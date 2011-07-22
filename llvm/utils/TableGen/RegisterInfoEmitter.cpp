//===- RegisterInfoEmitter.cpp - Generate a Register File Desc. -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of a target
// register file for a code generator.  It uses instances of the Register,
// RegisterAliases, and RegisterClass classes to gather this information.
//
//===----------------------------------------------------------------------===//

#include "RegisterInfoEmitter.h"
#include "CodeGenTarget.h"
#include "CodeGenRegisters.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include <algorithm>
#include <set>
using namespace llvm;

// runEnums - Print out enum values for all of the registers.
void
RegisterInfoEmitter::runEnums(raw_ostream &OS,
                              CodeGenTarget &Target, CodeGenRegBank &Bank) {
  const std::vector<CodeGenRegister*> &Registers = Bank.getRegisters();

  std::string Namespace = Registers[0]->TheDef->getValueAsString("Namespace");

  EmitSourceFileHeader("Target Register Enum Values", OS);

  OS << "\n#ifdef GET_REGINFO_ENUM\n";
  OS << "#undef GET_REGINFO_ENUM\n";

  OS << "namespace llvm {\n\n";

  if (!Namespace.empty())
    OS << "namespace " << Namespace << " {\n";
  OS << "enum {\n  NoRegister,\n";

  for (unsigned i = 0, e = Registers.size(); i != e; ++i)
    OS << "  " << Registers[i]->getName() << " = " <<
      Registers[i]->EnumValue << ",\n";
  assert(Registers.size() == Registers[Registers.size()-1]->EnumValue &&
         "Register enum value mismatch!");
  OS << "  NUM_TARGET_REGS \t// " << Registers.size()+1 << "\n";
  OS << "};\n";
  if (!Namespace.empty())
    OS << "}\n";

  const std::vector<CodeGenRegisterClass> &RegisterClasses =
    Target.getRegisterClasses();
  if (!RegisterClasses.empty()) {
    OS << "\n// Register classes\n";
    if (!Namespace.empty())
      OS << "namespace " << Namespace << " {\n";
    OS << "enum {\n";
    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      if (i) OS << ",\n";
      OS << "  " << RegisterClasses[i].getName() << "RegClassID";
      OS << " = " << i;
    }
    OS << "\n  };\n";
    if (!Namespace.empty())
      OS << "}\n";
  }

  const std::vector<Record*> RegAltNameIndices = Target.getRegAltNameIndices();
  // If the only definition is the default NoRegAltName, we don't need to
  // emit anything.
  if (RegAltNameIndices.size() > 1) {
    OS << "\n// Register alternate name indices\n";
    if (!Namespace.empty())
      OS << "namespace " << Namespace << " {\n";
    OS << "enum {\n";
    for (unsigned i = 0, e = RegAltNameIndices.size(); i != e; ++i)
      OS << "  " << RegAltNameIndices[i]->getName() << ",\t// " << i << "\n";
    OS << "  NUM_TARGET_REG_ALT_NAMES = " << RegAltNameIndices.size() << "\n";
    OS << "};\n";
    if (!Namespace.empty())
      OS << "}\n";
  }


  OS << "} // End llvm namespace \n";
  OS << "#endif // GET_REGINFO_ENUM\n\n";
}

void
RegisterInfoEmitter::EmitRegMapping(raw_ostream &OS,
                                    const std::vector<CodeGenRegister*> &Regs,
                                    bool isCtor) {

  // Collect all information about dwarf register numbers
  typedef std::map<Record*, std::vector<int64_t>, LessRecord> DwarfRegNumsMapTy;
  DwarfRegNumsMapTy DwarfRegNums;

  // First, just pull all provided information to the map
  unsigned maxLength = 0;
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    Record *Reg = Regs[i]->TheDef;
    std::vector<int64_t> RegNums = Reg->getValueAsListOfInts("DwarfNumbers");
    maxLength = std::max((size_t)maxLength, RegNums.size());
    if (DwarfRegNums.count(Reg))
      errs() << "Warning: DWARF numbers for register " << getQualifiedName(Reg)
             << "specified multiple times\n";
    DwarfRegNums[Reg] = RegNums;
  }

  if (!maxLength)
    return;

  // Now we know maximal length of number list. Append -1's, where needed
  for (DwarfRegNumsMapTy::iterator
       I = DwarfRegNums.begin(), E = DwarfRegNums.end(); I != E; ++I)
    for (unsigned i = I->second.size(), e = maxLength; i != e; ++i)
      I->second.push_back(-1);

  // Emit reverse information about the dwarf register numbers.
  for (unsigned j = 0; j < 2; ++j) {
    OS << "  switch (";
    if (j == 0)
      OS << "DwarfFlavour";
    else
      OS << "EHFlavour";
    OS << ") {\n"
     << "  default:\n"
     << "    assert(0 && \"Unknown DWARF flavour\");\n"
     << "    break;\n";

    for (unsigned i = 0, e = maxLength; i != e; ++i) {
      OS << "  case " << i << ":\n";
      for (DwarfRegNumsMapTy::iterator
             I = DwarfRegNums.begin(), E = DwarfRegNums.end(); I != E; ++I) {
        int DwarfRegNo = I->second[i];
        if (DwarfRegNo < 0)
          continue;
        OS << "    ";
        if (!isCtor)
          OS << "RI->";
        OS << "mapDwarfRegToLLVMReg(" << DwarfRegNo << ", "
           << getQualifiedName(I->first) << ", ";
        if (j == 0)
          OS << "false";
        else
          OS << "true";
        OS << " );\n";
      }
      OS << "    break;\n";
    }
    OS << "  }\n";
  }

  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    Record *Reg = Regs[i]->TheDef;
    const RecordVal *V = Reg->getValue("DwarfAlias");
    if (!V || !V->getValue())
      continue;

    DefInit *DI = dynamic_cast<DefInit*>(V->getValue());
    Record *Alias = DI->getDef();
    DwarfRegNums[Reg] = DwarfRegNums[Alias];
  }

  // Emit information about the dwarf register numbers.
  for (unsigned j = 0; j < 2; ++j) {
    OS << "  switch (";
    if (j == 0)
      OS << "DwarfFlavour";
    else
      OS << "EHFlavour";
    OS << ") {\n"
       << "  default:\n"
       << "    assert(0 && \"Unknown DWARF flavour\");\n"
       << "    break;\n";

    for (unsigned i = 0, e = maxLength; i != e; ++i) {
      OS << "  case " << i << ":\n";
      // Sort by name to get a stable order.
      for (DwarfRegNumsMapTy::iterator
             I = DwarfRegNums.begin(), E = DwarfRegNums.end(); I != E; ++I) {
        int RegNo = I->second[i];
        OS << "    ";
        if (!isCtor)
          OS << "RI->";
        OS << "mapLLVMRegToDwarfReg(" << getQualifiedName(I->first) << ", "
           <<  RegNo << ", ";
        if (j == 0)
          OS << "false";
        else
          OS << "true";
        OS << " );\n";
      }
      OS << "    break;\n";
    }
    OS << "  }\n";
  }
}

//
// runMCDesc - Print out MC register descriptions.
//
void
RegisterInfoEmitter::runMCDesc(raw_ostream &OS, CodeGenTarget &Target,
                               CodeGenRegBank &RegBank) {
  EmitSourceFileHeader("MC Register Information", OS);

  OS << "\n#ifdef GET_REGINFO_MC_DESC\n";
  OS << "#undef GET_REGINFO_MC_DESC\n";

  std::map<const CodeGenRegister*, CodeGenRegister::Set> Overlaps;
  RegBank.computeOverlaps(Overlaps);

  OS << "namespace llvm {\n\n";

  const std::string &TargetName = Target.getName();
  std::string ClassName = TargetName + "GenMCRegisterInfo";
  OS << "struct " << ClassName << " : public MCRegisterInfo {\n"
     << "  explicit " << ClassName << "(const MCRegisterDesc *D);\n";
  OS << "};\n";

  OS << "\nnamespace {\n";

  const std::vector<CodeGenRegister*> &Regs = RegBank.getRegisters();

  // Emit an overlap list for all registers.
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    const CodeGenRegister *Reg = Regs[i];
    const CodeGenRegister::Set &O = Overlaps[Reg];
    // Move Reg to the front so TRI::getAliasSet can share the list.
    OS << "  const unsigned " << Reg->getName() << "_Overlaps[] = { "
       << getQualifiedName(Reg->TheDef) << ", ";
    for (CodeGenRegister::Set::const_iterator I = O.begin(), E = O.end();
         I != E; ++I)
      if (*I != Reg)
        OS << getQualifiedName((*I)->TheDef) << ", ";
    OS << "0 };\n";
  }

  // Emit the empty sub-registers list
  OS << "  const unsigned Empty_SubRegsSet[] = { 0 };\n";
  // Loop over all of the registers which have sub-registers, emitting the
  // sub-registers list to memory.
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    const CodeGenRegister &Reg = *Regs[i];
    if (Reg.getSubRegs().empty())
     continue;
    // getSubRegs() orders by SubRegIndex. We want a topological order.
    SetVector<CodeGenRegister*> SR;
    Reg.addSubRegsPreOrder(SR);
    OS << "  const unsigned " << Reg.getName() << "_SubRegsSet[] = { ";
    for (unsigned j = 0, je = SR.size(); j != je; ++j)
      OS << getQualifiedName(SR[j]->TheDef) << ", ";
    OS << "0 };\n";
  }

  // Emit the empty super-registers list
  OS << "  const unsigned Empty_SuperRegsSet[] = { 0 };\n";
  // Loop over all of the registers which have super-registers, emitting the
  // super-registers list to memory.
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    const CodeGenRegister &Reg = *Regs[i];
    const CodeGenRegister::SuperRegList &SR = Reg.getSuperRegs();
    if (SR.empty())
      continue;
    OS << "  const unsigned " << Reg.getName() << "_SuperRegsSet[] = { ";
    for (unsigned j = 0, je = SR.size(); j != je; ++j)
      OS << getQualifiedName(SR[j]->TheDef) << ", ";
    OS << "0 };\n";
  }
  OS << "}\n";       // End of anonymous namespace...

  OS << "\nMCRegisterDesc " << TargetName
     << "RegDesc[] = { // Descriptors\n";
  OS << "  { \"NOREG\",\t0,\t0,\t0 },\n";

  // Now that register alias and sub-registers sets have been emitted, emit the
  // register descriptors now.
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    const CodeGenRegister &Reg = *Regs[i];
    OS << "  { \"";
    OS << Reg.getName() << "\",\t" << Reg.getName() << "_Overlaps,\t";
    if (!Reg.getSubRegs().empty())
      OS << Reg.getName() << "_SubRegsSet,\t";
    else
      OS << "Empty_SubRegsSet,\t";
    if (!Reg.getSuperRegs().empty())
      OS << Reg.getName() << "_SuperRegsSet";
    else
      OS << "Empty_SuperRegsSet";
    OS << " },\n";
  }
  OS << "};\n\n";      // End of register descriptors...

  // FIXME: This code is duplicated in the TargetRegisterClass emitter.
  const std::vector<CodeGenRegisterClass> &RegisterClasses =
    Target.getRegisterClasses();

  // Loop over all of the register classes... emitting each one.
  OS << "namespace {     // Register classes...\n";

  // Emit the register enum value arrays for each RegisterClass
  for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
    const CodeGenRegisterClass &RC = RegisterClasses[rc];
    ArrayRef<Record*> Order = RC.getOrder();

    // Give the register class a legal C name if it's anonymous.
    std::string Name = RC.getName();

    // Emit the register list now.
    OS << "  // " << Name << " Register Class...\n"
       << "  static const unsigned " << Name
       << "[] = {\n    ";
    for (unsigned i = 0, e = Order.size(); i != e; ++i) {
      Record *Reg = Order[i];
      OS << getQualifiedName(Reg) << ", ";
    }
    OS << "\n  };\n\n";
  }
  OS << "}\n\n";

  OS << "MCRegisterClass " << TargetName << "MCRegisterClasses[] = {\n";

  for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
    const CodeGenRegisterClass &RC = RegisterClasses[rc];
    OS << "  MCRegisterClass(";
    if (!RC.Namespace.empty())
      OS << RC.Namespace << "::";
    OS << RC.getName() + "RegClassID" << ", "
       << '\"' << RC.getName() << "\", "
       << RC.SpillSize/8 << ", "
       << RC.SpillAlignment/8 << ", "
       << RC.CopyCost << ", "
       << RC.Allocatable << ", "
       << RC.getName() << ", " << RC.getName() << " + "
       << RC.getOrder().size()
       << "),\n";
  }

  OS << "};\n\n";

  // MCRegisterInfo initialization routine.
  OS << "static inline void Init" << TargetName
     << "MCRegisterInfo(MCRegisterInfo *RI, unsigned RA, "
     << "unsigned DwarfFlavour = 0, unsigned EHFlavour = 0) {\n";
  OS << "  RI->InitMCRegisterInfo(" << TargetName << "RegDesc, "
     << Regs.size()+1 << ", RA, " << TargetName << "MCRegisterClasses, "
     << RegisterClasses.size() << ");\n\n";

  EmitRegMapping(OS, Regs, false);

  OS << "}\n\n";


  OS << "} // End llvm namespace \n";
  OS << "#endif // GET_REGINFO_MC_DESC\n\n";
}

void
RegisterInfoEmitter::runTargetHeader(raw_ostream &OS, CodeGenTarget &Target,
                                     CodeGenRegBank &RegBank) {
  EmitSourceFileHeader("Register Information Header Fragment", OS);

  OS << "\n#ifdef GET_REGINFO_HEADER\n";
  OS << "#undef GET_REGINFO_HEADER\n";

  const std::string &TargetName = Target.getName();
  std::string ClassName = TargetName + "GenRegisterInfo";

  OS << "#include \"llvm/Target/TargetRegisterInfo.h\"\n";
  OS << "#include <string>\n\n";

  OS << "namespace llvm {\n\n";

  OS << "struct " << ClassName << " : public TargetRegisterInfo {\n"
     << "  explicit " << ClassName
     << "(unsigned RA, unsigned D = 0, unsigned E = 0);\n"
     << "  virtual bool needsStackRealignment(const MachineFunction &) const\n"
     << "     { return false; }\n"
     << "  unsigned getSubReg(unsigned RegNo, unsigned Index) const;\n"
     << "  unsigned getSubRegIndex(unsigned RegNo, unsigned SubRegNo) const;\n"
     << "  unsigned composeSubRegIndices(unsigned, unsigned) const;\n"
     << "};\n\n";

  const std::vector<Record*> &SubRegIndices = RegBank.getSubRegIndices();
  if (!SubRegIndices.empty()) {
    OS << "\n// Subregister indices\n";
    std::string Namespace = SubRegIndices[0]->getValueAsString("Namespace");
    if (!Namespace.empty())
      OS << "namespace " << Namespace << " {\n";
    OS << "enum {\n  NoSubRegister,\n";
    for (unsigned i = 0, e = RegBank.getNumNamedIndices(); i != e; ++i)
      OS << "  " << SubRegIndices[i]->getName() << ",\t// " << i+1 << "\n";
    OS << "  NUM_TARGET_NAMED_SUBREGS = " << SubRegIndices.size()+1 << "\n";
    OS << "};\n";
    if (!Namespace.empty())
      OS << "}\n";
  }

  const std::vector<CodeGenRegisterClass> &RegisterClasses =
    Target.getRegisterClasses();

  if (!RegisterClasses.empty()) {
    OS << "namespace " << RegisterClasses[0].Namespace
       << " { // Register classes\n";

    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      const CodeGenRegisterClass &RC = RegisterClasses[i];
      const std::string &Name = RC.getName();

      // Output the register class definition.
      OS << "  struct " << Name << "Class : public TargetRegisterClass {\n"
         << "    " << Name << "Class();\n";
      if (!RC.AltOrderSelect.empty())
        OS << "    ArrayRef<unsigned> "
              "getRawAllocationOrder(const MachineFunction&) const;\n";
      OS << "  };\n";

      // Output the extern for the instance.
      OS << "  extern " << Name << "Class\t" << Name << "RegClass;\n";
      // Output the extern for the pointer to the instance (should remove).
      OS << "  static TargetRegisterClass * const "<< Name <<"RegisterClass = &"
         << Name << "RegClass;\n";
    }
    OS << "} // end of namespace " << TargetName << "\n\n";
  }
  OS << "} // End llvm namespace \n";
  OS << "#endif // GET_REGINFO_HEADER\n\n";
}

//
// runTargetDesc - Output the target register and register file descriptions.
//
void
RegisterInfoEmitter::runTargetDesc(raw_ostream &OS, CodeGenTarget &Target,
                                   CodeGenRegBank &RegBank){
  EmitSourceFileHeader("Target Register and Register Classes Information", OS);

  OS << "\n#ifdef GET_REGINFO_TARGET_DESC\n";
  OS << "#undef GET_REGINFO_TARGET_DESC\n";

  OS << "namespace llvm {\n\n";

  // Start out by emitting each of the register classes.
  const std::vector<CodeGenRegisterClass> &RegisterClasses =
    Target.getRegisterClasses();

  // Collect all registers belonging to any allocatable class.
  std::set<Record*> AllocatableRegs;

  // Loop over all of the register classes... emitting each one.
  OS << "namespace {     // Register classes...\n";

  // Emit the register enum value arrays for each RegisterClass
  for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
    const CodeGenRegisterClass &RC = RegisterClasses[rc];
    ArrayRef<Record*> Order = RC.getOrder();

    // Collect allocatable registers.
    if (RC.Allocatable)
      AllocatableRegs.insert(Order.begin(), Order.end());

    // Give the register class a legal C name if it's anonymous.
    std::string Name = RC.getName();

    // Emit the register list now.
    OS << "  // " << Name << " Register Class...\n"
       << "  static const unsigned " << Name
       << "[] = {\n    ";
    for (unsigned i = 0, e = Order.size(); i != e; ++i) {
      Record *Reg = Order[i];
      OS << getQualifiedName(Reg) << ", ";
    }
    OS << "\n  };\n\n";
  }

  // Emit the ValueType arrays for each RegisterClass
  for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
    const CodeGenRegisterClass &RC = RegisterClasses[rc];

    // Give the register class a legal C name if it's anonymous.
    std::string Name = RC.getName() + "VTs";

    // Emit the register list now.
    OS << "  // " << Name
       << " Register Class Value Types...\n"
       << "  static const EVT " << Name
       << "[] = {\n    ";
    for (unsigned i = 0, e = RC.VTs.size(); i != e; ++i)
      OS << getEnumName(RC.VTs[i]) << ", ";
    OS << "MVT::Other\n  };\n\n";
  }
  OS << "}  // end anonymous namespace\n\n";

  // Now that all of the structs have been emitted, emit the instances.
  if (!RegisterClasses.empty()) {
    OS << "namespace " << RegisterClasses[0].Namespace
       << " {   // Register class instances\n";
    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i)
      OS << "  " << RegisterClasses[i].getName()  << "Class\t"
         << RegisterClasses[i].getName() << "RegClass;\n";

    std::map<unsigned, std::set<unsigned> > SuperClassMap;
    std::map<unsigned, std::set<unsigned> > SuperRegClassMap;
    OS << "\n";

    unsigned NumSubRegIndices = RegBank.getSubRegIndices().size();

    if (NumSubRegIndices) {
      // Emit the sub-register classes for each RegisterClass
      for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
        const CodeGenRegisterClass &RC = RegisterClasses[rc];
        std::vector<Record*> SRC(NumSubRegIndices);
        for (DenseMap<Record*,Record*>::const_iterator
             i = RC.SubRegClasses.begin(),
             e = RC.SubRegClasses.end(); i != e; ++i) {
          // Build SRC array.
          unsigned idx = RegBank.getSubRegIndexNo(i->first);
          SRC.at(idx-1) = i->second;

          // Find the register class number of i->second for SuperRegClassMap.
          for (unsigned rc2 = 0, e2 = RegisterClasses.size(); rc2 != e2; ++rc2) {
            const CodeGenRegisterClass &RC2 =  RegisterClasses[rc2];
            if (RC2.TheDef == i->second) {
              SuperRegClassMap[rc2].insert(rc);
              break;
            }
          }
        }

        // Give the register class a legal C name if it's anonymous.
        std::string Name = RC.TheDef->getName();

        OS << "  // " << Name
           << " Sub-register Classes...\n"
           << "  static const TargetRegisterClass* const "
           << Name << "SubRegClasses[] = {\n    ";

        for (unsigned idx = 0; idx != NumSubRegIndices; ++idx) {
          if (idx)
            OS << ", ";
          if (SRC[idx])
            OS << "&" << getQualifiedName(SRC[idx]) << "RegClass";
          else
            OS << "0";
        }
        OS << "\n  };\n\n";
      }

      // Emit the super-register classes for each RegisterClass
      for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
        const CodeGenRegisterClass &RC = RegisterClasses[rc];

        // Give the register class a legal C name if it's anonymous.
        std::string Name = RC.TheDef->getName();

        OS << "  // " << Name
           << " Super-register Classes...\n"
           << "  static const TargetRegisterClass* const "
           << Name << "SuperRegClasses[] = {\n    ";

        bool Empty = true;
        std::map<unsigned, std::set<unsigned> >::iterator I =
          SuperRegClassMap.find(rc);
        if (I != SuperRegClassMap.end()) {
          for (std::set<unsigned>::iterator II = I->second.begin(),
                 EE = I->second.end(); II != EE; ++II) {
            const CodeGenRegisterClass &RC2 = RegisterClasses[*II];
            if (!Empty)
              OS << ", ";
            OS << "&" << getQualifiedName(RC2.TheDef) << "RegClass";
            Empty = false;
          }
        }

        OS << (!Empty ? ", " : "") << "NULL";
        OS << "\n  };\n\n";
      }
    } else {
      // No subregindices in this target
      OS << "  static const TargetRegisterClass* const "
         << "NullRegClasses[] = { NULL };\n\n";
    }

    // Emit the sub-classes array for each RegisterClass
    for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
      const CodeGenRegisterClass &RC = RegisterClasses[rc];

      // Give the register class a legal C name if it's anonymous.
      std::string Name = RC.TheDef->getName();

      OS << "  // " << Name
         << " Register Class sub-classes...\n"
         << "  static const TargetRegisterClass* const "
         << Name << "Subclasses[] = {\n    ";

      bool Empty = true;
      for (unsigned rc2 = 0, e2 = RegisterClasses.size(); rc2 != e2; ++rc2) {
        const CodeGenRegisterClass &RC2 = RegisterClasses[rc2];

        // Sub-classes are used to determine if a virtual register can be used
        // as an instruction operand, or if it must be copied first.
        if (rc == rc2 || !RC.hasSubClass(&RC2)) continue;

        if (!Empty) OS << ", ";
        OS << "&" << getQualifiedName(RC2.TheDef) << "RegClass";
        Empty = false;

        std::map<unsigned, std::set<unsigned> >::iterator SCMI =
          SuperClassMap.find(rc2);
        if (SCMI == SuperClassMap.end()) {
          SuperClassMap.insert(std::make_pair(rc2, std::set<unsigned>()));
          SCMI = SuperClassMap.find(rc2);
        }
        SCMI->second.insert(rc);
      }

      OS << (!Empty ? ", " : "") << "NULL";
      OS << "\n  };\n\n";
    }

    for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
      const CodeGenRegisterClass &RC = RegisterClasses[rc];

      // Give the register class a legal C name if it's anonymous.
      std::string Name = RC.TheDef->getName();

      OS << "  // " << Name
         << " Register Class super-classes...\n"
         << "  static const TargetRegisterClass* const "
         << Name << "Superclasses[] = {\n    ";

      bool Empty = true;
      std::map<unsigned, std::set<unsigned> >::iterator I =
        SuperClassMap.find(rc);
      if (I != SuperClassMap.end()) {
        for (std::set<unsigned>::iterator II = I->second.begin(),
               EE = I->second.end(); II != EE; ++II) {
          const CodeGenRegisterClass &RC2 = RegisterClasses[*II];
          if (!Empty) OS << ", ";
          OS << "&" << getQualifiedName(RC2.TheDef) << "RegClass";
          Empty = false;
        }
      }

      OS << (!Empty ? ", " : "") << "NULL";
      OS << "\n  };\n\n";
    }

    // Emit methods.
    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      const CodeGenRegisterClass &RC = RegisterClasses[i];
      OS << RC.getName() << "Class::" << RC.getName()
         << "Class()  : TargetRegisterClass("
         << RC.getName() + "RegClassID" << ", "
         << '\"' << RC.getName() << "\", "
         << RC.getName() + "VTs" << ", "
         << RC.getName() + "Subclasses" << ", "
         << RC.getName() + "Superclasses" << ", "
         << (NumSubRegIndices ? RC.getName() + "Sub" : std::string("Null"))
         << "RegClasses, "
         << (NumSubRegIndices ? RC.getName() + "Super" : std::string("Null"))
         << "RegClasses, "
         << RC.SpillSize/8 << ", "
         << RC.SpillAlignment/8 << ", "
         << RC.CopyCost << ", "
         << RC.Allocatable << ", "
         << RC.getName() << ", " << RC.getName() << " + "
         << RC.getOrder().size()
         << ") {}\n";
      if (!RC.AltOrderSelect.empty()) {
        OS << "\nstatic inline unsigned " << RC.getName()
           << "AltOrderSelect(const MachineFunction &MF) {"
           << RC.AltOrderSelect << "}\n\nArrayRef<unsigned> "
           << RC.getName() << "Class::"
           << "getRawAllocationOrder(const MachineFunction &MF) const {\n";
        for (unsigned oi = 1 , oe = RC.getNumOrders(); oi != oe; ++oi) {
          ArrayRef<Record*> Elems = RC.getOrder(oi);
          OS << "  static const unsigned AltOrder" << oi << "[] = {";
          for (unsigned elem = 0; elem != Elems.size(); ++elem)
            OS << (elem ? ", " : " ") << getQualifiedName(Elems[elem]);
          OS << " };\n";
        }
        OS << "  static const ArrayRef<unsigned> Order[] = {\n"
           << "    makeArrayRef(" << RC.getName();
        for (unsigned oi = 1, oe = RC.getNumOrders(); oi != oe; ++oi)
          OS << "),\n    makeArrayRef(AltOrder" << oi;
        OS << ")\n  };\n  const unsigned Select = " << RC.getName()
           << "AltOrderSelect(MF);\n  assert(Select < " << RC.getNumOrders()
           << ");\n  return Order[Select];\n}\n";
        }
    }

    OS << "}\n";
  }

  OS << "\nnamespace {\n";
  OS << "  const TargetRegisterClass* const RegisterClasses[] = {\n";
  for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i)
    OS << "    &" << getQualifiedName(RegisterClasses[i].TheDef)
       << "RegClass,\n";
  OS << "  };\n";
  OS << "}\n";       // End of anonymous namespace...

  // Emit extra information about registers.
  const std::string &TargetName = Target.getName();
  OS << "\n  static const TargetRegisterInfoDesc "
     << TargetName << "RegInfoDesc[] = "
     << "{ // Extra Descriptors\n";
  OS << "    { 0, 0 },\n";

  const std::vector<CodeGenRegister*> &Regs = RegBank.getRegisters();
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    const CodeGenRegister &Reg = *Regs[i];
    OS << "    { ";
    OS << Reg.CostPerUse << ", "
       << int(AllocatableRegs.count(Reg.TheDef)) << " },\n";
  }
  OS << "  };\n";      // End of register descriptors...


  // Calculate the mapping of subregister+index pairs to physical registers.
  // This will also create further anonymous indexes.
  unsigned NamedIndices = RegBank.getNumNamedIndices();

  // Emit SubRegIndex names, skipping 0
  const std::vector<Record*> &SubRegIndices = RegBank.getSubRegIndices();
  OS << "\n  static const char *const " << TargetName
     << "SubRegIndexTable[] = { \"";
  for (unsigned i = 0, e = SubRegIndices.size(); i != e; ++i) {
    OS << SubRegIndices[i]->getName();
    if (i+1 != e)
      OS << "\", \"";
  }
  OS << "\" };\n\n";

  // Emit names of the anonymus subreg indexes.
  if (SubRegIndices.size() > NamedIndices) {
    OS << "  enum {";
    for (unsigned i = NamedIndices, e = SubRegIndices.size(); i != e; ++i) {
      OS << "\n    " << SubRegIndices[i]->getName() << " = " << i+1;
      if (i+1 != e)
        OS << ',';
    }
    OS << "\n  };\n\n";
  }
  OS << "\n";

  std::string ClassName = Target.getName() + "GenRegisterInfo";

  // Emit the subregister + index mapping function based on the information
  // calculated above.
  OS << "unsigned " << ClassName
     << "::getSubReg(unsigned RegNo, unsigned Index) const {\n"
     << "  switch (RegNo) {\n"
     << "  default:\n    return 0;\n";
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    const CodeGenRegister::SubRegMap &SRM = Regs[i]->getSubRegs();
    if (SRM.empty())
      continue;
    OS << "  case " << getQualifiedName(Regs[i]->TheDef) << ":\n";
    OS << "    switch (Index) {\n";
    OS << "    default: return 0;\n";
    for (CodeGenRegister::SubRegMap::const_iterator ii = SRM.begin(),
         ie = SRM.end(); ii != ie; ++ii)
      OS << "    case " << getQualifiedName(ii->first)
         << ": return " << getQualifiedName(ii->second->TheDef) << ";\n";
    OS << "    };\n" << "    break;\n";
  }
  OS << "  };\n";
  OS << "  return 0;\n";
  OS << "}\n\n";

  OS << "unsigned " << ClassName
     << "::getSubRegIndex(unsigned RegNo, unsigned SubRegNo) const {\n"
     << "  switch (RegNo) {\n"
     << "  default:\n    return 0;\n";
   for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
     const CodeGenRegister::SubRegMap &SRM = Regs[i]->getSubRegs();
     if (SRM.empty())
       continue;
    OS << "  case " << getQualifiedName(Regs[i]->TheDef) << ":\n";
    for (CodeGenRegister::SubRegMap::const_iterator ii = SRM.begin(),
         ie = SRM.end(); ii != ie; ++ii)
      OS << "    if (SubRegNo == " << getQualifiedName(ii->second->TheDef)
         << ")  return " << getQualifiedName(ii->first) << ";\n";
    OS << "    return 0;\n";
  }
  OS << "  };\n";
  OS << "  return 0;\n";
  OS << "}\n\n";

  // Emit composeSubRegIndices
  OS << "unsigned " << ClassName
     << "::composeSubRegIndices(unsigned IdxA, unsigned IdxB) const {\n"
     << "  switch (IdxA) {\n"
     << "  default:\n    return IdxB;\n";
  for (unsigned i = 0, e = SubRegIndices.size(); i != e; ++i) {
    bool Open = false;
    for (unsigned j = 0; j != e; ++j) {
      if (Record *Comp = RegBank.getCompositeSubRegIndex(SubRegIndices[i],
                                                         SubRegIndices[j])) {
        if (!Open) {
          OS << "  case " << getQualifiedName(SubRegIndices[i])
             << ": switch(IdxB) {\n    default: return IdxB;\n";
          Open = true;
        }
        OS << "    case " << getQualifiedName(SubRegIndices[j])
           << ": return " << getQualifiedName(Comp) << ";\n";
      }
    }
    if (Open)
      OS << "    }\n";
  }
  OS << "  }\n}\n\n";

  // Emit the constructor of the class...
  OS << "extern MCRegisterDesc " << TargetName << "RegDesc[];\n";
  OS << "extern MCRegisterClass " << TargetName << "MCRegisterClasses[];\n";

  OS << ClassName << "::" << ClassName
     << "(unsigned RA, unsigned DwarfFlavour, unsigned EHFlavour)\n"
     << "  : TargetRegisterInfo(" << TargetName << "RegInfoDesc"
     << ", RegisterClasses, RegisterClasses+" << RegisterClasses.size() <<",\n"
     << "                 " << TargetName << "SubRegIndexTable) {\n"
     << "  InitMCRegisterInfo(" << TargetName << "RegDesc, "
     << Regs.size()+1 << ", RA, " << TargetName << "MCRegisterClasses, "
     << RegisterClasses.size() << ");\n\n";

  EmitRegMapping(OS, Regs, true);

  OS << "}\n\n";

  OS << "} // End llvm namespace \n";
  OS << "#endif // GET_REGINFO_TARGET_DESC\n\n";
}

void RegisterInfoEmitter::run(raw_ostream &OS) {
  CodeGenTarget Target(Records);
  CodeGenRegBank &RegBank = Target.getRegBank();
  RegBank.computeDerivedInfo();

  runEnums(OS, Target, RegBank);
  runMCDesc(OS, Target, RegBank);
  runTargetHeader(OS, Target, RegBank);
  runTargetDesc(OS, Target, RegBank);
}
