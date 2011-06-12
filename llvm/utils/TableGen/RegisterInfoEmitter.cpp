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
#include <algorithm>
#include <set>
using namespace llvm;

// runEnums - Print out enum values for all of the registers.
void RegisterInfoEmitter::runEnums(raw_ostream &OS) {
  CodeGenTarget Target(Records);
  CodeGenRegBank &Bank = Target.getRegBank();
  const std::vector<CodeGenRegister> &Registers = Target.getRegisters();

  std::string Namespace = Registers[0].TheDef->getValueAsString("Namespace");

  EmitSourceFileHeader("Target Register Enum Values", OS);
  OS << "namespace llvm {\n\n";

  if (!Namespace.empty())
    OS << "namespace " << Namespace << " {\n";
  OS << "enum {\n  NoRegister,\n";

  for (unsigned i = 0, e = Registers.size(); i != e; ++i)
    OS << "  " << Registers[i].getName() << " = " <<
      Registers[i].EnumValue << ",\n";
  assert(Registers.size() == Registers[Registers.size()-1].EnumValue &&
         "Register enum value mismatch!");
  OS << "  NUM_TARGET_REGS \t// " << Registers.size()+1 << "\n";
  OS << "};\n";
  if (!Namespace.empty())
    OS << "}\n";

  const std::vector<Record*> &SubRegIndices = Bank.getSubRegIndices();
  if (!SubRegIndices.empty()) {
    OS << "\n// Subregister indices\n";
    Namespace = SubRegIndices[0]->getValueAsString("Namespace");
    if (!Namespace.empty())
      OS << "namespace " << Namespace << " {\n";
    OS << "enum {\n  NoSubRegister,\n";
    for (unsigned i = 0, e = Bank.getNumNamedIndices(); i != e; ++i)
      OS << "  " << SubRegIndices[i]->getName() << ",\t// " << i+1 << "\n";
    OS << "  NUM_TARGET_NAMED_SUBREGS = " << SubRegIndices.size()+1 << "\n";
    OS << "};\n";
    if (!Namespace.empty())
      OS << "}\n";
  }
  OS << "} // End llvm namespace \n";
}

void RegisterInfoEmitter::runHeader(raw_ostream &OS) {
  EmitSourceFileHeader("Register Information Header Fragment", OS);
  CodeGenTarget Target(Records);
  const std::string &TargetName = Target.getName();
  std::string ClassName = TargetName + "GenRegisterInfo";

  OS << "#include \"llvm/Target/TargetRegisterInfo.h\"\n";
  OS << "#include <string>\n\n";

  OS << "namespace llvm {\n\n";

  OS << "struct " << ClassName << " : public TargetRegisterInfo {\n"
     << "  explicit " << ClassName
     << "(int CallFrameSetupOpcode = -1, int CallFrameDestroyOpcode = -1);\n"
     << "  virtual int getDwarfRegNumFull(unsigned RegNum, "
     << "unsigned Flavour) const;\n"
     << "  virtual int getLLVMRegNumFull(unsigned DwarfRegNum, "
     << "unsigned Flavour) const;\n"
     << "  virtual int getDwarfRegNum(unsigned RegNum, bool isEH) const = 0;\n"
     << "  virtual bool needsStackRealignment(const MachineFunction &) const\n"
     << "     { return false; }\n"
     << "  unsigned getSubReg(unsigned RegNo, unsigned Index) const;\n"
     << "  unsigned getSubRegIndex(unsigned RegNo, unsigned SubRegNo) const;\n"
     << "  unsigned composeSubRegIndices(unsigned, unsigned) const;\n"
     << "};\n\n";

  const std::vector<CodeGenRegisterClass> &RegisterClasses =
    Target.getRegisterClasses();

  if (!RegisterClasses.empty()) {
    OS << "namespace " << RegisterClasses[0].Namespace
       << " { // Register classes\n";

    OS << "  enum {\n";
    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      if (i) OS << ",\n";
      OS << "    " << RegisterClasses[i].getName() << "RegClassID";
      OS << " = " << i;
    }
    OS << "\n  };\n\n";

    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      const std::string &Name = RegisterClasses[i].getName();

      // Output the register class definition.
      OS << "  struct " << Name << "Class : public TargetRegisterClass {\n"
         << "    " << Name << "Class();\n"
         << RegisterClasses[i].MethodProtos << "  };\n";

      // Output the extern for the instance.
      OS << "  extern " << Name << "Class\t" << Name << "RegClass;\n";
      // Output the extern for the pointer to the instance (should remove).
      OS << "  static TargetRegisterClass * const "<< Name <<"RegisterClass = &"
         << Name << "RegClass;\n";
    }
    OS << "} // end of namespace " << TargetName << "\n\n";
  }
  OS << "} // End llvm namespace \n";
}

static void addSuperReg(Record *R, Record *S,
                  std::map<Record*, std::set<Record*>, LessRecord> &SubRegs,
                  std::map<Record*, std::set<Record*>, LessRecord> &SuperRegs,
                  std::map<Record*, std::set<Record*>, LessRecord> &Aliases) {
  if (R == S) {
    errs() << "Error: recursive sub-register relationship between"
           << " register " << getQualifiedName(R)
           << " and its sub-registers?\n";
    abort();
  }
  if (!SuperRegs[R].insert(S).second)
    return;
  SubRegs[S].insert(R);
  Aliases[R].insert(S);
  Aliases[S].insert(R);
  if (SuperRegs.count(S))
    for (std::set<Record*>::iterator I = SuperRegs[S].begin(),
           E = SuperRegs[S].end(); I != E; ++I)
      addSuperReg(R, *I, SubRegs, SuperRegs, Aliases);
}

static void addSubSuperReg(Record *R, Record *S,
                   std::map<Record*, std::set<Record*>, LessRecord> &SubRegs,
                   std::map<Record*, std::set<Record*>, LessRecord> &SuperRegs,
                   std::map<Record*, std::set<Record*>, LessRecord> &Aliases) {
  if (R == S) {
    errs() << "Error: recursive sub-register relationship between"
           << " register " << getQualifiedName(R)
           << " and its sub-registers?\n";
    abort();
  }

  if (!SubRegs[R].insert(S).second)
    return;
  addSuperReg(S, R, SubRegs, SuperRegs, Aliases);
  Aliases[R].insert(S);
  Aliases[S].insert(R);
  if (SubRegs.count(S))
    for (std::set<Record*>::iterator I = SubRegs[S].begin(),
           E = SubRegs[S].end(); I != E; ++I)
      addSubSuperReg(R, *I, SubRegs, SuperRegs, Aliases);
}

// RegisterInfoEmitter::run - Main register file description emitter.
//
void RegisterInfoEmitter::run(raw_ostream &OS) {
  CodeGenTarget Target(Records);
  CodeGenRegBank &RegBank = Target.getRegBank();
  RegBank.computeDerivedInfo();
  std::map<const CodeGenRegister*, CodeGenRegister::Set> Overlaps;
  RegBank.computeOverlaps(Overlaps);

  EmitSourceFileHeader("Register Information Source Fragment", OS);

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

    // Collect allocatable registers.
    if (RC.Allocatable)
      AllocatableRegs.insert(RC.Elements.begin(), RC.Elements.end());

    // Give the register class a legal C name if it's anonymous.
    std::string Name = RC.TheDef->getName();

    // Emit the register list now.
    OS << "  // " << Name << " Register Class...\n"
       << "  static const unsigned " << Name
       << "[] = {\n    ";
    for (unsigned i = 0, e = RC.Elements.size(); i != e; ++i) {
      Record *Reg = RC.Elements[i];
      OS << getQualifiedName(Reg) << ", ";
    }
    OS << "\n  };\n\n";
  }

  // Emit the ValueType arrays for each RegisterClass
  for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
    const CodeGenRegisterClass &RC = RegisterClasses[rc];

    // Give the register class a legal C name if it's anonymous.
    std::string Name = RC.TheDef->getName() + "VTs";

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


    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      const CodeGenRegisterClass &RC = RegisterClasses[i];
      OS << RC.MethodBodies << "\n";
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
         << RC.getName() << ", " << RC.getName() << " + " << RC.Elements.size()
         << ") {}\n";
    }

    OS << "}\n";
  }

  OS << "\nnamespace {\n";
  OS << "  const TargetRegisterClass* const RegisterClasses[] = {\n";
  for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i)
    OS << "    &" << getQualifiedName(RegisterClasses[i].TheDef)
       << "RegClass,\n";
  OS << "  };\n";

  // Emit register sub-registers / super-registers, aliases...
  std::map<Record*, std::set<Record*>, LessRecord> RegisterSubRegs;
  std::map<Record*, std::set<Record*>, LessRecord> RegisterSuperRegs;
  std::map<Record*, std::set<Record*>, LessRecord> RegisterAliases;
  typedef std::map<Record*, std::vector<int64_t>, LessRecord> DwarfRegNumsMapTy;
  DwarfRegNumsMapTy DwarfRegNums;

  const std::vector<CodeGenRegister> &Regs = Target.getRegisters();

  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    Record *R = Regs[i].TheDef;
    std::vector<Record*> LI = Regs[i].TheDef->getValueAsListOfDefs("Aliases");
    // Add information that R aliases all of the elements in the list... and
    // that everything in the list aliases R.
    for (unsigned j = 0, e = LI.size(); j != e; ++j) {
      Record *Reg = LI[j];
      if (RegisterAliases[R].count(Reg))
        errs() << "Warning: register alias between " << getQualifiedName(R)
               << " and " << getQualifiedName(Reg)
               << " specified multiple times!\n";
      RegisterAliases[R].insert(Reg);

      if (RegisterAliases[Reg].count(R))
        errs() << "Warning: register alias between " << getQualifiedName(R)
               << " and " << getQualifiedName(Reg)
               << " specified multiple times!\n";
      RegisterAliases[Reg].insert(R);
    }
  }

  // Process sub-register sets.
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    Record *R = Regs[i].TheDef;
    std::vector<Record*> LI = Regs[i].TheDef->getValueAsListOfDefs("SubRegs");
    // Process sub-register set and add aliases information.
    for (unsigned j = 0, e = LI.size(); j != e; ++j) {
      Record *SubReg = LI[j];
      if (RegisterSubRegs[R].count(SubReg))
        errs() << "Warning: register " << getQualifiedName(SubReg)
               << " specified as a sub-register of " << getQualifiedName(R)
               << " multiple times!\n";
      addSubSuperReg(R, SubReg, RegisterSubRegs, RegisterSuperRegs,
                     RegisterAliases);
    }
  }

  // Print the SubregHashTable, a simple quadratically probed
  // hash table for determining if a register is a subregister
  // of another register.
  unsigned NumSubRegs = 0;
  std::map<Record*, unsigned> RegNo;
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    RegNo[Regs[i].TheDef] = i;
    NumSubRegs += RegisterSubRegs[Regs[i].TheDef].size();
  }

  unsigned SubregHashTableSize = 2 * NextPowerOf2(2 * NumSubRegs);
  unsigned* SubregHashTable = new unsigned[2 * SubregHashTableSize];
  std::fill(SubregHashTable, SubregHashTable + 2 * SubregHashTableSize, ~0U);

  unsigned hashMisses = 0;

  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    Record* R = Regs[i].TheDef;
    for (std::set<Record*>::iterator I = RegisterSubRegs[R].begin(),
         E = RegisterSubRegs[R].end(); I != E; ++I) {
      Record* RJ = *I;
      // We have to increase the indices of both registers by one when
      // computing the hash because, in the generated code, there
      // will be an extra empty slot at register 0.
      size_t index = ((i+1) + (RegNo[RJ]+1) * 37) & (SubregHashTableSize-1);
      unsigned ProbeAmt = 2;
      while (SubregHashTable[index*2] != ~0U &&
             SubregHashTable[index*2+1] != ~0U) {
        index = (index + ProbeAmt) & (SubregHashTableSize-1);
        ProbeAmt += 2;

        hashMisses++;
      }

      SubregHashTable[index*2] = i;
      SubregHashTable[index*2+1] = RegNo[RJ];
    }
  }

  OS << "\n\n  // Number of hash collisions: " << hashMisses << "\n";

  if (SubregHashTableSize) {
    std::string Namespace = Regs[0].TheDef->getValueAsString("Namespace");

    OS << "  const unsigned SubregHashTable[] = { ";
    for (unsigned i = 0; i < SubregHashTableSize - 1; ++i) {
      if (i != 0)
        // Insert spaces for nice formatting.
        OS << "                                       ";

      if (SubregHashTable[2*i] != ~0U) {
        OS << getQualifiedName(Regs[SubregHashTable[2*i]].TheDef) << ", "
           << getQualifiedName(Regs[SubregHashTable[2*i+1]].TheDef) << ", \n";
      } else {
        OS << Namespace << "::NoRegister, " << Namespace << "::NoRegister, \n";
      }
    }

    unsigned Idx = SubregHashTableSize*2-2;
    if (SubregHashTable[Idx] != ~0U) {
      OS << "                                       "
         << getQualifiedName(Regs[SubregHashTable[Idx]].TheDef) << ", "
         << getQualifiedName(Regs[SubregHashTable[Idx+1]].TheDef) << " };\n";
    } else {
      OS << Namespace << "::NoRegister, " << Namespace << "::NoRegister };\n";
    }

    OS << "  const unsigned SubregHashTableSize = "
       << SubregHashTableSize << ";\n";
  } else {
    OS << "  const unsigned SubregHashTable[] = { ~0U, ~0U };\n"
       << "  const unsigned SubregHashTableSize = 1;\n";
  }

  delete [] SubregHashTable;


  // Print the AliasHashTable, a simple quadratically probed
  // hash table for determining if a register aliases another register.
  unsigned NumAliases = 0;
  RegNo.clear();
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    RegNo[Regs[i].TheDef] = i;
    NumAliases += RegisterAliases[Regs[i].TheDef].size();
  }

  unsigned AliasesHashTableSize = 2 * NextPowerOf2(2 * NumAliases);
  unsigned* AliasesHashTable = new unsigned[2 * AliasesHashTableSize];
  std::fill(AliasesHashTable, AliasesHashTable + 2 * AliasesHashTableSize, ~0U);

  hashMisses = 0;

  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    Record* R = Regs[i].TheDef;
    for (std::set<Record*>::iterator I = RegisterAliases[R].begin(),
         E = RegisterAliases[R].end(); I != E; ++I) {
      Record* RJ = *I;
      // We have to increase the indices of both registers by one when
      // computing the hash because, in the generated code, there
      // will be an extra empty slot at register 0.
      size_t index = ((i+1) + (RegNo[RJ]+1) * 37) & (AliasesHashTableSize-1);
      unsigned ProbeAmt = 2;
      while (AliasesHashTable[index*2] != ~0U &&
             AliasesHashTable[index*2+1] != ~0U) {
        index = (index + ProbeAmt) & (AliasesHashTableSize-1);
        ProbeAmt += 2;

        hashMisses++;
      }

      AliasesHashTable[index*2] = i;
      AliasesHashTable[index*2+1] = RegNo[RJ];
    }
  }

  OS << "\n\n  // Number of hash collisions: " << hashMisses << "\n";

  if (AliasesHashTableSize) {
    std::string Namespace = Regs[0].TheDef->getValueAsString("Namespace");

    OS << "  const unsigned AliasesHashTable[] = { ";
    for (unsigned i = 0; i < AliasesHashTableSize - 1; ++i) {
      if (i != 0)
        // Insert spaces for nice formatting.
        OS << "                                       ";

      if (AliasesHashTable[2*i] != ~0U) {
        OS << getQualifiedName(Regs[AliasesHashTable[2*i]].TheDef) << ", "
           << getQualifiedName(Regs[AliasesHashTable[2*i+1]].TheDef) << ", \n";
      } else {
        OS << Namespace << "::NoRegister, " << Namespace << "::NoRegister, \n";
      }
    }

    unsigned Idx = AliasesHashTableSize*2-2;
    if (AliasesHashTable[Idx] != ~0U) {
      OS << "                                       "
         << getQualifiedName(Regs[AliasesHashTable[Idx]].TheDef) << ", "
         << getQualifiedName(Regs[AliasesHashTable[Idx+1]].TheDef) << " };\n";
    } else {
      OS << Namespace << "::NoRegister, " << Namespace << "::NoRegister };\n";
    }

    OS << "  const unsigned AliasesHashTableSize = "
       << AliasesHashTableSize << ";\n";
  } else {
    OS << "  const unsigned AliasesHashTable[] = { ~0U, ~0U };\n"
       << "  const unsigned AliasesHashTableSize = 1;\n";
  }

  delete [] AliasesHashTable;

  if (!RegisterAliases.empty())
    OS << "\n\n  // Register Overlap Lists...\n";

  // Emit an overlap list for all registers.
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    const CodeGenRegister *Reg = &Regs[i];
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
    const CodeGenRegister &Reg = Regs[i];
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
    const CodeGenRegister &Reg = Regs[i];
    const CodeGenRegister::SuperRegList &SR = Reg.getSuperRegs();
    if (SR.empty())
      continue;
    OS << "  const unsigned " << Reg.getName() << "_SuperRegsSet[] = { ";
    for (unsigned j = 0, je = SR.size(); j != je; ++j)
      OS << getQualifiedName(SR[j]->TheDef) << ", ";
    OS << "0 };\n";
  }

  OS<<"\n  const TargetRegisterDesc RegisterDescriptors[] = { // Descriptors\n";
  OS << "    { \"NOREG\",\t0,\t0,\t0,\t0,\t0 },\n";

  // Now that register alias and sub-registers sets have been emitted, emit the
  // register descriptors now.
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    const CodeGenRegister &Reg = Regs[i];
    OS << "    { \"";
    OS << Reg.getName() << "\",\t" << Reg.getName() << "_Overlaps,\t";
    if (!Reg.getSubRegs().empty())
      OS << Reg.getName() << "_SubRegsSet,\t";
    else
      OS << "Empty_SubRegsSet,\t";
    if (!Reg.getSuperRegs().empty())
      OS << Reg.getName() << "_SuperRegsSet,\t";
    else
      OS << "Empty_SuperRegsSet,\t";
    OS << Reg.CostPerUse << ",\t"
       << int(AllocatableRegs.count(Reg.TheDef)) << " },\n";
  }
  OS << "  };\n";      // End of register descriptors...

  // Calculate the mapping of subregister+index pairs to physical registers.
  // This will also create further anonymous indexes.
  unsigned NamedIndices = RegBank.getNumNamedIndices();

  // Emit SubRegIndex names, skipping 0
  const std::vector<Record*> &SubRegIndices = RegBank.getSubRegIndices();
  OS << "\n  const char *const SubRegIndexTable[] = { \"";
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
  OS << "}\n\n";       // End of anonymous namespace...

  std::string ClassName = Target.getName() + "GenRegisterInfo";

  // Emit the subregister + index mapping function based on the information
  // calculated above.
  OS << "unsigned " << ClassName
     << "::getSubReg(unsigned RegNo, unsigned Index) const {\n"
     << "  switch (RegNo) {\n"
     << "  default:\n    return 0;\n";
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    const CodeGenRegister::SubRegMap &SRM = Regs[i].getSubRegs();
    if (SRM.empty())
      continue;
    OS << "  case " << getQualifiedName(Regs[i].TheDef) << ":\n";
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
     const CodeGenRegister::SubRegMap &SRM = Regs[i].getSubRegs();
     if (SRM.empty())
       continue;
    OS << "  case " << getQualifiedName(Regs[i].TheDef) << ":\n";
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
  OS << ClassName << "::" << ClassName
     << "(int CallFrameSetupOpcode, int CallFrameDestroyOpcode)\n"
     << "  : TargetRegisterInfo(RegisterDescriptors, " << Regs.size()+1
     << ", RegisterClasses, RegisterClasses+" << RegisterClasses.size() <<",\n"
     << "                 SubRegIndexTable,\n"
     << "                 CallFrameSetupOpcode, CallFrameDestroyOpcode,\n"
     << "                 SubregHashTable, SubregHashTableSize,\n"
     << "                 AliasesHashTable, AliasesHashTableSize) {\n"
     << "}\n\n";

  // Collect all information about dwarf register numbers

  // First, just pull all provided information to the map
  unsigned maxLength = 0;
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    Record *Reg = Regs[i].TheDef;
    std::vector<int64_t> RegNums = Reg->getValueAsListOfInts("DwarfNumbers");
    maxLength = std::max((size_t)maxLength, RegNums.size());
    if (DwarfRegNums.count(Reg))
      errs() << "Warning: DWARF numbers for register " << getQualifiedName(Reg)
             << "specified multiple times\n";
    DwarfRegNums[Reg] = RegNums;
  }

  // Now we know maximal length of number list. Append -1's, where needed
  for (DwarfRegNumsMapTy::iterator
       I = DwarfRegNums.begin(), E = DwarfRegNums.end(); I != E; ++I)
    for (unsigned i = I->second.size(), e = maxLength; i != e; ++i)
      I->second.push_back(-1);

  // Emit reverse information about the dwarf register numbers.
  OS << "int " << ClassName << "::getLLVMRegNumFull(unsigned DwarfRegNum, "
     << "unsigned Flavour) const {\n"
     << "  switch (Flavour) {\n"
     << "  default:\n"
     << "    assert(0 && \"Unknown DWARF flavour\");\n"
     << "    return -1;\n";

  for (unsigned i = 0, e = maxLength; i != e; ++i) {
    OS << "  case " << i << ":\n"
       << "    switch (DwarfRegNum) {\n"
       << "    default:\n"
       << "      assert(0 && \"Invalid DwarfRegNum\");\n"
       << "      return -1;\n";

    for (DwarfRegNumsMapTy::iterator
           I = DwarfRegNums.begin(), E = DwarfRegNums.end(); I != E; ++I) {
      int DwarfRegNo = I->second[i];
      if (DwarfRegNo >= 0)
        OS << "    case " <<  DwarfRegNo << ":\n"
           << "      return " << getQualifiedName(I->first) << ";\n";
    }
    OS << "    };\n";
  }

  OS << "  };\n}\n\n";

  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    Record *Reg = Regs[i].TheDef;
    const RecordVal *V = Reg->getValue("DwarfAlias");
    if (!V || !V->getValue())
      continue;

    DefInit *DI = dynamic_cast<DefInit*>(V->getValue());
    Record *Alias = DI->getDef();
    DwarfRegNums[Reg] = DwarfRegNums[Alias];
  }

  // Emit information about the dwarf register numbers.
  OS << "int " << ClassName << "::getDwarfRegNumFull(unsigned RegNum, "
     << "unsigned Flavour) const {\n"
     << "  switch (Flavour) {\n"
     << "  default:\n"
     << "    assert(0 && \"Unknown DWARF flavour\");\n"
     << "    return -1;\n";

  for (unsigned i = 0, e = maxLength; i != e; ++i) {
    OS << "  case " << i << ":\n"
       << "    switch (RegNum) {\n"
       << "    default:\n"
       << "      assert(0 && \"Invalid RegNum\");\n"
       << "      return -1;\n";

    // Sort by name to get a stable order.


    for (DwarfRegNumsMapTy::iterator
           I = DwarfRegNums.begin(), E = DwarfRegNums.end(); I != E; ++I) {
      int RegNo = I->second[i];
      OS << "    case " << getQualifiedName(I->first) << ":\n"
         << "      return " << RegNo << ";\n";
    }
    OS << "    };\n";
  }

  OS << "  };\n}\n\n";

  OS << "} // End llvm namespace \n";
}
