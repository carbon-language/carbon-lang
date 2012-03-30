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
#include "SequenceToOffsetTable.h"
#include "llvm/TableGen/Record.h"
#include "llvm/ADT/BitVector.h"
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

  // Register enums are stored as uint16_t in the tables. Make sure we'll fit
  assert(Registers.size() <= 0xffff && "Too many regs to fit in tables");

  std::string Namespace = Registers[0]->TheDef->getValueAsString("Namespace");

  EmitSourceFileHeader("Target Register Enum Values", OS);

  OS << "\n#ifdef GET_REGINFO_ENUM\n";
  OS << "#undef GET_REGINFO_ENUM\n";

  OS << "namespace llvm {\n\n";

  OS << "class MCRegisterClass;\n"
     << "extern const MCRegisterClass " << Namespace
     << "MCRegisterClasses[];\n\n";

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

  ArrayRef<CodeGenRegisterClass*> RegisterClasses = Bank.getRegClasses();
  if (!RegisterClasses.empty()) {

    // RegisterClass enums are stored as uint16_t in the tables.
    assert(RegisterClasses.size() <= 0xffff &&
           "Too many register classes to fit in tables");

    OS << "\n// Register classes\n";
    if (!Namespace.empty())
      OS << "namespace " << Namespace << " {\n";
    OS << "enum {\n";
    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      if (i) OS << ",\n";
      OS << "  " << RegisterClasses[i]->getName() << "RegClassID";
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

  ArrayRef<CodeGenSubRegIndex*> SubRegIndices = Bank.getSubRegIndices();
  if (!SubRegIndices.empty()) {
    OS << "\n// Subregister indices\n";
    std::string Namespace =
      SubRegIndices[0]->getNamespace();
    if (!Namespace.empty())
      OS << "namespace " << Namespace << " {\n";
    OS << "enum {\n  NoSubRegister,\n";
    for (unsigned i = 0, e = Bank.getNumNamedIndices(); i != e; ++i)
      OS << "  " << SubRegIndices[i]->getName() << ",\t// " << i+1 << "\n";
    OS << "  NUM_TARGET_NAMED_SUBREGS\n};\n";
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
     << "    llvm_unreachable(\"Unknown DWARF flavour\");\n";

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
       << "    llvm_unreachable(\"Unknown DWARF flavour\");\n";

    for (unsigned i = 0, e = maxLength; i != e; ++i) {
      OS << "  case " << i << ":\n";
      // Sort by name to get a stable order.
      for (DwarfRegNumsMapTy::iterator
             I = DwarfRegNums.begin(), E = DwarfRegNums.end(); I != E; ++I) {
        int RegNo = I->second[i];
        if (RegNo == -1) // -1 is the default value, don't emit a mapping.
          continue;

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

// Print a BitVector as a sequence of hex numbers using a little-endian mapping.
// Width is the number of bits per hex number.
static void printBitVectorAsHex(raw_ostream &OS,
                                const BitVector &Bits,
                                unsigned Width) {
  assert(Width <= 32 && "Width too large");
  unsigned Digits = (Width + 3) / 4;
  for (unsigned i = 0, e = Bits.size(); i < e; i += Width) {
    unsigned Value = 0;
    for (unsigned j = 0; j != Width && i + j != e; ++j)
      Value |= Bits.test(i + j) << j;
    OS << format("0x%0*x, ", Digits, Value);
  }
}

// Helper to emit a set of bits into a constant byte array.
class BitVectorEmitter {
  BitVector Values;
public:
  void add(unsigned v) {
    if (v >= Values.size())
      Values.resize(((v/8)+1)*8); // Round up to the next byte.
    Values[v] = true;
  }

  void print(raw_ostream &OS) {
    printBitVectorAsHex(OS, Values, 8);
  }
};

static void printRegister(raw_ostream &OS, const CodeGenRegister *Reg) {
  OS << getQualifiedName(Reg->TheDef);
}

static void printSimpleValueType(raw_ostream &OS, MVT::SimpleValueType VT) {
  OS << getEnumName(VT);
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

  const std::vector<CodeGenRegister*> &Regs = RegBank.getRegisters();
  std::map<const CodeGenRegister*, CodeGenRegister::Set> Overlaps;
  RegBank.computeOverlaps(Overlaps);

  // The lists of sub-registers, super-registers, and overlaps all go in the
  // same array. That allows us to share suffixes.
  typedef std::vector<const CodeGenRegister*> RegVec;
  SmallVector<RegVec, 4> SubRegLists(Regs.size());
  SmallVector<RegVec, 4> OverlapLists(Regs.size());
  SequenceToOffsetTable<RegVec, CodeGenRegister::Less> RegSeqs;

  // Precompute register lists for the SequenceToOffsetTable.
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    const CodeGenRegister *Reg = Regs[i];

    // Compute the ordered sub-register list.
    SetVector<const CodeGenRegister*> SR;
    Reg->addSubRegsPreOrder(SR, RegBank);
    RegVec &SubRegList = SubRegLists[i];
    SubRegList.assign(SR.begin(), SR.end());
    RegSeqs.add(SubRegList);

    // Super-registers are already computed.
    const RegVec &SuperRegList = Reg->getSuperRegs();
    RegSeqs.add(SuperRegList);

    // The list of overlaps doesn't need to have any particular order, except
    // Reg itself must be the first element. Pick an ordering that has one of
    // the other lists as a suffix.
    RegVec &OverlapList = OverlapLists[i];
    const RegVec &Suffix = SubRegList.size() > SuperRegList.size() ?
                           SubRegList : SuperRegList;
    CodeGenRegister::Set Omit(Suffix.begin(), Suffix.end());

    // First element is Reg itself.
    OverlapList.push_back(Reg);
    Omit.insert(Reg);

    // Any elements not in Suffix.
    const CodeGenRegister::Set &OSet = Overlaps[Reg];
    std::set_difference(OSet.begin(), OSet.end(),
                        Omit.begin(), Omit.end(),
                        std::back_inserter(OverlapList));

    // Finally, Suffix itself.
    OverlapList.insert(OverlapList.end(), Suffix.begin(), Suffix.end());
    RegSeqs.add(OverlapList);
  }

  // Compute the final layout of the sequence table.
  RegSeqs.layout();

  OS << "namespace llvm {\n\n";

  const std::string &TargetName = Target.getName();

  // Emit the shared table of register lists.
  OS << "extern const uint16_t " << TargetName << "RegLists[] = {\n";
  RegSeqs.emit(OS, printRegister);
  OS << "};\n\n";

  OS << "extern const MCRegisterDesc " << TargetName
     << "RegDesc[] = { // Descriptors\n";
  OS << "  { \"NOREG\", 0, 0, 0 },\n";

  // Emit the register descriptors now.
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    const CodeGenRegister *Reg = Regs[i];
    OS << "  { \"" << Reg->getName() << "\", "
       << RegSeqs.get(OverlapLists[i]) << ", "
       << RegSeqs.get(SubRegLists[i]) << ", "
       << RegSeqs.get(Reg->getSuperRegs()) << " },\n";
  }
  OS << "};\n\n";      // End of register descriptors...

  ArrayRef<CodeGenRegisterClass*> RegisterClasses = RegBank.getRegClasses();

  // Loop over all of the register classes... emitting each one.
  OS << "namespace {     // Register classes...\n";

  // Emit the register enum value arrays for each RegisterClass
  for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
    const CodeGenRegisterClass &RC = *RegisterClasses[rc];
    ArrayRef<Record*> Order = RC.getOrder();

    // Give the register class a legal C name if it's anonymous.
    std::string Name = RC.getName();

    // Emit the register list now.
    OS << "  // " << Name << " Register Class...\n"
       << "  const uint16_t " << Name
       << "[] = {\n    ";
    for (unsigned i = 0, e = Order.size(); i != e; ++i) {
      Record *Reg = Order[i];
      OS << getQualifiedName(Reg) << ", ";
    }
    OS << "\n  };\n\n";

    OS << "  // " << Name << " Bit set.\n"
       << "  const uint8_t " << Name
       << "Bits[] = {\n    ";
    BitVectorEmitter BVE;
    for (unsigned i = 0, e = Order.size(); i != e; ++i) {
      Record *Reg = Order[i];
      BVE.add(Target.getRegBank().getReg(Reg)->EnumValue);
    }
    BVE.print(OS);
    OS << "\n  };\n\n";

  }
  OS << "}\n\n";

  OS << "extern const MCRegisterClass " << TargetName
     << "MCRegisterClasses[] = {\n";

  for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
    const CodeGenRegisterClass &RC = *RegisterClasses[rc];

    // Asserts to make sure values will fit in table assuming types from
    // MCRegisterInfo.h
    assert((RC.SpillSize/8) <= 0xffff && "SpillSize too large.");
    assert((RC.SpillAlignment/8) <= 0xffff && "SpillAlignment too large.");
    assert(RC.CopyCost >= -128 && RC.CopyCost <= 127 && "Copy cost too large.");

    OS << "  { " << '\"' << RC.getName() << "\", "
       << RC.getName() << ", " << RC.getName() << "Bits, "
       << RC.getOrder().size() << ", sizeof(" << RC.getName() << "Bits), "
       << RC.getQualifiedName() + "RegClassID" << ", "
       << RC.SpillSize/8 << ", "
       << RC.SpillAlignment/8 << ", "
       << RC.CopyCost << ", "
       << RC.Allocatable << " },\n";
  }

  OS << "};\n\n";

  // Emit the data table for getSubReg().
  ArrayRef<CodeGenSubRegIndex*> SubRegIndices = RegBank.getSubRegIndices();
  if (SubRegIndices.size()) {
    OS << "const uint16_t " << TargetName << "SubRegTable[]["
       << SubRegIndices.size() << "] = {\n";
    for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
      const CodeGenRegister::SubRegMap &SRM = Regs[i]->getSubRegs();
      OS << "  /* " << Regs[i]->TheDef->getName() << " */\n";
      if (SRM.empty()) {
        OS << "  {0},\n";
        continue;
      }
      OS << "  {";
      for (unsigned j = 0, je = SubRegIndices.size(); j != je; ++j) {
        // FIXME: We really should keep this to 80 columns...
        CodeGenRegister::SubRegMap::const_iterator SubReg =
          SRM.find(SubRegIndices[j]);
        if (SubReg != SRM.end())
          OS << getQualifiedName(SubReg->second->TheDef);
        else
          OS << "0";
        if (j != je - 1)
          OS << ", ";
      }
      OS << "}" << (i != e ? "," : "") << "\n";
    }
    OS << "};\n\n";
    OS << "const uint16_t *get" << TargetName
       << "SubRegTable() {\n  return (const uint16_t *)" << TargetName
       << "SubRegTable;\n}\n\n";
  }

  // MCRegisterInfo initialization routine.
  OS << "static inline void Init" << TargetName
     << "MCRegisterInfo(MCRegisterInfo *RI, unsigned RA, "
     << "unsigned DwarfFlavour = 0, unsigned EHFlavour = 0) {\n";
  OS << "  RI->InitMCRegisterInfo(" << TargetName << "RegDesc, "
     << Regs.size()+1 << ", RA, " << TargetName << "MCRegisterClasses, "
     << RegisterClasses.size() << ", " << TargetName << "RegLists, ";
  if (SubRegIndices.size() != 0)
    OS << "(uint16_t*)" << TargetName << "SubRegTable, "
       << SubRegIndices.size() << ");\n\n";
  else
    OS << "NULL, 0);\n\n";

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
     << "  unsigned composeSubRegIndices(unsigned, unsigned) const;\n"
     << "  const TargetRegisterClass *"
        "getSubClassWithSubReg(const TargetRegisterClass*, unsigned) const;\n"
     << "  const TargetRegisterClass *getMatchingSuperRegClass("
        "const TargetRegisterClass*, const TargetRegisterClass*, "
        "unsigned) const;\n"
     << "};\n\n";

  ArrayRef<CodeGenRegisterClass*> RegisterClasses = RegBank.getRegClasses();

  if (!RegisterClasses.empty()) {
    OS << "namespace " << RegisterClasses[0]->Namespace
       << " { // Register classes\n";

    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      const CodeGenRegisterClass &RC = *RegisterClasses[i];
      const std::string &Name = RC.getName();

      // Output the extern for the instance.
      OS << "  extern const TargetRegisterClass " << Name << "RegClass;\n";
      // Output the extern for the pointer to the instance (should remove).
      OS << "  static const TargetRegisterClass * const " << Name
         << "RegisterClass = &" << Name << "RegClass;\n";
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

  // Get access to MCRegisterClass data.
  OS << "extern const MCRegisterClass " << Target.getName()
     << "MCRegisterClasses[];\n";

  // Start out by emitting each of the register classes.
  ArrayRef<CodeGenRegisterClass*> RegisterClasses = RegBank.getRegClasses();

  // Collect all registers belonging to any allocatable class.
  std::set<Record*> AllocatableRegs;

  // Collect allocatable registers.
  for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
    const CodeGenRegisterClass &RC = *RegisterClasses[rc];
    ArrayRef<Record*> Order = RC.getOrder();

    if (RC.Allocatable)
      AllocatableRegs.insert(Order.begin(), Order.end());
  }

  // Build a shared array of value types.
  SequenceToOffsetTable<std::vector<MVT::SimpleValueType> > VTSeqs;
  for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc)
    VTSeqs.add(RegisterClasses[rc]->VTs);
  VTSeqs.layout();
  OS << "\nstatic const MVT::SimpleValueType VTLists[] = {\n";
  VTSeqs.emit(OS, printSimpleValueType, "MVT::Other");
  OS << "};\n";

  // Now that all of the structs have been emitted, emit the instances.
  if (!RegisterClasses.empty()) {
    std::map<unsigned, std::set<unsigned> > SuperRegClassMap;

    OS << "\nstatic const TargetRegisterClass *const "
       << "NullRegClasses[] = { NULL };\n\n";

    unsigned NumSubRegIndices = RegBank.getSubRegIndices().size();

    if (NumSubRegIndices) {
      // Compute the super-register classes for each RegisterClass
      for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
        const CodeGenRegisterClass &RC = *RegisterClasses[rc];
        for (DenseMap<Record*,Record*>::const_iterator
             i = RC.SubRegClasses.begin(),
             e = RC.SubRegClasses.end(); i != e; ++i) {
          // Find the register class number of i->second for SuperRegClassMap.
          const CodeGenRegisterClass *RC2 = RegBank.getRegClass(i->second);
          assert(RC2 && "Invalid register class in SubRegClasses");
          SuperRegClassMap[RC2->EnumValue].insert(rc);
        }
      }

      // Emit the super-register classes for each RegisterClass
      for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
        const CodeGenRegisterClass &RC = *RegisterClasses[rc];

        // Give the register class a legal C name if it's anonymous.
        std::string Name = RC.getName();

        OS << "// " << Name
           << " Super-register Classes...\n"
           << "static const TargetRegisterClass *const "
           << Name << "SuperRegClasses[] = {\n  ";

        bool Empty = true;
        std::map<unsigned, std::set<unsigned> >::iterator I =
          SuperRegClassMap.find(rc);
        if (I != SuperRegClassMap.end()) {
          for (std::set<unsigned>::iterator II = I->second.begin(),
                 EE = I->second.end(); II != EE; ++II) {
            const CodeGenRegisterClass &RC2 = *RegisterClasses[*II];
            if (!Empty)
              OS << ", ";
            OS << "&" << RC2.getQualifiedName() << "RegClass";
            Empty = false;
          }
        }

        OS << (!Empty ? ", " : "") << "NULL";
        OS << "\n};\n\n";
      }
    }

    // Emit the sub-classes array for each RegisterClass
    for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
      const CodeGenRegisterClass &RC = *RegisterClasses[rc];

      // Give the register class a legal C name if it's anonymous.
      std::string Name = RC.getName();

      OS << "static const uint32_t " << Name << "SubclassMask[] = {\n  ";
      printBitVectorAsHex(OS, RC.getSubClasses(), 32);
      OS << "\n};\n\n";
    }

    // Emit NULL terminated super-class lists.
    for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
      const CodeGenRegisterClass &RC = *RegisterClasses[rc];
      ArrayRef<CodeGenRegisterClass*> Supers = RC.getSuperClasses();

      // Skip classes without supers.  We can reuse NullRegClasses.
      if (Supers.empty())
        continue;

      OS << "static const TargetRegisterClass *const "
         << RC.getName() << "Superclasses[] = {\n";
      for (unsigned i = 0; i != Supers.size(); ++i)
        OS << "  &" << Supers[i]->getQualifiedName() << "RegClass,\n";
      OS << "  NULL\n};\n\n";
    }

    // Emit methods.
    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      const CodeGenRegisterClass &RC = *RegisterClasses[i];
      if (!RC.AltOrderSelect.empty()) {
        OS << "\nstatic inline unsigned " << RC.getName()
           << "AltOrderSelect(const MachineFunction &MF) {"
           << RC.AltOrderSelect << "}\n\n"
           << "static ArrayRef<uint16_t> " << RC.getName()
           << "GetRawAllocationOrder(const MachineFunction &MF) {\n";
        for (unsigned oi = 1 , oe = RC.getNumOrders(); oi != oe; ++oi) {
          ArrayRef<Record*> Elems = RC.getOrder(oi);
          if (!Elems.empty()) {
            OS << "  static const uint16_t AltOrder" << oi << "[] = {";
            for (unsigned elem = 0; elem != Elems.size(); ++elem)
              OS << (elem ? ", " : " ") << getQualifiedName(Elems[elem]);
            OS << " };\n";
          }
        }
        OS << "  const MCRegisterClass &MCR = " << Target.getName()
           << "MCRegisterClasses[" << RC.getQualifiedName() + "RegClassID];\n"
           << "  const ArrayRef<uint16_t> Order[] = {\n"
           << "    makeArrayRef(MCR.begin(), MCR.getNumRegs()";
        for (unsigned oi = 1, oe = RC.getNumOrders(); oi != oe; ++oi)
          if (RC.getOrder(oi).empty())
            OS << "),\n    ArrayRef<uint16_t>(";
          else
            OS << "),\n    makeArrayRef(AltOrder" << oi;
        OS << ")\n  };\n  const unsigned Select = " << RC.getName()
           << "AltOrderSelect(MF);\n  assert(Select < " << RC.getNumOrders()
           << ");\n  return Order[Select];\n}\n";
        }
    }

    // Now emit the actual value-initialized register class instances.
    OS << "namespace " << RegisterClasses[0]->Namespace
       << " {   // Register class instances\n";

    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      const CodeGenRegisterClass &RC = *RegisterClasses[i];
      OS << "  extern const TargetRegisterClass "
         << RegisterClasses[i]->getName() << "RegClass = {\n    "
         << '&' << Target.getName() << "MCRegisterClasses[" << RC.getName()
         << "RegClassID],\n    "
         << "VTLists + " << VTSeqs.get(RC.VTs) << ",\n    "
         << RC.getName() << "SubclassMask,\n    ";
      if (RC.getSuperClasses().empty())
        OS << "NullRegClasses,\n    ";
      else
        OS << RC.getName() << "Superclasses,\n    ";
      OS << (NumSubRegIndices ? RC.getName() + "Super" : std::string("Null"))
         << "RegClasses,\n    ";
      if (RC.AltOrderSelect.empty())
        OS << "0\n";
      else
        OS << RC.getName() << "GetRawAllocationOrder\n";
      OS << "  };\n\n";
    }

    OS << "}\n";
  }

  OS << "\nnamespace {\n";
  OS << "  const TargetRegisterClass* const RegisterClasses[] = {\n";
  for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i)
    OS << "    &" << RegisterClasses[i]->getQualifiedName()
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
  // This will also create further anonymous indices.
  unsigned NamedIndices = RegBank.getNumNamedIndices();

  // Emit SubRegIndex names, skipping 0
  ArrayRef<CodeGenSubRegIndex*> SubRegIndices = RegBank.getSubRegIndices();
  OS << "\n  static const char *const " << TargetName
     << "SubRegIndexTable[] = { \"";
  for (unsigned i = 0, e = SubRegIndices.size(); i != e; ++i) {
    OS << SubRegIndices[i]->getName();
    if (i+1 != e)
      OS << "\", \"";
  }
  OS << "\" };\n\n";

  // Emit names of the anonymous subreg indices.
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

  // Emit composeSubRegIndices
  OS << "unsigned " << ClassName
     << "::composeSubRegIndices(unsigned IdxA, unsigned IdxB) const {\n"
     << "  switch (IdxA) {\n"
     << "  default:\n    return IdxB;\n";
  for (unsigned i = 0, e = SubRegIndices.size(); i != e; ++i) {
    bool Open = false;
    for (unsigned j = 0; j != e; ++j) {
      if (CodeGenSubRegIndex *Comp =
            SubRegIndices[i]->compose(SubRegIndices[j])) {
        if (!Open) {
          OS << "  case " << SubRegIndices[i]->getQualifiedName()
             << ": switch(IdxB) {\n    default: return IdxB;\n";
          Open = true;
        }
        OS << "    case " << SubRegIndices[j]->getQualifiedName()
           << ": return " << Comp->getQualifiedName() << ";\n";
      }
    }
    if (Open)
      OS << "    }\n";
  }
  OS << "  }\n}\n\n";

  // Emit getSubClassWithSubReg.
  OS << "const TargetRegisterClass *" << ClassName
     << "::getSubClassWithSubReg(const TargetRegisterClass *RC, unsigned Idx)"
        " const {\n";
  if (SubRegIndices.empty()) {
    OS << "  assert(Idx == 0 && \"Target has no sub-registers\");\n"
       << "  return RC;\n";
  } else {
    // Use the smallest type that can hold a regclass ID with room for a
    // sentinel.
    if (RegisterClasses.size() < UINT8_MAX)
      OS << "  static const uint8_t Table[";
    else if (RegisterClasses.size() < UINT16_MAX)
      OS << "  static const uint16_t Table[";
    else
      throw "Too many register classes.";
    OS << RegisterClasses.size() << "][" << SubRegIndices.size() << "] = {\n";
    for (unsigned rci = 0, rce = RegisterClasses.size(); rci != rce; ++rci) {
      const CodeGenRegisterClass &RC = *RegisterClasses[rci];
      OS << "    {\t// " << RC.getName() << "\n";
      for (unsigned sri = 0, sre = SubRegIndices.size(); sri != sre; ++sri) {
        CodeGenSubRegIndex *Idx = SubRegIndices[sri];
        if (CodeGenRegisterClass *SRC = RC.getSubClassWithSubReg(Idx))
          OS << "      " << SRC->EnumValue + 1 << ",\t// " << Idx->getName()
             << " -> " << SRC->getName() << "\n";
        else
          OS << "      0,\t// " << Idx->getName() << "\n";
      }
      OS << "    },\n";
    }
    OS << "  };\n  assert(RC && \"Missing regclass\");\n"
       << "  if (!Idx) return RC;\n  --Idx;\n"
       << "  assert(Idx < " << SubRegIndices.size() << " && \"Bad subreg\");\n"
       << "  unsigned TV = Table[RC->getID()][Idx];\n"
       << "  return TV ? getRegClass(TV - 1) : 0;\n";
  }
  OS << "}\n\n";

  // Emit getMatchingSuperRegClass.
  OS << "const TargetRegisterClass *" << ClassName
     << "::getMatchingSuperRegClass(const TargetRegisterClass *A,"
        " const TargetRegisterClass *B, unsigned Idx) const {\n";
  if (SubRegIndices.empty()) {
    OS << "  llvm_unreachable(\"Target has no sub-registers\");\n";
  } else {
    // We need to find the largest sub-class of A such that every register has
    // an Idx sub-register in B.  Map (B, Idx) to a bit-vector of
    // super-register classes that map into B. Then compute the largest common
    // sub-class with A by taking advantage of the register class ordering,
    // like getCommonSubClass().

    // Bitvector table is NumRCs x NumSubIndexes x BVWords, where BVWords is
    // the number of 32-bit words required to represent all register classes.
    const unsigned BVWords = (RegisterClasses.size()+31)/32;
    BitVector BV(RegisterClasses.size());

    OS << "  static const uint32_t Table[" << RegisterClasses.size()
       << "][" << SubRegIndices.size() << "][" << BVWords << "] = {\n";
    for (unsigned rci = 0, rce = RegisterClasses.size(); rci != rce; ++rci) {
      const CodeGenRegisterClass &RC = *RegisterClasses[rci];
      OS << "    {\t// " << RC.getName() << "\n";
      for (unsigned sri = 0, sre = SubRegIndices.size(); sri != sre; ++sri) {
        CodeGenSubRegIndex *Idx = SubRegIndices[sri];
        BV.reset();
        RC.getSuperRegClasses(Idx, BV);
        OS << "      { ";
        printBitVectorAsHex(OS, BV, 32);
        OS << "},\t// " << Idx->getName() << '\n';
      }
      OS << "    },\n";
    }
    OS << "  };\n  assert(A && B && \"Missing regclass\");\n"
       << "  --Idx;\n"
       << "  assert(Idx < " << SubRegIndices.size() << " && \"Bad subreg\");\n"
       << "  const uint32_t *TV = Table[B->getID()][Idx];\n"
       << "  const uint32_t *SC = A->getSubClassMask();\n"
       << "  for (unsigned i = 0; i != " << BVWords << "; ++i)\n"
       << "    if (unsigned Common = TV[i] & SC[i])\n"
       << "      return getRegClass(32*i + CountTrailingZeros_32(Common));\n"
       << "  return 0;\n";
  }
  OS << "}\n\n";

  // Emit the constructor of the class...
  OS << "extern const MCRegisterDesc " << TargetName << "RegDesc[];\n";
  OS << "extern const uint16_t " << TargetName << "RegLists[];\n";
  if (SubRegIndices.size() != 0)
    OS << "extern const uint16_t *get" << TargetName
       << "SubRegTable();\n";

  OS << ClassName << "::\n" << ClassName
     << "(unsigned RA, unsigned DwarfFlavour, unsigned EHFlavour)\n"
     << "  : TargetRegisterInfo(" << TargetName << "RegInfoDesc"
     << ", RegisterClasses, RegisterClasses+" << RegisterClasses.size() <<",\n"
     << "             " << TargetName << "SubRegIndexTable) {\n"
     << "  InitMCRegisterInfo(" << TargetName << "RegDesc, "
     << Regs.size()+1 << ", RA,\n                     " << TargetName
     << "MCRegisterClasses, " << RegisterClasses.size() << ",\n"
     << "                     " << TargetName << "RegLists,\n"
     << "                     ";
  if (SubRegIndices.size() != 0)
    OS << "get" << TargetName << "SubRegTable(), "
       << SubRegIndices.size() << ");\n\n";
  else
    OS << "NULL, 0);\n\n";

  EmitRegMapping(OS, Regs, true);

  OS << "}\n\n";


  // Emit CalleeSavedRegs information.
  std::vector<Record*> CSRSets =
    Records.getAllDerivedDefinitions("CalleeSavedRegs");
  for (unsigned i = 0, e = CSRSets.size(); i != e; ++i) {
    Record *CSRSet = CSRSets[i];
    const SetTheory::RecVec *Regs = RegBank.getSets().expand(CSRSet);
    assert(Regs && "Cannot expand CalleeSavedRegs instance");

    // Emit the *_SaveList list of callee-saved registers.
    OS << "static const uint16_t " << CSRSet->getName()
       << "_SaveList[] = { ";
    for (unsigned r = 0, re = Regs->size(); r != re; ++r)
      OS << getQualifiedName((*Regs)[r]) << ", ";
    OS << "0 };\n";

    // Emit the *_RegMask bit mask of call-preserved registers.
    OS << "static const uint32_t " << CSRSet->getName()
       << "_RegMask[] = { ";
    printBitVectorAsHex(OS, RegBank.computeCoveredRegisters(*Regs), 32);
    OS << "};\n";
  }
  OS << "\n\n";

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
