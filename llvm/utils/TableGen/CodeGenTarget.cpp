//===- CodeGenTarget.cpp - CodeGen Target Class Wrapper -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class wraps target description classes used by the various code
// generation TableGen backends.  This makes it easier to access the data and
// provides a single place that needs to check it for validity.  All of these
// classes throw exceptions on error conditions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenTarget.h"
#include "CodeGenIntrinsics.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
using namespace llvm;

static cl::opt<unsigned>
AsmParserNum("asmparsernum", cl::init(0),
             cl::desc("Make -gen-asm-parser emit assembly parser #N"));

static cl::opt<unsigned>
AsmWriterNum("asmwriternum", cl::init(0),
             cl::desc("Make -gen-asm-writer emit assembly writer #N"));

/// getValueType - Return the MVT::SimpleValueType that the specified TableGen
/// record corresponds to.
MVT::SimpleValueType llvm::getValueType(Record *Rec) {
  return (MVT::SimpleValueType)Rec->getValueAsInt("Value");
}

std::string llvm::getName(MVT::SimpleValueType T) {
  switch (T) {
  case MVT::Other:   return "UNKNOWN";
  case MVT::iPTR:    return "TLI.getPointerTy()";
  case MVT::iPTRAny: return "TLI.getPointerTy()";
  default: return getEnumName(T);
  }
}

std::string llvm::getEnumName(MVT::SimpleValueType T) {
  switch (T) {
  case MVT::Other: return "MVT::Other";
  case MVT::i1:    return "MVT::i1";
  case MVT::i8:    return "MVT::i8";
  case MVT::i16:   return "MVT::i16";
  case MVT::i32:   return "MVT::i32";
  case MVT::i64:   return "MVT::i64";
  case MVT::i128:  return "MVT::i128";
  case MVT::iAny:  return "MVT::iAny";
  case MVT::fAny:  return "MVT::fAny";
  case MVT::vAny:  return "MVT::vAny";
  case MVT::f32:   return "MVT::f32";
  case MVT::f64:   return "MVT::f64";
  case MVT::f80:   return "MVT::f80";
  case MVT::f128:  return "MVT::f128";
  case MVT::ppcf128:  return "MVT::ppcf128";
  case MVT::Flag:  return "MVT::Flag";
  case MVT::isVoid:return "MVT::isVoid";
  case MVT::v2i8:  return "MVT::v2i8";
  case MVT::v4i8:  return "MVT::v4i8";
  case MVT::v8i8:  return "MVT::v8i8";
  case MVT::v16i8: return "MVT::v16i8";
  case MVT::v32i8: return "MVT::v32i8";
  case MVT::v2i16: return "MVT::v2i16";
  case MVT::v4i16: return "MVT::v4i16";
  case MVT::v8i16: return "MVT::v8i16";
  case MVT::v16i16: return "MVT::v16i16";
  case MVT::v2i32: return "MVT::v2i32";
  case MVT::v4i32: return "MVT::v4i32";
  case MVT::v8i32: return "MVT::v8i32";
  case MVT::v1i64: return "MVT::v1i64";
  case MVT::v2i64: return "MVT::v2i64";
  case MVT::v4i64: return "MVT::v4i64";
  case MVT::v2f32: return "MVT::v2f32";
  case MVT::v4f32: return "MVT::v4f32";
  case MVT::v8f32: return "MVT::v8f32";
  case MVT::v2f64: return "MVT::v2f64";
  case MVT::v4f64: return "MVT::v4f64";
  case MVT::Metadata: return "MVT::Metadata";
  case MVT::iPTR:  return "MVT::iPTR";
  case MVT::iPTRAny:  return "MVT::iPTRAny";
  default: assert(0 && "ILLEGAL VALUE TYPE!"); return "";
  }
}

/// getQualifiedName - Return the name of the specified record, with a
/// namespace qualifier if the record contains one.
///
std::string llvm::getQualifiedName(const Record *R) {
  std::string Namespace = R->getValueAsString("Namespace");
  if (Namespace.empty()) return R->getName();
  return Namespace + "::" + R->getName();
}




/// getTarget - Return the current instance of the Target class.
///
CodeGenTarget::CodeGenTarget() {
  std::vector<Record*> Targets = Records.getAllDerivedDefinitions("Target");
  if (Targets.size() == 0)
    throw std::string("ERROR: No 'Target' subclasses defined!");
  if (Targets.size() != 1)
    throw std::string("ERROR: Multiple subclasses of Target defined!");
  TargetRec = Targets[0];
}


const std::string &CodeGenTarget::getName() const {
  return TargetRec->getName();
}

std::string CodeGenTarget::getInstNamespace() const {
  for (inst_iterator i = inst_begin(), e = inst_end(); i != e; ++i) {
    // Make sure not to pick up "TargetOpcode" by accidentally getting
    // the namespace off the PHI instruction or something.
    if ((*i)->Namespace != "TargetOpcode")
      return (*i)->Namespace;
  }

  return "";
}

Record *CodeGenTarget::getInstructionSet() const {
  return TargetRec->getValueAsDef("InstructionSet");
}


/// getAsmParser - Return the AssemblyParser definition for this target.
///
Record *CodeGenTarget::getAsmParser() const {
  std::vector<Record*> LI = TargetRec->getValueAsListOfDefs("AssemblyParsers");
  if (AsmParserNum >= LI.size())
    throw "Target does not have an AsmParser #" + utostr(AsmParserNum) + "!";
  return LI[AsmParserNum];
}

/// getAsmWriter - Return the AssemblyWriter definition for this target.
///
Record *CodeGenTarget::getAsmWriter() const {
  std::vector<Record*> LI = TargetRec->getValueAsListOfDefs("AssemblyWriters");
  if (AsmWriterNum >= LI.size())
    throw "Target does not have an AsmWriter #" + utostr(AsmWriterNum) + "!";
  return LI[AsmWriterNum];
}

void CodeGenTarget::ReadRegisters() const {
  std::vector<Record*> Regs = Records.getAllDerivedDefinitions("Register");
  if (Regs.empty())
    throw std::string("No 'Register' subclasses defined!");

  Registers.reserve(Regs.size());
  Registers.assign(Regs.begin(), Regs.end());
}

CodeGenRegister::CodeGenRegister(Record *R) : TheDef(R) {
  DeclaredSpillSize = R->getValueAsInt("SpillSize");
  DeclaredSpillAlignment = R->getValueAsInt("SpillAlignment");
}

const std::string &CodeGenRegister::getName() const {
  return TheDef->getName();
}

void CodeGenTarget::ReadRegisterClasses() const {
  std::vector<Record*> RegClasses =
    Records.getAllDerivedDefinitions("RegisterClass");
  if (RegClasses.empty())
    throw std::string("No 'RegisterClass' subclasses defined!");

  RegisterClasses.reserve(RegClasses.size());
  RegisterClasses.assign(RegClasses.begin(), RegClasses.end());
}

std::vector<MVT::SimpleValueType> CodeGenTarget::
getRegisterVTs(Record *R) const {
  std::vector<MVT::SimpleValueType> Result;
  const std::vector<CodeGenRegisterClass> &RCs = getRegisterClasses();
  for (unsigned i = 0, e = RCs.size(); i != e; ++i) {
    const CodeGenRegisterClass &RC = RegisterClasses[i];
    for (unsigned ei = 0, ee = RC.Elements.size(); ei != ee; ++ei) {
      if (R == RC.Elements[ei]) {
        const std::vector<MVT::SimpleValueType> &InVTs = RC.getValueTypes();
        Result.insert(Result.end(), InVTs.begin(), InVTs.end());
      }
    }
  }
  
  // Remove duplicates.
  array_pod_sort(Result.begin(), Result.end());
  Result.erase(std::unique(Result.begin(), Result.end()), Result.end());
  return Result;
}


CodeGenRegisterClass::CodeGenRegisterClass(Record *R) : TheDef(R) {
  // Rename anonymous register classes.
  if (R->getName().size() > 9 && R->getName()[9] == '.') {
    static unsigned AnonCounter = 0;
    R->setName("AnonRegClass_"+utostr(AnonCounter++));
  } 
  
  std::vector<Record*> TypeList = R->getValueAsListOfDefs("RegTypes");
  for (unsigned i = 0, e = TypeList.size(); i != e; ++i) {
    Record *Type = TypeList[i];
    if (!Type->isSubClassOf("ValueType"))
      throw "RegTypes list member '" + Type->getName() +
        "' does not derive from the ValueType class!";
    VTs.push_back(getValueType(Type));
  }
  assert(!VTs.empty() && "RegisterClass must contain at least one ValueType!");
  
  std::vector<Record*> RegList = R->getValueAsListOfDefs("MemberList");
  for (unsigned i = 0, e = RegList.size(); i != e; ++i) {
    Record *Reg = RegList[i];
    if (!Reg->isSubClassOf("Register"))
      throw "Register Class member '" + Reg->getName() +
            "' does not derive from the Register class!";
    Elements.push_back(Reg);
  }
  
  std::vector<Record*> SubRegClassList = 
                        R->getValueAsListOfDefs("SubRegClassList");
  for (unsigned i = 0, e = SubRegClassList.size(); i != e; ++i) {
    Record *SubRegClass = SubRegClassList[i];
    if (!SubRegClass->isSubClassOf("RegisterClass"))
      throw "Register Class member '" + SubRegClass->getName() +
            "' does not derive from the RegisterClass class!";
    SubRegClasses.push_back(SubRegClass);
  }  
  
  // Allow targets to override the size in bits of the RegisterClass.
  unsigned Size = R->getValueAsInt("Size");

  Namespace = R->getValueAsString("Namespace");
  SpillSize = Size ? Size : EVT(VTs[0]).getSizeInBits();
  SpillAlignment = R->getValueAsInt("Alignment");
  CopyCost = R->getValueAsInt("CopyCost");
  MethodBodies = R->getValueAsCode("MethodBodies");
  MethodProtos = R->getValueAsCode("MethodProtos");
}

const std::string &CodeGenRegisterClass::getName() const {
  return TheDef->getName();
}

void CodeGenTarget::ReadLegalValueTypes() const {
  const std::vector<CodeGenRegisterClass> &RCs = getRegisterClasses();
  for (unsigned i = 0, e = RCs.size(); i != e; ++i)
    for (unsigned ri = 0, re = RCs[i].VTs.size(); ri != re; ++ri)
      LegalValueTypes.push_back(RCs[i].VTs[ri]);
  
  // Remove duplicates.
  std::sort(LegalValueTypes.begin(), LegalValueTypes.end());
  LegalValueTypes.erase(std::unique(LegalValueTypes.begin(),
                                    LegalValueTypes.end()),
                        LegalValueTypes.end());
}


void CodeGenTarget::ReadInstructions() const {
  std::vector<Record*> Insts = Records.getAllDerivedDefinitions("Instruction");
  if (Insts.size() <= 2)
    throw std::string("No 'Instruction' subclasses defined!");

  // Parse the instructions defined in the .td file.
  std::string InstFormatName =
    getAsmWriter()->getValueAsString("InstFormatName");

  for (unsigned i = 0, e = Insts.size(); i != e; ++i) {
    std::string AsmStr = Insts[i]->getValueAsString(InstFormatName);
    Instructions[Insts[i]] = new CodeGenInstruction(Insts[i], AsmStr);
  }
}

static const CodeGenInstruction *
GetInstByName(const char *Name,
              const DenseMap<const Record*, CodeGenInstruction*> &Insts) {
  const Record *Rec = Records.getDef(Name);
  
  DenseMap<const Record*, CodeGenInstruction*>::const_iterator
    I = Insts.find(Rec);
  if (Rec == 0 || I == Insts.end())
    throw std::string("Could not find '") + Name + "' instruction!";
  return I->second;
}

namespace {
/// SortInstByName - Sorting predicate to sort instructions by name.
///
struct SortInstByName {
  bool operator()(const CodeGenInstruction *Rec1,
                  const CodeGenInstruction *Rec2) const {
    return Rec1->TheDef->getName() < Rec2->TheDef->getName();
  }
};
}

/// getInstructionsByEnumValue - Return all of the instructions defined by the
/// target, ordered by their enum value.
void CodeGenTarget::ComputeInstrsByEnum() const {
  const DenseMap<const Record*, CodeGenInstruction*> &Insts = getInstructions();
  const CodeGenInstruction *PHI = GetInstByName("PHI", Insts);
  const CodeGenInstruction *INLINEASM = GetInstByName("INLINEASM", Insts);
  const CodeGenInstruction *DBG_LABEL = GetInstByName("DBG_LABEL", Insts);
  const CodeGenInstruction *EH_LABEL = GetInstByName("EH_LABEL", Insts);
  const CodeGenInstruction *GC_LABEL = GetInstByName("GC_LABEL", Insts);
  const CodeGenInstruction *KILL = GetInstByName("KILL", Insts);
  const CodeGenInstruction *EXTRACT_SUBREG =
    GetInstByName("EXTRACT_SUBREG", Insts);
  const CodeGenInstruction *INSERT_SUBREG =
    GetInstByName("INSERT_SUBREG", Insts);
  const CodeGenInstruction *IMPLICIT_DEF = GetInstByName("IMPLICIT_DEF", Insts);
  const CodeGenInstruction *SUBREG_TO_REG =
    GetInstByName("SUBREG_TO_REG", Insts);
  const CodeGenInstruction *COPY_TO_REGCLASS =
    GetInstByName("COPY_TO_REGCLASS", Insts);
  const CodeGenInstruction *DBG_VALUE = GetInstByName("DBG_VALUE", Insts);
  const CodeGenInstruction *REG_SEQUENCE = GetInstByName("REG_SEQUENCE", Insts);

  // Print out the rest of the instructions now.
  InstrsByEnum.push_back(PHI);
  InstrsByEnum.push_back(INLINEASM);
  InstrsByEnum.push_back(DBG_LABEL);
  InstrsByEnum.push_back(EH_LABEL);
  InstrsByEnum.push_back(GC_LABEL);
  InstrsByEnum.push_back(KILL);
  InstrsByEnum.push_back(EXTRACT_SUBREG);
  InstrsByEnum.push_back(INSERT_SUBREG);
  InstrsByEnum.push_back(IMPLICIT_DEF);
  InstrsByEnum.push_back(SUBREG_TO_REG);
  InstrsByEnum.push_back(COPY_TO_REGCLASS);
  InstrsByEnum.push_back(DBG_VALUE);
  InstrsByEnum.push_back(REG_SEQUENCE);
  
  unsigned EndOfPredefines = InstrsByEnum.size();
  
  for (DenseMap<const Record*, CodeGenInstruction*>::const_iterator
       I = Insts.begin(), E = Insts.end(); I != E; ++I) {
    const CodeGenInstruction *CGI = I->second;
    if (CGI != PHI &&
        CGI != INLINEASM &&
        CGI != DBG_LABEL &&
        CGI != EH_LABEL &&
        CGI != GC_LABEL &&
        CGI != KILL &&
        CGI != EXTRACT_SUBREG &&
        CGI != INSERT_SUBREG &&
        CGI != IMPLICIT_DEF &&
        CGI != SUBREG_TO_REG &&
        CGI != COPY_TO_REGCLASS &&
        CGI != DBG_VALUE &&
        CGI != REG_SEQUENCE)
      InstrsByEnum.push_back(CGI);
  }
  
  // All of the instructions are now in random order based on the map iteration.
  // Sort them by name.
  std::sort(InstrsByEnum.begin()+EndOfPredefines, InstrsByEnum.end(),
            SortInstByName());
}


/// isLittleEndianEncoding - Return whether this target encodes its instruction
/// in little-endian format, i.e. bits laid out in the order [0..n]
///
bool CodeGenTarget::isLittleEndianEncoding() const {
  return getInstructionSet()->getValueAsBit("isLittleEndianEncoding");
}

//===----------------------------------------------------------------------===//
// ComplexPattern implementation
//
ComplexPattern::ComplexPattern(Record *R) {
  Ty          = ::getValueType(R->getValueAsDef("Ty"));
  NumOperands = R->getValueAsInt("NumOperands");
  SelectFunc  = R->getValueAsString("SelectFunc");
  RootNodes   = R->getValueAsListOfDefs("RootNodes");

  // Parse the properties.
  Properties = 0;
  std::vector<Record*> PropList = R->getValueAsListOfDefs("Properties");
  for (unsigned i = 0, e = PropList.size(); i != e; ++i)
    if (PropList[i]->getName() == "SDNPHasChain") {
      Properties |= 1 << SDNPHasChain;
    } else if (PropList[i]->getName() == "SDNPOptInFlag") {
      Properties |= 1 << SDNPOptInFlag;
    } else if (PropList[i]->getName() == "SDNPMayStore") {
      Properties |= 1 << SDNPMayStore;
    } else if (PropList[i]->getName() == "SDNPMayLoad") {
      Properties |= 1 << SDNPMayLoad;
    } else if (PropList[i]->getName() == "SDNPSideEffect") {
      Properties |= 1 << SDNPSideEffect;
    } else if (PropList[i]->getName() == "SDNPMemOperand") {
      Properties |= 1 << SDNPMemOperand;
    } else if (PropList[i]->getName() == "SDNPVariadic") {
      Properties |= 1 << SDNPVariadic;
    } else {
      errs() << "Unsupported SD Node property '" << PropList[i]->getName()
             << "' on ComplexPattern '" << R->getName() << "'!\n";
      exit(1);
    }
}

//===----------------------------------------------------------------------===//
// CodeGenIntrinsic Implementation
//===----------------------------------------------------------------------===//

std::vector<CodeGenIntrinsic> llvm::LoadIntrinsics(const RecordKeeper &RC,
                                                   bool TargetOnly) {
  std::vector<Record*> I = RC.getAllDerivedDefinitions("Intrinsic");
  
  std::vector<CodeGenIntrinsic> Result;

  for (unsigned i = 0, e = I.size(); i != e; ++i) {
    bool isTarget = I[i]->getValueAsBit("isTarget");
    if (isTarget == TargetOnly)
      Result.push_back(CodeGenIntrinsic(I[i]));
  }
  return Result;
}

CodeGenIntrinsic::CodeGenIntrinsic(Record *R) {
  TheDef = R;
  std::string DefName = R->getName();
  ModRef = WriteMem;
  isOverloaded = false;
  isCommutative = false;
  
  if (DefName.size() <= 4 || 
      std::string(DefName.begin(), DefName.begin() + 4) != "int_")
    throw "Intrinsic '" + DefName + "' does not start with 'int_'!";

  EnumName = std::string(DefName.begin()+4, DefName.end());

  if (R->getValue("GCCBuiltinName"))  // Ignore a missing GCCBuiltinName field.
    GCCBuiltinName = R->getValueAsString("GCCBuiltinName");

  TargetPrefix = R->getValueAsString("TargetPrefix");
  Name = R->getValueAsString("LLVMName");

  if (Name == "") {
    // If an explicit name isn't specified, derive one from the DefName.
    Name = "llvm.";

    for (unsigned i = 0, e = EnumName.size(); i != e; ++i)
      Name += (EnumName[i] == '_') ? '.' : EnumName[i];
  } else {
    // Verify it starts with "llvm.".
    if (Name.size() <= 5 || 
        std::string(Name.begin(), Name.begin() + 5) != "llvm.")
      throw "Intrinsic '" + DefName + "'s name does not start with 'llvm.'!";
  }
  
  // If TargetPrefix is specified, make sure that Name starts with
  // "llvm.<targetprefix>.".
  if (!TargetPrefix.empty()) {
    if (Name.size() < 6+TargetPrefix.size() ||
        std::string(Name.begin() + 5, Name.begin() + 6 + TargetPrefix.size())
        != (TargetPrefix + "."))
      throw "Intrinsic '" + DefName + "' does not start with 'llvm." +
        TargetPrefix + ".'!";
  }
  
  // Parse the list of return types.
  std::vector<MVT::SimpleValueType> OverloadedVTs;
  ListInit *TypeList = R->getValueAsListInit("RetTypes");
  for (unsigned i = 0, e = TypeList->getSize(); i != e; ++i) {
    Record *TyEl = TypeList->getElementAsRecord(i);
    assert(TyEl->isSubClassOf("LLVMType") && "Expected a type!");
    MVT::SimpleValueType VT;
    if (TyEl->isSubClassOf("LLVMMatchType")) {
      unsigned MatchTy = TyEl->getValueAsInt("Number");
      assert(MatchTy < OverloadedVTs.size() &&
             "Invalid matching number!");
      VT = OverloadedVTs[MatchTy];
      // It only makes sense to use the extended and truncated vector element
      // variants with iAny types; otherwise, if the intrinsic is not
      // overloaded, all the types can be specified directly.
      assert(((!TyEl->isSubClassOf("LLVMExtendedElementVectorType") &&
               !TyEl->isSubClassOf("LLVMTruncatedElementVectorType")) ||
              VT == MVT::iAny || VT == MVT::vAny) &&
             "Expected iAny or vAny type");
    } else {
      VT = getValueType(TyEl->getValueAsDef("VT"));
    }
    if (EVT(VT).isOverloaded()) {
      OverloadedVTs.push_back(VT);
      isOverloaded = true;
    }

    // Reject invalid types.
    if (VT == MVT::isVoid)
      throw "Intrinsic '" + DefName + " has void in result type list!";
    
    IS.RetVTs.push_back(VT);
    IS.RetTypeDefs.push_back(TyEl);
  }
  
  // Parse the list of parameter types.
  TypeList = R->getValueAsListInit("ParamTypes");
  for (unsigned i = 0, e = TypeList->getSize(); i != e; ++i) {
    Record *TyEl = TypeList->getElementAsRecord(i);
    assert(TyEl->isSubClassOf("LLVMType") && "Expected a type!");
    MVT::SimpleValueType VT;
    if (TyEl->isSubClassOf("LLVMMatchType")) {
      unsigned MatchTy = TyEl->getValueAsInt("Number");
      assert(MatchTy < OverloadedVTs.size() &&
             "Invalid matching number!");
      VT = OverloadedVTs[MatchTy];
      // It only makes sense to use the extended and truncated vector element
      // variants with iAny types; otherwise, if the intrinsic is not
      // overloaded, all the types can be specified directly.
      assert(((!TyEl->isSubClassOf("LLVMExtendedElementVectorType") &&
               !TyEl->isSubClassOf("LLVMTruncatedElementVectorType")) ||
              VT == MVT::iAny || VT == MVT::vAny) &&
             "Expected iAny or vAny type");
    } else
      VT = getValueType(TyEl->getValueAsDef("VT"));
    
    if (EVT(VT).isOverloaded()) {
      OverloadedVTs.push_back(VT);
      isOverloaded = true;
    }
    
    // Reject invalid types.
    if (VT == MVT::isVoid && i != e-1 /*void at end means varargs*/)
      throw "Intrinsic '" + DefName + " has void in result type list!";
    
    IS.ParamVTs.push_back(VT);
    IS.ParamTypeDefs.push_back(TyEl);
  }

  // Parse the intrinsic properties.
  ListInit *PropList = R->getValueAsListInit("Properties");
  for (unsigned i = 0, e = PropList->getSize(); i != e; ++i) {
    Record *Property = PropList->getElementAsRecord(i);
    assert(Property->isSubClassOf("IntrinsicProperty") &&
           "Expected a property!");
    
    if (Property->getName() == "IntrNoMem")
      ModRef = NoMem;
    else if (Property->getName() == "IntrReadArgMem")
      ModRef = ReadArgMem;
    else if (Property->getName() == "IntrReadMem")
      ModRef = ReadMem;
    else if (Property->getName() == "IntrWriteArgMem")
      ModRef = WriteArgMem;
    else if (Property->getName() == "IntrWriteMem")
      ModRef = WriteMem;
    else if (Property->getName() == "Commutative")
      isCommutative = true;
    else if (Property->isSubClassOf("NoCapture")) {
      unsigned ArgNo = Property->getValueAsInt("ArgNo");
      ArgumentAttributes.push_back(std::make_pair(ArgNo, NoCapture));
    } else
      assert(0 && "Unknown property!");
  }
}
