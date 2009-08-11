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
#include "llvm/Support/CommandLine.h"
#include <algorithm>
using namespace llvm;

static cl::opt<unsigned>
AsmParserNum("asmparsernum", cl::init(0),
             cl::desc("Make -gen-asm-parser emit assembly parser #N"));

static cl::opt<unsigned>
AsmWriterNum("asmwriternum", cl::init(0),
             cl::desc("Make -gen-asm-writer emit assembly writer #N"));

/// getValueType - Return the EVT::SimpleValueType that the specified TableGen
/// record corresponds to.
EVT::SimpleValueType llvm::getValueType(Record *Rec) {
  return (EVT::SimpleValueType)Rec->getValueAsInt("Value");
}

std::string llvm::getName(EVT::SimpleValueType T) {
  switch (T) {
  case EVT::Other:   return "UNKNOWN";
  case EVT::iPTR:    return "TLI.getPointerTy()";
  case EVT::iPTRAny: return "TLI.getPointerTy()";
  default: return getEnumName(T);
  }
}

std::string llvm::getEnumName(EVT::SimpleValueType T) {
  switch (T) {
  case EVT::Other: return "EVT::Other";
  case EVT::i1:    return "EVT::i1";
  case EVT::i8:    return "EVT::i8";
  case EVT::i16:   return "EVT::i16";
  case EVT::i32:   return "EVT::i32";
  case EVT::i64:   return "EVT::i64";
  case EVT::i128:  return "EVT::i128";
  case EVT::iAny:  return "EVT::iAny";
  case EVT::fAny:  return "EVT::fAny";
  case EVT::vAny:  return "EVT::vAny";
  case EVT::f32:   return "EVT::f32";
  case EVT::f64:   return "EVT::f64";
  case EVT::f80:   return "EVT::f80";
  case EVT::f128:  return "EVT::f128";
  case EVT::ppcf128:  return "EVT::ppcf128";
  case EVT::Flag:  return "EVT::Flag";
  case EVT::isVoid:return "EVT::isVoid";
  case EVT::v2i8:  return "EVT::v2i8";
  case EVT::v4i8:  return "EVT::v4i8";
  case EVT::v8i8:  return "EVT::v8i8";
  case EVT::v16i8: return "EVT::v16i8";
  case EVT::v32i8: return "EVT::v32i8";
  case EVT::v2i16: return "EVT::v2i16";
  case EVT::v4i16: return "EVT::v4i16";
  case EVT::v8i16: return "EVT::v8i16";
  case EVT::v16i16: return "EVT::v16i16";
  case EVT::v2i32: return "EVT::v2i32";
  case EVT::v4i32: return "EVT::v4i32";
  case EVT::v8i32: return "EVT::v8i32";
  case EVT::v1i64: return "EVT::v1i64";
  case EVT::v2i64: return "EVT::v2i64";
  case EVT::v4i64: return "EVT::v4i64";
  case EVT::v2f32: return "EVT::v2f32";
  case EVT::v4f32: return "EVT::v4f32";
  case EVT::v8f32: return "EVT::v8f32";
  case EVT::v2f64: return "EVT::v2f64";
  case EVT::v4f64: return "EVT::v4f64";
  case EVT::Metadata: return "EVT::Metadata";
  case EVT::iPTR:  return "EVT::iPTR";
  case EVT::iPTRAny:  return "EVT::iPTRAny";
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
  std::string InstNS;

  for (inst_iterator i = inst_begin(), e = inst_end(); i != e; ++i) {
    InstNS = i->second.Namespace;

    // Make sure not to pick up "TargetInstrInfo" by accidentally getting
    // the namespace off the PHI instruction or something.
    if (InstNS != "TargetInstrInfo")
      break;
  }

  return InstNS;
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

std::vector<unsigned char> CodeGenTarget::getRegisterVTs(Record *R) const {
  std::vector<unsigned char> Result;
  const std::vector<CodeGenRegisterClass> &RCs = getRegisterClasses();
  for (unsigned i = 0, e = RCs.size(); i != e; ++i) {
    const CodeGenRegisterClass &RC = RegisterClasses[i];
    for (unsigned ei = 0, ee = RC.Elements.size(); ei != ee; ++ei) {
      if (R == RC.Elements[ei]) {
        const std::vector<EVT::SimpleValueType> &InVTs = RC.getValueTypes();
        for (unsigned i = 0, e = InVTs.size(); i != e; ++i)
          Result.push_back(InVTs[i]);
      }
    }
  }
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
    Instructions.insert(std::make_pair(Insts[i]->getName(),
                                       CodeGenInstruction(Insts[i], AsmStr)));
  }
}

/// getInstructionsByEnumValue - Return all of the instructions defined by the
/// target, ordered by their enum value.
void CodeGenTarget::
getInstructionsByEnumValue(std::vector<const CodeGenInstruction*>
                                                 &NumberedInstructions) {
  std::map<std::string, CodeGenInstruction>::const_iterator I;
  I = getInstructions().find("PHI");
  if (I == Instructions.end()) throw "Could not find 'PHI' instruction!";
  const CodeGenInstruction *PHI = &I->second;
  
  I = getInstructions().find("INLINEASM");
  if (I == Instructions.end()) throw "Could not find 'INLINEASM' instruction!";
  const CodeGenInstruction *INLINEASM = &I->second;
  
  I = getInstructions().find("DBG_LABEL");
  if (I == Instructions.end()) throw "Could not find 'DBG_LABEL' instruction!";
  const CodeGenInstruction *DBG_LABEL = &I->second;
  
  I = getInstructions().find("EH_LABEL");
  if (I == Instructions.end()) throw "Could not find 'EH_LABEL' instruction!";
  const CodeGenInstruction *EH_LABEL = &I->second;
  
  I = getInstructions().find("GC_LABEL");
  if (I == Instructions.end()) throw "Could not find 'GC_LABEL' instruction!";
  const CodeGenInstruction *GC_LABEL = &I->second;
  
  I = getInstructions().find("DECLARE");
  if (I == Instructions.end()) throw "Could not find 'DECLARE' instruction!";
  const CodeGenInstruction *DECLARE = &I->second;
  
  I = getInstructions().find("EXTRACT_SUBREG");
  if (I == Instructions.end()) 
    throw "Could not find 'EXTRACT_SUBREG' instruction!";
  const CodeGenInstruction *EXTRACT_SUBREG = &I->second;
  
  I = getInstructions().find("INSERT_SUBREG");
  if (I == Instructions.end()) 
    throw "Could not find 'INSERT_SUBREG' instruction!";
  const CodeGenInstruction *INSERT_SUBREG = &I->second;
  
  I = getInstructions().find("IMPLICIT_DEF");
  if (I == Instructions.end())
    throw "Could not find 'IMPLICIT_DEF' instruction!";
  const CodeGenInstruction *IMPLICIT_DEF = &I->second;
  
  I = getInstructions().find("SUBREG_TO_REG");
  if (I == Instructions.end())
    throw "Could not find 'SUBREG_TO_REG' instruction!";
  const CodeGenInstruction *SUBREG_TO_REG = &I->second;

  I = getInstructions().find("COPY_TO_REGCLASS");
  if (I == Instructions.end())
    throw "Could not find 'COPY_TO_REGCLASS' instruction!";
  const CodeGenInstruction *COPY_TO_REGCLASS = &I->second;

  // Print out the rest of the instructions now.
  NumberedInstructions.push_back(PHI);
  NumberedInstructions.push_back(INLINEASM);
  NumberedInstructions.push_back(DBG_LABEL);
  NumberedInstructions.push_back(EH_LABEL);
  NumberedInstructions.push_back(GC_LABEL);
  NumberedInstructions.push_back(DECLARE);
  NumberedInstructions.push_back(EXTRACT_SUBREG);
  NumberedInstructions.push_back(INSERT_SUBREG);
  NumberedInstructions.push_back(IMPLICIT_DEF);
  NumberedInstructions.push_back(SUBREG_TO_REG);
  NumberedInstructions.push_back(COPY_TO_REGCLASS);
  for (inst_iterator II = inst_begin(), E = inst_end(); II != E; ++II)
    if (&II->second != PHI &&
        &II->second != INLINEASM &&
        &II->second != DBG_LABEL &&
        &II->second != EH_LABEL &&
        &II->second != GC_LABEL &&
        &II->second != DECLARE &&
        &II->second != EXTRACT_SUBREG &&
        &II->second != INSERT_SUBREG &&
        &II->second != IMPLICIT_DEF &&
        &II->second != SUBREG_TO_REG &&
        &II->second != COPY_TO_REGCLASS)
      NumberedInstructions.push_back(&II->second);
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
    } else {
      errs() << "Unsupported SD Node property '" << PropList[i]->getName()
             << "' on ComplexPattern '" << R->getName() << "'!\n";
      exit(1);
    }
  
  // Parse the attributes.  
  Attributes = 0;
  PropList = R->getValueAsListOfDefs("Attributes");
  for (unsigned i = 0, e = PropList.size(); i != e; ++i)
    if (PropList[i]->getName() == "CPAttrParentAsRoot") {
      Attributes |= 1 << CPAttrParentAsRoot;
    } else {
      errs() << "Unsupported pattern attribute '" << PropList[i]->getName()
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
  std::vector<EVT::SimpleValueType> OverloadedVTs;
  ListInit *TypeList = R->getValueAsListInit("RetTypes");
  for (unsigned i = 0, e = TypeList->getSize(); i != e; ++i) {
    Record *TyEl = TypeList->getElementAsRecord(i);
    assert(TyEl->isSubClassOf("LLVMType") && "Expected a type!");
    EVT::SimpleValueType VT;
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
              VT == EVT::iAny) && "Expected iAny type");
    } else {
      VT = getValueType(TyEl->getValueAsDef("VT"));
    }
    if (EVT(VT).isOverloaded()) {
      OverloadedVTs.push_back(VT);
      isOverloaded |= true;
    }
    IS.RetVTs.push_back(VT);
    IS.RetTypeDefs.push_back(TyEl);
  }

  if (IS.RetVTs.size() == 0)
    throw "Intrinsic '"+DefName+"' needs at least a type for the ret value!";

  // Parse the list of parameter types.
  TypeList = R->getValueAsListInit("ParamTypes");
  for (unsigned i = 0, e = TypeList->getSize(); i != e; ++i) {
    Record *TyEl = TypeList->getElementAsRecord(i);
    assert(TyEl->isSubClassOf("LLVMType") && "Expected a type!");
    EVT::SimpleValueType VT;
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
              VT == EVT::iAny) && "Expected iAny type");
    } else
      VT = getValueType(TyEl->getValueAsDef("VT"));
    if (EVT(VT).isOverloaded()) {
      OverloadedVTs.push_back(VT);
      isOverloaded |= true;
    }
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
