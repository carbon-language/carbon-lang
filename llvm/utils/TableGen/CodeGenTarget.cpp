//===- CodeGenTarget.cpp - CodeGen Target Class Wrapper ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class wrap target description classes used by the various code
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
#include "llvm/Support/Streams.h"
#include <set>
#include <algorithm>
using namespace llvm;

static cl::opt<unsigned>
AsmWriterNum("asmwriternum", cl::init(0),
             cl::desc("Make -gen-asm-writer emit assembly writer #N"));

/// getValueType - Return the MCV::ValueType that the specified TableGen record
/// corresponds to.
MVT::ValueType llvm::getValueType(Record *Rec) {
  return (MVT::ValueType)Rec->getValueAsInt("Value");
}

std::string llvm::getName(MVT::ValueType T) {
  switch (T) {
  case MVT::Other: return "UNKNOWN";
  case MVT::i1:    return "MVT::i1";
  case MVT::i8:    return "MVT::i8";
  case MVT::i16:   return "MVT::i16";
  case MVT::i32:   return "MVT::i32";
  case MVT::i64:   return "MVT::i64";
  case MVT::i128:  return "MVT::i128";
  case MVT::iAny:  return "MVT::iAny";
  case MVT::f32:   return "MVT::f32";
  case MVT::f64:   return "MVT::f64";
  case MVT::f80:   return "MVT::f80";
  case MVT::f128:  return "MVT::f128";
  case MVT::Flag:  return "MVT::Flag";
  case MVT::isVoid:return "MVT::void";
  case MVT::v8i8:  return "MVT::v8i8";
  case MVT::v4i16: return "MVT::v4i16";
  case MVT::v2i32: return "MVT::v2i32";
  case MVT::v1i64: return "MVT::v1i64";
  case MVT::v16i8: return "MVT::v16i8";
  case MVT::v8i16: return "MVT::v8i16";
  case MVT::v4i32: return "MVT::v4i32";
  case MVT::v2i64: return "MVT::v2i64";
  case MVT::v2f32: return "MVT::v2f32";
  case MVT::v4f32: return "MVT::v4f32";
  case MVT::v2f64: return "MVT::v2f64";
  case MVT::v3i32: return "MVT::v3i32";
  case MVT::v3f32: return "MVT::v3f32";
  case MVT::iPTR:  return "TLI.getPointerTy()";
  default: assert(0 && "ILLEGAL VALUE TYPE!"); return "";
  }
}

std::string llvm::getEnumName(MVT::ValueType T) {
  switch (T) {
  case MVT::Other: return "MVT::Other";
  case MVT::i1:    return "MVT::i1";
  case MVT::i8:    return "MVT::i8";
  case MVT::i16:   return "MVT::i16";
  case MVT::i32:   return "MVT::i32";
  case MVT::i64:   return "MVT::i64";
  case MVT::i128:  return "MVT::i128";
  case MVT::iAny:  return "MVT::iAny";
  case MVT::f32:   return "MVT::f32";
  case MVT::f64:   return "MVT::f64";
  case MVT::f80:   return "MVT::f80";
  case MVT::f128:  return "MVT::f128";
  case MVT::Flag:  return "MVT::Flag";
  case MVT::isVoid:return "MVT::isVoid";
  case MVT::v8i8:  return "MVT::v8i8";
  case MVT::v4i16: return "MVT::v4i16";
  case MVT::v2i32: return "MVT::v2i32";
  case MVT::v1i64: return "MVT::v1i64";
  case MVT::v16i8: return "MVT::v16i8";
  case MVT::v8i16: return "MVT::v8i16";
  case MVT::v4i32: return "MVT::v4i32";
  case MVT::v2i64: return "MVT::v2i64";
  case MVT::v2f32: return "MVT::v2f32";
  case MVT::v4f32: return "MVT::v4f32";
  case MVT::v2f64: return "MVT::v2f64";
  case MVT::v3i32: return "MVT::v3i32";
  case MVT::v3f32: return "MVT::v3f32";
  case MVT::iPTR:  return "TLI.getPointerTy()";
  default: assert(0 && "ILLEGAL VALUE TYPE!"); return "";
  }
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

Record *CodeGenTarget::getInstructionSet() const {
  return TargetRec->getValueAsDef("InstructionSet");
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
        const std::vector<MVT::ValueType> &InVTs = RC.getValueTypes();
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
  SpillSize = Size ? Size : MVT::getSizeInBits(VTs[0]);
  SpillAlignment = R->getValueAsInt("Alignment");
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
  
  I = getInstructions().find("LABEL");
  if (I == Instructions.end()) throw "Could not find 'LABEL' instruction!";
  const CodeGenInstruction *LABEL = &I->second;
  
  // Print out the rest of the instructions now.
  NumberedInstructions.push_back(PHI);
  NumberedInstructions.push_back(INLINEASM);
  NumberedInstructions.push_back(LABEL);
  for (inst_iterator II = inst_begin(), E = inst_end(); II != E; ++II)
    if (&II->second != PHI &&
        &II->second != INLINEASM &&
        &II->second != LABEL)
      NumberedInstructions.push_back(&II->second);
}


/// isLittleEndianEncoding - Return whether this target encodes its instruction
/// in little-endian format, i.e. bits laid out in the order [0..n]
///
bool CodeGenTarget::isLittleEndianEncoding() const {
  return getInstructionSet()->getValueAsBit("isLittleEndianEncoding");
}



static void ParseConstraint(const std::string &CStr, CodeGenInstruction *I) {
  // FIXME: Only supports TIED_TO for now.
  std::string::size_type pos = CStr.find_first_of('=');
  assert(pos != std::string::npos && "Unrecognized constraint");
  std::string Name = CStr.substr(0, pos);

  // TIED_TO: $src1 = $dst
  std::string::size_type wpos = Name.find_first_of(" \t");
  if (wpos == std::string::npos)
    throw "Illegal format for tied-to constraint: '" + CStr + "'";
  std::string DestOpName = Name.substr(0, wpos);
  std::pair<unsigned,unsigned> DestOp = I->ParseOperandName(DestOpName, false);

  Name = CStr.substr(pos+1);
  wpos = Name.find_first_not_of(" \t");
  if (wpos == std::string::npos)
    throw "Illegal format for tied-to constraint: '" + CStr + "'";
    
  std::pair<unsigned,unsigned> SrcOp =
    I->ParseOperandName(Name.substr(wpos), false);
  if (SrcOp > DestOp)
    throw "Illegal tied-to operand constraint '" + CStr + "'";
  
  
  unsigned FlatOpNo = I->getFlattenedOperandNumber(SrcOp);
  // Build the string for the operand.
  std::string OpConstraint =
    "((" + utostr(FlatOpNo) + " << 16) | (1 << TOI::TIED_TO))";

  
  if (!I->OperandList[DestOp.first].Constraints[DestOp.second].empty())
    throw "Operand '" + DestOpName + "' cannot have multiple constraints!";
  I->OperandList[DestOp.first].Constraints[DestOp.second] = OpConstraint;
}

static void ParseConstraints(const std::string &CStr, CodeGenInstruction *I) {
  // Make sure the constraints list for each operand is large enough to hold
  // constraint info, even if none is present.
  for (unsigned i = 0, e = I->OperandList.size(); i != e; ++i) 
    I->OperandList[i].Constraints.resize(I->OperandList[i].MINumOperands);
  
  if (CStr.empty()) return;
  
  const std::string delims(",");
  std::string::size_type bidx, eidx;

  bidx = CStr.find_first_not_of(delims);
  while (bidx != std::string::npos) {
    eidx = CStr.find_first_of(delims, bidx);
    if (eidx == std::string::npos)
      eidx = CStr.length();
    
    ParseConstraint(CStr.substr(bidx, eidx), I);
    bidx = CStr.find_first_not_of(delims, eidx);
  }
}

CodeGenInstruction::CodeGenInstruction(Record *R, const std::string &AsmStr)
  : TheDef(R), AsmString(AsmStr) {
  Name      = R->getValueAsString("Name");
  Namespace = R->getValueAsString("Namespace");

  isReturn     = R->getValueAsBit("isReturn");
  isBranch     = R->getValueAsBit("isBranch");
  isBarrier    = R->getValueAsBit("isBarrier");
  isCall       = R->getValueAsBit("isCall");
  isLoad       = R->getValueAsBit("isLoad");
  isStore      = R->getValueAsBit("isStore");
  bool isTwoAddress = R->getValueAsBit("isTwoAddress");
  isPredicable = R->getValueAsBit("isPredicable");
  isConvertibleToThreeAddress = R->getValueAsBit("isConvertibleToThreeAddress");
  isCommutable = R->getValueAsBit("isCommutable");
  isTerminator = R->getValueAsBit("isTerminator");
  isReMaterializable = R->getValueAsBit("isReMaterializable");
  hasDelaySlot = R->getValueAsBit("hasDelaySlot");
  usesCustomDAGSchedInserter = R->getValueAsBit("usesCustomDAGSchedInserter");
  hasCtrlDep   = R->getValueAsBit("hasCtrlDep");
  isNotDuplicable = R->getValueAsBit("isNotDuplicable");
  hasOptionalDef = false;
  hasVariableNumberOfOperands = false;
  
  DagInit *DI;
  try {
    DI = R->getValueAsDag("OutOperandList");
  } catch (...) {
    // Error getting operand list, just ignore it (sparcv9).
    AsmString.clear();
    OperandList.clear();
    return;
  }
  NumDefs = DI->getNumArgs();

  DagInit *IDI;
  try {
    IDI = R->getValueAsDag("InOperandList");
  } catch (...) {
    // Error getting operand list, just ignore it (sparcv9).
    AsmString.clear();
    OperandList.clear();
    return;
  }
  DI = (DagInit*)(new BinOpInit(BinOpInit::CONCAT, DI, IDI))->Fold();

  unsigned MIOperandNo = 0;
  std::set<std::string> OperandNames;
  for (unsigned i = 0, e = DI->getNumArgs(); i != e; ++i) {
    DefInit *Arg = dynamic_cast<DefInit*>(DI->getArg(i));
    if (!Arg)
      throw "Illegal operand for the '" + R->getName() + "' instruction!";

    Record *Rec = Arg->getDef();
    std::string PrintMethod = "printOperand";
    unsigned NumOps = 1;
    DagInit *MIOpInfo = 0;
    if (Rec->isSubClassOf("Operand")) {
      PrintMethod = Rec->getValueAsString("PrintMethod");
      MIOpInfo = Rec->getValueAsDag("MIOperandInfo");
      
      // Verify that MIOpInfo has an 'ops' root value.
      if (!dynamic_cast<DefInit*>(MIOpInfo->getOperator()) ||
          dynamic_cast<DefInit*>(MIOpInfo->getOperator())
               ->getDef()->getName() != "ops")
        throw "Bad value for MIOperandInfo in operand '" + Rec->getName() +
              "'\n";

      // If we have MIOpInfo, then we have #operands equal to number of entries
      // in MIOperandInfo.
      if (unsigned NumArgs = MIOpInfo->getNumArgs())
        NumOps = NumArgs;

      if (Rec->isSubClassOf("PredicateOperand"))
        isPredicable = true;
      else if (Rec->isSubClassOf("OptionalDefOperand"))
        hasOptionalDef = true;
    } else if (Rec->getName() == "variable_ops") {
      hasVariableNumberOfOperands = true;
      continue;
    } else if (!Rec->isSubClassOf("RegisterClass") && 
               Rec->getName() != "ptr_rc")
      throw "Unknown operand class '" + Rec->getName() +
            "' in instruction '" + R->getName() + "' instruction!";

    // Check that the operand has a name and that it's unique.
    if (DI->getArgName(i).empty())
      throw "In instruction '" + R->getName() + "', operand #" + utostr(i) +
        " has no name!";
    if (!OperandNames.insert(DI->getArgName(i)).second)
      throw "In instruction '" + R->getName() + "', operand #" + utostr(i) +
        " has the same name as a previous operand!";
    
    OperandList.push_back(OperandInfo(Rec, DI->getArgName(i), PrintMethod, 
                                      MIOperandNo, NumOps, MIOpInfo));
    MIOperandNo += NumOps;
  }

  // Parse Constraints.
  ParseConstraints(R->getValueAsString("Constraints"), this);
  
  // For backward compatibility: isTwoAddress means operand 1 is tied to
  // operand 0.
  if (isTwoAddress) {
    if (!OperandList[1].Constraints[0].empty())
      throw R->getName() + ": cannot use isTwoAddress property: instruction "
            "already has constraint set!";
    OperandList[1].Constraints[0] = "((0 << 16) | (1 << TOI::TIED_TO))";
  }
  
  // Any operands with unset constraints get 0 as their constraint.
  for (unsigned op = 0, e = OperandList.size(); op != e; ++op)
    for (unsigned j = 0, e = OperandList[op].MINumOperands; j != e; ++j)
      if (OperandList[op].Constraints[j].empty())
        OperandList[op].Constraints[j] = "0";
  
  // Parse the DisableEncoding field.
  std::string DisableEncoding = R->getValueAsString("DisableEncoding");
  while (1) {
    std::string OpName = getToken(DisableEncoding, " ,\t");
    if (OpName.empty()) break;

    // Figure out which operand this is.
    std::pair<unsigned,unsigned> Op = ParseOperandName(OpName, false);

    // Mark the operand as not-to-be encoded.
    if (Op.second >= OperandList[Op.first].DoNotEncode.size())
      OperandList[Op.first].DoNotEncode.resize(Op.second+1);
    OperandList[Op.first].DoNotEncode[Op.second] = true;
  }
}



/// getOperandNamed - Return the index of the operand with the specified
/// non-empty name.  If the instruction does not have an operand with the
/// specified name, throw an exception.
///
unsigned CodeGenInstruction::getOperandNamed(const std::string &Name) const {
  assert(!Name.empty() && "Cannot search for operand with no name!");
  for (unsigned i = 0, e = OperandList.size(); i != e; ++i)
    if (OperandList[i].Name == Name) return i;
  throw "Instruction '" + TheDef->getName() +
        "' does not have an operand named '$" + Name + "'!";
}

std::pair<unsigned,unsigned> 
CodeGenInstruction::ParseOperandName(const std::string &Op,
                                     bool AllowWholeOp) {
  if (Op.empty() || Op[0] != '$')
    throw TheDef->getName() + ": Illegal operand name: '" + Op + "'";
  
  std::string OpName = Op.substr(1);
  std::string SubOpName;
  
  // Check to see if this is $foo.bar.
  std::string::size_type DotIdx = OpName.find_first_of(".");
  if (DotIdx != std::string::npos) {
    SubOpName = OpName.substr(DotIdx+1);
    if (SubOpName.empty())
      throw TheDef->getName() + ": illegal empty suboperand name in '" +Op +"'";
    OpName = OpName.substr(0, DotIdx);
  }
  
  unsigned OpIdx = getOperandNamed(OpName);

  if (SubOpName.empty()) {  // If no suboperand name was specified:
    // If one was needed, throw.
    if (OperandList[OpIdx].MINumOperands > 1 && !AllowWholeOp &&
        SubOpName.empty())
      throw TheDef->getName() + ": Illegal to refer to"
            " whole operand part of complex operand '" + Op + "'";
  
    // Otherwise, return the operand.
    return std::make_pair(OpIdx, 0U);
  }
  
  // Find the suboperand number involved.
  DagInit *MIOpInfo = OperandList[OpIdx].MIOperandInfo;
  if (MIOpInfo == 0)
    throw TheDef->getName() + ": unknown suboperand name in '" + Op + "'";
  
  // Find the operand with the right name.
  for (unsigned i = 0, e = MIOpInfo->getNumArgs(); i != e; ++i)
    if (MIOpInfo->getArgName(i) == SubOpName)
      return std::make_pair(OpIdx, i);

  // Otherwise, didn't find it!
  throw TheDef->getName() + ": unknown suboperand name in '" + Op + "'";
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
    } else {
      cerr << "Unsupported SD Node property '" << PropList[i]->getName()
           << "' on ComplexPattern '" << R->getName() << "'!\n";
      exit(1);
    }
}

//===----------------------------------------------------------------------===//
// CodeGenIntrinsic Implementation
//===----------------------------------------------------------------------===//

std::vector<CodeGenIntrinsic> llvm::LoadIntrinsics(const RecordKeeper &RC) {
  std::vector<Record*> I = RC.getAllDerivedDefinitions("Intrinsic");
  
  std::vector<CodeGenIntrinsic> Result;

  // If we are in the context of a target .td file, get the target info so that
  // we can decode the current intptr_t.
  CodeGenTarget *CGT = 0;
  if (Records.getClass("Target") &&
      Records.getAllDerivedDefinitions("Target").size() == 1)
    CGT = new CodeGenTarget();
  
  for (unsigned i = 0, e = I.size(); i != e; ++i)
    Result.push_back(CodeGenIntrinsic(I[i], CGT));
  delete CGT;
  return Result;
}

CodeGenIntrinsic::CodeGenIntrinsic(Record *R, CodeGenTarget *CGT) {
  TheDef = R;
  std::string DefName = R->getName();
  ModRef = WriteMem;
  isOverloaded = false;
  
  if (DefName.size() <= 4 || 
      std::string(DefName.begin(), DefName.begin()+4) != "int_")
    throw "Intrinsic '" + DefName + "' does not start with 'int_'!";
  EnumName = std::string(DefName.begin()+4, DefName.end());
  if (R->getValue("GCCBuiltinName"))  // Ignore a missing GCCBuiltinName field.
    GCCBuiltinName = R->getValueAsString("GCCBuiltinName");
  TargetPrefix   = R->getValueAsString("TargetPrefix");
  Name = R->getValueAsString("LLVMName");
  if (Name == "") {
    // If an explicit name isn't specified, derive one from the DefName.
    Name = "llvm.";
    for (unsigned i = 0, e = EnumName.size(); i != e; ++i)
      if (EnumName[i] == '_')
        Name += '.';
      else
        Name += EnumName[i];
  } else {
    // Verify it starts with "llvm.".
    if (Name.size() <= 5 || 
        std::string(Name.begin(), Name.begin()+5) != "llvm.")
      throw "Intrinsic '" + DefName + "'s name does not start with 'llvm.'!";
  }
  
  // If TargetPrefix is specified, make sure that Name starts with
  // "llvm.<targetprefix>.".
  if (!TargetPrefix.empty()) {
    if (Name.size() < 6+TargetPrefix.size() ||
        std::string(Name.begin()+5, Name.begin()+6+TargetPrefix.size()) 
        != (TargetPrefix+"."))
      throw "Intrinsic '" + DefName + "' does not start with 'llvm." + 
        TargetPrefix + ".'!";
  }
  
  // Parse the list of argument types.
  ListInit *TypeList = R->getValueAsListInit("Types");
  for (unsigned i = 0, e = TypeList->getSize(); i != e; ++i) {
    Record *TyEl = TypeList->getElementAsRecord(i);
    assert(TyEl->isSubClassOf("LLVMType") && "Expected a type!");
    ArgTypes.push_back(TyEl->getValueAsString("TypeVal"));
    MVT::ValueType VT = getValueType(TyEl->getValueAsDef("VT"));
    isOverloaded |= VT == MVT::iAny;
    ArgVTs.push_back(VT);
    ArgTypeDefs.push_back(TyEl);
  }
  if (ArgTypes.size() == 0)
    throw "Intrinsic '"+DefName+"' needs at least a type for the ret value!";

  
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
    else
      assert(0 && "Unknown property!");
  }
}
