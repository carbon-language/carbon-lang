//===- CodeGenInstruction.cpp - CodeGen Instruction Class Wrapper ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CodeGenInstruction class.
//
//===----------------------------------------------------------------------===//

#include "CodeGenInstruction.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include <set>
using namespace llvm;

static void ParseConstraint(const std::string &CStr, CodeGenInstruction *I) {
  // EARLY_CLOBBER: @early $reg
  std::string::size_type wpos = CStr.find_first_of(" \t");
  std::string::size_type start = CStr.find_first_not_of(" \t");
  std::string Tok = CStr.substr(start, wpos - start);
  if (Tok == "@earlyclobber") {
    std::string Name = CStr.substr(wpos+1);
    wpos = Name.find_first_not_of(" \t");
    if (wpos == std::string::npos)
      throw "Illegal format for @earlyclobber constraint: '" + CStr + "'";
    Name = Name.substr(wpos);
    std::pair<unsigned,unsigned> Op =
      I->ParseOperandName(Name, false);

    // Build the string for the operand
    if (!I->OperandList[Op.first].Constraints[Op.second].isNone())
      throw "Operand '" + Name + "' cannot have multiple constraints!";
    I->OperandList[Op.first].Constraints[Op.second] =
      CodeGenInstruction::ConstraintInfo::getEarlyClobber();
    return;
  }

  // Only other constraint is "TIED_TO" for now.
  std::string::size_type pos = CStr.find_first_of('=');
  assert(pos != std::string::npos && "Unrecognized constraint");
  start = CStr.find_first_not_of(" \t");
  std::string Name = CStr.substr(start, pos - start);

  // TIED_TO: $src1 = $dst
  wpos = Name.find_first_of(" \t");
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

  if (!I->OperandList[DestOp.first].Constraints[DestOp.second].isNone())
    throw "Operand '" + DestOpName + "' cannot have multiple constraints!";
  I->OperandList[DestOp.first].Constraints[DestOp.second] =
    CodeGenInstruction::ConstraintInfo::getTied(FlatOpNo);
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

    ParseConstraint(CStr.substr(bidx, eidx - bidx), I);
    bidx = CStr.find_first_not_of(delims, eidx);
  }
}

CodeGenInstruction::CodeGenInstruction(Record *R, const std::string &AsmStr)
  : TheDef(R), AsmString(AsmStr) {
  Namespace = R->getValueAsString("Namespace");

  isReturn     = R->getValueAsBit("isReturn");
  isBranch     = R->getValueAsBit("isBranch");
  isIndirectBranch = R->getValueAsBit("isIndirectBranch");
  isBarrier    = R->getValueAsBit("isBarrier");
  isCall       = R->getValueAsBit("isCall");
  canFoldAsLoad = R->getValueAsBit("canFoldAsLoad");
  mayLoad      = R->getValueAsBit("mayLoad");
  mayStore     = R->getValueAsBit("mayStore");
  bool isTwoAddress = R->getValueAsBit("isTwoAddress");
  isPredicable = R->getValueAsBit("isPredicable");
  isConvertibleToThreeAddress = R->getValueAsBit("isConvertibleToThreeAddress");
  isCommutable = R->getValueAsBit("isCommutable");
  isTerminator = R->getValueAsBit("isTerminator");
  isReMaterializable = R->getValueAsBit("isReMaterializable");
  hasDelaySlot = R->getValueAsBit("hasDelaySlot");
  usesCustomInserter = R->getValueAsBit("usesCustomInserter");
  hasCtrlDep   = R->getValueAsBit("hasCtrlDep");
  isNotDuplicable = R->getValueAsBit("isNotDuplicable");
  hasSideEffects = R->getValueAsBit("hasSideEffects");
  neverHasSideEffects = R->getValueAsBit("neverHasSideEffects");
  isAsCheapAsAMove = R->getValueAsBit("isAsCheapAsAMove");
  hasExtraSrcRegAllocReq = R->getValueAsBit("hasExtraSrcRegAllocReq");
  hasExtraDefRegAllocReq = R->getValueAsBit("hasExtraDefRegAllocReq");
  hasOptionalDef = false;
  isVariadic = false;
  ImplicitDefs = R->getValueAsListOfDefs("Defs");
  ImplicitUses = R->getValueAsListOfDefs("Uses");

  if (neverHasSideEffects + hasSideEffects > 1)
    throw R->getName() + ": multiple conflicting side-effect flags set!";

  DagInit *OutDI = R->getValueAsDag("OutOperandList");

  if (DefInit *Init = dynamic_cast<DefInit*>(OutDI->getOperator())) {
    if (Init->getDef()->getName() != "outs")
      throw R->getName() + ": invalid def name for output list: use 'outs'";
  } else
    throw R->getName() + ": invalid output list: use 'outs'";
    
  NumDefs = OutDI->getNumArgs();
    
  DagInit *InDI = R->getValueAsDag("InOperandList");
  if (DefInit *Init = dynamic_cast<DefInit*>(InDI->getOperator())) {
    if (Init->getDef()->getName() != "ins")
      throw R->getName() + ": invalid def name for input list: use 'ins'";
  } else
    throw R->getName() + ": invalid input list: use 'ins'";
    
  unsigned MIOperandNo = 0;
  std::set<std::string> OperandNames;
  for (unsigned i = 0, e = InDI->getNumArgs()+OutDI->getNumArgs(); i != e; ++i){
    Init *ArgInit;
    std::string ArgName;
    if (i < NumDefs) {
      ArgInit = OutDI->getArg(i);
      ArgName = OutDI->getArgName(i);
    } else {
      ArgInit = InDI->getArg(i-NumDefs);
      ArgName = InDI->getArgName(i-NumDefs);
    }
    
    DefInit *Arg = dynamic_cast<DefInit*>(ArgInit);
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
      isVariadic = true;
      continue;
    } else if (!Rec->isSubClassOf("RegisterClass") &&
               Rec->getName() != "ptr_rc" && Rec->getName() != "unknown")
      throw "Unknown operand class '" + Rec->getName() +
            "' in '" + R->getName() + "' instruction!";

    // Check that the operand has a name and that it's unique.
    if (ArgName.empty())
      throw "In instruction '" + R->getName() + "', operand #" + utostr(i) +
        " has no name!";
    if (!OperandNames.insert(ArgName).second)
      throw "In instruction '" + R->getName() + "', operand #" + utostr(i) +
        " has the same name as a previous operand!";

    OperandList.push_back(OperandInfo(Rec, ArgName, PrintMethod,
                                      MIOperandNo, NumOps, MIOpInfo));
    MIOperandNo += NumOps;
  }

  // Parse Constraints.
  ParseConstraints(R->getValueAsString("Constraints"), this);

  // For backward compatibility: isTwoAddress means operand 1 is tied to
  // operand 0.
  if (isTwoAddress) {
    if (!OperandList[1].Constraints[0].isNone())
      throw R->getName() + ": cannot use isTwoAddress property: instruction "
            "already has constraint set!";
    OperandList[1].Constraints[0] =
      CodeGenInstruction::ConstraintInfo::getTied(0);
  }

  // Parse the DisableEncoding field.
  std::string DisableEncoding = R->getValueAsString("DisableEncoding");
  while (1) {
    std::string OpName;
    tie(OpName, DisableEncoding) = getToken(DisableEncoding, " ,\t");
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


/// HasOneImplicitDefWithKnownVT - If the instruction has at least one
/// implicit def and it has a known VT, return the VT, otherwise return
/// MVT::Other.
MVT::SimpleValueType CodeGenInstruction::
HasOneImplicitDefWithKnownVT(const CodeGenTarget &TargetInfo) const {
  if (ImplicitDefs.empty()) return MVT::Other;
  
  // Check to see if the first implicit def has a resolvable type.
  Record *FirstImplicitDef = ImplicitDefs[0];
  assert(FirstImplicitDef->isSubClassOf("Register"));
  const std::vector<MVT::SimpleValueType> &RegVTs = 
    TargetInfo.getRegisterVTs(FirstImplicitDef);
  if (RegVTs.size() == 1)
    return RegVTs[0];
  return MVT::Other;
}

