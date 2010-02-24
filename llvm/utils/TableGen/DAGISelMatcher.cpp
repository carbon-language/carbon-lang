//===- DAGISelMatcher.cpp - Representation of DAG pattern matcher ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DAGISelMatcher.h"
#include "CodeGenDAGPatterns.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

void MatcherNode::dump() const {
  print(errs());
}

void MatcherNode::printNext(raw_ostream &OS, unsigned indent) const {
  if (Next)
    return Next->print(OS, indent);
}


void PushMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "Push\n";
  printNext(OS, indent+2);
  Failure->print(OS, indent);
}

void RecordMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "Record\n";
  printNext(OS, indent);
}

void RecordChildMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "RecordChild: " << ChildNo << '\n';
  printNext(OS, indent);
}

void RecordMemRefMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "RecordMemRef\n";
  printNext(OS, indent);
}

void CaptureFlagInputMatcherNode::print(raw_ostream &OS, unsigned indent) const{
  OS.indent(indent) << "CaptureFlagInput\n";
  printNext(OS, indent);
}

void MoveChildMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MoveChild " << ChildNo << '\n';
  printNext(OS, indent);
}

void MoveParentMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MoveParent\n";
  printNext(OS, indent);
}

void CheckSameMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckSame " << MatchNumber << '\n';
  printNext(OS, indent);
}

void CheckPatternPredicateMatcherNode::
print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckPatternPredicate " << Predicate << '\n';
  printNext(OS, indent);
}

void CheckPredicateMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckPredicate " << PredName << '\n';
  printNext(OS, indent);
}

void CheckOpcodeMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckOpcode " << OpcodeName << '\n';
  printNext(OS, indent);
}

void CheckMultiOpcodeMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckMultiOpcode <todo args>\n";
  printNext(OS, indent);
}

void CheckTypeMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckType " << getEnumName(Type) << '\n';
  printNext(OS, indent);
}

void CheckIntegerMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckInteger " << Value << '\n';
  printNext(OS, indent);
}

void CheckCondCodeMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckCondCode ISD::" << CondCodeName << '\n';
  printNext(OS, indent);
}

void CheckValueTypeMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckValueType MVT::" << TypeName << '\n';
  printNext(OS, indent);
}

void CheckComplexPatMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckComplexPat " << Pattern.getSelectFunc() << '\n';
  printNext(OS, indent);
}

void CheckAndImmMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckAndImm " << Value << '\n';
  printNext(OS, indent);
}

void CheckOrImmMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckOrImm " << Value << '\n';
  printNext(OS, indent);
}

void CheckFoldableChainNodeMatcherNode::print(raw_ostream &OS,
                                              unsigned indent) const {
  OS.indent(indent) << "CheckFoldableChainNode\n";
  printNext(OS, indent);
}

void CheckChainCompatibleMatcherNode::print(raw_ostream &OS,
                                              unsigned indent) const {
  OS.indent(indent) << "CheckChainCompatible " << PreviousOp << "\n";
  printNext(OS, indent);
}

void EmitIntegerMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitInteger " << Val << " VT=" << VT << '\n';
  printNext(OS, indent);
}

void EmitStringIntegerMatcherNode::
print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitStringInteger " << Val << " VT=" << VT << '\n';
  printNext(OS, indent);
}

void EmitRegisterMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitRegister ";
  if (Reg)
    OS << Reg->getName();
  else
    OS << "zero_reg";
  OS << " VT=" << VT << '\n';
  printNext(OS, indent);
}

void EmitConvertToTargetMatcherNode::
print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitConvertToTarget " << Slot << '\n';
  printNext(OS, indent);
}

void EmitMergeInputChainsMatcherNode::
print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitMergeInputChains <todo: args>\n";
  printNext(OS, indent);
}

void EmitCopyToRegMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitCopyToReg <todo: args>\n";
  printNext(OS, indent);
}

void EmitNodeXFormMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitNodeXForm " << NodeXForm->getName()
     << " Slot=" << Slot << '\n';
  printNext(OS, indent);
}


void EmitNodeMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitNode: " << OpcodeName << ": <todo flags> ";

  for (unsigned i = 0, e = VTs.size(); i != e; ++i)
    OS << ' ' << getEnumName(VTs[i]);
  OS << '(';
  for (unsigned i = 0, e = Operands.size(); i != e; ++i)
    OS << Operands[i] << ' ';
  OS << ")\n";
  printNext(OS, indent);
}

void MarkFlagResultsMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MarkFlagResults <todo: args>\n";
  printNext(OS, indent);
}

void CompleteMatchMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CompleteMatch <todo args>\n";
  OS.indent(indent) << "Src = " << *Pattern.getSrcPattern() << "\n";
  OS.indent(indent) << "Dst = " << *Pattern.getDstPattern() << "\n";
  printNext(OS, indent);
}

