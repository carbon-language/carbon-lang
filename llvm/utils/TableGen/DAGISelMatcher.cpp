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
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

void MatcherNode::dump() const {
  print(errs());
}

void EmitNodeMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitNode: Src = " << *Pattern.getSrcPattern() << "\n";
  OS.indent(indent) << "EmitNode: Dst = " << *Pattern.getDstPattern() << "\n";
}

void MatcherNodeWithChild::printChild(raw_ostream &OS, unsigned indent) const {
  if (Child)
    return Child->print(OS, indent);
  OS.indent(indent) << "<null child>\n";
}


void PushMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "Push\n";
  printChild(OS, indent+2);
  Failure->print(OS, indent);
}

void RecordMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "Record\n";
  printChild(OS, indent);
}

void MoveChildMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MoveChild " << ChildNo << '\n';
  printChild(OS, indent);
}

void MoveParentMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MoveParent\n";
  printChild(OS, indent);
}

void CheckSameMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckSame " << MatchNumber << '\n';
  printChild(OS, indent);
}

void CheckPatternPredicateMatcherNode::
print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckPatternPredicate " << Predicate << '\n';
  printChild(OS, indent);
}

void CheckPredicateMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckPredicate " << PredName << '\n';
  printChild(OS, indent);
}

void CheckOpcodeMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckOpcode " << OpcodeName << '\n';
  printChild(OS, indent);
}

void CheckTypeMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckType " << getEnumName(Type) << '\n';
  printChild(OS, indent);
}

void CheckIntegerMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckInteger " << Value << '\n';
  printChild(OS, indent);
}

void CheckCondCodeMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckCondCode ISD::" << CondCodeName << '\n';
  printChild(OS, indent);
}

void CheckValueTypeMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckValueType MVT::" << TypeName << '\n';
  printChild(OS, indent);
}

void CheckComplexPatMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckComplexPat " << Pattern.getSelectFunc() << '\n';
  printChild(OS, indent);
}

void CheckAndImmMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckAndImm " << Value << '\n';
  printChild(OS, indent);
}

void CheckOrImmMatcherNode::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckOrImm " << Value << '\n';
  printChild(OS, indent);
}

