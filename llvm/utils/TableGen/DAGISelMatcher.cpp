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

void Matcher::dump() const {
  print(errs());
}

void Matcher::printNext(raw_ostream &OS, unsigned indent) const {
  if (Next)
    return Next->print(OS, indent);
}


void ScopeMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "Scope\n";
  Check->print(OS, indent+2);
  printNext(OS, indent);
}

void RecordMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "Record\n";
  printNext(OS, indent);
}

void RecordChildMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "RecordChild: " << ChildNo << '\n';
  printNext(OS, indent);
}

void RecordMemRefMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "RecordMemRef\n";
  printNext(OS, indent);
}

void CaptureFlagInputMatcher::print(raw_ostream &OS, unsigned indent) const{
  OS.indent(indent) << "CaptureFlagInput\n";
  printNext(OS, indent);
}

void MoveChildMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MoveChild " << ChildNo << '\n';
  printNext(OS, indent);
}

void MoveParentMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MoveParent\n";
  printNext(OS, indent);
}

void CheckSameMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckSame " << MatchNumber << '\n';
  printNext(OS, indent);
}

void CheckPatternPredicateMatcher::
print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckPatternPredicate " << Predicate << '\n';
  printNext(OS, indent);
}

void CheckPredicateMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckPredicate " << PredName << '\n';
  printNext(OS, indent);
}

void CheckOpcodeMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckOpcode " << OpcodeName << '\n';
  printNext(OS, indent);
}

void CheckMultiOpcodeMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckMultiOpcode <todo args>\n";
  printNext(OS, indent);
}

void CheckTypeMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckType " << getEnumName(Type) << '\n';
  printNext(OS, indent);
}

void CheckChildTypeMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckChildType " << ChildNo << " "
    << getEnumName(Type) << '\n';
  printNext(OS, indent);
}


void CheckIntegerMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckInteger " << Value << '\n';
  printNext(OS, indent);
}

void CheckCondCodeMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckCondCode ISD::" << CondCodeName << '\n';
  printNext(OS, indent);
}

void CheckValueTypeMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckValueType MVT::" << TypeName << '\n';
  printNext(OS, indent);
}

void CheckComplexPatMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckComplexPat " << Pattern.getSelectFunc() << '\n';
  printNext(OS, indent);
}

void CheckAndImmMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckAndImm " << Value << '\n';
  printNext(OS, indent);
}

void CheckOrImmMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckOrImm " << Value << '\n';
  printNext(OS, indent);
}

void CheckFoldableChainNodeMatcher::print(raw_ostream &OS,
                                              unsigned indent) const {
  OS.indent(indent) << "CheckFoldableChainNode\n";
  printNext(OS, indent);
}

void CheckChainCompatibleMatcher::print(raw_ostream &OS,
                                              unsigned indent) const {
  OS.indent(indent) << "CheckChainCompatible " << PreviousOp << "\n";
  printNext(OS, indent);
}

void EmitIntegerMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitInteger " << Val << " VT=" << VT << '\n';
  printNext(OS, indent);
}

void EmitStringIntegerMatcher::
print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitStringInteger " << Val << " VT=" << VT << '\n';
  printNext(OS, indent);
}

void EmitRegisterMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitRegister ";
  if (Reg)
    OS << Reg->getName();
  else
    OS << "zero_reg";
  OS << " VT=" << VT << '\n';
  printNext(OS, indent);
}

void EmitConvertToTargetMatcher::
print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitConvertToTarget " << Slot << '\n';
  printNext(OS, indent);
}

void EmitMergeInputChainsMatcher::
print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitMergeInputChains <todo: args>\n";
  printNext(OS, indent);
}

void EmitCopyToRegMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitCopyToReg <todo: args>\n";
  printNext(OS, indent);
}

void EmitNodeXFormMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitNodeXForm " << NodeXForm->getName()
     << " Slot=" << Slot << '\n';
  printNext(OS, indent);
}


void EmitNodeMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitNode: " << OpcodeName << ": <todo flags> ";

  for (unsigned i = 0, e = VTs.size(); i != e; ++i)
    OS << ' ' << getEnumName(VTs[i]);
  OS << '(';
  for (unsigned i = 0, e = Operands.size(); i != e; ++i)
    OS << Operands[i] << ' ';
  OS << ")\n";
  printNext(OS, indent);
}

void MarkFlagResultsMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MarkFlagResults <todo: args>\n";
  printNext(OS, indent);
}

void CompleteMatchMatcher::print(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CompleteMatch <todo args>\n";
  OS.indent(indent) << "Src = " << *Pattern.getSrcPattern() << "\n";
  OS.indent(indent) << "Dst = " << *Pattern.getDstPattern() << "\n";
  printNext(OS, indent);
}

