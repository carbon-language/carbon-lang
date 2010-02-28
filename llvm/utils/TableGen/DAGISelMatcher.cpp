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
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

void Matcher::dump() const {
  print(errs(), 0);
}

void Matcher::print(raw_ostream &OS, unsigned indent) const {
  printImpl(OS, indent);
  if (Next)
    return Next->print(OS, indent);
}

void Matcher::printOne(raw_ostream &OS) const {
  printImpl(OS, 0);
}

ScopeMatcher::~ScopeMatcher() {
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    delete Children[i];
}


// printImpl methods.

void ScopeMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "Scope\n";
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i) {
    if (getChild(i) == 0)
      OS.indent(indent+1) << "NULL POINTER\n";
    else
      getChild(i)->print(OS, indent+2);
  }
}

void RecordMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "Record\n";
}

void RecordChildMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "RecordChild: " << ChildNo << '\n';
}

void RecordMemRefMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "RecordMemRef\n";
}

void CaptureFlagInputMatcher::printImpl(raw_ostream &OS, unsigned indent) const{
  OS.indent(indent) << "CaptureFlagInput\n";
}

void MoveChildMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MoveChild " << ChildNo << '\n';
}

void MoveParentMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MoveParent\n";
}

void CheckSameMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckSame " << MatchNumber << '\n';
}

void CheckPatternPredicateMatcher::
printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckPatternPredicate " << Predicate << '\n';
}

void CheckPredicateMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckPredicate " << PredName << '\n';
}

void CheckOpcodeMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckOpcode " << Opcode.getEnumName() << '\n';
}

void CheckMultiOpcodeMatcher::printImpl(raw_ostream &OS, unsigned indent) const{
  OS.indent(indent) << "CheckMultiOpcode <todo args>\n";
}

void CheckTypeMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckType " << getEnumName(Type) << '\n';
}

void CheckChildTypeMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckChildType " << ChildNo << " "
    << getEnumName(Type) << '\n';
}


void CheckIntegerMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckInteger " << Value << '\n';
}

void CheckCondCodeMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckCondCode ISD::" << CondCodeName << '\n';
}

void CheckValueTypeMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckValueType MVT::" << TypeName << '\n';
}

void CheckComplexPatMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckComplexPat " << Pattern.getSelectFunc() << '\n';
}

void CheckAndImmMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckAndImm " << Value << '\n';
}

void CheckOrImmMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckOrImm " << Value << '\n';
}

void CheckFoldableChainNodeMatcher::printImpl(raw_ostream &OS,
                                              unsigned indent) const {
  OS.indent(indent) << "CheckFoldableChainNode\n";
}

void CheckChainCompatibleMatcher::printImpl(raw_ostream &OS,
                                              unsigned indent) const {
  OS.indent(indent) << "CheckChainCompatible " << PreviousOp << "\n";
}

void EmitIntegerMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitInteger " << Val << " VT=" << VT << '\n';
}

void EmitStringIntegerMatcher::
printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitStringInteger " << Val << " VT=" << VT << '\n';
}

void EmitRegisterMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitRegister ";
  if (Reg)
    OS << Reg->getName();
  else
    OS << "zero_reg";
  OS << " VT=" << VT << '\n';
}

void EmitConvertToTargetMatcher::
printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitConvertToTarget " << Slot << '\n';
}

void EmitMergeInputChainsMatcher::
printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitMergeInputChains <todo: args>\n";
}

void EmitCopyToRegMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitCopyToReg <todo: args>\n";
}

void EmitNodeXFormMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitNodeXForm " << NodeXForm->getName()
     << " Slot=" << Slot << '\n';
}


void EmitNodeMatcherCommon::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent);
  OS << (isa<MorphNodeToMatcher>(this) ? "MorphNodeTo: " : "EmitNode: ")
     << OpcodeName << ": <todo flags> ";

  for (unsigned i = 0, e = VTs.size(); i != e; ++i)
    OS << ' ' << getEnumName(VTs[i]);
  OS << '(';
  for (unsigned i = 0, e = Operands.size(); i != e; ++i)
    OS << Operands[i] << ' ';
  OS << ")\n";
}

void MarkFlagResultsMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MarkFlagResults <todo: args>\n";
}

void CompleteMatchMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CompleteMatch <todo args>\n";
  OS.indent(indent) << "Src = " << *Pattern.getSrcPattern() << "\n";
  OS.indent(indent) << "Dst = " << *Pattern.getDstPattern() << "\n";
}

// getHashImpl Implementation.

unsigned CheckPatternPredicateMatcher::getHashImpl() const {
  return HashString(Predicate);
}

unsigned CheckPredicateMatcher::getHashImpl() const {
  return HashString(PredName);
}

unsigned CheckOpcodeMatcher::getHashImpl() const {
  return HashString(Opcode.getEnumName());
}

unsigned CheckMultiOpcodeMatcher::getHashImpl() const {
  unsigned Result = 0;
  for (unsigned i = 0, e = Opcodes.size(); i != e; ++i)
    Result |= HashString(Opcodes[i]->getEnumName());
  return Result;
}

unsigned CheckCondCodeMatcher::getHashImpl() const {
  return HashString(CondCodeName);
}

unsigned CheckValueTypeMatcher::getHashImpl() const {
  return HashString(TypeName);
}

unsigned EmitStringIntegerMatcher::getHashImpl() const {
  return HashString(Val) ^ VT;
}

template<typename It>
static unsigned HashUnsigneds(It I, It E) {
  unsigned Result = 0;
  for (; I != E; ++I)
    Result = (Result<<3) ^ *I;
  return Result;
}

unsigned EmitMergeInputChainsMatcher::getHashImpl() const {
  return HashUnsigneds(ChainNodes.begin(), ChainNodes.end());
}

bool EmitNodeMatcherCommon::isEqualImpl(const Matcher *m) const {
  const EmitNodeMatcherCommon *M = cast<EmitNodeMatcherCommon>(m);
  return M->OpcodeName == OpcodeName && M->VTs == VTs &&
         M->Operands == Operands && M->HasChain == HasChain &&
         M->HasInFlag == HasInFlag && M->HasOutFlag == HasOutFlag &&
         M->HasMemRefs == HasMemRefs &&
         M->NumFixedArityOperands == NumFixedArityOperands;
}

unsigned EmitNodeMatcherCommon::getHashImpl() const {
  return (HashString(OpcodeName) << 4) | Operands.size();
}


unsigned MarkFlagResultsMatcher::getHashImpl() const {
  return HashUnsigneds(FlagResultNodes.begin(), FlagResultNodes.end());
}

unsigned CompleteMatchMatcher::getHashImpl() const {
  return HashUnsigneds(Results.begin(), Results.end()) ^ 
          ((unsigned)(intptr_t)&Pattern << 8);
}

// isContradictoryImpl Implementations.

static bool TypesAreContradictory(MVT::SimpleValueType T1,
                                  MVT::SimpleValueType T2) {
  // If the two types are the same, then they are the same, so they don't
  // contradict.
  if (T1 == T2) return false;
  
  // If either type is about iPtr, then they don't conflict unless the other
  // one is not a scalar integer type.
  if (T1 == MVT::iPTR)
    return !MVT(T2).isInteger() || MVT(T2).isVector();
  
  if (T2 == MVT::iPTR)
    return !MVT(T1).isInteger() || MVT(T1).isVector();
  
  // Otherwise, they are two different non-iPTR types, they conflict.
  return true;
}

bool CheckOpcodeMatcher::isContradictoryImpl(const Matcher *M) const {
  if (const CheckOpcodeMatcher *COM = dyn_cast<CheckOpcodeMatcher>(M)) {
    // One node can't have two different opcodes!
    return &COM->getOpcode() != &getOpcode();
  }
  
  // TODO: CheckMultiOpcodeMatcher?
  
  // If the node has a known type, and if the type we're checking for is
  // different, then we know they contradict.  For example, a check for
  // ISD::STORE will never be true at the same time a check for Type i32 is.
  if (const CheckTypeMatcher *CT = dyn_cast<CheckTypeMatcher>(M)) {
    // FIXME: What result is this referring to?
    unsigned NodeType;
    if (getOpcode().getNumResults() == 0)
      NodeType = MVT::isVoid;
    else
      NodeType = getOpcode().getKnownType();
    if (NodeType != EEVT::isUnknown)
      return TypesAreContradictory((MVT::SimpleValueType)NodeType,
                                   CT->getType());
  }
  
  return false;
}

bool CheckTypeMatcher::isContradictoryImpl(const Matcher *M) const {
  if (const CheckTypeMatcher *CT = dyn_cast<CheckTypeMatcher>(M))
    return TypesAreContradictory(getType(), CT->getType());
  return false;
}

bool CheckChildTypeMatcher::isContradictoryImpl(const Matcher *M) const {
  if (const CheckChildTypeMatcher *CC = dyn_cast<CheckChildTypeMatcher>(M)) {
    // If the two checks are about different nodes, we don't know if they
    // conflict!
    if (CC->getChildNo() != getChildNo())
      return false;
    
    return TypesAreContradictory(getType(), CC->getType());
  }
  return false;
}
  
bool CheckIntegerMatcher::isContradictoryImpl(const Matcher *M) const {
  if (const CheckIntegerMatcher *CIM = dyn_cast<CheckIntegerMatcher>(M))
    return CIM->getValue() != getValue();
  return false;
}
