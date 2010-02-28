//===- DAGISelMatcherEmitter.cpp - Matcher Emitter ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to generate C++ code a matcher.
//
//===----------------------------------------------------------------------===//

#include "DAGISelMatcher.h"
#include "CodeGenDAGPatterns.h"
#include "Record.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

enum {
  CommentIndent = 30
};

/// ClassifyInt - Classify an integer by size, return '1','2','4','8' if this
/// fits in 1, 2, 4, or 8 sign extended bytes.
static char ClassifyInt(int64_t Val) {
  if (Val == int8_t(Val))  return '1';
  if (Val == int16_t(Val)) return '2';
  if (Val == int32_t(Val)) return '4';
  return '8';
}

/// EmitInt - Emit the specified integer, returning the number of bytes emitted.
static unsigned EmitInt(int64_t Val, formatted_raw_ostream &OS) {
  unsigned BytesEmitted = 1;
  OS << (int)(unsigned char)Val << ", ";
  if (Val == int8_t(Val)) {
    OS << '\n';
    return BytesEmitted;
  }
  
  OS << (int)(unsigned char)(Val >> 8) << ", ";
  ++BytesEmitted;
  
  if (Val != int16_t(Val)) {
    OS << (int)(unsigned char)(Val >> 16) << ", "
       << (int)(unsigned char)(Val >> 24) << ", ";
    BytesEmitted += 2;
    
    if (Val != int32_t(Val)) {
      OS << (int)(unsigned char)(Val >> 32) << ", "
         << (int)(unsigned char)(Val >> 40) << ", "
         << (int)(unsigned char)(Val >> 48) << ", "
         << (int)(unsigned char)(Val >> 56) << ", ";
      BytesEmitted += 4;
    }   
  }
  
  OS.PadToColumn(CommentIndent) << "// " << Val << " aka 0x";
  OS.write_hex(Val) << '\n';
  return BytesEmitted;
}

namespace {
class MatcherTableEmitter {
  StringMap<unsigned> NodePredicateMap, PatternPredicateMap;
  std::vector<std::string> NodePredicates, PatternPredicates;

  DenseMap<const ComplexPattern*, unsigned> ComplexPatternMap;
  std::vector<const ComplexPattern*> ComplexPatterns;


  DenseMap<Record*, unsigned> NodeXFormMap;
  std::vector<const Record*> NodeXForms;

  // Per opcode frequence count. 
  std::vector<unsigned> Histogram;
public:
  MatcherTableEmitter() {}

  unsigned EmitMatcherList(const Matcher *N, unsigned Indent,
                           unsigned StartIdx, formatted_raw_ostream &OS);
  
  void EmitPredicateFunctions(formatted_raw_ostream &OS);
  
  void EmitHistogram(formatted_raw_ostream &OS);
private:
  unsigned EmitMatcher(const Matcher *N, unsigned Indent, unsigned CurrentIdx,
                       formatted_raw_ostream &OS);
  
  unsigned getNodePredicate(StringRef PredName) {
    unsigned &Entry = NodePredicateMap[PredName];
    if (Entry == 0) {
      NodePredicates.push_back(PredName.str());
      Entry = NodePredicates.size();
    }
    return Entry-1;
  }
  unsigned getPatternPredicate(StringRef PredName) {
    unsigned &Entry = PatternPredicateMap[PredName];
    if (Entry == 0) {
      PatternPredicates.push_back(PredName.str());
      Entry = PatternPredicates.size();
    }
    return Entry-1;
  }
  
  unsigned getComplexPat(const ComplexPattern &P) {
    unsigned &Entry = ComplexPatternMap[&P];
    if (Entry == 0) {
      ComplexPatterns.push_back(&P);
      Entry = ComplexPatterns.size();
    }
    return Entry-1;
  }
  
  unsigned getNodeXFormID(Record *Rec) {
    unsigned &Entry = NodeXFormMap[Rec];
    if (Entry == 0) {
      NodeXForms.push_back(Rec);
      Entry = NodeXForms.size();
    }
    return Entry-1;
  }
  
};
} // end anonymous namespace.

static unsigned GetVBRSize(unsigned Val) {
  if (Val <= 127) return 1;
  
  unsigned NumBytes = 0;
  while (Val >= 128) {
    Val >>= 7;
    ++NumBytes;
  }
  return NumBytes+1;
}

/// EmitVBRValue - Emit the specified value as a VBR, returning the number of
/// bytes emitted.
static unsigned EmitVBRValue(unsigned Val, raw_ostream &OS) {
  if (Val <= 127) {
    OS << Val << ", ";
    return 1;
  }
  
  unsigned InVal = Val;
  unsigned NumBytes = 0;
  while (Val >= 128) {
    OS << (Val&127) << "|128,";
    Val >>= 7;
    ++NumBytes;
  }
  OS << Val << "/*" << InVal << "*/, ";
  return NumBytes+1;
}

/// EmitMatcherOpcodes - Emit bytes for the specified matcher and return
/// the number of bytes emitted.
unsigned MatcherTableEmitter::
EmitMatcher(const Matcher *N, unsigned Indent, unsigned CurrentIdx,
            formatted_raw_ostream &OS) {
  OS.PadToColumn(Indent*2);
  
  switch (N->getKind()) {
  case Matcher::Scope: {
    const ScopeMatcher *SM = cast<ScopeMatcher>(N);
    assert(SM->getNext() == 0 && "Shouldn't have next after scope");
    
    unsigned StartIdx = CurrentIdx;
    
    // Emit all of the children.
    for (unsigned i = 0, e = SM->getNumChildren(); i != e; ++i) {
      if (i == 0) {
        OS << "OPC_Scope, ";
        ++CurrentIdx;
      } else {
        OS << "/*" << CurrentIdx << "*/";
        OS.PadToColumn(Indent*2) << "/*Scope*/ ";
      }

      // We need to encode the child and the offset of the failure code before
      // emitting either of them.  Handle this by buffering the output into a
      // string while we get the size.  Unfortunately, the offset of the
      // children depends on the VBR size of the child, so for large children we
      // have to iterate a bit.
      SmallString<128> TmpBuf;
      unsigned ChildSize = 0;
      unsigned VBRSize = 0;
      do {
        VBRSize = GetVBRSize(ChildSize);
        
        TmpBuf.clear();
        raw_svector_ostream OS(TmpBuf);
        formatted_raw_ostream FOS(OS);
        ChildSize = EmitMatcherList(cast<ScopeMatcher>(N)->getChild(i),
                                   Indent+1, CurrentIdx+VBRSize, FOS);
      } while (GetVBRSize(ChildSize) != VBRSize);
      
      assert(ChildSize != 0 && "Should not have a zero-sized child!");
    
      CurrentIdx += EmitVBRValue(ChildSize, OS);
      OS << "/*->" << CurrentIdx+ChildSize << "*/";
      
      if (i == 0)
        OS.PadToColumn(CommentIndent) << "// " << SM->getNumChildren()
          << " children in Scope";
      
      OS << '\n' << TmpBuf.str();
      CurrentIdx += ChildSize;
    }
    
    // Emit a zero as a sentinel indicating end of 'Scope'.
    OS << "/*" << CurrentIdx << "*/";
    OS.PadToColumn(Indent*2) << "0, /*End of Scope*/\n";
    return CurrentIdx - StartIdx + 1;
  }
      
  case Matcher::RecordNode:
    OS << "OPC_RecordNode,";
    OS.PadToColumn(CommentIndent) << "// "
       << cast<RecordMatcher>(N)->getWhatFor() << '\n';
    return 1;

  case Matcher::RecordChild:
    OS << "OPC_RecordChild" << cast<RecordChildMatcher>(N)->getChildNo()
       << ',';
    OS.PadToColumn(CommentIndent) << "// "
      << cast<RecordChildMatcher>(N)->getWhatFor() << '\n';
    return 1;
      
  case Matcher::RecordMemRef:
    OS << "OPC_RecordMemRef,\n";
    return 1;
      
  case Matcher::CaptureFlagInput:
    OS << "OPC_CaptureFlagInput,\n";
    return 1;
      
  case Matcher::MoveChild:
    OS << "OPC_MoveChild, " << cast<MoveChildMatcher>(N)->getChildNo() << ",\n";
    return 2;
      
  case Matcher::MoveParent:
    OS << "OPC_MoveParent,\n";
    return 1;
      
  case Matcher::CheckSame:
    OS << "OPC_CheckSame, "
       << cast<CheckSameMatcher>(N)->getMatchNumber() << ",\n";
    return 2;

  case Matcher::CheckPatternPredicate: {
    StringRef Pred = cast<CheckPatternPredicateMatcher>(N)->getPredicate();
    OS << "OPC_CheckPatternPredicate, " << getPatternPredicate(Pred) << ',';
    OS.PadToColumn(CommentIndent) << "// " << Pred << '\n';
    return 2;
  }
  case Matcher::CheckPredicate: {
    StringRef Pred = cast<CheckPredicateMatcher>(N)->getPredicateName();
    OS << "OPC_CheckPredicate, " << getNodePredicate(Pred) << ',';
    OS.PadToColumn(CommentIndent) << "// " << Pred << '\n';
    return 2;
  }

  case Matcher::CheckOpcode:
    OS << "OPC_CheckOpcode, "
       << cast<CheckOpcodeMatcher>(N)->getOpcode().getEnumName() << ",\n";
    return 2;
      
  case Matcher::CheckMultiOpcode: {
    const CheckMultiOpcodeMatcher *CMO = cast<CheckMultiOpcodeMatcher>(N);
    OS << "OPC_CheckMultiOpcode, " << CMO->getNumOpcodes() << ", ";
    for (unsigned i = 0, e = CMO->getNumOpcodes(); i != e; ++i)
      OS << CMO->getOpcode(i).getEnumName() << ", ";
    OS << '\n';
    return 2 + CMO->getNumOpcodes();
  }
      
  case Matcher::CheckType:
    OS << "OPC_CheckType, "
       << getEnumName(cast<CheckTypeMatcher>(N)->getType()) << ",\n";
    return 2;
  case Matcher::CheckChildType:
    OS << "OPC_CheckChild"
       << cast<CheckChildTypeMatcher>(N)->getChildNo() << "Type, "
       << getEnumName(cast<CheckChildTypeMatcher>(N)->getType()) << ",\n";
    return 2;
      
  case Matcher::CheckInteger: {
    int64_t Val = cast<CheckIntegerMatcher>(N)->getValue();
    OS << "OPC_CheckInteger" << ClassifyInt(Val) << ", ";
    return EmitInt(Val, OS)+1;
  }   
  case Matcher::CheckCondCode:
    OS << "OPC_CheckCondCode, ISD::"
       << cast<CheckCondCodeMatcher>(N)->getCondCodeName() << ",\n";
    return 2;
      
  case Matcher::CheckValueType:
    OS << "OPC_CheckValueType, MVT::"
       << cast<CheckValueTypeMatcher>(N)->getTypeName() << ",\n";
    return 2;

  case Matcher::CheckComplexPat: {
    const ComplexPattern &Pattern =
      cast<CheckComplexPatMatcher>(N)->getPattern();
    OS << "OPC_CheckComplexPat, " << getComplexPat(Pattern) << ',';
    OS.PadToColumn(CommentIndent) << "// " << Pattern.getSelectFunc();
    OS << ": " << Pattern.getNumOperands() << " operands";
    if (Pattern.hasProperty(SDNPHasChain))
      OS << " + chain result and input";
    OS << '\n';
    return 2;
  }
      
  case Matcher::CheckAndImm: {
    int64_t Val = cast<CheckAndImmMatcher>(N)->getValue();
    OS << "OPC_CheckAndImm" << ClassifyInt(Val) << ", ";
    return EmitInt(Val, OS)+1;
  }

  case Matcher::CheckOrImm: {
    int64_t Val = cast<CheckOrImmMatcher>(N)->getValue();
    OS << "OPC_CheckOrImm" << ClassifyInt(Val) << ", ";
    return EmitInt(Val, OS)+1;
  }
  case Matcher::CheckFoldableChainNode:
    OS << "OPC_CheckFoldableChainNode,\n";
    return 1;
  case Matcher::CheckChainCompatible:
    OS << "OPC_CheckChainCompatible, "
       << cast<CheckChainCompatibleMatcher>(N)->getPreviousOp() << ",\n";
    return 2;
      
  case Matcher::EmitInteger: {
    int64_t Val = cast<EmitIntegerMatcher>(N)->getValue();
    OS << "OPC_EmitInteger" << ClassifyInt(Val) << ", "
       << getEnumName(cast<EmitIntegerMatcher>(N)->getVT()) << ", ";
    return EmitInt(Val, OS)+2;
  }
  case Matcher::EmitStringInteger: {
    const std::string &Val = cast<EmitStringIntegerMatcher>(N)->getValue();
    // These should always fit into one byte.
    OS << "OPC_EmitInteger1, "
      << getEnumName(cast<EmitStringIntegerMatcher>(N)->getVT()) << ", "
      << Val << ",\n";
    return 3;
  }
      
  case Matcher::EmitRegister:
    OS << "OPC_EmitRegister, "
       << getEnumName(cast<EmitRegisterMatcher>(N)->getVT()) << ", ";
    if (Record *R = cast<EmitRegisterMatcher>(N)->getReg())
      OS << getQualifiedName(R) << ",\n";
    else
      OS << "0 /*zero_reg*/,\n";
    return 3;
      
  case Matcher::EmitConvertToTarget:
    OS << "OPC_EmitConvertToTarget, "
       << cast<EmitConvertToTargetMatcher>(N)->getSlot() << ",\n";
    return 2;
      
  case Matcher::EmitMergeInputChains: {
    const EmitMergeInputChainsMatcher *MN =
      cast<EmitMergeInputChainsMatcher>(N);
    OS << "OPC_EmitMergeInputChains, " << MN->getNumNodes() << ", ";
    for (unsigned i = 0, e = MN->getNumNodes(); i != e; ++i)
      OS << MN->getNode(i) << ", ";
    OS << '\n';
    return 2+MN->getNumNodes();
  }
  case Matcher::EmitCopyToReg:
    OS << "OPC_EmitCopyToReg, "
       << cast<EmitCopyToRegMatcher>(N)->getSrcSlot() << ", "
       << getQualifiedName(cast<EmitCopyToRegMatcher>(N)->getDestPhysReg())
       << ",\n";
    return 3;
  case Matcher::EmitNodeXForm: {
    const EmitNodeXFormMatcher *XF = cast<EmitNodeXFormMatcher>(N);
    OS << "OPC_EmitNodeXForm, " << getNodeXFormID(XF->getNodeXForm()) << ", "
       << XF->getSlot() << ',';
    OS.PadToColumn(CommentIndent) << "// "<<XF->getNodeXForm()->getName()<<'\n';
    return 3;
  }
      
  case Matcher::EmitNode:
  case Matcher::MorphNodeTo: {
    const EmitNodeMatcherCommon *EN = cast<EmitNodeMatcherCommon>(N);
    OS << (isa<EmitNodeMatcher>(EN) ? "OPC_EmitNode" : "OPC_MorphNodeTo");
    OS << ", TARGET_OPCODE(" << EN->getOpcodeName() << "), 0";
    
    if (EN->hasChain())   OS << "|OPFL_Chain";
    if (EN->hasFlag())    OS << "|OPFL_Flag";
    if (EN->hasMemRefs()) OS << "|OPFL_MemRefs";
    if (EN->getNumFixedArityOperands() != -1)
      OS << "|OPFL_Variadic" << EN->getNumFixedArityOperands();
    OS << ",\n";
    
    OS.PadToColumn(Indent*2+4) << EN->getNumVTs() << "/*#VTs*/, ";
    for (unsigned i = 0, e = EN->getNumVTs(); i != e; ++i)
      OS << getEnumName(EN->getVT(i)) << ", ";

    OS << EN->getNumOperands() << "/*#Ops*/, ";
    unsigned NumOperandBytes = 0;
    for (unsigned i = 0, e = EN->getNumOperands(); i != e; ++i) {
      // We emit the operand numbers in VBR encoded format, in case the number
      // is too large to represent with a byte.
      NumOperandBytes += EmitVBRValue(EN->getOperand(i), OS);
    }
    
    // Print the result #'s for EmitNode.
    if (const EmitNodeMatcher *E = dyn_cast<EmitNodeMatcher>(EN)) {
      if (unsigned NumResults = EN->getNumNonChainFlagVTs()) {
        OS.PadToColumn(CommentIndent) << "// Results = ";
        unsigned First = E->getFirstResultSlot();
        for (unsigned i = 0; i != NumResults; ++i)
          OS << "#" << First+i << " ";
      }
    }
    OS << '\n';
    
    if (const MorphNodeToMatcher *SNT = dyn_cast<MorphNodeToMatcher>(N)) {
      OS.PadToColumn(Indent*2) << "// Src: "
      << *SNT->getPattern().getSrcPattern() << '\n';
      OS.PadToColumn(Indent*2) << "// Dst: " 
      << *SNT->getPattern().getDstPattern() << '\n';
      
    }
    
    return 6+EN->getNumVTs()+NumOperandBytes;
  }
  case Matcher::MarkFlagResults: {
    const MarkFlagResultsMatcher *CFR = cast<MarkFlagResultsMatcher>(N);
    OS << "OPC_MarkFlagResults, " << CFR->getNumNodes() << ", ";
    unsigned NumOperandBytes = 0;
    for (unsigned i = 0, e = CFR->getNumNodes(); i != e; ++i)
      NumOperandBytes += EmitVBRValue(CFR->getNode(i), OS);
    OS << '\n';
    return 2+NumOperandBytes;
  }
  case Matcher::CompleteMatch: {
    const CompleteMatchMatcher *CM = cast<CompleteMatchMatcher>(N);
    OS << "OPC_CompleteMatch, " << CM->getNumResults() << ", ";
    unsigned NumResultBytes = 0;
    for (unsigned i = 0, e = CM->getNumResults(); i != e; ++i)
      NumResultBytes += EmitVBRValue(CM->getResult(i), OS);
    OS << '\n';
    OS.PadToColumn(Indent*2) << "// Src: "
      << *CM->getPattern().getSrcPattern() << '\n';
    OS.PadToColumn(Indent*2) << "// Dst: " 
      << *CM->getPattern().getDstPattern() << '\n';
    return 2 + NumResultBytes;
  }
  }
  assert(0 && "Unreachable");
  return 0;
}

/// EmitMatcherList - Emit the bytes for the specified matcher subtree.
unsigned MatcherTableEmitter::
EmitMatcherList(const Matcher *N, unsigned Indent, unsigned CurrentIdx,
                formatted_raw_ostream &OS) {
  unsigned Size = 0;
  while (N) {
    if (unsigned(N->getKind()) >= Histogram.size())
      Histogram.resize(N->getKind()+1);
    Histogram[N->getKind()]++;
    
    OS << "/*" << CurrentIdx << "*/";
    unsigned MatcherSize = EmitMatcher(N, Indent, CurrentIdx, OS);
    Size += MatcherSize;
    CurrentIdx += MatcherSize;
    
    // If there are other nodes in this list, iterate to them, otherwise we're
    // done.
    N = N->getNext();
  }
  return Size;
}

void MatcherTableEmitter::EmitPredicateFunctions(formatted_raw_ostream &OS) {
  // FIXME: Don't build off the DAGISelEmitter's predicates, emit them directly
  // here into the case stmts.
  
  // Emit pattern predicates.
  OS << "bool CheckPatternPredicate(unsigned PredNo) const {\n";
  OS << "  switch (PredNo) {\n";
  OS << "  default: assert(0 && \"Invalid predicate in table?\");\n";
  for (unsigned i = 0, e = PatternPredicates.size(); i != e; ++i)
    OS << "  case " << i << ": return "  << PatternPredicates[i] << ";\n";
  OS << "  }\n";
  OS << "}\n\n";

  // Emit Node predicates.
  OS << "bool CheckNodePredicate(SDNode *N, unsigned PredNo) const {\n";
  OS << "  switch (PredNo) {\n";
  OS << "  default: assert(0 && \"Invalid predicate in table?\");\n";
  for (unsigned i = 0, e = NodePredicates.size(); i != e; ++i)
    OS << "  case " << i << ": return "  << NodePredicates[i] << "(N);\n";
  OS << "  }\n";
  OS << "}\n\n";
  
  // Emit CompletePattern matchers.
  // FIXME: This should be const.
  OS << "bool CheckComplexPattern(SDNode *Root, SDValue N,\n";
  OS << "      unsigned PatternNo, SmallVectorImpl<SDValue> &Result) {\n";
  OS << "  switch (PatternNo) {\n";
  OS << "  default: assert(0 && \"Invalid pattern # in table?\");\n";
  for (unsigned i = 0, e = ComplexPatterns.size(); i != e; ++i) {
    const ComplexPattern &P = *ComplexPatterns[i];
    unsigned NumOps = P.getNumOperands();

    if (P.hasProperty(SDNPHasChain))
      ++NumOps;  // Get the chained node too.
    
    OS << "  case " << i << ":\n";
    OS << "    Result.resize(Result.size()+" << NumOps << ");\n";
    OS << "    return "  << P.getSelectFunc();

    // FIXME: Temporary hack until old isel dies.
    if (P.hasProperty(SDNPHasChain))
      OS << "XXX";
    
    OS << "(Root, N";
    for (unsigned i = 0; i != NumOps; ++i)
      OS << ", Result[Result.size()-" << (NumOps-i) << ']';
    OS << ");\n";
  }
  OS << "  }\n";
  OS << "}\n\n";
  
  // Emit SDNodeXForm handlers.
  // FIXME: This should be const.
  OS << "SDValue RunSDNodeXForm(SDValue V, unsigned XFormNo) {\n";
  OS << "  switch (XFormNo) {\n";
  OS << "  default: assert(0 && \"Invalid xform # in table?\");\n";
  
  // FIXME: The node xform could take SDValue's instead of SDNode*'s.
  for (unsigned i = 0, e = NodeXForms.size(); i != e; ++i)
    OS << "  case " << i << ": return Transform_" << NodeXForms[i]->getName()
       << "(V.getNode());\n";
  OS << "  }\n";
  OS << "}\n\n";
}

void MatcherTableEmitter::EmitHistogram(formatted_raw_ostream &OS) {
  OS << "  // Opcode Histogram:\n";
  for (unsigned i = 0, e = Histogram.size(); i != e; ++i) {
    OS << "  // #";
    switch ((Matcher::KindTy)i) {
    case Matcher::Scope: OS << "OPC_Scope"; break; 
    case Matcher::RecordNode: OS << "OPC_RecordNode"; break; 
    case Matcher::RecordChild: OS << "OPC_RecordChild"; break;
    case Matcher::RecordMemRef: OS << "OPC_RecordMemRef"; break;
    case Matcher::CaptureFlagInput: OS << "OPC_CaptureFlagInput"; break;
    case Matcher::MoveChild: OS << "OPC_MoveChild"; break;
    case Matcher::MoveParent: OS << "OPC_MoveParent"; break;
    case Matcher::CheckSame: OS << "OPC_CheckSame"; break;
    case Matcher::CheckPatternPredicate:
      OS << "OPC_CheckPatternPredicate"; break;
    case Matcher::CheckPredicate: OS << "OPC_CheckPredicate"; break;
    case Matcher::CheckOpcode: OS << "OPC_CheckOpcode"; break;
    case Matcher::CheckMultiOpcode: OS << "OPC_CheckMultiOpcode"; break;
    case Matcher::CheckType: OS << "OPC_CheckType"; break;
    case Matcher::CheckChildType: OS << "OPC_CheckChildType"; break;
    case Matcher::CheckInteger: OS << "OPC_CheckInteger"; break;
    case Matcher::CheckCondCode: OS << "OPC_CheckCondCode"; break;
    case Matcher::CheckValueType: OS << "OPC_CheckValueType"; break;
    case Matcher::CheckComplexPat: OS << "OPC_CheckComplexPat"; break;
    case Matcher::CheckAndImm: OS << "OPC_CheckAndImm"; break;
    case Matcher::CheckOrImm: OS << "OPC_CheckOrImm"; break;
    case Matcher::CheckFoldableChainNode:
      OS << "OPC_CheckFoldableChainNode"; break;
    case Matcher::CheckChainCompatible: OS << "OPC_CheckChainCompatible"; break;
    case Matcher::EmitInteger: OS << "OPC_EmitInteger"; break;
    case Matcher::EmitStringInteger: OS << "OPC_EmitStringInteger"; break;
    case Matcher::EmitRegister: OS << "OPC_EmitRegister"; break;
    case Matcher::EmitConvertToTarget: OS << "OPC_EmitConvertToTarget"; break;
    case Matcher::EmitMergeInputChains: OS << "OPC_EmitMergeInputChains"; break;
    case Matcher::EmitCopyToReg: OS << "OPC_EmitCopyToReg"; break;
    case Matcher::EmitNode: OS << "OPC_EmitNode"; break;
    case Matcher::MorphNodeTo: OS << "OPC_MorphNodeTo"; break;
    case Matcher::EmitNodeXForm: OS << "OPC_EmitNodeXForm"; break;
    case Matcher::MarkFlagResults: OS << "OPC_MarkFlagResults"; break;
    case Matcher::CompleteMatch: OS << "OPC_CompleteMatch"; break;    
    }
    
    OS.PadToColumn(40) << " = " << Histogram[i] << '\n';
  }
  OS << '\n';
}


void llvm::EmitMatcherTable(const Matcher *TheMatcher, raw_ostream &O) {
  formatted_raw_ostream OS(O);
  
  OS << "// The main instruction selector code.\n";
  OS << "SDNode *SelectCode(SDNode *N) {\n";

  MatcherTableEmitter MatcherEmitter;

  OS << "  // Opcodes are emitted as 2 bytes, TARGET_OPCODE handles this.\n";
  OS << "  #define TARGET_OPCODE(X) X & 255, unsigned(X) >> 8\n";
  OS << "  static const unsigned char MatcherTable[] = {\n";
  unsigned TotalSize = MatcherEmitter.EmitMatcherList(TheMatcher, 5, 0, OS);
  OS << "    0\n  }; // Total Array size is " << (TotalSize+1) << " bytes\n\n";
  
  MatcherEmitter.EmitHistogram(OS);
  
  OS << "  #undef TARGET_OPCODE\n";
  OS << "  return SelectCodeCommon(N, MatcherTable,sizeof(MatcherTable));\n}\n";
  OS << "\n";
  
  // Next up, emit the function for node and pattern predicates:
  MatcherEmitter.EmitPredicateFunctions(OS);
}
