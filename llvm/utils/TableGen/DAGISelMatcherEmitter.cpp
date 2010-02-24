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

  unsigned EmitMatcherList(const MatcherNode *N, unsigned Indent,
                           unsigned StartIdx, formatted_raw_ostream &OS);
  
  void EmitPredicateFunctions(formatted_raw_ostream &OS);
  
  void EmitHistogram(formatted_raw_ostream &OS);
private:
  unsigned EmitMatcher(const MatcherNode *N, unsigned Indent,
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
EmitMatcher(const MatcherNode *N, unsigned Indent, formatted_raw_ostream &OS) {
  OS.PadToColumn(Indent*2);
  
  switch (N->getKind()) {
  case MatcherNode::Push: assert(0 && "Should be handled by caller");
  case MatcherNode::RecordNode:
    OS << "OPC_RecordNode,";
    OS.PadToColumn(CommentIndent) << "// "
       << cast<RecordMatcherNode>(N)->getWhatFor() << '\n';
    return 1;

  case MatcherNode::RecordChild:
    OS << "OPC_RecordChild" << cast<RecordChildMatcherNode>(N)->getChildNo()
       << ',';
    OS.PadToColumn(CommentIndent) << "// "
      << cast<RecordChildMatcherNode>(N)->getWhatFor() << '\n';
    return 1;
      
  case MatcherNode::RecordMemRef:
    OS << "OPC_RecordMemRef,\n";
    return 1;
      
  case MatcherNode::CaptureFlagInput:
    OS << "OPC_CaptureFlagInput,\n";
    return 1;
      
  case MatcherNode::MoveChild:
    OS << "OPC_MoveChild, "
       << cast<MoveChildMatcherNode>(N)->getChildNo() << ",\n";
    return 2;
      
  case MatcherNode::MoveParent:
    OS << "OPC_MoveParent,\n";
    return 1;
      
  case MatcherNode::CheckSame:
    OS << "OPC_CheckSame, "
       << cast<CheckSameMatcherNode>(N)->getMatchNumber() << ",\n";
    return 2;

  case MatcherNode::CheckPatternPredicate: {
    StringRef Pred = cast<CheckPatternPredicateMatcherNode>(N)->getPredicate();
    OS << "OPC_CheckPatternPredicate, " << getPatternPredicate(Pred) << ',';
    OS.PadToColumn(CommentIndent) << "// " << Pred << '\n';
    return 2;
  }
  case MatcherNode::CheckPredicate: {
    StringRef Pred = cast<CheckPredicateMatcherNode>(N)->getPredicateName();
    OS << "OPC_CheckPredicate, " << getNodePredicate(Pred) << ',';
    OS.PadToColumn(CommentIndent) << "// " << Pred << '\n';
    return 2;
  }

  case MatcherNode::CheckOpcode:
    OS << "OPC_CheckOpcode, "
       << cast<CheckOpcodeMatcherNode>(N)->getOpcodeName() << ",\n";
    return 2;
      
  case MatcherNode::CheckMultiOpcode: {
    const CheckMultiOpcodeMatcherNode *CMO=cast<CheckMultiOpcodeMatcherNode>(N);
    OS << "OPC_CheckMultiOpcode, " << CMO->getNumOpcodeNames() << ", ";
    for (unsigned i = 0, e = CMO->getNumOpcodeNames(); i != e; ++i)
      OS << CMO->getOpcodeName(i) << ", ";
    OS << '\n';
    return 2 + CMO->getNumOpcodeNames();
  }
      
  case MatcherNode::CheckType:
    OS << "OPC_CheckType, "
       << getEnumName(cast<CheckTypeMatcherNode>(N)->getType()) << ",\n";
    return 2;
  case MatcherNode::CheckChildType:
    OS << "OPC_CheckChild"
       << cast<CheckChildTypeMatcherNode>(N)->getChildNo() << "Type, "
       << getEnumName(cast<CheckChildTypeMatcherNode>(N)->getType()) << ",\n";
    return 2;
      
  case MatcherNode::CheckInteger: {
    int64_t Val = cast<CheckIntegerMatcherNode>(N)->getValue();
    OS << "OPC_CheckInteger" << ClassifyInt(Val) << ", ";
    return EmitInt(Val, OS)+1;
  }   
  case MatcherNode::CheckCondCode:
    OS << "OPC_CheckCondCode, ISD::"
       << cast<CheckCondCodeMatcherNode>(N)->getCondCodeName() << ",\n";
    return 2;
      
  case MatcherNode::CheckValueType:
    OS << "OPC_CheckValueType, MVT::"
       << cast<CheckValueTypeMatcherNode>(N)->getTypeName() << ",\n";
    return 2;

  case MatcherNode::CheckComplexPat: {
    const ComplexPattern &Pattern =
      cast<CheckComplexPatMatcherNode>(N)->getPattern();
    OS << "OPC_CheckComplexPat, " << getComplexPat(Pattern) << ',';
    OS.PadToColumn(CommentIndent) << "// " << Pattern.getSelectFunc();
    OS << ": " << Pattern.getNumOperands() << " operands";
    if (Pattern.hasProperty(SDNPHasChain))
      OS << " + chain result and input";
    OS << '\n';
    return 2;
  }
      
  case MatcherNode::CheckAndImm: {
    int64_t Val = cast<CheckAndImmMatcherNode>(N)->getValue();
    OS << "OPC_CheckAndImm" << ClassifyInt(Val) << ", ";
    return EmitInt(Val, OS)+1;
  }

  case MatcherNode::CheckOrImm: {
    int64_t Val = cast<CheckOrImmMatcherNode>(N)->getValue();
    OS << "OPC_CheckOrImm" << ClassifyInt(Val) << ", ";
    return EmitInt(Val, OS)+1;
  }
  case MatcherNode::CheckFoldableChainNode:
    OS << "OPC_CheckFoldableChainNode,\n";
    return 1;
  case MatcherNode::CheckChainCompatible:
    OS << "OPC_CheckChainCompatible, "
       << cast<CheckChainCompatibleMatcherNode>(N)->getPreviousOp() << ",\n";
    return 2;
      
  case MatcherNode::EmitInteger: {
    int64_t Val = cast<EmitIntegerMatcherNode>(N)->getValue();
    OS << "OPC_EmitInteger" << ClassifyInt(Val) << ", "
       << getEnumName(cast<EmitIntegerMatcherNode>(N)->getVT()) << ", ";
    return EmitInt(Val, OS)+2;
  }
  case MatcherNode::EmitStringInteger: {
    const std::string &Val = cast<EmitStringIntegerMatcherNode>(N)->getValue();
    // These should always fit into one byte.
    OS << "OPC_EmitInteger1, "
      << getEnumName(cast<EmitStringIntegerMatcherNode>(N)->getVT()) << ", "
      << Val << ",\n";
    return 3;
  }
      
  case MatcherNode::EmitRegister:
    OS << "OPC_EmitRegister, "
       << getEnumName(cast<EmitRegisterMatcherNode>(N)->getVT()) << ", ";
    if (Record *R = cast<EmitRegisterMatcherNode>(N)->getReg())
      OS << getQualifiedName(R) << ",\n";
    else
      OS << "0 /*zero_reg*/,\n";
    return 3;
      
  case MatcherNode::EmitConvertToTarget:
    OS << "OPC_EmitConvertToTarget, "
       << cast<EmitConvertToTargetMatcherNode>(N)->getSlot() << ",\n";
    return 2;
      
  case MatcherNode::EmitMergeInputChains: {
    const EmitMergeInputChainsMatcherNode *MN =
      cast<EmitMergeInputChainsMatcherNode>(N);
    OS << "OPC_EmitMergeInputChains, " << MN->getNumNodes() << ", ";
    for (unsigned i = 0, e = MN->getNumNodes(); i != e; ++i)
      OS << MN->getNode(i) << ", ";
    OS << '\n';
    return 2+MN->getNumNodes();
  }
  case MatcherNode::EmitCopyToReg:
    OS << "OPC_EmitCopyToReg, "
       << cast<EmitCopyToRegMatcherNode>(N)->getSrcSlot() << ", "
       << getQualifiedName(cast<EmitCopyToRegMatcherNode>(N)->getDestPhysReg())
       << ",\n";
    return 3;
  case MatcherNode::EmitNodeXForm: {
    const EmitNodeXFormMatcherNode *XF = cast<EmitNodeXFormMatcherNode>(N);
    OS << "OPC_EmitNodeXForm, " << getNodeXFormID(XF->getNodeXForm()) << ", "
       << XF->getSlot() << ',';
    OS.PadToColumn(CommentIndent) << "// "<<XF->getNodeXForm()->getName()<<'\n';
    return 3;
  }
      
  case MatcherNode::EmitNode: {
    const EmitNodeMatcherNode *EN = cast<EmitNodeMatcherNode>(N);
    OS << "OPC_EmitNode, TARGET_OPCODE(" << EN->getOpcodeName() << "), 0";
    
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
    OS << '\n';
    return 6+EN->getNumVTs()+NumOperandBytes;
  }
  case MatcherNode::MarkFlagResults: {
    const MarkFlagResultsMatcherNode *CFR = cast<MarkFlagResultsMatcherNode>(N);
    OS << "OPC_MarkFlagResults, " << CFR->getNumNodes() << ", ";
    unsigned NumOperandBytes = 0;
    for (unsigned i = 0, e = CFR->getNumNodes(); i != e; ++i)
      NumOperandBytes += EmitVBRValue(CFR->getNode(i), OS);
    OS << '\n';
    return 2+NumOperandBytes;
  }
  case MatcherNode::CompleteMatch: {
    const CompleteMatchMatcherNode *CM = cast<CompleteMatchMatcherNode>(N);
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
EmitMatcherList(const MatcherNode *N, unsigned Indent, unsigned CurrentIdx,
                formatted_raw_ostream &OS) {
  unsigned Size = 0;
  while (N) {
    if (unsigned(N->getKind()) >= Histogram.size())
      Histogram.resize(N->getKind()+1);
    Histogram[N->getKind()]++;
    
    // Push is a special case since it is binary.
    if (const PushMatcherNode *PMN = dyn_cast<PushMatcherNode>(N)) {
      // We need to encode the child and the offset of the failure code before
      // emitting either of them.  Handle this by buffering the output into a
      // string while we get the size.
      SmallString<128> TmpBuf;
      unsigned NextSize;
      {
        raw_svector_ostream OS(TmpBuf);
        formatted_raw_ostream FOS(OS);
        NextSize = EmitMatcherList(cast<PushMatcherNode>(N)->getNext(),
                                   Indent+1, CurrentIdx+2, FOS);
      }

      // In the unlikely event that we have something too big to emit with a
      // one byte offset, regenerate it with a two-byte one.
      if (NextSize > 255) {
        TmpBuf.clear();
        raw_svector_ostream OS(TmpBuf);
        formatted_raw_ostream FOS(OS);
        NextSize = EmitMatcherList(cast<PushMatcherNode>(N)->getNext(),
                                   Indent+1, CurrentIdx+3, FOS);
        if (NextSize > 65535) {
          errs() <<
            "Tblgen internal error: can't handle pattern this complex yet\n";
          exit(1);
        }
      }
      
      OS << "/*" << CurrentIdx << "*/";
      OS.PadToColumn(Indent*2);
      
      if (NextSize < 256)
        OS << "OPC_Push, " << NextSize << ",\n";
      else
        OS << "OPC_Push2, " << (NextSize&255) << ", " << (NextSize>>8) << ",\n";
      OS << TmpBuf.str();
      
      Size += 2+NextSize;
      CurrentIdx += 2+NextSize;
      N = PMN->getFailure();
      continue;
    }
  
    OS << "/*" << CurrentIdx << "*/";
    unsigned MatcherSize = EmitMatcher(N, Indent, OS);
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
    switch ((MatcherNode::KindTy)i) {
    case MatcherNode::Push: OS << "OPC_Push"; break; 
    case MatcherNode::RecordNode: OS << "OPC_RecordNode"; break; 
    case MatcherNode::RecordChild: OS << "OPC_RecordChild"; break;
    case MatcherNode::RecordMemRef: OS << "OPC_RecordMemRef"; break;
    case MatcherNode::CaptureFlagInput: OS << "OPC_CaptureFlagInput"; break;
    case MatcherNode::MoveChild: OS << "OPC_MoveChild"; break;
    case MatcherNode::MoveParent: OS << "OPC_MoveParent"; break;
    case MatcherNode::CheckSame: OS << "OPC_CheckSame"; break;
    case MatcherNode::CheckPatternPredicate:
      OS << "OPC_CheckPatternPredicate"; break;
    case MatcherNode::CheckPredicate: OS << "OPC_CheckPredicate"; break;
    case MatcherNode::CheckOpcode: OS << "OPC_CheckOpcode"; break;
    case MatcherNode::CheckMultiOpcode: OS << "OPC_CheckMultiOpcode"; break;
    case MatcherNode::CheckType: OS << "OPC_CheckType"; break;
    case MatcherNode::CheckChildType: OS << "OPC_CheckChildType"; break;
    case MatcherNode::CheckInteger: OS << "OPC_CheckInteger"; break;
    case MatcherNode::CheckCondCode: OS << "OPC_CheckCondCode"; break;
    case MatcherNode::CheckValueType: OS << "OPC_CheckValueType"; break;
    case MatcherNode::CheckComplexPat: OS << "OPC_CheckComplexPat"; break;
    case MatcherNode::CheckAndImm: OS << "OPC_CheckAndImm"; break;
    case MatcherNode::CheckOrImm: OS << "OPC_CheckOrImm"; break;
    case MatcherNode::CheckFoldableChainNode:
      OS << "OPC_CheckFoldableChainNode"; break;
    case MatcherNode::CheckChainCompatible:
      OS << "OPC_CheckChainCompatible"; break;
    case MatcherNode::EmitInteger: OS << "OPC_EmitInteger"; break;
    case MatcherNode::EmitStringInteger: OS << "OPC_EmitStringInteger"; break;
    case MatcherNode::EmitRegister: OS << "OPC_EmitRegister"; break;
    case MatcherNode::EmitConvertToTarget:
      OS << "OPC_EmitConvertToTarget"; break;
    case MatcherNode::EmitMergeInputChains:
      OS << "OPC_EmitMergeInputChains"; break;
    case MatcherNode::EmitCopyToReg: OS << "OPC_EmitCopyToReg"; break;
    case MatcherNode::EmitNode: OS << "OPC_EmitNode"; break;
    case MatcherNode::EmitNodeXForm: OS << "OPC_EmitNodeXForm"; break;
    case MatcherNode::MarkFlagResults: OS << "OPC_MarkFlagResults"; break;
    case MatcherNode::CompleteMatch: OS << "OPC_CompleteMatch"; break;    
    }
    
    OS.PadToColumn(40) << " = " << Histogram[i] << '\n';
  }
  OS << '\n';
}


void llvm::EmitMatcherTable(const MatcherNode *Matcher, raw_ostream &O) {
  formatted_raw_ostream OS(O);
  
  OS << "// The main instruction selector code.\n";
  OS << "SDNode *SelectCode(SDNode *N) {\n";

  MatcherTableEmitter MatcherEmitter;

  OS << "  // Opcodes are emitted as 2 bytes, TARGET_OPCODE handles this.\n";
  OS << "  #define TARGET_OPCODE(X) X & 255, unsigned(X) >> 8\n";
  OS << "  static const unsigned char MatcherTable[] = {\n";
  unsigned TotalSize = MatcherEmitter.EmitMatcherList(Matcher, 5, 0, OS);
  OS << "    0\n  }; // Total Array size is " << (TotalSize+1) << " bytes\n\n";
  
  MatcherEmitter.EmitHistogram(OS);
  
  OS << "  #undef TARGET_OPCODE\n";
  OS << "  return SelectCodeCommon(N, MatcherTable,sizeof(MatcherTable));\n}\n";
  OS << "\n";
  
  // Next up, emit the function for node and pattern predicates:
  MatcherEmitter.EmitPredicateFunctions(OS);
}
