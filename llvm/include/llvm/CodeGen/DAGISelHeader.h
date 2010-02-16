//==-llvm/CodeGen/DAGISelHeader.h - Common DAG ISel definitions  -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides definitions of the common, target-independent methods and 
// data, which is used by SelectionDAG-based instruction selectors.
//
// *** NOTE: This file is #included into the middle of the target
// instruction selector class.  These functions are really methods.
// This is a little awkward, but it allows this code to be shared
// by all the targets while still being able to call into
// target-specific code without using a virtual function call.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DAGISEL_HEADER_H
#define LLVM_CODEGEN_DAGISEL_HEADER_H

/// ISelPosition - Node iterator marking the current position of
/// instruction selection as it procedes through the topologically-sorted
/// node list.
SelectionDAG::allnodes_iterator ISelPosition;

/// IsChainCompatible - Returns true if Chain is Op or Chain does
/// not reach Op.
static bool IsChainCompatible(SDNode *Chain, SDNode *Op) {
  if (Chain->getOpcode() == ISD::EntryToken)
    return true;
  if (Chain->getOpcode() == ISD::TokenFactor)
    return false;
  if (Chain->getNumOperands() > 0) {
    SDValue C0 = Chain->getOperand(0);
    if (C0.getValueType() == MVT::Other)
      return C0.getNode() != Op && IsChainCompatible(C0.getNode(), Op);
  }
  return true;
}

/// ISelUpdater - helper class to handle updates of the 
/// instruciton selection graph.
class VISIBILITY_HIDDEN ISelUpdater : public SelectionDAG::DAGUpdateListener {
  SelectionDAG::allnodes_iterator &ISelPosition;
public:
  explicit ISelUpdater(SelectionDAG::allnodes_iterator &isp)
    : ISelPosition(isp) {}
  
  /// NodeDeleted - Handle nodes deleted from the graph. If the
  /// node being deleted is the current ISelPosition node, update
  /// ISelPosition.
  ///
  virtual void NodeDeleted(SDNode *N, SDNode *E) {
    if (ISelPosition == SelectionDAG::allnodes_iterator(N))
      ++ISelPosition;
  }

  /// NodeUpdated - Ignore updates for now.
  virtual void NodeUpdated(SDNode *N) {}
};

/// ReplaceUses - replace all uses of the old node F with the use
/// of the new node T.
DISABLE_INLINE void ReplaceUses(SDValue F, SDValue T) {
  ISelUpdater ISU(ISelPosition);
  CurDAG->ReplaceAllUsesOfValueWith(F, T, &ISU);
}

/// ReplaceUses - replace all uses of the old nodes F with the use
/// of the new nodes T.
DISABLE_INLINE void ReplaceUses(const SDValue *F, const SDValue *T,
                                unsigned Num) {
  ISelUpdater ISU(ISelPosition);
  CurDAG->ReplaceAllUsesOfValuesWith(F, T, Num, &ISU);
}

/// ReplaceUses - replace all uses of the old node F with the use
/// of the new node T.
DISABLE_INLINE void ReplaceUses(SDNode *F, SDNode *T) {
  ISelUpdater ISU(ISelPosition);
  CurDAG->ReplaceAllUsesWith(F, T, &ISU);
}

/// SelectRoot - Top level entry to DAG instruction selector.
/// Selects instructions starting at the root of the current DAG.
void SelectRoot(SelectionDAG &DAG) {
  SelectRootInit();

  // Create a dummy node (which is not added to allnodes), that adds
  // a reference to the root node, preventing it from being deleted,
  // and tracking any changes of the root.
  HandleSDNode Dummy(CurDAG->getRoot());
  ISelPosition = llvm::next(SelectionDAG::allnodes_iterator(CurDAG->getRoot().getNode()));

  // The AllNodes list is now topological-sorted. Visit the
  // nodes by starting at the end of the list (the root of the
  // graph) and preceding back toward the beginning (the entry
  // node).
  while (ISelPosition != CurDAG->allnodes_begin()) {
    SDNode *Node = --ISelPosition;
    // Skip dead nodes. DAGCombiner is expected to eliminate all dead nodes,
    // but there are currently some corner cases that it misses. Also, this
    // makes it theoretically possible to disable the DAGCombiner.
    if (Node->use_empty())
      continue;
#if 0
    DAG.setSubgraphColor(Node, "red");
#endif
    SDNode *ResNode = Select(Node);
    // If node should not be replaced, continue with the next one.
    if (ResNode == Node)
      continue;
    // Replace node.
    if (ResNode) {
#if 0
      DAG.setSubgraphColor(ResNode, "yellow");
      DAG.setSubgraphColor(ResNode, "black");
#endif
      ReplaceUses(Node, ResNode);
    }
    // If after the replacement this node is not used any more,
    // remove this dead node.
    if (Node->use_empty()) { // Don't delete EntryToken, etc.
      ISelUpdater ISU(ISelPosition);
      CurDAG->RemoveDeadNode(Node, &ISU);
    }
  }

  CurDAG->setRoot(Dummy.getValue());
}


/// CheckInteger - Return true if the specified node is not a ConstantSDNode or
/// if it doesn't have the specified value.
static bool CheckInteger(SDValue V, int64_t Val) {
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(V);
  return C == 0 || C->getSExtValue() != Val;
}

/// CheckAndImmediate - Check to see if the specified node is an and with an
/// immediate returning true on failure.
///
/// FIXME: Inline this gunk into CheckAndMask.
bool CheckAndImmediate(SDValue V, int64_t Val) {
  if (V->getOpcode() == ISD::AND)
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(V->getOperand(1)))
      if (CheckAndMask(V.getOperand(0), C, Val))
        return false;
  return true;
}

/// CheckOrImmediate - Check to see if the specified node is an or with an
/// immediate returning true on failure.
///
/// FIXME: Inline this gunk into CheckOrMask.
bool CheckOrImmediate(SDValue V, int64_t Val) {
  if (V->getOpcode() == ISD::OR)
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(V->getOperand(1)))
      if (CheckOrMask(V.getOperand(0), C, Val))
        return false;
  return true;
}

// These functions are marked always inline so that Idx doesn't get pinned to
// the stack.
ALWAYS_INLINE static int8_t
GetInt1(const unsigned char *MatcherTable, unsigned &Idx) {
  return MatcherTable[Idx++];
}

ALWAYS_INLINE static int16_t
GetInt2(const unsigned char *MatcherTable, unsigned &Idx) {
  int16_t Val = GetInt1(MatcherTable, Idx);
  Val |= int16_t(GetInt1(MatcherTable, Idx)) << 8;
  return Val;
}

ALWAYS_INLINE static int32_t
GetInt4(const unsigned char *MatcherTable, unsigned &Idx) {
  int32_t Val = GetInt2(MatcherTable, Idx);
  Val |= int32_t(GetInt2(MatcherTable, Idx)) << 16;
  return Val;
}

ALWAYS_INLINE static int64_t
GetInt8(const unsigned char *MatcherTable, unsigned &Idx) {
  int64_t Val = GetInt4(MatcherTable, Idx);
  Val |= int64_t(GetInt4(MatcherTable, Idx)) << 32;
  return Val;
}

enum BuiltinOpcodes {
  OPC_Emit,
  OPC_Push,
  OPC_Record,
  OPC_MoveChild,
  OPC_MoveParent,
  OPC_CheckSame,
  OPC_CheckPatternPredicate,
  OPC_CheckPredicate,
  OPC_CheckOpcode,
  OPC_CheckType,
  OPC_CheckInteger1, OPC_CheckInteger2, OPC_CheckInteger4, OPC_CheckInteger8,
  OPC_CheckCondCode,
  OPC_CheckValueType,
  OPC_CheckComplexPat,
  OPC_CheckAndImm1, OPC_CheckAndImm2, OPC_CheckAndImm4, OPC_CheckAndImm8,
  OPC_CheckOrImm1, OPC_CheckOrImm2, OPC_CheckOrImm4, OPC_CheckOrImm8,
  OPC_CheckFoldableChainNode
};

struct MatchScope {
  /// FailIndex - If this match fails, this is the index to continue with.
  unsigned FailIndex;
  
  /// NodeStackSize - The size of the node stack when the scope was formed.
  unsigned NodeStackSize;
  
  /// NumRecordedNodes - The number of recorded nodes when the scope was formed.
  unsigned NumRecordedNodes;
};

SDNode *SelectCodeCommon(SDNode *NodeToMatch, const unsigned char *MatcherTable,
                         unsigned TableSize) {
  switch (NodeToMatch->getOpcode()) {
  default:
    break;
  case ISD::EntryToken:       // These nodes remain the same.
  case ISD::BasicBlock:
  case ISD::Register:
  case ISD::HANDLENODE:
  case ISD::TargetConstant:
  case ISD::TargetConstantFP:
  case ISD::TargetConstantPool:
  case ISD::TargetFrameIndex:
  case ISD::TargetExternalSymbol:
  case ISD::TargetBlockAddress:
  case ISD::TargetJumpTable:
  case ISD::TargetGlobalTLSAddress:
  case ISD::TargetGlobalAddress:
  case ISD::TokenFactor:
  case ISD::CopyFromReg:
  case ISD::CopyToReg:
    return 0;
  case ISD::AssertSext:
  case ISD::AssertZext:
    ReplaceUses(SDValue(NodeToMatch, 0), NodeToMatch->getOperand(0));
    return 0;
  case ISD::INLINEASM: return Select_INLINEASM(NodeToMatch);
  case ISD::EH_LABEL:  return Select_EH_LABEL(NodeToMatch);
  case ISD::UNDEF:     return Select_UNDEF(NodeToMatch);
  }
  
  assert(!NodeToMatch->isMachineOpcode() && "Node already selected!");
  
  SmallVector<MatchScope, 8> MatchScopes;
  
  // RecordedNodes - This is the set of nodes that have been recorded by the
  // state machine.
  SmallVector<SDValue, 8> RecordedNodes;
  
  // Set up the node stack with NodeToMatch as the only node on the stack.
  SmallVector<SDValue, 8> NodeStack;
  SDValue N = SDValue(NodeToMatch, 0);
  NodeStack.push_back(N);
  
  // Interpreter starts at opcode #0.
  unsigned MatcherIndex = 0;
  while (1) {
    assert(MatcherIndex < TableSize && "Invalid index");
    switch ((BuiltinOpcodes)MatcherTable[MatcherIndex++]) {
    case OPC_Emit: {
      errs() << "EMIT NODE\n";
      return 0;
    }
    case OPC_Push: {
      unsigned NumToSkip = MatcherTable[MatcherIndex++];
      MatchScope NewEntry;
      NewEntry.FailIndex = MatcherIndex+NumToSkip;
      NewEntry.NodeStackSize = NodeStack.size();
      NewEntry.NumRecordedNodes = RecordedNodes.size();
      MatchScopes.push_back(NewEntry);
      continue;
    }
    case OPC_Record:
      // Remember this node, it may end up being an operand in the pattern.
      RecordedNodes.push_back(N);
      continue;
        
    case OPC_MoveChild: {
      unsigned Child = MatcherTable[MatcherIndex++];
      if (Child >= N.getNumOperands())
        break;  // Match fails if out of range child #.
      N = N.getOperand(Child);
      NodeStack.push_back(N);
      continue;
    }
        
    case OPC_MoveParent:
      // Pop the current node off the NodeStack.
      NodeStack.pop_back();
      assert(!NodeStack.empty() && "Node stack imbalance!");
      N = NodeStack.back();  
      continue;
     
    case OPC_CheckSame: {
      // Accept if it is exactly the same as a previously recorded node.
      unsigned RecNo = MatcherTable[MatcherIndex++];
      assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
      if (N != RecordedNodes[RecNo]) break;
      continue;
    }
    case OPC_CheckPatternPredicate:
      if (!CheckPatternPredicate(MatcherTable[MatcherIndex++])) break;
      continue;
    case OPC_CheckPredicate:
      if (!CheckNodePredicate(N.getNode(), MatcherTable[MatcherIndex++])) break;
      continue;
    case OPC_CheckComplexPat: {
      unsigned PatNo = MatcherTable[MatcherIndex++];
      (void)PatNo;
      // FIXME: CHECK IT.
      continue;
    }
        
    case OPC_CheckOpcode:
      if (N->getOpcode() != MatcherTable[MatcherIndex++]) break;
      continue;
    case OPC_CheckType:
      if (N.getValueType() !=
          (MVT::SimpleValueType)MatcherTable[MatcherIndex++]) break;
      continue;
    case OPC_CheckCondCode:
      if (cast<CondCodeSDNode>(N)->get() !=
          (ISD::CondCode)MatcherTable[MatcherIndex++]) break;
      continue;
    case OPC_CheckValueType:
      if (cast<VTSDNode>(N)->getVT() !=
          (MVT::SimpleValueType)MatcherTable[MatcherIndex++]) break;
      continue;

    case OPC_CheckInteger1:
      if (CheckInteger(N, GetInt1(MatcherTable, MatcherIndex))) break;
      continue;
    case OPC_CheckInteger2:
      if (CheckInteger(N, GetInt2(MatcherTable, MatcherIndex))) break;
      continue;
    case OPC_CheckInteger4:
      if (CheckInteger(N, GetInt4(MatcherTable, MatcherIndex))) break;
      continue;
    case OPC_CheckInteger8:
      if (CheckInteger(N, GetInt8(MatcherTable, MatcherIndex))) break;
      continue;
        
    case OPC_CheckAndImm1:
      if (CheckAndImmediate(N, GetInt1(MatcherTable, MatcherIndex))) break;
      continue;
    case OPC_CheckAndImm2:
      if (CheckAndImmediate(N, GetInt2(MatcherTable, MatcherIndex))) break;
      continue;
    case OPC_CheckAndImm4:
      if (CheckAndImmediate(N, GetInt4(MatcherTable, MatcherIndex))) break;
      continue;
    case OPC_CheckAndImm8:
      if (CheckAndImmediate(N, GetInt8(MatcherTable, MatcherIndex))) break;
      continue;

    case OPC_CheckOrImm1:
      if (CheckOrImmediate(N, GetInt1(MatcherTable, MatcherIndex))) break;
      continue;
    case OPC_CheckOrImm2:
      if (CheckOrImmediate(N, GetInt2(MatcherTable, MatcherIndex))) break;
      continue;
    case OPC_CheckOrImm4:
      if (CheckOrImmediate(N, GetInt4(MatcherTable, MatcherIndex))) break;
      continue;
    case OPC_CheckOrImm8:
      if (CheckOrImmediate(N, GetInt8(MatcherTable, MatcherIndex))) break;
      continue;
        
    case OPC_CheckFoldableChainNode: {
      assert(!NodeStack.size() == 1 && "No parent node");
      // Verify that all intermediate nodes between the root and this one have
      // a single use.
      bool HasMultipleUses = false;
      for (unsigned i = 1, e = NodeStack.size()-1; i != e; ++i)
        if (!NodeStack[i].hasOneUse()) {
          HasMultipleUses = true;
          break;
        }
      if (HasMultipleUses) break;

      // Check to see that the target thinks this is profitable to fold and that
      // we can fold it without inducing cycles in the graph.
      if (!IsProfitableToFold(N, NodeStack[NodeStack.size()-2].getNode(),
                              NodeToMatch) ||
          !IsLegalToFold(N, NodeStack[NodeStack.size()-2].getNode(),
                         NodeToMatch))
        break;
      
      continue;
    }
    }
    
    // If the code reached this point, then the match failed pop out to the next
    // match scope.
    if (MatchScopes.empty()) {
      CannotYetSelect(NodeToMatch);
      return 0;
    }
    
    RecordedNodes.resize(MatchScopes.back().NumRecordedNodes);
    NodeStack.resize(MatchScopes.back().NodeStackSize);
    MatcherIndex = MatchScopes.back().FailIndex;
    MatchScopes.pop_back();
  }
}
    

#endif /* LLVM_CODEGEN_DAGISEL_HEADER_H */
