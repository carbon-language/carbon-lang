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

/// ChainNotReachable - Returns true if Chain does not reach Op.
static bool ChainNotReachable(SDNode *Chain, SDNode *Op) {
  if (Chain->getOpcode() == ISD::EntryToken)
    return true;
  if (Chain->getOpcode() == ISD::TokenFactor)
    return false;
  if (Chain->getNumOperands() > 0) {
    SDValue C0 = Chain->getOperand(0);
    if (C0.getValueType() == MVT::Other)
      return C0.getNode() != Op && ChainNotReachable(C0.getNode(), Op);
  }
  return true;
}

/// IsChainCompatible - Returns true if Chain is Op or Chain does not reach Op.
/// This is used to ensure that there are no nodes trapped between Chain, which
/// is the first chain node discovered in a pattern and Op, a later node, that
/// will not be selected into the pattern.
static bool IsChainCompatible(SDNode *Chain, SDNode *Op) {
  return Chain == Op || ChainNotReachable(Chain, Op);
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
  ISelPosition = SelectionDAG::allnodes_iterator(CurDAG->getRoot().getNode());
  ++ISelPosition;

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

    SDNode *ResNode = Select(Node);
    // If node should not be replaced, continue with the next one.
    if (ResNode == Node)
      continue;
    // Replace node.
    if (ResNode)
      ReplaceUses(Node, ResNode);

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

void EmitInteger(int64_t Val, MVT::SimpleValueType VT,
                 SmallVectorImpl<SDValue> &RecordedNodes) {
  RecordedNodes.push_back(CurDAG->getTargetConstant(Val, VT));
}

// These functions are marked always inline so that Idx doesn't get pinned to
// the stack.
ALWAYS_INLINE static int8_t
GetInt1(const unsigned char *MatcherTable, unsigned &Idx) {
  return MatcherTable[Idx++];
}

ALWAYS_INLINE static int16_t
GetInt2(const unsigned char *MatcherTable, unsigned &Idx) {
  int16_t Val = (uint8_t)GetInt1(MatcherTable, Idx);
  Val |= int16_t(GetInt1(MatcherTable, Idx)) << 8;
  return Val;
}

ALWAYS_INLINE static int32_t
GetInt4(const unsigned char *MatcherTable, unsigned &Idx) {
  int32_t Val = (uint16_t)GetInt2(MatcherTable, Idx);
  Val |= int32_t(GetInt2(MatcherTable, Idx)) << 16;
  return Val;
}

ALWAYS_INLINE static int64_t
GetInt8(const unsigned char *MatcherTable, unsigned &Idx) {
  int64_t Val = (uint32_t)GetInt4(MatcherTable, Idx);
  Val |= int64_t(GetInt4(MatcherTable, Idx)) << 32;
  return Val;
}

enum BuiltinOpcodes {
  OPC_Push, OPC_Push2,
  OPC_RecordNode,
  OPC_RecordMemRef,
  OPC_CaptureFlagInput,
  OPC_MoveChild,
  OPC_MoveParent,
  OPC_CheckSame,
  OPC_CheckPatternPredicate,
  OPC_CheckPredicate,
  OPC_CheckOpcode,
  OPC_CheckMultiOpcode,
  OPC_CheckType,
  OPC_CheckInteger1, OPC_CheckInteger2, OPC_CheckInteger4, OPC_CheckInteger8,
  OPC_CheckCondCode,
  OPC_CheckValueType,
  OPC_CheckComplexPat,
  OPC_CheckAndImm1, OPC_CheckAndImm2, OPC_CheckAndImm4, OPC_CheckAndImm8,
  OPC_CheckOrImm1, OPC_CheckOrImm2, OPC_CheckOrImm4, OPC_CheckOrImm8,
  OPC_CheckFoldableChainNode,
  OPC_CheckChainCompatible,
  
  OPC_EmitInteger1, OPC_EmitInteger2, OPC_EmitInteger4, OPC_EmitInteger8,
  OPC_EmitRegister,
  OPC_EmitConvertToTarget,
  OPC_EmitMergeInputChains,
  OPC_EmitCopyToReg,
  OPC_EmitNodeXForm,
  OPC_EmitNode,
  OPC_CompleteMatch
};

enum {
  OPFL_None      = 0,     // Node has no chain or flag input and isn't variadic.
  OPFL_Chain     = 1,     // Node has a chain input.
  OPFL_Flag      = 2,     // Node has a flag input.
  OPFL_MemRefs   = 4,     // Node gets accumulated MemRefs.
  OPFL_Variadic0 = 1<<3,  // Node is variadic, root has 0 fixed inputs.
  OPFL_Variadic1 = 2<<3,  // Node is variadic, root has 1 fixed inputs.
  OPFL_Variadic2 = 3<<3,  // Node is variadic, root has 2 fixed inputs.
  OPFL_Variadic3 = 4<<3,  // Node is variadic, root has 3 fixed inputs.
  OPFL_Variadic4 = 5<<3,  // Node is variadic, root has 4 fixed inputs.
  OPFL_Variadic5 = 6<<3,  // Node is variadic, root has 5 fixed inputs.
  OPFL_Variadic6 = 7<<3,  // Node is variadic, root has 6 fixed inputs.
  
  OPFL_VariadicInfo = OPFL_Variadic6
};

/// getNumFixedFromVariadicInfo - Transform an EmitNode flags word into the
/// number of fixed arity values that should be skipped when copying from the
/// root.
static inline int getNumFixedFromVariadicInfo(unsigned Flags) {
  return ((Flags&OPFL_VariadicInfo) >> 3)-1;
}

struct MatchScope {
  /// FailIndex - If this match fails, this is the index to continue with.
  unsigned FailIndex;
  
  /// NodeStackSize - The size of the node stack when the scope was formed.
  unsigned NodeStackSize;
  
  /// NumRecordedNodes - The number of recorded nodes when the scope was formed.
  unsigned NumRecordedNodes;
  
  /// NumMatchedMemRefs - The number of matched memref entries.
  unsigned NumMatchedMemRefs;
  
  /// InputChain/InputFlag - The current chain/flag 
  SDValue InputChain, InputFlag;

  /// HasChainNodesMatched - True if the ChainNodesMatched list is non-empty.
  bool HasChainNodesMatched;
};

SDNode *SelectCodeCommon(SDNode *NodeToMatch, const unsigned char *MatcherTable,
                         unsigned TableSize) {
  // FIXME: Should these even be selected?  Handle these cases in the caller?
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

  // Set up the node stack with NodeToMatch as the only node on the stack.
  SmallVector<SDValue, 8> NodeStack;
  SDValue N = SDValue(NodeToMatch, 0);
  NodeStack.push_back(N);

  // MatchScopes - Scopes used when matching, if a match failure happens, this
  // indicates where to continue checking.
  SmallVector<MatchScope, 8> MatchScopes;
  
  // RecordedNodes - This is the set of nodes that have been recorded by the
  // state machine.
  SmallVector<SDValue, 8> RecordedNodes;
  
  // MatchedMemRefs - This is the set of MemRef's we've seen in the input
  // pattern.
  SmallVector<MachineMemOperand*, 2> MatchedMemRefs;
  
  // These are the current input chain and flag for use when generating nodes.
  // Various Emit operations change these.  For example, emitting a copytoreg
  // uses and updates these.
  SDValue InputChain, InputFlag;
  
  // ChainNodesMatched - If a pattern matches nodes that have input/output
  // chains, the OPC_EmitMergeInputChains operation is emitted which indicates
  // which ones they are.  The result is captured into this list so that we can
  // update the chain results when the pattern is complete.
  SmallVector<SDNode*, 3> ChainNodesMatched;
  
  DEBUG(errs() << "ISEL: Starting pattern match on root node: ";
        NodeToMatch->dump(CurDAG);
        errs() << '\n');
  
  // Interpreter starts at opcode #0.
  unsigned MatcherIndex = 0;
  while (1) {
    assert(MatcherIndex < TableSize && "Invalid index");
    switch ((BuiltinOpcodes)MatcherTable[MatcherIndex++]) {
    case OPC_Push: {
      unsigned NumToSkip = MatcherTable[MatcherIndex++];
      MatchScope NewEntry;
      NewEntry.FailIndex = MatcherIndex+NumToSkip;
      NewEntry.NodeStackSize = NodeStack.size();
      NewEntry.NumRecordedNodes = RecordedNodes.size();
      NewEntry.NumMatchedMemRefs = MatchedMemRefs.size();
      NewEntry.InputChain = InputChain;
      NewEntry.InputFlag = InputFlag;
      NewEntry.HasChainNodesMatched = !ChainNodesMatched.empty();
      MatchScopes.push_back(NewEntry);
      continue;
    }
    case OPC_Push2: {
      unsigned NumToSkip = GetInt2(MatcherTable, MatcherIndex);
      MatchScope NewEntry;
      NewEntry.FailIndex = MatcherIndex+NumToSkip;
      NewEntry.NodeStackSize = NodeStack.size();
      NewEntry.NumRecordedNodes = RecordedNodes.size();
      NewEntry.NumMatchedMemRefs = MatchedMemRefs.size();
      NewEntry.InputChain = InputChain;
      NewEntry.InputFlag = InputFlag;
      NewEntry.HasChainNodesMatched = !ChainNodesMatched.empty();
      MatchScopes.push_back(NewEntry);
      continue;
    }
    case OPC_RecordNode:
      // Remember this node, it may end up being an operand in the pattern.
      RecordedNodes.push_back(N);
      continue;
    case OPC_RecordMemRef:
      MatchedMemRefs.push_back(cast<MemSDNode>(N)->getMemOperand());
      continue;
        
    case OPC_CaptureFlagInput:
      // If the current node has an input flag, capture it in InputFlag.
      if (N->getNumOperands() != 0 &&
          N->getOperand(N->getNumOperands()-1).getValueType() == MVT::Flag)
        InputFlag = N->getOperand(N->getNumOperands()-1);
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
    case OPC_CheckComplexPat:
      if (!CheckComplexPattern(NodeToMatch, N, 
                               MatcherTable[MatcherIndex++], RecordedNodes))
        break;
      continue;
    case OPC_CheckOpcode:
      if (N->getOpcode() != MatcherTable[MatcherIndex++]) break;
      continue;
        
    case OPC_CheckMultiOpcode: {
      unsigned NumOps = MatcherTable[MatcherIndex++];
      bool OpcodeEquals = false;
      for (unsigned i = 0; i != NumOps; ++i)
        OpcodeEquals |= N->getOpcode() == MatcherTable[MatcherIndex++];
      if (!OpcodeEquals) break;
      continue;
    }
        
    case OPC_CheckType: {
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      if (N.getValueType() != VT) {
        // Handle the case when VT is iPTR.
        if (VT != MVT::iPTR || N.getValueType() != TLI.getPointerTy())
          break;
      }
      continue;
    }
    case OPC_CheckCondCode:
      if (cast<CondCodeSDNode>(N)->get() !=
          (ISD::CondCode)MatcherTable[MatcherIndex++]) break;
      continue;
    case OPC_CheckValueType: {
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      if (cast<VTSDNode>(N)->getVT() != VT) {
        // Handle the case when VT is iPTR.
        if (VT != MVT::iPTR || cast<VTSDNode>(N)->getVT() != TLI.getPointerTy())
          break;
      }
      continue;
    }
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
      assert(NodeStack.size() != 1 && "No parent node");
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
    case OPC_CheckChainCompatible: {
      unsigned PrevNode = MatcherTable[MatcherIndex++];
      assert(PrevNode < RecordedNodes.size() && "Invalid CheckChainCompatible");
      SDValue PrevChainedNode = RecordedNodes[PrevNode];
      SDValue ThisChainedNode = RecordedNodes.back();
      
      // We have two nodes with chains, verify that their input chains are good.
      assert(PrevChainedNode.getOperand(0).getValueType() == MVT::Other &&
             ThisChainedNode.getOperand(0).getValueType() == MVT::Other &&
             "Invalid chained nodes");
      
      if (!IsChainCompatible(// Input chain of the previous node.
                             PrevChainedNode.getOperand(0).getNode(),
                             // Node with chain.
                             ThisChainedNode.getNode()))
        break;
      continue;
    }
        
    case OPC_EmitInteger1: {
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      EmitInteger(GetInt1(MatcherTable, MatcherIndex), VT, RecordedNodes);
      continue;
    }
    case OPC_EmitInteger2: {
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      EmitInteger(GetInt2(MatcherTable, MatcherIndex), VT, RecordedNodes);
      continue;
    }
    case OPC_EmitInteger4: {
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      EmitInteger(GetInt4(MatcherTable, MatcherIndex), VT, RecordedNodes);
      continue;
    }
    case OPC_EmitInteger8: {
      MVT::SimpleValueType VT =
       (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      EmitInteger(GetInt8(MatcherTable, MatcherIndex), VT, RecordedNodes);
      continue;
    }
        
    case OPC_EmitRegister: {
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      unsigned RegNo = MatcherTable[MatcherIndex++];
      RecordedNodes.push_back(CurDAG->getRegister(RegNo, VT));
      continue;
    }
        
    case OPC_EmitConvertToTarget:  {
      // Convert from IMM/FPIMM to target version.
      unsigned RecNo = MatcherTable[MatcherIndex++];
      assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
      SDValue Imm = RecordedNodes[RecNo];

      if (Imm->getOpcode() == ISD::Constant) {
        int64_t Val = cast<ConstantSDNode>(Imm)->getZExtValue();
        Imm = CurDAG->getTargetConstant(Val, Imm.getValueType());
      } else if (Imm->getOpcode() == ISD::ConstantFP) {
        const ConstantFP *Val=cast<ConstantFPSDNode>(Imm)->getConstantFPValue();
        Imm = CurDAG->getTargetConstantFP(*Val, Imm.getValueType());
      }
      
      RecordedNodes.push_back(Imm);
      continue;
    }
        
    case OPC_EmitMergeInputChains: {
      assert(InputChain.getNode() == 0 &&
             "EmitMergeInputChains should be the first chain producing node");
      // This node gets a list of nodes we matched in the input that have
      // chains.  We want to token factor all of the input chains to these nodes
      // together.  However, if any of the input chains is actually one of the
      // nodes matched in this pattern, then we have an intra-match reference.
      // Ignore these because the newly token factored chain should not refer to
      // the old nodes.
      unsigned NumChains = MatcherTable[MatcherIndex++];
      assert(NumChains != 0 && "Can't TF zero chains");
      
      // The common case here is that we have exactly one chain, which is really
      // cheap to handle, just do it.
      if (NumChains == 1) {
        unsigned RecNo = MatcherTable[MatcherIndex++];
        assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
        ChainNodesMatched.push_back(RecordedNodes[RecNo].getNode());
        InputChain = RecordedNodes[RecNo].getOperand(0);
        assert(InputChain.getValueType() == MVT::Other && "Not a chain");
        continue;
      }
      
      // Read all of the chained nodes.
      assert(ChainNodesMatched.empty() &&
             "Should only have one EmitMergeInputChains per match");
      for (unsigned i = 0; i != NumChains; ++i) {
        unsigned RecNo = MatcherTable[MatcherIndex++];
        assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
        ChainNodesMatched.push_back(RecordedNodes[RecNo].getNode());
      }

      // Walk all the chained nodes, adding the input chains if they are not in
      // ChainedNodes (and this, not in the matched pattern).  This is an N^2
      // algorithm, but # chains is usually 2 here, at most 3 for MSP430.
      SmallVector<SDValue, 3> InputChains;
      for (unsigned i = 0, e = ChainNodesMatched.size(); i != e; ++i) {
        SDValue InChain = ChainNodesMatched[i]->getOperand(0);
        assert(InChain.getValueType() == MVT::Other && "Not a chain");
        bool Invalid = false;
        for (unsigned j = 0; j != e; ++j)
          Invalid |= ChainNodesMatched[j] == InChain.getNode();
        if (!Invalid)
          InputChains.push_back(InChain);
      }

      SDValue Res;
      if (InputChains.size() == 1)
        InputChain = InputChains[0];
      else
        InputChain = CurDAG->getNode(ISD::TokenFactor,
                                     NodeToMatch->getDebugLoc(), MVT::Other,
                                     &InputChains[0], InputChains.size());
      continue;
    }
        
    case OPC_EmitCopyToReg: {
      unsigned RecNo = MatcherTable[MatcherIndex++];
      assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
      unsigned DestPhysReg = MatcherTable[MatcherIndex++];
      
      if (InputChain.getNode() == 0)
        InputChain = CurDAG->getEntryNode();
      
      InputChain = CurDAG->getCopyToReg(InputChain, NodeToMatch->getDebugLoc(),
                                        DestPhysReg, RecordedNodes[RecNo],
                                        InputFlag);
      
      InputFlag = InputChain.getValue(1);
      continue;
    }
        
    case OPC_EmitNodeXForm: {
      unsigned XFormNo = MatcherTable[MatcherIndex++];
      unsigned RecNo = MatcherTable[MatcherIndex++];
      assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
      RecordedNodes.push_back(RunSDNodeXForm(RecordedNodes[RecNo], XFormNo));
      continue;
    }
        
    case OPC_EmitNode: {
      uint16_t TargetOpc = GetInt2(MatcherTable, MatcherIndex);
      unsigned EmitNodeInfo = MatcherTable[MatcherIndex++];
      // Get the result VT list.
      unsigned NumVTs = MatcherTable[MatcherIndex++];
      assert(NumVTs != 0 && "Invalid node result");
      SmallVector<EVT, 4> VTs;
      for (unsigned i = 0; i != NumVTs; ++i) {
        MVT::SimpleValueType VT =
          (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
        if (VT == MVT::iPTR) VT = TLI.getPointerTy().SimpleTy;
        VTs.push_back(VT);
      }
      
      // FIXME: Use faster version for the common 'one VT' case?
      SDVTList VTList = CurDAG->getVTList(VTs.data(), VTs.size());

      // Get the operand list.
      unsigned NumOps = MatcherTable[MatcherIndex++];
      SmallVector<SDValue, 8> Ops;
      for (unsigned i = 0; i != NumOps; ++i) {
        unsigned RecNo = MatcherTable[MatcherIndex++];
        assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
        Ops.push_back(RecordedNodes[RecNo]);
      }
      
      // If there are variadic operands to add, handle them now.
      if (EmitNodeInfo & OPFL_VariadicInfo) {
        // Determine the start index to copy from.
        unsigned FirstOpToCopy = getNumFixedFromVariadicInfo(EmitNodeInfo);
        FirstOpToCopy += (EmitNodeInfo & OPFL_Chain) ? 1 : 0;
        assert(NodeToMatch->getNumOperands() >= FirstOpToCopy &&
               "Invalid variadic node");
        // Copy all of the variadic operands, not including a potential flag
        // input.
        for (unsigned i = FirstOpToCopy, e = NodeToMatch->getNumOperands();
             i != e; ++i) {
          SDValue V = NodeToMatch->getOperand(i);
          if (V.getValueType() == MVT::Flag) break;
          Ops.push_back(V);
        }
      }
      
      // If this has chain/flag inputs, add them.
      if (EmitNodeInfo & OPFL_Chain)
        Ops.push_back(InputChain);
      if ((EmitNodeInfo & OPFL_Flag) && InputFlag.getNode() != 0)
        Ops.push_back(InputFlag);
      
      // Create the node.
      MachineSDNode *Res = CurDAG->getMachineNode(TargetOpc,
                                                  NodeToMatch->getDebugLoc(),
                                                  VTList,
                                                  Ops.data(), Ops.size());
      // Add all the non-flag/non-chain results to the RecordedNodes list.
      for (unsigned i = 0, e = VTs.size(); i != e; ++i) {
        if (VTs[i] == MVT::Other || VTs[i] == MVT::Flag) break;
        RecordedNodes.push_back(SDValue(Res, i));
      }
      
      // If the node had chain/flag results, update our notion of the current
      // chain and flag.
      if (VTs.back() == MVT::Flag) {
        InputFlag = SDValue(Res, VTs.size()-1);
        if (EmitNodeInfo & OPFL_Chain)
          InputChain = SDValue(Res, VTs.size()-2);
      } else if (EmitNodeInfo & OPFL_Chain)
        InputChain = SDValue(Res, VTs.size()-1);

      // If the OPFL_MemRefs flag is set on this node, slap all of the
      // accumulated memrefs onto it.
      //
      // FIXME: This is vastly incorrect for patterns with multiple outputs
      // instructions that access memory and for ComplexPatterns that match
      // loads.
      if (EmitNodeInfo & OPFL_MemRefs) {
        MachineSDNode::mmo_iterator MemRefs =
          MF->allocateMemRefsArray(MatchedMemRefs.size());
        std::copy(MatchedMemRefs.begin(), MatchedMemRefs.end(), MemRefs);
        Res->setMemRefs(MemRefs, MemRefs + MatchedMemRefs.size());
      }
      
      DEBUG(errs() << "  Created node: "; Res->dump(CurDAG); errs() << "\n");
      continue;
    }
      
    case OPC_CompleteMatch: {
      // The match has been completed, and any new nodes (if any) have been
      // created.  Patch up references to the matched dag to use the newly
      // created nodes.
      unsigned NumResults = MatcherTable[MatcherIndex++];

      for (unsigned i = 0; i != NumResults; ++i) {
        unsigned ResSlot = MatcherTable[MatcherIndex++];
        assert(ResSlot < RecordedNodes.size() && "Invalid CheckSame");
        SDValue Res = RecordedNodes[ResSlot];
        
        // FIXME2: Eliminate this horrible hack by fixing the 'Gen' program
        // after (parallel) on input patterns are removed.  This would also
        // allow us to stop encoding #results in OPC_CompleteMatch's table
        // entry.
        if (NodeToMatch->getNumValues() <= i ||
            NodeToMatch->getValueType(i) == MVT::Other ||
            NodeToMatch->getValueType(i) == MVT::Flag)
          break;
        assert((NodeToMatch->getValueType(i) == Res.getValueType() ||
                NodeToMatch->getValueType(i) == MVT::iPTR ||
                Res.getValueType() == MVT::iPTR ||
                NodeToMatch->getValueType(i).getSizeInBits() ==
                    Res.getValueType().getSizeInBits()) &&
               "invalid replacement");
        ReplaceUses(SDValue(NodeToMatch, i), Res);
      }
      
      // Now that all the normal results are replaced, we replace the chain and
      // flag results if present.
      if (!ChainNodesMatched.empty()) {
        assert(InputChain.getNode() != 0 &&
               "Matched input chains but didn't produce a chain");
        // Loop over all of the nodes we matched that produced a chain result.
        // Replace all the chain results with the final chain we ended up with.
        for (unsigned i = 0, e = ChainNodesMatched.size(); i != e; ++i) {
          SDNode *ChainNode = ChainNodesMatched[i];
          SDValue ChainVal = SDValue(ChainNode, ChainNode->getNumValues()-1);
          if (ChainVal.getValueType() == MVT::Flag)
            ChainVal = ChainVal.getValue(ChainVal->getNumValues()-2);
          assert(ChainVal.getValueType() == MVT::Other && "Not a chain?");
          ReplaceUses(ChainVal, InputChain);
        }
      }
      // If the root node produces a flag, make sure to replace its flag
      // result with the resultant flag.
      if (NodeToMatch->getValueType(NodeToMatch->getNumValues()-1) ==
            MVT::Flag)
        ReplaceUses(SDValue(NodeToMatch, NodeToMatch->getNumValues()-1),
                    InputFlag);
      
      assert(NodeToMatch->use_empty() &&
             "Didn't replace all uses of the node?");
      
      DEBUG(errs() << "ISEL: Match complete!\n");
      
      // FIXME: We just return here, which interacts correctly with SelectRoot
      // above.  We should fix this to not return an SDNode* anymore.
      return 0;
    }
    }
    
    // If the code reached this point, then the match failed pop out to the next
    // match scope.
    if (MatchScopes.empty()) {
      CannotYetSelect(NodeToMatch);
      return 0;
    }
    
    const MatchScope &LastScope = MatchScopes.back();
    RecordedNodes.resize(LastScope.NumRecordedNodes);
    NodeStack.resize(LastScope.NodeStackSize);
    N = NodeStack.back();

    DEBUG(errs() << "  Match failed at index " << MatcherIndex
                 << " continuing at " << LastScope.FailIndex << "\n");
    
    if (LastScope.NumMatchedMemRefs != MatchedMemRefs.size())
      MatchedMemRefs.resize(LastScope.NumMatchedMemRefs);
    MatcherIndex = LastScope.FailIndex;
    
    InputChain = LastScope.InputChain;
    InputFlag = LastScope.InputFlag;
    if (!LastScope.HasChainNodesMatched)
      ChainNodesMatched.clear();
    
    MatchScopes.pop_back();
  }
}
    

#endif /* LLVM_CODEGEN_DAGISEL_HEADER_H */
