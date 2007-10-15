//===-- LegalizeDAGTypes.cpp - Implement SelectionDAG::LegalizeTypes ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SelectionDAG::LegalizeTypes method.  It transforms
// an arbitrary well-formed SelectionDAG to only consist of legal types.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "legalize-types"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
/// DAGTypeLegalizer - This takes an arbitrary SelectionDAG as input and
/// hacks on it until the target machine can handle it.  This involves
/// eliminating value sizes the machine cannot handle (promoting small sizes to
/// large sizes or splitting up large values into small values) as well as
/// eliminating operations the machine cannot handle.
///
/// This code also does a small amount of optimization and recognition of idioms
/// as part of its processing.  For example, if a target does not support a
/// 'setcc' instruction efficiently, but does support 'brcc' instruction, this
/// will attempt merge setcc and brc instructions into brcc's.
///
namespace {
class VISIBILITY_HIDDEN DAGTypeLegalizer {
  TargetLowering &TLI;
  SelectionDAG &DAG;
  
  // NodeIDFlags - This pass uses the NodeID on the SDNodes to hold information
  // about the state of the node.  The enum has all the values.
  enum NodeIDFlags {
    /// ReadyToProcess - All operands have been processed, so this node is ready
    /// to be handled.
    ReadyToProcess = 0,
    
    /// NewNode - This is a new node that was created in the process of
    /// legalizing some other node.
    NewNode = -1,
    
    /// Processed - This is a node that has already been processed.
    Processed = -2
    
    // 1+ - This is a node which has this many unlegalized operands.
  };
  
  enum LegalizeAction {
    Legal,      // The target natively supports this operation.
    Promote,    // This operation should be executed in a larger type.
    Expand      // Try to expand this to other ops, otherwise use a libcall.
  };
  
  /// ValueTypeActions - This is a bitvector that contains two bits for each
  /// simple value type, where the two bits correspond to the LegalizeAction
  /// enum.  This can be queried with "getTypeAction(VT)".
  TargetLowering::ValueTypeActionImpl ValueTypeActions;
  
  /// getTypeAction - Return how we should legalize values of this type, either
  /// it is already legal or we need to expand it into multiple registers of
  /// smaller integer type, or we need to promote it to a larger type.
  LegalizeAction getTypeAction(MVT::ValueType VT) const {
    return (LegalizeAction)ValueTypeActions.getTypeAction(VT);
  }
  
  /// isTypeLegal - Return true if this type is legal on this target.
  ///
  bool isTypeLegal(MVT::ValueType VT) const {
    return getTypeAction(VT) == Legal;
  }
  
  SDOperand getIntPtrConstant(uint64_t Val) {
    return DAG.getConstant(Val, TLI.getPointerTy());
  }
  
  /// PromotedNodes - For nodes that are below legal width, and that have more
  /// than one use, this map indicates what promoted value to use.
  DenseMap<SDOperand, SDOperand> PromotedNodes;
  
  /// ExpandedNodes - For nodes that need to be expanded this map indicates
  /// which operands are the expanded version of the input.
  DenseMap<SDOperand, std::pair<SDOperand, SDOperand> > ExpandedNodes;
  
  /// Worklist - This defines a worklist of nodes to process.  In order to be
  /// pushed onto this worklist, all operands of a node must have already been
  /// processed.
  SmallVector<SDNode*, 128> Worklist;
  
public:
  DAGTypeLegalizer(SelectionDAG &dag)
    : TLI(dag.getTargetLoweringInfo()), DAG(dag),
    ValueTypeActions(TLI.getValueTypeActions()) {
    assert(MVT::LAST_VALUETYPE <= 32 &&
           "Too many value types for ValueTypeActions to hold!");
  }      
  
  void run();
  
private:
  void MarkNewNodes(SDNode *N);
  
  void ReplaceLegalValueWith(SDOperand From, SDOperand To);
  
  SDOperand GetPromotedOp(SDOperand Op) {
    Op = PromotedNodes[Op];
    assert(Op.Val && "Operand wasn't promoted?");
    return Op;
  }    
  void SetPromotedOp(SDOperand Op, SDOperand Result);

  void GetExpandedOp(SDOperand Op, SDOperand &Lo, SDOperand &Hi);
  void SetExpandedOp(SDOperand Op, SDOperand Lo, SDOperand Hi);
    
  // Result Promotion.
  void PromoteResult(SDNode *N, unsigned ResNo);
  SDOperand PromoteResult_UNDEF(SDNode *N);
  SDOperand PromoteResult_Constant(SDNode *N);
  SDOperand PromoteResult_TRUNCATE(SDNode *N);
  SDOperand PromoteResult_INT_EXTEND(SDNode *N);
  SDOperand PromoteResult_FP_ROUND(SDNode *N);
  SDOperand PromoteResult_SETCC(SDNode *N);
  SDOperand PromoteResult_LOAD(LoadSDNode *N);
  SDOperand PromoteResult_SimpleIntBinOp(SDNode *N);
  
  // Result Expansion.
  void ExpandResult(SDNode *N, unsigned ResNo);
  void ExpandResult_UNDEF      (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_Constant   (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_BUILD_PAIR (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_ANY_EXTEND (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_ZERO_EXTEND(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_SIGN_EXTEND(SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_LOAD       (LoadSDNode *N, SDOperand &Lo, SDOperand &Hi);

  void ExpandResult_Logical    (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_ADDSUB     (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_SELECT     (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_SELECT_CC  (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_MUL        (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  void ExpandResult_Shift      (SDNode *N, SDOperand &Lo, SDOperand &Hi);
  
  void ExpandShiftByConstant(SDNode *N, unsigned Amt, 
                             SDOperand &Lo, SDOperand &Hi);
  bool ExpandShiftWithKnownAmountBit(SDNode *N, SDOperand &Lo, SDOperand &Hi);

  // Operand Promotion.
  bool PromoteOperand(SDNode *N, unsigned OperandNo);
  SDOperand PromoteOperand_ANY_EXTEND(SDNode *N);
  SDOperand PromoteOperand_ZERO_EXTEND(SDNode *N);
  SDOperand PromoteOperand_SIGN_EXTEND(SDNode *N);
  SDOperand PromoteOperand_FP_EXTEND(SDNode *N);
  SDOperand PromoteOperand_FP_ROUND(SDNode *N);
  SDOperand PromoteOperand_SELECT(SDNode *N, unsigned OpNo);
  SDOperand PromoteOperand_BRCOND(SDNode *N, unsigned OpNo);
  SDOperand PromoteOperand_STORE(StoreSDNode *N, unsigned OpNo);
  
  // Operand Expansion.
  bool ExpandOperand(SDNode *N, unsigned OperandNo);
  SDOperand ExpandOperand_TRUNCATE(SDNode *N);
  SDOperand ExpandOperand_EXTRACT_ELEMENT(SDNode *N);
  SDOperand ExpandOperand_SETCC(SDNode *N);
  SDOperand ExpandOperand_STORE(StoreSDNode *N, unsigned OpNo);

  void ExpandSetCCOperands(SDOperand &NewLHS, SDOperand &NewRHS,
                           ISD::CondCode &CCCode);
};
}  // end anonymous namespace



/// run - This is the main entry point for the type legalizer.  This does a
/// top-down traversal of the dag, legalizing types as it goes.
void DAGTypeLegalizer::run() {
  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted, and tracking any
  // changes of the root.
  HandleSDNode Dummy(DAG.getRoot());

  // The root of the dag may dangle to deleted nodes until the type legalizer is
  // done.  Set it to null to avoid confusion.
  DAG.setRoot(SDOperand());
  
  // Walk all nodes in the graph, assigning them a NodeID of 'ReadyToProcess'
  // (and remembering them) if they are leaves and assigning 'NewNode' if
  // non-leaves.
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I) {
    if (I->getNumOperands() == 0) {
      I->setNodeId(ReadyToProcess);
      Worklist.push_back(I);
    } else {
      I->setNodeId(NewNode);
    }
  }
  
  // Now that we have a set of nodes to process, handle them all.
  while (!Worklist.empty()) {
    SDNode *N = Worklist.back();
    Worklist.pop_back();
    assert(N->getNodeId() == ReadyToProcess &&
           "Node should be ready if on worklist!");
    
    // Scan the values produced by the node, checking to see if any result
    // types are illegal.
    unsigned i = 0;
    unsigned NumResults = N->getNumValues();
    do {
      LegalizeAction Action = getTypeAction(N->getValueType(i));
      if (Action == Promote) {
        PromoteResult(N, i);
        goto NodeDone;
      } else if (Action == Expand) {
        ExpandResult(N, i);
        goto NodeDone;
      } else {
        assert(Action == Legal && "Unknown action!");
      }
    } while (++i < NumResults);
    
    // Scan the operand list for the node, handling any nodes with operands that
    // are illegal.
    {
    unsigned NumOperands = N->getNumOperands();
    bool NeedsRevisit = false;
    for (i = 0; i != NumOperands; ++i) {
      LegalizeAction Action = getTypeAction(N->getOperand(i).getValueType());
      if (Action == Promote) {
        NeedsRevisit = PromoteOperand(N, i);
        break;
      } else if (Action == Expand) {
        NeedsRevisit = ExpandOperand(N, i);
        break;
      } else {
        assert(Action == Legal && "Unknown action!");
      }
    }

    // If the node needs revisitation, don't add all users to the worklist etc.
    if (NeedsRevisit)
      continue;
    
    if (i == NumOperands)
      DEBUG(cerr << "Legally typed node: "; N->dump(&DAG); cerr << "\n");
    }
NodeDone:

    // If we reach here, the node was processed, potentially creating new nodes.
    // Mark it as processed and add its users to the worklist as appropriate.
    N->setNodeId(Processed);
    
    for (SDNode::use_iterator UI = N->use_begin(), E = N->use_end();
         UI != E; ++UI) {
      SDNode *User = *UI;
      int NodeID = User->getNodeId();
      assert(NodeID != ReadyToProcess && NodeID != Processed &&
             "Invalid node id for user of unprocessed node!");
      
      // This node has two options: it can either be a new node or its Node ID
      // may be a count of the number of operands it has that are not ready.
      if (NodeID > 0) {
        User->setNodeId(NodeID-1);
        
        // If this was the last use it was waiting on, add it to the ready list.
        if (NodeID-1 == ReadyToProcess)
          Worklist.push_back(User);
        continue;
      }
      
      // Otherwise, this node is new: this is the first operand of it that
      // became ready.  Its new NodeID is the number of operands it has minus 1
      // (as this node is now processed).
      assert(NodeID == NewNode && "Unknown node ID!");
      User->setNodeId(User->getNumOperands()-1);
      
      // If the node only has a single operand, it is now ready.
      if (User->getNumOperands() == 1)
        Worklist.push_back(User);
    }
  }
  
  // If the root changed (e.g. it was a dead load, update the root).
  DAG.setRoot(Dummy.getValue());

  //DAG.viewGraph();

  // Remove dead nodes.  This is important to do for cleanliness but also before
  // the checking loop below.  Implicit folding by the DAG.getNode operators can
  // cause unreachable nodes to be around with their flags set to new.
  DAG.RemoveDeadNodes();

  // In a debug build, scan all the nodes to make sure we found them all.  This
  // ensures that there are no cycles and that everything got processed.
#ifndef NDEBUG
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I) {
    if (I->getNodeId() == Processed)
      continue;
    cerr << "Unprocessed node: ";
    I->dump(&DAG); cerr << "\n";

    if (I->getNodeId() == NewNode)
      cerr << "New node not 'noticed'?\n";
    else if (I->getNodeId() > 0)
      cerr << "Operand not processed?\n";
    else if (I->getNodeId() == ReadyToProcess)
      cerr << "Not added to worklist?\n";
    abort();
  }
#endif
}

/// MarkNewNodes - The specified node is the root of a subtree of potentially
/// new nodes.  Add the correct NodeId to mark it.
void DAGTypeLegalizer::MarkNewNodes(SDNode *N) {
  // If this was an existing node that is already done, we're done.
  if (N->getNodeId() != NewNode)
    return;

  // Okay, we know that this node is new.  Recursively walk all of its operands
  // to see if they are new also.  The depth of this walk is bounded by the size
  // of the new tree that was constructed (usually 2-3 nodes), so we don't worry
  // about revisitation of nodes.
  //
  // As we walk the operands, keep track of the number of nodes that are
  // processed.  If non-zero, this will become the new nodeid of this node.
  unsigned NumProcessed = 0;
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    int OpId = N->getOperand(i).Val->getNodeId();
    if (OpId == NewNode)
      MarkNewNodes(N->getOperand(i).Val);
    else if (OpId == Processed)
      ++NumProcessed;
  }
  
  N->setNodeId(N->getNumOperands()-NumProcessed);
  if (N->getNodeId() == ReadyToProcess)
    Worklist.push_back(N);
}

/// ReplaceLegalValueWith - The specified value with a legal type was legalized
/// to the specified other value.  If they are different, update the DAG and
/// NodeIDs replacing any uses of From to use To instead.
void DAGTypeLegalizer::ReplaceLegalValueWith(SDOperand From, SDOperand To) {
  if (From == To) return;
  
  // If expansion produced new nodes, make sure they are properly marked.
  if (To.Val->getNodeId() == NewNode)
    MarkNewNodes(To.Val);
  
  // Anything that used the old node should now use the new one.  Note that this
  // can potentially cause recursive merging.
  DAG.ReplaceAllUsesOfValueWith(From, To);
  
  // Since we just made an unstructured update to the DAG, which could wreak
  // general havoc on anything that once used N and now uses Res, walk all users
  // of the result, updating their flags.
  for (SDNode::use_iterator I = To.Val->use_begin(), E = To.Val->use_end();
       I != E; ++I) {
    SDNode *User = *I;
    // If the node isn't already processed or in the worklist, mark it as new,
    // then use MarkNewNodes to recompute its ID.
    int NodeId = User->getNodeId();
    if (NodeId != ReadyToProcess && NodeId != Processed) {
      User->setNodeId(NewNode);
      MarkNewNodes(User);
    }
  }
}

void DAGTypeLegalizer::SetPromotedOp(SDOperand Op, SDOperand Result) {
  if (Result.Val->getNodeId() == NewNode) 
    MarkNewNodes(Result.Val);

  SDOperand &OpEntry = PromotedNodes[Op];
  assert(OpEntry.Val == 0 && "Node is already promoted!");
  OpEntry = Result;
}


void DAGTypeLegalizer::GetExpandedOp(SDOperand Op, SDOperand &Lo, 
                                     SDOperand &Hi) {
  std::pair<SDOperand, SDOperand> &Entry = ExpandedNodes[Op];
  assert(Entry.first.Val && "Operand isn't expanded");
  Lo = Entry.first;
  Hi = Entry.second;
}

void DAGTypeLegalizer::SetExpandedOp(SDOperand Op, SDOperand Lo, 
                                     SDOperand Hi) {
  // Remember that this is the result of the node.
  std::pair<SDOperand, SDOperand> &Entry = ExpandedNodes[Op];
  assert(Entry.first.Val == 0 && "Node already expanded");
  Entry.first = Lo;
  Entry.second = Hi;
  
  // Lo/Hi may have been newly allocated, if so, add nodeid's as relevant.
  if (Lo.Val->getNodeId() == NewNode) 
    MarkNewNodes(Lo.Val);
  if (Hi.Val->getNodeId() == NewNode) 
    MarkNewNodes(Hi.Val);
}

//===----------------------------------------------------------------------===//
//  Result Promotion
//===----------------------------------------------------------------------===//

/// PromoteResult - This method is called when a result of a node is found to be
/// in need of promotion to a larger type.  At this point, the node may also
/// have invalid operands or may have other results that need expansion, we just
/// know that (at least) one result needs promotion.
void DAGTypeLegalizer::PromoteResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Promote node result: "; N->dump(&DAG); cerr << "\n");
  SDOperand Result = SDOperand();
  
  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "PromoteResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to promote this operator!");
    abort();
  case ISD::UNDEF:    Result = PromoteResult_UNDEF(N); break;
  case ISD::Constant: Result = PromoteResult_Constant(N); break;
    
  case ISD::TRUNCATE:    Result = PromoteResult_TRUNCATE(N); break;
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:  Result = PromoteResult_INT_EXTEND(N); break;
  case ISD::FP_ROUND:    Result = PromoteResult_FP_ROUND(N); break;
    
  case ISD::SETCC:    Result = PromoteResult_SETCC(N); break;
  case ISD::LOAD:     Result = PromoteResult_LOAD(cast<LoadSDNode>(N)); break;
    
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:      Result = PromoteResult_SimpleIntBinOp(N); break;
  }      
  
  // If Result is null, the sub-method took care of registering the result.
  if (Result.Val)
    SetPromotedOp(SDOperand(N, ResNo), Result);
}

SDOperand DAGTypeLegalizer::PromoteResult_UNDEF(SDNode *N) {
  return DAG.getNode(ISD::UNDEF, TLI.getTypeToTransformTo(N->getValueType(0)));
}

SDOperand DAGTypeLegalizer::PromoteResult_Constant(SDNode *N) {
  MVT::ValueType VT = N->getValueType(0);
  // Zero extend things like i1, sign extend everything else.  It shouldn't
  // matter in theory which one we pick, but this tends to give better code?
  unsigned Opc = VT != MVT::i1 ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
  SDOperand Result = DAG.getNode(Opc, TLI.getTypeToTransformTo(VT),
                                 SDOperand(N, 0));
  assert(isa<ConstantSDNode>(Result) && "Didn't constant fold ext?");
  return Result;
}

SDOperand DAGTypeLegalizer::PromoteResult_TRUNCATE(SDNode *N) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  switch (getTypeAction(N->getOperand(0).getValueType())) {
  default: assert(0 && "Unknown type action!");
  case Legal: {
    SDOperand Res = N->getOperand(0);
    assert(Res.getValueType() >= NVT && "Truncation doesn't make sense!");
    if (Res.getValueType() > NVT)             // Truncate to NVT instead of VT
      return DAG.getNode(ISD::TRUNCATE, NVT, Res);
    return Res;
  }
  case Promote:
    // The truncation is not required, because we don't guarantee anything
    // about high bits anyway.
    return GetPromotedOp(N->getOperand(0));
  case Expand:
    // Truncate the low part of the expanded value to the result type
    SDOperand Lo, Hi;
    GetExpandedOp(N->getOperand(0), Lo, Hi);
    return DAG.getNode(ISD::TRUNCATE, NVT, Lo);
  }
}
SDOperand DAGTypeLegalizer::PromoteResult_INT_EXTEND(SDNode *N) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  switch (getTypeAction(N->getOperand(0).getValueType())) {
  default: assert(0 && "BUG: Smaller reg should have been promoted!");
  case Legal:
    // Input is legal?  Just do extend all the way to the larger type.
    return DAG.getNode(N->getOpcode(), NVT, N->getOperand(0));
  case Promote:
    // Get promoted operand if it is smaller.
    SDOperand Res = GetPromotedOp(N->getOperand(0));
    // The high bits are not guaranteed to be anything.  Insert an extend.
    if (N->getOpcode() == ISD::SIGN_EXTEND)
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, NVT, Res,
                         DAG.getValueType(N->getOperand(0).getValueType()));
    if (N->getOpcode() == ISD::ZERO_EXTEND)
      return DAG.getZeroExtendInReg(Res, N->getOperand(0).getValueType());
    assert(N->getOpcode() == ISD::ANY_EXTEND && "Unknown integer extension!");
    return Res;
  }
}

SDOperand DAGTypeLegalizer::PromoteResult_FP_ROUND(SDNode *N) {
  // NOTE: Assumes input is legal.
  return DAG.getNode(ISD::FP_ROUND_INREG, N->getOperand(0).getValueType(),
                     N->getOperand(0), DAG.getValueType(N->getValueType(0)));
}


SDOperand DAGTypeLegalizer::PromoteResult_SETCC(SDNode *N) {
  assert(isTypeLegal(TLI.getSetCCResultTy()) && "SetCC type is not legal??");
  return DAG.getNode(ISD::SETCC, TLI.getSetCCResultTy(), N->getOperand(0),
                     N->getOperand(1), N->getOperand(2));
}

SDOperand DAGTypeLegalizer::PromoteResult_LOAD(LoadSDNode *N) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  ISD::LoadExtType ExtType =
    ISD::isNON_EXTLoad(N) ? ISD::EXTLOAD : N->getExtensionType();
  SDOperand Res = DAG.getExtLoad(ExtType, NVT, N->getChain(), N->getBasePtr(),
                                 N->getSrcValue(), N->getSrcValueOffset(),
                                 N->getLoadedVT(), N->isVolatile(),
                                 N->getAlignment());
  
  // Legalized the chain result, switching anything that used the old chain to
  // use the new one.
  ReplaceLegalValueWith(SDOperand(N, 1), Res.getValue(1));
  return Res;
}

SDOperand DAGTypeLegalizer::PromoteResult_SimpleIntBinOp(SDNode *N) {
  // The input may have strange things in the top bits of the registers, but
  // these operations don't care.  They may have weird bits going out, but
  // that too is okay if they are integer operations.
  SDOperand LHS = GetPromotedOp(N->getOperand(0));
  SDOperand RHS = GetPromotedOp(N->getOperand(1));
  return DAG.getNode(N->getOpcode(), LHS.getValueType(), LHS, RHS);
}

//===----------------------------------------------------------------------===//
//  Result Expansion
//===----------------------------------------------------------------------===//

/// ExpandResult - This method is called when the specified result of the
/// specified node is found to need expansion.  At this point, the node may also
/// have invalid operands or may have other results that need promotion, we just
/// know that (at least) one result needs expansion.
void DAGTypeLegalizer::ExpandResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Expand node result: "; N->dump(&DAG); cerr << "\n");
  SDOperand Lo, Hi;
  Lo = Hi = SDOperand();
  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "ExpandResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to expand this operator!");
    abort();
      
  case ISD::UNDEF:       ExpandResult_UNDEF(N, Lo, Hi); break;
  case ISD::Constant:    ExpandResult_Constant(N, Lo, Hi); break;
  case ISD::BUILD_PAIR:  ExpandResult_BUILD_PAIR(N, Lo, Hi); break;
  case ISD::ANY_EXTEND:  ExpandResult_ANY_EXTEND(N, Lo, Hi); break;
  case ISD::ZERO_EXTEND: ExpandResult_ZERO_EXTEND(N, Lo, Hi); break;
  case ISD::SIGN_EXTEND: ExpandResult_SIGN_EXTEND(N, Lo, Hi); break;
  case ISD::LOAD:        ExpandResult_LOAD(cast<LoadSDNode>(N), Lo, Hi); break;
    
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:         ExpandResult_Logical(N, Lo, Hi); break;
  case ISD::ADD:
  case ISD::SUB:         ExpandResult_ADDSUB(N, Lo, Hi); break;
  case ISD::SELECT:      ExpandResult_SELECT(N, Lo, Hi); break;
  case ISD::SELECT_CC:   ExpandResult_SELECT_CC(N, Lo, Hi); break;
  case ISD::MUL:         ExpandResult_MUL(N, Lo, Hi); break;
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:         ExpandResult_Shift(N, Lo, Hi); break;

  }
  
  // If Lo/Hi is null, the sub-method took care of registering results etc.
  if (Lo.Val)
    SetExpandedOp(SDOperand(N, ResNo), Lo, Hi);
}

void DAGTypeLegalizer::ExpandResult_UNDEF(SDNode *N,
                                          SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  Lo = Hi = DAG.getNode(ISD::UNDEF, NVT);
}

void DAGTypeLegalizer::ExpandResult_Constant(SDNode *N,
                                             SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  uint64_t Cst = cast<ConstantSDNode>(N)->getValue();
  Lo = DAG.getConstant(Cst, NVT);
  Hi = DAG.getConstant(Cst >> MVT::getSizeInBits(NVT), NVT);
}

void DAGTypeLegalizer::ExpandResult_BUILD_PAIR(SDNode *N,
                                               SDOperand &Lo, SDOperand &Hi) {
  // Return the operands.
  Lo = N->getOperand(0);
  Hi = N->getOperand(1);
}

void DAGTypeLegalizer::ExpandResult_ANY_EXTEND(SDNode *N, 
                                               SDOperand &Lo, SDOperand &Hi) {
  
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  // The low part is any extension of the input (which degenerates to a copy).
  Lo = DAG.getNode(ISD::ANY_EXTEND, NVT, N->getOperand(0));
  Hi = DAG.getNode(ISD::UNDEF, NVT);   // The high part is undefined.
}

void DAGTypeLegalizer::ExpandResult_ZERO_EXTEND(SDNode *N,
                                                SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  // The low part is zero extension of the input (which degenerates to a copy).
  Lo = DAG.getNode(ISD::ZERO_EXTEND, NVT, N->getOperand(0));
  Hi = DAG.getConstant(0, NVT);   // The high part is just a zero.
}

void DAGTypeLegalizer::ExpandResult_SIGN_EXTEND(SDNode *N,
                                                SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  // The low part is sign extension of the input (which degenerates to a copy).
  Lo = DAG.getNode(ISD::SIGN_EXTEND, NVT, N->getOperand(0));

  // The high part is obtained by SRA'ing all but one of the bits of low part.
  unsigned LoSize = MVT::getSizeInBits(NVT);
  Hi = DAG.getNode(ISD::SRA, NVT, Lo,
                   DAG.getConstant(LoSize-1, TLI.getShiftAmountTy()));
}


void DAGTypeLegalizer::ExpandResult_LOAD(LoadSDNode *N,
                                         SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);
  SDOperand Ch  = N->getChain();    // Legalize the chain.
  SDOperand Ptr = N->getBasePtr();  // Legalize the pointer.
  ISD::LoadExtType ExtType = N->getExtensionType();
  int SVOffset = N->getSrcValueOffset();
  unsigned Alignment = N->getAlignment();
  bool isVolatile = N->isVolatile();
  
  if (ExtType == ISD::NON_EXTLOAD) {
    Lo = DAG.getLoad(NVT, Ch, Ptr, N->getSrcValue(), SVOffset,
                     isVolatile, Alignment);
    if (VT == MVT::f32 || VT == MVT::f64) {
      assert(0 && "FIXME: softfp should use promotion!");
#if 0
      // f32->i32 or f64->i64 one to one expansion.
      // Remember that we legalized the chain.
      AddLegalizedOperand(SDOperand(Node, 1), LegalizeOp(Lo.getValue(1)));
      // Recursively expand the new load.
      if (getTypeAction(NVT) == Expand)
        ExpandOp(Lo, Lo, Hi);
      break;
#endif
    }
    
    // Increment the pointer to the other half.
    unsigned IncrementSize = MVT::getSizeInBits(Lo.getValueType())/8;
    Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                      getIntPtrConstant(IncrementSize));
    Hi = DAG.getLoad(NVT, Ch, Ptr, N->getSrcValue(), SVOffset+IncrementSize,
                     isVolatile, std::max(Alignment, IncrementSize));
    
    // Build a factor node to remember that this load is independent of the
    // other one.
    Ch = DAG.getNode(ISD::TokenFactor, MVT::Other, Lo.getValue(1),
                     Hi.getValue(1));

    // Handle endianness of the load.
    if (!TLI.isLittleEndian())
      std::swap(Lo, Hi);
  } else {
    MVT::ValueType EVT = N->getLoadedVT();
    
    if (VT == MVT::f64 && EVT == MVT::f32) {
      assert(0 && "FIXME: softfp should use promotion!");
#if 0
      // f64 = EXTLOAD f32 should expand to LOAD, FP_EXTEND
      SDOperand Load = DAG.getLoad(EVT, Ch, Ptr, N->getSrcValue(),
                                   SVOffset, isVolatile, Alignment);
      // Remember that we legalized the chain.
      AddLegalizedOperand(SDOperand(Node, 1), LegalizeOp(Load.getValue(1)));
      ExpandOp(DAG.getNode(ISD::FP_EXTEND, VT, Load), Lo, Hi);
      break;
#endif
    }
    
    if (EVT == NVT)
      Lo = DAG.getLoad(NVT, Ch, Ptr, N->getSrcValue(),
                       SVOffset, isVolatile, Alignment);
    else
      Lo = DAG.getExtLoad(ExtType, NVT, Ch, Ptr, N->getSrcValue(),
                          SVOffset, EVT, isVolatile,
                          Alignment);
    // Remember the chain.
    Ch = Lo.getValue(1);
    
    if (ExtType == ISD::SEXTLOAD) {
      // The high part is obtained by SRA'ing all but one of the bits of the
      // lo part.
      unsigned LoSize = MVT::getSizeInBits(Lo.getValueType());
      Hi = DAG.getNode(ISD::SRA, NVT, Lo,
                       DAG.getConstant(LoSize-1, TLI.getShiftAmountTy()));
    } else if (ExtType == ISD::ZEXTLOAD) {
      // The high part is just a zero.
      Hi = DAG.getConstant(0, NVT);
    } else {
      assert(ExtType == ISD::EXTLOAD && "Unknown extload!");
      // The high part is undefined.
      Hi = DAG.getNode(ISD::UNDEF, NVT);
    }
  }
  
  // Legalized the chain result, switching anything that used the old chain to
  // use the new one.
  ReplaceLegalValueWith(SDOperand(N, 1), Ch);
}  


void DAGTypeLegalizer::ExpandResult_Logical(SDNode *N,
                                            SDOperand &Lo, SDOperand &Hi) {
  SDOperand LL, LH, RL, RH;
  GetExpandedOp(N->getOperand(0), LL, LH);
  GetExpandedOp(N->getOperand(1), RL, RH);
  Lo = DAG.getNode(N->getOpcode(), LL.getValueType(), LL, RL);
  Hi = DAG.getNode(N->getOpcode(), LL.getValueType(), LH, RH);
}

void DAGTypeLegalizer::ExpandResult_SELECT(SDNode *N,
                                           SDOperand &Lo, SDOperand &Hi) {
  SDOperand LL, LH, RL, RH;
  GetExpandedOp(N->getOperand(1), LL, LH);
  GetExpandedOp(N->getOperand(2), RL, RH);
  Lo = DAG.getNode(ISD::SELECT, LL.getValueType(), N->getOperand(0), LL, RL);
  
  assert(N->getOperand(0).getValueType() != MVT::f32 &&
         "FIXME: softfp shouldn't use expand!");
  Hi = DAG.getNode(ISD::SELECT, LL.getValueType(), N->getOperand(0), LH, RH);
}

void DAGTypeLegalizer::ExpandResult_SELECT_CC(SDNode *N,
                                              SDOperand &Lo, SDOperand &Hi) {
  SDOperand LL, LH, RL, RH;
  GetExpandedOp(N->getOperand(2), LL, LH);
  GetExpandedOp(N->getOperand(3), RL, RH);
  Lo = DAG.getNode(ISD::SELECT_CC, LL.getValueType(), N->getOperand(0), 
                   N->getOperand(1), LL, RL, N->getOperand(4));
  
  assert(N->getOperand(0).getValueType() != MVT::f32 &&
         "FIXME: softfp shouldn't use expand!");
  Hi = DAG.getNode(ISD::SELECT_CC, LL.getValueType(), N->getOperand(0), 
                   N->getOperand(1), LH, RH, N->getOperand(4));
}

void DAGTypeLegalizer::ExpandResult_ADDSUB(SDNode *N,
                                           SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType VT = N->getValueType(0);
  
  // If the target wants to custom expand this, let them.
  if (TLI.getOperationAction(N->getOpcode(), VT) ==
      TargetLowering::Custom) {
    SDOperand Op = TLI.LowerOperation(SDOperand(N, 0), DAG);
    // FIXME: Do a replace all uses with here!
    assert(0 && "Custom not impl yet!");
    if (Op.Val) {
#if 0
      ExpandOp(Op, Lo, Hi);
#endif
      return;
    }
  }
  
  // Expand the subcomponents.
  SDOperand LHSL, LHSH, RHSL, RHSH;
  GetExpandedOp(N->getOperand(0), LHSL, LHSH);
  GetExpandedOp(N->getOperand(1), RHSL, RHSH);
  SDVTList VTList = DAG.getVTList(LHSL.getValueType(), MVT::Flag);
  SDOperand LoOps[2], HiOps[3];
  LoOps[0] = LHSL;
  LoOps[1] = RHSL;
  HiOps[0] = LHSH;
  HiOps[1] = RHSH;
  if (N->getOpcode() == ISD::ADD) {
    Lo = DAG.getNode(ISD::ADDC, VTList, LoOps, 2);
    HiOps[2] = Lo.getValue(1);
    Hi = DAG.getNode(ISD::ADDE, VTList, HiOps, 3);
  } else {
    Lo = DAG.getNode(ISD::SUBC, VTList, LoOps, 2);
    HiOps[2] = Lo.getValue(1);
    Hi = DAG.getNode(ISD::SUBE, VTList, HiOps, 3);
  }
}


void DAGTypeLegalizer::ExpandResult_MUL(SDNode *N,
                                        SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);
  
  // If the target wants to custom expand this, let them.
  if (TLI.getOperationAction(ISD::MUL, VT) == TargetLowering::Custom) {
    SDOperand New = TLI.LowerOperation(SDOperand(N, 0), DAG);
    if (New.Val) {
      // FIXME: Do a replace all uses with here!
      assert(0 && "Custom not impl yet!");
#if 0
      ExpandOp(New, Lo, Hi);
#endif
      return;
    }
  }
  
  bool HasMULHS = TLI.isOperationLegal(ISD::MULHS, NVT);
  bool HasMULHU = TLI.isOperationLegal(ISD::MULHU, NVT);
  bool HasSMUL_LOHI = TLI.isOperationLegal(ISD::SMUL_LOHI, NVT);
  bool HasUMUL_LOHI = TLI.isOperationLegal(ISD::UMUL_LOHI, NVT);
  if (HasMULHU || HasMULHS || HasUMUL_LOHI || HasSMUL_LOHI) {
    SDOperand LL, LH, RL, RH;
    GetExpandedOp(N->getOperand(0), LL, LH);
    GetExpandedOp(N->getOperand(1), RL, RH);
    unsigned BitSize = MVT::getSizeInBits(RH.getValueType());
    unsigned LHSSB = DAG.ComputeNumSignBits(N->getOperand(0));
    unsigned RHSSB = DAG.ComputeNumSignBits(N->getOperand(1));
    
    // FIXME: generalize this to handle other bit sizes
    if (LHSSB == 32 && RHSSB == 32 &&
        DAG.MaskedValueIsZero(N->getOperand(0), 0xFFFFFFFF00000000ULL) &&
        DAG.MaskedValueIsZero(N->getOperand(1), 0xFFFFFFFF00000000ULL)) {
      // The inputs are both zero-extended.
      if (HasUMUL_LOHI) {
        // We can emit a umul_lohi.
        Lo = DAG.getNode(ISD::UMUL_LOHI, DAG.getVTList(NVT, NVT), LL, RL);
        Hi = SDOperand(Lo.Val, 1);
        return;
      }
      if (HasMULHU) {
        // We can emit a mulhu+mul.
        Lo = DAG.getNode(ISD::MUL, NVT, LL, RL);
        Hi = DAG.getNode(ISD::MULHU, NVT, LL, RL);
        return;
      }
    }
    if (LHSSB > BitSize && RHSSB > BitSize) {
      // The input values are both sign-extended.
      if (HasSMUL_LOHI) {
        // We can emit a smul_lohi.
        Lo = DAG.getNode(ISD::SMUL_LOHI, DAG.getVTList(NVT, NVT), LL, RL);
        Hi = SDOperand(Lo.Val, 1);
        return;
      }
      if (HasMULHS) {
        // We can emit a mulhs+mul.
        Lo = DAG.getNode(ISD::MUL, NVT, LL, RL);
        Hi = DAG.getNode(ISD::MULHS, NVT, LL, RL);
        return;
      }
    }
    if (HasUMUL_LOHI) {
      // Lo,Hi = umul LHS, RHS.
      SDOperand UMulLOHI = DAG.getNode(ISD::UMUL_LOHI,
                                       DAG.getVTList(NVT, NVT), LL, RL);
      Lo = UMulLOHI;
      Hi = UMulLOHI.getValue(1);
      RH = DAG.getNode(ISD::MUL, NVT, LL, RH);
      LH = DAG.getNode(ISD::MUL, NVT, LH, RL);
      Hi = DAG.getNode(ISD::ADD, NVT, Hi, RH);
      Hi = DAG.getNode(ISD::ADD, NVT, Hi, LH);
      return;
    }
  }
  
  abort();
#if 0 // FIXME!
  // If nothing else, we can make a libcall.
  Lo = ExpandLibCall(TLI.getLibcallName(RTLIB::MUL_I64), N,
                     false/*sign irrelevant*/, Hi);
#endif
}  


void DAGTypeLegalizer::ExpandResult_Shift(SDNode *N,
                                          SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType VT = N->getValueType(0);
  
  // If the target wants custom lowering, do so.
  if (TLI.getOperationAction(N->getOpcode(), VT) == TargetLowering::Custom) {
    SDOperand Op = TLI.LowerOperation(SDOperand(N, 0), DAG);
    if (Op.Val) {
      // Now that the custom expander is done, expand the result, which is
      // still VT.
      // FIXME: Do a replace all uses with here!
      abort();
#if 0
      ExpandOp(Op, Lo, Hi);
#endif
      return;
    }
  }
  
  // If we can emit an efficient shift operation, do so now.  Check to see if 
  // the RHS is a constant.
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N->getOperand(1)))
    return ExpandShiftByConstant(N, CN->getValue(), Lo, Hi);

  // If we can determine that the high bit of the shift is zero or one, even if
  // the low bits are variable, emit this shift in an optimized form.
  if (ExpandShiftWithKnownAmountBit(N, Lo, Hi))
    return;
  
  // If this target supports shift_PARTS, use it.  First, map to the _PARTS opc.
  unsigned PartsOpc;
  if (N->getOpcode() == ISD::SHL)
    PartsOpc = ISD::SHL_PARTS;
  else if (N->getOpcode() == ISD::SRL)
    PartsOpc = ISD::SRL_PARTS;
  else {
    assert(N->getOpcode() == ISD::SRA && "Unknown shift!");
    PartsOpc = ISD::SRA_PARTS;
  }
  
  // Next check to see if the target supports this SHL_PARTS operation or if it
  // will custom expand it.
  MVT::ValueType NVT = TLI.getTypeToTransformTo(VT);
  TargetLowering::LegalizeAction Action = TLI.getOperationAction(PartsOpc, NVT);
  if ((Action == TargetLowering::Legal && TLI.isTypeLegal(NVT)) ||
      Action == TargetLowering::Custom) {
    // Expand the subcomponents.
    SDOperand LHSL, LHSH;
    GetExpandedOp(N->getOperand(0), LHSL, LHSH);
    
    SDOperand Ops[] = { LHSL, LHSH, N->getOperand(1) };
    MVT::ValueType VT = LHSL.getValueType();
    Lo = DAG.getNode(PartsOpc, DAG.getNodeValueTypes(VT, VT), 2, Ops, 3);
    Hi = Lo.getValue(1);
    return;
  }
  
  abort();
#if 0 // FIXME!
  // Otherwise, emit a libcall.
  unsigned RuntimeCode = ; // SRL -> SRL_I64 etc.
  bool Signed = ;
  Lo = ExpandLibCall(TLI.getLibcallName(RTLIB::SRL_I64), N,
                     false/*lshr is unsigned*/, Hi);
#endif
}  


/// ExpandShiftByConstant - N is a shift by a value that needs to be expanded,
/// and the shift amount is a constant 'Amt'.  Expand the operation.
void DAGTypeLegalizer::ExpandShiftByConstant(SDNode *N, unsigned Amt, 
                                             SDOperand &Lo, SDOperand &Hi) {
  // Expand the incoming operand to be shifted, so that we have its parts
  SDOperand InL, InH;
  GetExpandedOp(N->getOperand(0), InL, InH);
  
  MVT::ValueType NVT = InL.getValueType();
  unsigned VTBits = MVT::getSizeInBits(N->getValueType(0));
  unsigned NVTBits = MVT::getSizeInBits(NVT);
  MVT::ValueType ShTy = N->getOperand(1).getValueType();

  if (N->getOpcode() == ISD::SHL) {
    if (Amt > VTBits) {
      Lo = Hi = DAG.getConstant(0, NVT);
    } else if (Amt > NVTBits) {
      Lo = DAG.getConstant(0, NVT);
      Hi = DAG.getNode(ISD::SHL, NVT, InL, DAG.getConstant(Amt-NVTBits,ShTy));
    } else if (Amt == NVTBits) {
      Lo = DAG.getConstant(0, NVT);
      Hi = InL;
    } else {
      Lo = DAG.getNode(ISD::SHL, NVT, InL, DAG.getConstant(Amt, ShTy));
      Hi = DAG.getNode(ISD::OR, NVT,
                       DAG.getNode(ISD::SHL, NVT, InH,
                                   DAG.getConstant(Amt, ShTy)),
                       DAG.getNode(ISD::SRL, NVT, InL,
                                   DAG.getConstant(NVTBits-Amt, ShTy)));
    }
    return;
  }
  
  if (N->getOpcode() == ISD::SRL) {
    if (Amt > VTBits) {
      Lo = DAG.getConstant(0, NVT);
      Hi = DAG.getConstant(0, NVT);
    } else if (Amt > NVTBits) {
      Lo = DAG.getNode(ISD::SRL, NVT, InH, DAG.getConstant(Amt-NVTBits,ShTy));
      Hi = DAG.getConstant(0, NVT);
    } else if (Amt == NVTBits) {
      Lo = InH;
      Hi = DAG.getConstant(0, NVT);
    } else {
      Lo = DAG.getNode(ISD::OR, NVT,
                       DAG.getNode(ISD::SRL, NVT, InL,
                                   DAG.getConstant(Amt, ShTy)),
                       DAG.getNode(ISD::SHL, NVT, InH,
                                   DAG.getConstant(NVTBits-Amt, ShTy)));
      Hi = DAG.getNode(ISD::SRL, NVT, InH, DAG.getConstant(Amt, ShTy));
    }
    return;
  }
  
  assert(N->getOpcode() == ISD::SRA && "Unknown shift!");
  if (Amt > VTBits) {
    Hi = Lo = DAG.getNode(ISD::SRA, NVT, InH,
                          DAG.getConstant(NVTBits-1, ShTy));
  } else if (Amt > NVTBits) {
    Lo = DAG.getNode(ISD::SRA, NVT, InH,
                     DAG.getConstant(Amt-NVTBits, ShTy));
    Hi = DAG.getNode(ISD::SRA, NVT, InH,
                     DAG.getConstant(NVTBits-1, ShTy));
  } else if (Amt == NVTBits) {
    Lo = InH;
    Hi = DAG.getNode(ISD::SRA, NVT, InH,
                     DAG.getConstant(NVTBits-1, ShTy));
  } else {
    Lo = DAG.getNode(ISD::OR, NVT,
                     DAG.getNode(ISD::SRL, NVT, InL,
                                 DAG.getConstant(Amt, ShTy)),
                     DAG.getNode(ISD::SHL, NVT, InH,
                                 DAG.getConstant(NVTBits-Amt, ShTy)));
    Hi = DAG.getNode(ISD::SRA, NVT, InH, DAG.getConstant(Amt, ShTy));
  }
}

/// ExpandShiftWithKnownAmountBit - Try to determine whether we can simplify
/// this shift based on knowledge of the high bit of the shift amount.  If we
/// can tell this, we know that it is >= 32 or < 32, without knowing the actual
/// shift amount.
bool DAGTypeLegalizer::
ExpandShiftWithKnownAmountBit(SDNode *N, SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  unsigned NVTBits = MVT::getSizeInBits(NVT);

  uint64_t HighBitMask = NVTBits, KnownZero, KnownOne;
  DAG.ComputeMaskedBits(N->getOperand(1), HighBitMask, KnownZero, KnownOne);
  
  // If we don't know anything about the high bit, exit.
  if (((KnownZero|KnownOne) & HighBitMask) == 0)
    return false;

  // Get the incoming operand to be shifted.
  SDOperand InL, InH;
  GetExpandedOp(N->getOperand(0), InL, InH);
  SDOperand Amt = N->getOperand(1);

  // If we know that the high bit of the shift amount is one, then we can do
  // this as a couple of simple shifts.
  if (KnownOne & HighBitMask) {
    // Mask out the high bit, which we know is set.
    Amt = DAG.getNode(ISD::AND, Amt.getValueType(), Amt,
                      DAG.getConstant(NVTBits-1, Amt.getValueType()));
    
    switch (N->getOpcode()) {
    default: assert(0 && "Unknown shift");
    case ISD::SHL:
      Lo = DAG.getConstant(0, NVT);              // Low part is zero.
      Hi = DAG.getNode(ISD::SHL, NVT, InL, Amt); // High part from Lo part.
      return true;
    case ISD::SRL:
      Hi = DAG.getConstant(0, NVT);              // Hi part is zero.
      Lo = DAG.getNode(ISD::SRL, NVT, InH, Amt); // Lo part from Hi part.
      return true;
    case ISD::SRA:
      Hi = DAG.getNode(ISD::SRA, NVT, InH,       // Sign extend high part.
                       DAG.getConstant(NVTBits-1, Amt.getValueType()));
      Lo = DAG.getNode(ISD::SRA, NVT, InH, Amt); // Lo part from Hi part.
      return true;
    }
  }
  
  // If we know that the high bit of the shift amount is zero, then we can do
  // this as a couple of simple shifts.
  assert((KnownZero & HighBitMask) && "Bad mask computation above");

  // Compute 32-amt.
  SDOperand Amt2 = DAG.getNode(ISD::SUB, Amt.getValueType(),
                               DAG.getConstant(NVTBits, Amt.getValueType()),
                               Amt);
  unsigned Op1, Op2;
  switch (N->getOpcode()) {
  default: assert(0 && "Unknown shift");
  case ISD::SHL:  Op1 = ISD::SHL; Op2 = ISD::SRL; break;
  case ISD::SRL:
  case ISD::SRA:  Op1 = ISD::SRL; Op2 = ISD::SHL; break;
  }
    
  Lo = DAG.getNode(N->getOpcode(), NVT, InL, Amt);
  Hi = DAG.getNode(ISD::OR, NVT,
                   DAG.getNode(Op1, NVT, InH, Amt),
                   DAG.getNode(Op2, NVT, InL, Amt2));
  return true;
}

//===----------------------------------------------------------------------===//
//  Operand Promotion
//===----------------------------------------------------------------------===//

/// PromoteOperand - This method is called when the specified operand of the
/// specified node is found to need promotion.  At this point, all of the result
/// types of the node are known to be legal, but other operands of the node may
/// need promotion or expansion as well as the specified one.
bool DAGTypeLegalizer::PromoteOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Promote node operand: "; N->dump(&DAG); cerr << "\n");
  SDOperand Res;
  switch (N->getOpcode()) {
    default:
#ifndef NDEBUG
    cerr << "PromoteOperand Op #" << OpNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to promote this operator's operand!");
    abort();
    
  case ISD::ANY_EXTEND:  Res = PromoteOperand_ANY_EXTEND(N); break;
  case ISD::ZERO_EXTEND: Res = PromoteOperand_ZERO_EXTEND(N); break;
  case ISD::SIGN_EXTEND: Res = PromoteOperand_SIGN_EXTEND(N); break;
  case ISD::FP_EXTEND:   Res = PromoteOperand_FP_EXTEND(N); break;
  case ISD::FP_ROUND:    Res = PromoteOperand_FP_ROUND(N); break;
    
  case ISD::SELECT:      Res = PromoteOperand_SELECT(N, OpNo); break;
  case ISD::BRCOND:      Res = PromoteOperand_BRCOND(N, OpNo); break;
  case ISD::STORE:       Res = PromoteOperand_STORE(cast<StoreSDNode>(N),
                                                    OpNo); break;
  }
  
  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.Val) return false;
  // If the result is N, the sub-method updated N in place.
  if (Res.Val == N) {
    // Mark N as new and remark N and its operands.  This allows us to correctly
    // revisit N if it needs another step of promotion and allows us to visit
    // any new operands to N.
    N->setNodeId(NewNode);
    MarkNewNodes(N);
    return true;
  }
  
  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");
  
  ReplaceLegalValueWith(SDOperand(N, 0), Res);
  return false;
}

SDOperand DAGTypeLegalizer::PromoteOperand_ANY_EXTEND(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  return DAG.getNode(ISD::ANY_EXTEND, N->getValueType(0), Op);
}

SDOperand DAGTypeLegalizer::PromoteOperand_ZERO_EXTEND(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  Op = DAG.getNode(ISD::ANY_EXTEND, N->getValueType(0), Op);
  return DAG.getZeroExtendInReg(Op, N->getOperand(0).getValueType());
}
SDOperand DAGTypeLegalizer::PromoteOperand_SIGN_EXTEND(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  Op = DAG.getNode(ISD::ANY_EXTEND, N->getValueType(0), Op);
  return DAG.getNode(ISD::SIGN_EXTEND_INREG, Op.getValueType(),
                     Op, DAG.getValueType(N->getOperand(0).getValueType()));
}

SDOperand DAGTypeLegalizer::PromoteOperand_FP_EXTEND(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  return DAG.getNode(ISD::FP_EXTEND, N->getValueType(0), Op);
}
SDOperand DAGTypeLegalizer::PromoteOperand_FP_ROUND(SDNode *N) {
  SDOperand Op = GetPromotedOp(N->getOperand(0));
  return DAG.getNode(ISD::FP_ROUND, N->getValueType(0), Op);
}


SDOperand DAGTypeLegalizer::PromoteOperand_SELECT(SDNode *N, unsigned OpNo) {
  assert(OpNo == 0 && "Only know how to promote condition");
  SDOperand Cond = GetPromotedOp(N->getOperand(0));  // Promote the condition.

  // The top bits of the promoted condition are not necessarily zero, ensure
  // that the value is properly zero extended.
  if (!DAG.MaskedValueIsZero(Cond, 
                             MVT::getIntVTBitMask(Cond.getValueType())^1)) {
    Cond = DAG.getZeroExtendInReg(Cond, MVT::i1);
    MarkNewNodes(Cond.Val); 
  }

  // The chain (Op#0) and basic block destination (Op#2) are always legal types.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), Cond, N->getOperand(1),
                                N->getOperand(2));
}


SDOperand DAGTypeLegalizer::PromoteOperand_BRCOND(SDNode *N, unsigned OpNo) {
  assert(OpNo == 1 && "only know how to promote condition");
  SDOperand Cond = GetPromotedOp(N->getOperand(1));  // Promote the condition.
  
  // The top bits of the promoted condition are not necessarily zero, ensure
  // that the value is properly zero extended.
  if (!DAG.MaskedValueIsZero(Cond, 
                             MVT::getIntVTBitMask(Cond.getValueType())^1)) {
    Cond = DAG.getZeroExtendInReg(Cond, MVT::i1);
    MarkNewNodes(Cond.Val); 
  }
  
  // The chain (Op#0) and basic block destination (Op#2) are always legal types.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), N->getOperand(0), Cond,
                                N->getOperand(2));
}

SDOperand DAGTypeLegalizer::PromoteOperand_STORE(StoreSDNode *N, unsigned OpNo){
  SDOperand Ch = N->getChain(), Ptr = N->getBasePtr();
  int SVOffset = N->getSrcValueOffset();
  unsigned Alignment = N->getAlignment();
  bool isVolatile = N->isVolatile();
  
  SDOperand Val = GetPromotedOp(N->getValue());  // Get promoted value.

  assert(!N->isTruncatingStore() && "Cannot promote this store operand!");
  
  // Truncate the value and store the result.
  return DAG.getTruncStore(Ch, Val, Ptr, N->getSrcValue(),
                           SVOffset, N->getStoredVT(),
                           isVolatile, Alignment);
}


//===----------------------------------------------------------------------===//
//  Operand Expansion
//===----------------------------------------------------------------------===//

/// ExpandOperand - This method is called when the specified operand of the
/// specified node is found to need expansion.  At this point, all of the result
/// types of the node are known to be legal, but other operands of the node may
/// need promotion or expansion as well as the specified one.
bool DAGTypeLegalizer::ExpandOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Expand node operand: "; N->dump(&DAG); cerr << "\n");
  SDOperand Res;
  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "ExpandOperand Op #" << OpNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to expand this operator's operand!");
    abort();
    
  case ISD::TRUNCATE:        Res = ExpandOperand_TRUNCATE(N); break;
  case ISD::EXTRACT_ELEMENT: Res = ExpandOperand_EXTRACT_ELEMENT(N); break;
  case ISD::SETCC:           Res = ExpandOperand_SETCC(N); break;

  case ISD::STORE: Res = ExpandOperand_STORE(cast<StoreSDNode>(N), OpNo); break;
  }
  
  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.Val) return false;
  // If the result is N, the sub-method updated N in place.  Check to see if any
  // operands are new, and if so, mark them.
  if (Res.Val == N) {
    // Mark N as new and remark N and its operands.  This allows us to correctly
    // revisit N if it needs another step of promotion and allows us to visit
    // any new operands to N.
    N->setNodeId(NewNode);
    MarkNewNodes(N);
    return true;
  }

  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");
  
  ReplaceLegalValueWith(SDOperand(N, 0), Res);
  return false;
}

SDOperand DAGTypeLegalizer::ExpandOperand_TRUNCATE(SDNode *N) {
  SDOperand InL, InH;
  GetExpandedOp(N->getOperand(0), InL, InH);
  // Just truncate the low part of the source.
  return DAG.getNode(ISD::TRUNCATE, N->getValueType(0), InL);
}

SDOperand DAGTypeLegalizer::ExpandOperand_EXTRACT_ELEMENT(SDNode *N) {
  SDOperand Lo, Hi;
  GetExpandedOp(N->getOperand(0), Lo, Hi);
  return cast<ConstantSDNode>(N->getOperand(1))->getValue() ? Hi : Lo;
}

SDOperand DAGTypeLegalizer::ExpandOperand_SETCC(SDNode *N) {
  SDOperand NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(2))->get();
  ExpandSetCCOperands(NewLHS, NewRHS, CCCode);
  
  // If ExpandSetCCOperands returned a scalar, use it.
  if (NewRHS.Val == 0) return NewLHS;

  // Otherwise, update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDOperand(N, 0), NewLHS, NewRHS,
                                DAG.getCondCode(CCCode));
}

/// ExpandSetCCOperands - Expand the operands to a comparison.  This code is
/// shared among BR_CC, SELECT_CC, and SETCC handlers.
void DAGTypeLegalizer::ExpandSetCCOperands(SDOperand &NewLHS, SDOperand &NewRHS,
                                           ISD::CondCode &CCCode) {
  SDOperand LHSLo, LHSHi, RHSLo, RHSHi;
  GetExpandedOp(NewLHS, LHSLo, LHSHi);
  GetExpandedOp(NewRHS, RHSLo, RHSHi);
  
  MVT::ValueType VT = NewLHS.getValueType();
  if (VT == MVT::f32 || VT == MVT::f64) {
    assert(0 && "FIXME: softfp not implemented yet! should be promote not exp");
  }
  
  if (VT == MVT::ppcf128) {
    // FIXME:  This generated code sucks.  We want to generate
    //         FCMP crN, hi1, hi2
    //         BNE crN, L:
    //         FCMP crN, lo1, lo2
    // The following can be improved, but not that much.
    SDOperand Tmp1, Tmp2, Tmp3;
    Tmp1 = DAG.getSetCC(TLI.getSetCCResultTy(), LHSHi, RHSHi, ISD::SETEQ);
    Tmp2 = DAG.getSetCC(TLI.getSetCCResultTy(), LHSLo, RHSLo, CCCode);
    Tmp3 = DAG.getNode(ISD::AND, Tmp1.getValueType(), Tmp1, Tmp2);
    Tmp1 = DAG.getSetCC(TLI.getSetCCResultTy(), LHSHi, RHSHi, ISD::SETNE);
    Tmp2 = DAG.getSetCC(TLI.getSetCCResultTy(), LHSHi, RHSHi, CCCode);
    Tmp1 = DAG.getNode(ISD::AND, Tmp1.getValueType(), Tmp1, Tmp2);
    NewLHS = DAG.getNode(ISD::OR, Tmp1.getValueType(), Tmp1, Tmp3);
    NewRHS = SDOperand();   // LHS is the result, not a compare.
    return;
  }
  
  
  if (CCCode == ISD::SETEQ || CCCode == ISD::SETNE) {
    if (RHSLo == RHSHi)
      if (ConstantSDNode *RHSCST = dyn_cast<ConstantSDNode>(RHSLo))
        if (RHSCST->isAllOnesValue()) {
          // Equality comparison to -1.
          NewLHS = DAG.getNode(ISD::AND, LHSLo.getValueType(), LHSLo, LHSHi);
          NewRHS = RHSLo;
          return;
        }
          
    NewLHS = DAG.getNode(ISD::XOR, LHSLo.getValueType(), LHSLo, RHSLo);
    NewRHS = DAG.getNode(ISD::XOR, LHSLo.getValueType(), LHSHi, RHSHi);
    NewLHS = DAG.getNode(ISD::OR, NewLHS.getValueType(), NewLHS, NewRHS);
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    return;
  }
  
  // If this is a comparison of the sign bit, just look at the top part.
  // X > -1,  x < 0
  if (ConstantSDNode *CST = dyn_cast<ConstantSDNode>(NewRHS))
    if ((CCCode == ISD::SETLT && CST->getValue() == 0) ||   // X < 0
        (CCCode == ISD::SETGT && CST->isAllOnesValue())) {  // X > -1
      NewLHS = LHSHi;
      NewRHS = RHSHi;
      return;
    }
      
  // FIXME: This generated code sucks.
  ISD::CondCode LowCC;
  switch (CCCode) {
  default: assert(0 && "Unknown integer setcc!");
  case ISD::SETLT:
  case ISD::SETULT: LowCC = ISD::SETULT; break;
  case ISD::SETGT:
  case ISD::SETUGT: LowCC = ISD::SETUGT; break;
  case ISD::SETLE:
  case ISD::SETULE: LowCC = ISD::SETULE; break;
  case ISD::SETGE:
  case ISD::SETUGE: LowCC = ISD::SETUGE; break;
  }
  
  // Tmp1 = lo(op1) < lo(op2)   // Always unsigned comparison
  // Tmp2 = hi(op1) < hi(op2)   // Signedness depends on operands
  // dest = hi(op1) == hi(op2) ? Tmp1 : Tmp2;
  
  // NOTE: on targets without efficient SELECT of bools, we can always use
  // this identity: (B1 ? B2 : B3) --> (B1 & B2)|(!B1&B3)
  TargetLowering::DAGCombinerInfo DagCombineInfo(DAG, false, true, NULL);
  SDOperand Tmp1, Tmp2;
  Tmp1 = TLI.SimplifySetCC(TLI.getSetCCResultTy(), LHSLo, RHSLo, LowCC,
                           false, DagCombineInfo);
  if (!Tmp1.Val)
    Tmp1 = DAG.getSetCC(TLI.getSetCCResultTy(), LHSLo, RHSLo, LowCC);
  Tmp2 = TLI.SimplifySetCC(TLI.getSetCCResultTy(), LHSHi, RHSHi,
                           CCCode, false, DagCombineInfo);
  if (!Tmp2.Val)
    Tmp2 = DAG.getNode(ISD::SETCC, TLI.getSetCCResultTy(), LHSHi, RHSHi,
                       DAG.getCondCode(CCCode));
  
  ConstantSDNode *Tmp1C = dyn_cast<ConstantSDNode>(Tmp1.Val);
  ConstantSDNode *Tmp2C = dyn_cast<ConstantSDNode>(Tmp2.Val);
  if ((Tmp1C && Tmp1C->getValue() == 0) ||
      (Tmp2C && Tmp2C->getValue() == 0 &&
       (CCCode == ISD::SETLE || CCCode == ISD::SETGE ||
        CCCode == ISD::SETUGE || CCCode == ISD::SETULE)) ||
      (Tmp2C && Tmp2C->getValue() == 1 &&
       (CCCode == ISD::SETLT || CCCode == ISD::SETGT ||
        CCCode == ISD::SETUGT || CCCode == ISD::SETULT))) {
    // low part is known false, returns high part.
    // For LE / GE, if high part is known false, ignore the low part.
    // For LT / GT, if high part is known true, ignore the low part.
    NewLHS = Tmp2;
    NewRHS = SDOperand();
    return;
  }
  
  NewLHS = TLI.SimplifySetCC(TLI.getSetCCResultTy(), LHSHi, RHSHi,
                             ISD::SETEQ, false, DagCombineInfo);
  if (!NewLHS.Val)
    NewLHS = DAG.getSetCC(TLI.getSetCCResultTy(), LHSHi, RHSHi, ISD::SETEQ);
  NewLHS = DAG.getNode(ISD::SELECT, Tmp1.getValueType(),
                       NewLHS, Tmp1, Tmp2);
  NewRHS = SDOperand();
}


SDOperand DAGTypeLegalizer::ExpandOperand_STORE(StoreSDNode *N, unsigned OpNo) {
  assert(OpNo == 1 && "Can only expand the stored value so far");
  assert(!N->isTruncatingStore() && "Can't expand truncstore!");

  unsigned IncrementSize = 0;
  SDOperand Lo, Hi;
  
  // If this is a vector type, then we have to calculate the increment as
  // the product of the element size in bytes, and the number of elements
  // in the high half of the vector.
  if (MVT::isVector(N->getValue().getValueType())) {
    assert(0 && "Vectors not supported yet");
#if 0
    SDNode *InVal = ST->getValue().Val;
    unsigned NumElems = MVT::getVectorNumElements(InVal->getValueType(0));
    MVT::ValueType EVT = MVT::getVectorElementType(InVal->getValueType(0));
    
    // Figure out if there is a simple type corresponding to this Vector
    // type.  If so, convert to the vector type.
    MVT::ValueType TVT = MVT::getVectorType(EVT, NumElems);
    if (TLI.isTypeLegal(TVT)) {
      // Turn this into a normal store of the vector type.
      Tmp3 = LegalizeOp(Node->getOperand(1));
      Result = DAG.getStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(),
                            SVOffset, isVolatile, Alignment);
      Result = LegalizeOp(Result);
      break;
    } else if (NumElems == 1) {
      // Turn this into a normal store of the scalar type.
      Tmp3 = ScalarizeVectorOp(Node->getOperand(1));
      Result = DAG.getStore(Tmp1, Tmp3, Tmp2, ST->getSrcValue(),
                            SVOffset, isVolatile, Alignment);
      // The scalarized value type may not be legal, e.g. it might require
      // promotion or expansion.  Relegalize the scalar store.
      return LegalizeOp(Result);
    } else {
      SplitVectorOp(Node->getOperand(1), Lo, Hi);
      IncrementSize = NumElems/2 * MVT::getSizeInBits(EVT)/8;
    }
#endif
  } else {
    GetExpandedOp(N->getValue(), Lo, Hi);
    IncrementSize = Hi.Val ? MVT::getSizeInBits(Hi.getValueType())/8 : 0;
    
    if (!TLI.isLittleEndian())
      std::swap(Lo, Hi);
  }
  
  SDOperand Chain    = N->getChain();
  SDOperand Ptr      = N->getBasePtr();
  int SVOffset       = N->getSrcValueOffset();
  unsigned Alignment = N->getAlignment();
  bool isVolatile    = N->isVolatile();
  
  Lo = DAG.getStore(Chain, Lo, Ptr, N->getSrcValue(),
                    SVOffset, isVolatile, Alignment);
  
  assert(Hi.Val && "FIXME: int <-> float should be handled with promote!");
#if 0
  if (Hi.Val == NULL) {
    // Must be int <-> float one-to-one expansion.
    return Lo;
  }
#endif
  
  Ptr = DAG.getNode(ISD::ADD, Ptr.getValueType(), Ptr,
                    getIntPtrConstant(IncrementSize));
  assert(isTypeLegal(Ptr.getValueType()) && "Pointers must be legal!");
  Hi = DAG.getStore(Chain, Hi, Ptr, N->getSrcValue(), SVOffset+IncrementSize,
                    isVolatile, std::max(Alignment, IncrementSize));
  return DAG.getNode(ISD::TokenFactor, MVT::Other, Lo, Hi);
}

//===----------------------------------------------------------------------===//
//  Entry Point
//===----------------------------------------------------------------------===//

/// LegalizeTypes - This transforms the SelectionDAG into a SelectionDAG that
/// only uses types natively supported by the target.
///
/// Note that this is an involved process that may invalidate pointers into
/// the graph.
void SelectionDAG::LegalizeTypes() {
  DAGTypeLegalizer(*this).run();
}

