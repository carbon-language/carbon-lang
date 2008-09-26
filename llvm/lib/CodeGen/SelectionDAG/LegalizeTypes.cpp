//===-- LegalizeTypes.cpp - Common code for DAG type legalizer ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SelectionDAG::LegalizeTypes method.  It transforms
// an arbitrary well-formed SelectionDAG to only consist of legal types.  This
// is common code shared among the LegalizeTypes*.cpp files.
//
//===----------------------------------------------------------------------===//

#include "LegalizeTypes.h"
#include "llvm/CallingConv.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetData.h"
using namespace llvm;

/// run - This is the main entry point for the type legalizer.  This does a
/// top-down traversal of the dag, legalizing types as it goes.
void DAGTypeLegalizer::run() {
  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted, and tracking any
  // changes of the root.
  HandleSDNode Dummy(DAG.getRoot());

  // The root of the dag may dangle to deleted nodes until the type legalizer is
  // done.  Set it to null to avoid confusion.
  DAG.setRoot(SDValue());

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

    if (IgnoreNodeResults(N))
      goto ScanOperands;

    // Scan the values produced by the node, checking to see if any result
    // types are illegal.
    for (unsigned i = 0, NumResults = N->getNumValues(); i < NumResults; ++i) {
      MVT ResultVT = N->getValueType(i);
      switch (getTypeAction(ResultVT)) {
      default:
        assert(false && "Unknown action!");
      case Legal:
        break;
      case PromoteInteger:
        PromoteIntegerResult(N, i);
        goto NodeDone;
      case ExpandInteger:
        ExpandIntegerResult(N, i);
        goto NodeDone;
      case SoftenFloat:
        SoftenFloatResult(N, i);
        goto NodeDone;
      case ExpandFloat:
        ExpandFloatResult(N, i);
        goto NodeDone;
      case ScalarizeVector:
        ScalarizeVectorResult(N, i);
        goto NodeDone;
      case SplitVector:
        SplitVectorResult(N, i);
        goto NodeDone;
      }
    }

ScanOperands:
    // Scan the operand list for the node, handling any nodes with operands that
    // are illegal.
    {
    unsigned NumOperands = N->getNumOperands();
    bool NeedsRevisit = false;
    unsigned i;
    for (i = 0; i != NumOperands; ++i) {
      if (IgnoreNodeResults(N->getOperand(i).getNode()))
        continue;

      MVT OpVT = N->getOperand(i).getValueType();
      switch (getTypeAction(OpVT)) {
      default:
        assert(false && "Unknown action!");
      case Legal:
        continue;
      case PromoteInteger:
        NeedsRevisit = PromoteIntegerOperand(N, i);
        break;
      case ExpandInteger:
        NeedsRevisit = ExpandIntegerOperand(N, i);
        break;
      case SoftenFloat:
        NeedsRevisit = SoftenFloatOperand(N, i);
        break;
      case ExpandFloat:
        NeedsRevisit = ExpandFloatOperand(N, i);
        break;
      case ScalarizeVector:
        NeedsRevisit = ScalarizeVectorOperand(N, i);
        break;
      case SplitVector:
        NeedsRevisit = SplitVectorOperand(N, i);
        break;
      }
      break;
    }

    // If the node needs revisiting, don't add all users to the worklist etc.
    if (NeedsRevisit)
      continue;

    if (i == NumOperands) {
      DEBUG(cerr << "Legally typed node: "; N->dump(&DAG); cerr << "\n");
    }
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
    bool Failed = false;

    // Check that all result types are legal.
    if (!IgnoreNodeResults(I))
      for (unsigned i = 0, NumVals = I->getNumValues(); i < NumVals; ++i)
        if (!isTypeLegal(I->getValueType(i))) {
          cerr << "Result type " << i << " illegal!\n";
          Failed = true;
        }

    // Check that all operand types are legal.
    for (unsigned i = 0, NumOps = I->getNumOperands(); i < NumOps; ++i)
      if (!IgnoreNodeResults(I->getOperand(i).getNode()) &&
          !isTypeLegal(I->getOperand(i).getValueType())) {
        cerr << "Operand type " << i << " illegal!\n";
        Failed = true;
      }

    if (I->getNodeId() != Processed) {
       if (I->getNodeId() == NewNode)
         cerr << "New node not 'noticed'?\n";
       else if (I->getNodeId() > 0)
         cerr << "Operand not processed?\n";
       else if (I->getNodeId() == ReadyToProcess)
         cerr << "Not added to worklist?\n";
       Failed = true;
    }

    if (Failed) {
      I->dump(&DAG); cerr << "\n";
      abort();
    }
  }
#endif
}

/// AnalyzeNewNode - The specified node is the root of a subtree of potentially
/// new nodes.  Correct any processed operands (this may change the node) and
/// calculate the NodeId.
/// Returns the potentially changed node.
SDNode *DAGTypeLegalizer::AnalyzeNewNode(SDNode *N) {
  // If this was an existing node that is already done, we're done.
  if (N->getNodeId() != NewNode)
    return N;

  // Remove any stale map entries.
  ExpungeNode(N);

  // Okay, we know that this node is new.  Recursively walk all of its operands
  // to see if they are new also.  The depth of this walk is bounded by the size
  // of the new tree that was constructed (usually 2-3 nodes), so we don't worry
  // about revisiting of nodes.
  //
  // As we walk the operands, keep track of the number of nodes that are
  // processed.  If non-zero, this will become the new nodeid of this node.
  // Already processed operands may need to be remapped to the node that
  // replaced them, which can result in our node changing.  Since remapping
  // is rare, the code tries to minimize overhead in the non-remapping case.

  SmallVector<SDValue, 8> NewOps;
  unsigned NumProcessed = 0;
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    SDValue OrigOp = N->getOperand(i);
    SDValue Op = OrigOp;

    if (Op.getNode()->getNodeId() == Processed)
      RemapNode(Op);

    if (Op.getNode()->getNodeId() == NewNode)
      AnalyzeNewNode(Op);
    else if (Op.getNode()->getNodeId() == Processed)
      ++NumProcessed;

    if (!NewOps.empty()) {
      // Some previous operand changed.  Add this one to the list.
      NewOps.push_back(Op);
    } else if (Op != OrigOp) {
      // This is the first operand to change - add all operands so far.
      for (unsigned j = 0; j < i; ++j)
        NewOps.push_back(N->getOperand(j));
      NewOps.push_back(Op);
    }
  }

  // Some operands changed - update the node.
  if (!NewOps.empty())
    N = DAG.UpdateNodeOperands(SDValue(N, 0),
                               &NewOps[0],
                               NewOps.size()).getNode();

  N->setNodeId(N->getNumOperands()-NumProcessed);
  if (N->getNodeId() == ReadyToProcess)
    Worklist.push_back(N);
  return N;
}

/// AnalyzeNewNode - call AnalyzeNewNode(SDNode *N)
/// and update the node in SDValue if necessary.
void DAGTypeLegalizer::AnalyzeNewNode(SDValue &Val) {
  SDNode *N(Val.getNode());
  SDNode *M(AnalyzeNewNode(N));
  if (N != M)
    Val.setNode(M);
}


namespace {
  /// NodeUpdateListener - This class is a DAGUpdateListener that listens for
  /// updates to nodes and recomputes their ready state.
  class VISIBILITY_HIDDEN NodeUpdateListener :
    public SelectionDAG::DAGUpdateListener {
    DAGTypeLegalizer &DTL;
  public:
    explicit NodeUpdateListener(DAGTypeLegalizer &dtl) : DTL(dtl) {}

    virtual void NodeDeleted(SDNode *N, SDNode *E) {
      assert(N->getNodeId() != DAGTypeLegalizer::Processed &&
             N->getNodeId() != DAGTypeLegalizer::ReadyToProcess &&
             "RAUW deleted processed node!");
      // It is possible, though rare, for the deleted node N to occur as a
      // target in a map, so note the replacement N -> E in ReplacedNodes.
      assert(E && "Node not replaced?");
      DTL.NoteDeletion(N, E);
    }

    virtual void NodeUpdated(SDNode *N) {
      // Node updates can mean pretty much anything.  It is possible that an
      // operand was set to something already processed (f.e.) in which case
      // this node could become ready.  Recompute its flags.
      assert(N->getNodeId() != DAGTypeLegalizer::Processed &&
             N->getNodeId() != DAGTypeLegalizer::ReadyToProcess &&
             "RAUW updated processed node!");
      DTL.ReanalyzeNode(N);
    }
  };
}


/// ReplaceValueWith - The specified value was legalized to the specified other
/// value.  If they are different, update the DAG and NodeIDs replacing any uses
/// of From to use To instead.
void DAGTypeLegalizer::ReplaceValueWith(SDValue From, SDValue To) {
  if (From == To) return;

  // If expansion produced new nodes, make sure they are properly marked.
  ExpungeNode(From.getNode());
  AnalyzeNewNode(To); // Expunges To.

  // Anything that used the old node should now use the new one.  Note that this
  // can potentially cause recursive merging.
  NodeUpdateListener NUL(*this);
  DAG.ReplaceAllUsesOfValueWith(From, To, &NUL);

  // The old node may still be present in a map like ExpandedIntegers or
  // PromotedIntegers.  Inform maps about the replacement.
  ReplacedNodes[From] = To;
}

/// ReplaceNodeWith - Replace uses of the 'from' node's results with the 'to'
/// node's results.  The from and to node must define identical result types.
void DAGTypeLegalizer::ReplaceNodeWith(SDNode *From, SDNode *To) {
  if (From == To) return;

  // If expansion produced new nodes, make sure they are properly marked.
  ExpungeNode(From);

  To = AnalyzeNewNode(To); // Expunges To.

  assert(From->getNumValues() == To->getNumValues() &&
         "Node results don't match");

  // Anything that used the old node should now use the new one.  Note that this
  // can potentially cause recursive merging.
  NodeUpdateListener NUL(*this);
  DAG.ReplaceAllUsesWith(From, To, &NUL);

  // The old node may still be present in a map like ExpandedIntegers or
  // PromotedIntegers.  Inform maps about the replacement.
  for (unsigned i = 0, e = From->getNumValues(); i != e; ++i) {
    assert(From->getValueType(i) == To->getValueType(i) &&
           "Node results don't match");
    ReplacedNodes[SDValue(From, i)] = SDValue(To, i);
  }
}

/// RemapNode - If the specified value was already legalized to another value,
/// replace it by that value.
void DAGTypeLegalizer::RemapNode(SDValue &N) {
  DenseMap<SDValue, SDValue>::iterator I = ReplacedNodes.find(N);
  if (I != ReplacedNodes.end()) {
    // Use path compression to speed up future lookups if values get multiply
    // replaced with other values.
    RemapNode(I->second);
    N = I->second;
  }
}

/// ExpungeNode - If N has a bogus mapping in ReplacedNodes, eliminate it.
/// This can occur when a node is deleted then reallocated as a new node -
/// the mapping in ReplacedNodes applies to the deleted node, not the new
/// one.
/// The only map that can have a deleted node as a source is ReplacedNodes.
/// Other maps can have deleted nodes as targets, but since their looked-up
/// values are always immediately remapped using RemapNode, resulting in a
/// not-deleted node, this is harmless as long as ReplacedNodes/RemapNode
/// always performs correct mappings.  In order to keep the mapping correct,
/// ExpungeNode should be called on any new nodes *before* adding them as
/// either source or target to ReplacedNodes (which typically means calling
/// Expunge when a new node is first seen, since it may no longer be marked
/// NewNode by the time it is added to ReplacedNodes).
void DAGTypeLegalizer::ExpungeNode(SDNode *N) {
  if (N->getNodeId() != NewNode)
    return;

  // If N is not remapped by ReplacedNodes then there is nothing to do.
  unsigned i, e;
  for (i = 0, e = N->getNumValues(); i != e; ++i)
    if (ReplacedNodes.find(SDValue(N, i)) != ReplacedNodes.end())
      break;

  if (i == e)
    return;

  // Remove N from all maps - this is expensive but rare.

  for (DenseMap<SDValue, SDValue>::iterator I = PromotedIntegers.begin(),
       E = PromotedIntegers.end(); I != E; ++I) {
    assert(I->first.getNode() != N);
    RemapNode(I->second);
  }

  for (DenseMap<SDValue, SDValue>::iterator I = SoftenedFloats.begin(),
       E = SoftenedFloats.end(); I != E; ++I) {
    assert(I->first.getNode() != N);
    RemapNode(I->second);
  }

  for (DenseMap<SDValue, SDValue>::iterator I = ScalarizedVectors.begin(),
       E = ScalarizedVectors.end(); I != E; ++I) {
    assert(I->first.getNode() != N);
    RemapNode(I->second);
  }

  for (DenseMap<SDValue, std::pair<SDValue, SDValue> >::iterator
       I = ExpandedIntegers.begin(), E = ExpandedIntegers.end(); I != E; ++I){
    assert(I->first.getNode() != N);
    RemapNode(I->second.first);
    RemapNode(I->second.second);
  }

  for (DenseMap<SDValue, std::pair<SDValue, SDValue> >::iterator
       I = ExpandedFloats.begin(), E = ExpandedFloats.end(); I != E; ++I) {
    assert(I->first.getNode() != N);
    RemapNode(I->second.first);
    RemapNode(I->second.second);
  }

  for (DenseMap<SDValue, std::pair<SDValue, SDValue> >::iterator
       I = SplitVectors.begin(), E = SplitVectors.end(); I != E; ++I) {
    assert(I->first.getNode() != N);
    RemapNode(I->second.first);
    RemapNode(I->second.second);
  }

  for (DenseMap<SDValue, SDValue>::iterator I = ReplacedNodes.begin(),
       E = ReplacedNodes.end(); I != E; ++I)
    RemapNode(I->second);

  for (unsigned i = 0, e = N->getNumValues(); i != e; ++i)
    ReplacedNodes.erase(SDValue(N, i));
}

void DAGTypeLegalizer::SetPromotedInteger(SDValue Op, SDValue Result) {
  AnalyzeNewNode(Result);

  SDValue &OpEntry = PromotedIntegers[Op];
  assert(OpEntry.getNode() == 0 && "Node is already promoted!");
  OpEntry = Result;
}

void DAGTypeLegalizer::SetSoftenedFloat(SDValue Op, SDValue Result) {
  AnalyzeNewNode(Result);

  SDValue &OpEntry = SoftenedFloats[Op];
  assert(OpEntry.getNode() == 0 && "Node is already converted to integer!");
  OpEntry = Result;
}

void DAGTypeLegalizer::SetScalarizedVector(SDValue Op, SDValue Result) {
  AnalyzeNewNode(Result);

  SDValue &OpEntry = ScalarizedVectors[Op];
  assert(OpEntry.getNode() == 0 && "Node is already scalarized!");
  OpEntry = Result;
}

void DAGTypeLegalizer::GetExpandedInteger(SDValue Op, SDValue &Lo,
                                          SDValue &Hi) {
  std::pair<SDValue, SDValue> &Entry = ExpandedIntegers[Op];
  RemapNode(Entry.first);
  RemapNode(Entry.second);
  assert(Entry.first.getNode() && "Operand isn't expanded");
  Lo = Entry.first;
  Hi = Entry.second;
}

void DAGTypeLegalizer::SetExpandedInteger(SDValue Op, SDValue Lo,
                                          SDValue Hi) {
  // Lo/Hi may have been newly allocated, if so, add nodeid's as relevant.
  AnalyzeNewNode(Lo);
  AnalyzeNewNode(Hi);

  // Remember that this is the result of the node.
  std::pair<SDValue, SDValue> &Entry = ExpandedIntegers[Op];
  assert(Entry.first.getNode() == 0 && "Node already expanded");
  Entry.first = Lo;
  Entry.second = Hi;
}

void DAGTypeLegalizer::GetExpandedFloat(SDValue Op, SDValue &Lo,
                                        SDValue &Hi) {
  std::pair<SDValue, SDValue> &Entry = ExpandedFloats[Op];
  RemapNode(Entry.first);
  RemapNode(Entry.second);
  assert(Entry.first.getNode() && "Operand isn't expanded");
  Lo = Entry.first;
  Hi = Entry.second;
}

void DAGTypeLegalizer::SetExpandedFloat(SDValue Op, SDValue Lo,
                                        SDValue Hi) {
  // Lo/Hi may have been newly allocated, if so, add nodeid's as relevant.
  AnalyzeNewNode(Lo);
  AnalyzeNewNode(Hi);

  // Remember that this is the result of the node.
  std::pair<SDValue, SDValue> &Entry = ExpandedFloats[Op];
  assert(Entry.first.getNode() == 0 && "Node already expanded");
  Entry.first = Lo;
  Entry.second = Hi;
}

void DAGTypeLegalizer::GetSplitVector(SDValue Op, SDValue &Lo,
                                      SDValue &Hi) {
  std::pair<SDValue, SDValue> &Entry = SplitVectors[Op];
  RemapNode(Entry.first);
  RemapNode(Entry.second);
  assert(Entry.first.getNode() && "Operand isn't split");
  Lo = Entry.first;
  Hi = Entry.second;
}

void DAGTypeLegalizer::SetSplitVector(SDValue Op, SDValue Lo,
                                      SDValue Hi) {
  // Lo/Hi may have been newly allocated, if so, add nodeid's as relevant.
  AnalyzeNewNode(Lo);
  AnalyzeNewNode(Hi);

  // Remember that this is the result of the node.
  std::pair<SDValue, SDValue> &Entry = SplitVectors[Op];
  assert(Entry.first.getNode() == 0 && "Node already split");
  Entry.first = Lo;
  Entry.second = Hi;
}


//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

/// BitConvertToInteger - Convert to an integer of the same size.
SDValue DAGTypeLegalizer::BitConvertToInteger(SDValue Op) {
  unsigned BitWidth = Op.getValueType().getSizeInBits();
  return DAG.getNode(ISD::BIT_CONVERT, MVT::getIntegerVT(BitWidth), Op);
}

SDValue DAGTypeLegalizer::CreateStackStoreLoad(SDValue Op,
                                               MVT DestVT) {
  // Create the stack frame object.  Make sure it is aligned for both
  // the source and destination types.
  unsigned SrcAlign =
   TLI.getTargetData()->getPrefTypeAlignment(Op.getValueType().getTypeForMVT());
  SDValue FIPtr = DAG.CreateStackTemporary(DestVT, SrcAlign);

  // Emit a store to the stack slot.
  SDValue Store = DAG.getStore(DAG.getEntryNode(), Op, FIPtr, NULL, 0);
  // Result is a load from the stack slot.
  return DAG.getLoad(DestVT, Store, FIPtr, NULL, 0);
}

/// JoinIntegers - Build an integer with low bits Lo and high bits Hi.
SDValue DAGTypeLegalizer::JoinIntegers(SDValue Lo, SDValue Hi) {
  MVT LVT = Lo.getValueType();
  MVT HVT = Hi.getValueType();
  MVT NVT = MVT::getIntegerVT(LVT.getSizeInBits() + HVT.getSizeInBits());

  Lo = DAG.getNode(ISD::ZERO_EXTEND, NVT, Lo);
  Hi = DAG.getNode(ISD::ANY_EXTEND, NVT, Hi);
  Hi = DAG.getNode(ISD::SHL, NVT, Hi, DAG.getConstant(LVT.getSizeInBits(),
                                                      TLI.getShiftAmountTy()));
  return DAG.getNode(ISD::OR, NVT, Lo, Hi);
}

/// SplitInteger - Return the lower LoVT bits of Op in Lo and the upper HiVT
/// bits in Hi.
void DAGTypeLegalizer::SplitInteger(SDValue Op,
                                    MVT LoVT, MVT HiVT,
                                    SDValue &Lo, SDValue &Hi) {
  assert(LoVT.getSizeInBits() + HiVT.getSizeInBits() ==
         Op.getValueType().getSizeInBits() && "Invalid integer splitting!");
  Lo = DAG.getNode(ISD::TRUNCATE, LoVT, Op);
  Hi = DAG.getNode(ISD::SRL, Op.getValueType(), Op,
                   DAG.getConstant(LoVT.getSizeInBits(),
                                   TLI.getShiftAmountTy()));
  Hi = DAG.getNode(ISD::TRUNCATE, HiVT, Hi);
}

/// SplitInteger - Return the lower and upper halves of Op's bits in a value type
/// half the size of Op's.
void DAGTypeLegalizer::SplitInteger(SDValue Op,
                                    SDValue &Lo, SDValue &Hi) {
  MVT HalfVT = MVT::getIntegerVT(Op.getValueType().getSizeInBits()/2);
  SplitInteger(Op, HalfVT, HalfVT, Lo, Hi);
}

/// MakeLibCall - Generate a libcall taking the given operands as arguments and
/// returning a result of type RetVT.
SDValue DAGTypeLegalizer::MakeLibCall(RTLIB::Libcall LC, MVT RetVT,
                                      const SDValue *Ops, unsigned NumOps,
                                      bool isSigned) {
  TargetLowering::ArgListTy Args;
  Args.reserve(NumOps);

  TargetLowering::ArgListEntry Entry;
  for (unsigned i = 0; i != NumOps; ++i) {
    Entry.Node = Ops[i];
    Entry.Ty = Entry.Node.getValueType().getTypeForMVT();
    Entry.isSExt = isSigned;
    Entry.isZExt = !isSigned;
    Args.push_back(Entry);
  }
  SDValue Callee = DAG.getExternalSymbol(TLI.getLibcallName(LC),
                                           TLI.getPointerTy());

  const Type *RetTy = RetVT.getTypeForMVT();
  std::pair<SDValue,SDValue> CallInfo =
    TLI.LowerCallTo(DAG.getEntryNode(), RetTy, isSigned, !isSigned, false,
                    false, CallingConv::C, false, Callee, Args, DAG);
  return CallInfo.first;
}

SDValue DAGTypeLegalizer::GetVectorElementPointer(SDValue VecPtr, MVT EltVT,
                                                  SDValue Index) {
  // Make sure the index type is big enough to compute in.
  if (Index.getValueType().bitsGT(TLI.getPointerTy()))
    Index = DAG.getNode(ISD::TRUNCATE, TLI.getPointerTy(), Index);
  else
    Index = DAG.getNode(ISD::ZERO_EXTEND, TLI.getPointerTy(), Index);

  // Calculate the element offset and add it to the pointer.
  unsigned EltSize = EltVT.getSizeInBits() / 8; // FIXME: should be ABI size.

  Index = DAG.getNode(ISD::MUL, Index.getValueType(), Index,
                      DAG.getConstant(EltSize, Index.getValueType()));
  return DAG.getNode(ISD::ADD, Index.getValueType(), Index, VecPtr);
}

/// GetSplitDestVTs - Compute the VTs needed for the low/hi parts of a type
/// which is split into two not necessarily identical pieces.
void DAGTypeLegalizer::GetSplitDestVTs(MVT InVT, MVT &LoVT, MVT &HiVT) {
  if (!InVT.isVector()) {
    LoVT = HiVT = TLI.getTypeToTransformTo(InVT);
  } else {
    MVT NewEltVT = InVT.getVectorElementType();
    unsigned NumElements = InVT.getVectorNumElements();
    if ((NumElements & (NumElements-1)) == 0) {  // Simple power of two vector.
      NumElements >>= 1;
      LoVT = HiVT =  MVT::getVectorVT(NewEltVT, NumElements);
    } else {                                     // Non-power-of-two vectors.
      unsigned NewNumElts_Lo = 1 << Log2_32(NumElements);
      unsigned NewNumElts_Hi = NumElements - NewNumElts_Lo;
      LoVT = MVT::getVectorVT(NewEltVT, NewNumElts_Lo);
      HiVT = MVT::getVectorVT(NewEltVT, NewNumElts_Hi);
    }
  }
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
