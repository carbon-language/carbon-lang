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
// *** instruction selector class.  These functions are really methods.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DAGISEL_HEADER_H
#define LLVM_CODEGEN_DAGISEL_HEADER_H

/// ISelQueue - Instruction selector priority queue sorted 
/// in the order of decreasing NodeId() values.
std::vector<SDNode*> ISelQueue;

/// Keep track of nodes which have already been added to queue.
unsigned char *ISelQueued;

/// Keep track of nodes which have already been selected.
unsigned char *ISelSelected;

/// IsChainCompatible - Returns true if Chain is Op or Chain does
/// not reach Op.
static bool IsChainCompatible(SDNode *Chain, SDNode *Op) {
  if (Chain->getOpcode() == ISD::EntryToken)
    return true;
  else if (Chain->getOpcode() == ISD::TokenFactor)
    return false;
  else if (Chain->getNumOperands() > 0) {
    SDValue C0 = Chain->getOperand(0);
    if (C0.getValueType() == MVT::Other)
      return C0.getNode() != Op && IsChainCompatible(C0.getNode(), Op);
  }
  return true;
}

/// isel_sort - Sorting functions for the selection queue in the
/// decreasing NodeId order.
struct isel_sort : public std::binary_function<SDNode*, SDNode*, bool> {
  bool operator()(const SDNode* left, const SDNode* right) const {
    return left->getNodeId() < right->getNodeId();
  }
};

/// setQueued - marks the node with a given NodeId() as element of the 
/// instruction selection queue.
inline void setQueued(int Id) {
  ISelQueued[Id / 8] |= 1 << (Id % 8);
}

/// isSelected - checks if the node with a given NodeId() is
/// in the instruction selection queue already.
inline bool isQueued(int Id) {
  return ISelQueued[Id / 8] & (1 << (Id % 8));
}

/// setSelected - marks the node with a given NodeId() as selected.
inline void setSelected(int Id) {
  ISelSelected[Id / 8] |= 1 << (Id % 8);
}

/// isSelected - checks if the node with a given NodeId() is
/// selected already.
inline bool isSelected(int Id) {
  return ISelSelected[Id / 8] & (1 << (Id % 8));
}

/// AddToISelQueue - adds a node to the instruction 
/// selection queue.
void AddToISelQueue(SDValue N) DISABLE_INLINE {
  int Id = N.getNode()->getNodeId();
  if (Id != -1 && !isQueued(Id)) {
    ISelQueue.push_back(N.getNode());
    std::push_heap(ISelQueue.begin(), ISelQueue.end(), isel_sort());
    setQueued(Id);
  }
}

/// ISelQueueUpdater - helper class to handle updates of the 
/// instruciton selection queue.
class VISIBILITY_HIDDEN ISelQueueUpdater :
  public SelectionDAG::DAGUpdateListener {
    std::vector<SDNode*> &ISelQueue;
    bool HadDelete; // Indicate if any deletions were done.
  public:
    explicit ISelQueueUpdater(std::vector<SDNode*> &isq)
      : ISelQueue(isq), HadDelete(false) {}
    
    bool hadDelete() const { return HadDelete; }

    /// NodeDeleted - remove node from the selection queue.
    virtual void NodeDeleted(SDNode *N, SDNode *E) {
      ISelQueue.erase(std::remove(ISelQueue.begin(), ISelQueue.end(), N),
                      ISelQueue.end());
      HadDelete = true;
    }

    /// NodeUpdated - Ignore updates for now.
    virtual void NodeUpdated(SDNode *N) {}
  };

/// UpdateQueue - update the instruction selction queue to maintain 
/// the decreasing NodeId() ordering property.
inline void UpdateQueue(const ISelQueueUpdater &ISQU) {
  if (ISQU.hadDelete())
    std::make_heap(ISelQueue.begin(), ISelQueue.end(),isel_sort());
}


/// ReplaceUses - replace all uses of the old node F with the use
/// of the new node T.
void ReplaceUses(SDValue F, SDValue T) DISABLE_INLINE {
  ISelQueueUpdater ISQU(ISelQueue);
  CurDAG->ReplaceAllUsesOfValueWith(F, T, &ISQU);
  setSelected(F.getNode()->getNodeId());
  UpdateQueue(ISQU);
}

/// ReplaceUses - replace all uses of the old nodes F with the use
/// of the new nodes T.
void ReplaceUses(const SDValue *F, const SDValue *T,
                 unsigned Num) DISABLE_INLINE {
  ISelQueueUpdater ISQU(ISelQueue);
  CurDAG->ReplaceAllUsesOfValuesWith(F, T, Num, &ISQU);
  for (unsigned i = 0; i != Num; ++i)
    setSelected(F[i].getNode()->getNodeId());
  UpdateQueue(ISQU);
}

/// ReplaceUses - replace all uses of the old node F with the use
/// of the new node T.
void ReplaceUses(SDNode *F, SDNode *T) DISABLE_INLINE {
  unsigned FNumVals = F->getNumValues();
  unsigned TNumVals = T->getNumValues();
  ISelQueueUpdater ISQU(ISelQueue);
  if (FNumVals != TNumVals) {
    for (unsigned i = 0, e = std::min(FNumVals, TNumVals); i < e; ++i)
     CurDAG->ReplaceAllUsesOfValueWith(SDValue(F, i), SDValue(T, i), &ISQU);
  } else {
    CurDAG->ReplaceAllUsesWith(F, T, &ISQU);
  }
  setSelected(F->getNodeId());
  UpdateQueue(ISQU);
}

/// SelectRoot - Top level entry to DAG instruction selector.
/// Selects instructions starting at the root of the current DAG.
void SelectRoot() {
  SelectRootInit();
  unsigned NumBytes = (DAGSize + 7) / 8;
  ISelQueued   = new unsigned char[NumBytes];
  ISelSelected = new unsigned char[NumBytes];
  memset(ISelQueued,   0, NumBytes);
  memset(ISelSelected, 0, NumBytes);

  // Create a dummy node (which is not added to allnodes), that adds
  // a reference to the root node, preventing it from being deleted,
  // and tracking any changes of the root.
  HandleSDNode Dummy(CurDAG->getRoot());
  ISelQueue.push_back(CurDAG->getRoot().getNode());

  // Select pending nodes from the instruction selection queue
  // until no more nodes are left for selection.
  while (!ISelQueue.empty()) {
    SDNode *Node = ISelQueue.front();
    std::pop_heap(ISelQueue.begin(), ISelQueue.end(), isel_sort());
    ISelQueue.pop_back();
    // Skip already selected nodes.
    if (isSelected(Node->getNodeId()))
      continue;
    SDNode *ResNode = Select(SDValue(Node, 0));
    // If node should not be replaced, 
    // continue with the next one.
    if (ResNode == Node)
      continue;
    // Replace node.
    if (ResNode)
      ReplaceUses(Node, ResNode);
    // If after the replacement this node is not used any more,
    // remove this dead node.
    if (Node->use_empty()) { // Don't delete EntryToken, etc.
      ISelQueueUpdater ISQU(ISelQueue);
      CurDAG->RemoveDeadNode(Node, &ISQU);
      UpdateQueue(ISQU);
    }
  }

  delete[] ISelQueued;
  ISelQueued = NULL;
  delete[] ISelSelected;
  ISelSelected = NULL;
  CurDAG->setRoot(Dummy.getValue());
}

#endif /* LLVM_CODEGEN_DAGISEL_HEADER_H */
