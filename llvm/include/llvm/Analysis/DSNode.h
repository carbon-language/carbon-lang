//===- DSNode.h - Node definition for datastructure graphs ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Data structure graph nodes and some implementation of DSNodeHandle.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DSNODE_H
#define LLVM_ANALYSIS_DSNODE_H

#include "llvm/Analysis/DSSupport.h"

namespace llvm {

template<typename BaseType>
class DSNodeIterator;          // Data structure graph traversal iterator
class TargetData;

//===----------------------------------------------------------------------===//
/// DSNode - Data structure node class
///
/// This class represents an untyped memory object of Size bytes.  It keeps
/// track of any pointers that have been stored into the object as well as the
/// different types represented in this object.
///
class DSNode {
  /// NumReferrers - The number of DSNodeHandles pointing to this node... if
  /// this is a forwarding node, then this is the number of node handles which
  /// are still forwarding over us.
  ///
  unsigned NumReferrers;

  /// ForwardNH - This NodeHandle contain the node (and offset into the node)
  /// that this node really is.  When nodes get folded together, the node to be
  /// eliminated has these fields filled in, otherwise ForwardNH.getNode() is
  /// null.
  ///
  DSNodeHandle ForwardNH;

  /// Next, Prev - These instance variables are used to keep the node on a
  /// doubly-linked ilist in the DSGraph.
  ///
  DSNode *Next, *Prev;
  friend class ilist_traits<DSNode>;

  /// Size - The current size of the node.  This should be equal to the size of
  /// the current type record.
  ///
  unsigned Size;

  /// ParentGraph - The graph this node is currently embedded into.
  ///
  DSGraph *ParentGraph;

  /// Ty - Keep track of the current outer most type of this object, in addition
  /// to whether or not it has been indexed like an array or not.  If the
  /// isArray bit is set, the node cannot grow.
  ///
  const Type *Ty;                 // The type itself...

  /// Links - Contains one entry for every sizeof(void*) bytes in this memory
  /// object.  Note that if the node is not a multiple of size(void*) bytes
  /// large, that there is an extra entry for the "remainder" of the node as
  /// well.  For this reason, nodes of 1 byte in size do have one link.
  ///
  std::vector<DSNodeHandle> Links;

  /// Globals - The list of global values that are merged into this node.
  ///
  std::vector<GlobalValue*> Globals;

  void operator=(const DSNode &); // DO NOT IMPLEMENT
  DSNode(const DSNode &);         // DO NOT IMPLEMENT
public:
  enum NodeTy {
    ShadowNode  = 0,        // Nothing is known about this node...
    AllocaNode  = 1 << 0,   // This node was allocated with alloca
    HeapNode    = 1 << 1,   // This node was allocated with malloc
    GlobalNode  = 1 << 2,   // This node was allocated by a global var decl
    UnknownNode = 1 << 3,   // This node points to unknown allocated memory 
    Incomplete  = 1 << 4,   // This node may not be complete

    Modified    = 1 << 5,   // This node is modified in this context
    Read        = 1 << 6,   // This node is read in this context

    Array       = 1 << 7,   // This node is treated like an array
    //#ifndef NDEBUG
    DEAD        = 1 << 8,   // This node is dead and should not be pointed to
    //#endif

    Composition = AllocaNode | HeapNode | GlobalNode | UnknownNode,
  };
  
  /// NodeType - A union of the above bits.  "Shadow" nodes do not add any flags
  /// to the nodes in the data structure graph, so it is possible to have nodes
  /// with a value of 0 for their NodeType.
  ///
private:
  unsigned short NodeType;
public:
  
  /// DSNode ctor - Create a node of the specified type, inserting it into the
  /// specified graph.
  ///
  DSNode(const Type *T, DSGraph *G);

  /// DSNode "copy ctor" - Copy the specified node, inserting it into the
  /// specified graph.  If NullLinks is true, then null out all of the links,
  /// but keep the same number of them.  This can be used for efficiency if the
  /// links are just going to be clobbered anyway.
  ///
  DSNode(const DSNode &, DSGraph *G, bool NullLinks = false);

  ~DSNode() {
    dropAllReferences();
    assert(hasNoReferrers() && "Referrers to dead node exist!");
  }

  // Iterator for graph interface... Defined in DSGraphTraits.h
  typedef DSNodeIterator<DSNode> iterator;
  typedef DSNodeIterator<const DSNode> const_iterator;
  inline iterator begin();
  inline iterator end();
  inline const_iterator begin() const;
  inline const_iterator end() const;

  //===--------------------------------------------------
  // Accessors

  /// getSize - Return the maximum number of bytes occupied by this object...
  ///
  unsigned getSize() const { return Size; }

  /// getType - Return the node type of this object...
  ///
  const Type *getType() const { return Ty; }

  bool isArray() const { return NodeType & Array; }

  /// hasNoReferrers - Return true if nothing is pointing to this node at all.
  ///
  bool hasNoReferrers() const { return getNumReferrers() == 0; }

  /// getNumReferrers - This method returns the number of referrers to the
  /// current node.  Note that if this node is a forwarding node, this will
  /// return the number of nodes forwarding over the node!
  unsigned getNumReferrers() const { return NumReferrers; }

  DSGraph *getParentGraph() const { return ParentGraph; }
  void setParentGraph(DSGraph *G) { ParentGraph = G; }


  /// getTargetData - Get the target data object used to construct this node.
  ///
  const TargetData &getTargetData() const;

  /// getForwardNode - This method returns the node that this node is forwarded
  /// to, if any.
  ///
  DSNode *getForwardNode() const { return ForwardNH.getNode(); }

  /// isForwarding - Return true if this node is forwarding to another.
  ///
  bool isForwarding() const { return !ForwardNH.isNull(); }

  /// stopForwarding - When the last reference to this forwarding node has been
  /// dropped, delete the node.
  ///
  void stopForwarding() {
    assert(isForwarding() &&
           "Node isn't forwarding, cannot stopForwarding()!");
    ForwardNH.setNode(0);
    assert(ParentGraph == 0 &&
           "Forwarding nodes must have been removed from graph!");
    delete this;
  }

  /// hasLink - Return true if this memory object has a link in slot #LinkNo
  ///
  bool hasLink(unsigned Offset) const {
    assert((Offset & ((1 << DS::PointerShift)-1)) == 0 &&
           "Pointer offset not aligned correctly!");
    unsigned Index = Offset >> DS::PointerShift;
    assert(Index < Links.size() && "Link index is out of range!");
    return Links[Index].getNode();
  }

  /// getLink - Return the link at the specified offset.
  ///
  DSNodeHandle &getLink(unsigned Offset) {
    assert((Offset & ((1 << DS::PointerShift)-1)) == 0 &&
           "Pointer offset not aligned correctly!");
    unsigned Index = Offset >> DS::PointerShift;
    assert(Index < Links.size() && "Link index is out of range!");
    return Links[Index];
  }
  const DSNodeHandle &getLink(unsigned Offset) const {
    assert((Offset & ((1 << DS::PointerShift)-1)) == 0 &&
           "Pointer offset not aligned correctly!");
    unsigned Index = Offset >> DS::PointerShift;
    assert(Index < Links.size() && "Link index is out of range!");
    return Links[Index];
  }

  /// getNumLinks - Return the number of links in a node...
  ///
  unsigned getNumLinks() const { return Links.size(); }

  /// mergeTypeInfo - This method merges the specified type into the current
  /// node at the specified offset.  This may update the current node's type
  /// record if this gives more information to the node, it may do nothing to
  /// the node if this information is already known, or it may merge the node
  /// completely (and return true) if the information is incompatible with what
  /// is already known.
  ///
  /// This method returns true if the node is completely folded, otherwise
  /// false.
  ///
  bool mergeTypeInfo(const Type *Ty, unsigned Offset,
                     bool FoldIfIncompatible = true);

  /// foldNodeCompletely - If we determine that this node has some funny
  /// behavior happening to it that we cannot represent, we fold it down to a
  /// single, completely pessimistic, node.  This node is represented as a
  /// single byte with a single TypeEntry of "void" with isArray = true.
  ///
  void foldNodeCompletely();

  /// isNodeCompletelyFolded - Return true if this node has been completely
  /// folded down to something that can never be expanded, effectively losing
  /// all of the field sensitivity that may be present in the node.
  ///
  bool isNodeCompletelyFolded() const;

  /// setLink - Set the link at the specified offset to the specified
  /// NodeHandle, replacing what was there.  It is uncommon to use this method,
  /// instead one of the higher level methods should be used, below.
  ///
  void setLink(unsigned Offset, const DSNodeHandle &NH) {
    assert((Offset & ((1 << DS::PointerShift)-1)) == 0 &&
           "Pointer offset not aligned correctly!");
    unsigned Index = Offset >> DS::PointerShift;
    assert(Index < Links.size() && "Link index is out of range!");
    Links[Index] = NH;
  }

  /// getPointerSize - Return the size of a pointer for the current target.
  ///
  unsigned getPointerSize() const { return DS::PointerSize; }

  /// addEdgeTo - Add an edge from the current node to the specified node.  This
  /// can cause merging of nodes in the graph.
  ///
  void addEdgeTo(unsigned Offset, const DSNodeHandle &NH);

  /// mergeWith - Merge this node and the specified node, moving all links to
  /// and from the argument node into the current node, deleting the node
  /// argument.  Offset indicates what offset the specified node is to be merged
  /// into the current node.
  ///
  /// The specified node may be a null pointer (in which case, nothing happens).
  ///
  void mergeWith(const DSNodeHandle &NH, unsigned Offset);

  /// addGlobal - Add an entry for a global value to the Globals list.  This
  /// also marks the node with the 'G' flag if it does not already have it.
  ///
  void addGlobal(GlobalValue *GV);
  void mergeGlobals(const std::vector<GlobalValue*> &RHS);
  const std::vector<GlobalValue*> &getGlobals() const { return Globals; }

  typedef std::vector<GlobalValue*>::const_iterator global_iterator;
  global_iterator global_begin() const { return Globals.begin(); }
  global_iterator global_end() const { return Globals.end(); }


  /// maskNodeTypes - Apply a mask to the node types bitfield.
  ///
  void maskNodeTypes(unsigned Mask) {
    NodeType &= Mask;
  }

  void mergeNodeFlags(unsigned RHS) {
    NodeType |= RHS;
  }

  /// getNodeFlags - Return all of the flags set on the node.  If the DEAD flag
  /// is set, hide it from the caller.
  ///
  unsigned getNodeFlags() const { return NodeType & ~DEAD; }

  bool isAllocaNode()  const { return NodeType & AllocaNode; }
  bool isHeapNode()    const { return NodeType & HeapNode; }
  bool isGlobalNode()  const { return NodeType & GlobalNode; }
  bool isUnknownNode() const { return NodeType & UnknownNode; }

  bool isModified() const   { return NodeType & Modified; }
  bool isRead() const       { return NodeType & Read; }

  bool isIncomplete() const { return NodeType & Incomplete; }
  bool isComplete() const   { return !isIncomplete(); }
  bool isDeadNode() const   { return NodeType & DEAD; }

  DSNode *setAllocaNodeMarker()  { NodeType |= AllocaNode;  return this; }
  DSNode *setHeapNodeMarker()    { NodeType |= HeapNode;    return this; }
  DSNode *setGlobalNodeMarker()  { NodeType |= GlobalNode;  return this; }
  DSNode *setUnknownNodeMarker() { NodeType |= UnknownNode; return this; }

  DSNode *setIncompleteMarker() { NodeType |= Incomplete; return this; }
  DSNode *setModifiedMarker()   { NodeType |= Modified;   return this; }
  DSNode *setReadMarker()       { NodeType |= Read;       return this; }
  DSNode *setArrayMarker()      { NodeType |= Array; return this; }

  void makeNodeDead() {
    Globals.clear();
    assert(hasNoReferrers() && "Dead node shouldn't have refs!");
    NodeType = DEAD;
  }

  /// forwardNode - Mark this node as being obsolete, and all references to it
  /// should be forwarded to the specified node and offset.
  ///
  void forwardNode(DSNode *To, unsigned Offset);

  void print(std::ostream &O, const DSGraph *G) const;
  void dump() const;

  void assertOK() const;

  void dropAllReferences() {
    Links.clear();
    if (isForwarding())
      ForwardNH.setNode(0);
  }

  /// remapLinks - Change all of the Links in the current node according to the
  /// specified mapping.
  ///
  void remapLinks(hash_map<const DSNode*, DSNodeHandle> &OldNodeMap);

  /// markReachableNodes - This method recursively traverses the specified
  /// DSNodes, marking any nodes which are reachable.  All reachable nodes it
  /// adds to the set, which allows it to only traverse visited nodes once.
  ///
  void markReachableNodes(hash_set<DSNode*> &ReachableNodes);

private:
  friend class DSNodeHandle;

  // static mergeNodes - Helper for mergeWith()
  static void MergeNodes(DSNodeHandle& CurNodeH, DSNodeHandle& NH);
};

//===----------------------------------------------------------------------===//
// Define the ilist_traits specialization for the DSGraph ilist.
//
template<>
struct ilist_traits<DSNode> {
  static DSNode *getPrev(const DSNode *N) { return N->Prev; }
  static DSNode *getNext(const DSNode *N) { return N->Next; }

  static void setPrev(DSNode *N, DSNode *Prev) { N->Prev = Prev; }
  static void setNext(DSNode *N, DSNode *Next) { N->Next = Next; }

  static DSNode *createNode() { return new DSNode(0,0); }
  //static DSNode *createNode(const DSNode &V) { return new DSNode(V); }


  void addNodeToList(DSNode *NTy) {}
  void removeNodeFromList(DSNode *NTy) {}
  void transferNodesFromList(iplist<DSNode, ilist_traits> &L2,
                             ilist_iterator<DSNode> first,
                             ilist_iterator<DSNode> last) {}
};

template<>
struct ilist_traits<const DSNode> : public ilist_traits<DSNode> {};

//===----------------------------------------------------------------------===//
// Define inline DSNodeHandle functions that depend on the definition of DSNode
//
inline DSNode *DSNodeHandle::getNode() const {
  // Disabling this assertion because it is failing on a "magic" struct
  // in named (from bind).  The fourth field is an array of length 0,
  // presumably used to create struct instances of different sizes.
  assert((!N ||
          N->isNodeCompletelyFolded() ||
          (N->Size == 0 && Offset == 0) ||
          (int(Offset) >= 0 && Offset < N->Size) ||
          (int(Offset) < 0 && -int(Offset) < int(N->Size)) ||
          N->isForwarding()) && "Node handle offset out of range!");
  if (N == 0 || !N->isForwarding())
    return N;

  return HandleForwarding();
}

inline void DSNodeHandle::setNode(DSNode *n) const {
  assert(!n || !n->isForwarding() && "Cannot set node to a forwarded node!");
  if (N) getNode()->NumReferrers--;
  N = n;
  if (N) {
    N->NumReferrers++;
    if (Offset >= N->Size) {
      assert((Offset == 0 || N->Size == 1) &&
             "Pointer to non-collapsed node with invalid offset!");
      Offset = 0;
    }
  }
  assert(!N || ((N->NodeType & DSNode::DEAD) == 0));
  assert((!N || Offset < N->Size || (N->Size == 0 && Offset == 0) ||
          N->isForwarding()) && "Node handle offset out of range!");
}

inline bool DSNodeHandle::hasLink(unsigned Num) const {
  assert(N && "DSNodeHandle does not point to a node yet!");
  return getNode()->hasLink(Num+Offset);
}


/// getLink - Treat this current node pointer as a pointer to a structure of
/// some sort.  This method will return the pointer a mem[this+Num]
///
inline const DSNodeHandle &DSNodeHandle::getLink(unsigned Off) const {
  assert(N && "DSNodeHandle does not point to a node yet!");
  return getNode()->getLink(Offset+Off);
}
inline DSNodeHandle &DSNodeHandle::getLink(unsigned Off) {
  assert(N && "DSNodeHandle does not point to a node yet!");
  return getNode()->getLink(Off+Offset);
}

inline void DSNodeHandle::setLink(unsigned Off, const DSNodeHandle &NH) {
  assert(N && "DSNodeHandle does not point to a node yet!");
  getNode()->setLink(Off+Offset, NH);
}

/// addEdgeTo - Add an edge from the current node to the specified node.  This
/// can cause merging of nodes in the graph.
///
inline void DSNodeHandle::addEdgeTo(unsigned Off, const DSNodeHandle &Node) {
  assert(N && "DSNodeHandle does not point to a node yet!");
  getNode()->addEdgeTo(Off+Offset, Node);
}

/// mergeWith - Merge the logical node pointed to by 'this' with the node
/// pointed to by 'N'.
///
inline void DSNodeHandle::mergeWith(const DSNodeHandle &Node) const {
  if (!isNull())
    getNode()->mergeWith(Node, Offset);
  else {   // No node to merge with, so just point to Node
    Offset = 0;
    setNode(Node.getNode());
    Offset = Node.getOffset();
  }
}

} // End llvm namespace

#endif
