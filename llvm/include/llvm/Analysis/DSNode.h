//===- DSSupport.h - Support for datastructure graphs -----------*- C++ -*-===//
//
// Support for graph nodes, call sites, and types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DSNODE_H
#define LLVM_ANALYSIS_DSNODE_H

#include "llvm/Analysis/DSSupport.h"

//===----------------------------------------------------------------------===//
/// DSNode - Data structure node class
///
/// This class represents an untyped memory object of Size bytes.  It keeps
/// track of any pointers that have been stored into the object as well as the
/// different types represented in this object.
///
class DSNode {
  /// Links - Contains one entry for every _distinct_ pointer field in the
  /// memory block.  These are demand allocated and indexed by the MergeMap
  /// vector.
  ///
  std::vector<DSNodeHandle> Links;

  /// MergeMap - Maps from every byte in the object to a signed byte number.
  /// This map is neccesary due to the merging that is possible as part of the
  /// unification algorithm.  To merge two distinct bytes of the object together
  /// into a single logical byte, the indexes for the two bytes are set to the
  /// same value.  This fully general merging is capable of representing all
  /// manners of array merging if neccesary.
  ///
  /// This map is also used to map outgoing pointers to various byte offsets in
  /// this data structure node.  If this value is >= 0, then it indicates that
  /// the numbered entry in the Links vector contains the outgoing edge for this
  /// byte offset.  In this way, the Links vector can be demand allocated and
  /// byte elements of the node may be merged without needing a Link allocated
  /// for it.
  ///
  /// Initially, each each element of the MergeMap is assigned a unique negative
  /// number, which are then merged as the unification occurs.
  ///
  std::vector<signed char> MergeMap;

  /// Referrers - Keep track of all of the node handles that point to this
  /// DSNode.  These pointers may need to be updated to point to a different
  /// node if this node gets merged with it.
  ///
  std::vector<DSNodeHandle*> Referrers;

  /// TypeEntries - As part of the merging process of this algorithm, nodes of
  /// different types can be represented by this single DSNode.  This vector is
  /// kept sorted.
  ///
  std::vector<DSTypeRec> TypeEntries;

  /// Globals - The list of global values that are merged into this node.
  ///
  std::vector<GlobalValue*> Globals;

  void operator=(const DSNode &); // DO NOT IMPLEMENT
public:
  enum NodeTy {
    ShadowNode = 0,        // Nothing is known about this node...
    ScalarNode = 1 << 0,   // Scalar of the current function contains this value
    AllocaNode = 1 << 1,   // This node was allocated with alloca
    NewNode    = 1 << 2,   // This node was allocated with malloc
    GlobalNode = 1 << 3,   // This node was allocated by a global var decl
    Incomplete = 1 << 4,   // This node may not be complete
    Modified   = 1 << 5,   // This node is modified in this context
    Read       = 1 << 6,   // This node is read in this context
  };
  
  /// NodeType - A union of the above bits.  "Shadow" nodes do not add any flags
  /// to the nodes in the data structure graph, so it is possible to have nodes
  /// with a value of 0 for their NodeType.  Scalar and Alloca markers go away
  /// when function graphs are inlined.
  ///
  unsigned char NodeType;

  DSNode(enum NodeTy NT, const Type *T);
  DSNode(const DSNode &);

  ~DSNode() {
#ifndef NDEBUG
    dropAllReferences();  // Only needed to satisfy assertion checks...
    assert(Referrers.empty() && "Referrers to dead node exist!");
#endif
  }

  // Iterator for graph interface...
  typedef DSNodeIterator iterator;
  typedef DSNodeIterator const_iterator;
  inline iterator begin() const;   // Defined in DSGraphTraits.h
  inline iterator end() const;

  //===--------------------------------------------------
  // Accessors

  /// getSize - Return the maximum number of bytes occupied by this object...
  ///
  unsigned getSize() const { return MergeMap.size(); }

  // getTypeEntries - Return the possible types and their offsets in this object
  const std::vector<DSTypeRec> &getTypeEntries() const { return TypeEntries; }

  /// getReferrers - Return a list of the pointers to this node...
  ///
  const std::vector<DSNodeHandle*> &getReferrers() const { return Referrers; }

  /// isModified - Return true if this node may be modified in this context
  ///
  bool isModified() const { return (NodeType & Modified) != 0; }

  /// isRead - Return true if this node may be read in this context
  ///
  bool isRead() const { return (NodeType & Read) != 0; }


  /// hasLink - Return true if this memory object has a link at the specified
  /// location.
  ///
  bool hasLink(unsigned i) const {
    assert(i < getSize() && "Field Link index is out of range!");
    return MergeMap[i] >= 0;
  }

  DSNodeHandle *getLink(unsigned i) {
    if (hasLink(i))
      return &Links[MergeMap[i]];
    return 0;
  }
  const DSNodeHandle *getLink(unsigned i) const {
    if (hasLink(i))
      return &Links[MergeMap[i]];
    return 0;
  }

  /// getMergeMapLabel - Return the merge map entry specified, to allow printing
  /// out of DSNodes nicely for DOT graphs.
  ///
  int getMergeMapLabel(unsigned i) const {
    assert(i < MergeMap.size() && "MergeMap index out of range!");
    return MergeMap[i];
  }

  /// getTypeRec - This method returns the specified type record if it exists.
  /// If it does not yet exist, the method checks to see whether or not the
  /// request would result in an untrackable state.  If adding it would cause
  /// untrackable state, we foldNodeCompletely the node and return the void
  /// record, otherwise we add an new TypeEntry and return it.
  ///
  DSTypeRec &getTypeRec(const Type *Ty, unsigned Offset);

  /// foldNodeCompletely - If we determine that this node has some funny
  /// behavior happening to it that we cannot represent, we fold it down to a
  /// single, completely pessimistic, node.  This node is represented as a
  /// single byte with a single TypeEntry of "void".
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
  void setLink(unsigned i, const DSNodeHandle &NH);

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

  /// mergeIndexes - If we discover that two indexes are equivalent and must be
  /// merged, this function is used to do the dirty work.
  ///
  void mergeIndexes(unsigned idx1, unsigned idx2) {
    assert(idx1 < getSize() && idx2 < getSize() && "Indexes out of range!");
    signed char MV1 = MergeMap[idx1];
    signed char MV2 = MergeMap[idx2];
    if (MV1 != MV2)
      mergeMappedValues(MV1, MV2);
  }


  /// addGlobal - Add an entry for a global value to the Globals list.  This
  /// also marks the node with the 'G' flag if it does not already have it.
  ///
  void addGlobal(GlobalValue *GV);
  const std::vector<GlobalValue*> &getGlobals() const { return Globals; }
  std::vector<GlobalValue*> &getGlobals() { return Globals; }

  void print(std::ostream &O, const DSGraph *G) const;
  void dump() const;

  void dropAllReferences() {
    Links.clear();
  }

  /// remapLinks - Change all of the Links in the current node according to the
  /// specified mapping.
  void remapLinks(std::map<const DSNode*, DSNode*> &OldNodeMap);

private:
  friend class DSNodeHandle;
  // addReferrer - Keep the referrer set up to date...
  void addReferrer(DSNodeHandle *H) { Referrers.push_back(H); }
  void removeReferrer(DSNodeHandle *H);

  /// rewriteMergeMap - Loop over the mergemap, replacing any references to the
  /// index From to be references to the index To.
  ///
  void rewriteMergeMap(signed char From, signed char To) {
    assert(From != To && "Cannot change something into itself!");
    for (unsigned i = 0, e = MergeMap.size(); i != e; ++i)
      if (MergeMap[i] == From)
        MergeMap[i] = To;
  }

  /// mergeMappedValues - This is the higher level form of rewriteMergeMap.  It
  /// is fully capable of merging links together if neccesary as well as simply
  /// rewriting the map entries.
  ///
  void mergeMappedValues(signed char V1, signed char V2);

  /// growNode - Attempt to grow the node to the specified size.  This may do
  /// one of three things:
  ///   1. Grow the node, return false
  ///   2. Refuse to grow the node, but maintain a trackable situation, return
  ///      false.
  ///   3. Be unable to track if node was that size, so collapse the node and
  ///      return true.
  ///
  bool growNode(unsigned RequestedSize);
};


//===----------------------------------------------------------------------===//
// Define inline DSNodeHandle functions that depend on the definition of DSNode
//

inline void DSNodeHandle::setNode(DSNode *n) {
  if (N) N->removeReferrer(this);
  N = n;
  if (N) N->addReferrer(this);
}

inline bool DSNodeHandle::hasLink(unsigned Num) const {
  assert(N && "DSNodeHandle does not point to a node yet!");
  return N->hasLink(Num+Offset);
}


/// getLink - Treat this current node pointer as a pointer to a structure of
/// some sort.  This method will return the pointer a mem[this+Num]
///
inline const DSNodeHandle *DSNodeHandle::getLink(unsigned Num) const {
  assert(N && "DSNodeHandle does not point to a node yet!");
  return N->getLink(Num+Offset);
}
inline DSNodeHandle *DSNodeHandle::getLink(unsigned Num) {
  assert(N && "DSNodeHandle does not point to a node yet!");
  return N->getLink(Num+Offset);
}

inline void DSNodeHandle::setLink(unsigned Num, const DSNodeHandle &NH) {
  assert(N && "DSNodeHandle does not point to a node yet!");
  N->setLink(Num+Offset, NH);
}

///  addEdgeTo - Add an edge from the current node to the specified node.  This
/// can cause merging of nodes in the graph.
///
inline void DSNodeHandle::addEdgeTo(unsigned LinkNo, const DSNodeHandle &Node) {
  assert(N && "DSNodeHandle does not point to a node yet!");
  N->addEdgeTo(LinkNo+Offset, Node);
}

/// mergeWith - Merge the logical node pointed to by 'this' with the node
/// pointed to by 'N'.
///
inline void DSNodeHandle::mergeWith(const DSNodeHandle &Node) {
  assert(N && "DSNodeHandle does not point to a node yet!");
  N->mergeWith(Node, Offset);
}

#endif
