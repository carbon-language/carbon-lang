//===- DSGraph.h - Represent a collection of data structures ----*- C++ -*-===//
//
// This header defines the primative classes that make up a data structure
// graph.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DSGRAPH_H
#define LLVM_ANALYSIS_DSGRAPH_H

#include "llvm/Pass.h"

class GlobalValue;
class Type;

class DSNode;                  // Each node in the graph
class DSGraph;                 // A graph for a function
class DSNodeIterator;          // Data structure graph traversal iterator

//===----------------------------------------------------------------------===//
/// DSNodeHandle - Implement a "handle" to a data structure node that takes care
/// of all of the add/un'refing of the node to prevent the backpointers in the
/// graph from getting out of date.  This class represents a "pointer" in the
/// graph, whose destination is an indexed offset into a node.
///
class DSNodeHandle {
  DSNode *N;
  unsigned Offset;
public:
  // Allow construction, destruction, and assignment...
  DSNodeHandle(DSNode *n = 0, unsigned offs = 0) : N(0), Offset(offs) {
    setNode(n);
  }
  DSNodeHandle(const DSNodeHandle &H) : N(0), Offset(H.Offset) { setNode(H.N); }
  ~DSNodeHandle() { setNode((DSNode*)0); }
  DSNodeHandle &operator=(const DSNodeHandle &H) {
    setNode(H.N); Offset = H.Offset;
    return *this;
  }

  bool operator<(const DSNodeHandle &H) const {  // Allow sorting
    return N < H.N || (N == H.N && Offset < H.Offset);
  }
  bool operator==(const DSNodeHandle &H) const { // Allow comparison
    return N == H.N && Offset == H.Offset;
  }
  bool operator!=(const DSNodeHandle &H) const { return !operator==(H); }

  // Allow explicit conversion to DSNode...
  DSNode *getNode() const { return N; }
  unsigned getOffset() const { return Offset; }

  inline void setNode(DSNode *N);  // Defined inline later...
  void setOffset(unsigned O) { Offset = O; }

  void addEdgeTo(unsigned LinkNo, const DSNodeHandle &N);
  void addEdgeTo(const DSNodeHandle &N) { addEdgeTo(0, N); }

  /// mergeWith - Merge the logical node pointed to by 'this' with the node
  /// pointed to by 'N'.
  ///
  void mergeWith(const DSNodeHandle &N);

  // hasLink - Return true if there is a link at the specified offset...
  inline bool hasLink(unsigned Num) const;

  /// getLink - Treat this current node pointer as a pointer to a structure of
  /// some sort.  This method will return the pointer a mem[this+Num]
  ///
  inline const DSNodeHandle *getLink(unsigned Num) const;
  inline DSNodeHandle *getLink(unsigned Num);

  inline void setLink(unsigned Num, const DSNodeHandle &NH);
};


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
  typedef std::pair<const Type *, unsigned> TypeRec;
  std::vector<TypeRec> TypeEntries;

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
  inline iterator begin();   // Defined in DataStructureGraph.h
  inline iterator end();

  //===--------------------------------------------------
  // Accessors

  // getSize - Return the maximum number of bytes occupied by this object...
  unsigned getSize() const { return MergeMap.size(); }

  // getTypeEntries - Return the possible types and their offsets in this object
  const std::vector<TypeRec> &getTypeEntries() const { return TypeEntries; }

  // getReferrers - Return a list of the pointers to this node...
  const std::vector<DSNodeHandle*> &getReferrers() const { return Referrers; }


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


//===----------------------------------------------------------------------===//
/// DSGraph - The graph that represents a function.
///
class DSGraph {
  Function *Func;
  std::vector<DSNode*> Nodes;
  DSNodeHandle RetNode;                          // Node that gets returned...
  std::map<Value*, DSNodeHandle> ValueMap;

#if 0
  // GlobalsGraph -- Reference to the common graph of globally visible objects.
  // This includes GlobalValues, New nodes, Cast nodes, and Calls.
  // 
  GlobalDSGraph* GlobalsGraph;
#endif

  // FunctionCalls - This vector maintains a single entry for each call
  // instruction in the current graph.  Each call entry contains DSNodeHandles
  // that refer to the arguments that are passed into the function call.  The
  // first entry in the vector is the scalar that holds the return value for the
  // call, the second is the function scalar being invoked, and the rest are
  // pointer arguments to the function.
  //
  std::vector<std::vector<DSNodeHandle> > FunctionCalls;

#if 0
  // OrigFunctionCalls - This vector retains a copy of the original function
  // calls of the current graph.  This is needed to support top-down inlining
  // after bottom-up inlining is complete, since the latter deletes call nodes.
  // 
  std::vector<std::vector<DSNodeHandle> > OrigFunctionCalls;

  // PendingCallers - This vector records all unresolved callers of the
  // current function, i.e., ones whose graphs have not been inlined into
  // the current graph.  As long as there are unresolved callers, the nodes
  // for formal arguments in the current graph cannot be eliminated, and
  // nodes in the graph reachable from the formal argument nodes or
  // global variable nodes must be considered incomplete. 
  std::set<Function*> PendingCallers;
#endif
  
protected:

#if 0
  // clone all the call nodes and save the copies in OrigFunctionCalls
  void saveOrigFunctionCalls() {
    assert(OrigFunctionCalls.size() == 0 && "Do this only once!");
    OrigFunctionCalls = FunctionCalls;
  }

  // get the saved copies of the original function call nodes
  std::vector<std::vector<DSNodeHandle> > &getOrigFunctionCalls() {
    return OrigFunctionCalls;
  }
#endif

  void operator=(const DSGraph &); // DO NOT IMPLEMENT
public:
  DSGraph() : Func(0) {}           // Create a new, empty, DSGraph.
  DSGraph(Function &F);            // Compute the local DSGraph
  DSGraph(const DSGraph &DSG);     // Copy ctor
  ~DSGraph();

  bool hasFunction() const { return Func != 0; }
  Function &getFunction() const { return *Func; }

  /// getNodes - Get a vector of all the nodes in the graph
  /// 
  const std::vector<DSNode*> &getNodes() const { return Nodes; }
        std::vector<DSNode*> &getNodes()       { return Nodes; }

  /// addNode - Add a new node to the graph.
  ///
  void addNode(DSNode *N) { Nodes.push_back(N); }

  /// getValueMap - Get a map that describes what the nodes the scalars in this
  /// function point to...
  ///
  std::map<Value*, DSNodeHandle> &getValueMap() { return ValueMap; }
  const std::map<Value*, DSNodeHandle> &getValueMap() const { return ValueMap;}

  std::vector<std::vector<DSNodeHandle> > &getFunctionCalls() {
    return FunctionCalls;
  }
  const std::vector<std::vector<DSNodeHandle> > &getFunctionCalls() const {
    return FunctionCalls;
  }

  const DSNodeHandle &getRetNode() const { return RetNode; }
        DSNodeHandle &getRetNode()       { return RetNode; }

  unsigned getGraphSize() const {
    return Nodes.size();
  }

  void print(std::ostream &O) const;
  void dump() const;
  void writeGraphToFile(std::ostream &O, const std::string &GraphName);

  // maskNodeTypes - Apply a mask to all of the node types in the graph.  This
  // is useful for clearing out markers like Scalar or Incomplete.
  //
  void maskNodeTypes(unsigned char Mask);
  void maskIncompleteMarkers() { maskNodeTypes(~DSNode::Incomplete); }

  // markIncompleteNodes - Traverse the graph, identifying nodes that may be
  // modified by other functions that have not been resolved yet.  This marks
  // nodes that are reachable through three sources of "unknownness":
  //   Global Variables, Function Calls, and Incoming Arguments
  //
  // For any node that may have unknown components (because something outside
  // the scope of current analysis may have modified it), the 'Incomplete' flag
  // is added to the NodeType.
  //
  void markIncompleteNodes(bool markFormalArgs = true);

  // removeTriviallyDeadNodes - After the graph has been constructed, this
  // method removes all unreachable nodes that are created because they got
  // merged with other nodes in the graph.
  //
  void removeTriviallyDeadNodes(bool KeepAllGlobals = false);

  // removeDeadNodes - Use a more powerful reachability analysis to eliminate
  // subgraphs that are unreachable.  This often occurs because the data
  // structure doesn't "escape" into it's caller, and thus should be eliminated
  // from the caller's graph entirely.  This is only appropriate to use when
  // inlining graphs.
  //
  void removeDeadNodes(bool KeepAllGlobals = false, bool KeepCalls = true);

#if 0
  // AddCaller - add a known caller node into the graph and mark it pending.
  // getCallers - get a vector of the functions that call this one
  // getCallersPending - get a matching vector of bools indicating if each
  //                     caller's DSGraph has been resolved into this one.
  // 
  void addCaller(Function &caller) {
    PendingCallers.insert(&caller);
  }
  std::set<Function*> &getPendingCallers() {
    return PendingCallers;
  }
#endif

  // cloneInto - Clone the specified DSGraph into the current graph, returning
  // the Return node of the graph.  The translated ValueMap for the old function
  // is filled into the OldValMap member.
  // If StripScalars (StripAllocas) is set to true, Scalar (Alloca) markers
  // are removed from the graph as the graph is being cloned.
  // If CopyCallers is set to true, the PendingCallers list is copied.
  // If CopyOrigCalls is set to true, the OrigFunctionCalls list is copied.
  //
  DSNodeHandle cloneInto(const DSGraph &G,
                         std::map<Value*, DSNodeHandle> &OldValMap,
                         std::map<const DSNode*, DSNode*> &OldNodeMap,
                         bool StripScalars = false, bool StripAllocas = false,
                         bool CopyCallers = true, bool CopyOrigCalls = true);

#if 0
  // cloneGlobalInto - Clone the given global node (or the node for the given
  // GlobalValue) from the GlobalsGraph and all its target links (recursively).
  // 
  DSNode* cloneGlobalInto(const DSNode* GNode);
  DSNode* cloneGlobalInto(GlobalValue* GV) {
    assert(!GV || (((DSGraph*) GlobalsGraph)->ValueMap[GV] != 0));
    return GV? cloneGlobalInto(((DSGraph*) GlobalsGraph)->ValueMap[GV]) : 0;
  }
#endif

private:
  bool isNodeDead(DSNode *N);
};

#endif
