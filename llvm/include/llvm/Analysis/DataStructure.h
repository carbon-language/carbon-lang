//===- DataStructure.h - Build data structure graphs -------------*- C++ -*--=//
//
// Implement the LLVM data structure analysis library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DATA_STRUCTURE_H
#define LLVM_ANALYSIS_DATA_STRUCTURE_H

#include "llvm/Pass.h"
#include <string>

class Type;
class GlobalValue;
class DSNode;                  // Each node in the graph
class DSGraph;                 // A graph for a function
class DSNodeIterator;          // Data structure graph traversal iterator
class LocalDataStructures;     // A collection of local graphs for a program
class BUDataStructures;        // A collection of bu graphs for a program
class TDDataStructures;        // A collection of td graphs for a program

//===----------------------------------------------------------------------===//
// DSNodeHandle - Implement a "handle" to a data structure node that takes care
// of all of the add/un'refing of the node to prevent the backpointers in the
// graph from getting out of date.
//
class DSNodeHandle {
  DSNode *N;
public:
  // Allow construction, destruction, and assignment...
  DSNodeHandle(DSNode *n = 0) : N(0) { operator=(n); }
  DSNodeHandle(const DSNodeHandle &H) : N(0) { operator=(H.N); }
  ~DSNodeHandle() { operator=(0); }
  DSNodeHandle &operator=(const DSNodeHandle &H) {operator=(H.N); return *this;}

  // Assignment of DSNode*, implement all of the add/un'refing (defined later)
  inline DSNodeHandle &operator=(DSNode *n);

  // Allow automatic, implicit, conversion to DSNode*
  operator DSNode*() { return N; }
  operator const DSNode*() const { return N; }
  operator bool() const { return N != 0; }
  operator bool() { return N != 0; }

  bool operator<(const DSNodeHandle &H) const {  // Allow sorting
    return N < H.N;
  }
  bool operator==(const DSNodeHandle &H) const { return N == H.N; }
  bool operator!=(const DSNodeHandle &H) const { return N != H.N; }
  bool operator==(const DSNode *Node) const { return N == Node; }
  bool operator!=(const DSNode *Node) const { return N != Node; }
  bool operator==(DSNode *Node) const { return N == Node; }
  bool operator!=(DSNode *Node) const { return N != Node; }

  // Avoid having comparisons to null cause errors...
  bool operator==(int X) const {
    assert(X == 0 && "Bad comparison!");
    return operator==((DSNode*)0);
  }
  bool operator!=(int X) const { return !operator==(X); }

  // Allow explicit conversion to DSNode...
  DSNode *get() { return N; }
  const DSNode *get() const { return N; }

  // Allow this to be treated like a pointer...
  DSNode *operator->() { return N; }
  const DSNode *operator->() const { return N; }
};


//===----------------------------------------------------------------------===//
// DSNode - Data structure node class
//
// This class keeps track of a node's type, and the fields in the data
// structure.
//
//
class DSNode {
  const Type *Ty;
  std::vector<DSNodeHandle> Links;
  std::vector<DSNodeHandle*> Referrers;

  // Globals - The list of global values that are merged into this node.
  std::vector<GlobalValue*> Globals;

  void operator=(const DSNode &); // DO NOT IMPLEMENT
public:
  enum NodeTy {
    ShadowNode = 0 << 0,   // Nothing is known about this node...
    ScalarNode = 1 << 0,   // Scalar of the current function contains this value
    AllocaNode = 1 << 1,   // This node was allocated with alloca
    NewNode    = 1 << 2,   // This node was allocated with malloc
    GlobalNode = 1 << 3,   // This node was allocated by a global var decl
    SubElement = 1 << 4,   // This node is a part of some other node
    CastNode   = 1 << 5,   // This node is accessed in unsafe ways
    Incomplete = 1 << 6,   // This node may not be complete
  };

  // NodeType - A union of the above bits.  "Shadow" nodes do not add any flags
  // to the nodes in the data structure graph, so it is possible to have nodes
  // with a value of 0 for their NodeType.  Scalar and Alloca markers go away
  // when function graphs are inlined.
  //
  unsigned char NodeType;

  DSNode(enum NodeTy NT, const Type *T);
  DSNode(const DSNode &);

  ~DSNode() {
#ifndef NDEBUG
    dropAllReferences();  // Only needed to satisfy assertion checks...
#endif
    assert(Referrers.empty() && "Referrers to dead node exist!");
  }

  // Iterator for graph interface...
  typedef DSNodeIterator iterator;
  inline iterator begin();   // Defined in DataStructureGraph.h
  inline iterator end();

  // Accessors
  const Type *getType() const { return Ty; }

  unsigned getNumLinks() const { return Links.size(); }
  DSNode *getLink(unsigned i) {
    assert(i < getNumLinks() && "Field links access out of range...");
    return Links[i];
  }
  const DSNode *getLink(unsigned i) const {
    assert(i < getNumLinks() && "Field links access out of range...");
    return Links[i];
  }

  void setLink(unsigned i, DSNode *N) {
    assert(i < getNumLinks() && "Field links access out of range...");
    Links[i] = N;
  }

  // addGlobal - Add an entry for a global value to the Globals list.  This also
  // marks the node with the 'G' flag if it does not already have it.
  //
  void addGlobal(GlobalValue *GV);
  const std::vector<GlobalValue*> &getGlobals() const { return Globals; }
  std::vector<GlobalValue*> &getGlobals() { return Globals; }

  // addEdgeTo - Add an edge from the current node to the specified node.  This
  // can cause merging of nodes in the graph.
  //
  void addEdgeTo(unsigned LinkNo, DSNode *N);
  void addEdgeTo(DSNode *N) {
    assert(getNumLinks() == 1 && "Must specify a field number to add edge if "
           " more than one field exists!");
    addEdgeTo(0, N);
  }

  // mergeWith - Merge this node into the specified node, moving all links to
  // and from the argument node into the current node.  The specified node may
  // be a null pointer (in which case, nothing happens).
  //
  void mergeWith(DSNode *N);

  // addReferrer - Keep the referrer set up to date...
  void addReferrer(DSNodeHandle *H) { Referrers.push_back(H); }
  void removeReferrer(DSNodeHandle *H);
  const std::vector<DSNodeHandle*> &getReferrers() const { return Referrers; }

  void print(std::ostream &O, const DSGraph *G) const;
  void dump() const;

  std::string getCaption(const DSGraph *G) const;

  void dropAllReferences() {
    Links.clear();
  }
};


inline DSNodeHandle &DSNodeHandle::operator=(DSNode *n) {
  if (N) N->removeReferrer(this);
  N = n;
  if (N) N->addReferrer(this);
  return *this;
}


// DSGraph - The graph that represents a function.
//
class DSGraph {
  Function &Func;
  std::vector<DSNode*> Nodes;
  DSNodeHandle RetNode;               // Node that gets returned...
  std::map<Value*, DSNodeHandle> ValueMap;

  // FunctionCalls - This vector maintains a single entry for each call
  // instruction in the current graph.  Each call entry contains DSNodeHandles
  // that refer to the arguments that are passed into the function call.  The
  // first entry in the vector is the scalar that holds the return value for the
  // call, the second is the function scalar being invoked, and the rest are
  // pointer arguments to the function.
  //
  std::vector<std::vector<DSNodeHandle> > FunctionCalls;

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
  std::vector<Function*> PendingCallers;
  
private:
  // Define the interface only accessable to DataStructure
  friend class LocalDataStructures;
  friend class BUDataStructures;
  friend class TDDataStructures;
  DSGraph(Function &F);            // Compute the local DSGraph
  DSGraph(const DSGraph &DSG);     // Copy ctor
  ~DSGraph();

  // clone all the call nodes and save the copies in OrigFunctionCalls
  void saveOrigFunctionCalls() {
    assert(OrigFunctionCalls.size() == 0 && "Do this only once!");
    OrigFunctionCalls = FunctionCalls;
  }
  
  // get the saved copies of the original function call nodes
  std::vector<std::vector<DSNodeHandle> > &getOrigFunctionCalls() {
    return OrigFunctionCalls;
  }

  void operator=(const DSGraph &); // DO NOT IMPLEMENT
public:

  Function &getFunction() const { return Func; }

  // getValueMap - Get a map that describes what the nodes the scalars in this
  // function point to...
  //
  std::map<Value*, DSNodeHandle> &getValueMap() { return ValueMap; }
  const std::map<Value*, DSNodeHandle> &getValueMap() const { return ValueMap;}

  std::vector<std::vector<DSNodeHandle> > &getFunctionCalls() {
    return FunctionCalls;
  }

  const DSNode *getRetNode() const { return RetNode; }

  unsigned getGraphSize() const {
    return Nodes.size();
  }

  void print(std::ostream &O) const;
  void dump() const;

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
  void markIncompleteNodes();

  // removeTriviallyDeadNodes - After the graph has been constructed, this
  // method removes all unreachable nodes that are created because they got
  // merged with other nodes in the graph.
  //
  void removeTriviallyDeadNodes();

  // removeDeadNodes - Use a more powerful reachability analysis to eliminate
  // subgraphs that are unreachable.  This often occurs because the data
  // structure doesn't "escape" into it's caller, and thus should be eliminated
  // from the caller's graph entirely.  This is only appropriate to use when
  // inlining graphs.
  //
  void removeDeadNodes();


  // AddCaller - add a known caller node into the graph and mark it pending.
  // getCallers - get a vector of the functions that call this one
  // getCallersPending - get a matching vector of bools indicating if each
  //                     caller's DSGraph has been resolved into this one.
  // 
  void addCaller(Function& caller) {
    PendingCallers.push_back(&caller);
  }
  std::vector<Function*>& getPendingCallers() {
    return PendingCallers;
  }
  
  // cloneInto - Clone the specified DSGraph into the current graph, returning
  // the Return node of the graph.  The translated ValueMap for the old function
  // is filled into the OldValMap member.  If StripLocals is set to true, Scalar
  // and Alloca markers are removed from the graph, as the graph is being cloned
  // into a calling function's graph.
  //
  DSNode *cloneInto(const DSGraph &G, std::map<Value*, DSNodeHandle> &OldValMap,
                    std::map<const DSNode*, DSNode*>& OldNodeMap,
                    bool StripLocals = true);
private:
  bool isNodeDead(DSNode *N);
};



// LocalDataStructures - The analysis that computes the local data structure
// graphs for all of the functions in the program.
//
// FIXME: This should be a Function pass that can be USED by a Pass, and would
// be automatically preserved.  Until we can do that, this is a Pass.
//
class LocalDataStructures : public Pass {
  // DSInfo, one graph for each function
  std::map<Function*, DSGraph*> DSInfo;
public:
  static AnalysisID ID;            // DataStructure Analysis ID 

  LocalDataStructures(AnalysisID id) { assert(id == ID); }
  ~LocalDataStructures() { releaseMemory(); }

  virtual const char *getPassName() const {
    return "Local Data Structure Analysis";
  }

  virtual bool run(Module &M);

  // getDSGraph - Return the data structure graph for the specified function.
  DSGraph &getDSGraph(Function &F) const {
    std::map<Function*, DSGraph*>::const_iterator I = DSInfo.find(&F);
    assert(I != DSInfo.end() && "Function not in module!");
    return *I->second;
  }

  // print - Print out the analysis results...
  void print(std::ostream &O, Module *M) const;

  // If the pass pipeline is done with this pass, we can release our memory...
  virtual void releaseMemory();

  // getAnalysisUsage - This obviously provides a data structure graph.
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addProvided(ID);
  }
};


// BUDataStructures - The analysis that computes the interprocedurally closed
// data structure graphs for all of the functions in the program.  This pass
// only performs a "Bottom Up" propogation (hence the name).
//
class BUDataStructures : public Pass {
  // DSInfo, one graph for each function
  std::map<Function*, DSGraph*> DSInfo;
public:
  static AnalysisID ID;            // BUDataStructure Analysis ID 

  BUDataStructures(AnalysisID id) { assert(id == ID); }
  ~BUDataStructures() { releaseMemory(); }

  virtual const char *getPassName() const {
    return "Bottom-Up Data Structure Analysis Closure";
  }

  virtual bool run(Module &M);

  // getDSGraph - Return the data structure graph for the specified function.
  DSGraph &getDSGraph(Function &F) const {
    std::map<Function*, DSGraph*>::const_iterator I = DSInfo.find(&F);
    assert(I != DSInfo.end() && "Function not in module!");
    return *I->second;
  }
  
  // print - Print out the analysis results...
  void print(std::ostream &O, Module *M) const;

  // If the pass pipeline is done with this pass, we can release our memory...
  virtual void releaseMemory();

  // getAnalysisUsage - This obviously provides a data structure graph.
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addProvided(ID);
    AU.addRequired(LocalDataStructures::ID);
  }
private:
  DSGraph &calculateGraph(Function &F);
};


// TDDataStructures - Analysis that computes new data structure graphs
// for each function using the closed graphs for the callers computed
// by the bottom-up pass.
//
class TDDataStructures : public Pass {
  // DSInfo, one graph for each function
  std::map<Function*, DSGraph*> DSInfo;
public:
  static AnalysisID ID;            // TDDataStructure Analysis ID 

  TDDataStructures(AnalysisID id) { assert(id == ID); }
  ~TDDataStructures() { releaseMemory(); }

  virtual const char *getPassName() const {
    return "Top-down Data Structure Analysis Closure";
  }

  virtual bool run(Module &M);

  // getDSGraph - Return the data structure graph for the specified function.
  DSGraph &getDSGraph(Function &F) const {
    std::map<Function*, DSGraph*>::const_iterator I = DSInfo.find(&F);
    assert(I != DSInfo.end() && "Function not in module!");
    return *I->second;
  }

  // print - Print out the analysis results...
  void print(std::ostream &O, Module *M) const;

  // If the pass pipeline is done with this pass, we can release our memory...
  virtual void releaseMemory();

  // getAnalysisUsage - This obviously provides a data structure graph.
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addProvided(ID);
    AU.addRequired(BUDataStructures::ID);
  }
private:
  DSGraph &calculateGraph(Function &F);
  void pushGraphIntoCallee(DSGraph &callerGraph, DSGraph &calleeGraph,
                           std::map<Value*, DSNodeHandle> &OldValMap,
                           std::map<const DSNode*, DSNode*> &OldNodeMap);
};
#endif
