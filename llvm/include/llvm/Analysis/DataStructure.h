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
class DSNode;                  // Each node in the graph
class DSGraph;                 // A graph for a function
class DSNodeIterator;          // Data structure graph traversal iterator
class LocalDataStructures;     // A collection of local graphs for a program


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

  // Allow explicit conversion to DSNode...
  DSNode *get() { return N; }
  const DSNode *get() const { return N; }

  // Allow this to be treated like a pointer...
  DSNode *operator->() { return N; }

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

  DSNode(const DSNode &);         // DO NOT IMPLEMENT
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
  };

  // NodeType - A union of the above bits.  "Shadow" nodes do not add any flags
  // to the nodes in the data structure graph, so it is possible to have nodes
  // with a value of 0 for their NodeType.  Scalar and Alloca markers go away
  // when function graphs are inlined.
  //
  unsigned char NodeType;

  DSNode(enum NodeTy NT, const Type *T);
  virtual ~DSNode() {
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

  void print(std::ostream &O, Function *F) const;
  void dump() const;

  std::string getCaption(Function *F) const;

  virtual void dropAllReferences() {
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
  // that refer to the arguments that are passed into the function call.
  //
  std::vector<std::vector<DSNodeHandle> > FunctionCalls;
#if 0
  // cloneFunctionIntoSelf - Clone the specified method graph into the current
  // method graph, returning the Return's set of the graph.  If ValueMap is set
  // to true, the ValueMap of the function is cloned into this function as well
  // as the data structure graph itself.  Regardless, the arguments value sets
  // of DSG are copied into Args.
  //
  PointerValSet cloneFunctionIntoSelf(const DSGraph &G, bool ValueMap,
                                      std::vector<PointerValSet> &Args);

  bool RemoveUnreachableNodes();
  bool UnlinkUndistinguishableNodes();
  void MarkEscapeableNodesReachable(std::vector<bool> &RSN,
                                    std::vector<bool> &RAN);
#endif

private:
  // Define the interface only accessable to DataStructure
  friend class LocalDataStructures;
  DSGraph(Function &F);            // Compute the local DSGraph
  ~DSGraph();

  DSGraph(const DSGraph &DSG);     // DO NOT IMPLEMENT
  void operator=(const DSGraph &); // DO NOT IMPLEMENT
public:

  Function &getFunction() const { return Func; }

#if 0
  // getEscapingAllocations - Add all allocations that escape the current
  // function to the specified vector.
  //
  void getEscapingAllocations(std::vector<AllocDSNode*> &Allocs);

  // getNonEscapingAllocations - Add all allocations that do not escape the
  // current function to the specified vector.
  //
  void getNonEscapingAllocations(std::vector<AllocDSNode*> &Allocs);
#endif

  // getValueMap - Get a map that describes what the nodes the scalars in this
  // function point to...
  //
  std::map<Value*, DSNodeHandle> &getValueMap() { return ValueMap; }
  const std::map<Value*, DSNodeHandle> &getValueMap() const { return ValueMap;}

  const DSNode *getRetNode() const { return RetNode; }

  unsigned getGraphSize() const {
    return Nodes.size();
  }

  void print(std::ostream &O) const;
};



// LocalDataStructures - The analysis that computes the local data structure
// graphs for all of the functions in the program.
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

#endif
