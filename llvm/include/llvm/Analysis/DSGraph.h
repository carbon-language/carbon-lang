//===- DSGraph.h - Represent a collection of data structures ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This header defines the data structure graph (DSGraph) and the
// ReachabilityCloner class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DSGRAPH_H
#define LLVM_ANALYSIS_DSGRAPH_H

#include "llvm/Analysis/DSNode.h"

namespace llvm {

class GlobalValue;

//===----------------------------------------------------------------------===//
/// DSScalarMap - An instance of this class is used to keep track of all of 
/// which DSNode each scalar in a function points to.  This is specialized to
/// keep track of globals with nodes in the function, and to keep track of the 
/// unique DSNodeHandle being used by the scalar map.
///
/// This class is crucial to the efficiency of DSA with some large SCC's.  In 
/// these cases, the cost of iterating over the scalar map dominates the cost
/// of DSA.  In all of these cases, the DSA phase is really trying to identify 
/// globals or unique node handles active in the function.
///
class DSScalarMap {
  typedef hash_map<Value*, DSNodeHandle> ValueMapTy;
  ValueMapTy ValueMap;

  typedef hash_set<GlobalValue*> GlobalSetTy;
  GlobalSetTy GlobalSet;
public:

  // Compatibility methods: provide an interface compatible with a map of 
  // Value* to DSNodeHandle's.
  typedef ValueMapTy::const_iterator const_iterator;
  typedef ValueMapTy::iterator iterator;
  iterator begin() { return ValueMap.begin(); }
  iterator end()   { return ValueMap.end(); }
  const_iterator begin() const { return ValueMap.begin(); }
  const_iterator end() const { return ValueMap.end(); }
  iterator find(Value *V) { return ValueMap.find(V); }
  const_iterator find(Value *V) const { return ValueMap.find(V); }
  unsigned count(Value *V) const { return ValueMap.count(V); }

  void erase(Value *V) { erase(find(V)); }

  /// replaceScalar - When an instruction needs to be modified, this method can
  /// be used to update the scalar map to remove the old and insert the new.
  void replaceScalar(Value *Old, Value *New) {
    iterator I = find(Old);
    assert(I != end() && "Old value is not in the map!");
    ValueMap.insert(std::make_pair(New, I->second));
    erase(I);
  }

  DSNodeHandle &operator[](Value *V) {
    std::pair<iterator,bool> IP = 
      ValueMap.insert(std::make_pair(V, DSNodeHandle()));
    if (IP.second) {  // Inserted the new entry into the map.
      if (GlobalValue *GV = dyn_cast<GlobalValue>(V))
        GlobalSet.insert(GV);
    }
    return IP.first->second;
  }

  void erase(iterator I) { 
    assert(I != ValueMap.end() && "Cannot erase end!");
    if (GlobalValue *GV = dyn_cast<GlobalValue>(I->first))
      GlobalSet.erase(GV);
    ValueMap.erase(I); 
  }

  void clear() {
    ValueMap.clear();
    GlobalSet.clear();
  }

  // Access to the global set: the set of all globals currently in the
  // scalar map.
  typedef GlobalSetTy::const_iterator global_iterator;
  global_iterator global_begin() const { return GlobalSet.begin(); }
  global_iterator global_end() const { return GlobalSet.end(); }
};


//===----------------------------------------------------------------------===//
/// DSGraph - The graph that represents a function.
///
struct DSGraph {
  // Public data-type declarations...
  typedef DSScalarMap ScalarMapTy;
  typedef hash_map<Function*, DSNodeHandle> ReturnNodesTy;
  typedef hash_set<GlobalValue*> GlobalSetTy;
  typedef ilist<DSNode> NodeListTy;

  /// NodeMapTy - This data type is used when cloning one graph into another to
  /// keep track of the correspondence between the nodes in the old and new
  /// graphs.
  typedef hash_map<const DSNode*, DSNodeHandle> NodeMapTy;
private:
  DSGraph *GlobalsGraph;   // Pointer to the common graph of global objects
  bool PrintAuxCalls;      // Should this graph print the Aux calls vector?

  NodeListTy Nodes;
  ScalarMapTy ScalarMap;

  // ReturnNodes - A return value for every function merged into this graph.
  // Each DSGraph may have multiple functions merged into it at any time, which
  // is used for representing SCCs.
  //
  ReturnNodesTy ReturnNodes;

  // FunctionCalls - This vector maintains a single entry for each call
  // instruction in the current graph.  The first entry in the vector is the
  // scalar that holds the return value for the call, the second is the function
  // scalar being invoked, and the rest are pointer arguments to the function.
  // This vector is built by the Local graph and is never modified after that.
  //
  std::vector<DSCallSite> FunctionCalls;

  // AuxFunctionCalls - This vector contains call sites that have been processed
  // by some mechanism.  In pratice, the BU Analysis uses this vector to hold
  // the _unresolved_ call sites, because it cannot modify FunctionCalls.
  //
  std::vector<DSCallSite> AuxFunctionCalls;

  // InlinedGlobals - This set records which globals have been inlined from
  // other graphs (callers or callees, depending on the pass) into this one.
  // 
  GlobalSetTy InlinedGlobals;

  /// TD - This is the target data object for the machine this graph is
  /// constructed for.
  const TargetData &TD;

  void operator=(const DSGraph &); // DO NOT IMPLEMENT

public:
  // Create a new, empty, DSGraph.
  DSGraph(const TargetData &td)
    : GlobalsGraph(0), PrintAuxCalls(false), TD(td) {}

  // Compute the local DSGraph
  DSGraph(const TargetData &td, Function &F, DSGraph *GlobalsGraph);

  // Copy ctor - If you want to capture the node mapping between the source and
  // destination graph, you may optionally do this by specifying a map to record
  // this into.
  //
  // Note that a copied graph does not retain the GlobalsGraph pointer of the
  // source.  You need to set a new GlobalsGraph with the setGlobalsGraph
  // method.
  //
  DSGraph(const DSGraph &DSG);
  DSGraph(const DSGraph &DSG, NodeMapTy &NodeMap);
  ~DSGraph();

  DSGraph *getGlobalsGraph() const { return GlobalsGraph; }
  void setGlobalsGraph(DSGraph *G) { GlobalsGraph = G; }

  /// getTargetData - Return the TargetData object for the current target.
  ///
  const TargetData &getTargetData() const { return TD; }

  /// setPrintAuxCalls - If you call this method, the auxillary call vector will
  /// be printed instead of the standard call vector to the dot file.
  ///
  void setPrintAuxCalls() { PrintAuxCalls = true; }
  bool shouldPrintAuxCalls() const { return PrintAuxCalls; }

  /// node_iterator/begin/end - Iterate over all of the nodes in the graph.  Be
  /// extremely careful with these methods because any merging of nodes could
  /// cause the node to be removed from this list.  This means that if you are
  /// iterating over nodes and doing something that could cause _any_ node to
  /// merge, your node_iterators into this graph can be invalidated.
  typedef NodeListTy::compat_iterator node_iterator;
  node_iterator node_begin() const { return Nodes.compat_begin(); }
  node_iterator node_end()   const { return Nodes.compat_end(); }

  /// getFunctionNames - Return a space separated list of the name of the
  /// functions in this graph (if any)
  std::string getFunctionNames() const;

  /// addNode - Add a new node to the graph.
  ///
  void addNode(DSNode *N) { Nodes.push_back(N); }
  void unlinkNode(DSNode *N) { Nodes.remove(N); }

  /// getScalarMap - Get a map that describes what the nodes the scalars in this
  /// function point to...
  ///
  ScalarMapTy &getScalarMap() { return ScalarMap; }
  const ScalarMapTy &getScalarMap() const { return ScalarMap; }

  /// getFunctionCalls - Return the list of call sites in the original local
  /// graph...
  ///
  const std::vector<DSCallSite> &getFunctionCalls() const {
    return FunctionCalls;
  }

  /// getAuxFunctionCalls - Get the call sites as modified by whatever passes
  /// have been run.
  ///
  std::vector<DSCallSite> &getAuxFunctionCalls() {
    return AuxFunctionCalls;
  }
  const std::vector<DSCallSite> &getAuxFunctionCalls() const {
    return AuxFunctionCalls;
  }

  /// getInlinedGlobals - Get the set of globals that are have been inlined
  /// (from callees in BU or from callers in TD) into the current graph.
  ///
  GlobalSetTy& getInlinedGlobals() {
    return InlinedGlobals;
  }

  /// getNodeForValue - Given a value that is used or defined in the body of the
  /// current function, return the DSNode that it points to.
  ///
  DSNodeHandle &getNodeForValue(Value *V) { return ScalarMap[V]; }

  const DSNodeHandle &getNodeForValue(Value *V) const {
    ScalarMapTy::const_iterator I = ScalarMap.find(V);
    assert(I != ScalarMap.end() &&
           "Use non-const lookup function if node may not be in the map");
    return I->second;
  }

  /// getReturnNodes - Return the mapping of functions to their return nodes for
  /// this graph.
  const ReturnNodesTy &getReturnNodes() const { return ReturnNodes; }
        ReturnNodesTy &getReturnNodes()       { return ReturnNodes; }

  /// getReturnNodeFor - Return the return node for the specified function.
  ///
  DSNodeHandle &getReturnNodeFor(Function &F) {
    ReturnNodesTy::iterator I = ReturnNodes.find(&F);
    assert(I != ReturnNodes.end() && "F not in this DSGraph!");
    return I->second;
  }

  const DSNodeHandle &getReturnNodeFor(Function &F) const {
    ReturnNodesTy::const_iterator I = ReturnNodes.find(&F);
    assert(I != ReturnNodes.end() && "F not in this DSGraph!");
    return I->second;
  }

  /// getGraphSize - Return the number of nodes in this graph.
  ///
  unsigned getGraphSize() const {
    return Nodes.size();
  }

  /// print - Print a dot graph to the specified ostream...
  ///
  void print(std::ostream &O) const;

  /// dump - call print(std::cerr), for use from the debugger...
  ///
  void dump() const;

  /// viewGraph - Emit a dot graph, run 'dot', run gv on the postscript file,
  /// then cleanup.  For use from the debugger.
  void viewGraph() const;

  void writeGraphToFile(std::ostream &O, const std::string &GraphName) const;

  /// maskNodeTypes - Apply a mask to all of the node types in the graph.  This
  /// is useful for clearing out markers like Incomplete.
  ///
  void maskNodeTypes(unsigned Mask) {
    for (node_iterator I = node_begin(), E = node_end(); I != E; ++I)
      (*I)->maskNodeTypes(Mask);
  }
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
  enum MarkIncompleteFlags {
    MarkFormalArgs = 1, IgnoreFormalArgs = 0,
    IgnoreGlobals = 2, MarkGlobalsIncomplete = 0,
  };
  void markIncompleteNodes(unsigned Flags);

  // removeDeadNodes - Use a reachability analysis to eliminate subgraphs that
  // are unreachable.  This often occurs because the data structure doesn't
  // "escape" into it's caller, and thus should be eliminated from the caller's
  // graph entirely.  This is only appropriate to use when inlining graphs.
  //
  enum RemoveDeadNodesFlags {
    RemoveUnreachableGlobals = 1, KeepUnreachableGlobals = 0,
  };
  void removeDeadNodes(unsigned Flags);

  /// CloneFlags enum - Bits that may be passed into the cloneInto method to
  /// specify how to clone the function graph.
  enum CloneFlags {
    StripAllocaBit        = 1 << 0, KeepAllocaBit     = 0,
    DontCloneCallNodes    = 1 << 1, CloneCallNodes    = 0,
    DontCloneAuxCallNodes = 1 << 2, CloneAuxCallNodes = 0,
    StripModRefBits       = 1 << 3, KeepModRefBits    = 0,
    StripIncompleteBit    = 1 << 4, KeepIncompleteBit = 0,
    UpdateInlinedGlobals  = 1 << 5, DontUpdateInlinedGlobals = 0,
  };

  void updateFromGlobalGraph();

  /// computeNodeMapping - Given roots in two different DSGraphs, traverse the
  /// nodes reachable from the two graphs, computing the mapping of nodes from
  /// the first to the second graph.
  ///
  static void computeNodeMapping(const DSNodeHandle &NH1,
                                 const DSNodeHandle &NH2, NodeMapTy &NodeMap,
                                 bool StrictChecking = true);


  /// cloneInto - Clone the specified DSGraph into the current graph.  The
  /// translated ScalarMap for the old function is filled into the OldValMap
  /// member, and the translated ReturnNodes map is returned into ReturnNodes.
  /// OldNodeMap contains a mapping from the original nodes to the newly cloned
  /// nodes.
  ///
  /// The CloneFlags member controls various aspects of the cloning process.
  ///
  void cloneInto(const DSGraph &G, ScalarMapTy &OldValMap,
                 ReturnNodesTy &OldReturnNodes, NodeMapTy &OldNodeMap,
                 unsigned CloneFlags = 0);

  /// mergeInGraph - The method is used for merging graphs together.  If the
  /// argument graph is not *this, it makes a clone of the specified graph, then
  /// merges the nodes specified in the call site with the formal arguments in
  /// the graph.  If the StripAlloca's argument is 'StripAllocaBit' then Alloca
  /// markers are removed from nodes.
  ///
  void mergeInGraph(const DSCallSite &CS, Function &F, const DSGraph &Graph,
                    unsigned CloneFlags);


  /// getCallSiteForArguments - Get the arguments and return value bindings for
  /// the specified function in the current graph.
  ///
  DSCallSite getCallSiteForArguments(Function &F) const;

  /// getDSCallSiteForCallSite - Given an LLVM CallSite object that is live in
  /// the context of this graph, return the DSCallSite for it.
  DSCallSite getDSCallSiteForCallSite(CallSite CS) const;

  // Methods for checking to make sure graphs are well formed...
  void AssertNodeInGraph(const DSNode *N) const {
    assert((!N || N->getParentGraph() == this) &&
           "AssertNodeInGraph: Node is not in graph!");
  }
  void AssertNodeContainsGlobal(const DSNode *N, GlobalValue *GV) const {
    assert(std::find(N->getGlobals().begin(), N->getGlobals().end(), GV) !=
           N->getGlobals().end() && "Global value not in node!");
  }

  void AssertCallSiteInGraph(const DSCallSite &CS) const;
  void AssertCallNodesInGraph() const;
  void AssertAuxCallNodesInGraph() const;

  void AssertGraphOK() const;

  /// removeTriviallyDeadNodes - After the graph has been constructed, this
  /// method removes all unreachable nodes that are created because they got
  /// merged with other nodes in the graph.  This is used as the first step of
  /// removeDeadNodes.
  ///
  void removeTriviallyDeadNodes();
};


  /// ReachabilityCloner - This class is used to incrementally clone and merge
  /// nodes from a non-changing source graph into a potentially mutating
  /// destination graph.  Nodes are only cloned over on demand, either in
  /// responds to a merge() or getClonedNH() call.  When a node is cloned over,
  /// all of the nodes reachable from it are automatically brought over as well.
  class ReachabilityCloner {
    DSGraph &Dest;
    const DSGraph &Src;

    /// BitsToKeep - These bits are retained from the source node when the
    /// source nodes are merged into the destination graph.
    unsigned BitsToKeep;
    unsigned CloneFlags;

    // NodeMap - A mapping from nodes in the source graph to the nodes that
    // represent them in the destination graph.
    DSGraph::NodeMapTy NodeMap;
  public:
    ReachabilityCloner(DSGraph &dest, const DSGraph &src, unsigned cloneFlags)
      : Dest(dest), Src(src), CloneFlags(cloneFlags) {
      assert(&Dest != &Src && "Cannot clone from graph to same graph!");
      BitsToKeep = ~DSNode::DEAD;
      if (CloneFlags & DSGraph::StripAllocaBit)
        BitsToKeep &= ~DSNode::AllocaNode;
      if (CloneFlags & DSGraph::StripModRefBits)
        BitsToKeep &= ~(DSNode::Modified | DSNode::Read);
      if (CloneFlags & DSGraph::StripIncompleteBit)
        BitsToKeep &= ~DSNode::Incomplete;
    }
    
    DSNodeHandle getClonedNH(const DSNodeHandle &SrcNH);

    void merge(const DSNodeHandle &NH, const DSNodeHandle &SrcNH);

    /// mergeCallSite - Merge the nodes reachable from the specified src call
    /// site into the nodes reachable from DestCS.
    void mergeCallSite(const DSCallSite &DestCS, const DSCallSite &SrcCS);

    bool clonedAnyNodes() const { return !NodeMap.empty(); }

    /// hasClonedNode - Return true if the specified node has been cloned from
    /// the source graph into the destination graph.
    bool hasClonedNode(const DSNode *N) {
      return NodeMap.count(N);
    }

    void destroy() { NodeMap.clear(); }
  };
} // End llvm namespace

#endif
