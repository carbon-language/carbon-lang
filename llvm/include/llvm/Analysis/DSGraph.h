//===- DSGraph.h - Represent a collection of data structures ----*- C++ -*-===//
//
// This header defines the data structure graph.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DSGRAPH_H
#define LLVM_ANALYSIS_DSGRAPH_H

#include "llvm/Analysis/DSNode.h"

//===----------------------------------------------------------------------===//
/// DSGraph - The graph that represents a function.
///
class DSGraph {
  Function *Func;
  std::vector<DSNode*> Nodes;
  DSNodeHandle RetNode;                          // Node that gets returned...
  std::map<Value*, DSNodeHandle> ScalarMap;

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
  std::vector<DSCallSite> FunctionCalls;

  void operator=(const DSGraph &); // DO NOT IMPLEMENT
public:
  DSGraph() : Func(0) {}           // Create a new, empty, DSGraph.
  DSGraph(Function &F);            // Compute the local DSGraph

  // Copy ctor - If you want to capture the node mapping between the source and
  // destination graph, you may optionally do this by specifying a map to record
  // this into.
  DSGraph(const DSGraph &DSG);
  DSGraph(const DSGraph &DSG, std::map<const DSNode*, DSNode*> &BUNodeMap);
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

  /// getScalarMap - Get a map that describes what the nodes the scalars in this
  /// function point to...
  ///
  std::map<Value*, DSNodeHandle> &getScalarMap() { return ScalarMap; }
  const std::map<Value*, DSNodeHandle> &getScalarMap() const {return ScalarMap;}

  std::vector<DSCallSite> &getFunctionCalls() {
    return FunctionCalls;
  }
  const std::vector<DSCallSite> &getFunctionCalls() const {
    return FunctionCalls;
  }

  /// getNodeForValue - Given a value that is used or defined in the body of the
  /// current function, return the DSNode that it points to.
  ///
  DSNodeHandle &getNodeForValue(Value *V) { return ScalarMap[V]; }

  const DSNodeHandle &getRetNode() const { return RetNode; }
        DSNodeHandle &getRetNode()       { return RetNode; }

  unsigned getGraphSize() const {
    return Nodes.size();
  }

  void print(std::ostream &O) const;
  void dump() const;
  void writeGraphToFile(std::ostream &O, const std::string &GraphName) const;

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

  // cloneInto - Clone the specified DSGraph into the current graph, returning
  // the Return node of the graph.  The translated ScalarMap for the old
  // function is filled into the OldValMap member.  If StripScalars
  // (StripAllocas) is set to true, Scalar (Alloca) markers are removed from the
  // graph as the graph is being cloned.
  //
  DSNodeHandle cloneInto(const DSGraph &G,
                         std::map<Value*, DSNodeHandle> &OldValMap,
                         std::map<const DSNode*, DSNode*> &OldNodeMap,
                         bool StripScalars = false, bool StripAllocas = false);

#if 0
  // cloneGlobalInto - Clone the given global node (or the node for the given
  // GlobalValue) from the GlobalsGraph and all its target links (recursively).
  // 
  DSNode* cloneGlobalInto(const DSNode* GNode);
  DSNode* cloneGlobalInto(GlobalValue* GV) {
    assert(!GV || (((DSGraph*) GlobalsGraph)->ScalarMap[GV] != 0));
    return GV? cloneGlobalInto(((DSGraph*) GlobalsGraph)->ScalarMap[GV]) : 0;
  }
#endif

private:
  bool isNodeDead(DSNode *N);
};

#endif
