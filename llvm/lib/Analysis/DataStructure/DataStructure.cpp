//===- DataStructure.cpp - Implement the core data structure analysis -----===//
//
// This file implements the core data structure functionality.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "Support/STLExtras.h"
#include "Support/StatisticReporter.h"
#include "Support/STLExtras.h"
#include <algorithm>
#include <set>
#include "llvm/Analysis/DataStructure.h"

using std::vector;

//===----------------------------------------------------------------------===//
// DSNode Implementation
//===----------------------------------------------------------------------===//

DSNode::DSNode(enum NodeTy NT, const Type *T) : Ty(T), NodeType(NT) {
  // If this node has any fields, allocate them now, but leave them null.
  switch (T->getPrimitiveID()) {
  case Type::PointerTyID: Links.resize(1); break;
  case Type::ArrayTyID:   Links.resize(1); break;
  case Type::StructTyID:
    Links.resize(cast<StructType>(T)->getNumContainedTypes());
    break;
  default: break;
  }
}

// DSNode copy constructor... do not copy over the referrers list!
DSNode::DSNode(const DSNode &N)
  : Ty(N.Ty), Links(N.Links), Globals(N.Globals), NodeType(N.NodeType) {
}

void DSNode::removeReferrer(DSNodeHandle *H) {
  // Search backwards, because we depopulate the list from the back for
  // efficiency (because it's a vector).
  vector<DSNodeHandle*>::reverse_iterator I =
    std::find(Referrers.rbegin(), Referrers.rend(), H);
  assert(I != Referrers.rend() && "Referrer not pointing to node!");
  Referrers.erase(I.base()-1);
}

// addGlobal - Add an entry for a global value to the Globals list.  This also
// marks the node with the 'G' flag if it does not already have it.
//
void DSNode::addGlobal(GlobalValue *GV) {
  // Keep the list sorted.
  vector<GlobalValue*>::iterator I =
    std::lower_bound(Globals.begin(), Globals.end(), GV);

  if (I == Globals.end() || *I != GV) {
    assert(GV->getType()->getElementType() == Ty);
    Globals.insert(I, GV);
    NodeType |= GlobalNode;
  }
}


// addEdgeTo - Add an edge from the current node to the specified node.  This
// can cause merging of nodes in the graph.
//
void DSNode::addEdgeTo(unsigned LinkNo, DSNode *N) {
  assert(LinkNo < Links.size() && "LinkNo out of range!");
  if (N == 0 || Links[LinkNo] == N) return;  // Nothing to do
  if (Links[LinkNo] == 0) {                  // No merging to perform
    Links[LinkNo] = N;
    return;
  }

  // Merge the two nodes...
  Links[LinkNo]->mergeWith(N);
}


// mergeWith - Merge this node into the specified node, moving all links to and
// from the argument node into the current node.  The specified node may be a
// null pointer (in which case, nothing happens).
//
void DSNode::mergeWith(DSNode *N) {
  if (N == 0 || N == this) return;  // Noop
  assert(N->Ty == Ty && N->Links.size() == Links.size() &&
         "Cannot merge nodes of two different types!");

  // Remove all edges pointing at N, causing them to point to 'this' instead.
  while (!N->Referrers.empty())
    *N->Referrers.back() = this;

  // Make all of the outgoing links of N now be outgoing links of this.  This
  // can cause recursive merging!
  //
  for (unsigned i = 0, e = Links.size(); i != e; ++i) {
    addEdgeTo(i, N->Links[i]);
    N->Links[i] = 0;  // Reduce unneccesary edges in graph. N is dead
  }

  // Merge the node types
  NodeType |= N->NodeType;
  N->NodeType = 0;   // N is now a dead node.

  // Merge the globals list...
  if (!N->Globals.empty()) {
    // Save the current globals off to the side...
    vector<GlobalValue*> OldGlobals(Globals);

    // Resize the globals vector to be big enough to hold both of them...
    Globals.resize(Globals.size()+N->Globals.size());

    // Merge the two sorted globals lists together...
    std::merge(OldGlobals.begin(), OldGlobals.end(),
               N->Globals.begin(), N->Globals.end(), Globals.begin());

    // Erase duplicate entries from the globals list...
    Globals.erase(std::unique(Globals.begin(), Globals.end()), Globals.end());

    // Delete the globals from the old node...
    N->Globals.clear();
  }
}

//===----------------------------------------------------------------------===//
// DSGraph Implementation
//===----------------------------------------------------------------------===//

DSGraph::DSGraph(const DSGraph &G) : Func(G.Func) {
  std::map<const DSNode*, DSNode*> NodeMap; // ignored
  RetNode = cloneInto(G, ValueMap, NodeMap, false);
}

DSGraph::~DSGraph() {
  FunctionCalls.clear();
  OrigFunctionCalls.clear();
  ValueMap.clear();
  RetNode = 0;

#ifndef NDEBUG
  // Drop all intra-node references, so that assertions don't fail...
  std::for_each(Nodes.begin(), Nodes.end(),
                std::mem_fun(&DSNode::dropAllReferences));
#endif

  // Delete all of the nodes themselves...
  std::for_each(Nodes.begin(), Nodes.end(), deleter<DSNode>);
}

// dump - Allow inspection of graph in a debugger.
void DSGraph::dump() const { print(std::cerr); }


// cloneInto - Clone the specified DSGraph into the current graph, returning the
// Return node of the graph.  The translated ValueMap for the old function is
// filled into the OldValMap member.  If StripLocals is set to true, Scalar and
// Alloca markers are removed from the graph, as the graph is being cloned into
// a calling function's graph.
//
DSNode *DSGraph::cloneInto(const DSGraph &G, 
                           std::map<Value*, DSNodeHandle> &OldValMap,
                           std::map<const DSNode*, DSNode*> &OldNodeMap,
                           bool StripLocals) {

  assert(OldNodeMap.size()==0 && "Return argument OldNodeMap should be empty");

  OldNodeMap[0] = 0;  // Null pointer maps to null

  unsigned FN = Nodes.size();  // FirstNode...

  // Duplicate all of the nodes, populating the node map...
  Nodes.reserve(FN+G.Nodes.size());
  for (unsigned i = 0, e = G.Nodes.size(); i != e; ++i) {
    DSNode *Old = G.Nodes[i], *New = new DSNode(*Old);
    Nodes.push_back(New);
    OldNodeMap[Old] = New;
  }

  // Rewrite the links in the nodes to point into the current graph now.
  for (unsigned i = FN, e = Nodes.size(); i != e; ++i)
    for (unsigned j = 0, e = Nodes[i]->getNumLinks(); j != e; ++j)
      Nodes[i]->setLink(j, OldNodeMap[Nodes[i]->getLink(j)]);

  // If we are inlining this graph into the called function graph, remove local
  // markers.
  if (StripLocals)
    for (unsigned i = FN, e = Nodes.size(); i != e; ++i)
      Nodes[i]->NodeType &= ~(DSNode::AllocaNode | DSNode::ScalarNode);

  // Copy the value map...
  for (std::map<Value*, DSNodeHandle>::const_iterator I = G.ValueMap.begin(),
         E = G.ValueMap.end(); I != E; ++I)
    OldValMap[I->first] = OldNodeMap[I->second];

  // Copy the function calls list...
  unsigned FC = FunctionCalls.size();  // FirstCall
  FunctionCalls.reserve(FC+G.FunctionCalls.size());
  for (unsigned i = 0, e = G.FunctionCalls.size(); i != e; ++i) {
    FunctionCalls.push_back(std::vector<DSNodeHandle>());
    FunctionCalls[FC+i].reserve(G.FunctionCalls[i].size());
    for (unsigned j = 0, e = G.FunctionCalls[i].size(); j != e; ++j)
      FunctionCalls[FC+i].push_back(OldNodeMap[G.FunctionCalls[i][j]]);
  }

  // Copy the list of unresolved callers
  PendingCallers.insert(PendingCallers.end(),
                        G.PendingCallers.begin(), G.PendingCallers.end());

  // Return the returned node pointer...
  return OldNodeMap[G.RetNode];
}


// markIncompleteNodes - Mark the specified node as having contents that are not
// known with the current analysis we have performed.  Because a node makes all
// of the nodes it can reach imcomplete if the node itself is incomplete, we
// must recursively traverse the data structure graph, marking all reachable
// nodes as incomplete.
//
static void markIncompleteNode(DSNode *N) {
  // Stop recursion if no node, or if node already marked...
  if (N == 0 || (N->NodeType & DSNode::Incomplete)) return;

  // Actually mark the node
  N->NodeType |= DSNode::Incomplete;

  // Recusively process children...
  for (unsigned i = 0, e = N->getNumLinks(); i != e; ++i)
    markIncompleteNode(N->getLink(i));
}


// markIncompleteNodes - Traverse the graph, identifying nodes that may be
// modified by other functions that have not been resolved yet.  This marks
// nodes that are reachable through three sources of "unknownness":
//
//  Global Variables, Function Calls, and Incoming Arguments
//
// For any node that may have unknown components (because something outside the
// scope of current analysis may have modified it), the 'Incomplete' flag is
// added to the NodeType.
//
void DSGraph::markIncompleteNodes() {
  // Mark any incoming arguments as incomplete...
  for (Function::aiterator I = Func.abegin(), E = Func.aend(); I != E; ++I)
    if (isa<PointerType>(I->getType()))
      markIncompleteNode(ValueMap[I]->getLink(0));

  // Mark stuff passed into functions calls as being incomplete...
  for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i) {
    vector<DSNodeHandle> &Args = FunctionCalls[i];
    // Then the return value is certainly incomplete!
    markIncompleteNode(Args[0]);

    // The call does not make the function argument incomplete...
 
    // All arguments to the function call are incomplete though!
    for (unsigned i = 2, e = Args.size(); i != e; ++i)
      markIncompleteNode(Args[i]);
  }

  // Mark all of the nodes pointed to by global or cast nodes as incomplete...
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
    if (Nodes[i]->NodeType & (DSNode::GlobalNode | DSNode::CastNode)) {
      DSNode *N = Nodes[i];
      for (unsigned i = 0, e = N->getNumLinks(); i != e; ++i)
        markIncompleteNode(N->getLink(i));
    }
}

// isNodeDead - This method checks to see if a node is dead, and if it isn't, it
// checks to see if there are simple transformations that it can do to make it
// dead.
//
bool DSGraph::isNodeDead(DSNode *N) {
  // Is it a trivially dead shadow node...
  if (N->getReferrers().empty() && N->NodeType == 0)
    return true;

  // Is it a function node or some other trivially unused global?
  if ((N->NodeType & ~DSNode::GlobalNode) == 0 && 
      N->getNumLinks() == 0 &&
      N->getReferrers().size() == N->getGlobals().size()) {

    // Remove the globals from the valuemap, so that the referrer count will go
    // down to zero.
    while (!N->getGlobals().empty()) {
      GlobalValue *GV = N->getGlobals().back();
      N->getGlobals().pop_back();      
      ValueMap.erase(GV);
    }
    assert(N->getReferrers().empty() && "Referrers should all be gone now!");
    return true;
  }

  return false;
}


// removeTriviallyDeadNodes - After the graph has been constructed, this method
// removes all unreachable nodes that are created because they got merged with
// other nodes in the graph.  These nodes will all be trivially unreachable, so
// we don't have to perform any non-trivial analysis here.
//
void DSGraph::removeTriviallyDeadNodes() {
  for (unsigned i = 0; i != Nodes.size(); ++i)
    if (isNodeDead(Nodes[i])) {               // This node is dead!
      delete Nodes[i];                        // Free memory...
      Nodes.erase(Nodes.begin()+i--);         // Remove from node list...
    }

  // Remove trivially identical function calls
  unsigned NumFns = FunctionCalls.size();
  std::sort(FunctionCalls.begin(), FunctionCalls.end());
  FunctionCalls.erase(std::unique(FunctionCalls.begin(), FunctionCalls.end()),
                      FunctionCalls.end());

  DEBUG(if (NumFns != FunctionCalls.size())
        std::cerr << "Merged " << (NumFns-FunctionCalls.size())
        << " call nodes in " << Func.getName() << "\n";);
}


// markAlive - Simple graph traverser that recursively walks the graph marking
// stuff to be alive.
//
static void markAlive(DSNode *N, std::set<DSNode*> &Alive) {
  if (N == 0 || Alive.count(N)) return;

  Alive.insert(N);
  for (unsigned i = 0, e = N->getNumLinks(); i != e; ++i)
    markAlive(N->getLink(i), Alive);
}


// removeDeadNodes - Use a more powerful reachability analysis to eliminate
// subgraphs that are unreachable.  This often occurs because the data
// structure doesn't "escape" into it's caller, and thus should be eliminated
// from the caller's graph entirely.  This is only appropriate to use when
// inlining graphs.
//
void DSGraph::removeDeadNodes() {
  // Reduce the amount of work we have to do...
  removeTriviallyDeadNodes();
  
  // FIXME: Merge nontrivially identical call nodes...

  // Alive - a set that holds all nodes found to be reachable/alive.
  std::set<DSNode*> Alive;

  // Mark all nodes reachable by call nodes as alive...
  for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i)
    for (unsigned j = 0, e = FunctionCalls[i].size(); j != e; ++j)
      markAlive(FunctionCalls[i][j], Alive);

  for (unsigned i = 0, e = OrigFunctionCalls.size(); i != e; ++i)
    for (unsigned j = 0, e = OrigFunctionCalls[i].size(); j != e; ++j)
      markAlive(OrigFunctionCalls[i][j], Alive);

  // Mark all nodes reachable by scalar, global, or incomplete nodes as
  // reachable...
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
    if (Nodes[i]->NodeType & (DSNode::ScalarNode | DSNode::GlobalNode))
      markAlive(Nodes[i], Alive);

  // Loop over all unreachable nodes, dropping their references...
  std::vector<DSNode*> DeadNodes;
  DeadNodes.reserve(Nodes.size());     // Only one allocation is allowed.
  for (unsigned i = 0; i != Nodes.size(); ++i)
    if (!Alive.count(Nodes[i])) {
      DSNode *N = Nodes[i];
      Nodes.erase(Nodes.begin()+i--);  // Erase node from alive list.
      DeadNodes.push_back(N);          // Add node to our list of dead nodes
      N->dropAllReferences();          // Drop all outgoing edges
    }
  
  // The return value is alive as well...
  markAlive(RetNode, Alive);

  // Delete all dead nodes...
  std::for_each(DeadNodes.begin(), DeadNodes.end(), deleter<DSNode>);
}



// maskNodeTypes - Apply a mask to all of the node types in the graph.  This
// is useful for clearing out markers like Scalar or Incomplete.
//
void DSGraph::maskNodeTypes(unsigned char Mask) {
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
    Nodes[i]->NodeType &= Mask;
}


//===----------------------------------------------------------------------===//
// LocalDataStructures Implementation
//===----------------------------------------------------------------------===//

// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
//
void LocalDataStructures::releaseMemory() {
  for (std::map<Function*, DSGraph*>::iterator I = DSInfo.begin(),
         E = DSInfo.end(); I != E; ++I)
    delete I->second;

  // Empty map so next time memory is released, data structures are not
  // re-deleted.
  DSInfo.clear();
}

bool LocalDataStructures::run(Module &M) {
  // Calculate all of the graphs...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal())
      DSInfo.insert(std::make_pair(&*I, new DSGraph(*I)));

  return false;
}
