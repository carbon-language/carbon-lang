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

DSGraph::DSGraph(const DSGraph &G) : Func(G.Func), GlobalsGraph(G.GlobalsGraph){
  GlobalsGraph->addReference(this);
  std::map<const DSNode*, DSNode*> NodeMap; // ignored
  RetNode = cloneInto(G, ValueMap, NodeMap);
}

DSGraph::~DSGraph() {
  GlobalsGraph->removeReference(this);
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


// Helper function used to clone a function list.
// Each call really shd have an explicit representation as a separate class. 
void
CopyFunctionCallsList(const std::vector<std::vector<DSNodeHandle> >& fromCalls,
                      std::vector<std::vector<DSNodeHandle> >& toCalls,
                      std::map<const DSNode*, DSNode*>& NodeMap) {
  
  unsigned FC = toCalls.size();  // FirstCall
  toCalls.reserve(FC+fromCalls.size());
  for (unsigned i = 0, ei = fromCalls.size(); i != ei; ++i) {
    toCalls.push_back(std::vector<DSNodeHandle>());
    toCalls[FC+i].reserve(fromCalls[i].size());
    for (unsigned j = 0, ej = fromCalls[i].size(); j != ej; ++j)
      toCalls[FC+i].push_back(NodeMap[fromCalls[i][j]]);
  }
}


// cloneInto - Clone the specified DSGraph into the current graph, returning the
// Return node of the graph.  The translated ValueMap for the old function is
// filled into the OldValMap member.  If StripLocals is set to true, Scalar and
// Alloca markers are removed from the graph, as the graph is being cloned into
// a calling function's graph.
//
DSNode *DSGraph::cloneInto(const DSGraph &G, 
                           std::map<Value*, DSNodeHandle> &OldValMap,
                           std::map<const DSNode*, DSNode*> &OldNodeMap,
                           bool StripScalars, bool StripAllocas,
                           bool CopyCallers, bool CopyOrigCalls) {

  assert(OldNodeMap.size()==0 && "Return arg. OldNodeMap shd be empty");

  OldNodeMap[0] = 0;                    // Null pointer maps to null

  unsigned FN = Nodes.size();           // First new node...

  // Duplicate all of the nodes, populating the node map...
  Nodes.reserve(FN+G.Nodes.size());
  for (unsigned i = 0, e = G.Nodes.size(); i != e; ++i) {
    DSNode *Old = G.Nodes[i], *New = new DSNode(*Old);
    Nodes.push_back(New);
    OldNodeMap[Old] = New;
  }

  // Rewrite the links in the new nodes to point into the current graph now.
  for (unsigned i = FN, e = Nodes.size(); i != e; ++i)
    for (unsigned j = 0, e = Nodes[i]->getNumLinks(); j != e; ++j)
      Nodes[i]->setLink(j, OldNodeMap.find(Nodes[i]->getLink(j))->second);

  // Remove local markers as specified
  if (StripScalars || StripAllocas) {
    char keepBits = ~((StripScalars? DSNode::ScalarNode : 0) |
                      (StripAllocas? DSNode::AllocaNode : 0));
    for (unsigned i = FN, e = Nodes.size(); i != e; ++i)
      Nodes[i]->NodeType &= keepBits;
  }

  // Copy the value map...
  for (std::map<Value*, DSNodeHandle>::const_iterator I = G.ValueMap.begin(),
         E = G.ValueMap.end(); I != E; ++I)
    OldValMap[I->first] = OldNodeMap[I->second];

  // Copy the function calls list...
  CopyFunctionCallsList(G.FunctionCalls, FunctionCalls, OldNodeMap);
  if (CopyOrigCalls) 
    CopyFunctionCallsList(G.OrigFunctionCalls, OrigFunctionCalls, OldNodeMap);

  // Copy the list of unresolved callers
  if (CopyCallers)
    PendingCallers.insert(G.PendingCallers.begin(), G.PendingCallers.end());

  // Return the returned node pointer...
  return OldNodeMap[G.RetNode];
}


// cloneGlobalInto - Clone the given global node and all its target links
// (and all their llinks, recursively).
// 
DSNode* DSGraph::cloneGlobalInto(const DSNode* GNode) {
  if (GNode == 0 || GNode->getGlobals().size() == 0) return 0;

  // If a clone has already been created for GNode, return it.
  DSNodeHandle& ValMapEntry = ValueMap[GNode->getGlobals()[0]];
  if (ValMapEntry != 0)
    return ValMapEntry;

  // Clone the node and update the ValMap.
  DSNode* NewNode = new DSNode(*GNode);
  ValMapEntry = NewNode;                // j=0 case of loop below!
  Nodes.push_back(NewNode);
  for (unsigned j = 1, N = NewNode->getGlobals().size(); j < N; ++j)
    ValueMap[NewNode->getGlobals()[j]] = NewNode;

  // Rewrite the links in the new node to point into the current graph.
  for (unsigned j = 0, e = GNode->getNumLinks(); j != e; ++j)
    NewNode->setLink(j, cloneGlobalInto(GNode->getLink(j)));

  return NewNode;
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
void DSGraph::markIncompleteNodes(bool markFormalArgs) {
  // Mark any incoming arguments as incomplete...
  if (markFormalArgs)
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

// removeRefsToGlobal - Helper function that removes globals from the
// ValueMap so that the referrer count will go down to zero.
static void
removeRefsToGlobal(DSNode* N, std::map<Value*, DSNodeHandle>& ValueMap) {
  while (!N->getGlobals().empty()) {
    GlobalValue *GV = N->getGlobals().back();
    N->getGlobals().pop_back();      
    ValueMap.erase(GV);
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
  if (N->NodeType != 0 &&
      (N->NodeType & ~DSNode::GlobalNode) == 0 && 
      N->getNumLinks() == 0 &&
      N->getReferrers().size() == N->getGlobals().size()) {

    // Remove the globals from the valuemap, so that the referrer count will go
    // down to zero.
    removeRefsToGlobal(N, ValueMap);
    assert(N->getReferrers().empty() && "Referrers should all be gone now!");
    return true;
  }

  return false;
}

static void
removeIdenticalCalls(std::vector<std::vector<DSNodeHandle> >& Calls,
                     const string& where) {
  // Remove trivially identical function calls
  unsigned NumFns = Calls.size();
  std::sort(Calls.begin(), Calls.end());
  Calls.erase(std::unique(Calls.begin(), Calls.end()),
              Calls.end());

  DEBUG(if (NumFns != Calls.size())
        std::cerr << "Merged " << (NumFns-Calls.size())
        << " call nodes in " << where << "\n";);
}

// removeTriviallyDeadNodes - After the graph has been constructed, this method
// removes all unreachable nodes that are created because they got merged with
// other nodes in the graph.  These nodes will all be trivially unreachable, so
// we don't have to perform any non-trivial analysis here.
//
void DSGraph::removeTriviallyDeadNodes(bool KeepAllGlobals) {
  for (unsigned i = 0; i != Nodes.size(); ++i)
    if (! KeepAllGlobals || ! (Nodes[i]->NodeType & DSNode::GlobalNode))
      if (isNodeDead(Nodes[i])) {               // This node is dead!
        delete Nodes[i];                        // Free memory...
        Nodes.erase(Nodes.begin()+i--);         // Remove from node list...
      }

  removeIdenticalCalls(FunctionCalls, Func.getName());
}


// markAlive - Simple graph traverser that recursively walks the graph marking
// stuff to be alive.
//
static void markAlive(DSNode *N, std::set<DSNode*> &Alive) {
  if (N == 0) return;

  Alive.insert(N);
  for (unsigned i = 0, e = N->getNumLinks(); i != e; ++i)
    if (N->getLink(i) && !Alive.count(N->getLink(i)))
      markAlive(N->getLink(i), Alive);
}

static bool checkGlobalAlive(DSNode *N, std::set<DSNode*> &Alive,
                             std::set<DSNode*> &Visiting) {
  if (N == 0) return false;

  if (Visiting.count(N) > 0) return false; // terminate recursion on a cycle
  Visiting.insert(N);

  // If any immediate successor is alive, N is alive
  for (unsigned i = 0, e = N->getNumLinks(); i != e; ++i)
    if (N->getLink(i) && Alive.count(N->getLink(i)))
      { Visiting.erase(N); return true; }

  // Else if any successor reaches a live node, N is alive
  for (unsigned i = 0, e = N->getNumLinks(); i != e; ++i)
    if (N->getLink(i) && checkGlobalAlive(N->getLink(i), Alive, Visiting))
      { Visiting.erase(N); return true; }

  Visiting.erase(N);
  return false;
}


// markGlobalsIteration - Recursive helper function for markGlobalsAlive().
// This would be unnecessary if function calls were real nodes!  In that case,
// the simple iterative loop in the first few lines below suffice.
// 
static void markGlobalsIteration(std::set<DSNode*>& GlobalNodes,
                                 std::vector<std::vector<DSNodeHandle> > &Calls,
                                 std::set<DSNode*> &Alive,
                                 bool FilterCalls) {

  // Iterate, marking globals or cast nodes alive until no new live nodes
  // are added to Alive
  std::set<DSNode*> Visiting;           // Used to identify cycles 
  std::set<DSNode*>::iterator I=GlobalNodes.begin(), E=GlobalNodes.end();
  for (size_t liveCount = 0; liveCount < Alive.size(); ) {
    liveCount = Alive.size();
    for ( ; I != E; ++I)
      if (Alive.count(*I) == 0) {
        Visiting.clear();
        if (checkGlobalAlive(*I, Alive, Visiting))
          markAlive(*I, Alive);
      }
  }

  // Find function calls with some dead and some live nodes.
  // Since all call nodes must be live if any one is live, we have to mark
  // all nodes of the call as live and continue the iteration (via recursion).
  if (FilterCalls) {
    bool recurse = false;
    for (int i = 0, ei = Calls.size(); i < ei; ++i) {
      bool CallIsDead = true, CallHasDeadArg = false;
      for (unsigned j = 0, ej = Calls[i].size(); j != ej; ++j) {
        bool argIsDead = Calls[i][j] == 0 || Alive.count(Calls[i][j]) == 0;
        CallHasDeadArg = CallHasDeadArg || (Calls[i][j] != 0 && argIsDead);
        CallIsDead = CallIsDead && argIsDead;
      }
      if (!CallIsDead && CallHasDeadArg) {
        // Some node in this call is live and another is dead.
        // Mark all nodes of call as live and iterate once more.
        recurse = true;
        for (unsigned j = 0, ej = Calls[i].size(); j != ej; ++j)
          markAlive(Calls[i][j], Alive);
      }
    }
    if (recurse)
      markGlobalsIteration(GlobalNodes, Calls, Alive, FilterCalls);
  }
}


// markGlobalsAlive - Mark global nodes and cast nodes alive if they
// can reach any other live node.  Since this can produce new live nodes,
// we use a simple iterative algorithm.
// 
static void markGlobalsAlive(DSGraph& G, std::set<DSNode*> &Alive,
                             bool FilterCalls) {
  // Add global and cast nodes to a set so we don't walk all nodes every time
  std::set<DSNode*> GlobalNodes;
  for (unsigned i = 0, e = G.getNodes().size(); i != e; ++i)
    if (G.getNodes()[i]->NodeType & (DSNode::CastNode | DSNode::GlobalNode))
      GlobalNodes.insert(G.getNodes()[i]);

  // Add all call nodes to the same set
  std::vector<std::vector<DSNodeHandle> > &Calls = G.getFunctionCalls();
  if (FilterCalls) {
    for (unsigned i = 0, e = Calls.size(); i != e; ++i)
      for (unsigned j = 0, e = Calls[i].size(); j != e; ++j)
        if (Calls[i][j])
          GlobalNodes.insert(Calls[i][j]);
  }

  // Iterate and recurse until no new live node are discovered.
  // This would be a simple iterative loop if function calls were real nodes!
  markGlobalsIteration(GlobalNodes, Calls, Alive, FilterCalls);

  // Free up references to dead globals from the ValueMap
  std::set<DSNode*>::iterator I=GlobalNodes.begin(), E=GlobalNodes.end();
  for( ; I != E; ++I)
    if (Alive.count(*I) == 0)
      removeRefsToGlobal(*I, G.getValueMap());

  // Delete dead function calls
  if (FilterCalls)
    for (int ei = Calls.size(), i = ei-1; i >= 0; --i) {
      bool CallIsDead = true;
      for (unsigned j = 0, ej= Calls[i].size(); CallIsDead && j != ej; ++j)
        CallIsDead = (Alive.count(Calls[i][j]) == 0);
      if (CallIsDead)
        Calls.erase(Calls.begin() + i); // remove the call entirely
    }
}

// removeDeadNodes - Use a more powerful reachability analysis to eliminate
// subgraphs that are unreachable.  This often occurs because the data
// structure doesn't "escape" into it's caller, and thus should be eliminated
// from the caller's graph entirely.  This is only appropriate to use when
// inlining graphs.
//
void DSGraph::removeDeadNodes(bool KeepAllGlobals, bool KeepCalls) {
  assert((!KeepAllGlobals || KeepCalls) &&
         "KeepAllGlobals without KeepCalls is meaningless");

  // Reduce the amount of work we have to do...
  removeTriviallyDeadNodes(KeepAllGlobals);

  // FIXME: Merge nontrivially identical call nodes...

  // Alive - a set that holds all nodes found to be reachable/alive.
  std::set<DSNode*> Alive;

  // If KeepCalls, mark all nodes reachable by call nodes as alive...
  if (KeepCalls)
    for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i)
      for (unsigned j = 0, e = FunctionCalls[i].size(); j != e; ++j)
        markAlive(FunctionCalls[i][j], Alive);

  for (unsigned i = 0, e = OrigFunctionCalls.size(); i != e; ++i)
    for (unsigned j = 0, e = OrigFunctionCalls[i].size(); j != e; ++j)
      markAlive(OrigFunctionCalls[i][j], Alive);

  // Mark all nodes reachable by scalar nodes (and global nodes, if
  // keeping them was specified) as alive...
  char keepBits = DSNode::ScalarNode | (KeepAllGlobals? DSNode::GlobalNode : 0);
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
    if (Nodes[i]->NodeType & keepBits)
      markAlive(Nodes[i], Alive);

  // The return value is alive as well...
  markAlive(RetNode, Alive);

  // Mark all globals or cast nodes that can reach a live node as alive.
  // This also marks all nodes reachable from such nodes as alive.
  // Of course, if KeepAllGlobals is specified, they would be live already.
  if (! KeepAllGlobals)
    markGlobalsAlive(*this, Alive, ! KeepCalls);

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
// GlobalDSGraph Implementation
//===----------------------------------------------------------------------===//

GlobalDSGraph::GlobalDSGraph() : DSGraph(*(Function*)0, this) {
}

GlobalDSGraph::~GlobalDSGraph() {
  assert(Referrers.size() == 0 &&
         "Deleting global graph while references from other graphs exist");
}

void GlobalDSGraph::addReference(const DSGraph* referrer) {
  if (referrer != this)
    Referrers.insert(referrer);
}

void GlobalDSGraph::removeReference(const DSGraph* referrer) {
  if (referrer != this) {
    assert(Referrers.find(referrer) != Referrers.end() && "This is very bad!");
    Referrers.erase(referrer);
    if (Referrers.size() == 0)
      delete this;
  }
}

// Bits used in the next function
static const char ExternalTypeBits = (DSNode::GlobalNode | DSNode::NewNode |
                                      DSNode::SubElement | DSNode::CastNode);


// GlobalDSGraph::cloneNodeInto - Clone a global node and all its externally
// visible target links (and recursively their such links) into this graph.
// NodeCache maps the node being cloned to its clone in the Globals graph,
// in order to track cycles.
// GlobalsAreFinal is a flag that says whether it is safe to assume that
// an existing global node is complete.  This is important to avoid
// reinserting all globals when inserting Calls to functions.
// This is a helper function for cloneGlobals and cloneCalls.
// 
DSNode* GlobalDSGraph::cloneNodeInto(DSNode *OldNode,
                                    std::map<const DSNode*, DSNode*> &NodeCache,
                                    bool GlobalsAreFinal) {
  if (OldNode == 0) return 0;

  // The caller should check this is an external node.  Just more  efficient...
  assert((OldNode->NodeType & ExternalTypeBits) && "Non-external node");

  // If a clone has already been created for OldNode, return it.
  DSNode*& CacheEntry = NodeCache[OldNode];
  if (CacheEntry != 0)
    return CacheEntry;

  // The result value...
  DSNode* NewNode = 0;

  // If nodes already exist for any of the globals of OldNode,
  // merge all such nodes together since they are merged in OldNode.
  // If ValueCacheIsFinal==true, look for an existing node that has
  // an identical list of globals and return it if it exists.
  //
  for (unsigned j = 0, N = OldNode->getGlobals().size(); j < N; ++j)
    if (DSNode* PrevNode = ValueMap[OldNode->getGlobals()[j]]) {
      if (NewNode == 0) {
        NewNode = PrevNode;             // first existing node found
        if (GlobalsAreFinal && j == 0)
          if (OldNode->getGlobals() == PrevNode->getGlobals()) {
            CacheEntry = NewNode;
            return NewNode;
          }
      }
      else if (NewNode != PrevNode) {   // found another, different from prev
        // update ValMap *before* merging PrevNode into NewNode
        for (unsigned k = 0, NK = PrevNode->getGlobals().size(); k < NK; ++k)
          ValueMap[PrevNode->getGlobals()[k]] = NewNode;
        NewNode->mergeWith(PrevNode);
      }
    } else if (NewNode != 0) {
      ValueMap[OldNode->getGlobals()[j]] = NewNode; // add the merged node
    }

  // If no existing node was found, clone the node and update the ValMap.
  if (NewNode == 0) {
    NewNode = new DSNode(*OldNode);
    Nodes.push_back(NewNode);
    for (unsigned j = 0, e = NewNode->getNumLinks(); j != e; ++j)
      NewNode->setLink(j, 0);
    for (unsigned j = 0, N = NewNode->getGlobals().size(); j < N; ++j)
      ValueMap[NewNode->getGlobals()[j]] = NewNode;
  }
  else
    NewNode->NodeType |= OldNode->NodeType; // Markers may be different!

  // Add the entry to NodeCache
  CacheEntry = NewNode;

  // Rewrite the links in the new node to point into the current graph,
  // but only for links to external nodes.  Set other links to NULL.
  for (unsigned j = 0, e = OldNode->getNumLinks(); j != e; ++j) {
    DSNode* OldTarget = OldNode->getLink(j);
    if (OldTarget && (OldTarget->NodeType & ExternalTypeBits)) {
      DSNode* NewLink = this->cloneNodeInto(OldTarget, NodeCache);
      if (NewNode->getLink(j))
        NewNode->getLink(j)->mergeWith(NewLink);
      else
        NewNode->setLink(j, NewLink);
    }
  }

  // Remove all local markers
  NewNode->NodeType &= ~(DSNode::AllocaNode | DSNode::ScalarNode);

  return NewNode;
}


// GlobalDSGraph::cloneGlobals - Clone global nodes and all their externally
// visible target links (and recursively their such links) into this graph.
// 
void GlobalDSGraph::cloneGlobals(DSGraph& Graph, bool CloneCalls) {
  std::map<const DSNode*, DSNode*> NodeCache;
  for (unsigned i = 0, N = Graph.Nodes.size(); i < N; ++i)
    if (Graph.Nodes[i]->NodeType & DSNode::GlobalNode)
      GlobalsGraph->cloneNodeInto(Graph.Nodes[i], NodeCache, false);

  if (CloneCalls)
    GlobalsGraph->cloneCalls(Graph);

  GlobalsGraph->removeDeadNodes(/*KeepAllGlobals*/ true, /*KeepCalls*/ true);
}


// GlobalDSGraph::cloneCalls - Clone function calls and their visible target
// links (and recursively their such links) into this graph.
// 
void GlobalDSGraph::cloneCalls(DSGraph& Graph) {
  std::map<const DSNode*, DSNode*> NodeCache;
  std::vector<std::vector<DSNodeHandle> >& FromCalls =Graph.FunctionCalls;

  FunctionCalls.reserve(FunctionCalls.size() + FromCalls.size());

  for (int i = 0, ei = FromCalls.size(); i < ei; ++i) {
    FunctionCalls.push_back(std::vector<DSNodeHandle>());
    FunctionCalls.back().reserve(FromCalls[i].size());
    for (unsigned j = 0, ej = FromCalls[i].size(); j != ej; ++j)
      FunctionCalls.back().push_back
        ((FromCalls[i][j] && (FromCalls[i][j]->NodeType & ExternalTypeBits))
         ? cloneNodeInto(FromCalls[i][j], NodeCache, true)
         : 0);
  }

  // remove trivially identical function calls
  removeIdenticalCalls(FunctionCalls, string("Globals Graph"));
}


//===----------------------------------------------------------------------===//
// LocalDataStructures Implementation
//===----------------------------------------------------------------------===//

// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
//
void LocalDataStructures::releaseMemory() {
  for (std::map<const Function*, DSGraph*>::iterator I = DSInfo.begin(),
         E = DSInfo.end(); I != E; ++I)
    delete I->second;

  // Empty map so next time memory is released, data structures are not
  // re-deleted.
  DSInfo.clear();
}

bool LocalDataStructures::run(Module &M) {
  // Create a globals graph for the module.  Deleted when all graphs go away.
  GlobalDSGraph* GG = new GlobalDSGraph;
  
  // Calculate all of the graphs...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal())
      DSInfo.insert(std::make_pair(&*I, new DSGraph(*I, GG)));

  return false;
}
