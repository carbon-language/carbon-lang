//===- DataStructure.cpp - Implement the core data structure analysis -----===//
//
// This file implements the core data structure functionality.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DSGraph.h"
#include "llvm/Function.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Target/TargetData.h"
#include "Support/STLExtras.h"
#include "Support/Statistic.h"
#include <algorithm>
#include <set>

using std::vector;

// TODO: FIXME
namespace DataStructureAnalysis {
  // isPointerType - Return true if this first class type is big enough to hold
  // a pointer.
  //
  bool isPointerType(const Type *Ty);
  extern TargetData TD;
}
using namespace DataStructureAnalysis;

//===----------------------------------------------------------------------===//
// DSNode Implementation
//===----------------------------------------------------------------------===//

DSNode::DSNode(enum NodeTy NT, const Type *T) : NodeType(NT) {
  // If this node is big enough to have pointer fields, add space for them now.
  if (T != Type::VoidTy && !isa<FunctionType>(T)) { // Avoid TargetData assert's
    MergeMap.resize(TD.getTypeSize(T));

    // Assign unique values to all of the elements of MergeMap
    if (MergeMap.size() < 128) {
      // Handle the common case of reasonable size structures...
      for (unsigned i = 0, e = MergeMap.size(); i != e; ++i)
        MergeMap[i] = -1-i;   // Assign -1, -2, -3, ...
    } else {
      // It's possible that we have something really big here.  In this case,
      // divide the object into chunks until it will fit into 128 elements.
      unsigned Multiple = MergeMap.size()/128;

      // It's probably an array, and probably some power of two in size.
      // Because of this, find the biggest power of two that is bigger than
      // multiple to use as our real Multiple.
      unsigned RealMultiple = 2;
      while (RealMultiple <= Multiple) RealMultiple <<= 1;

      unsigned RealBound = MergeMap.size()/RealMultiple;
      assert(RealBound <= 128 && "Math didn't work out right");

      // Now go through and assign indexes that are between -1 and -128
      // inclusive
      //
      for (unsigned i = 0, e = MergeMap.size(); i != e; ++i)
        MergeMap[i] = -1-(i % RealBound);   // Assign -1, -2, -3...
    }
  }

  TypeEntries.push_back(TypeRec(T, 0));
}

// DSNode copy constructor... do not copy over the referrers list!
DSNode::DSNode(const DSNode &N)
  : Links(N.Links), MergeMap(N.MergeMap),
    TypeEntries(N.TypeEntries), Globals(N.Globals), NodeType(N.NodeType) {
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
    //assert(GV->getType()->getElementType() == Ty);
    Globals.insert(I, GV);
    NodeType |= GlobalNode;
  }
}


/// setLink - Set the link at the specified offset to the specified
/// NodeHandle, replacing what was there.  It is uncommon to use this method,
/// instead one of the higher level methods should be used, below.
///
void DSNode::setLink(unsigned i, const DSNodeHandle &NH) {
  // Create a new entry in the Links vector to hold a new element for offset.
  if (!hasLink(i)) {
    signed char NewIdx = Links.size();
    // Check to see if we allocate more than 128 distinct links for this node.
    // If so, just merge with the last one.  This really shouldn't ever happen,
    // but it should work regardless of whether it does or not.
    //
    if (NewIdx >= 0) {
      Links.push_back(NH);             // Allocate space: common case
    } else {                           // Wrap around?  Too many links?
      NewIdx--;                        // Merge with whatever happened last
      assert(NewIdx > 0 && "Should wrap back around");
      std::cerr << "\n*** DSNode found that requires more than 128 "
                << "active links at once!\n\n";
    } 

    signed char OldIdx = MergeMap[i];
    assert (OldIdx < 0 && "Shouldn't contain link!");

    // Make sure that anything aliasing this field gets updated to point to the
    // new link field.
    rewriteMergeMap(OldIdx, NewIdx);
    assert(MergeMap[i] == NewIdx && "Field not replaced!");
  } else {
    Links[MergeMap[i]] = NH;
  }
}

// addEdgeTo - Add an edge from the current node to the specified node.  This
// can cause merging of nodes in the graph.
//
void DSNode::addEdgeTo(unsigned Offset, const DSNodeHandle &NH) {
  assert(Offset < getSize() && "Offset out of range!");
  if (NH.getNode() == 0) return;       // Nothing to do

  if (DSNodeHandle *ExistingNH = getLink(Offset)) {
    // Merge the two nodes...
    ExistingNH->mergeWith(NH);
  } else {                             // No merging to perform...
    setLink(Offset, NH);               // Just force a link in there...
  }
}

/// mergeMappedValues - This is the higher level form of rewriteMergeMap.  It is
/// fully capable of merging links together if neccesary as well as simply
/// rewriting the map entries.
///
void DSNode::mergeMappedValues(signed char V1, signed char V2) {
  assert(V1 != V2 && "Cannot merge two identical mapped values!");
  
  if (V1 < 0) {  // If there is no outgoing link from V1, merge it with V2
    if (V2 < 0 && V1 > V2)
       // If both are not linked, merge to the field closer to 0
      rewriteMergeMap(V2, V1);
    else
      rewriteMergeMap(V1, V2);
  } else if (V2 < 0) {           // Is V2 < 0 && V1 >= 0?
    rewriteMergeMap(V2, V1);     // Merge into the one with the link...
  } else {                       // Otherwise, links exist at both locations
    // Merge Links[V1] with Links[V2] so they point to the same place now...
    Links[V1].mergeWith(Links[V2]);

    // Merge the V2 link into V1 so that we reduce the overall value of the
    // links are reduced...
    //
    if (V2 < V1) std::swap(V1, V2);     // Ensure V1 < V2
    rewriteMergeMap(V2, V1);            // After this, V2 is "dead"

    // Change the user of the last link to use V2 instead
    if ((unsigned)V2 != Links.size()-1) {
      rewriteMergeMap(Links.size()-1, V2);  // Point to V2 instead of last el...
      // Make sure V2 points the right DSNode
      Links[V2] = Links.back();
    }

    // Reduce the number of distinct outgoing links...
    Links.pop_back();
  }
}


// MergeSortedVectors - Efficiently merge a vector into another vector where
// duplicates are not allowed and both are sorted.  This assumes that 'T's are
// efficiently copyable and have sane comparison semantics.
//
template<typename T>
void MergeSortedVectors(vector<T> &Dest, const vector<T> &Src) {
  // By far, the most common cases will be the simple ones.  In these cases,
  // avoid having to allocate a temporary vector...
  //
  if (Src.empty()) {             // Nothing to merge in...
    return;
  } else if (Dest.empty()) {     // Just copy the result in...
    Dest = Src;
  } else if (Src.size() == 1) {  // Insert a single element...
    const T &V = Src[0];
    typename vector<T>::iterator I =
      std::lower_bound(Dest.begin(), Dest.end(), V);
    if (I == Dest.end() || *I != Src[0])  // If not already contained...
      Dest.insert(I, Src[0]);
  } else if (Dest.size() == 1) {
    T Tmp = Dest[0];                      // Save value in temporary...
    Dest = Src;                           // Copy over list...
    typename vector<T>::iterator I =
      std::lower_bound(Dest.begin(), Dest.end(),Tmp);
    if (I == Dest.end() || *I != Src[0])  // If not already contained...
      Dest.insert(I, Src[0]);

  } else {
    // Make a copy to the side of Dest...
    vector<T> Old(Dest);
    
    // Make space for all of the type entries now...
    Dest.resize(Dest.size()+Src.size());
    
    // Merge the two sorted ranges together... into Dest.
    std::merge(Old.begin(), Old.end(), Src.begin(), Src.end(), Dest.begin());
    
    // Now erase any duplicate entries that may have accumulated into the 
    // vectors (because they were in both of the input sets)
    Dest.erase(std::unique(Dest.begin(), Dest.end()), Dest.end());
  }
}


// mergeWith - Merge this node and the specified node, moving all links to and
// from the argument node into the current node, deleting the node argument.
// Offset indicates what offset the specified node is to be merged into the
// current node.
//
// The specified node may be a null pointer (in which case, nothing happens).
//
void DSNode::mergeWith(const DSNodeHandle &NH, unsigned Offset) {
  DSNode *N = NH.getNode();
  if (N == 0 || (N == this && NH.getOffset() == Offset))
      return;  // Noop

  assert(NH.getNode() != this &&
         "Cannot merge two portions of the same node yet!");

  // If both nodes are not at offset 0, make sure that we are merging the node
  // at an later offset into the node with the zero offset.
  //
  if (Offset > NH.getOffset()) {
    N->mergeWith(DSNodeHandle(this, Offset), NH.getOffset());
    return;
  }

#if 0
  std::cerr << "\n\nMerging:\n";
  N->print(std::cerr, 0);
  std::cerr << " and:\n";
  print(std::cerr, 0);
#endif

  // Now we know that Offset <= NH.Offset, so convert it so our "Offset" (with
  // respect to NH.Offset) is now zero.
  //
  unsigned NOffset = NH.getOffset()-Offset;

  unsigned NSize = N->getSize();
  assert(NSize+NOffset <= getSize() &&
         "Don't know how to merge extend a merged nodes size yet!");

  // Remove all edges pointing at N, causing them to point to 'this' instead.
  // Make sure to adjust their offset, not just the node pointer.
  //
  while (!N->Referrers.empty()) {
    DSNodeHandle &Ref = *N->Referrers.back();
    Ref = DSNodeHandle(this, NOffset+Ref.getOffset());
  }

  // We must merge fields in this node due to nodes merged in the source node.
  // In order to handle this we build a map that converts from the source node's
  // MergeMap values to our MergeMap values.  This map is indexed by the
  // expression: MergeMap[SMM+SourceNodeSize] so we need to allocate at least
  // 2*SourceNodeSize elements of space for the mapping.  We can do this because
  // we know that there are at most SourceNodeSize outgoing links in the node
  // (thus that many positive values) and at most SourceNodeSize distinct fields
  // (thus that many negative values).
  //
  std::vector<signed char> MergeMapMap(NSize*2, 127);

  // Loop through the structures, merging them together...
  for (unsigned i = 0, e = NSize; i != e; ++i) {
    // Get what this byte of N maps to...
    signed char NElement = N->MergeMap[i];

    // Get what we map this byte to...
    signed char Element = MergeMap[i+NOffset];
    // We use 127 as a sentinal and don't check for it's existence yet...
    assert(Element != 127 && "MergeMapMap doesn't permit 127 values yet!");

    signed char CurMappedVal = MergeMapMap[NElement+NSize];
    if (CurMappedVal == 127) {               // Haven't seen this NElement yet?
      MergeMapMap[NElement+NSize] = Element; // Map the two together...
    } else if (CurMappedVal != Element) {
      // If we are mapping two different fields together this means that we need
      // to merge fields in the current node due to merging in the source node.
      //
      mergeMappedValues(CurMappedVal, Element);
      MergeMapMap[NElement+NSize] = MergeMap[i+NOffset];
    }
  }

  // Make all of the outgoing links of N now be outgoing links of this.  This
  // can cause recursive merging!
  //
  for (unsigned i = 0, e = NSize; i != e; ++i)
    if (DSNodeHandle *Link = N->getLink(i)) {
      addEdgeTo(i+NOffset, *Link);
      N->MergeMap[i] = -1;  // Kill outgoing edge
    }

  // Now that there are no outgoing edges, all of the Links are dead.
  N->Links.clear();

  // Merge the node types
  NodeType |= N->NodeType;
  N->NodeType = 0;   // N is now a dead node.

  // If this merging into node has more than just void nodes in it, merge!
  assert(!N->TypeEntries.empty() && "TypeEntries is empty for a node?");
  if (N->TypeEntries.size() != 1 || N->TypeEntries[0].Ty != Type::VoidTy) {
    // If the current node just has a Void entry in it, remove it.
    if (TypeEntries.size() == 1 && TypeEntries[0].Ty == Type::VoidTy)
      TypeEntries.clear();

    // Adjust all of the type entries we are merging in by the offset... and add
    // them to the TypeEntries list.
    //
    if (NOffset != 0) {  // This case is common enough to optimize for
      // Offset all of the TypeEntries in N with their new offset
      for (unsigned i = 0, e = N->TypeEntries.size(); i != e; ++i)
        N->TypeEntries[i].Offset += NOffset;
    }

    MergeSortedVectors(TypeEntries, N->TypeEntries);

    N->TypeEntries.clear();
  }

  // Merge the globals list...
  if (!N->Globals.empty()) {
    MergeSortedVectors(Globals, N->Globals);

    // Delete the globals from the old node...
    N->Globals.clear();
  }
}

//===----------------------------------------------------------------------===//
// DSGraph Implementation
//===----------------------------------------------------------------------===//

DSGraph::DSGraph(const DSGraph &G) : Func(G.Func) {
  std::map<const DSNode*, DSNode*> NodeMap;
  RetNode = cloneInto(G, ValueMap, NodeMap);
}

DSGraph::~DSGraph() {
  FunctionCalls.clear();
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
//
static void CopyFunctionCallsList(const vector<vector<DSNodeHandle> >&fromCalls,
                                  vector<vector<DSNodeHandle> > &toCalls,
                                  std::map<const DSNode*, DSNode*> &NodeMap) {

  unsigned FC = toCalls.size();  // FirstCall
  toCalls.reserve(FC+fromCalls.size());
  for (unsigned i = 0, ei = fromCalls.size(); i != ei; ++i) {
    toCalls.push_back(vector<DSNodeHandle>());
    
    const vector<DSNodeHandle> &CurCall = fromCalls[i];
    toCalls.back().reserve(CurCall.size());
    for (unsigned j = 0, ej = fromCalls[i].size(); j != ej; ++j)
      toCalls[FC+i].push_back(DSNodeHandle(NodeMap[CurCall[j].getNode()],
                                           CurCall[j].getOffset()));
  }
}

/// remapLinks - Change all of the Links in the current node according to the
/// specified mapping.
void DSNode::remapLinks(std::map<const DSNode*, DSNode*> &OldNodeMap) {
  for (unsigned i = 0, e = Links.size(); i != e; ++i) 
    Links[i].setNode(OldNodeMap[Links[i].getNode()]);
}

// cloneInto - Clone the specified DSGraph into the current graph, returning the
// Return node of the graph.  The translated ValueMap for the old function is
// filled into the OldValMap member.  If StripLocals is set to true, Scalar and
// Alloca markers are removed from the graph, as the graph is being cloned into
// a calling function's graph.
//
DSNodeHandle DSGraph::cloneInto(const DSGraph &G, 
                                std::map<Value*, DSNodeHandle> &OldValMap,
                                std::map<const DSNode*, DSNode*> &OldNodeMap,
                                bool StripScalars, bool StripAllocas,
                                bool CopyCallers, bool CopyOrigCalls) {
  assert(OldNodeMap.empty() && "Returned OldNodeMap should be empty!");

  unsigned FN = Nodes.size();           // First new node...

  // Duplicate all of the nodes, populating the node map...
  Nodes.reserve(FN+G.Nodes.size());
  for (unsigned i = 0, e = G.Nodes.size(); i != e; ++i) {
    DSNode *Old = G.Nodes[i];
    DSNode *New = new DSNode(*Old);
    Nodes.push_back(New);
    OldNodeMap[Old] = New;
  }

  // Rewrite the links in the new nodes to point into the current graph now.
  for (unsigned i = FN, e = Nodes.size(); i != e; ++i)
    Nodes[i]->remapLinks(OldNodeMap);

  // Remove local markers as specified
  unsigned char StripBits = (StripScalars ? DSNode::ScalarNode : 0) |
                            (StripAllocas ? DSNode::AllocaNode : 0);
  if (StripBits)
    for (unsigned i = FN, e = Nodes.size(); i != e; ++i)
      Nodes[i]->NodeType &= ~StripBits;

  // Copy the value map... and merge all of the global nodes...
  for (std::map<Value*, DSNodeHandle>::const_iterator I = G.ValueMap.begin(),
         E = G.ValueMap.end(); I != E; ++I) {
    DSNodeHandle &H = OldValMap[I->first];
    H = DSNodeHandle(OldNodeMap[I->second.getNode()], I->second.getOffset());

    if (isa<GlobalValue>(I->first)) {  // Is this a global?
      std::map<Value*, DSNodeHandle>::iterator GVI = ValueMap.find(I->first);
      if (GVI != ValueMap.end()) {   // Is the global value in this fun already?
        GVI->second.mergeWith(H);
      } else {
        ValueMap[I->first] = H;      // Add global pointer to this graph
      }
    }
  }
  // Copy the function calls list...
  CopyFunctionCallsList(G.FunctionCalls, FunctionCalls, OldNodeMap);


  // Return the returned node pointer...
  return DSNodeHandle(OldNodeMap[G.RetNode.getNode()], G.RetNode.getOffset());
}

#if 0
// cloneGlobalInto - Clone the given global node and all its target links
// (and all their llinks, recursively).
// 
DSNode *DSGraph::cloneGlobalInto(const DSNode *GNode) {
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
#endif


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
  for (unsigned i = 0, e = N->getSize(); i != e; ++i)
    if (DSNodeHandle *DSNH = N->getLink(i))
      markIncompleteNode(DSNH->getNode());
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
  if (markFormalArgs && Func)
    for (Function::aiterator I = Func->abegin(), E = Func->aend(); I != E; ++I)
      if (isPointerType(I->getType()) && ValueMap.find(I) != ValueMap.end()) {
        DSNodeHandle &INH = ValueMap[I];
        if (INH.getNode() && INH.hasLink(0))
          markIncompleteNode(ValueMap[I].getLink(0)->getNode());
      }

  // Mark stuff passed into functions calls as being incomplete...
  for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i) {
    vector<DSNodeHandle> &Args = FunctionCalls[i];
    // Then the return value is certainly incomplete!
    markIncompleteNode(Args[0].getNode());

    // The call does not make the function argument incomplete...
 
    // All arguments to the function call are incomplete though!
    for (unsigned i = 2, e = Args.size(); i != e; ++i)
      markIncompleteNode(Args[i].getNode());
  }

  // Mark all of the nodes pointed to by global or cast nodes as incomplete...
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
    if (Nodes[i]->NodeType & DSNode::GlobalNode) {
      DSNode *N = Nodes[i];
      for (unsigned i = 0, e = N->getSize(); i != e; ++i)
        if (DSNodeHandle *DSNH = N->getLink(i))
          markIncompleteNode(DSNH->getNode());
    }
}

// removeRefsToGlobal - Helper function that removes globals from the
// ValueMap so that the referrer count will go down to zero.
static void removeRefsToGlobal(DSNode* N,
                               std::map<Value*, DSNodeHandle> &ValueMap) {
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
      N->getSize() == 0 &&
      N->getReferrers().size() == N->getGlobals().size()) {

    // Remove the globals from the ValueMap, so that the referrer count will go
    // down to zero.
    removeRefsToGlobal(N, ValueMap);
    assert(N->getReferrers().empty() && "Referrers should all be gone now!");
    return true;
  }

  return false;
}

static void removeIdenticalCalls(vector<vector<DSNodeHandle> > &Calls,
                                 const std::string &where) {
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
    if (!KeepAllGlobals || !(Nodes[i]->NodeType & DSNode::GlobalNode))
      if (isNodeDead(Nodes[i])) {               // This node is dead!
        delete Nodes[i];                        // Free memory...
        Nodes.erase(Nodes.begin()+i--);         // Remove from node list...
      }

  removeIdenticalCalls(FunctionCalls, Func ? Func->getName() : "");
}


// markAlive - Simple graph walker that recursively traverses the graph, marking
// stuff to be alive.
//
static void markAlive(DSNode *N, std::set<DSNode*> &Alive) {
  if (N == 0) return;

  Alive.insert(N);
  for (unsigned i = 0, e = N->getSize(); i != e; ++i)
    if (DSNodeHandle *DSNH = N->getLink(i))
      if (!Alive.count(DSNH->getNode()))
        markAlive(DSNH->getNode(), Alive);
}

static bool checkGlobalAlive(DSNode *N, std::set<DSNode*> &Alive,
                             std::set<DSNode*> &Visiting) {
  if (N == 0) return false;

  if (Visiting.count(N)) return false; // terminate recursion on a cycle
  Visiting.insert(N);

  // If any immediate successor is alive, N is alive
  for (unsigned i = 0, e = N->getSize(); i != e; ++i)
    if (DSNodeHandle *DSNH = N->getLink(i))
      if (Alive.count(DSNH->getNode())) {
        Visiting.erase(N);
        return true;
      }

  // Else if any successor reaches a live node, N is alive
  for (unsigned i = 0, e = N->getSize(); i != e; ++i)
    if (DSNodeHandle *DSNH = N->getLink(i))
      if (checkGlobalAlive(DSNH->getNode(), Alive, Visiting)) {
        Visiting.erase(N); return true;
      }

  Visiting.erase(N);
  return false;
}


// markGlobalsIteration - Recursive helper function for markGlobalsAlive().
// This would be unnecessary if function calls were real nodes!  In that case,
// the simple iterative loop in the first few lines below suffice.
// 
static void markGlobalsIteration(std::set<DSNode*>& GlobalNodes,
                                 vector<vector<DSNodeHandle> > &Calls,
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
        bool argIsDead = Calls[i][j].getNode() == 0 ||
                         Alive.count(Calls[i][j].getNode()) == 0;
        CallHasDeadArg |= (Calls[i][j].getNode() != 0 && argIsDead);
        CallIsDead &= argIsDead;
      }
      if (!CallIsDead && CallHasDeadArg) {
        // Some node in this call is live and another is dead.
        // Mark all nodes of call as live and iterate once more.
        recurse = true;
        for (unsigned j = 0, ej = Calls[i].size(); j != ej; ++j)
          markAlive(Calls[i][j].getNode(), Alive);
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
static void markGlobalsAlive(DSGraph &G, std::set<DSNode*> &Alive,
                             bool FilterCalls) {
  // Add global and cast nodes to a set so we don't walk all nodes every time
  std::set<DSNode*> GlobalNodes;
  for (unsigned i = 0, e = G.getNodes().size(); i != e; ++i)
    if (G.getNodes()[i]->NodeType & DSNode::GlobalNode)
      GlobalNodes.insert(G.getNodes()[i]);

  // Add all call nodes to the same set
  vector<vector<DSNodeHandle> > &Calls = G.getFunctionCalls();
  if (FilterCalls) {
    for (unsigned i = 0, e = Calls.size(); i != e; ++i)
      for (unsigned j = 0, e = Calls[i].size(); j != e; ++j)
        if (Calls[i][j].getNode())
          GlobalNodes.insert(Calls[i][j].getNode());
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
      for (unsigned j = 0, ej = Calls[i].size(); CallIsDead && j != ej; ++j)
        CallIsDead = Alive.count(Calls[i][j].getNode()) == 0;
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
        markAlive(FunctionCalls[i][j].getNode(), Alive);

#if 0
  for (unsigned i = 0, e = OrigFunctionCalls.size(); i != e; ++i)
    for (unsigned j = 0, e = OrigFunctionCalls[i].size(); j != e; ++j)
      markAlive(OrigFunctionCalls[i][j].getNode(), Alive);
#endif

  // Mark all nodes reachable by scalar nodes (and global nodes, if
  // keeping them was specified) as alive...
  unsigned char keepBits = DSNode::ScalarNode |
                           (KeepAllGlobals ? DSNode::GlobalNode : 0);
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
    if (Nodes[i]->NodeType & keepBits)
      markAlive(Nodes[i], Alive);

  // The return value is alive as well...
  markAlive(RetNode.getNode(), Alive);

  // Mark all globals or cast nodes that can reach a live node as alive.
  // This also marks all nodes reachable from such nodes as alive.
  // Of course, if KeepAllGlobals is specified, they would be live already.
  if (! KeepAllGlobals)
    markGlobalsAlive(*this, Alive, ! KeepCalls);

  // Loop over all unreachable nodes, dropping their references...
  vector<DSNode*> DeadNodes;
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


#if 0
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
static const char ExternalTypeBits = DSNode::GlobalNode | DSNode::NewNode;

#if 0
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
  for (unsigned j = 0, N = OldNode->getGlobals().size(); j != N; ++j)
    if (DSNode *PrevNode = ValueMap[OldNode->getGlobals()[j]].getNode()) {
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
#if 0
  for (unsigned i = 0, N = Graph.Nodes.size(); i < N; ++i)
    if (Graph.Nodes[i]->NodeType & DSNode::GlobalNode)
      GlobalsGraph->cloneNodeInto(Graph.Nodes[i], NodeCache, false);
  if (CloneCalls)
    GlobalsGraph->cloneCalls(Graph);

  GlobalsGraph->removeDeadNodes(/*KeepAllGlobals*/ true, /*KeepCalls*/ true);
#endif
}


// GlobalDSGraph::cloneCalls - Clone function calls and their visible target
// links (and recursively their such links) into this graph.
// 
void GlobalDSGraph::cloneCalls(DSGraph& Graph) {
  std::map<const DSNode*, DSNode*> NodeCache;
  vector<vector<DSNodeHandle> >& FromCalls =Graph.FunctionCalls;

  FunctionCalls.reserve(FunctionCalls.size() + FromCalls.size());

  for (int i = 0, ei = FromCalls.size(); i < ei; ++i) {
    FunctionCalls.push_back(vector<DSNodeHandle>());
    FunctionCalls.back().reserve(FromCalls[i].size());
    for (unsigned j = 0, ej = FromCalls[i].size(); j != ej; ++j)
      FunctionCalls.back().push_back
        ((FromCalls[i][j] && (FromCalls[i][j]->NodeType & ExternalTypeBits))
         ? cloneNodeInto(FromCalls[i][j], NodeCache, true)
         : 0);
  }

  // remove trivially identical function calls
  removeIdenticalCalls(FunctionCalls, "Globals Graph");
}
#endif

#endif
