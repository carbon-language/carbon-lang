//===- DataStructure.cpp - Implement the core data structure analysis -----===//
//
// This file implements the core data structure functionality.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DSGraph.h"
#include "llvm/Function.h"
#include "llvm/iOther.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Target/TargetData.h"
#include "Support/STLExtras.h"
#include "Support/Statistic.h"
#include <algorithm>
#include <set>

using std::vector;

namespace {
  Statistic<> NumFolds          ("dsnode", "Number of nodes completely folded");
  Statistic<> NumCallNodesMerged("dsnode", "Number of call nodes merged");
};

namespace DS {   // TODO: FIXME
  extern TargetData TD;
}
using namespace DS;

//===----------------------------------------------------------------------===//
// DSNode Implementation
//===----------------------------------------------------------------------===//

DSNode::DSNode(enum NodeTy NT, const Type *T)
  : Ty(Type::VoidTy), Size(0), NodeType(NT) {
  // Add the type entry if it is specified...
  if (T) mergeTypeInfo(T, 0);
}

// DSNode copy constructor... do not copy over the referrers list!
DSNode::DSNode(const DSNode &N)
  : Links(N.Links), Globals(N.Globals), Ty(N.Ty), Size(N.Size), 
    NodeType(N.NodeType) {
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

/// foldNodeCompletely - If we determine that this node has some funny
/// behavior happening to it that we cannot represent, we fold it down to a
/// single, completely pessimistic, node.  This node is represented as a
/// single byte with a single TypeEntry of "void".
///
void DSNode::foldNodeCompletely() {
  if (isNodeCompletelyFolded()) return;

  ++NumFolds;

  // We are no longer typed at all...
  Ty = DSTypeRec(Type::VoidTy, true);
  Size = 1;

  // Loop over all of our referrers, making them point to our zero bytes of
  // space.
  for (vector<DSNodeHandle*>::iterator I = Referrers.begin(), E=Referrers.end();
       I != E; ++I)
    (*I)->setOffset(0);

  // If we have links, merge all of our outgoing links together...
  for (unsigned i = 1, e = Links.size(); i < e; ++i)
    Links[0].mergeWith(Links[i]);
  Links.resize(1);
}

/// isNodeCompletelyFolded - Return true if this node has been completely
/// folded down to something that can never be expanded, effectively losing
/// all of the field sensitivity that may be present in the node.
///
bool DSNode::isNodeCompletelyFolded() const {
  return getSize() == 1 && Ty.Ty == Type::VoidTy && Ty.isArray;
}


/// mergeTypeInfo - This method merges the specified type into the current node
/// at the specified offset.  This may update the current node's type record if
/// this gives more information to the node, it may do nothing to the node if
/// this information is already known, or it may merge the node completely (and
/// return true) if the information is incompatible with what is already known.
///
/// This method returns true if the node is completely folded, otherwise false.
///
bool DSNode::mergeTypeInfo(const Type *NewTy, unsigned Offset) {
  // Check to make sure the Size member is up-to-date.  Size can be one of the
  // following:
  //  Size = 0, Ty = Void: Nothing is known about this node.
  //  Size = 0, Ty = FnTy: FunctionPtr doesn't have a size, so we use zero
  //  Size = 1, Ty = Void, Array = 1: The node is collapsed
  //  Otherwise, sizeof(Ty) = Size
  //
  assert(((Size == 0 && Ty.Ty == Type::VoidTy && !Ty.isArray) ||
          (Size == 0 && !Ty.Ty->isSized() && !Ty.isArray) ||
          (Size == 1 && Ty.Ty == Type::VoidTy && Ty.isArray) ||
          (Size == 0 && !Ty.Ty->isSized() && !Ty.isArray) ||
          (TD.getTypeSize(Ty.Ty) == Size)) &&
         "Size member of DSNode doesn't match the type structure!");
  assert(NewTy != Type::VoidTy && "Cannot merge void type into DSNode!");

  if (Offset == 0 && NewTy == Ty.Ty)
    return false;  // This should be a common case, handle it efficiently

  // Return true immediately if the node is completely folded.
  if (isNodeCompletelyFolded()) return true;

  // Figure out how big the new type we're merging in is...
  unsigned NewTySize = NewTy->isSized() ? TD.getTypeSize(NewTy) : 0;

  // Otherwise check to see if we can fold this type into the current node.  If
  // we can't, we fold the node completely, if we can, we potentially update our
  // internal state.
  //
  if (Ty.Ty == Type::VoidTy) {
    // If this is the first type that this node has seen, just accept it without
    // question....
    assert(Offset == 0 && "Cannot have an offset into a void node!");
    assert(Ty.isArray == false && "This shouldn't happen!");
    Ty.Ty = NewTy;
    Size = NewTySize;

    // Calculate the number of outgoing links from this node.
    Links.resize((Size+DS::PointerSize-1) >> DS::PointerShift);
    return false;
  }

  // Handle node expansion case here...
  if (Offset+NewTySize > Size) {
    // It is illegal to grow this node if we have treated it as an array of
    // objects...
    if (Ty.isArray) {
      foldNodeCompletely();
      return true;
    }

    if (Offset) {  // We could handle this case, but we don't for now...
      DEBUG(std::cerr << "UNIMP: Trying to merge a growth type into "
                      << "offset != 0: Collapsing!\n");
      foldNodeCompletely();
      return true;
    }

    // Okay, the situation is nice and simple, we are trying to merge a type in
    // at offset 0 that is bigger than our current type.  Implement this by
    // switching to the new type and then merge in the smaller one, which should
    // hit the other code path here.  If the other code path decides it's not
    // ok, it will collapse the node as appropriate.
    //
    const Type *OldTy = Ty.Ty;
    Ty.Ty = NewTy;
    Size = NewTySize;

    // Must grow links to be the appropriate size...
    Links.resize((Size+DS::PointerSize-1) >> DS::PointerShift);

    // Merge in the old type now... which is guaranteed to be smaller than the
    // "current" type.
    return mergeTypeInfo(OldTy, 0);
  }

  assert(Offset <= Size &&
         "Cannot merge something into a part of our type that doesn't exist!");

  // Find the section of Ty.Ty that NewTy overlaps with... first we find the
  // type that starts at offset Offset.
  //
  unsigned O = 0;
  const Type *SubType = Ty.Ty;
  while (O < Offset) {
    assert(Offset-O < TD.getTypeSize(SubType) && "Offset out of range!");

    switch (SubType->getPrimitiveID()) {
    case Type::StructTyID: {
      const StructType *STy = cast<StructType>(SubType);
      const StructLayout &SL = *TD.getStructLayout(STy);

      unsigned i = 0, e = SL.MemberOffsets.size();
      for (; i+1 < e && SL.MemberOffsets[i+1] <= Offset-O; ++i)
        /* empty */;

      // The offset we are looking for must be in the i'th element...
      SubType = STy->getElementTypes()[i];
      O += SL.MemberOffsets[i];
      break;
    }
    case Type::ArrayTyID: {
      SubType = cast<ArrayType>(SubType)->getElementType();
      unsigned ElSize = TD.getTypeSize(SubType);
      unsigned Remainder = (Offset-O) % ElSize;
      O = Offset-Remainder;
      break;
    }
    default:
      assert(0 && "Unknown type!");
    }
  }

  assert(O == Offset && "Could not achieve the correct offset!");

  // If we found our type exactly, early exit
  if (SubType == NewTy) return false;

  // Okay, so we found the leader type at the offset requested.  Search the list
  // of types that starts at this offset.  If SubType is currently an array or
  // structure, the type desired may actually be the first element of the
  // composite type...
  //
  unsigned SubTypeSize = SubType->isSized() ? TD.getTypeSize(SubType) : 0;
  while (SubType != NewTy) {
    const Type *NextSubType = 0;
    unsigned NextSubTypeSize;
    switch (SubType->getPrimitiveID()) {
    case Type::StructTyID:
      NextSubType = cast<StructType>(SubType)->getElementTypes()[0];
      NextSubTypeSize = TD.getTypeSize(SubType);
      break;
    case Type::ArrayTyID:
      NextSubType = cast<ArrayType>(SubType)->getElementType();
      NextSubTypeSize = TD.getTypeSize(SubType);
      break;
    default: ;
      // fall out 
    }

    if (NextSubType == 0)
      break;   // In the default case, break out of the loop

    if (NextSubTypeSize < NewTySize)
      break;   // Don't allow shrinking to a smaller type than NewTySize
    SubType = NextSubType;
    SubTypeSize = NextSubTypeSize;
  }

  // If we found the type exactly, return it...
  if (SubType == NewTy)
    return false;

  // Check to see if we have a compatible, but different type...
  if (NewTySize == SubTypeSize) {
    // Check to see if this type is obviously convertable... int -> uint f.e.
    if (NewTy->isLosslesslyConvertableTo(SubType))
      return false;

    // Check to see if we have a pointer & integer mismatch going on here,
    // loading a pointer as a long, for example.
    //
    if (SubType->isInteger() && isa<PointerType>(NewTy) ||
        NewTy->isInteger() && isa<PointerType>(SubType))
      return false;

  }


  DEBUG(std::cerr << "MergeTypeInfo Folding OrigTy: " << Ty.Ty
                  << "\n due to:" << NewTy << " @ " << Offset << "!\n"
                  << "SubType: " << SubType << "\n\n");

  foldNodeCompletely();
  return true;
}



// addEdgeTo - Add an edge from the current node to the specified node.  This
// can cause merging of nodes in the graph.
//
void DSNode::addEdgeTo(unsigned Offset, const DSNodeHandle &NH) {
  if (NH.getNode() == 0) return;       // Nothing to do

  DSNodeHandle &ExistingEdge = getLink(Offset);
  if (ExistingEdge.getNode()) {
    // Merge the two nodes...
    ExistingEdge.mergeWith(NH);
  } else {                             // No merging to perform...
    setLink(Offset, NH);               // Just force a link in there...
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

  if (N == this) {
    // We cannot merge two pieces of the same node together, collapse the node
    // completely.
    DEBUG(std::cerr << "Attempting to merge two chunks of"
                    << " the same node together!\n");
    foldNodeCompletely();
    return;
  }

  // Merge the type entries of the two nodes together...
  if (N->Ty.Ty != Type::VoidTy)
    mergeTypeInfo(N->Ty.Ty, Offset);

  // If we are merging a node with a completely folded node, then both nodes are
  // now completely folded.
  //
  if (isNodeCompletelyFolded()) {
    if (!N->isNodeCompletelyFolded())
      N->foldNodeCompletely();
  } else if (N->isNodeCompletelyFolded()) {
    foldNodeCompletely();
    Offset = 0;
  }
  N = NH.getNode();

  if (this == N || N == 0) return;

  // If both nodes are not at offset 0, make sure that we are merging the node
  // at an later offset into the node with the zero offset.
  //
  if (Offset > NH.getOffset()) {
    N->mergeWith(DSNodeHandle(this, Offset), NH.getOffset());
    return;
  } else if (Offset == NH.getOffset() && getSize() < N->getSize()) {
    // If the offsets are the same, merge the smaller node into the bigger node
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

  // Remove all edges pointing at N, causing them to point to 'this' instead.
  // Make sure to adjust their offset, not just the node pointer.
  //
  while (!N->Referrers.empty()) {
    DSNodeHandle &Ref = *N->Referrers.back();
    Ref = DSNodeHandle(this, NOffset+Ref.getOffset());
  }

  // Make all of the outgoing links of N now be outgoing links of this.  This
  // can cause recursive merging!
  //
  for (unsigned i = 0; i < NSize; i += DS::PointerSize) {
    DSNodeHandle &Link = N->getLink(i);
    if (Link.getNode()) {
      addEdgeTo((i+NOffset) % getSize(), Link);

      // It's possible that after adding the new edge that some recursive
      // merging just occured, causing THIS node to get merged into oblivion.
      // If that happens, we must not try to merge any more edges into it!
      //
      if (Size == 0) return;
    }
  }

  // Now that there are no outgoing edges, all of the Links are dead.
  N->Links.clear();
  N->Size = 0;
  N->Ty.Ty = Type::VoidTy;
  N->Ty.isArray = false;

  // Merge the node types
  NodeType |= N->NodeType;
  N->NodeType = DEAD;   // N is now a dead node.

  // Merge the globals list...
  if (!N->Globals.empty()) {
    MergeSortedVectors(Globals, N->Globals);

    // Delete the globals from the old node...
    N->Globals.clear();
  }
}

//===----------------------------------------------------------------------===//
// DSCallSite Implementation
//===----------------------------------------------------------------------===//

// Define here to avoid including iOther.h and BasicBlock.h in DSGraph.h
Function &DSCallSite::getCaller() const {
  return *Inst->getParent()->getParent();
}


//===----------------------------------------------------------------------===//
// DSGraph Implementation
//===----------------------------------------------------------------------===//

DSGraph::DSGraph(const DSGraph &G) : Func(G.Func) {
  std::map<const DSNode*, DSNodeHandle> NodeMap;
  RetNode = cloneInto(G, ScalarMap, NodeMap);
}

DSGraph::DSGraph(const DSGraph &G,
                 std::map<const DSNode*, DSNodeHandle> &NodeMap)
  : Func(G.Func) {
  RetNode = cloneInto(G, ScalarMap, NodeMap);
}

DSGraph::~DSGraph() {
  FunctionCalls.clear();
  ScalarMap.clear();
  RetNode.setNode(0);

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


/// remapLinks - Change all of the Links in the current node according to the
/// specified mapping.
///
void DSNode::remapLinks(std::map<const DSNode*, DSNodeHandle> &OldNodeMap) {
  for (unsigned i = 0, e = Links.size(); i != e; ++i) {
    DSNodeHandle &H = OldNodeMap[Links[i].getNode()];
    Links[i].setNode(H.getNode());
    Links[i].setOffset(Links[i].getOffset()+H.getOffset());
  }
}


// cloneInto - Clone the specified DSGraph into the current graph, returning the
// Return node of the graph.  The translated ScalarMap for the old function is
// filled into the OldValMap member.  If StripAllocas is set to true, Alloca
// markers are removed from the graph, as the graph is being cloned into a
// calling function's graph.
//
DSNodeHandle DSGraph::cloneInto(const DSGraph &G, 
                                std::map<Value*, DSNodeHandle> &OldValMap,
                              std::map<const DSNode*, DSNodeHandle> &OldNodeMap,
                                AllocaBit StripAllocas) {
  assert(OldNodeMap.empty() && "Returned OldNodeMap should be empty!");
  assert(&G != this && "Cannot clone graph into itself!");

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

  // Remove alloca markers as specified
  if (StripAllocas == StripAllocaBit)
    for (unsigned i = FN, e = Nodes.size(); i != e; ++i)
      Nodes[i]->NodeType &= ~DSNode::AllocaNode;

  // Copy the value map... and merge all of the global nodes...
  for (std::map<Value*, DSNodeHandle>::const_iterator I = G.ScalarMap.begin(),
         E = G.ScalarMap.end(); I != E; ++I) {
    DSNodeHandle &H = OldValMap[I->first];
    DSNodeHandle &MappedNode = OldNodeMap[I->second.getNode()];
    H.setNode(MappedNode.getNode());
    H.setOffset(I->second.getOffset()+MappedNode.getOffset());

    if (isa<GlobalValue>(I->first)) {  // Is this a global?
      std::map<Value*, DSNodeHandle>::iterator GVI = ScalarMap.find(I->first);
      if (GVI != ScalarMap.end()) {   // Is the global value in this fn already?
        GVI->second.mergeWith(H);
      } else {
        ScalarMap[I->first] = H;      // Add global pointer to this graph
      }
    }
  }

  // Copy the function calls list...
  unsigned FC = FunctionCalls.size();  // FirstCall
  FunctionCalls.reserve(FC+G.FunctionCalls.size());
  for (unsigned i = 0, ei = G.FunctionCalls.size(); i != ei; ++i)
    FunctionCalls.push_back(DSCallSite(G.FunctionCalls[i], OldNodeMap));

  // Return the returned node pointer...
  DSNodeHandle &MappedRet = OldNodeMap[G.RetNode.getNode()];
  return DSNodeHandle(MappedRet.getNode(),
                      MappedRet.getOffset()+G.RetNode.getOffset());
}

/// mergeInGraph - The method is used for merging graphs together.  If the
/// argument graph is not *this, it makes a clone of the specified graph, then
/// merges the nodes specified in the call site with the formal arguments in the
/// graph.
///
void DSGraph::mergeInGraph(DSCallSite &CS, const DSGraph &Graph,
                           AllocaBit StripAllocas) {
  std::map<Value*, DSNodeHandle> OldValMap;
  DSNodeHandle RetVal;
  std::map<Value*, DSNodeHandle> *ScalarMap = &OldValMap;

  // If this is not a recursive call, clone the graph into this graph...
  if (&Graph != this) {
    // Clone the callee's graph into the current graph, keeping
    // track of where scalars in the old graph _used_ to point,
    // and of the new nodes matching nodes of the old graph.
    std::map<const DSNode*, DSNodeHandle> OldNodeMap;
    
    // The clone call may invalidate any of the vectors in the data
    // structure graph.  Strip locals and don't copy the list of callers
    RetVal = cloneInto(Graph, OldValMap, OldNodeMap, StripAllocas);
    ScalarMap = &OldValMap;
  } else {
    RetVal = getRetNode();
    ScalarMap = &getScalarMap();
  }

  // Merge the return value with the return value of the context...
  RetVal.mergeWith(CS.getRetVal());

  // Resolve all of the function arguments...
  Function &F = Graph.getFunction();
  Function::aiterator AI = F.abegin();
  for (unsigned i = 0, e = CS.getNumPtrArgs(); i != e; ++i, ++AI) {
    // Advance the argument iterator to the first pointer argument...
    while (!isPointerType(AI->getType())) {
      ++AI;
#ifndef NDEBUG
      if (AI == F.aend())
        std::cerr << "Bad call to Function: " << F.getName() << "\n";
#endif
      assert(AI != F.aend() && "# Args provided is not # Args required!");
    }
    
    // Add the link from the argument scalar to the provided value
    DSNodeHandle &NH = (*ScalarMap)[AI];
    assert(NH.getNode() && "Pointer argument without scalarmap entry?");
    NH.mergeWith(CS.getPtrArg(i));
  }
}

#if 0
// cloneGlobalInto - Clone the given global node and all its target links
// (and all their llinks, recursively).
// 
DSNode *DSGraph::cloneGlobalInto(const DSNode *GNode) {
  if (GNode == 0 || GNode->getGlobals().size() == 0) return 0;

  // If a clone has already been created for GNode, return it.
  DSNodeHandle& ValMapEntry = ScalarMap[GNode->getGlobals()[0]];
  if (ValMapEntry != 0)
    return ValMapEntry;

  // Clone the node and update the ValMap.
  DSNode* NewNode = new DSNode(*GNode);
  ValMapEntry = NewNode;                // j=0 case of loop below!
  Nodes.push_back(NewNode);
  for (unsigned j = 1, N = NewNode->getGlobals().size(); j < N; ++j)
    ScalarMap[NewNode->getGlobals()[j]] = NewNode;

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
  for (unsigned i = 0, e = N->getSize(); i < e; i += DS::PointerSize)
    if (DSNode *DSN = N->getLink(i).getNode())
      markIncompleteNode(DSN);
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
      if (isPointerType(I->getType()) && ScalarMap.find(I) != ScalarMap.end())
        markIncompleteNode(ScalarMap[I].getNode());

  // Mark stuff passed into functions calls as being incomplete...
  for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i) {
    DSCallSite &Call = FunctionCalls[i];
    // Then the return value is certainly incomplete!
    markIncompleteNode(Call.getRetVal().getNode());

    // All objects pointed to by function arguments are incomplete though!
    for (unsigned i = 0, e = Call.getNumPtrArgs(); i != e; ++i)
      markIncompleteNode(Call.getPtrArg(i).getNode());
  }

  // Mark all of the nodes pointed to by global nodes as incomplete...
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
    if (Nodes[i]->NodeType & DSNode::GlobalNode) {
      DSNode *N = Nodes[i];
      // FIXME: Make more efficient by looking over Links directly
      for (unsigned i = 0, e = N->getSize(); i < e; i += DS::PointerSize)
        if (DSNode *DSN = N->getLink(i).getNode())
          markIncompleteNode(DSN);
    }
}

// removeRefsToGlobal - Helper function that removes globals from the
// ScalarMap so that the referrer count will go down to zero.
static void removeRefsToGlobal(DSNode* N,
                               std::map<Value*, DSNodeHandle> &ScalarMap) {
  while (!N->getGlobals().empty()) {
    GlobalValue *GV = N->getGlobals().back();
    N->getGlobals().pop_back();      
    ScalarMap.erase(GV);
  }
}


// isNodeDead - This method checks to see if a node is dead, and if it isn't, it
// checks to see if there are simple transformations that it can do to make it
// dead.
//
bool DSGraph::isNodeDead(DSNode *N) {
  // Is it a trivially dead shadow node...
  if (N->getReferrers().empty() &&
      (N->NodeType == 0 || N->NodeType == DSNode::DEAD))
    return true;

  // Is it a function node or some other trivially unused global?
  if ((N->NodeType & ~DSNode::GlobalNode) == 0 && N->getSize() == 0 &&
      N->getReferrers().size() == N->getGlobals().size()) {

    // Remove the globals from the ScalarMap, so that the referrer count will go
    // down to zero.
    removeRefsToGlobal(N, ScalarMap);
    assert(N->getReferrers().empty() && "Referrers should all be gone now!");
    return true;
  }

  return false;
}

static void removeIdenticalCalls(vector<DSCallSite> &Calls,
                                 const std::string &where) {
  // Remove trivially identical function calls
  unsigned NumFns = Calls.size();
  std::sort(Calls.begin(), Calls.end());
  Calls.erase(std::unique(Calls.begin(), Calls.end()),
              Calls.end());

  // Track the number of call nodes merged away...
  NumCallNodesMerged += NumFns-Calls.size();

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
  for (unsigned i = 0, e = N->getSize(); i < e; i += DS::PointerSize)
    if (DSNode *DSN = N->getLink(i).getNode())
      if (!Alive.count(DSN))
        markAlive(DSN, Alive);
}

static bool checkGlobalAlive(DSNode *N, std::set<DSNode*> &Alive,
                             std::set<DSNode*> &Visiting) {
  if (N == 0) return false;

  if (Visiting.count(N)) return false; // terminate recursion on a cycle
  Visiting.insert(N);

  // If any immediate successor is alive, N is alive
  for (unsigned i = 0, e = N->getSize(); i < e; i += DS::PointerSize)
    if (DSNode *DSN = N->getLink(i).getNode())
      if (Alive.count(DSN)) {
        Visiting.erase(N);
        return true;
      }

  // Else if any successor reaches a live node, N is alive
  for (unsigned i = 0, e = N->getSize(); i < e; i += DS::PointerSize)
    if (DSNode *DSN = N->getLink(i).getNode())
      if (checkGlobalAlive(DSN, Alive, Visiting)) {
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
                                 vector<DSCallSite> &Calls,
                                 std::set<DSNode*> &Alive,
                                 bool FilterCalls) {

  // Iterate, marking globals or cast nodes alive until no new live nodes
  // are added to Alive
  std::set<DSNode*> Visiting;           // Used to identify cycles 
  std::set<DSNode*>::iterator I = GlobalNodes.begin(), E = GlobalNodes.end();
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
    bool Recurse = false;
    for (unsigned i = 0, ei = Calls.size(); i < ei; ++i) {
      bool CallIsDead = true, CallHasDeadArg = false;
      DSCallSite &CS = Calls[i];
      for (unsigned j = 0, ej = CS.getNumPtrArgs(); j != ej; ++j)
        if (DSNode *N = CS.getPtrArg(j).getNode()) {
          bool ArgIsDead  = !Alive.count(N);
          CallHasDeadArg |= ArgIsDead;
          CallIsDead     &= ArgIsDead;
        }

      if (DSNode *N = CS.getRetVal().getNode()) {
        bool RetIsDead  = !Alive.count(N);
        CallHasDeadArg |= RetIsDead;
        CallIsDead     &= RetIsDead;
      }

      DSNode *N = CS.getCallee().getNode();
      bool FnIsDead  = !Alive.count(N);
      CallHasDeadArg |= FnIsDead;
      CallIsDead     &= FnIsDead;

      if (!CallIsDead && CallHasDeadArg) {
        // Some node in this call is live and another is dead.
        // Mark all nodes of call as live and iterate once more.
        Recurse = true;
        for (unsigned j = 0, ej = CS.getNumPtrArgs(); j != ej; ++j)
          markAlive(CS.getPtrArg(j).getNode(), Alive);
        markAlive(CS.getRetVal().getNode(), Alive);
        markAlive(CS.getCallee().getNode(), Alive);
      }
    }
    if (Recurse)
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
  vector<DSCallSite> &Calls = G.getFunctionCalls();
  if (FilterCalls) {
    for (unsigned i = 0, e = Calls.size(); i != e; ++i) {
      for (unsigned j = 0, e = Calls[i].getNumPtrArgs(); j != e; ++j)
        if (DSNode *N = Calls[i].getPtrArg(j).getNode())
          GlobalNodes.insert(N);
      if (DSNode *N = Calls[i].getRetVal().getNode())
        GlobalNodes.insert(N);
      if (DSNode *N = Calls[i].getCallee().getNode())
        GlobalNodes.insert(N);
    }
  }

  // Iterate and recurse until no new live node are discovered.
  // This would be a simple iterative loop if function calls were real nodes!
  markGlobalsIteration(GlobalNodes, Calls, Alive, FilterCalls);

  // Free up references to dead globals from the ScalarMap
  std::set<DSNode*>::iterator I = GlobalNodes.begin(), E = GlobalNodes.end();
  for( ; I != E; ++I)
    if (Alive.count(*I) == 0)
      removeRefsToGlobal(*I, G.getScalarMap());

  // Delete dead function calls
  if (FilterCalls)
    for (int ei = Calls.size(), i = ei-1; i >= 0; --i) {
      bool CallIsDead = true;
      for (unsigned j = 0, ej = Calls[i].getNumPtrArgs();
           CallIsDead && j != ej; ++j)
        CallIsDead = Alive.count(Calls[i].getPtrArg(j).getNode()) == 0;
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
  assert((!KeepAllGlobals || KeepCalls) &&  // FIXME: This should be an enum!
         "KeepAllGlobals without KeepCalls is meaningless");

  // Reduce the amount of work we have to do...
  removeTriviallyDeadNodes(KeepAllGlobals);

  // FIXME: Merge nontrivially identical call nodes...

  // Alive - a set that holds all nodes found to be reachable/alive.
  std::set<DSNode*> Alive;

  // If KeepCalls, mark all nodes reachable by call nodes as alive...
  if (KeepCalls)
    for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i) {
      for (unsigned j = 0, e = FunctionCalls[i].getNumPtrArgs(); j != e; ++j)
        markAlive(FunctionCalls[i].getPtrArg(j).getNode(), Alive);
      markAlive(FunctionCalls[i].getRetVal().getNode(), Alive);
      markAlive(FunctionCalls[i].getCallee().getNode(), Alive);
    }

  // Mark all nodes reachable by scalar nodes as alive...
  for (std::map<Value*, DSNodeHandle>::iterator I = ScalarMap.begin(),
         E = ScalarMap.end(); I != E; ++I)
    markAlive(I->second.getNode(), Alive);

  // The return value is alive as well...
  markAlive(RetNode.getNode(), Alive);

  // Mark all globals or cast nodes that can reach a live node as alive.
  // This also marks all nodes reachable from such nodes as alive.
  // Of course, if KeepAllGlobals is specified, they would be live already.

  if (!KeepAllGlobals)
    markGlobalsAlive(*this, Alive, !KeepCalls);

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

#if 0
// Bits used in the next function
static const char ExternalTypeBits = DSNode::GlobalNode | DSNode::HeapNode;


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
    if (DSNode *PrevNode = ScalarMap[OldNode->getGlobals()[j]].getNode()) {
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
          ScalarMap[PrevNode->getGlobals()[k]] = NewNode;
        NewNode->mergeWith(PrevNode);
      }
    } else if (NewNode != 0) {
      ScalarMap[OldNode->getGlobals()[j]] = NewNode; // add the merged node
    }

  // If no existing node was found, clone the node and update the ValMap.
  if (NewNode == 0) {
    NewNode = new DSNode(*OldNode);
    Nodes.push_back(NewNode);
    for (unsigned j = 0, e = NewNode->getNumLinks(); j != e; ++j)
      NewNode->setLink(j, 0);
    for (unsigned j = 0, N = NewNode->getGlobals().size(); j < N; ++j)
      ScalarMap[NewNode->getGlobals()[j]] = NewNode;
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
  vector<DSCallSite >& FromCalls =Graph.FunctionCalls;

  FunctionCalls.reserve(FunctionCalls.size() + FromCalls.size());

  for (int i = 0, ei = FromCalls.size(); i < ei; ++i) {
    DSCallSite& callCopy = FunctionCalls.back();
    callCopy.reserve(FromCalls[i].size());
    for (unsigned j = 0, ej = FromCalls[i].size(); j != ej; ++j)
      callCopy.push_back
        ((FromCalls[i][j] && (FromCalls[i][j]->NodeType & ExternalTypeBits))
         ? cloneNodeInto(FromCalls[i][j], NodeCache, true)
         : 0);
  }

  // remove trivially identical function calls
  removeIdenticalCalls(FunctionCalls, "Globals Graph");
}
#endif

#endif
