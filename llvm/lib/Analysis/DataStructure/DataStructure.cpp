//===- DataStructure.cpp - Implement the core data structure analysis -----===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the core data structure functionality.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DSGraph.h"
#include "llvm/Function.h"
#include "llvm/iOther.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Assembly/Writer.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/STLExtras.h"
#include "Support/Statistic.h"
#include "Support/Timer.h"
#include <algorithm>
using namespace llvm;

namespace {
  Statistic<> NumFolds          ("dsa", "Number of nodes completely folded");
  Statistic<> NumCallNodesMerged("dsa", "Number of call nodes merged");
  Statistic<> NumNodeAllocated  ("dsa", "Number of nodes allocated");
  Statistic<> NumDNE            ("dsa", "Number of nodes removed by reachability");
  Statistic<> NumTrivialDNE     ("dsa", "Number of nodes trivially removed");
  Statistic<> NumTrivialGlobalDNE("dsa", "Number of globals trivially removed");

  cl::opt<bool>
  EnableDSNodeGlobalRootsHack("enable-dsa-globalrootshack", cl::Hidden,
                cl::desc("Make DSA less aggressive when cloning graphs"));
};

#if 1
#define TIME_REGION(VARNAME, DESC) \
   NamedRegionTimer VARNAME(DESC)
#else
#define TIME_REGION(VARNAME, DESC)
#endif

using namespace DS;

DSNode *DSNodeHandle::HandleForwarding() const {
  assert(N->isForwarding() && "Can only be invoked if forwarding!");

  // Handle node forwarding here!
  DSNode *Next = N->ForwardNH.getNode();  // Cause recursive shrinkage
  Offset += N->ForwardNH.getOffset();

  if (--N->NumReferrers == 0) {
    // Removing the last referrer to the node, sever the forwarding link
    N->stopForwarding();
  }

  N = Next;
  N->NumReferrers++;
  if (N->Size <= Offset) {
    assert(N->Size <= 1 && "Forwarded to shrunk but not collapsed node?");
    Offset = 0;
  }
  return N;
}

//===----------------------------------------------------------------------===//
// DSNode Implementation
//===----------------------------------------------------------------------===//

DSNode::DSNode(const Type *T, DSGraph *G)
  : NumReferrers(0), Size(0), ParentGraph(G), Ty(Type::VoidTy), NodeType(0) {
  // Add the type entry if it is specified...
  if (T) mergeTypeInfo(T, 0);
  if (G) G->addNode(this);
  ++NumNodeAllocated;
}

// DSNode copy constructor... do not copy over the referrers list!
DSNode::DSNode(const DSNode &N, DSGraph *G, bool NullLinks)
  : NumReferrers(0), Size(N.Size), ParentGraph(G),
    Ty(N.Ty), Globals(N.Globals), NodeType(N.NodeType) {
  if (!NullLinks)
    Links = N.Links;
  else
    Links.resize(N.Links.size()); // Create the appropriate number of null links
  G->addNode(this);
  ++NumNodeAllocated;
}

/// getTargetData - Get the target data object used to construct this node.
///
const TargetData &DSNode::getTargetData() const {
  return ParentGraph->getTargetData();
}

void DSNode::assertOK() const {
  assert((Ty != Type::VoidTy ||
          Ty == Type::VoidTy && (Size == 0 ||
                                 (NodeType & DSNode::Array))) &&
         "Node not OK!");

  assert(ParentGraph && "Node has no parent?");
  const DSScalarMap &SM = ParentGraph->getScalarMap();
  for (unsigned i = 0, e = Globals.size(); i != e; ++i) {
    assert(SM.count(Globals[i]));
    assert(SM.find(Globals[i])->second.getNode() == this);
  }
}

/// forwardNode - Mark this node as being obsolete, and all references to it
/// should be forwarded to the specified node and offset.
///
void DSNode::forwardNode(DSNode *To, unsigned Offset) {
  assert(this != To && "Cannot forward a node to itself!");
  assert(ForwardNH.isNull() && "Already forwarding from this node!");
  if (To->Size <= 1) Offset = 0;
  assert((Offset < To->Size || (Offset == To->Size && Offset == 0)) &&
         "Forwarded offset is wrong!");
  ForwardNH.setNode(To);
  ForwardNH.setOffset(Offset);
  NodeType = DEAD;
  Size = 0;
  Ty = Type::VoidTy;

  // Remove this node from the parent graph's Nodes list.
  ParentGraph->unlinkNode(this);  
  ParentGraph = 0;
}

// addGlobal - Add an entry for a global value to the Globals list.  This also
// marks the node with the 'G' flag if it does not already have it.
//
void DSNode::addGlobal(GlobalValue *GV) {
  // Keep the list sorted.
  std::vector<GlobalValue*>::iterator I =
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
  if (isNodeCompletelyFolded()) return;  // If this node is already folded...

  ++NumFolds;

  // If this node has a size that is <= 1, we don't need to create a forwarding
  // node.
  if (getSize() <= 1) {
    NodeType |= DSNode::Array;
    Ty = Type::VoidTy;
    Size = 1;
    assert(Links.size() <= 1 && "Size is 1, but has more links?");
    Links.resize(1);
  } else {
    // Create the node we are going to forward to.  This is required because
    // some referrers may have an offset that is > 0.  By forcing them to
    // forward, the forwarder has the opportunity to correct the offset.
    DSNode *DestNode = new DSNode(0, ParentGraph);
    DestNode->NodeType = NodeType|DSNode::Array;
    DestNode->Ty = Type::VoidTy;
    DestNode->Size = 1;
    DestNode->Globals.swap(Globals);
    
    // Start forwarding to the destination node...
    forwardNode(DestNode, 0);
    
    if (!Links.empty()) {
      DestNode->Links.reserve(1);
      
      DSNodeHandle NH(DestNode);
      DestNode->Links.push_back(Links[0]);
      
      // If we have links, merge all of our outgoing links together...
      for (unsigned i = Links.size()-1; i != 0; --i)
        NH.getNode()->Links[0].mergeWith(Links[i]);
      Links.clear();
    } else {
      DestNode->Links.resize(1);
    }
  }
}

/// isNodeCompletelyFolded - Return true if this node has been completely
/// folded down to something that can never be expanded, effectively losing
/// all of the field sensitivity that may be present in the node.
///
bool DSNode::isNodeCompletelyFolded() const {
  return getSize() == 1 && Ty == Type::VoidTy && isArray();
}

namespace {
  /// TypeElementWalker Class - Used for implementation of physical subtyping...
  ///
  class TypeElementWalker {
    struct StackState {
      const Type *Ty;
      unsigned Offset;
      unsigned Idx;
      StackState(const Type *T, unsigned Off = 0)
        : Ty(T), Offset(Off), Idx(0) {}
    };

    std::vector<StackState> Stack;
    const TargetData &TD;
  public:
    TypeElementWalker(const Type *T, const TargetData &td) : TD(td) {
      Stack.push_back(T);
      StepToLeaf();
    }

    bool isDone() const { return Stack.empty(); }
    const Type *getCurrentType()   const { return Stack.back().Ty;     }
    unsigned    getCurrentOffset() const { return Stack.back().Offset; }

    void StepToNextType() {
      PopStackAndAdvance();
      StepToLeaf();
    }

  private:
    /// PopStackAndAdvance - Pop the current element off of the stack and
    /// advance the underlying element to the next contained member.
    void PopStackAndAdvance() {
      assert(!Stack.empty() && "Cannot pop an empty stack!");
      Stack.pop_back();
      while (!Stack.empty()) {
        StackState &SS = Stack.back();
        if (const StructType *ST = dyn_cast<StructType>(SS.Ty)) {
          ++SS.Idx;
          if (SS.Idx != ST->getNumElements()) {
            const StructLayout *SL = TD.getStructLayout(ST);
            SS.Offset += SL->MemberOffsets[SS.Idx]-SL->MemberOffsets[SS.Idx-1];
            return;
          }
          Stack.pop_back();  // At the end of the structure
        } else {
          const ArrayType *AT = cast<ArrayType>(SS.Ty);
          ++SS.Idx;
          if (SS.Idx != AT->getNumElements()) {
            SS.Offset += TD.getTypeSize(AT->getElementType());
            return;
          }
          Stack.pop_back();  // At the end of the array
        }
      }
    }

    /// StepToLeaf - Used by physical subtyping to move to the first leaf node
    /// on the type stack.
    void StepToLeaf() {
      if (Stack.empty()) return;
      while (!Stack.empty() && !Stack.back().Ty->isFirstClassType()) {
        StackState &SS = Stack.back();
        if (const StructType *ST = dyn_cast<StructType>(SS.Ty)) {
          if (ST->getNumElements() == 0) {
            assert(SS.Idx == 0);
            PopStackAndAdvance();
          } else {
            // Step into the structure...
            assert(SS.Idx < ST->getNumElements());
            const StructLayout *SL = TD.getStructLayout(ST);
            Stack.push_back(StackState(ST->getElementType(SS.Idx),
                                       SS.Offset+SL->MemberOffsets[SS.Idx]));
          }
        } else {
          const ArrayType *AT = cast<ArrayType>(SS.Ty);
          if (AT->getNumElements() == 0) {
            assert(SS.Idx == 0);
            PopStackAndAdvance();
          } else {
            // Step into the array...
            assert(SS.Idx < AT->getNumElements());
            Stack.push_back(StackState(AT->getElementType(),
                                       SS.Offset+SS.Idx*
                                       TD.getTypeSize(AT->getElementType())));
          }
        }
      }
    }
  };
} // end anonymous namespace

/// ElementTypesAreCompatible - Check to see if the specified types are
/// "physically" compatible.  If so, return true, else return false.  We only
/// have to check the fields in T1: T2 may be larger than T1.  If AllowLargerT1
/// is true, then we also allow a larger T1.
///
static bool ElementTypesAreCompatible(const Type *T1, const Type *T2,
                                      bool AllowLargerT1, const TargetData &TD){
  TypeElementWalker T1W(T1, TD), T2W(T2, TD);
  
  while (!T1W.isDone() && !T2W.isDone()) {
    if (T1W.getCurrentOffset() != T2W.getCurrentOffset())
      return false;

    const Type *T1 = T1W.getCurrentType();
    const Type *T2 = T2W.getCurrentType();
    if (T1 != T2 && !T1->isLosslesslyConvertibleTo(T2))
      return false;
    
    T1W.StepToNextType();
    T2W.StepToNextType();
  }
  
  return AllowLargerT1 || T1W.isDone();
}


/// mergeTypeInfo - This method merges the specified type into the current node
/// at the specified offset.  This may update the current node's type record if
/// this gives more information to the node, it may do nothing to the node if
/// this information is already known, or it may merge the node completely (and
/// return true) if the information is incompatible with what is already known.
///
/// This method returns true if the node is completely folded, otherwise false.
///
bool DSNode::mergeTypeInfo(const Type *NewTy, unsigned Offset,
                           bool FoldIfIncompatible) {
  const TargetData &TD = getTargetData();
  // Check to make sure the Size member is up-to-date.  Size can be one of the
  // following:
  //  Size = 0, Ty = Void: Nothing is known about this node.
  //  Size = 0, Ty = FnTy: FunctionPtr doesn't have a size, so we use zero
  //  Size = 1, Ty = Void, Array = 1: The node is collapsed
  //  Otherwise, sizeof(Ty) = Size
  //
  assert(((Size == 0 && Ty == Type::VoidTy && !isArray()) ||
          (Size == 0 && !Ty->isSized() && !isArray()) ||
          (Size == 1 && Ty == Type::VoidTy && isArray()) ||
          (Size == 0 && !Ty->isSized() && !isArray()) ||
          (TD.getTypeSize(Ty) == Size)) &&
         "Size member of DSNode doesn't match the type structure!");
  assert(NewTy != Type::VoidTy && "Cannot merge void type into DSNode!");

  if (Offset == 0 && NewTy == Ty)
    return false;  // This should be a common case, handle it efficiently

  // Return true immediately if the node is completely folded.
  if (isNodeCompletelyFolded()) return true;

  // If this is an array type, eliminate the outside arrays because they won't
  // be used anyway.  This greatly reduces the size of large static arrays used
  // as global variables, for example.
  //
  bool WillBeArray = false;
  while (const ArrayType *AT = dyn_cast<ArrayType>(NewTy)) {
    // FIXME: we might want to keep small arrays, but must be careful about
    // things like: [2 x [10000 x int*]]
    NewTy = AT->getElementType();
    WillBeArray = true;
  }

  // Figure out how big the new type we're merging in is...
  unsigned NewTySize = NewTy->isSized() ? TD.getTypeSize(NewTy) : 0;

  // Otherwise check to see if we can fold this type into the current node.  If
  // we can't, we fold the node completely, if we can, we potentially update our
  // internal state.
  //
  if (Ty == Type::VoidTy) {
    // If this is the first type that this node has seen, just accept it without
    // question....
    assert(Offset == 0 && !isArray() &&
           "Cannot have an offset into a void node!");
    Ty = NewTy;
    NodeType &= ~Array;
    if (WillBeArray) NodeType |= Array;
    Size = NewTySize;

    // Calculate the number of outgoing links from this node.
    Links.resize((Size+DS::PointerSize-1) >> DS::PointerShift);
    return false;
  }

  // Handle node expansion case here...
  if (Offset+NewTySize > Size) {
    // It is illegal to grow this node if we have treated it as an array of
    // objects...
    if (isArray()) {
      if (FoldIfIncompatible) foldNodeCompletely();
      return true;
    }

    if (Offset) {  // We could handle this case, but we don't for now...
      std::cerr << "UNIMP: Trying to merge a growth type into "
                << "offset != 0: Collapsing!\n";
      if (FoldIfIncompatible) foldNodeCompletely();
      return true;
    }

    // Okay, the situation is nice and simple, we are trying to merge a type in
    // at offset 0 that is bigger than our current type.  Implement this by
    // switching to the new type and then merge in the smaller one, which should
    // hit the other code path here.  If the other code path decides it's not
    // ok, it will collapse the node as appropriate.
    //
    const Type *OldTy = Ty;
    Ty = NewTy;
    NodeType &= ~Array;
    if (WillBeArray) NodeType |= Array;
    Size = NewTySize;

    // Must grow links to be the appropriate size...
    Links.resize((Size+DS::PointerSize-1) >> DS::PointerShift);

    // Merge in the old type now... which is guaranteed to be smaller than the
    // "current" type.
    return mergeTypeInfo(OldTy, 0);
  }

  assert(Offset <= Size &&
         "Cannot merge something into a part of our type that doesn't exist!");

  // Find the section of Ty that NewTy overlaps with... first we find the
  // type that starts at offset Offset.
  //
  unsigned O = 0;
  const Type *SubType = Ty;
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
      SubType = STy->getElementType(i);
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
      if (FoldIfIncompatible) foldNodeCompletely();
      return true;
    }
  }

  assert(O == Offset && "Could not achieve the correct offset!");

  // If we found our type exactly, early exit
  if (SubType == NewTy) return false;

  // Differing function types don't require us to merge.  They are not values anyway.
  if (isa<FunctionType>(SubType) &&
      isa<FunctionType>(NewTy)) return false;

  unsigned SubTypeSize = SubType->isSized() ? TD.getTypeSize(SubType) : 0;

  // Ok, we are getting desperate now.  Check for physical subtyping, where we
  // just require each element in the node to be compatible.
  if (NewTySize <= SubTypeSize && NewTySize && NewTySize < 256 &&
      SubTypeSize && SubTypeSize < 256 && 
      ElementTypesAreCompatible(NewTy, SubType, !isArray(), TD))
    return false;

  // Okay, so we found the leader type at the offset requested.  Search the list
  // of types that starts at this offset.  If SubType is currently an array or
  // structure, the type desired may actually be the first element of the
  // composite type...
  //
  unsigned PadSize = SubTypeSize; // Size, including pad memory which is ignored
  while (SubType != NewTy) {
    const Type *NextSubType = 0;
    unsigned NextSubTypeSize = 0;
    unsigned NextPadSize = 0;
    switch (SubType->getPrimitiveID()) {
    case Type::StructTyID: {
      const StructType *STy = cast<StructType>(SubType);
      const StructLayout &SL = *TD.getStructLayout(STy);
      if (SL.MemberOffsets.size() > 1)
        NextPadSize = SL.MemberOffsets[1];
      else
        NextPadSize = SubTypeSize;
      NextSubType = STy->getElementType(0);
      NextSubTypeSize = TD.getTypeSize(NextSubType);
      break;
    }
    case Type::ArrayTyID:
      NextSubType = cast<ArrayType>(SubType)->getElementType();
      NextSubTypeSize = TD.getTypeSize(NextSubType);
      NextPadSize = NextSubTypeSize;
      break;
    default: ;
      // fall out 
    }

    if (NextSubType == 0)
      break;   // In the default case, break out of the loop

    if (NextPadSize < NewTySize)
      break;   // Don't allow shrinking to a smaller type than NewTySize
    SubType = NextSubType;
    SubTypeSize = NextSubTypeSize;
    PadSize = NextPadSize;
  }

  // If we found the type exactly, return it...
  if (SubType == NewTy)
    return false;

  // Check to see if we have a compatible, but different type...
  if (NewTySize == SubTypeSize) {
    // Check to see if this type is obviously convertible... int -> uint f.e.
    if (NewTy->isLosslesslyConvertibleTo(SubType))
      return false;

    // Check to see if we have a pointer & integer mismatch going on here,
    // loading a pointer as a long, for example.
    //
    if (SubType->isInteger() && isa<PointerType>(NewTy) ||
        NewTy->isInteger() && isa<PointerType>(SubType))
      return false;
  } else if (NewTySize > SubTypeSize && NewTySize <= PadSize) {
    // We are accessing the field, plus some structure padding.  Ignore the
    // structure padding.
    return false;
  }

  Module *M = 0;
  if (getParentGraph()->getReturnNodes().size())
    M = getParentGraph()->getReturnNodes().begin()->first->getParent();
  DEBUG(std::cerr << "MergeTypeInfo Folding OrigTy: ";
        WriteTypeSymbolic(std::cerr, Ty, M) << "\n due to:";
        WriteTypeSymbolic(std::cerr, NewTy, M) << " @ " << Offset << "!\n"
                  << "SubType: ";
        WriteTypeSymbolic(std::cerr, SubType, M) << "\n\n");

  if (FoldIfIncompatible) foldNodeCompletely();
  return true;
}



// addEdgeTo - Add an edge from the current node to the specified node.  This
// can cause merging of nodes in the graph.
//
void DSNode::addEdgeTo(unsigned Offset, const DSNodeHandle &NH) {
  if (NH.isNull()) return;       // Nothing to do

  DSNodeHandle &ExistingEdge = getLink(Offset);
  if (!ExistingEdge.isNull()) {
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
static void MergeSortedVectors(std::vector<GlobalValue*> &Dest,
                               const std::vector<GlobalValue*> &Src) {
  // By far, the most common cases will be the simple ones.  In these cases,
  // avoid having to allocate a temporary vector...
  //
  if (Src.empty()) {             // Nothing to merge in...
    return;
  } else if (Dest.empty()) {     // Just copy the result in...
    Dest = Src;
  } else if (Src.size() == 1) {  // Insert a single element...
    const GlobalValue *V = Src[0];
    std::vector<GlobalValue*>::iterator I =
      std::lower_bound(Dest.begin(), Dest.end(), V);
    if (I == Dest.end() || *I != Src[0])  // If not already contained...
      Dest.insert(I, Src[0]);
  } else if (Dest.size() == 1) {
    GlobalValue *Tmp = Dest[0];           // Save value in temporary...
    Dest = Src;                           // Copy over list...
    std::vector<GlobalValue*>::iterator I =
      std::lower_bound(Dest.begin(), Dest.end(), Tmp);
    if (I == Dest.end() || *I != Tmp)     // If not already contained...
      Dest.insert(I, Tmp);

  } else {
    // Make a copy to the side of Dest...
    std::vector<GlobalValue*> Old(Dest);
    
    // Make space for all of the type entries now...
    Dest.resize(Dest.size()+Src.size());
    
    // Merge the two sorted ranges together... into Dest.
    std::merge(Old.begin(), Old.end(), Src.begin(), Src.end(), Dest.begin());
    
    // Now erase any duplicate entries that may have accumulated into the 
    // vectors (because they were in both of the input sets)
    Dest.erase(std::unique(Dest.begin(), Dest.end()), Dest.end());
  }
}

void DSNode::mergeGlobals(const std::vector<GlobalValue*> &RHS) {
  MergeSortedVectors(Globals, RHS);
}

// MergeNodes - Helper function for DSNode::mergeWith().
// This function does the hard work of merging two nodes, CurNodeH
// and NH after filtering out trivial cases and making sure that
// CurNodeH.offset >= NH.offset.
// 
// ***WARNING***
// Since merging may cause either node to go away, we must always
// use the node-handles to refer to the nodes.  These node handles are
// automatically updated during merging, so will always provide access
// to the correct node after a merge.
//
void DSNode::MergeNodes(DSNodeHandle& CurNodeH, DSNodeHandle& NH) {
  assert(CurNodeH.getOffset() >= NH.getOffset() &&
         "This should have been enforced in the caller.");

  // Now we know that Offset >= NH.Offset, so convert it so our "Offset" (with
  // respect to NH.Offset) is now zero.  NOffset is the distance from the base
  // of our object that N starts from.
  //
  unsigned NOffset = CurNodeH.getOffset()-NH.getOffset();
  unsigned NSize = NH.getNode()->getSize();

  // If the two nodes are of different size, and the smaller node has the array
  // bit set, collapse!
  if (NSize != CurNodeH.getNode()->getSize()) {
    if (NSize < CurNodeH.getNode()->getSize()) {
      if (NH.getNode()->isArray())
        NH.getNode()->foldNodeCompletely();
    } else if (CurNodeH.getNode()->isArray()) {
      NH.getNode()->foldNodeCompletely();
    }
  }

  // Merge the type entries of the two nodes together...    
  if (NH.getNode()->Ty != Type::VoidTy)
    CurNodeH.getNode()->mergeTypeInfo(NH.getNode()->Ty, NOffset);
  assert(!CurNodeH.getNode()->isDeadNode());

  // If we are merging a node with a completely folded node, then both nodes are
  // now completely folded.
  //
  if (CurNodeH.getNode()->isNodeCompletelyFolded()) {
    if (!NH.getNode()->isNodeCompletelyFolded()) {
      NH.getNode()->foldNodeCompletely();
      assert(NH.getNode() && NH.getOffset() == 0 &&
             "folding did not make offset 0?");
      NOffset = NH.getOffset();
      NSize = NH.getNode()->getSize();
      assert(NOffset == 0 && NSize == 1);
    }
  } else if (NH.getNode()->isNodeCompletelyFolded()) {
    CurNodeH.getNode()->foldNodeCompletely();
    assert(CurNodeH.getNode() && CurNodeH.getOffset() == 0 &&
           "folding did not make offset 0?");
    NOffset = NH.getOffset();
    NSize = NH.getNode()->getSize();
    assert(NOffset == 0 && NSize == 1);
  }

  DSNode *N = NH.getNode();
  if (CurNodeH.getNode() == N || N == 0) return;
  assert(!CurNodeH.getNode()->isDeadNode());

  // Merge the NodeType information.
  CurNodeH.getNode()->NodeType |= N->NodeType;

  // Start forwarding to the new node!
  N->forwardNode(CurNodeH.getNode(), NOffset);
  assert(!CurNodeH.getNode()->isDeadNode());

  // Make all of the outgoing links of N now be outgoing links of CurNodeH.
  //
  for (unsigned i = 0; i < N->getNumLinks(); ++i) {
    DSNodeHandle &Link = N->getLink(i << DS::PointerShift);
    if (Link.getNode()) {
      // Compute the offset into the current node at which to
      // merge this link.  In the common case, this is a linear
      // relation to the offset in the original node (with
      // wrapping), but if the current node gets collapsed due to
      // recursive merging, we must make sure to merge in all remaining
      // links at offset zero.
      unsigned MergeOffset = 0;
      DSNode *CN = CurNodeH.getNode();
      if (CN->Size != 1)
        MergeOffset = ((i << DS::PointerShift)+NOffset) % CN->getSize();
      CN->addEdgeTo(MergeOffset, Link);
    }
  }

  // Now that there are no outgoing edges, all of the Links are dead.
  N->Links.clear();

  // Merge the globals list...
  if (!N->Globals.empty()) {
    CurNodeH.getNode()->mergeGlobals(N->Globals);

    // Delete the globals from the old node...
    std::vector<GlobalValue*>().swap(N->Globals);
  }
}


// mergeWith - Merge this node and the specified node, moving all links to and
// from the argument node into the current node, deleting the node argument.
// Offset indicates what offset the specified node is to be merged into the
// current node.
//
// The specified node may be a null pointer (in which case, we update it to
// point to this node).
//
void DSNode::mergeWith(const DSNodeHandle &NH, unsigned Offset) {
  DSNode *N = NH.getNode();
  if (N == this && NH.getOffset() == Offset)
    return;  // Noop

  // If the RHS is a null node, make it point to this node!
  if (N == 0) {
    NH.mergeWith(DSNodeHandle(this, Offset));
    return;
  }

  assert(!N->isDeadNode() && !isDeadNode());
  assert(!hasNoReferrers() && "Should not try to fold a useless node!");

  if (N == this) {
    // We cannot merge two pieces of the same node together, collapse the node
    // completely.
    DEBUG(std::cerr << "Attempting to merge two chunks of"
                    << " the same node together!\n");
    foldNodeCompletely();
    return;
  }

  // If both nodes are not at offset 0, make sure that we are merging the node
  // at an later offset into the node with the zero offset.
  //
  if (Offset < NH.getOffset()) {
    N->mergeWith(DSNodeHandle(this, Offset), NH.getOffset());
    return;
  } else if (Offset == NH.getOffset() && getSize() < N->getSize()) {
    // If the offsets are the same, merge the smaller node into the bigger node
    N->mergeWith(DSNodeHandle(this, Offset), NH.getOffset());
    return;
  }

  // Ok, now we can merge the two nodes.  Use a static helper that works with
  // two node handles, since "this" may get merged away at intermediate steps.
  DSNodeHandle CurNodeH(this, Offset);
  DSNodeHandle NHCopy(NH);
  DSNode::MergeNodes(CurNodeH, NHCopy);
}


//===----------------------------------------------------------------------===//
// ReachabilityCloner Implementation
//===----------------------------------------------------------------------===//

DSNodeHandle ReachabilityCloner::getClonedNH(const DSNodeHandle &SrcNH) {
  if (SrcNH.isNull()) return DSNodeHandle();
  const DSNode *SN = SrcNH.getNode();

  DSNodeHandle &NH = NodeMap[SN];
  if (!NH.isNull())    // Node already mapped?
    return DSNodeHandle(NH.getNode(), NH.getOffset()+SrcNH.getOffset());

  DSNode *DN = new DSNode(*SN, &Dest, true /* Null out all links */);
  DN->maskNodeTypes(BitsToKeep);
  NH = DN;
  
  // Next, recursively clone all outgoing links as necessary.  Note that
  // adding these links can cause the node to collapse itself at any time, and
  // the current node may be merged with arbitrary other nodes.  For this
  // reason, we must always go through NH.
  DN = 0;
  for (unsigned i = 0, e = SN->getNumLinks(); i != e; ++i) {
    const DSNodeHandle &SrcEdge = SN->getLink(i << DS::PointerShift);
    if (!SrcEdge.isNull()) {
      const DSNodeHandle &DestEdge = getClonedNH(SrcEdge);
      // Compute the offset into the current node at which to
      // merge this link.  In the common case, this is a linear
      // relation to the offset in the original node (with
      // wrapping), but if the current node gets collapsed due to
      // recursive merging, we must make sure to merge in all remaining
      // links at offset zero.
      unsigned MergeOffset = 0;
      DSNode *CN = NH.getNode();
      if (CN->getSize() != 1)
        MergeOffset = ((i << DS::PointerShift)+NH.getOffset()
                       - SrcNH.getOffset()) %CN->getSize();
      CN->addEdgeTo(MergeOffset, DestEdge);
    }
  }
  
  // If this node contains any globals, make sure they end up in the scalar
  // map with the correct offset.
  for (DSNode::global_iterator I = SN->global_begin(), E = SN->global_end();
       I != E; ++I) {
    GlobalValue *GV = *I;
    const DSNodeHandle &SrcGNH = Src.getNodeForValue(GV);
    DSNodeHandle &DestGNH = NodeMap[SrcGNH.getNode()];
    assert(DestGNH.getNode() == NH.getNode() &&"Global mapping inconsistent");
    Dest.getNodeForValue(GV).mergeWith(DSNodeHandle(DestGNH.getNode(),
                                       DestGNH.getOffset()+SrcGNH.getOffset()));
    
    if (CloneFlags & DSGraph::UpdateInlinedGlobals)
      Dest.getInlinedGlobals().insert(GV);
  }

  return DSNodeHandle(NH.getNode(), NH.getOffset()+SrcNH.getOffset());
}

void ReachabilityCloner::merge(const DSNodeHandle &NH,
                               const DSNodeHandle &SrcNH) {
  if (SrcNH.isNull()) return;  // Noop
  if (NH.isNull()) {
    // If there is no destination node, just clone the source and assign the
    // destination node to be it.
    NH.mergeWith(getClonedNH(SrcNH));
    return;
  }

  // Okay, at this point, we know that we have both a destination and a source
  // node that need to be merged.  Check to see if the source node has already
  // been cloned.
  const DSNode *SN = SrcNH.getNode();
  DSNodeHandle &SCNH = NodeMap[SN];  // SourceClonedNodeHandle
  if (!SCNH.isNull()) {   // Node already cloned?
    NH.mergeWith(DSNodeHandle(SCNH.getNode(),
                              SCNH.getOffset()+SrcNH.getOffset()));

    return;  // Nothing to do!
  }

  // Okay, so the source node has not already been cloned.  Instead of creating
  // a new DSNode, only to merge it into the one we already have, try to perform
  // the merge in-place.  The only case we cannot handle here is when the offset
  // into the existing node is less than the offset into the virtual node we are
  // merging in.  In this case, we have to extend the existing node, which
  // requires an allocation anyway.
  DSNode *DN = NH.getNode();   // Make sure the Offset is up-to-date
  if (NH.getOffset() >= SrcNH.getOffset()) {
    if (!DN->isNodeCompletelyFolded()) {
      // Make sure the destination node is folded if the source node is folded.
      if (SN->isNodeCompletelyFolded()) {
        DN->foldNodeCompletely();
        DN = NH.getNode();
      } else if (SN->getSize() != DN->getSize()) {
        // If the two nodes are of different size, and the smaller node has the
        // array bit set, collapse!
        if (SN->getSize() < DN->getSize()) {
          if (SN->isArray()) {
            DN->foldNodeCompletely();
            DN = NH.getNode();
          }
        } else if (DN->isArray()) {
          DN->foldNodeCompletely();
          DN = NH.getNode();
        }
      }
    
      // Merge the type entries of the two nodes together...    
      if (SN->getType() != Type::VoidTy && !DN->isNodeCompletelyFolded()) {
        DN->mergeTypeInfo(SN->getType(), NH.getOffset()-SrcNH.getOffset());
        DN = NH.getNode();
      }
    }

    assert(!DN->isDeadNode());
    
    // Merge the NodeType information.
    DN->mergeNodeFlags(SN->getNodeFlags() & BitsToKeep);

    // Before we start merging outgoing links and updating the scalar map, make
    // sure it is known that this is the representative node for the src node.
    SCNH = DSNodeHandle(DN, NH.getOffset()-SrcNH.getOffset());

    // If the source node contains any globals, make sure they end up in the
    // scalar map with the correct offset.
    if (SN->global_begin() != SN->global_end()) {
      // Update the globals in the destination node itself.
      DN->mergeGlobals(SN->getGlobals());

      // Update the scalar map for the graph we are merging the source node
      // into.
      for (DSNode::global_iterator I = SN->global_begin(), E = SN->global_end();
           I != E; ++I) {
        GlobalValue *GV = *I;
        const DSNodeHandle &SrcGNH = Src.getNodeForValue(GV);
        DSNodeHandle &DestGNH = NodeMap[SrcGNH.getNode()];
        assert(DestGNH.getNode()==NH.getNode() &&"Global mapping inconsistent");
        Dest.getNodeForValue(GV).mergeWith(DSNodeHandle(DestGNH.getNode(),
                                      DestGNH.getOffset()+SrcGNH.getOffset()));
        
        if (CloneFlags & DSGraph::UpdateInlinedGlobals)
          Dest.getInlinedGlobals().insert(GV);
      }
    }
  } else {
    // We cannot handle this case without allocating a temporary node.  Fall
    // back on being simple.
    DSNode *NewDN = new DSNode(*SN, &Dest, true /* Null out all links */);
    NewDN->maskNodeTypes(BitsToKeep);

    unsigned NHOffset = NH.getOffset();
    NH.mergeWith(DSNodeHandle(NewDN, SrcNH.getOffset()));

    assert(NH.getNode() &&
           (NH.getOffset() > NHOffset ||
            (NH.getOffset() == 0 && NH.getNode()->isNodeCompletelyFolded())) &&
           "Merging did not adjust the offset!");

    // Before we start merging outgoing links and updating the scalar map, make
    // sure it is known that this is the representative node for the src node.
    SCNH = DSNodeHandle(NH.getNode(), NH.getOffset()-SrcNH.getOffset());

    // If the source node contained any globals, make sure to create entries 
    // in the scalar map for them!
    for (DSNode::global_iterator I = SN->global_begin(), E = SN->global_end();
         I != E; ++I) {
      GlobalValue *GV = *I;
      const DSNodeHandle &SrcGNH = Src.getNodeForValue(GV);
      DSNodeHandle &DestGNH = NodeMap[SrcGNH.getNode()];
      assert(DestGNH.getNode()==NH.getNode() &&"Global mapping inconsistent");
      assert(SrcGNH.getNode() == SN && "Global mapping inconsistent");
      Dest.getNodeForValue(GV).mergeWith(DSNodeHandle(DestGNH.getNode(),
                                    DestGNH.getOffset()+SrcGNH.getOffset()));
      
      if (CloneFlags & DSGraph::UpdateInlinedGlobals)
        Dest.getInlinedGlobals().insert(GV);
    }
  }


  // Next, recursively merge all outgoing links as necessary.  Note that
  // adding these links can cause the destination node to collapse itself at
  // any time, and the current node may be merged with arbitrary other nodes.
  // For this reason, we must always go through NH.
  DN = 0;
  for (unsigned i = 0, e = SN->getNumLinks(); i != e; ++i) {
    const DSNodeHandle &SrcEdge = SN->getLink(i << DS::PointerShift);
    if (!SrcEdge.isNull()) {
      // Compute the offset into the current node at which to
      // merge this link.  In the common case, this is a linear
      // relation to the offset in the original node (with
      // wrapping), but if the current node gets collapsed due to
      // recursive merging, we must make sure to merge in all remaining
      // links at offset zero.
      unsigned MergeOffset = 0;
      DSNode *CN = SCNH.getNode();
      if (CN->getSize() != 1)
        MergeOffset = ((i << DS::PointerShift)+SCNH.getOffset()) %CN->getSize();
      
      DSNodeHandle &Link = CN->getLink(MergeOffset);
      if (!Link.isNull()) {
        // Perform the recursive merging.  Make sure to create a temporary NH,
        // because the Link can disappear in the process of recursive merging.
        DSNodeHandle Tmp = Link;
        merge(Tmp, SrcEdge);
      } else {
        merge(Link, SrcEdge);
      }
    }
  }
}

/// mergeCallSite - Merge the nodes reachable from the specified src call
/// site into the nodes reachable from DestCS.
void ReachabilityCloner::mergeCallSite(const DSCallSite &DestCS,
                                       const DSCallSite &SrcCS) {
  merge(DestCS.getRetVal(), SrcCS.getRetVal());
  unsigned MinArgs = DestCS.getNumPtrArgs();
  if (SrcCS.getNumPtrArgs() < MinArgs) MinArgs = SrcCS.getNumPtrArgs();
  
  for (unsigned a = 0; a != MinArgs; ++a)
    merge(DestCS.getPtrArg(a), SrcCS.getPtrArg(a));
}


//===----------------------------------------------------------------------===//
// DSCallSite Implementation
//===----------------------------------------------------------------------===//

// Define here to avoid including iOther.h and BasicBlock.h in DSGraph.h
Function &DSCallSite::getCaller() const {
  return *Site.getInstruction()->getParent()->getParent();
}

void DSCallSite::InitNH(DSNodeHandle &NH, const DSNodeHandle &Src,
                        ReachabilityCloner &RC) {
  NH = RC.getClonedNH(Src);
}

//===----------------------------------------------------------------------===//
// DSGraph Implementation
//===----------------------------------------------------------------------===//

/// getFunctionNames - Return a space separated list of the name of the
/// functions in this graph (if any)
std::string DSGraph::getFunctionNames() const {
  switch (getReturnNodes().size()) {
  case 0: return "Globals graph";
  case 1: return getReturnNodes().begin()->first->getName();
  default:
    std::string Return;
    for (DSGraph::ReturnNodesTy::const_iterator I = getReturnNodes().begin();
         I != getReturnNodes().end(); ++I)
      Return += I->first->getName() + " ";
    Return.erase(Return.end()-1, Return.end());   // Remove last space character
    return Return;
  }
}


DSGraph::DSGraph(const DSGraph &G) : GlobalsGraph(0), TD(G.TD) {
  PrintAuxCalls = false;
  NodeMapTy NodeMap;
  cloneInto(G, ScalarMap, ReturnNodes, NodeMap);
}

DSGraph::DSGraph(const DSGraph &G, NodeMapTy &NodeMap)
  : GlobalsGraph(0), TD(G.TD) {
  PrintAuxCalls = false;
  cloneInto(G, ScalarMap, ReturnNodes, NodeMap);
}

DSGraph::~DSGraph() {
  FunctionCalls.clear();
  AuxFunctionCalls.clear();
  InlinedGlobals.clear();
  ScalarMap.clear();
  ReturnNodes.clear();

  // Drop all intra-node references, so that assertions don't fail...
  for (node_iterator NI = node_begin(), E = node_end(); NI != E; ++NI)
    (*NI)->dropAllReferences();

  // Free all of the nodes.
  Nodes.clear();
}

// dump - Allow inspection of graph in a debugger.
void DSGraph::dump() const { print(std::cerr); }


/// remapLinks - Change all of the Links in the current node according to the
/// specified mapping.
///
void DSNode::remapLinks(DSGraph::NodeMapTy &OldNodeMap) {
  for (unsigned i = 0, e = Links.size(); i != e; ++i)
    if (DSNode *N = Links[i].getNode()) {
      DSGraph::NodeMapTy::const_iterator ONMI = OldNodeMap.find(N);
      if (ONMI != OldNodeMap.end()) {
        Links[i].setNode(ONMI->second.getNode());
        Links[i].setOffset(Links[i].getOffset()+ONMI->second.getOffset());
      }
    }
}

/// updateFromGlobalGraph - This function rematerializes global nodes and
/// nodes reachable from them from the globals graph into the current graph.
/// It uses the vector InlinedGlobals to avoid cloning and merging globals that
/// are already up-to-date in the current graph.  In practice, in the TD pass,
/// this is likely to be a large fraction of the live global nodes in each
/// function (since most live nodes are likely to have been brought up-to-date
/// in at _some_ caller or callee).
/// 
void DSGraph::updateFromGlobalGraph() {
  TIME_REGION(X, "updateFromGlobalGraph");
  ReachabilityCloner RC(*this, *GlobalsGraph, 0);

  // Clone the non-up-to-date global nodes into this graph.
  for (DSScalarMap::global_iterator I = getScalarMap().global_begin(),
         E = getScalarMap().global_end(); I != E; ++I)
    if (InlinedGlobals.count(*I) == 0) { // GNode is not up-to-date
      DSScalarMap::iterator It = GlobalsGraph->ScalarMap.find(*I);
      if (It != GlobalsGraph->ScalarMap.end())
        RC.merge(getNodeForValue(*I), It->second);
    }
}

/// cloneInto - Clone the specified DSGraph into the current graph.  The
/// translated ScalarMap for the old function is filled into the OldValMap
/// member, and the translated ReturnNodes map is returned into ReturnNodes.
///
/// The CloneFlags member controls various aspects of the cloning process.
///
void DSGraph::cloneInto(const DSGraph &G, DSScalarMap &OldValMap,
                        ReturnNodesTy &OldReturnNodes, NodeMapTy &OldNodeMap,
                        unsigned CloneFlags) {
  TIME_REGION(X, "cloneInto");
  assert(OldNodeMap.empty() && "Returned OldNodeMap should be empty!");
  assert(&G != this && "Cannot clone graph into itself!");

  // Remove alloca or mod/ref bits as specified...
  unsigned BitsToClear = ((CloneFlags & StripAllocaBit)? DSNode::AllocaNode : 0)
    | ((CloneFlags & StripModRefBits)? (DSNode::Modified | DSNode::Read) : 0)
    | ((CloneFlags & StripIncompleteBit)? DSNode::Incomplete : 0);
  BitsToClear |= DSNode::DEAD;  // Clear dead flag...

  for (node_iterator I = G.node_begin(), E = G.node_end(); I != E; ++I) {
    assert(!(*I)->isForwarding() &&
           "Forward nodes shouldn't be in node list!");
    DSNode *New = new DSNode(**I, this);
    New->maskNodeTypes(~BitsToClear);
    OldNodeMap[*I] = New;
  }
  
#ifndef NDEBUG
  Timer::addPeakMemoryMeasurement();
#endif
  
  // Rewrite the links in the new nodes to point into the current graph now.
  // Note that we don't loop over the node's list to do this.  The problem is
  // that remaping links can cause recursive merging to happen, which means
  // that node_iterator's can get easily invalidated!  Because of this, we
  // loop over the OldNodeMap, which contains all of the new nodes as the
  // .second element of the map elements.  Also note that if we remap a node
  // more than once, we won't break anything.
  for (NodeMapTy::iterator I = OldNodeMap.begin(), E = OldNodeMap.end();
       I != E; ++I)
    I->second.getNode()->remapLinks(OldNodeMap);

  // Copy the scalar map... merging all of the global nodes...
  for (DSScalarMap::const_iterator I = G.ScalarMap.begin(),
         E = G.ScalarMap.end(); I != E; ++I) {
    DSNodeHandle &MappedNode = OldNodeMap[I->second.getNode()];
    DSNodeHandle &H = OldValMap[I->first];
    H.mergeWith(DSNodeHandle(MappedNode.getNode(),
                             I->second.getOffset()+MappedNode.getOffset()));

    // If this is a global, add the global to this fn or merge if already exists
    if (GlobalValue* GV = dyn_cast<GlobalValue>(I->first)) {
      ScalarMap[GV].mergeWith(H);
      if (CloneFlags & DSGraph::UpdateInlinedGlobals)
        InlinedGlobals.insert(GV);
    }
  }

  if (!(CloneFlags & DontCloneCallNodes)) {
    // Copy the function calls list...
    unsigned FC = FunctionCalls.size();  // FirstCall
    FunctionCalls.reserve(FC+G.FunctionCalls.size());
    for (unsigned i = 0, ei = G.FunctionCalls.size(); i != ei; ++i)
      FunctionCalls.push_back(DSCallSite(G.FunctionCalls[i], OldNodeMap));
  }

  if (!(CloneFlags & DontCloneAuxCallNodes)) {
    // Copy the auxiliary function calls list...
    unsigned FC = AuxFunctionCalls.size();  // FirstCall
    AuxFunctionCalls.reserve(FC+G.AuxFunctionCalls.size());
    for (unsigned i = 0, ei = G.AuxFunctionCalls.size(); i != ei; ++i)
      AuxFunctionCalls.push_back(DSCallSite(G.AuxFunctionCalls[i], OldNodeMap));
  }

  // Map the return node pointers over...
  for (ReturnNodesTy::const_iterator I = G.getReturnNodes().begin(),
         E = G.getReturnNodes().end(); I != E; ++I) {
    const DSNodeHandle &Ret = I->second;
    DSNodeHandle &MappedRet = OldNodeMap[Ret.getNode()];
    OldReturnNodes.insert(std::make_pair(I->first,
                          DSNodeHandle(MappedRet.getNode(),
                                       MappedRet.getOffset()+Ret.getOffset())));
  }
}


/// mergeInGraph - The method is used for merging graphs together.  If the
/// argument graph is not *this, it makes a clone of the specified graph, then
/// merges the nodes specified in the call site with the formal arguments in the
/// graph.
///
void DSGraph::mergeInGraph(const DSCallSite &CS, Function &F,
                           const DSGraph &Graph, unsigned CloneFlags) {
  TIME_REGION(X, "mergeInGraph");

  // If this is not a recursive call, clone the graph into this graph...
  if (&Graph != this) {
    // Clone the callee's graph into the current graph, keeping track of where
    // scalars in the old graph _used_ to point, and of the new nodes matching
    // nodes of the old graph.
    ReachabilityCloner RC(*this, Graph, CloneFlags);
    
    // Set up argument bindings
    Function::aiterator AI = F.abegin();
    for (unsigned i = 0, e = CS.getNumPtrArgs(); i != e; ++i, ++AI) {
      // Advance the argument iterator to the first pointer argument...
      while (AI != F.aend() && !isPointerType(AI->getType())) {
        ++AI;
#ifndef NDEBUG  // FIXME: We should merge vararg arguments!
        if (AI == F.aend() && !F.getFunctionType()->isVarArg())
          std::cerr << "Bad call to Function: " << F.getName() << "\n";
#endif
      }
      if (AI == F.aend()) break;
      
      // Add the link from the argument scalar to the provided value.
      RC.merge(CS.getPtrArg(i), Graph.getNodeForValue(AI));
    }
    
    // Map the return node pointer over.
    if (!CS.getRetVal().isNull())
      RC.merge(CS.getRetVal(), Graph.getReturnNodeFor(F));
    
    // If requested, copy the calls or aux-calls lists.
    if (!(CloneFlags & DontCloneCallNodes)) {
      // Copy the function calls list...
      FunctionCalls.reserve(FunctionCalls.size()+Graph.FunctionCalls.size());
      for (unsigned i = 0, ei = Graph.FunctionCalls.size(); i != ei; ++i)
        FunctionCalls.push_back(DSCallSite(Graph.FunctionCalls[i], RC));
    }
    
    if (!(CloneFlags & DontCloneAuxCallNodes)) {
      // Copy the auxiliary function calls list...
      AuxFunctionCalls.reserve(AuxFunctionCalls.size()+
                               Graph.AuxFunctionCalls.size());
      for (unsigned i = 0, ei = Graph.AuxFunctionCalls.size(); i != ei; ++i)
        AuxFunctionCalls.push_back(DSCallSite(Graph.AuxFunctionCalls[i], RC));
    }
    
    // If the user requested it, add the nodes that we need to clone to the
    // RootNodes set.
    if (!EnableDSNodeGlobalRootsHack)
      // FIXME: Why is this not iterating over the globals in the graph??
      for (node_iterator NI = Graph.node_begin(), E = Graph.node_end();
           NI != E; ++NI)
        if (!(*NI)->getGlobals().empty())
          RC.getClonedNH(*NI);
                                                 
  } else {
    DSNodeHandle RetVal = getReturnNodeFor(F);

    // Merge the return value with the return value of the context...
    RetVal.mergeWith(CS.getRetVal());
    
    // Resolve all of the function arguments...
    Function::aiterator AI = F.abegin();
    
    for (unsigned i = 0, e = CS.getNumPtrArgs(); i != e; ++i, ++AI) {
      // Advance the argument iterator to the first pointer argument...
      while (AI != F.aend() && !isPointerType(AI->getType())) {
        ++AI;
#ifndef NDEBUG // FIXME: We should merge varargs arguments!!
        if (AI == F.aend() && !F.getFunctionType()->isVarArg())
          std::cerr << "Bad call to Function: " << F.getName() << "\n";
#endif
      }
      if (AI == F.aend()) break;
      
      // Add the link from the argument scalar to the provided value
      DSNodeHandle &NH = getNodeForValue(AI);
      assert(NH.getNode() && "Pointer argument without scalarmap entry?");
      NH.mergeWith(CS.getPtrArg(i));
    }
  }
}

/// getCallSiteForArguments - Get the arguments and return value bindings for
/// the specified function in the current graph.
///
DSCallSite DSGraph::getCallSiteForArguments(Function &F) const {
  std::vector<DSNodeHandle> Args;

  for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I)
    if (isPointerType(I->getType()))
      Args.push_back(getNodeForValue(I));

  return DSCallSite(CallSite(), getReturnNodeFor(F), &F, Args);
}



// markIncompleteNodes - Mark the specified node as having contents that are not
// known with the current analysis we have performed.  Because a node makes all
// of the nodes it can reach incomplete if the node itself is incomplete, we
// must recursively traverse the data structure graph, marking all reachable
// nodes as incomplete.
//
static void markIncompleteNode(DSNode *N) {
  // Stop recursion if no node, or if node already marked...
  if (N == 0 || N->isIncomplete()) return;

  // Actually mark the node
  N->setIncompleteMarker();

  // Recursively process children...
  for (unsigned i = 0, e = N->getSize(); i < e; i += DS::PointerSize)
    if (DSNode *DSN = N->getLink(i).getNode())
      markIncompleteNode(DSN);
}

static void markIncomplete(DSCallSite &Call) {
  // Then the return value is certainly incomplete!
  markIncompleteNode(Call.getRetVal().getNode());

  // All objects pointed to by function arguments are incomplete!
  for (unsigned i = 0, e = Call.getNumPtrArgs(); i != e; ++i)
    markIncompleteNode(Call.getPtrArg(i).getNode());
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
void DSGraph::markIncompleteNodes(unsigned Flags) {
  // Mark any incoming arguments as incomplete...
  if (Flags & DSGraph::MarkFormalArgs)
    for (ReturnNodesTy::iterator FI = ReturnNodes.begin(), E =ReturnNodes.end();
         FI != E; ++FI) {
      Function &F = *FI->first;
      if (F.getName() != "main")
        for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I)
          if (isPointerType(I->getType()))
            markIncompleteNode(getNodeForValue(I).getNode());
    }

  // Mark stuff passed into functions calls as being incomplete...
  if (!shouldPrintAuxCalls())
    for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i)
      markIncomplete(FunctionCalls[i]);
  else
    for (unsigned i = 0, e = AuxFunctionCalls.size(); i != e; ++i)
      markIncomplete(AuxFunctionCalls[i]);
    

  // Mark all global nodes as incomplete...
  if ((Flags & DSGraph::IgnoreGlobals) == 0)
    for (node_iterator NI = node_begin(), E = node_end(); NI != E; ++NI)
      if ((*NI)->isGlobalNode() && (*NI)->getNumLinks())
        markIncompleteNode(*NI);
}

static inline void killIfUselessEdge(DSNodeHandle &Edge) {
  if (DSNode *N = Edge.getNode())  // Is there an edge?
    if (N->getNumReferrers() == 1)  // Does it point to a lonely node?
      // No interesting info?
      if ((N->getNodeFlags() & ~DSNode::Incomplete) == 0 &&
          N->getType() == Type::VoidTy && !N->isNodeCompletelyFolded())
        Edge.setNode(0);  // Kill the edge!
}

static inline bool nodeContainsExternalFunction(const DSNode *N) {
  const std::vector<GlobalValue*> &Globals = N->getGlobals();
  for (unsigned i = 0, e = Globals.size(); i != e; ++i)
    if (Globals[i]->isExternal())
      return true;
  return false;
}

static void removeIdenticalCalls(std::vector<DSCallSite> &Calls) {
  // Remove trivially identical function calls
  unsigned NumFns = Calls.size();
  std::sort(Calls.begin(), Calls.end());  // Sort by callee as primary key!

#if 1
  // Scan the call list cleaning it up as necessary...
  DSNode   *LastCalleeNode = 0;
  Function *LastCalleeFunc = 0;
  unsigned NumDuplicateCalls = 0;
  bool LastCalleeContainsExternalFunction = false;
  for (unsigned i = 0; i != Calls.size(); ++i) {
    DSCallSite &CS = Calls[i];

    // If the Callee is a useless edge, this must be an unreachable call site,
    // eliminate it.
    if (CS.isIndirectCall() && CS.getCalleeNode()->getNumReferrers() == 1 &&
        CS.getCalleeNode()->getNodeFlags() == 0) {  // No useful info?
#ifndef NDEBUG
      std::cerr << "WARNING: Useless call site found??\n";
#endif
      CS.swap(Calls.back());
      Calls.pop_back();
      --i;
    } else {
      // If the return value or any arguments point to a void node with no
      // information at all in it, and the call node is the only node to point
      // to it, remove the edge to the node (killing the node).
      //
      killIfUselessEdge(CS.getRetVal());
      for (unsigned a = 0, e = CS.getNumPtrArgs(); a != e; ++a)
        killIfUselessEdge(CS.getPtrArg(a));
      
      // If this call site calls the same function as the last call site, and if
      // the function pointer contains an external function, this node will
      // never be resolved.  Merge the arguments of the call node because no
      // information will be lost.
      //
      if ((CS.isDirectCall()   && CS.getCalleeFunc() == LastCalleeFunc) ||
          (CS.isIndirectCall() && CS.getCalleeNode() == LastCalleeNode)) {
        ++NumDuplicateCalls;
        if (NumDuplicateCalls == 1) {
          if (LastCalleeNode)
            LastCalleeContainsExternalFunction =
              nodeContainsExternalFunction(LastCalleeNode);
          else
            LastCalleeContainsExternalFunction = LastCalleeFunc->isExternal();
        }
     
        // It is not clear why, but enabling this code makes DSA really
        // sensitive to node forwarding.  Basically, with this enabled, DSA
        // performs different number of inlinings based on which nodes are
        // forwarding or not.  This is clearly a problem, so this code is
        // disabled until this can be resolved.
#if 1
        if (LastCalleeContainsExternalFunction
#if 0
            ||
            // This should be more than enough context sensitivity!
            // FIXME: Evaluate how many times this is tripped!
            NumDuplicateCalls > 20
#endif
            ) {
          DSCallSite &OCS = Calls[i-1];
          OCS.mergeWith(CS);
          
          // The node will now be eliminated as a duplicate!
          if (CS.getNumPtrArgs() < OCS.getNumPtrArgs())
            CS = OCS;
          else if (CS.getNumPtrArgs() > OCS.getNumPtrArgs())
            OCS = CS;
        }
#endif
      } else {
        if (CS.isDirectCall()) {
          LastCalleeFunc = CS.getCalleeFunc();
          LastCalleeNode = 0;
        } else {
          LastCalleeNode = CS.getCalleeNode();
          LastCalleeFunc = 0;
        }
        NumDuplicateCalls = 0;
      }
    }
  }
#endif
  Calls.erase(std::unique(Calls.begin(), Calls.end()), Calls.end());

  // Track the number of call nodes merged away...
  NumCallNodesMerged += NumFns-Calls.size();

  DEBUG(if (NumFns != Calls.size())
          std::cerr << "Merged " << (NumFns-Calls.size()) << " call nodes.\n";);
}


// removeTriviallyDeadNodes - After the graph has been constructed, this method
// removes all unreachable nodes that are created because they got merged with
// other nodes in the graph.  These nodes will all be trivially unreachable, so
// we don't have to perform any non-trivial analysis here.
//
void DSGraph::removeTriviallyDeadNodes() {
  TIME_REGION(X, "removeTriviallyDeadNodes");

  // Loop over all of the nodes in the graph, calling getNode on each field.
  // This will cause all nodes to update their forwarding edges, causing
  // forwarded nodes to be delete-able.
  for (node_iterator NI = node_begin(), E = node_end(); NI != E; ++NI) {
    DSNode *N = *NI;
    for (unsigned l = 0, e = N->getNumLinks(); l != e; ++l)
      N->getLink(l*N->getPointerSize()).getNode();
  }

  // NOTE: This code is disabled.  Though it should, in theory, allow us to
  // remove more nodes down below, the scan of the scalar map is incredibly
  // expensive for certain programs (with large SCCs).  In the future, if we can
  // make the scalar map scan more efficient, then we can reenable this.
#if 0
  { TIME_REGION(X, "removeTriviallyDeadNodes:scalarmap");

  // Likewise, forward any edges from the scalar nodes.  While we are at it,
  // clean house a bit.
  for (DSScalarMap::iterator I = ScalarMap.begin(),E = ScalarMap.end();I != E;){
    I->second.getNode();
    ++I;
  }
  }
#endif
  bool isGlobalsGraph = !GlobalsGraph;

  for (NodeListTy::iterator NI = Nodes.begin(), E = Nodes.end(); NI != E; ) {
    DSNode &Node = *NI;

    // Do not remove *any* global nodes in the globals graph.
    // This is a special case because such nodes may not have I, M, R flags set.
    if (Node.isGlobalNode() && isGlobalsGraph) {
      ++NI;
      continue;
    }

    if (Node.isComplete() && !Node.isModified() && !Node.isRead()) {
      // This is a useless node if it has no mod/ref info (checked above),
      // outgoing edges (which it cannot, as it is not modified in this
      // context), and it has no incoming edges.  If it is a global node it may
      // have all of these properties and still have incoming edges, due to the
      // scalar map, so we check those now.
      //
      if (Node.getNumReferrers() == Node.getGlobals().size()) {
        const std::vector<GlobalValue*> &Globals = Node.getGlobals();

        // Loop through and make sure all of the globals are referring directly
        // to the node...
        for (unsigned j = 0, e = Globals.size(); j != e; ++j) {
          DSNode *N = getNodeForValue(Globals[j]).getNode();
          assert(N == &Node && "ScalarMap doesn't match globals list!");
        }

        // Make sure NumReferrers still agrees, if so, the node is truly dead.
        if (Node.getNumReferrers() == Globals.size()) {
          for (unsigned j = 0, e = Globals.size(); j != e; ++j)
            ScalarMap.erase(Globals[j]);
          Node.makeNodeDead();
          ++NumTrivialGlobalDNE;
        }
      }
    }

    if (Node.getNodeFlags() == 0 && Node.hasNoReferrers()) {
      // This node is dead!
      NI = Nodes.erase(NI);    // Erase & remove from node list.
      ++NumTrivialDNE;
    } else {
      ++NI;
    }
  }

  removeIdenticalCalls(FunctionCalls);
  removeIdenticalCalls(AuxFunctionCalls);
}


/// markReachableNodes - This method recursively traverses the specified
/// DSNodes, marking any nodes which are reachable.  All reachable nodes it adds
/// to the set, which allows it to only traverse visited nodes once.
///
void DSNode::markReachableNodes(hash_set<DSNode*> &ReachableNodes) {
  if (this == 0) return;
  assert(getForwardNode() == 0 && "Cannot mark a forwarded node!");
  if (ReachableNodes.insert(this).second)        // Is newly reachable?
    for (unsigned i = 0, e = getSize(); i < e; i += DS::PointerSize)
      getLink(i).getNode()->markReachableNodes(ReachableNodes);
}

void DSCallSite::markReachableNodes(hash_set<DSNode*> &Nodes) {
  getRetVal().getNode()->markReachableNodes(Nodes);
  if (isIndirectCall()) getCalleeNode()->markReachableNodes(Nodes);
  
  for (unsigned i = 0, e = getNumPtrArgs(); i != e; ++i)
    getPtrArg(i).getNode()->markReachableNodes(Nodes);
}

// CanReachAliveNodes - Simple graph walker that recursively traverses the graph
// looking for a node that is marked alive.  If an alive node is found, return
// true, otherwise return false.  If an alive node is reachable, this node is
// marked as alive...
//
static bool CanReachAliveNodes(DSNode *N, hash_set<DSNode*> &Alive,
                               hash_set<DSNode*> &Visited,
                               bool IgnoreGlobals) {
  if (N == 0) return false;
  assert(N->getForwardNode() == 0 && "Cannot mark a forwarded node!");

  // If this is a global node, it will end up in the globals graph anyway, so we
  // don't need to worry about it.
  if (IgnoreGlobals && N->isGlobalNode()) return false;

  // If we know that this node is alive, return so!
  if (Alive.count(N)) return true;

  // Otherwise, we don't think the node is alive yet, check for infinite
  // recursion.
  if (Visited.count(N)) return false;  // Found a cycle
  Visited.insert(N);   // No recursion, insert into Visited...

  for (unsigned i = 0, e = N->getSize(); i < e; i += DS::PointerSize)
    if (CanReachAliveNodes(N->getLink(i).getNode(), Alive, Visited,
                           IgnoreGlobals)) {
      N->markReachableNodes(Alive);
      return true;
    }
  return false;
}

// CallSiteUsesAliveArgs - Return true if the specified call site can reach any
// alive nodes.
//
static bool CallSiteUsesAliveArgs(DSCallSite &CS, hash_set<DSNode*> &Alive,
                                  hash_set<DSNode*> &Visited,
                                  bool IgnoreGlobals) {
  if (CanReachAliveNodes(CS.getRetVal().getNode(), Alive, Visited,
                         IgnoreGlobals))
    return true;
  if (CS.isIndirectCall() &&
      CanReachAliveNodes(CS.getCalleeNode(), Alive, Visited, IgnoreGlobals))
    return true;
  for (unsigned i = 0, e = CS.getNumPtrArgs(); i != e; ++i)
    if (CanReachAliveNodes(CS.getPtrArg(i).getNode(), Alive, Visited,
                           IgnoreGlobals))
      return true;
  return false;
}

// removeDeadNodes - Use a more powerful reachability analysis to eliminate
// subgraphs that are unreachable.  This often occurs because the data
// structure doesn't "escape" into it's caller, and thus should be eliminated
// from the caller's graph entirely.  This is only appropriate to use when
// inlining graphs.
//
void DSGraph::removeDeadNodes(unsigned Flags) {
  DEBUG(AssertGraphOK(); if (GlobalsGraph) GlobalsGraph->AssertGraphOK());

  // Reduce the amount of work we have to do... remove dummy nodes left over by
  // merging...
  removeTriviallyDeadNodes();

  TIME_REGION(X, "removeDeadNodes");

  // FIXME: Merge non-trivially identical call nodes...

  // Alive - a set that holds all nodes found to be reachable/alive.
  hash_set<DSNode*> Alive;
  std::vector<std::pair<Value*, DSNode*> > GlobalNodes;

  // Copy and merge all information about globals to the GlobalsGraph if this is
  // not a final pass (where unreachable globals are removed).
  //
  // Strip all alloca bits since the current function is only for the BU pass.
  // Strip all incomplete bits since they are short-lived properties and they
  // will be correctly computed when rematerializing nodes into the functions.
  //
  ReachabilityCloner GGCloner(*GlobalsGraph, *this, DSGraph::StripAllocaBit |
                              DSGraph::StripIncompleteBit);

  // Mark all nodes reachable by (non-global) scalar nodes as alive...
  { TIME_REGION(Y, "removeDeadNodes:scalarscan");
  for (DSScalarMap::iterator I = ScalarMap.begin(), E = ScalarMap.end(); I !=E;)
    if (isa<GlobalValue>(I->first)) {             // Keep track of global nodes
      assert(I->second.getNode() && "Null global node?");
      assert(I->second.getNode()->isGlobalNode() && "Should be a global node!");
      GlobalNodes.push_back(std::make_pair(I->first, I->second.getNode()));

      // Make sure that all globals are cloned over as roots.
      if (!(Flags & DSGraph::RemoveUnreachableGlobals)) {
        DSGraph::ScalarMapTy::iterator SMI = 
          GlobalsGraph->getScalarMap().find(I->first);
        if (SMI != GlobalsGraph->getScalarMap().end())
          GGCloner.merge(SMI->second, I->second);
        else
          GGCloner.getClonedNH(I->second);
      }
      ++I;
    } else {
      DSNode *N = I->second.getNode();
#if 0
      // Check to see if this is a worthless node generated for non-pointer
      // values, such as integers.  Consider an addition of long types: A+B.
      // Assuming we can track all uses of the value in this context, and it is
      // NOT used as a pointer, we can delete the node.  We will be able to
      // detect this situation if the node pointed to ONLY has Unknown bit set
      // in the node.  In this case, the node is not incomplete, does not point
      // to any other nodes (no mod/ref bits set), and is therefore
      // uninteresting for data structure analysis.  If we run across one of
      // these, prune the scalar pointing to it.
      //
      if (N->getNodeFlags() == DSNode::UnknownNode && !isa<Argument>(I->first))
        ScalarMap.erase(I++);
      else {
#endif
        N->markReachableNodes(Alive);
        ++I;
      //}
    }
  }

  // The return values are alive as well.
  for (ReturnNodesTy::iterator I = ReturnNodes.begin(), E = ReturnNodes.end();
       I != E; ++I)
    I->second.getNode()->markReachableNodes(Alive);

  // Mark any nodes reachable by primary calls as alive...
  for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i)
    FunctionCalls[i].markReachableNodes(Alive);


  // Now find globals and aux call nodes that are already live or reach a live
  // value (which makes them live in turn), and continue till no more are found.
  // 
  bool Iterate;
  hash_set<DSNode*> Visited;
  std::vector<unsigned char> AuxFCallsAlive(AuxFunctionCalls.size());
  do {
    Visited.clear();
    // If any global node points to a non-global that is "alive", the global is
    // "alive" as well...  Remove it from the GlobalNodes list so we only have
    // unreachable globals in the list.
    //
    Iterate = false;
    if (!(Flags & DSGraph::RemoveUnreachableGlobals))
      for (unsigned i = 0; i != GlobalNodes.size(); ++i)
        if (CanReachAliveNodes(GlobalNodes[i].second, Alive, Visited, 
                               Flags & DSGraph::RemoveUnreachableGlobals)) {
          std::swap(GlobalNodes[i--], GlobalNodes.back()); // Move to end to...
          GlobalNodes.pop_back();                          // erase efficiently
          Iterate = true;
        }

    // Mark only unresolvable call nodes for moving to the GlobalsGraph since
    // call nodes that get resolved will be difficult to remove from that graph.
    // The final unresolved call nodes must be handled specially at the end of
    // the BU pass (i.e., in main or other roots of the call graph).
    for (unsigned i = 0, e = AuxFunctionCalls.size(); i != e; ++i)
      if (!AuxFCallsAlive[i] &&
          (AuxFunctionCalls[i].isIndirectCall()
           || CallSiteUsesAliveArgs(AuxFunctionCalls[i], Alive, Visited,
                                  Flags & DSGraph::RemoveUnreachableGlobals))) {
        AuxFunctionCalls[i].markReachableNodes(Alive);
        AuxFCallsAlive[i] = true;
        Iterate = true;
      }
  } while (Iterate);

  // Move dead aux function calls to the end of the list
  unsigned CurIdx = 0;
  for (unsigned i = 0, e = AuxFunctionCalls.size(); i != e; ++i)
    if (AuxFCallsAlive[i])
      AuxFunctionCalls[CurIdx++].swap(AuxFunctionCalls[i]);

  // Copy and merge all global nodes and dead aux call nodes into the
  // GlobalsGraph, and all nodes reachable from those nodes
  // 
  if (!(Flags & DSGraph::RemoveUnreachableGlobals)) {
    // Copy the unreachable call nodes to the globals graph, updating their
    // target pointers using the GGCloner
    for (unsigned i = CurIdx, e = AuxFunctionCalls.size(); i != e; ++i)
      GlobalsGraph->AuxFunctionCalls.push_back(DSCallSite(AuxFunctionCalls[i],
                                                          GGCloner));
  }
  // Crop all the useless ones out...
  AuxFunctionCalls.erase(AuxFunctionCalls.begin()+CurIdx,
                         AuxFunctionCalls.end());

  // We are finally done with the GGCloner so we can destroy it.
  GGCloner.destroy();

  // At this point, any nodes which are visited, but not alive, are nodes
  // which can be removed.  Loop over all nodes, eliminating completely
  // unreachable nodes.
  //
  std::vector<DSNode*> DeadNodes;
  DeadNodes.reserve(Nodes.size());
  for (NodeListTy::iterator NI = Nodes.begin(), E = Nodes.end(); NI != E;)
    if (!Alive.count(NI)) {
      ++NumDNE;
      DSNode *N = Nodes.remove(NI++);
      DeadNodes.push_back(N);
      N->dropAllReferences();
    } else {
      assert(NI->getForwardNode() == 0 && "Alive forwarded node?");
      ++NI;
    }

  // Remove all unreachable globals from the ScalarMap.
  // If flag RemoveUnreachableGlobals is set, GlobalNodes has only dead nodes.
  // In either case, the dead nodes will not be in the set Alive.
  for (unsigned i = 0, e = GlobalNodes.size(); i != e; ++i)
    if (!Alive.count(GlobalNodes[i].second))
      ScalarMap.erase(GlobalNodes[i].first);
    else
      assert((Flags & DSGraph::RemoveUnreachableGlobals) && "non-dead global");

  // Delete all dead nodes now since their referrer counts are zero.
  for (unsigned i = 0, e = DeadNodes.size(); i != e; ++i)
    delete DeadNodes[i];

  DEBUG(AssertGraphOK(); GlobalsGraph->AssertGraphOK());
}

void DSGraph::AssertGraphOK() const {
  for (node_iterator NI = node_begin(), E = node_end(); NI != E; ++NI)
    (*NI)->assertOK();

  for (ScalarMapTy::const_iterator I = ScalarMap.begin(),
         E = ScalarMap.end(); I != E; ++I) {
    assert(I->second.getNode() && "Null node in scalarmap!");
    AssertNodeInGraph(I->second.getNode());
    if (GlobalValue *GV = dyn_cast<GlobalValue>(I->first)) {
      assert(I->second.getNode()->isGlobalNode() &&
             "Global points to node, but node isn't global?");
      AssertNodeContainsGlobal(I->second.getNode(), GV);
    }
  }
  AssertCallNodesInGraph();
  AssertAuxCallNodesInGraph();
}

/// computeNodeMapping - Given roots in two different DSGraphs, traverse the
/// nodes reachable from the two graphs, computing the mapping of nodes from
/// the first to the second graph.
///
void DSGraph::computeNodeMapping(const DSNodeHandle &NH1,
                                 const DSNodeHandle &NH2, NodeMapTy &NodeMap,
                                 bool StrictChecking) {
  DSNode *N1 = NH1.getNode(), *N2 = NH2.getNode();
  if (N1 == 0 || N2 == 0) return;

  DSNodeHandle &Entry = NodeMap[N1];
  if (Entry.getNode()) {
    // Termination of recursion!
    assert(!StrictChecking ||
           (Entry.getNode() == N2 &&
            Entry.getOffset() == (NH2.getOffset()-NH1.getOffset())) &&
           "Inconsistent mapping detected!");
    return;
  }
  
  Entry.setNode(N2);
  Entry.setOffset(NH2.getOffset()-NH1.getOffset());

  // Loop over all of the fields that N1 and N2 have in common, recursively
  // mapping the edges together now.
  int N2Idx = NH2.getOffset()-NH1.getOffset();
  unsigned N2Size = N2->getSize();
  for (unsigned i = 0, e = N1->getSize(); i < e; i += DS::PointerSize)
    if (unsigned(N2Idx)+i < N2Size)
      computeNodeMapping(N1->getLink(i), N2->getLink(N2Idx+i), NodeMap);
}
