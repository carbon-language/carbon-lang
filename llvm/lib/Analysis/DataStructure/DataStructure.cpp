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
#include "llvm/Assembly/Writer.h"
#include "Support/STLExtras.h"
#include "Support/Statistic.h"
#include "Support/Timer.h"
#include <algorithm>

namespace {
  Statistic<> NumFolds          ("dsnode", "Number of nodes completely folded");
  Statistic<> NumCallNodesMerged("dsnode", "Number of call nodes merged");
};

namespace DS {   // TODO: FIXME
  extern TargetData TD;
}
using namespace DS;

DSNode *DSNodeHandle::HandleForwarding() const {
  assert(!N->ForwardNH.isNull() && "Can only be invoked if forwarding!");

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
  G->getNodes().push_back(this);
}

// DSNode copy constructor... do not copy over the referrers list!
DSNode::DSNode(const DSNode &N, DSGraph *G)
  : NumReferrers(0), Size(N.Size), ParentGraph(G),
    Ty(N.Ty), Links(N.Links), Globals(N.Globals), NodeType(N.NodeType) {
  G->getNodes().push_back(this);
}

void DSNode::assertOK() const {
  assert((Ty != Type::VoidTy ||
          Ty == Type::VoidTy && (Size == 0 ||
                                 (NodeType & DSNode::Array))) &&
         "Node not OK!");
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

  // Create the node we are going to forward to...
  DSNode *DestNode = new DSNode(0, ParentGraph);
  DestNode->NodeType = NodeType|DSNode::Array;
  DestNode->Ty = Type::VoidTy;
  DestNode->Size = 1;
  DestNode->Globals.swap(Globals);

  // Start forwarding to the destination node...
  forwardNode(DestNode, 0);
  
  if (Links.size()) {
    DestNode->Links.push_back(Links[0]);
    DSNodeHandle NH(DestNode);

    // If we have links, merge all of our outgoing links together...
    for (unsigned i = Links.size()-1; i != 0; --i)
      NH.getNode()->Links[0].mergeWith(Links[i]);
    Links.clear();
  } else {
    DestNode->Links.resize(1);
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
  public:
    TypeElementWalker(const Type *T) {
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
          if (SS.Idx != ST->getElementTypes().size()) {
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
          if (ST->getElementTypes().empty()) {
            assert(SS.Idx == 0);
            PopStackAndAdvance();
          } else {
            // Step into the structure...
            assert(SS.Idx < ST->getElementTypes().size());
            const StructLayout *SL = TD.getStructLayout(ST);
            Stack.push_back(StackState(ST->getElementTypes()[SS.Idx],
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
}

/// ElementTypesAreCompatible - Check to see if the specified types are
/// "physically" compatible.  If so, return true, else return false.  We only
/// have to check the fields in T1: T2 may be larger than T1.
///
static bool ElementTypesAreCompatible(const Type *T1, const Type *T2) {
  TypeElementWalker T1W(T1), T2W(T2);
  
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
  
  return T1W.isDone();
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
    assert(Offset == 0 && "Cannot have an offset into a void node!");
    assert(!isArray() && "This shouldn't happen!");
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
      if (FoldIfIncompatible) foldNodeCompletely();
      return true;
    }
  }

  assert(O == Offset && "Could not achieve the correct offset!");

  // If we found our type exactly, early exit
  if (SubType == NewTy) return false;

  unsigned SubTypeSize = SubType->isSized() ? TD.getTypeSize(SubType) : 0;

  // Ok, we are getting desperate now.  Check for physical subtyping, where we
  // just require each element in the node to be compatible.
  if (NewTySize <= SubTypeSize && NewTySize && NewTySize < 256 &&
      SubTypeSize && SubTypeSize < 256 && 
      ElementTypesAreCompatible(NewTy, SubType))
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
      NextSubType = STy->getElementTypes()[0];
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


// MergeNodes() - Helper function for DSNode::mergeWith().
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

  // Merge the NodeType information...
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
    MergeSortedVectors(CurNodeH.getNode()->Globals, N->Globals);

    // Delete the globals from the old node...
    std::vector<GlobalValue*>().swap(N->Globals);
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
// DSCallSite Implementation
//===----------------------------------------------------------------------===//

// Define here to avoid including iOther.h and BasicBlock.h in DSGraph.h
Function &DSCallSite::getCaller() const {
  return *Inst->getParent()->getParent();
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


DSGraph::DSGraph(const DSGraph &G) : GlobalsGraph(0) {
  PrintAuxCalls = false;
  NodeMapTy NodeMap;
  cloneInto(G, ScalarMap, ReturnNodes, NodeMap);
}

DSGraph::DSGraph(const DSGraph &G, NodeMapTy &NodeMap)
  : GlobalsGraph(0) {
  PrintAuxCalls = false;
  cloneInto(G, ScalarMap, ReturnNodes, NodeMap);
}

DSGraph::~DSGraph() {
  FunctionCalls.clear();
  AuxFunctionCalls.clear();
  ScalarMap.clear();
  ReturnNodes.clear();

  // Drop all intra-node references, so that assertions don't fail...
  std::for_each(Nodes.begin(), Nodes.end(),
                std::mem_fun(&DSNode::dropAllReferences));

  // Delete all of the nodes themselves...
  std::for_each(Nodes.begin(), Nodes.end(), deleter<DSNode>);
}

// dump - Allow inspection of graph in a debugger.
void DSGraph::dump() const { print(std::cerr); }


/// remapLinks - Change all of the Links in the current node according to the
/// specified mapping.
///
void DSNode::remapLinks(DSGraph::NodeMapTy &OldNodeMap) {
  for (unsigned i = 0, e = Links.size(); i != e; ++i) {
    DSNodeHandle &H = OldNodeMap[Links[i].getNode()];
    Links[i].setNode(H.getNode());
    Links[i].setOffset(Links[i].getOffset()+H.getOffset());
  }
}


/// cloneInto - Clone the specified DSGraph into the current graph.  The
/// translated ScalarMap for the old function is filled into the OldValMap
/// member, and the translated ReturnNodes map is returned into ReturnNodes.
///
/// The CloneFlags member controls various aspects of the cloning process.
///
void DSGraph::cloneInto(const DSGraph &G, ScalarMapTy &OldValMap,
                        ReturnNodesTy &OldReturnNodes, NodeMapTy &OldNodeMap,
                        unsigned CloneFlags) {
  assert(OldNodeMap.empty() && "Returned OldNodeMap should be empty!");
  assert(&G != this && "Cannot clone graph into itself!");

  unsigned FN = Nodes.size();           // First new node...

  // Duplicate all of the nodes, populating the node map...
  Nodes.reserve(FN+G.Nodes.size());

  // Remove alloca or mod/ref bits as specified...
  unsigned BitsToClear =((CloneFlags & StripAllocaBit) ? DSNode::AllocaNode : 0)
    | ((CloneFlags & StripModRefBits) ? (DSNode::Modified | DSNode::Read) : 0);
  BitsToClear |= DSNode::DEAD;  // Clear dead flag...
  for (unsigned i = 0, e = G.Nodes.size(); i != e; ++i) {
    DSNode *Old = G.Nodes[i];
    DSNode *New = new DSNode(*Old, this);
    New->maskNodeTypes(~BitsToClear);
    OldNodeMap[Old] = New;
  }

#ifndef NDEBUG
  Timer::addPeakMemoryMeasurement();
#endif

  // Rewrite the links in the new nodes to point into the current graph now.
  for (unsigned i = FN, e = Nodes.size(); i != e; ++i)
    Nodes[i]->remapLinks(OldNodeMap);

  // Copy the scalar map... merging all of the global nodes...
  for (ScalarMapTy::const_iterator I = G.ScalarMap.begin(),
         E = G.ScalarMap.end(); I != E; ++I) {
    DSNodeHandle &MappedNode = OldNodeMap[I->second.getNode()];
    DSNodeHandle &H = OldValMap[I->first];
    H.mergeWith(DSNodeHandle(MappedNode.getNode(),
                             I->second.getOffset()+MappedNode.getOffset()));

    // If this is a global, add the global to this fn or merge if already exists
    if (isa<GlobalValue>(I->first))
      ScalarMap[I->first].mergeWith(H);
  }

  if (!(CloneFlags & DontCloneCallNodes)) {
    // Copy the function calls list...
    unsigned FC = FunctionCalls.size();  // FirstCall
    FunctionCalls.reserve(FC+G.FunctionCalls.size());
    for (unsigned i = 0, ei = G.FunctionCalls.size(); i != ei; ++i)
      FunctionCalls.push_back(DSCallSite(G.FunctionCalls[i], OldNodeMap));
  }

  if (!(CloneFlags & DontCloneAuxCallNodes)) {
    // Copy the auxillary function calls list...
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
  ScalarMapTy OldValMap, *ScalarMap;
  DSNodeHandle RetVal;

  // If this is not a recursive call, clone the graph into this graph...
  if (&Graph != this) {
    // Clone the callee's graph into the current graph, keeping
    // track of where scalars in the old graph _used_ to point,
    // and of the new nodes matching nodes of the old graph.
    NodeMapTy OldNodeMap;
    
    // The clone call may invalidate any of the vectors in the data
    // structure graph.  Strip locals and don't copy the list of callers
    ReturnNodesTy OldRetNodes;
    cloneInto(Graph, OldValMap, OldRetNodes, OldNodeMap, CloneFlags);

    // We need to map the arguments for the function to the cloned nodes old
    // argument values.  Do this now.
    RetVal = OldRetNodes[&F];
    ScalarMap = &OldValMap;
  } else {
    RetVal = getReturnNodeFor(F);
    ScalarMap = &getScalarMap();
  }

  // Merge the return value with the return value of the context...
  RetVal.mergeWith(CS.getRetVal());

  // Resolve all of the function arguments...
  Function::aiterator AI = F.abegin();

  for (unsigned i = 0, e = CS.getNumPtrArgs(); i != e; ++i, ++AI) {
    // Advance the argument iterator to the first pointer argument...
    while (AI != F.aend() && !isPointerType(AI->getType())) {
      ++AI;
#ifndef NDEBUG
      if (AI == F.aend())
        std::cerr << "Bad call to Function: " << F.getName() << "\n";
#endif
    }
    if (AI == F.aend()) break;
    
    // Add the link from the argument scalar to the provided value
    assert(ScalarMap->count(AI) && "Argument not in scalar map?");
    DSNodeHandle &NH = (*ScalarMap)[AI];
    assert(NH.getNode() && "Pointer argument without scalarmap entry?");
    NH.mergeWith(CS.getPtrArg(i));
  }
}

/// getCallSiteForArguments - Get the arguments and return value bindings for
/// the specified function in the current graph.
///
DSCallSite DSGraph::getCallSiteForArguments(Function &F) const {
  std::vector<DSNodeHandle> Args;

  for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I)
    if (isPointerType(I->getType()))
      Args.push_back(getScalarMap().find(I)->second);

  return DSCallSite(*(CallInst*)0, getReturnNodeFor(F), &F, Args);
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

  // Recusively process children...
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
          if (isPointerType(I->getType()) &&
              ScalarMap.find(I) != ScalarMap.end())
            markIncompleteNode(ScalarMap[I].getNode());
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
    for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
      if (Nodes[i]->isGlobalNode() && Nodes[i]->getNumLinks())
        markIncompleteNode(Nodes[i]);
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
      std::cerr << "WARNING: Useless call site found??\n";
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
        
#if 1
        if (LastCalleeContainsExternalFunction ||
            // This should be more than enough context sensitivity!
            // FIXME: Evaluate how many times this is tripped!
            NumDuplicateCalls > 20) {
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

  Calls.erase(std::unique(Calls.begin(), Calls.end()),
              Calls.end());

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
  removeIdenticalCalls(FunctionCalls);
  removeIdenticalCalls(AuxFunctionCalls);

  for (unsigned i = 0; i != Nodes.size(); ++i) {
    DSNode *Node = Nodes[i];
    if (Node->isComplete() && !Node->isModified() && !Node->isRead()) {
      // This is a useless node if it has no mod/ref info (checked above),
      // outgoing edges (which it cannot, as it is not modified in this
      // context), and it has no incoming edges.  If it is a global node it may
      // have all of these properties and still have incoming edges, due to the
      // scalar map, so we check those now.
      //
      if (Node->getNumReferrers() == Node->getGlobals().size()) {
        const std::vector<GlobalValue*> &Globals = Node->getGlobals();

        // Loop through and make sure all of the globals are referring directly
        // to the node...
        for (unsigned j = 0, e = Globals.size(); j != e; ++j) {
          DSNode *N = ScalarMap.find(Globals[j])->second.getNode();
          assert(N == Node && "ScalarMap doesn't match globals list!");
        }

        // Make sure NumReferrers still agrees, if so, the node is truly dead.
        if (Node->getNumReferrers() == Globals.size()) {
          for (unsigned j = 0, e = Globals.size(); j != e; ++j)
            ScalarMap.erase(Globals[j]);
          Node->makeNodeDead();
        }
      }
    }

    if (Node->getNodeFlags() == 0 && Node->hasNoReferrers()) {
      // This node is dead!
      delete Node;                        // Free memory...
      Nodes[i--] = Nodes.back();
      Nodes.pop_back();                   // Remove from node list...
    }
  }
}


/// markReachableNodes - This method recursively traverses the specified
/// DSNodes, marking any nodes which are reachable.  All reachable nodes it adds
/// to the set, which allows it to only traverse visited nodes once.
///
void DSNode::markReachableNodes(hash_set<DSNode*> &ReachableNodes) {
  if (this == 0) return;
  assert(getForwardNode() == 0 && "Cannot mark a forwarded node!");
  if (ReachableNodes.count(this)) return;          // Already marked reachable
  ReachableNodes.insert(this);                     // Is reachable now

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
                               hash_set<DSNode*> &Visited) {
  if (N == 0) return false;
  assert(N->getForwardNode() == 0 && "Cannot mark a forwarded node!");

  // If we know that this node is alive, return so!
  if (Alive.count(N)) return true;

  // Otherwise, we don't think the node is alive yet, check for infinite
  // recursion.
  if (Visited.count(N)) return false;  // Found a cycle
  Visited.insert(N);   // No recursion, insert into Visited...

  for (unsigned i = 0, e = N->getSize(); i < e; i += DS::PointerSize)
    if (CanReachAliveNodes(N->getLink(i).getNode(), Alive, Visited)) {
      N->markReachableNodes(Alive);
      return true;
    }
  return false;
}

// CallSiteUsesAliveArgs - Return true if the specified call site can reach any
// alive nodes.
//
static bool CallSiteUsesAliveArgs(DSCallSite &CS, hash_set<DSNode*> &Alive,
                                  hash_set<DSNode*> &Visited) {
  if (CanReachAliveNodes(CS.getRetVal().getNode(), Alive, Visited))
    return true;
  if (CS.isIndirectCall() &&
      CanReachAliveNodes(CS.getCalleeNode(), Alive, Visited))
    return true;
  for (unsigned i = 0, e = CS.getNumPtrArgs(); i != e; ++i)
    if (CanReachAliveNodes(CS.getPtrArg(i).getNode(), Alive, Visited))
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
  // Reduce the amount of work we have to do... remove dummy nodes left over by
  // merging...
  removeTriviallyDeadNodes();

  // FIXME: Merge nontrivially identical call nodes...

  // Alive - a set that holds all nodes found to be reachable/alive.
  hash_set<DSNode*> Alive;
  std::vector<std::pair<Value*, DSNode*> > GlobalNodes;

  // Mark all nodes reachable by (non-global) scalar nodes as alive...
  for (ScalarMapTy::iterator I = ScalarMap.begin(), E = ScalarMap.end(); I !=E;)
    if (isa<GlobalValue>(I->first)) {             // Keep track of global nodes
      assert(I->second.getNode() && "Null global node?");
      GlobalNodes.push_back(std::make_pair(I->first, I->second.getNode()));
      ++I;
    } else {
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
      DSNode *N = I->second.getNode();
      if (N->isUnknownNode() && !isa<Argument>(I->first)) {
        ScalarMap.erase(I++);
      } else {
        I->second.getNode()->markReachableNodes(Alive);
        ++I;
      }
    }

  // The return value is alive as well...
  for (ReturnNodesTy::iterator I = ReturnNodes.begin(), E = ReturnNodes.end();
       I != E; ++I)
    I->second.getNode()->markReachableNodes(Alive);

  // Mark any nodes reachable by primary calls as alive...
  for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i)
    FunctionCalls[i].markReachableNodes(Alive);

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
    for (unsigned i = 0; i != GlobalNodes.size(); ++i)
      if (CanReachAliveNodes(GlobalNodes[i].second, Alive, Visited)) {
        std::swap(GlobalNodes[i--], GlobalNodes.back()); // Move to end to erase
        GlobalNodes.pop_back();                          // Erase efficiently
        Iterate = true;
      }

    for (unsigned i = 0, e = AuxFunctionCalls.size(); i != e; ++i)
      if (!AuxFCallsAlive[i] &&
          CallSiteUsesAliveArgs(AuxFunctionCalls[i], Alive, Visited)) {
        AuxFunctionCalls[i].markReachableNodes(Alive);
        AuxFCallsAlive[i] = true;
        Iterate = true;
      }
  } while (Iterate);

  // Remove all dead aux function calls...
  unsigned CurIdx = 0;
  for (unsigned i = 0, e = AuxFunctionCalls.size(); i != e; ++i)
    if (AuxFCallsAlive[i])
      AuxFunctionCalls[CurIdx++].swap(AuxFunctionCalls[i]);
  if (!(Flags & DSGraph::RemoveUnreachableGlobals)) {
    assert(GlobalsGraph && "No globals graph available??");
    // Move the unreachable call nodes to the globals graph...
    GlobalsGraph->AuxFunctionCalls.insert(GlobalsGraph->AuxFunctionCalls.end(),
                                          AuxFunctionCalls.begin()+CurIdx,
                                          AuxFunctionCalls.end());
  }
  // Crop all the useless ones out...
  AuxFunctionCalls.erase(AuxFunctionCalls.begin()+CurIdx,
                         AuxFunctionCalls.end());

  // At this point, any nodes which are visited, but not alive, are nodes which
  // should be moved to the globals graph.  Loop over all nodes, eliminating
  // completely unreachable nodes, and moving visited nodes to the globals graph
  //
  std::vector<DSNode*> DeadNodes;
  DeadNodes.reserve(Nodes.size());
  for (unsigned i = 0; i != Nodes.size(); ++i)
    if (!Alive.count(Nodes[i])) {
      DSNode *N = Nodes[i];
      Nodes[i--] = Nodes.back();            // move node to end of vector
      Nodes.pop_back();                     // Erase node from alive list.
      if (!(Flags & DSGraph::RemoveUnreachableGlobals) &&  // Not in TD pass
          Visited.count(N)) {                    // Visited but not alive?
        GlobalsGraph->Nodes.push_back(N);        // Move node to globals graph
        N->setParentGraph(GlobalsGraph);
      } else {                                 // Otherwise, delete the node
        assert((!N->isGlobalNode() ||
                (Flags & DSGraph::RemoveUnreachableGlobals))
               && "Killing a global?");
        //std::cerr << "[" << i+1 << "/" << DeadNodes.size()
        //          << "] Node is dead: "; N->dump();
        DeadNodes.push_back(N);
        N->dropAllReferences();
      }
    } else {
      assert(Nodes[i]->getForwardNode() == 0 && "Alive forwarded node?");
    }

  // Now that the nodes have either been deleted or moved to the globals graph,
  // loop over the scalarmap, updating the entries for globals...
  //
  if (!(Flags & DSGraph::RemoveUnreachableGlobals)) {  // Not in the TD pass?
    // In this array we start the remapping, which can cause merging.  Because
    // of this, the DSNode pointers in GlobalNodes may be invalidated, so we
    // must always go through the ScalarMap (which contains DSNodeHandles [which
    // cannot be invalidated by merging]).
    //
    for (unsigned i = 0, e = GlobalNodes.size(); i != e; ++i) {
      Value *G = GlobalNodes[i].first;
      ScalarMapTy::iterator I = ScalarMap.find(G);
      assert(I != ScalarMap.end() && "Global not in scalar map anymore?");
      assert(I->second.getNode() && "Global not pointing to anything?");
      assert(!Alive.count(I->second.getNode()) && "Node is alive??");
      GlobalsGraph->ScalarMap[G].mergeWith(I->second);
      assert(GlobalsGraph->ScalarMap[G].getNode() &&
             "Global not pointing to anything?");
      ScalarMap.erase(I);
    }

    // Merging leaves behind silly nodes, we remove them to avoid polluting the
    // globals graph.
    if (!GlobalNodes.empty())
      GlobalsGraph->removeTriviallyDeadNodes();
  } else {
    // If we are in the top-down pass, remove all unreachable globals from the
    // ScalarMap...
    for (unsigned i = 0, e = GlobalNodes.size(); i != e; ++i)
      ScalarMap.erase(GlobalNodes[i].first);
  }

  // Loop over all of the dead nodes now, deleting them since their referrer
  // count is zero.
  for (unsigned i = 0, e = DeadNodes.size(); i != e; ++i)
    delete DeadNodes[i];

  DEBUG(AssertGraphOK(); GlobalsGraph->AssertGraphOK());
}

void DSGraph::AssertGraphOK() const {
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
    Nodes[i]->assertOK();
  return;  // FIXME: remove
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
