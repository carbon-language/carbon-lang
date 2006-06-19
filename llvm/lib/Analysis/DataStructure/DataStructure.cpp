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

#include "llvm/Analysis/DataStructure/DSGraphTraits.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Timer.h"
#include <iostream>
#include <algorithm>
using namespace llvm;

#define COLLAPSE_ARRAYS_AGGRESSIVELY 0

namespace {
  Statistic<> NumFolds          ("dsa", "Number of nodes completely folded");
  Statistic<> NumCallNodesMerged("dsa", "Number of call nodes merged");
  Statistic<> NumNodeAllocated  ("dsa", "Number of nodes allocated");
  Statistic<> NumDNE            ("dsa", "Number of nodes removed by reachability");
  Statistic<> NumTrivialDNE     ("dsa", "Number of nodes trivially removed");
  Statistic<> NumTrivialGlobalDNE("dsa", "Number of globals trivially removed");
  static cl::opt<unsigned>
  DSAFieldLimit("dsa-field-limit", cl::Hidden,
                cl::desc("Number of fields to track before collapsing a node"),
                cl::init(256));
}

#if 0
#define TIME_REGION(VARNAME, DESC) \
   NamedRegionTimer VARNAME(DESC)
#else
#define TIME_REGION(VARNAME, DESC)
#endif

using namespace DS;

/// isForwarding - Return true if this NodeHandle is forwarding to another
/// one.
bool DSNodeHandle::isForwarding() const {
  return N && N->isForwarding();
}

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
// DSScalarMap Implementation
//===----------------------------------------------------------------------===//

DSNodeHandle &DSScalarMap::AddGlobal(GlobalValue *GV) {
  assert(ValueMap.count(GV) == 0 && "GV already exists!");

  // If the node doesn't exist, check to see if it's a global that is
  // equated to another global in the program.
  EquivalenceClasses<GlobalValue*>::iterator ECI = GlobalECs.findValue(GV);
  if (ECI != GlobalECs.end()) {
    GlobalValue *Leader = *GlobalECs.findLeader(ECI);
    if (Leader != GV) {
      GV = Leader;
      iterator I = ValueMap.find(GV);
      if (I != ValueMap.end())
        return I->second;
    }
  }

  // Okay, this is either not an equivalenced global or it is the leader, it
  // will be inserted into the scalar map now.
  GlobalSet.insert(GV);

  return ValueMap.insert(std::make_pair(GV, DSNodeHandle())).first->second;
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
  if (!NullLinks) {
    Links = N.Links;
  } else
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
    assert(SM.global_count(Globals[i]));
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
  ForwardNH.setTo(To, Offset);
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
  // First, check to make sure this is the leader if the global is in an
  // equivalence class.
  GV = getParentGraph()->getScalarMap().getLeaderForGlobal(GV);

  // Keep the list sorted.
  std::vector<GlobalValue*>::iterator I =
    std::lower_bound(Globals.begin(), Globals.end(), GV);

  if (I == Globals.end() || *I != GV) {
    Globals.insert(I, GV);
    NodeType |= GlobalNode;
  }
}

// removeGlobal - Remove the specified global that is explicitly in the globals
// list.
void DSNode::removeGlobal(GlobalValue *GV) {
  std::vector<GlobalValue*>::iterator I =
    std::lower_bound(Globals.begin(), Globals.end(), GV);
  assert(I != Globals.end() && *I == GV && "Global not in node!");
  Globals.erase(I);
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

/// addFullGlobalsList - Compute the full set of global values that are
/// represented by this node.  Unlike getGlobalsList(), this requires fair
/// amount of work to compute, so don't treat this method call as free.
void DSNode::addFullGlobalsList(std::vector<GlobalValue*> &List) const {
  if (globals_begin() == globals_end()) return;

  EquivalenceClasses<GlobalValue*> &EC = getParentGraph()->getGlobalECs();

  for (globals_iterator I = globals_begin(), E = globals_end(); I != E; ++I) {
    EquivalenceClasses<GlobalValue*>::iterator ECI = EC.findValue(*I);
    if (ECI == EC.end())
      List.push_back(*I);
    else
      List.insert(List.end(), EC.member_begin(ECI), EC.member_end());
  }
}

/// addFullFunctionList - Identical to addFullGlobalsList, but only return the
/// functions in the full list.
void DSNode::addFullFunctionList(std::vector<Function*> &List) const {
  if (globals_begin() == globals_end()) return;

  EquivalenceClasses<GlobalValue*> &EC = getParentGraph()->getGlobalECs();

  for (globals_iterator I = globals_begin(), E = globals_end(); I != E; ++I) {
    EquivalenceClasses<GlobalValue*>::iterator ECI = EC.findValue(*I);
    if (ECI == EC.end()) {
      if (Function *F = dyn_cast<Function>(*I))
        List.push_back(F);
    } else {
      for (EquivalenceClasses<GlobalValue*>::member_iterator MI =
             EC.member_begin(ECI), E = EC.member_end(); MI != E; ++MI)
        if (Function *F = dyn_cast<Function>(*MI))
          List.push_back(F);
    }
  }
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
            SS.Offset +=
               unsigned(SL->MemberOffsets[SS.Idx]-SL->MemberOffsets[SS.Idx-1]);
            return;
          }
          Stack.pop_back();  // At the end of the structure
        } else {
          const ArrayType *AT = cast<ArrayType>(SS.Ty);
          ++SS.Idx;
          if (SS.Idx != AT->getNumElements()) {
            SS.Offset += unsigned(TD.getTypeSize(AT->getElementType()));
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
                            SS.Offset+unsigned(SL->MemberOffsets[SS.Idx])));
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
                             unsigned(TD.getTypeSize(AT->getElementType()))));
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
  unsigned NewTySize = NewTy->isSized() ? (unsigned)TD.getTypeSize(NewTy) : 0;

  // Otherwise check to see if we can fold this type into the current node.  If
  // we can't, we fold the node completely, if we can, we potentially update our
  // internal state.
  //
  if (Ty == Type::VoidTy) {
    // If this is the first type that this node has seen, just accept it without
    // question....
    assert(Offset == 0 && !isArray() &&
           "Cannot have an offset into a void node!");

    // If this node would have to have an unreasonable number of fields, just
    // collapse it.  This can occur for fortran common blocks, which have stupid
    // things like { [100000000 x double], [1000000 x double] }.
    unsigned NumFields = (NewTySize+DS::PointerSize-1) >> DS::PointerShift;
    if (NumFields > DSAFieldLimit) {
      foldNodeCompletely();
      return true;
    }

    Ty = NewTy;
    NodeType &= ~Array;
    if (WillBeArray) NodeType |= Array;
    Size = NewTySize;

    // Calculate the number of outgoing links from this node.
    Links.resize(NumFields);
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

    // If this node would have to have an unreasonable number of fields, just
    // collapse it.  This can occur for fortran common blocks, which have stupid
    // things like { [100000000 x double], [1000000 x double] }.
    unsigned NumFields = (NewTySize+Offset+DS::PointerSize-1) >> DS::PointerShift;
    if (NumFields > DSAFieldLimit) {
      foldNodeCompletely();
      return true;
    }

    if (Offset) {
      //handle some common cases:
      // Ty:    struct { t1, t2, t3, t4, ..., tn}
      // NewTy: struct { offset, stuff...}
      // try merge with NewTy: struct {t1, t2, stuff...} if offset lands exactly on a field in Ty
      if (isa<StructType>(NewTy) && isa<StructType>(Ty)) {
        DEBUG(std::cerr << "Ty: " << *Ty << "\nNewTy: " << *NewTy << "@" << Offset << "\n");
        unsigned O = 0;
        const StructType *STy = cast<StructType>(Ty);
        const StructLayout &SL = *TD.getStructLayout(STy);
        unsigned i = SL.getElementContainingOffset(Offset);
        //Either we hit it exactly or give up
        if (SL.MemberOffsets[i] != Offset) {
          if (FoldIfIncompatible) foldNodeCompletely();
          return true;
        }
        std::vector<const Type*> nt;
        for (unsigned x = 0; x < i; ++x)
          nt.push_back(STy->getElementType(x));
        STy = cast<StructType>(NewTy);
        nt.insert(nt.end(), STy->element_begin(), STy->element_end());
        //and merge
        STy = StructType::get(nt);
        DEBUG(std::cerr << "Trying with: " << *STy << "\n");
        return mergeTypeInfo(STy, 0);
      }

      //Ty: struct { t1, t2, t3 ... tn}
      //NewTy T offset x
      //try merge with NewTy: struct : {t1, t2, T} if offset lands on a field in Ty
      if (isa<StructType>(Ty)) {
        DEBUG(std::cerr << "Ty: " << *Ty << "\nNewTy: " << *NewTy << "@" << Offset << "\n");
        unsigned O = 0;
        const StructType *STy = cast<StructType>(Ty);
        const StructLayout &SL = *TD.getStructLayout(STy);
        unsigned i = SL.getElementContainingOffset(Offset);
        //Either we hit it exactly or give up
        if (SL.MemberOffsets[i] != Offset) {
          if (FoldIfIncompatible) foldNodeCompletely();
          return true;
        }
        std::vector<const Type*> nt;
        for (unsigned x = 0; x < i; ++x)
          nt.push_back(STy->getElementType(x));
        nt.push_back(NewTy);
        //and merge
        STy = StructType::get(nt);
        DEBUG(std::cerr << "Trying with: " << *STy << "\n");
        return mergeTypeInfo(STy, 0);
      }

      std::cerr << "UNIMP: Trying to merge a growth type into "
                << "offset != 0: Collapsing!\n";
      abort();
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
    Links.resize(NumFields);

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

    switch (SubType->getTypeID()) {
    case Type::StructTyID: {
      const StructType *STy = cast<StructType>(SubType);
      const StructLayout &SL = *TD.getStructLayout(STy);
      unsigned i = SL.getElementContainingOffset(Offset-O);

      // The offset we are looking for must be in the i'th element...
      SubType = STy->getElementType(i);
      O += (unsigned)SL.MemberOffsets[i];
      break;
    }
    case Type::ArrayTyID: {
      SubType = cast<ArrayType>(SubType)->getElementType();
      unsigned ElSize = (unsigned)TD.getTypeSize(SubType);
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

  // Differing function types don't require us to merge.  They are not values
  // anyway.
  if (isa<FunctionType>(SubType) &&
      isa<FunctionType>(NewTy)) return false;

  unsigned SubTypeSize = SubType->isSized() ?
       (unsigned)TD.getTypeSize(SubType) : 0;

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
    switch (SubType->getTypeID()) {
    case Type::StructTyID: {
      const StructType *STy = cast<StructType>(SubType);
      const StructLayout &SL = *TD.getStructLayout(STy);
      if (SL.MemberOffsets.size() > 1)
        NextPadSize = (unsigned)SL.MemberOffsets[1];
      else
        NextPadSize = SubTypeSize;
      NextSubType = STy->getElementType(0);
      NextSubTypeSize = (unsigned)TD.getTypeSize(NextSubType);
      break;
    }
    case Type::ArrayTyID:
      NextSubType = cast<ArrayType>(SubType)->getElementType();
      NextSubTypeSize = (unsigned)TD.getTypeSize(NextSubType);
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
  if (getParentGraph()->retnodes_begin() != getParentGraph()->retnodes_end())
    M = getParentGraph()->retnodes_begin()->first->getParent();
  DEBUG(std::cerr << "MergeTypeInfo Folding OrigTy: ";
        WriteTypeSymbolic(std::cerr, Ty, M) << "\n due to:";
        WriteTypeSymbolic(std::cerr, NewTy, M) << " @ " << Offset << "!\n"
                  << "SubType: ";
        WriteTypeSymbolic(std::cerr, SubType, M) << "\n\n");

  if (FoldIfIncompatible) foldNodeCompletely();
  return true;
}



/// addEdgeTo - Add an edge from the current node to the specified node.  This
/// can cause merging of nodes in the graph.
///
void DSNode::addEdgeTo(unsigned Offset, const DSNodeHandle &NH) {
  if (NH.isNull()) return;       // Nothing to do

  if (isNodeCompletelyFolded())
    Offset = 0;

  DSNodeHandle &ExistingEdge = getLink(Offset);
  if (!ExistingEdge.isNull()) {
    // Merge the two nodes...
    ExistingEdge.mergeWith(NH);
  } else {                             // No merging to perform...
    setLink(Offset, NH);               // Just force a link in there...
  }
}


/// MergeSortedVectors - Efficiently merge a vector into another vector where
/// duplicates are not allowed and both are sorted.  This assumes that 'T's are
/// efficiently copyable and have sane comparison semantics.
///
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
  assert(CurNodeH.getNode()->getParentGraph()==NH.getNode()->getParentGraph() &&
         "Cannot merge two nodes that are not in the same graph!");

  // Now we know that Offset >= NH.Offset, so convert it so our "Offset" (with
  // respect to NH.Offset) is now zero.  NOffset is the distance from the base
  // of our object that N starts from.
  //
  unsigned NOffset = CurNodeH.getOffset()-NH.getOffset();
  unsigned NSize = NH.getNode()->getSize();

  // If the two nodes are of different size, and the smaller node has the array
  // bit set, collapse!
  if (NSize != CurNodeH.getNode()->getSize()) {
#if COLLAPSE_ARRAYS_AGGRESSIVELY
    if (NSize < CurNodeH.getNode()->getSize()) {
      if (NH.getNode()->isArray())
        NH.getNode()->foldNodeCompletely();
    } else if (CurNodeH.getNode()->isArray()) {
      NH.getNode()->foldNodeCompletely();
    }
#endif
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
    NSize = NH.getNode()->getSize();
    NOffset = NH.getOffset();
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


/// mergeWith - Merge this node and the specified node, moving all links to and
/// from the argument node into the current node, deleting the node argument.
/// Offset indicates what offset the specified node is to be merged into the
/// current node.
///
/// The specified node may be a null pointer (in which case, we update it to
/// point to this node).
///
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
  if (CurNodeH.getOffset() >= NHCopy.getOffset())
    DSNode::MergeNodes(CurNodeH, NHCopy);
  else
    DSNode::MergeNodes(NHCopy, CurNodeH);
}


//===----------------------------------------------------------------------===//
// ReachabilityCloner Implementation
//===----------------------------------------------------------------------===//

DSNodeHandle ReachabilityCloner::getClonedNH(const DSNodeHandle &SrcNH) {
  if (SrcNH.isNull()) return DSNodeHandle();
  const DSNode *SN = SrcNH.getNode();

  DSNodeHandle &NH = NodeMap[SN];
  if (!NH.isNull()) {   // Node already mapped?
    DSNode *NHN = NH.getNode();
    return DSNodeHandle(NHN, NH.getOffset()+SrcNH.getOffset());
  }

  // If SrcNH has globals and the destination graph has one of the same globals,
  // merge this node with the destination node, which is much more efficient.
  if (SN->globals_begin() != SN->globals_end()) {
    DSScalarMap &DestSM = Dest.getScalarMap();
    for (DSNode::globals_iterator I = SN->globals_begin(),E = SN->globals_end();
         I != E; ++I) {
      GlobalValue *GV = *I;
      DSScalarMap::iterator GI = DestSM.find(GV);
      if (GI != DestSM.end() && !GI->second.isNull()) {
        // We found one, use merge instead!
        merge(GI->second, Src.getNodeForValue(GV));
        assert(!NH.isNull() && "Didn't merge node!");
        DSNode *NHN = NH.getNode();
        return DSNodeHandle(NHN, NH.getOffset()+SrcNH.getOffset());
      }
    }
  }

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
        MergeOffset = ((i << DS::PointerShift)+NH.getOffset()) % CN->getSize();
      CN->addEdgeTo(MergeOffset, DestEdge);
    }
  }

  // If this node contains any globals, make sure they end up in the scalar
  // map with the correct offset.
  for (DSNode::globals_iterator I = SN->globals_begin(), E = SN->globals_end();
       I != E; ++I) {
    GlobalValue *GV = *I;
    const DSNodeHandle &SrcGNH = Src.getNodeForValue(GV);
    DSNodeHandle &DestGNH = NodeMap[SrcGNH.getNode()];
    assert(DestGNH.getNode() == NH.getNode() &&"Global mapping inconsistent");
    Dest.getNodeForValue(GV).mergeWith(DSNodeHandle(DestGNH.getNode(),
                                       DestGNH.getOffset()+SrcGNH.getOffset()));
  }
  NH.getNode()->mergeGlobals(SN->getGlobalsList());

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
    DSNode *SCNHN = SCNH.getNode();
    NH.mergeWith(DSNodeHandle(SCNHN,
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
#if COLLAPSE_ARRAYS_AGGRESSIVELY
        if (SN->getSize() < DN->getSize()) {
          if (SN->isArray()) {
            DN->foldNodeCompletely();
            DN = NH.getNode();
          }
        } else if (DN->isArray()) {
          DN->foldNodeCompletely();
          DN = NH.getNode();
        }
#endif
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
    if (SN->globals_begin() != SN->globals_end()) {
      // Update the globals in the destination node itself.
      DN->mergeGlobals(SN->getGlobalsList());

      // Update the scalar map for the graph we are merging the source node
      // into.
      for (DSNode::globals_iterator I = SN->globals_begin(),
             E = SN->globals_end(); I != E; ++I) {
        GlobalValue *GV = *I;
        const DSNodeHandle &SrcGNH = Src.getNodeForValue(GV);
        DSNodeHandle &DestGNH = NodeMap[SrcGNH.getNode()];
        assert(DestGNH.getNode()==NH.getNode() &&"Global mapping inconsistent");
        Dest.getNodeForValue(GV).mergeWith(DSNodeHandle(DestGNH.getNode(),
                                      DestGNH.getOffset()+SrcGNH.getOffset()));
      }
      NH.getNode()->mergeGlobals(SN->getGlobalsList());
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
    for (DSNode::globals_iterator I = SN->globals_begin(),
           E = SN->globals_end(); I != E; ++I) {
      GlobalValue *GV = *I;
      const DSNodeHandle &SrcGNH = Src.getNodeForValue(GV);
      DSNodeHandle &DestGNH = NodeMap[SrcGNH.getNode()];
      assert(DestGNH.getNode()==NH.getNode() &&"Global mapping inconsistent");
      assert(SrcGNH.getNode() == SN && "Global mapping inconsistent");
      Dest.getNodeForValue(GV).mergeWith(DSNodeHandle(DestGNH.getNode(),
                                    DestGNH.getOffset()+SrcGNH.getOffset()));
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
      DSNode *CN = SCNH.getNode();
      unsigned MergeOffset =
        ((i << DS::PointerShift)+SCNH.getOffset()) % CN->getSize();

      DSNodeHandle Tmp = CN->getLink(MergeOffset);
      if (!Tmp.isNull()) {
        // Perform the recursive merging.  Make sure to create a temporary NH,
        // because the Link can disappear in the process of recursive merging.
        merge(Tmp, SrcEdge);
      } else {
        Tmp.mergeWith(getClonedNH(SrcEdge));
        // Merging this could cause all kinds of recursive things to happen,
        // culminating in the current node being eliminated.  Since this is
        // possible, make sure to reaquire the link from 'CN'.

        unsigned MergeOffset = 0;
        CN = SCNH.getNode();
        MergeOffset = ((i << DS::PointerShift)+SCNH.getOffset()) %CN->getSize();
        CN->getLink(MergeOffset).mergeWith(Tmp);
      }
    }
  }
}

/// mergeCallSite - Merge the nodes reachable from the specified src call
/// site into the nodes reachable from DestCS.
void ReachabilityCloner::mergeCallSite(DSCallSite &DestCS,
                                       const DSCallSite &SrcCS) {
  merge(DestCS.getRetVal(), SrcCS.getRetVal());
  unsigned MinArgs = DestCS.getNumPtrArgs();
  if (SrcCS.getNumPtrArgs() < MinArgs) MinArgs = SrcCS.getNumPtrArgs();

  for (unsigned a = 0; a != MinArgs; ++a)
    merge(DestCS.getPtrArg(a), SrcCS.getPtrArg(a));

  for (unsigned a = MinArgs, e = SrcCS.getNumPtrArgs(); a != e; ++a)
    DestCS.addPtrArg(getClonedNH(SrcCS.getPtrArg(a)));
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
  case 1: return retnodes_begin()->first->getName();
  default:
    std::string Return;
    for (DSGraph::retnodes_iterator I = retnodes_begin();
         I != retnodes_end(); ++I)
      Return += I->first->getName() + " ";
    Return.erase(Return.end()-1, Return.end());   // Remove last space character
    return Return;
  }
}


DSGraph::DSGraph(const DSGraph &G, EquivalenceClasses<GlobalValue*> &ECs,
                 unsigned CloneFlags)
  : GlobalsGraph(0), ScalarMap(ECs), TD(G.TD) {
  PrintAuxCalls = false;
  cloneInto(G, CloneFlags);
}

DSGraph::~DSGraph() {
  FunctionCalls.clear();
  AuxFunctionCalls.clear();
  ScalarMap.clear();
  ReturnNodes.clear();

  // Drop all intra-node references, so that assertions don't fail...
  for (node_iterator NI = node_begin(), E = node_end(); NI != E; ++NI)
    NI->dropAllReferences();

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
        DSNode *ONMIN = ONMI->second.getNode();
        Links[i].setTo(ONMIN, Links[i].getOffset()+ONMI->second.getOffset());
      }
    }
}

/// addObjectToGraph - This method can be used to add global, stack, and heap
/// objects to the graph.  This can be used when updating DSGraphs due to the
/// introduction of new temporary objects.  The new object is not pointed to
/// and does not point to any other objects in the graph.
DSNode *DSGraph::addObjectToGraph(Value *Ptr, bool UseDeclaredType) {
  assert(isa<PointerType>(Ptr->getType()) && "Ptr is not a pointer!");
  const Type *Ty = cast<PointerType>(Ptr->getType())->getElementType();
  DSNode *N = new DSNode(UseDeclaredType ? Ty : 0, this);
  assert(ScalarMap[Ptr].isNull() && "Object already in this graph!");
  ScalarMap[Ptr] = N;

  if (GlobalValue *GV = dyn_cast<GlobalValue>(Ptr)) {
    N->addGlobal(GV);
  } else if (MallocInst *MI = dyn_cast<MallocInst>(Ptr)) {
    N->setHeapNodeMarker();
  } else if (AllocaInst *AI = dyn_cast<AllocaInst>(Ptr)) {
    N->setAllocaNodeMarker();
  } else {
    assert(0 && "Illegal memory object input!");
  }
  return N;
}


/// cloneInto - Clone the specified DSGraph into the current graph.  The
/// translated ScalarMap for the old function is filled into the ScalarMap
/// for the graph, and the translated ReturnNodes map is returned into
/// ReturnNodes.
///
/// The CloneFlags member controls various aspects of the cloning process.
///
void DSGraph::cloneInto(const DSGraph &G, unsigned CloneFlags) {
  TIME_REGION(X, "cloneInto");
  assert(&G != this && "Cannot clone graph into itself!");

  NodeMapTy OldNodeMap;

  // Remove alloca or mod/ref bits as specified...
  unsigned BitsToClear = ((CloneFlags & StripAllocaBit)? DSNode::AllocaNode : 0)
    | ((CloneFlags & StripModRefBits)? (DSNode::Modified | DSNode::Read) : 0)
    | ((CloneFlags & StripIncompleteBit)? DSNode::Incomplete : 0);
  BitsToClear |= DSNode::DEAD;  // Clear dead flag...

  for (node_const_iterator I = G.node_begin(), E = G.node_end(); I != E; ++I) {
    assert(!I->isForwarding() &&
           "Forward nodes shouldn't be in node list!");
    DSNode *New = new DSNode(*I, this);
    New->maskNodeTypes(~BitsToClear);
    OldNodeMap[I] = New;
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
    DSNodeHandle &H = ScalarMap.getRawEntryRef(I->first);
    DSNode *MappedNodeN = MappedNode.getNode();
    H.mergeWith(DSNodeHandle(MappedNodeN,
                             I->second.getOffset()+MappedNode.getOffset()));
  }

  if (!(CloneFlags & DontCloneCallNodes)) {
    // Copy the function calls list.
    for (fc_iterator I = G.fc_begin(), E = G.fc_end(); I != E; ++I)
      FunctionCalls.push_back(DSCallSite(*I, OldNodeMap));
  }

  if (!(CloneFlags & DontCloneAuxCallNodes)) {
    // Copy the auxiliary function calls list.
    for (afc_iterator I = G.afc_begin(), E = G.afc_end(); I != E; ++I)
      AuxFunctionCalls.push_back(DSCallSite(*I, OldNodeMap));
  }

  // Map the return node pointers over...
  for (retnodes_iterator I = G.retnodes_begin(),
         E = G.retnodes_end(); I != E; ++I) {
    const DSNodeHandle &Ret = I->second;
    DSNodeHandle &MappedRet = OldNodeMap[Ret.getNode()];
    DSNode *MappedRetN = MappedRet.getNode();
    ReturnNodes.insert(std::make_pair(I->first,
                                      DSNodeHandle(MappedRetN,
                                     MappedRet.getOffset()+Ret.getOffset())));
  }
}

/// spliceFrom - Logically perform the operation of cloning the RHS graph into
/// this graph, then clearing the RHS graph.  Instead of performing this as
/// two seperate operations, do it as a single, much faster, one.
///
void DSGraph::spliceFrom(DSGraph &RHS) {
  // Change all of the nodes in RHS to think we are their parent.
  for (NodeListTy::iterator I = RHS.Nodes.begin(), E = RHS.Nodes.end();
       I != E; ++I)
    I->setParentGraph(this);
  // Take all of the nodes.
  Nodes.splice(Nodes.end(), RHS.Nodes);

  // Take all of the calls.
  FunctionCalls.splice(FunctionCalls.end(), RHS.FunctionCalls);
  AuxFunctionCalls.splice(AuxFunctionCalls.end(), RHS.AuxFunctionCalls);

  // Take all of the return nodes.
  if (ReturnNodes.empty()) {
    ReturnNodes.swap(RHS.ReturnNodes);
  } else {
    ReturnNodes.insert(RHS.ReturnNodes.begin(), RHS.ReturnNodes.end());
    RHS.ReturnNodes.clear();
  }

  // Merge the scalar map in.
  ScalarMap.spliceFrom(RHS.ScalarMap);
}

/// spliceFrom - Copy all entries from RHS, then clear RHS.
///
void DSScalarMap::spliceFrom(DSScalarMap &RHS) {
  // Special case if this is empty.
  if (ValueMap.empty()) {
    ValueMap.swap(RHS.ValueMap);
    GlobalSet.swap(RHS.GlobalSet);
  } else {
    GlobalSet.insert(RHS.GlobalSet.begin(), RHS.GlobalSet.end());
    for (ValueMapTy::iterator I = RHS.ValueMap.begin(), E = RHS.ValueMap.end();
         I != E; ++I)
      ValueMap[I->first].mergeWith(I->second);
    RHS.ValueMap.clear();
  }
}


/// getFunctionArgumentsForCall - Given a function that is currently in this
/// graph, return the DSNodeHandles that correspond to the pointer-compatible
/// function arguments.  The vector is filled in with the return value (or
/// null if it is not pointer compatible), followed by all of the
/// pointer-compatible arguments.
void DSGraph::getFunctionArgumentsForCall(Function *F,
                                       std::vector<DSNodeHandle> &Args) const {
  Args.push_back(getReturnNodeFor(*F));
  for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
       AI != E; ++AI)
    if (isPointerType(AI->getType())) {
      Args.push_back(getNodeForValue(AI));
      assert(!Args.back().isNull() && "Pointer argument w/o scalarmap entry!?");
    }
}

namespace {
  // HackedGraphSCCFinder - This is used to find nodes that have a path from the
  // node to a node cloned by the ReachabilityCloner object contained.  To be
  // extra obnoxious it ignores edges from nodes that are globals, and truncates
  // search at RC marked nodes.  This is designed as an object so that
  // intermediate results can be memoized across invocations of
  // PathExistsToClonedNode.
  struct HackedGraphSCCFinder {
    ReachabilityCloner &RC;
    unsigned CurNodeId;
    std::vector<const DSNode*> SCCStack;
    std::map<const DSNode*, std::pair<unsigned, bool> > NodeInfo;

    HackedGraphSCCFinder(ReachabilityCloner &rc) : RC(rc), CurNodeId(1) {
      // Remove null pointer as a special case.
      NodeInfo[0] = std::make_pair(0, false);
    }

    std::pair<unsigned, bool> &VisitForSCCs(const DSNode *N);

    bool PathExistsToClonedNode(const DSNode *N) {
      return VisitForSCCs(N).second;
    }

    bool PathExistsToClonedNode(const DSCallSite &CS) {
      if (PathExistsToClonedNode(CS.getRetVal().getNode()))
        return true;
      for (unsigned i = 0, e = CS.getNumPtrArgs(); i != e; ++i)
        if (PathExistsToClonedNode(CS.getPtrArg(i).getNode()))
          return true;
      return false;
    }
  };
}

std::pair<unsigned, bool> &HackedGraphSCCFinder::
VisitForSCCs(const DSNode *N) {
  std::map<const DSNode*, std::pair<unsigned, bool> >::iterator
    NodeInfoIt = NodeInfo.lower_bound(N);
  if (NodeInfoIt != NodeInfo.end() && NodeInfoIt->first == N)
    return NodeInfoIt->second;

  unsigned Min = CurNodeId++;
  unsigned MyId = Min;
  std::pair<unsigned, bool> &ThisNodeInfo =
    NodeInfo.insert(NodeInfoIt,
                    std::make_pair(N, std::make_pair(MyId, false)))->second;

  // Base case: if we find a global, this doesn't reach the cloned graph
  // portion.
  if (N->isGlobalNode()) {
    ThisNodeInfo.second = false;
    return ThisNodeInfo;
  }

  // Base case: if this does reach the cloned graph portion... it does. :)
  if (RC.hasClonedNode(N)) {
    ThisNodeInfo.second = true;
    return ThisNodeInfo;
  }

  SCCStack.push_back(N);

  // Otherwise, check all successors.
  bool AnyDirectSuccessorsReachClonedNodes = false;
  for (DSNode::const_edge_iterator EI = N->edge_begin(), EE = N->edge_end();
       EI != EE; ++EI)
    if (DSNode *Succ = EI->getNode()) {
      std::pair<unsigned, bool> &SuccInfo = VisitForSCCs(Succ);
      if (SuccInfo.first < Min) Min = SuccInfo.first;
      AnyDirectSuccessorsReachClonedNodes |= SuccInfo.second;
    }

  if (Min != MyId)
    return ThisNodeInfo;  // Part of a large SCC.  Leave self on stack.

  if (SCCStack.back() == N) {  // Special case single node SCC.
    SCCStack.pop_back();
    ThisNodeInfo.second = AnyDirectSuccessorsReachClonedNodes;
    return ThisNodeInfo;
  }

  // Find out if any direct successors of any node reach cloned nodes.
  if (!AnyDirectSuccessorsReachClonedNodes)
    for (unsigned i = SCCStack.size()-1; SCCStack[i] != N; --i)
      for (DSNode::const_edge_iterator EI = N->edge_begin(), EE = N->edge_end();
           EI != EE; ++EI)
        if (DSNode *N = EI->getNode())
          if (NodeInfo[N].second) {
            AnyDirectSuccessorsReachClonedNodes = true;
            goto OutOfLoop;
          }
OutOfLoop:
  // If any successor reaches a cloned node, mark all nodes in this SCC as
  // reaching the cloned node.
  if (AnyDirectSuccessorsReachClonedNodes)
    while (SCCStack.back() != N) {
      NodeInfo[SCCStack.back()].second = true;
      SCCStack.pop_back();
    }
  SCCStack.pop_back();
  ThisNodeInfo.second = true;
  return ThisNodeInfo;
}

/// mergeInCallFromOtherGraph - This graph merges in the minimal number of
/// nodes from G2 into 'this' graph, merging the bindings specified by the
/// call site (in this graph) with the bindings specified by the vector in G2.
/// The two DSGraphs must be different.
///
void DSGraph::mergeInGraph(const DSCallSite &CS,
                           std::vector<DSNodeHandle> &Args,
                           const DSGraph &Graph, unsigned CloneFlags) {
  TIME_REGION(X, "mergeInGraph");

  assert((CloneFlags & DontCloneCallNodes) &&
         "Doesn't support copying of call nodes!");

  // If this is not a recursive call, clone the graph into this graph...
  if (&Graph == this) {
    // Merge the return value with the return value of the context.
    Args[0].mergeWith(CS.getRetVal());

    // Resolve all of the function arguments.
    for (unsigned i = 0, e = CS.getNumPtrArgs(); i != e; ++i) {
      if (i == Args.size()-1)
        break;

      // Add the link from the argument scalar to the provided value.
      Args[i+1].mergeWith(CS.getPtrArg(i));
    }
    return;
  }

  // Clone the callee's graph into the current graph, keeping track of where
  // scalars in the old graph _used_ to point, and of the new nodes matching
  // nodes of the old graph.
  ReachabilityCloner RC(*this, Graph, CloneFlags);

  // Map the return node pointer over.
  if (!CS.getRetVal().isNull())
    RC.merge(CS.getRetVal(), Args[0]);

  // Map over all of the arguments.
  for (unsigned i = 0, e = CS.getNumPtrArgs(); i != e; ++i) {
    if (i == Args.size()-1)
      break;

    // Add the link from the argument scalar to the provided value.
    RC.merge(CS.getPtrArg(i), Args[i+1]);
  }

  // We generally don't want to copy global nodes or aux calls from the callee
  // graph to the caller graph.  However, we have to copy them if there is a
  // path from the node to a node we have already copied which does not go
  // through another global.  Compute the set of node that can reach globals and
  // aux call nodes to copy over, then do it.
  std::vector<const DSCallSite*> AuxCallToCopy;
  std::vector<GlobalValue*> GlobalsToCopy;

  // NodesReachCopiedNodes - Memoize results for efficiency.  Contains a
  // true/false value for every visited node that reaches a copied node without
  // going through a global.
  HackedGraphSCCFinder SCCFinder(RC);

  if (!(CloneFlags & DontCloneAuxCallNodes))
    for (afc_iterator I = Graph.afc_begin(), E = Graph.afc_end(); I!=E; ++I)
      if (SCCFinder.PathExistsToClonedNode(*I))
        AuxCallToCopy.push_back(&*I);
      else if (I->isIndirectCall()){
 	//If the call node doesn't have any callees, clone it
 	std::vector< Function *> List;
 	I->getCalleeNode()->addFullFunctionList(List);
 	if (!List.size())
 	  AuxCallToCopy.push_back(&*I);
       }

  const DSScalarMap &GSM = Graph.getScalarMap();
  for (DSScalarMap::global_iterator GI = GSM.global_begin(),
         E = GSM.global_end(); GI != E; ++GI) {
    DSNode *GlobalNode = Graph.getNodeForValue(*GI).getNode();
    for (DSNode::edge_iterator EI = GlobalNode->edge_begin(),
           EE = GlobalNode->edge_end(); EI != EE; ++EI)
      if (SCCFinder.PathExistsToClonedNode(EI->getNode())) {
        GlobalsToCopy.push_back(*GI);
        break;
      }
  }

  // Copy aux calls that are needed.
  for (unsigned i = 0, e = AuxCallToCopy.size(); i != e; ++i)
    AuxFunctionCalls.push_back(DSCallSite(*AuxCallToCopy[i], RC));

  // Copy globals that are needed.
  for (unsigned i = 0, e = GlobalsToCopy.size(); i != e; ++i)
    RC.getClonedNH(Graph.getNodeForValue(GlobalsToCopy[i]));
}



/// mergeInGraph - The method is used for merging graphs together.  If the
/// argument graph is not *this, it makes a clone of the specified graph, then
/// merges the nodes specified in the call site with the formal arguments in the
/// graph.
///
void DSGraph::mergeInGraph(const DSCallSite &CS, Function &F,
                           const DSGraph &Graph, unsigned CloneFlags) {
  // Set up argument bindings.
  std::vector<DSNodeHandle> Args;
  Graph.getFunctionArgumentsForCall(&F, Args);

  mergeInGraph(CS, Args, Graph, CloneFlags);
}

/// getCallSiteForArguments - Get the arguments and return value bindings for
/// the specified function in the current graph.
///
DSCallSite DSGraph::getCallSiteForArguments(Function &F) const {
  std::vector<DSNodeHandle> Args;

  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I)
    if (isPointerType(I->getType()))
      Args.push_back(getNodeForValue(I));

  return DSCallSite(CallSite(), getReturnNodeFor(F), &F, Args);
}

/// getDSCallSiteForCallSite - Given an LLVM CallSite object that is live in
/// the context of this graph, return the DSCallSite for it.
DSCallSite DSGraph::getDSCallSiteForCallSite(CallSite CS) const {
  DSNodeHandle RetVal;
  Instruction *I = CS.getInstruction();
  if (isPointerType(I->getType()))
    RetVal = getNodeForValue(I);

  std::vector<DSNodeHandle> Args;
  Args.reserve(CS.arg_end()-CS.arg_begin());

  // Calculate the arguments vector...
  for (CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end(); I != E; ++I)
    if (isPointerType((*I)->getType()))
      if (isa<ConstantPointerNull>(*I))
        Args.push_back(DSNodeHandle());
      else
        Args.push_back(getNodeForValue(*I));

  // Add a new function call entry...
  if (Function *F = CS.getCalledFunction())
    return DSCallSite(CS, RetVal, F, Args);
  else
    return DSCallSite(CS, RetVal,
                      getNodeForValue(CS.getCalledValue()).getNode(), Args);
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
  for (DSNode::edge_iterator I = N->edge_begin(),E = N->edge_end(); I != E; ++I)
    if (DSNode *DSN = I->getNode())
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
  // Mark any incoming arguments as incomplete.
  if (Flags & DSGraph::MarkFormalArgs)
    for (ReturnNodesTy::iterator FI = ReturnNodes.begin(), E =ReturnNodes.end();
         FI != E; ++FI) {
      Function &F = *FI->first;
      for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end();
           I != E; ++I)
        if (isPointerType(I->getType()))
          markIncompleteNode(getNodeForValue(I).getNode());
      markIncompleteNode(FI->second.getNode());
    }

  // Mark stuff passed into functions calls as being incomplete.
  if (!shouldPrintAuxCalls())
    for (std::list<DSCallSite>::iterator I = FunctionCalls.begin(),
           E = FunctionCalls.end(); I != E; ++I)
      markIncomplete(*I);
  else
    for (std::list<DSCallSite>::iterator I = AuxFunctionCalls.begin(),
           E = AuxFunctionCalls.end(); I != E; ++I)
      markIncomplete(*I);

  // Mark all global nodes as incomplete.
  for (DSScalarMap::global_iterator I = ScalarMap.global_begin(),
         E = ScalarMap.global_end(); I != E; ++I)
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(*I))
      if (!GV->hasInitializer() ||    // Always mark external globals incomp.
          (!GV->isConstant() && (Flags & DSGraph::IgnoreGlobals) == 0))
        markIncompleteNode(ScalarMap[GV].getNode());
}

static inline void killIfUselessEdge(DSNodeHandle &Edge) {
  if (DSNode *N = Edge.getNode())  // Is there an edge?
    if (N->getNumReferrers() == 1)  // Does it point to a lonely node?
      // No interesting info?
      if ((N->getNodeFlags() & ~DSNode::Incomplete) == 0 &&
          N->getType() == Type::VoidTy && !N->isNodeCompletelyFolded())
        Edge.setTo(0, 0);  // Kill the edge!
}

static inline bool nodeContainsExternalFunction(const DSNode *N) {
  std::vector<Function*> Funcs;
  N->addFullFunctionList(Funcs);
  for (unsigned i = 0, e = Funcs.size(); i != e; ++i)
    if (Funcs[i]->isExternal()) return true;
  return false;
}

static void removeIdenticalCalls(std::list<DSCallSite> &Calls) {
  // Remove trivially identical function calls
  Calls.sort();  // Sort by callee as primary key!

  // Scan the call list cleaning it up as necessary...
  DSNodeHandle LastCalleeNode;
  Function *LastCalleeFunc = 0;
  unsigned NumDuplicateCalls = 0;
  bool LastCalleeContainsExternalFunction = false;

  unsigned NumDeleted = 0;
  for (std::list<DSCallSite>::iterator I = Calls.begin(), E = Calls.end();
       I != E;) {
    DSCallSite &CS = *I;
    std::list<DSCallSite>::iterator OldIt = I++;

    if (!CS.isIndirectCall()) {
      LastCalleeNode = 0;
    } else {
      DSNode *Callee = CS.getCalleeNode();

      // If the Callee is a useless edge, this must be an unreachable call site,
      // eliminate it.
      if (Callee->getNumReferrers() == 1 && Callee->isComplete() &&
          Callee->getGlobalsList().empty()) {  // No useful info?
#ifndef NDEBUG
        std::cerr << "WARNING: Useless call site found.\n";
#endif
        Calls.erase(OldIt);
        ++NumDeleted;
        continue;
      }

      // If the last call site in the list has the same callee as this one, and
      // if the callee contains an external function, it will never be
      // resolvable, just merge the call sites.
      if (!LastCalleeNode.isNull() && LastCalleeNode.getNode() == Callee) {
        LastCalleeContainsExternalFunction =
          nodeContainsExternalFunction(Callee);

        std::list<DSCallSite>::iterator PrevIt = OldIt;
        --PrevIt;
        PrevIt->mergeWith(CS);

        // No need to keep this call anymore.
        Calls.erase(OldIt);
        ++NumDeleted;
        continue;
      } else {
        LastCalleeNode = Callee;
      }
    }

    // If the return value or any arguments point to a void node with no
    // information at all in it, and the call node is the only node to point
    // to it, remove the edge to the node (killing the node).
    //
    killIfUselessEdge(CS.getRetVal());
    for (unsigned a = 0, e = CS.getNumPtrArgs(); a != e; ++a)
      killIfUselessEdge(CS.getPtrArg(a));

#if 0
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

        std::list<DSCallSite>::iterator PrevIt = OldIt;
        --PrevIt;
        PrevIt->mergeWith(CS);

        // No need to keep this call anymore.
        Calls.erase(OldIt);
        ++NumDeleted;
        continue;
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
#endif

    if (I != Calls.end() && CS == *I) {
      LastCalleeNode = 0;
      Calls.erase(OldIt);
      ++NumDeleted;
      continue;
    }
  }

  // Resort now that we simplified things.
  Calls.sort();

  // Now that we are in sorted order, eliminate duplicates.
  std::list<DSCallSite>::iterator CI = Calls.begin(), CE = Calls.end();
  if (CI != CE)
    while (1) {
      std::list<DSCallSite>::iterator OldIt = CI++;
      if (CI == CE) break;

      // If this call site is now the same as the previous one, we can delete it
      // as a duplicate.
      if (*OldIt == *CI) {
        Calls.erase(CI);
        CI = OldIt;
        ++NumDeleted;
      }
    }

  //Calls.erase(std::unique(Calls.begin(), Calls.end()), Calls.end());

  // Track the number of call nodes merged away...
  NumCallNodesMerged += NumDeleted;

  DEBUG(if (NumDeleted)
          std::cerr << "Merged " << NumDeleted << " call nodes.\n";);
}


// removeTriviallyDeadNodes - After the graph has been constructed, this method
// removes all unreachable nodes that are created because they got merged with
// other nodes in the graph.  These nodes will all be trivially unreachable, so
// we don't have to perform any non-trivial analysis here.
//
void DSGraph::removeTriviallyDeadNodes() {
  TIME_REGION(X, "removeTriviallyDeadNodes");

#if 0
  /// NOTE: This code is disabled.  This slows down DSA on 177.mesa
  /// substantially!

  // Loop over all of the nodes in the graph, calling getNode on each field.
  // This will cause all nodes to update their forwarding edges, causing
  // forwarded nodes to be delete-able.
  { TIME_REGION(X, "removeTriviallyDeadNodes:node_iterate");
  for (node_iterator NI = node_begin(), E = node_end(); NI != E; ++NI) {
    DSNode &N = *NI;
    for (unsigned l = 0, e = N.getNumLinks(); l != e; ++l)
      N.getLink(l*N.getPointerSize()).getNode();
  }
  }

  // NOTE: This code is disabled.  Though it should, in theory, allow us to
  // remove more nodes down below, the scan of the scalar map is incredibly
  // expensive for certain programs (with large SCCs).  In the future, if we can
  // make the scalar map scan more efficient, then we can reenable this.
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
      if (Node.getNumReferrers() == Node.getGlobalsList().size()) {
        const std::vector<GlobalValue*> &Globals = Node.getGlobalsList();

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
void DSNode::markReachableNodes(hash_set<const DSNode*> &ReachableNodes) const {
  if (this == 0) return;
  assert(getForwardNode() == 0 && "Cannot mark a forwarded node!");
  if (ReachableNodes.insert(this).second)        // Is newly reachable?
    for (DSNode::const_edge_iterator I = edge_begin(), E = edge_end();
         I != E; ++I)
      I->getNode()->markReachableNodes(ReachableNodes);
}

void DSCallSite::markReachableNodes(hash_set<const DSNode*> &Nodes) const {
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
static bool CanReachAliveNodes(DSNode *N, hash_set<const DSNode*> &Alive,
                               hash_set<const DSNode*> &Visited,
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

  for (DSNode::edge_iterator I = N->edge_begin(),E = N->edge_end(); I != E; ++I)
    if (CanReachAliveNodes(I->getNode(), Alive, Visited, IgnoreGlobals)) {
      N->markReachableNodes(Alive);
      return true;
    }
  return false;
}

// CallSiteUsesAliveArgs - Return true if the specified call site can reach any
// alive nodes.
//
static bool CallSiteUsesAliveArgs(const DSCallSite &CS,
                                  hash_set<const DSNode*> &Alive,
                                  hash_set<const DSNode*> &Visited,
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
  hash_set<const DSNode*> Alive;
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
  for (DSScalarMap::iterator I = ScalarMap.begin(), E = ScalarMap.end();
       I != E; ++I)
    if (isa<GlobalValue>(I->first)) {             // Keep track of global nodes
      assert(!I->second.isNull() && "Null global node?");
      assert(I->second.getNode()->isGlobalNode() && "Should be a global node!");
      GlobalNodes.push_back(std::make_pair(I->first, I->second.getNode()));

      // Make sure that all globals are cloned over as roots.
      if (!(Flags & DSGraph::RemoveUnreachableGlobals) && GlobalsGraph) {
        DSGraph::ScalarMapTy::iterator SMI =
          GlobalsGraph->getScalarMap().find(I->first);
        if (SMI != GlobalsGraph->getScalarMap().end())
          GGCloner.merge(SMI->second, I->second);
        else
          GGCloner.getClonedNH(I->second);
      }
    } else {
      I->second.getNode()->markReachableNodes(Alive);
    }
}

  // The return values are alive as well.
  for (ReturnNodesTy::iterator I = ReturnNodes.begin(), E = ReturnNodes.end();
       I != E; ++I)
    I->second.getNode()->markReachableNodes(Alive);

  // Mark any nodes reachable by primary calls as alive...
  for (fc_iterator I = fc_begin(), E = fc_end(); I != E; ++I)
    I->markReachableNodes(Alive);


  // Now find globals and aux call nodes that are already live or reach a live
  // value (which makes them live in turn), and continue till no more are found.
  //
  bool Iterate;
  hash_set<const DSNode*> Visited;
  hash_set<const DSCallSite*> AuxFCallsAlive;
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
    for (afc_iterator CI = afc_begin(), E = afc_end(); CI != E; ++CI)
      if (!AuxFCallsAlive.count(&*CI) &&
          (CI->isIndirectCall()
           || CallSiteUsesAliveArgs(*CI, Alive, Visited,
                                  Flags & DSGraph::RemoveUnreachableGlobals))) {
        CI->markReachableNodes(Alive);
        AuxFCallsAlive.insert(&*CI);
        Iterate = true;
      }
  } while (Iterate);

  // Move dead aux function calls to the end of the list
  unsigned CurIdx = 0;
  for (std::list<DSCallSite>::iterator CI = AuxFunctionCalls.begin(),
         E = AuxFunctionCalls.end(); CI != E; )
    if (AuxFCallsAlive.count(&*CI))
      ++CI;
    else {
      // Copy and merge global nodes and dead aux call nodes into the
      // GlobalsGraph, and all nodes reachable from those nodes.  Update their
      // target pointers using the GGCloner.
      //
      if (!(Flags & DSGraph::RemoveUnreachableGlobals))
        GlobalsGraph->AuxFunctionCalls.push_back(DSCallSite(*CI, GGCloner));

      AuxFunctionCalls.erase(CI++);
    }

  // We are finally done with the GGCloner so we can destroy it.
  GGCloner.destroy();

  // At this point, any nodes which are visited, but not alive, are nodes
  // which can be removed.  Loop over all nodes, eliminating completely
  // unreachable nodes.
  //
  std::vector<DSNode*> DeadNodes;
  DeadNodes.reserve(Nodes.size());
  for (NodeListTy::iterator NI = Nodes.begin(), E = Nodes.end(); NI != E;) {
    DSNode *N = NI++;
    assert(!N->isForwarding() && "Forwarded node in nodes list?");

    if (!Alive.count(N)) {
      Nodes.remove(N);
      assert(!N->isForwarding() && "Cannot remove a forwarding node!");
      DeadNodes.push_back(N);
      N->dropAllReferences();
      ++NumDNE;
    }
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

void DSGraph::AssertNodeContainsGlobal(const DSNode *N, GlobalValue *GV) const {
  assert(std::find(N->globals_begin(),N->globals_end(), GV) !=
         N->globals_end() && "Global value not in node!");
}

void DSGraph::AssertCallSiteInGraph(const DSCallSite &CS) const {
  if (CS.isIndirectCall()) {
    AssertNodeInGraph(CS.getCalleeNode());
#if 0
    if (CS.getNumPtrArgs() && CS.getCalleeNode() == CS.getPtrArg(0).getNode() &&
        CS.getCalleeNode() && CS.getCalleeNode()->getGlobals().empty())
      std::cerr << "WARNING: WEIRD CALL SITE FOUND!\n";
#endif
  }
  AssertNodeInGraph(CS.getRetVal().getNode());
  for (unsigned j = 0, e = CS.getNumPtrArgs(); j != e; ++j)
    AssertNodeInGraph(CS.getPtrArg(j).getNode());
}

void DSGraph::AssertCallNodesInGraph() const {
  for (fc_iterator I = fc_begin(), E = fc_end(); I != E; ++I)
    AssertCallSiteInGraph(*I);
}
void DSGraph::AssertAuxCallNodesInGraph() const {
  for (afc_iterator I = afc_begin(), E = afc_end(); I != E; ++I)
    AssertCallSiteInGraph(*I);
}

void DSGraph::AssertGraphOK() const {
  for (node_const_iterator NI = node_begin(), E = node_end(); NI != E; ++NI)
    NI->assertOK();

  for (ScalarMapTy::const_iterator I = ScalarMap.begin(),
         E = ScalarMap.end(); I != E; ++I) {
    assert(!I->second.isNull() && "Null node in scalarmap!");
    AssertNodeInGraph(I->second.getNode());
    if (GlobalValue *GV = dyn_cast<GlobalValue>(I->first)) {
      assert(I->second.getNode()->isGlobalNode() &&
             "Global points to node, but node isn't global?");
      AssertNodeContainsGlobal(I->second.getNode(), GV);
    }
  }
  AssertCallNodesInGraph();
  AssertAuxCallNodesInGraph();

  // Check that all pointer arguments to any functions in this graph have
  // destinations.
  for (ReturnNodesTy::const_iterator RI = ReturnNodes.begin(),
         E = ReturnNodes.end();
       RI != E; ++RI) {
    Function &F = *RI->first;
    for (Function::arg_iterator AI = F.arg_begin(); AI != F.arg_end(); ++AI)
      if (isPointerType(AI->getType()))
        assert(!getNodeForValue(AI).isNull() &&
               "Pointer argument must be in the scalar map!");
  }
}

/// computeNodeMapping - Given roots in two different DSGraphs, traverse the
/// nodes reachable from the two graphs, computing the mapping of nodes from the
/// first to the second graph.  This mapping may be many-to-one (i.e. the first
/// graph may have multiple nodes representing one node in the second graph),
/// but it will not work if there is a one-to-many or many-to-many mapping.
///
void DSGraph::computeNodeMapping(const DSNodeHandle &NH1,
                                 const DSNodeHandle &NH2, NodeMapTy &NodeMap,
                                 bool StrictChecking) {
  DSNode *N1 = NH1.getNode(), *N2 = NH2.getNode();
  if (N1 == 0 || N2 == 0) return;

  DSNodeHandle &Entry = NodeMap[N1];
  if (!Entry.isNull()) {
    // Termination of recursion!
    if (StrictChecking) {
      assert(Entry.getNode() == N2 && "Inconsistent mapping detected!");
      assert((Entry.getOffset() == (NH2.getOffset()-NH1.getOffset()) ||
              Entry.getNode()->isNodeCompletelyFolded()) &&
             "Inconsistent mapping detected!");
    }
    return;
  }

  Entry.setTo(N2, NH2.getOffset()-NH1.getOffset());

  // Loop over all of the fields that N1 and N2 have in common, recursively
  // mapping the edges together now.
  int N2Idx = NH2.getOffset()-NH1.getOffset();
  unsigned N2Size = N2->getSize();
  if (N2Size == 0) return;   // No edges to map to.

  for (unsigned i = 0, e = N1->getSize(); i < e; i += DS::PointerSize) {
    const DSNodeHandle &N1NH = N1->getLink(i);
    // Don't call N2->getLink if not needed (avoiding crash if N2Idx is not
    // aligned right).
    if (!N1NH.isNull()) {
      if (unsigned(N2Idx)+i < N2Size)
        computeNodeMapping(N1NH, N2->getLink(N2Idx+i), NodeMap);
      else
        computeNodeMapping(N1NH,
                           N2->getLink(unsigned(N2Idx+i) % N2Size), NodeMap);
    }
  }
}


/// computeGToGGMapping - Compute the mapping of nodes in the global graph to
/// nodes in this graph.
void DSGraph::computeGToGGMapping(NodeMapTy &NodeMap) {
  DSGraph &GG = *getGlobalsGraph();

  DSScalarMap &SM = getScalarMap();
  for (DSScalarMap::global_iterator I = SM.global_begin(),
         E = SM.global_end(); I != E; ++I)
    DSGraph::computeNodeMapping(SM[*I], GG.getNodeForValue(*I), NodeMap);
}

/// computeGGToGMapping - Compute the mapping of nodes in the global graph to
/// nodes in this graph.  Note that any uses of this method are probably bugs,
/// unless it is known that the globals graph has been merged into this graph!
void DSGraph::computeGGToGMapping(InvNodeMapTy &InvNodeMap) {
  NodeMapTy NodeMap;
  computeGToGGMapping(NodeMap);

  while (!NodeMap.empty()) {
    InvNodeMap.insert(std::make_pair(NodeMap.begin()->second,
                                     NodeMap.begin()->first));
    NodeMap.erase(NodeMap.begin());
  }
}


/// computeCalleeCallerMapping - Given a call from a function in the current
/// graph to the 'Callee' function (which lives in 'CalleeGraph'), compute the
/// mapping of nodes from the callee to nodes in the caller.
void DSGraph::computeCalleeCallerMapping(DSCallSite CS, const Function &Callee,
                                         DSGraph &CalleeGraph,
                                         NodeMapTy &NodeMap) {

  DSCallSite CalleeArgs =
    CalleeGraph.getCallSiteForArguments(const_cast<Function&>(Callee));

  computeNodeMapping(CalleeArgs.getRetVal(), CS.getRetVal(), NodeMap);

  unsigned NumArgs = CS.getNumPtrArgs();
  if (NumArgs > CalleeArgs.getNumPtrArgs())
    NumArgs = CalleeArgs.getNumPtrArgs();

  for (unsigned i = 0; i != NumArgs; ++i)
    computeNodeMapping(CalleeArgs.getPtrArg(i), CS.getPtrArg(i), NodeMap);

  // Map the nodes that are pointed to by globals.
  DSScalarMap &CalleeSM = CalleeGraph.getScalarMap();
  DSScalarMap &CallerSM = getScalarMap();

  if (CalleeSM.global_size() >= CallerSM.global_size()) {
    for (DSScalarMap::global_iterator GI = CallerSM.global_begin(),
           E = CallerSM.global_end(); GI != E; ++GI)
      if (CalleeSM.global_count(*GI))
        computeNodeMapping(CalleeSM[*GI], CallerSM[*GI], NodeMap);
  } else {
    for (DSScalarMap::global_iterator GI = CalleeSM.global_begin(),
           E = CalleeSM.global_end(); GI != E; ++GI)
      if (CallerSM.global_count(*GI))
        computeNodeMapping(CalleeSM[*GI], CallerSM[*GI], NodeMap);
  }
}

/// updateFromGlobalGraph - This function rematerializes global nodes and
/// nodes reachable from them from the globals graph into the current graph.
///
void DSGraph::updateFromGlobalGraph() {
  TIME_REGION(X, "updateFromGlobalGraph");
  ReachabilityCloner RC(*this, *GlobalsGraph, 0);

  // Clone the non-up-to-date global nodes into this graph.
  for (DSScalarMap::global_iterator I = getScalarMap().global_begin(),
         E = getScalarMap().global_end(); I != E; ++I) {
    DSScalarMap::iterator It = GlobalsGraph->ScalarMap.find(*I);
    if (It != GlobalsGraph->ScalarMap.end())
      RC.merge(getNodeForValue(*I), It->second);
  }
}
