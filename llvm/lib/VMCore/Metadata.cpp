//===-- Metadata.cpp - Implement Metadata classes -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Metadata classes.
//
//===----------------------------------------------------------------------===//

#include "LLVMContextImpl.h"
#include "llvm/Metadata.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "SymbolTableListTraitsImpl.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//MetadataBase implementation
//

/// resizeOperands - Metadata keeps track of other metadata uses using 
/// OperandList. Resize this list to hold anticipated number of metadata
/// operands.
void MetadataBase::resizeOperands(unsigned NumOps) {
  unsigned e = getNumOperands();
  if (NumOps == 0) {
    NumOps = e*2;
    if (NumOps < 2) NumOps = 2;  
  } else if (NumOps > NumOperands) {
    // No resize needed.
    if (ReservedSpace >= NumOps) return;
  } else if (NumOps == NumOperands) {
    if (ReservedSpace == NumOps) return;
  } else {
    return;
  }

  ReservedSpace = NumOps;
  Use *OldOps = OperandList;
  Use *NewOps = allocHungoffUses(NumOps);
  std::copy(OldOps, OldOps + e, NewOps);
  OperandList = NewOps;
  if (OldOps) Use::zap(OldOps, OldOps + e, true);
}
//===----------------------------------------------------------------------===//
//MDString implementation
//
MDString *MDString::get(LLVMContext &Context, const StringRef &Str) {
  LLVMContextImpl *pImpl = Context.pImpl;
  sys::SmartScopedWriter<true> Writer(pImpl->ConstantsLock);
  StringMapEntry<MDString *> &Entry = 
    pImpl->MDStringCache.GetOrCreateValue(Str);
  MDString *&S = Entry.getValue();
  if (!S) S = new MDString(Context, Entry.getKeyData(),
                           Entry.getKeyLength());

  return S;
}

//===----------------------------------------------------------------------===//
//MDNode implementation
//
MDNode::MDNode(LLVMContext &C, Value*const* Vals, unsigned NumVals)
  : MetadataBase(Type::getMetadataTy(C), Value::MDNodeVal) {
  NumOperands = 0;
  resizeOperands(NumVals);
  for (unsigned i = 0; i != NumVals; ++i) {
    // Only record metadata uses.
    if (MetadataBase *MB = dyn_cast_or_null<MetadataBase>(Vals[i]))
      OperandList[NumOperands++] = MB;
    else if(Vals[i] && 
            Vals[i]->getType()->getTypeID() == Type::MetadataTyID)
      OperandList[NumOperands++] = Vals[i];
    Node.push_back(ElementVH(Vals[i], this));
  }
}

void MDNode::Profile(FoldingSetNodeID &ID) const {
  for (const_elem_iterator I = elem_begin(), E = elem_end(); I != E; ++I)
    ID.AddPointer(*I);
}

MDNode *MDNode::get(LLVMContext &Context, Value*const* Vals, unsigned NumVals) {
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  for (unsigned i = 0; i != NumVals; ++i)
    ID.AddPointer(Vals[i]);

  // FIXME: MDNode uniquing disabled temporarily.
#ifndef ENABLE_MDNODE_UNIQUING
  return new MDNode(Context, Vals, NumVals);
#endif

  pImpl->ConstantsLock.reader_acquire();
  void *InsertPoint;
  MDNode *N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
  pImpl->ConstantsLock.reader_release();

  if (!N) {
    sys::SmartScopedWriter<true> Writer(pImpl->ConstantsLock);
    N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
    if (!N) {
      // InsertPoint will have been set by the FindNodeOrInsertPos call.
      N = new MDNode(Context, Vals, NumVals);
      pImpl->MDNodeSet.InsertNode(N, InsertPoint);
    }
  }

  return N;
}

/// dropAllReferences - Remove all uses and clear node vector.
void MDNode::dropAllReferences() {
  User::dropAllReferences();
  Node.clear();
}

MDNode::~MDNode() {
  // FIXME: MDNode uniquing disabled temporarily.
#ifdef ENABLE_MDNODE_UNIQUING
  getType()->getContext().pImpl->MDNodeSet.RemoveNode(this);
#endif
  dropAllReferences();
}

// Replace value from this node's element list.
void MDNode::replaceElement(Value *From, Value *To) {
  // FIXME: MDNode uniquing disabled temporarily.
#ifndef ENABLE_MDNODE_UNIQUING
  if (From == To || !getType())
    return;

  for (SmallVector<ElementVH, 4>::iterator I = Node.begin(),
         E = Node.end(); I != E; ++I)
    if (*I && *I == From)
      *I = ElementVH(To, this);
  return;
#endif

  if (From == To || !getType())
    return;
  LLVMContext &Context = getType()->getContext();
  LLVMContextImpl *pImpl = Context.pImpl;

  // Find value. This is a linear search, do something if it consumes 
  // lot of time. It is possible that to have multiple instances of
  // From in this MDNode's element list.
  SmallVector<unsigned, 4> Indexes;
  unsigned Index = 0;
  for (SmallVector<ElementVH, 4>::iterator I = Node.begin(),
         E = Node.end(); I != E; ++I, ++Index) {
    Value *V = *I;
    if (V && V == From) 
      Indexes.push_back(Index);
  }

  if (Indexes.empty())
    return;

  // Remove "this" from the context map. 
  {
    sys::SmartScopedWriter<true> Writer(pImpl->ConstantsLock);
    pImpl->MDNodeSet.RemoveNode(this);
  }

  // MDNode only lists metadata elements in operand list, because MDNode
  // used by MDNode is considered a valid use. However on the side, MDNode
  // using a non-metadata value is not considered a "use" of non-metadata
  // value.
  SmallVector<unsigned, 4> OpIndexes;
  unsigned OpIndex = 0;
  for (User::op_iterator OI = op_begin(), OE = op_end();
       OI != OE; ++OI, OpIndex++) {
    if (*OI == From)
      OpIndexes.push_back(OpIndex);
  }
  if (MetadataBase *MDTo = dyn_cast_or_null<MetadataBase>(To)) {
    for (SmallVector<unsigned, 4>::iterator OI = OpIndexes.begin(),
           OE = OpIndexes.end(); OI != OE; ++OI)
      setOperand(*OI, MDTo);
  } else {
    for (SmallVector<unsigned, 4>::iterator OI = OpIndexes.begin(),
           OE = OpIndexes.end(); OI != OE; ++OI)
      setOperand(*OI, 0);
  }

  // Replace From element(s) in place.
  for (SmallVector<unsigned, 4>::iterator I = Indexes.begin(), E = Indexes.end(); 
       I != E; ++I) {
    unsigned Index = *I;
    Node[Index] = ElementVH(To, this);
  }

  // Insert updated "this" into the context's folding node set.
  // If a node with same element list already exist then before inserting 
  // updated "this" into the folding node set, replace all uses of existing 
  // node with updated "this" node.
  FoldingSetNodeID ID;
  Profile(ID);
  pImpl->ConstantsLock.reader_acquire();
  void *InsertPoint;
  MDNode *N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
  pImpl->ConstantsLock.reader_release();

  if (N) {
    N->replaceAllUsesWith(this);
    delete N;
    N = 0;
  }

  {
    sys::SmartScopedWriter<true> Writer(pImpl->ConstantsLock);
    N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
    if (!N) {
      // InsertPoint will have been set by the FindNodeOrInsertPos call.
      N = this;
      pImpl->MDNodeSet.InsertNode(N, InsertPoint);
    }
  }
}

//===----------------------------------------------------------------------===//
//NamedMDNode implementation
//
NamedMDNode::NamedMDNode(LLVMContext &C, const Twine &N,
                         MetadataBase*const* MDs, 
                         unsigned NumMDs, Module *ParentModule)
  : MetadataBase(Type::getMetadataTy(C), Value::NamedMDNodeVal), Parent(0) {
  setName(N);
  NumOperands = 0;
  resizeOperands(NumMDs);

  for (unsigned i = 0; i != NumMDs; ++i) {
    if (MDs[i])
      OperandList[NumOperands++] = MDs[i];
    Node.push_back(WeakMetadataVH(MDs[i]));
  }
  if (ParentModule)
    ParentModule->getNamedMDList().push_back(this);
}

NamedMDNode *NamedMDNode::Create(const NamedMDNode *NMD, Module *M) {
  assert (NMD && "Invalid source NamedMDNode!");
  SmallVector<MetadataBase *, 4> Elems;
  for (unsigned i = 0, e = NMD->getNumElements(); i != e; ++i)
    Elems.push_back(NMD->getElement(i));
  return new NamedMDNode(NMD->getContext(), NMD->getName().data(),
                         Elems.data(), Elems.size(), M);
}

/// eraseFromParent - Drop all references and remove the node from parent
/// module.
void NamedMDNode::eraseFromParent() {
  getParent()->getNamedMDList().erase(this);
}

/// dropAllReferences - Remove all uses and clear node vector.
void NamedMDNode::dropAllReferences() {
  User::dropAllReferences();
  Node.clear();
}

NamedMDNode::~NamedMDNode() {
  dropAllReferences();
}
