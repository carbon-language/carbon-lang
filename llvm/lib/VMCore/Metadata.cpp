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
//MDString implementation
//
MDString *MDString::get(LLVMContext &Context, const StringRef &Str) {
  LLVMContextImpl *pImpl = Context.pImpl;
  sys::SmartScopedWriter<true> Writer(pImpl->ConstantsLock);
  StringMapEntry<MDString *> &Entry = 
    pImpl->MDStringCache.GetOrCreateValue(Str);
  MDString *&S = Entry.getValue();
  if (!S) S = new MDString(Entry.getKeyData(),
                           Entry.getKeyLength());

  return S;
}

//===----------------------------------------------------------------------===//
//MDNode implementation
//
MDNode::MDNode(Value*const* Vals, unsigned NumVals)
  : MetadataBase(Type::MetadataTy, Value::MDNodeVal) {
  for (unsigned i = 0; i != NumVals; ++i)
    Node.push_back(WeakVH(Vals[i]));
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

  pImpl->ConstantsLock.reader_acquire();
  void *InsertPoint;
  MDNode *N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
  pImpl->ConstantsLock.reader_release();
  
  if (!N) {
    sys::SmartScopedWriter<true> Writer(pImpl->ConstantsLock);
    N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
    if (!N) {
      // InsertPoint will have been set by the FindNodeOrInsertPos call.
      N = new MDNode(Vals, NumVals);
      pImpl->MDNodeSet.InsertNode(N, InsertPoint);
    }
  }

  return N;
}

//===----------------------------------------------------------------------===//
//NamedMDNode implementation
//
NamedMDNode::NamedMDNode(const Twine &N, MetadataBase*const* MDs, 
                         unsigned NumMDs, Module *ParentModule)
  : MetadataBase(Type::MetadataTy, Value::NamedMDNodeVal), Parent(0) {
  setName(N);
  for (unsigned i = 0; i != NumMDs; ++i)
    Node.push_back(WeakMetadataVH(MDs[i]));
  if (ParentModule)
    ParentModule->getNamedMDList().push_back(this);
}
