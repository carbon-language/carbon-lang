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

#include "llvm/Metadata.h"
#include "llvm/Module.h"
using namespace llvm;

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

//===----------------------------------------------------------------------===//
//NamedMDNode implementation
//
NamedMDNode::NamedMDNode(const char *N, unsigned NameLength,
                         MetadataBase*const* MDs, unsigned NumMDs,
                         Module *M)
  : MetadataBase(Type::MetadataTy, Value::NamedMDNodeVal),
    Parent(M), Name(N, NameLength) {
  for (unsigned i = 0; i != NumMDs; ++i)
    Node.push_back(WeakMetadataVH(MDs[i]));

  // FIXME : Add into the parent module.
}
