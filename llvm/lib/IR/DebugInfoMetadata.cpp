//===- DebugInfoMetadata.cpp - Implement debug info metadata --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the debug info Metadata classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoMetadata.h"
#include "LLVMContextImpl.h"
#include "MetadataImpl.h"

using namespace llvm;

MDLocation::MDLocation(LLVMContext &C, StorageType Storage, unsigned Line,
                       unsigned Column, ArrayRef<Metadata *> MDs)
    : MDNode(C, MDLocationKind, Storage, MDs) {
  assert((MDs.size() == 1 || MDs.size() == 2) &&
         "Expected a scope and optional inlined-at");

  // Set line and column.
  assert(Line < (1u << 24) && "Expected 24-bit line");
  assert(Column < (1u << 16) && "Expected 16-bit column");

  SubclassData32 = Line;
  SubclassData16 = Column;
}

static void adjustLine(unsigned &Line) {
  // Set to unknown on overflow.  Still use 24 bits for now.
  if (Line >= (1u << 24))
    Line = 0;
}

static void adjustColumn(unsigned &Column) {
  // Set to unknown on overflow.  We only have 16 bits to play with here.
  if (Column >= (1u << 16))
    Column = 0;
}

MDLocation *MDLocation::getImpl(LLVMContext &Context, unsigned Line,
                                unsigned Column, Metadata *Scope,
                                Metadata *InlinedAt, StorageType Storage,
                                bool ShouldCreate) {
  // Fixup line/column.
  adjustLine(Line);
  adjustColumn(Column);

  if (Storage == Uniqued) {
    if (auto *N =
            getUniqued(Context.pImpl->MDLocations,
                       MDLocationInfo::KeyTy(Line, Column, Scope, InlinedAt)))
      return N;
    if (!ShouldCreate)
      return nullptr;
  } else {
    assert(ShouldCreate && "Expected non-uniqued nodes to always be created");
  }

  SmallVector<Metadata *, 2> Ops;
  Ops.push_back(Scope);
  if (InlinedAt)
    Ops.push_back(InlinedAt);
  return storeImpl(new (Ops.size())
                       MDLocation(Context, Storage, Line, Column, Ops),
                   Storage, Context.pImpl->MDLocations);
}

static StringRef getString(const MDString *S) {
  if (S)
    return S->getString();
  return StringRef();
}

#ifndef NDEBUG
static bool isCanonical(const MDString *S) {
  return !S || !S->getString().empty();
}
#endif

GenericDebugNode *GenericDebugNode::getImpl(LLVMContext &Context, unsigned Tag,
                                            MDString *Header,
                                            ArrayRef<Metadata *> DwarfOps,
                                            StorageType Storage,
                                            bool ShouldCreate) {
  unsigned Hash = 0;
  if (Storage == Uniqued) {
    GenericDebugNodeInfo::KeyTy Key(Tag, getString(Header), DwarfOps);
    if (auto *N = getUniqued(Context.pImpl->GenericDebugNodes, Key))
      return N;
    if (!ShouldCreate)
      return nullptr;
    Hash = Key.getHash();
  } else {
    assert(ShouldCreate && "Expected non-uniqued nodes to always be created");
  }

  // Use a nullptr for empty headers.
  assert(isCanonical(Header) && "Expected canonical MDString");
  Metadata *PreOps[] = {Header};
  return storeImpl(new (DwarfOps.size() + 1) GenericDebugNode(
                       Context, Storage, Hash, Tag, PreOps, DwarfOps),
                   Storage, Context.pImpl->GenericDebugNodes);
}

void GenericDebugNode::recalculateHash() {
  setHash(GenericDebugNodeInfo::KeyTy::calculateHash(this));
}
