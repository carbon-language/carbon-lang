//===--- AttributeList.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AttributeList class implementation
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/AttributeList.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/StringMap.h"
using namespace clang;

size_t AttributeList::allocated_size() const {
  if (IsAvailability) return AttributeFactory::AvailabilityAllocSize;
  return (sizeof(AttributeList) + NumArgs * sizeof(Expr*));
}

AttributeFactory::AttributeFactory() {
  // Go ahead and configure all the inline capacity.  This is just a memset.
  FreeLists.resize(InlineFreeListsCapacity);
}
AttributeFactory::~AttributeFactory() {}

static size_t getFreeListIndexForSize(size_t size) {
  assert(size >= sizeof(AttributeList));
  assert((size % sizeof(void*)) == 0);
  return ((size - sizeof(AttributeList)) / sizeof(void*));
}

void *AttributeFactory::allocate(size_t size) {
  // Check for a previously reclaimed attribute.
  size_t index = getFreeListIndexForSize(size);
  if (index < FreeLists.size()) {
    if (AttributeList *attr = FreeLists[index]) {
      FreeLists[index] = attr->NextInPool;
      return attr;
    }
  }

  // Otherwise, allocate something new.
  return Alloc.Allocate(size, llvm::AlignOf<AttributeFactory>::Alignment);
}

void AttributeFactory::reclaimPool(AttributeList *cur) {
  assert(cur && "reclaiming empty pool!");
  do {
    // Read this here, because we're going to overwrite NextInPool
    // when we toss 'cur' into the appropriate queue.
    AttributeList *next = cur->NextInPool;

    size_t size = cur->allocated_size();
    size_t freeListIndex = getFreeListIndexForSize(size);

    // Expand FreeLists to the appropriate size, if required.
    if (freeListIndex >= FreeLists.size())
      FreeLists.resize(freeListIndex+1);

    // Add 'cur' to the appropriate free-list.
    cur->NextInPool = FreeLists[freeListIndex];
    FreeLists[freeListIndex] = cur;
    
    cur = next;
  } while (cur);
}

void AttributePool::takePool(AttributeList *pool) {
  assert(pool);

  // Fast path:  this pool is empty.
  if (!Head) {
    Head = pool;
    return;
  }

  // Reverse the pool onto the current head.  This optimizes for the
  // pattern of pulling a lot of pools into a single pool.
  do {
    AttributeList *next = pool->NextInPool;
    pool->NextInPool = Head;
    Head = pool;
    pool = next;
  } while (pool);
}

AttributeList *
AttributePool::createIntegerAttribute(ASTContext &C, IdentifierInfo *Name,
                                      SourceLocation TokLoc, int Arg) {
  Expr *IArg = IntegerLiteral::Create(C, llvm::APInt(32, (uint64_t) Arg),
                                      C.IntTy, TokLoc);
  return create(Name, TokLoc, 0, TokLoc, 0, TokLoc, &IArg, 1, 0);
}


typedef llvm::StringMap<AttributeList::Kind> AttributeNameKindMap;

static AttributeNameKindMap createAttributeNameKindMap(){
  AttributeNameKindMap Result;
#include "clang/Sema/AttrParsedAttrKinds.inc"
  Result["address_space"] = AttributeList::AT_address_space;
  Result["align"] = AttributeList::AT_aligned; // FIXME: should it be "aligned"?
  Result["base_check"] = AttributeList::AT_base_check;
  Result["bounded"] = AttributeList::IgnoredAttribute; // OpenBSD
  Result["__const"] = AttributeList::AT_const; // some GCC headers do contain this spelling
  Result["cf_returns_autoreleased"] = AttributeList::AT_cf_returns_autoreleased;
  Result["mode"] = AttributeList::AT_mode;
  Result["vec_type_hint"] = AttributeList::IgnoredAttribute;
  Result["ext_vector_type"] = AttributeList::AT_ext_vector_type;
  Result["neon_vector_type"] = AttributeList::AT_neon_vector_type;
  Result["neon_polyvector_type"] = AttributeList::AT_neon_polyvector_type;
  Result["opencl_image_access"] = AttributeList::AT_opencl_image_access;
  Result["objc_gc"] = AttributeList::AT_objc_gc;
  Result["objc_ownership"] = AttributeList::AT_objc_ownership;
  Result["vector_size"] = AttributeList::AT_vector_size;
  return Result;
}

AttributeList::Kind AttributeList::getKind(const IdentifierInfo *Name) {
  StringRef AttrName = Name->getName();

  // Normalize the attribute name, __foo__ becomes foo.
  if (AttrName.startswith("__") && AttrName.endswith("__") &&
      AttrName.size() >= 4)
    AttrName = AttrName.substr(2, AttrName.size() - 4);

  static AttributeNameKindMap Map = createAttributeNameKindMap();
  AttributeNameKindMap::iterator Pos = Map.find(AttrName);
  if (Pos != Map.end())
    return Pos->second;
  
  return UnknownAttribute;
}
