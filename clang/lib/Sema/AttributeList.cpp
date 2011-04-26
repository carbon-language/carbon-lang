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
#include "llvm/ADT/StringSwitch.h"
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

AttributeList::Kind AttributeList::getKind(const IdentifierInfo *Name) {
  llvm::StringRef AttrName = Name->getName();

  // Normalize the attribute name, __foo__ becomes foo.
  if (AttrName.startswith("__") && AttrName.endswith("__"))
    AttrName = AttrName.substr(2, AttrName.size() - 4);

  return llvm::StringSwitch<AttributeList::Kind>(AttrName)
    .Case("weak", AT_weak)
    .Case("weakref", AT_weakref)
    .Case("pure", AT_pure)
    .Case("mode", AT_mode)
    .Case("used", AT_used)
    .Case("alias", AT_alias)
    .Case("align", AT_aligned)
    .Case("cdecl", AT_cdecl)
    .Case("const", AT_const)
    .Case("__const", AT_const) // some GCC headers do contain this spelling
    .Case("blocks", AT_blocks)
    .Case("format", AT_format)
    .Case("malloc", AT_malloc)
    .Case("packed", AT_packed)
    .Case("unused", AT_unused)
    .Case("aligned", AT_aligned)
    .Case("cleanup", AT_cleanup)
    .Case("naked", AT_naked)
    .Case("nodebug", AT_nodebug)
    .Case("nonnull", AT_nonnull)
    .Case("nothrow", AT_nothrow)
    .Case("objc_gc", AT_objc_gc)
    .Case("regparm", AT_regparm)
    .Case("section", AT_section)
    .Case("stdcall", AT_stdcall)
    .Case("annotate", AT_annotate)
    .Case("fastcall", AT_fastcall)
    .Case("ibaction", AT_IBAction)
    .Case("iboutlet", AT_IBOutlet)
    .Case("iboutletcollection", AT_IBOutletCollection)
    .Case("noreturn", AT_noreturn)
    .Case("noinline", AT_noinline)
    .Case("sentinel", AT_sentinel)
    .Case("NSObject", AT_nsobject)
    .Case("dllimport", AT_dllimport)
    .Case("dllexport", AT_dllexport)
    .Case("may_alias", AT_may_alias)
    .Case("base_check", AT_base_check)
    .Case("deprecated", AT_deprecated)
    .Case("availability", AT_availability)
    .Case("visibility", AT_visibility)
    .Case("destructor", AT_destructor)
    .Case("format_arg", AT_format_arg)
    .Case("gnu_inline", AT_gnu_inline)
    .Case("weak_import", AT_weak_import)
    .Case("vecreturn", AT_vecreturn)
    .Case("vector_size", AT_vector_size)
    .Case("constructor", AT_constructor)
    .Case("unavailable", AT_unavailable)
    .Case("overloadable", AT_overloadable)
    .Case("address_space", AT_address_space)
    .Case("opencl_image_access", AT_opencl_image_access)
    .Case("always_inline", AT_always_inline)
    .Case("returns_twice", IgnoredAttribute)
    .Case("vec_type_hint", IgnoredAttribute)
    .Case("objc_exception", AT_objc_exception)
    .Case("objc_method_family", AT_objc_method_family)
    .Case("ext_vector_type", AT_ext_vector_type)
    .Case("neon_vector_type", AT_neon_vector_type)
    .Case("neon_polyvector_type", AT_neon_polyvector_type)
    .Case("transparent_union", AT_transparent_union)
    .Case("analyzer_noreturn", AT_analyzer_noreturn)
    .Case("warn_unused_result", AT_warn_unused_result)
    .Case("carries_dependency", AT_carries_dependency)
    .Case("ns_consumed", AT_ns_consumed)
    .Case("ns_consumes_self", AT_ns_consumes_self)
    .Case("ns_returns_autoreleased", AT_ns_returns_autoreleased)
    .Case("ns_returns_not_retained", AT_ns_returns_not_retained)
    .Case("ns_returns_retained", AT_ns_returns_retained)
    .Case("cf_consumed", AT_cf_consumed)
    .Case("cf_returns_not_retained", AT_cf_returns_not_retained)
    .Case("cf_returns_retained", AT_cf_returns_retained)
    .Case("ownership_returns", AT_ownership_returns)
    .Case("ownership_holds", AT_ownership_holds)
    .Case("ownership_takes", AT_ownership_takes)
    .Case("reqd_work_group_size", AT_reqd_wg_size)
    .Case("init_priority", AT_init_priority)
    .Case("no_instrument_function", AT_no_instrument_function)
    .Case("thiscall", AT_thiscall)
    .Case("bounded", IgnoredAttribute)       // OpenBSD
    .Case("pascal", AT_pascal)
    .Case("__cdecl", AT_cdecl)
    .Case("__stdcall", AT_stdcall)
    .Case("__fastcall", AT_fastcall)
    .Case("__thiscall", AT_thiscall)
    .Case("__pascal", AT_pascal)
    .Case("constant", AT_constant)
    .Case("device", AT_device)
    .Case("global", AT_global)
    .Case("host", AT_host)
    .Case("shared", AT_shared)
    .Case("launch_bounds", AT_launch_bounds)
    .Case("common", AT_common)
    .Case("nocommon", AT_nocommon)
    .Case("opencl_kernel_function", AT_opencl_kernel_function)
    .Case("uuid", AT_uuid)
    .Case("pcs", AT_pcs)
    .Case("ms_struct", AT_MsStruct)
    .Default(UnknownAttribute);
}
