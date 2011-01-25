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
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/StringSwitch.h"
using namespace clang;

AttributeList::AttributeList(llvm::BumpPtrAllocator &Alloc,
                             IdentifierInfo *aName, SourceLocation aLoc,
                             IdentifierInfo *sName, SourceLocation sLoc,
                             IdentifierInfo *pName, SourceLocation pLoc,
                             Expr **ExprList, unsigned numArgs,
                             bool declspec, bool cxx0x)
  : AttrName(aName), AttrLoc(aLoc), ScopeName(sName),
    ScopeLoc(sLoc),
    ParmName(pName), ParmLoc(pLoc), NumArgs(numArgs), Next(0),
    DeclspecAttribute(declspec), CXX0XAttribute(cxx0x), Invalid(false) {

  if (numArgs == 0)
    Args = 0;
  else {
    // Allocate the Args array using the BumpPtrAllocator.
    Args = Alloc.Allocate<Expr*>(numArgs);
    memcpy(Args, ExprList, numArgs*sizeof(Args[0]));
  }
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
    .Case("visibility", AT_visibility)
    .Case("destructor", AT_destructor)
    .Case("forbid_temporaries", AT_forbid_temporaries)
    .Case("format_arg", AT_format_arg)
    .Case("gnu_inline", AT_gnu_inline)
    .Case("weak_import", AT_weak_import)
    .Case("vecreturn", AT_vecreturn)
    .Case("vector_size", AT_vector_size)
    .Case("constructor", AT_constructor)
    .Case("unavailable", AT_unavailable)
    .Case("overloadable", AT_overloadable)
    .Case("address_space", AT_address_space)
    .Case("always_inline", AT_always_inline)
    .Case("returns_twice", IgnoredAttribute)
    .Case("vec_type_hint", IgnoredAttribute)
    .Case("objc_exception", AT_objc_exception)
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
    .Case("uuid", AT_uuid)
    .Default(UnknownAttribute);
}
