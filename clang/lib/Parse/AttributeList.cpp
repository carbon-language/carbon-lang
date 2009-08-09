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

#include "clang/Parse/AttributeList.h"
#include "clang/Basic/IdentifierTable.h"
using namespace clang;

AttributeList::AttributeList(IdentifierInfo *aName, SourceLocation aLoc,
                             IdentifierInfo *pName, SourceLocation pLoc,
                             ActionBase::ExprTy **ExprList, unsigned numArgs,
                             AttributeList *n, bool declspec)
  : AttrName(aName), AttrLoc(aLoc), ParmName(pName), ParmLoc(pLoc),
    NumArgs(numArgs), Next(n), DeclspecAttribute(declspec) {
  
  if (numArgs == 0)
    Args = 0;
  else {
    Args = new ActionBase::ExprTy*[numArgs];
    memcpy(Args, ExprList, numArgs*sizeof(Args[0]));
  }
}

AttributeList::~AttributeList() {
  if (Args) {
    // FIXME: before we delete the vector, we need to make sure the Expr's 
    // have been deleted. Since ActionBase::ExprTy is "void", we are dependent
    // on the actions module for actually freeing the memory. The specific
    // hooks are ActOnDeclarator, ActOnTypeName, ActOnParamDeclaratorType, 
    // ParseField, ParseTag. Once these routines have freed the expression, 
    // they should zero out the Args slot (to indicate the memory has been 
    // freed). If any element of the vector is non-null, we should assert.
    delete [] Args;
  }
  delete Next;
}

AttributeList::Kind AttributeList::getKind(const IdentifierInfo *Name) {
  const char *Str = Name->getName();
  unsigned Len = Name->getLength();

  // Normalize the attribute name, __foo__ becomes foo.
  if (Len > 4 && Str[0] == '_' && Str[1] == '_' &&
      Str[Len - 2] == '_' && Str[Len - 1] == '_') {
    Str += 2;
    Len -= 4;
  }
  
  // FIXME: Hand generating this is neither smart nor efficient.
  switch (Len) {
  case 4:
    if (!memcmp(Str, "weak", 4)) return AT_weak;
    if (!memcmp(Str, "pure", 4)) return AT_pure;
    if (!memcmp(Str, "mode", 4)) return AT_mode;
    if (!memcmp(Str, "used", 4)) return AT_used;
    break;
  case 5:
    if (!memcmp(Str, "alias", 5)) return AT_alias;
    if (!memcmp(Str, "const", 5)) return AT_const;
    break;
  case 6:
    if (!memcmp(Str, "packed", 6)) return AT_packed;
    if (!memcmp(Str, "malloc", 6)) return AT_malloc;
    if (!memcmp(Str, "format", 6)) return AT_format;
    if (!memcmp(Str, "unused", 6)) return AT_unused;
    if (!memcmp(Str, "blocks", 6)) return AT_blocks;
    break;
  case 7:
    if (!memcmp(Str, "aligned", 7)) return AT_aligned;
    if (!memcmp(Str, "cleanup", 7)) return AT_cleanup;
    if (!memcmp(Str, "nodebug", 7)) return AT_nodebug;
    if (!memcmp(Str, "nonnull", 7)) return AT_nonnull;
    if (!memcmp(Str, "nothrow", 7)) return AT_nothrow;
    if (!memcmp(Str, "objc_gc", 7)) return AT_objc_gc;
    if (!memcmp(Str, "regparm", 7)) return AT_regparm;
    if (!memcmp(Str, "section", 7)) return AT_section;
    if (!memcmp(Str, "stdcall", 7)) return AT_stdcall;
    break;
  case 8:
    if (!memcmp(Str, "annotate", 8)) return AT_annotate;
    if (!memcmp(Str, "noreturn", 8)) return AT_noreturn;
    if (!memcmp(Str, "noinline", 8)) return AT_noinline;
    if (!memcmp(Str, "fastcall", 8)) return AT_fastcall;
    if (!memcmp(Str, "iboutlet", 8)) return AT_IBOutlet;
    if (!memcmp(Str, "sentinel", 8)) return AT_sentinel;
    if (!memcmp(Str, "NSObject", 8)) return AT_nsobject;
    break;
  case 9:
    if (!memcmp(Str, "dllimport", 9)) return AT_dllimport;
    if (!memcmp(Str, "dllexport", 9)) return AT_dllexport;
    if (!memcmp(Str, "may_alias", 9)) return IgnoredAttribute; // FIXME: TBAA
    break;
  case 10:
    if (!memcmp(Str, "deprecated", 10)) return AT_deprecated;
    if (!memcmp(Str, "visibility", 10)) return AT_visibility;
    if (!memcmp(Str, "destructor", 10)) return AT_destructor;
    if (!memcmp(Str, "format_arg", 10)) return AT_format_arg; 
    if (!memcmp(Str, "gnu_inline", 10)) return AT_gnu_inline;
    break;
  case 11:
    if (!memcmp(Str, "weak_import", 11)) return AT_weak_import;
    if (!memcmp(Str, "vector_size", 11)) return AT_vector_size;
    if (!memcmp(Str, "constructor", 11)) return AT_constructor;
    if (!memcmp(Str, "unavailable", 11)) return AT_unavailable;
    break;
  case 12:
    if (!memcmp(Str, "overloadable", 12)) return AT_overloadable;
    break;
  case 13:
    if (!memcmp(Str, "address_space", 13)) return AT_address_space;
    if (!memcmp(Str, "always_inline", 13)) return AT_always_inline;
    if (!memcmp(Str, "vec_type_hint", 13)) return IgnoredAttribute;
    break;
  case 14:
    if (!memcmp(Str, "objc_exception", 14)) return AT_objc_exception;
    break;
  case 15:
    if (!memcmp(Str, "ext_vector_type", 15)) return AT_ext_vector_type;
    break;
  case 17:
    if (!memcmp(Str, "transparent_union", 17)) return AT_transparent_union;
    if (!memcmp(Str, "analyzer_noreturn", 17)) return AT_analyzer_noreturn;
    break;
  case 18:
    if (!memcmp(Str, "warn_unused_result", 18)) return AT_warn_unused_result;
    break;
  case 19:
    if (!memcmp(Str, "ns_returns_retained", 19)) return AT_ns_returns_retained;
    if (!memcmp(Str, "cf_returns_retained", 19)) return AT_cf_returns_retained;
    break;            
  case 20:
    if (!memcmp(Str, "reqd_work_group_size", 20)) return AT_reqd_wg_size;
  case 22:
    if (!memcmp(Str, "no_instrument_function", 22))
      return AT_no_instrument_function;
    break;
  }  
  return UnknownAttribute;
}
