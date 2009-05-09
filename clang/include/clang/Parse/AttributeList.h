//===--- AttributeList.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AttributeList class interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ATTRLIST_H
#define LLVM_CLANG_ATTRLIST_H

#include "clang/Parse/Ownership.h"
#include "clang/Basic/SourceLocation.h"
#include <cassert>

namespace clang {
  class IdentifierInfo;
  class Action;
  
/// AttributeList - Represents GCC's __attribute__ declaration. There are
/// 4 forms of this construct...they are:
///
/// 1: __attribute__(( const )). ParmName/Args/NumArgs will all be unused.
/// 2: __attribute__(( mode(byte) )). ParmName used, Args/NumArgs unused.
/// 3: __attribute__(( format(printf, 1, 2) )). ParmName/Args/NumArgs all used.
/// 4: __attribute__(( aligned(16) )). ParmName is unused, Args/Num used.
///
class AttributeList {
  IdentifierInfo *AttrName;
  SourceLocation AttrLoc;
  IdentifierInfo *ParmName;
  SourceLocation ParmLoc;
  ActionBase::ExprTy **Args;
  unsigned NumArgs;
  AttributeList *Next;
  AttributeList(const AttributeList &); // DO NOT IMPLEMENT
  void operator=(const AttributeList &); // DO NOT IMPLEMENT
public:
  AttributeList(IdentifierInfo *AttrName, SourceLocation AttrLoc,
                IdentifierInfo *ParmName, SourceLocation ParmLoc,
                ActionBase::ExprTy **args, unsigned numargs,
                AttributeList *Next);
  ~AttributeList();
  
  enum Kind {              // Please keep this list alphabetized.
    AT_IBOutlet,          // Clang-specific.
    AT_address_space,
    AT_alias,
    AT_aligned,
    AT_always_inline,
    AT_analyzer_noreturn,
    AT_annotate,
    AT_blocks,
    AT_cleanup,
    AT_const,
    AT_constructor,
    AT_deprecated,
    AT_destructor,
    AT_dllexport,
    AT_dllimport,
    AT_ext_vector_type,
    AT_fastcall,
    AT_format,
    AT_gnu_inline,
    AT_mode,
    AT_nodebug,
    AT_noinline,
    AT_no_instrument_function,
    AT_nonnull,
    AT_noreturn,
    AT_nothrow,
    AT_nsobject,
    AT_objc_exception,
    AT_cf_returns_retained,   // Clang-specific.
    AT_ns_returns_retained,   // Clang-specific.
    AT_objc_gc,
    AT_overloadable,       // Clang-specific.
    AT_packed,
    AT_pure,
    AT_regparm,
    AT_section,
    AT_sentinel,
    AT_stdcall,
    AT_transparent_union,
    AT_unavailable,
    AT_unused,
    AT_used,
    AT_vector_size,
    AT_visibility,
    AT_warn_unused_result,
    AT_weak,
    AT_weak_import,
    IgnoredAttribute,
    UnknownAttribute
  };
  
  IdentifierInfo *getName() const { return AttrName; }
  SourceLocation getLoc() const { return AttrLoc; }
  IdentifierInfo *getParameterName() const { return ParmName; }
  
  Kind getKind() const { return getKind(getName()); }
  static Kind getKind(const IdentifierInfo *Name);
  
  AttributeList *getNext() const { return Next; }
  void setNext(AttributeList *N) { Next = N; }
  
  void addAttributeList(AttributeList *alist) {
    assert((alist != 0) && "addAttributeList(): alist is null");
    AttributeList *next = this, *prev;
    do {
      prev = next;
      next = next->getNext();
    } while (next);
    prev->setNext(alist);
  }

  /// getNumArgs - Return the number of actual arguments to this attribute.
  unsigned getNumArgs() const { return NumArgs; }
  
  /// getArg - Return the specified argument.
  ActionBase::ExprTy *getArg(unsigned Arg) const {
    assert(Arg < NumArgs && "Arg access out of range!");
    return Args[Arg];
  }
  
  class arg_iterator {
    ActionBase::ExprTy** X;
    unsigned Idx;
  public:
    arg_iterator(ActionBase::ExprTy** x, unsigned idx) : X(x), Idx(idx) {}    

    arg_iterator& operator++() {
      ++Idx;
      return *this;
    }
    
    bool operator==(const arg_iterator& I) const {
      assert (X == I.X &&
              "compared arg_iterators are for different argument lists");
      return Idx == I.Idx;
    }
    
    bool operator!=(const arg_iterator& I) const {
      return !operator==(I);
    }
    
    ActionBase::ExprTy* operator*() const {
      return X[Idx];
    }
    
    unsigned getArgNum() const {
      return Idx+1;
    }
  };
  
  arg_iterator arg_begin() const {
    return arg_iterator(Args, 0);
  }

  arg_iterator arg_end() const {
    return arg_iterator(Args, NumArgs);
  }
};

}  // end namespace clang

#endif
