//===--- AttributeList.h - Parsed attribute sets ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AttributeList class, which is used to collect
// parsed attributes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_ATTRLIST_H
#define LLVM_CLANG_SEMA_ATTRLIST_H

#include "llvm/Support/Allocator.h"
#include "clang/Sema/Ownership.h"
#include "clang/Basic/SourceLocation.h"
#include <cassert>

namespace clang {
  class IdentifierInfo;
  class Expr;

/// AttributeList - Represents GCC's __attribute__ declaration. There are
/// 4 forms of this construct...they are:
///
/// 1: __attribute__(( const )). ParmName/Args/NumArgs will all be unused.
/// 2: __attribute__(( mode(byte) )). ParmName used, Args/NumArgs unused.
/// 3: __attribute__(( format(printf, 1, 2) )). ParmName/Args/NumArgs all used.
/// 4: __attribute__(( aligned(16) )). ParmName is unused, Args/Num used.
///
class AttributeList {
public:
  class Factory;
private:
  IdentifierInfo *AttrName;
  SourceLocation AttrLoc;
  IdentifierInfo *ScopeName;
  SourceLocation ScopeLoc;
  IdentifierInfo *ParmName;
  SourceLocation ParmLoc;
  Expr **Args;
  unsigned NumArgs;
  AttributeList *Next;
  bool DeclspecAttribute, CXX0XAttribute;
  mutable bool Invalid; /// True if already diagnosed as invalid.
  AttributeList(const AttributeList &); // DO NOT IMPLEMENT
  void operator=(const AttributeList &); // DO NOT IMPLEMENT
  void operator delete(void *); // DO NOT IMPLEMENT
  ~AttributeList(); // DO NOT IMPLEMENT
  AttributeList(llvm::BumpPtrAllocator &Alloc,
                IdentifierInfo *AttrName, SourceLocation AttrLoc,
                IdentifierInfo *ScopeName, SourceLocation ScopeLoc,
                IdentifierInfo *ParmName, SourceLocation ParmLoc,
                Expr **args, unsigned numargs,
                AttributeList *Next, bool declspec, bool cxx0x);
public:
  class Factory {
    llvm::BumpPtrAllocator Alloc;
  public:
    Factory() {}
    ~Factory() {}
    AttributeList *Create(IdentifierInfo *AttrName, SourceLocation AttrLoc,
      IdentifierInfo *ScopeName, SourceLocation ScopeLoc,
      IdentifierInfo *ParmName, SourceLocation ParmLoc,
      Expr **args, unsigned numargs,
      AttributeList *Next, bool declspec = false, bool cxx0x = false) {
        AttributeList *Mem = Alloc.Allocate<AttributeList>();
        new (Mem) AttributeList(Alloc, AttrName, AttrLoc, ScopeName, ScopeLoc,
                                ParmName, ParmLoc, args, numargs,
                                Next, declspec, cxx0x);
        return Mem;
      }
  };
  
  enum Kind {             // Please keep this list alphabetized.
    AT_IBAction,          // Clang-specific.
    AT_IBOutlet,          // Clang-specific.
    AT_IBOutletCollection, // Clang-specific.
    AT_address_space,
    AT_alias,
    AT_aligned,
    AT_always_inline,
    AT_analyzer_noreturn,
    AT_annotate,
    AT_base_check,
    AT_blocks,
    AT_carries_dependency,
    AT_cdecl,
    AT_cleanup,
    AT_common,
    AT_const,
    AT_constant,
    AT_constructor,
    AT_deprecated,
    AT_destructor,
    AT_device,
    AT_dllexport,
    AT_dllimport,
    AT_ext_vector_type,
    AT_fastcall,
    AT_final,
    AT_format,
    AT_format_arg,
    AT_global,
    AT_gnu_inline,
    AT_hiding,
    AT_host,
    AT_malloc,
    AT_may_alias,
    AT_mode,
    AT_neon_polyvector_type,    // Clang-specific.
    AT_neon_vector_type,        // Clang-specific.
    AT_naked,
    AT_nodebug,
    AT_noinline,
    AT_no_instrument_function,
    AT_nocommon,
    AT_nonnull,
    AT_noreturn,
    AT_nothrow,
    AT_nsobject,
    AT_objc_exception,
    AT_override,
    AT_cf_returns_not_retained, // Clang-specific.
    AT_cf_returns_retained,     // Clang-specific.
    AT_ns_returns_not_retained, // Clang-specific.
    AT_ns_returns_retained,     // Clang-specific.
    AT_objc_gc,
    AT_overloadable,       // Clang-specific.
    AT_ownership_holds,    // Clang-specific.
    AT_ownership_returns,  // Clang-specific.
    AT_ownership_takes,    // Clang-specific.
    AT_packed,
    AT_pascal,
    AT_pure,
    AT_regparm,
    AT_section,
    AT_sentinel,
    AT_shared,
    AT_stdcall,
    AT_thiscall,
    AT_transparent_union,
    AT_unavailable,
    AT_unused,
    AT_used,
    AT_vecreturn,     // PS3 PPU-specific.
    AT_vector_size,
    AT_visibility,
    AT_warn_unused_result,
    AT_weak,
    AT_weakref,
    AT_weak_import,
    AT_reqd_wg_size,
    AT_init_priority,
    IgnoredAttribute,
    UnknownAttribute
  };

  IdentifierInfo *getName() const { return AttrName; }
  SourceLocation getLoc() const { return AttrLoc; }
  
  bool hasScope() const { return ScopeName; }
  IdentifierInfo *getScopeName() const { return ScopeName; }
  SourceLocation getScopeLoc() const { return ScopeLoc; }
  
  IdentifierInfo *getParameterName() const { return ParmName; }
  SourceLocation getParameterLoc() const { return ParmLoc; }

  bool isDeclspecAttribute() const { return DeclspecAttribute; }
  bool isCXX0XAttribute() const { return CXX0XAttribute; }

  bool isInvalid() const { return Invalid; }
  void setInvalid(bool b = true) const { Invalid = b; }

  Kind getKind() const { return getKind(getName()); }
  static Kind getKind(const IdentifierInfo *Name);

  AttributeList *getNext() const { return Next; }
  void setNext(AttributeList *N) { Next = N; }

  /// getNumArgs - Return the number of actual arguments to this attribute.
  unsigned getNumArgs() const { return NumArgs; }

  /// getArg - Return the specified argument.
  Expr *getArg(unsigned Arg) const {
    assert(Arg < NumArgs && "Arg access out of range!");
    return Args[Arg];
  }

  class arg_iterator {
    Expr** X;
    unsigned Idx;
  public:
    arg_iterator(Expr** x, unsigned idx) : X(x), Idx(idx) {}

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

    Expr* operator*() const {
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

/// addAttributeLists - Add two AttributeLists together
/// The right-hand list is appended to the left-hand list, if any
/// A pointer to the joined list is returned.
/// Note: the lists are not left unmodified.
inline AttributeList* addAttributeLists (AttributeList *Left,
                                         AttributeList *Right) {
  if (!Left)
    return Right;

  AttributeList *next = Left, *prev;
  do {
    prev = next;
    next = next->getNext();
  } while (next);
  prev->setNext(Right);
  return Left;
}

/// CXX0XAttributeList - A wrapper around a C++0x attribute list.
/// Stores, in addition to the list proper, whether or not an actual list was
/// (as opposed to an empty list, which may be ill-formed in some places) and
/// the source range of the list.
struct CXX0XAttributeList { 
  AttributeList *AttrList;
  SourceRange Range;
  bool HasAttr;
  CXX0XAttributeList (AttributeList *attrList, SourceRange range, bool hasAttr)
    : AttrList(attrList), Range(range), HasAttr (hasAttr) {
  }
  CXX0XAttributeList ()
    : AttrList(0), Range(), HasAttr(false) {
  }
};

}  // end namespace clang

#endif
