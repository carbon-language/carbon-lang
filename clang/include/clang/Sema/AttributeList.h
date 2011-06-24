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
#include "llvm/ADT/SmallVector.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/VersionTuple.h"
#include <cassert>

namespace clang {
  class ASTContext;
  class IdentifierInfo;
  class Expr;

/// \brief Represents information about a change in availability for
/// an entity, which is part of the encoding of the 'availability'
/// attribute.
struct AvailabilityChange {
  /// \brief The location of the keyword indicating the kind of change.
  SourceLocation KeywordLoc;

  /// \brief The version number at which the change occurred.
  VersionTuple Version;

  /// \brief The source range covering the version number.
  SourceRange VersionRange;

  /// \brief Determine whether this availability change is valid.
  bool isValid() const { return !Version.empty(); }
};

/// AttributeList - Represents GCC's __attribute__ declaration. There are
/// 4 forms of this construct...they are:
///
/// 1: __attribute__(( const )). ParmName/Args/NumArgs will all be unused.
/// 2: __attribute__(( mode(byte) )). ParmName used, Args/NumArgs unused.
/// 3: __attribute__(( format(printf, 1, 2) )). ParmName/Args/NumArgs all used.
/// 4: __attribute__(( aligned(16) )). ParmName is unused, Args/Num used.
///
class AttributeList { // TODO: This should really be called ParsedAttribute
private:
  IdentifierInfo *AttrName;
  IdentifierInfo *ScopeName;
  IdentifierInfo *ParmName;
  SourceLocation AttrLoc;
  SourceLocation ScopeLoc;
  SourceLocation ParmLoc;

  /// The number of expression arguments this attribute has.
  /// The expressions themselves are stored after the object.
  unsigned NumArgs : 16;

  /// True if Microsoft style: declspec(foo).
  unsigned DeclspecAttribute : 1;

  /// True if C++0x-style: [[foo]].
  unsigned CXX0XAttribute : 1;

  /// True if already diagnosed as invalid.
  mutable unsigned Invalid : 1;

  /// True if this has the extra information associated with an
  /// availability attribute.
  unsigned IsAvailability : 1;

  /// \brief The location of the 'unavailable' keyword in an
  /// availability attribute.
  SourceLocation UnavailableLoc;

  /// The next attribute in the current position.
  AttributeList *NextInPosition;

  /// The next attribute allocated in the current Pool.
  AttributeList *NextInPool;

  Expr **getArgsBuffer() {
    return reinterpret_cast<Expr**>(this+1);
  }
  Expr * const *getArgsBuffer() const {
    return reinterpret_cast<Expr* const *>(this+1);
  }

  enum AvailabilitySlot {
    IntroducedSlot, DeprecatedSlot, ObsoletedSlot
  };

  AvailabilityChange &getAvailabilitySlot(AvailabilitySlot index) {
    return reinterpret_cast<AvailabilityChange*>(this+1)[index];
  }
  const AvailabilityChange &getAvailabilitySlot(AvailabilitySlot index) const {
    return reinterpret_cast<const AvailabilityChange*>(this+1)[index];
  }

  AttributeList(const AttributeList &); // DO NOT IMPLEMENT
  void operator=(const AttributeList &); // DO NOT IMPLEMENT
  void operator delete(void *); // DO NOT IMPLEMENT
  ~AttributeList(); // DO NOT IMPLEMENT

  size_t allocated_size() const;

  AttributeList(IdentifierInfo *attrName, SourceLocation attrLoc,
                IdentifierInfo *scopeName, SourceLocation scopeLoc,
                IdentifierInfo *parmName, SourceLocation parmLoc,
                Expr **args, unsigned numArgs,
                bool declspec, bool cxx0x)
    : AttrName(attrName), ScopeName(scopeName), ParmName(parmName),
      AttrLoc(attrLoc), ScopeLoc(scopeLoc), ParmLoc(parmLoc),
      NumArgs(numArgs),
      DeclspecAttribute(declspec), CXX0XAttribute(cxx0x), Invalid(false),
      IsAvailability(false), NextInPosition(0), NextInPool(0) {
    if (numArgs) memcpy(getArgsBuffer(), args, numArgs * sizeof(Expr*));
  }

  AttributeList(IdentifierInfo *attrName, SourceLocation attrLoc,
                IdentifierInfo *scopeName, SourceLocation scopeLoc,
                IdentifierInfo *parmName, SourceLocation parmLoc,
                const AvailabilityChange &introduced,
                const AvailabilityChange &deprecated,
                const AvailabilityChange &obsoleted,
                SourceLocation unavailable,
                bool declspec, bool cxx0x)
    : AttrName(attrName), ScopeName(scopeName), ParmName(parmName),
      AttrLoc(attrLoc), ScopeLoc(scopeLoc), ParmLoc(parmLoc),
      NumArgs(0), DeclspecAttribute(declspec), CXX0XAttribute(cxx0x),
      Invalid(false), IsAvailability(true), UnavailableLoc(unavailable),
      NextInPosition(0), NextInPool(0) {
    new (&getAvailabilitySlot(IntroducedSlot)) AvailabilityChange(introduced);
    new (&getAvailabilitySlot(DeprecatedSlot)) AvailabilityChange(deprecated);
    new (&getAvailabilitySlot(ObsoletedSlot)) AvailabilityChange(obsoleted);
  }

  friend class AttributePool;
  friend class AttributeFactory;

public:
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
    AT_availability,      // Clang-specific
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
    AT_format,
    AT_format_arg,
    AT_global,
    AT_gnu_inline,
    AT_host,
    AT_launch_bounds,
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
    AT_objc_method_family,
    AT_cf_returns_not_retained, // Clang-specific.
    AT_cf_returns_retained,     // Clang-specific.
    AT_cf_returns_autoreleased, // Clang-specific.
    AT_ns_returns_not_retained, // Clang-specific.
    AT_ns_returns_retained,     // Clang-specific.
    AT_ns_returns_autoreleased, // Clang-specific.
    AT_cf_consumed,             // Clang-specific.
    AT_ns_consumed,             // Clang-specific.
    AT_ns_consumes_self,        // Clang-specific.
    AT_objc_gc,
    AT_objc_ownership,          // Clang-specific.
    AT_objc_precise_lifetime,   // Clang-specific.
    AT_opencl_image_access,     // OpenCL-specific.
    AT_opencl_kernel_function,  // OpenCL-specific.
    AT_overloadable,       // Clang-specific.
    AT_ownership_holds,    // Clang-specific.
    AT_ownership_returns,  // Clang-specific.
    AT_ownership_takes,    // Clang-specific.
    AT_packed,
    AT_pascal,
    AT_pcs,  // ARM specific
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
    AT_uuid,
    AT_vecreturn,     // PS3 PPU-specific.
    AT_vector_size,
    AT_visibility,
    AT_warn_unused_result,
    AT_weak,
    AT_weakref,
    AT_weak_import,
    AT_reqd_wg_size,
    AT_init_priority,
    AT_MsStruct,
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

  AttributeList *getNext() const { return NextInPosition; }
  void setNext(AttributeList *N) { NextInPosition = N; }

  /// getNumArgs - Return the number of actual arguments to this attribute.
  unsigned getNumArgs() const { return NumArgs; }

  /// hasParameterOrArguments - Return true if this attribute has a parameter,
  /// or has a non empty argument expression list.
  bool hasParameterOrArguments() const { return ParmName || NumArgs; }

  /// getArg - Return the specified argument.
  Expr *getArg(unsigned Arg) const {
    assert(Arg < NumArgs && "Arg access out of range!");
    return getArgsBuffer()[Arg];
  }

  class arg_iterator {
    Expr * const *X;
    unsigned Idx;
  public:
    arg_iterator(Expr * const *x, unsigned idx) : X(x), Idx(idx) {}

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
    return arg_iterator(getArgsBuffer(), 0);
  }

  arg_iterator arg_end() const {
    return arg_iterator(getArgsBuffer(), NumArgs);
  }

  const AvailabilityChange &getAvailabilityIntroduced() const {
    assert(getKind() == AT_availability && "Not an availability attribute");
    return getAvailabilitySlot(IntroducedSlot);
  }

  const AvailabilityChange &getAvailabilityDeprecated() const {
    assert(getKind() == AT_availability && "Not an availability attribute");
    return getAvailabilitySlot(DeprecatedSlot);
  }

  const AvailabilityChange &getAvailabilityObsoleted() const {
    assert(getKind() == AT_availability && "Not an availability attribute");
    return getAvailabilitySlot(ObsoletedSlot);
  }

  SourceLocation getUnavailableLoc() const {
    assert(getKind() == AT_availability && "Not an availability attribute");
    return UnavailableLoc;
  }
};

/// A factory, from which one makes pools, from which one creates
/// individual attributes which are deallocated with the pool.
///
/// Note that it's tolerably cheap to create and destroy one of
/// these as long as you don't actually allocate anything in it.
class AttributeFactory {
public:
  enum {
    /// The required allocation size of an availability attribute,
    /// which we want to ensure is a multiple of sizeof(void*).
    AvailabilityAllocSize =
      sizeof(AttributeList)
      + ((3 * sizeof(AvailabilityChange) + sizeof(void*) - 1)
         / sizeof(void*) * sizeof(void*))
  };

private:
  enum {
    /// The number of free lists we want to be sure to support
    /// inline.  This is just enough that availability attributes
    /// don't surpass it.  It's actually very unlikely we'll see an
    /// attribute that needs more than that; on x86-64 you'd need 10
    /// expression arguments, and on i386 you'd need 19.
    InlineFreeListsCapacity =
      1 + (AvailabilityAllocSize - sizeof(AttributeList)) / sizeof(void*)
  };

  llvm::BumpPtrAllocator Alloc;

  /// Free lists.  The index is determined by the following formula:
  ///   (size - sizeof(AttributeList)) / sizeof(void*)
  llvm::SmallVector<AttributeList*, InlineFreeListsCapacity> FreeLists;

  // The following are the private interface used by AttributePool.
  friend class AttributePool;

  /// Allocate an attribute of the given size.
  void *allocate(size_t size);

  /// Reclaim all the attributes in the given pool chain, which is
  /// non-empty.  Note that the current implementation is safe
  /// against reclaiming things which were not actually allocated
  /// with the allocator, although of course it's important to make
  /// sure that their allocator lives at least as long as this one.
  void reclaimPool(AttributeList *head);

public:
  AttributeFactory();
  ~AttributeFactory();
};

class AttributePool {
  AttributeFactory &Factory;
  AttributeList *Head;

  void *allocate(size_t size) {
    return Factory.allocate(size);
  }

  AttributeList *add(AttributeList *attr) {
    // We don't care about the order of the pool.
    attr->NextInPool = Head;
    Head = attr;
    return attr;
  }

  void takePool(AttributeList *pool);

public:
  /// Create a new pool for a factory.
  AttributePool(AttributeFactory &factory) : Factory(factory), Head(0) {}

  /// Move the given pool's allocations to this pool.
  AttributePool(AttributePool &pool) : Factory(pool.Factory), Head(pool.Head) {
    pool.Head = 0;
  }

  AttributeFactory &getFactory() const { return Factory; }

  void clear() {
    if (Head) {
      Factory.reclaimPool(Head);
      Head = 0;
    }
  }

  /// Take the given pool's allocations and add them to this pool.
  void takeAllFrom(AttributePool &pool) {
    if (pool.Head) {
      takePool(pool.Head);
      pool.Head = 0;
    }
  }

  ~AttributePool() {
    if (Head) Factory.reclaimPool(Head);
  }

  AttributeList *create(IdentifierInfo *attrName, SourceLocation attrLoc,
                        IdentifierInfo *scopeName, SourceLocation scopeLoc,
                        IdentifierInfo *parmName, SourceLocation parmLoc,
                        Expr **args, unsigned numArgs,
                        bool declspec = false, bool cxx0x = false) {
    void *memory = allocate(sizeof(AttributeList)
                            + numArgs * sizeof(Expr*));
    return add(new (memory) AttributeList(attrName, attrLoc,
                                          scopeName, scopeLoc,
                                          parmName, parmLoc,
                                          args, numArgs,
                                          declspec, cxx0x));
  }

  AttributeList *create(IdentifierInfo *attrName, SourceLocation attrLoc,
                        IdentifierInfo *scopeName, SourceLocation scopeLoc,
                        IdentifierInfo *parmName, SourceLocation parmLoc,
                        const AvailabilityChange &introduced,
                        const AvailabilityChange &deprecated,
                        const AvailabilityChange &obsoleted,
                        SourceLocation unavailable,
                        bool declspec = false, bool cxx0x = false) {
    void *memory = allocate(AttributeFactory::AvailabilityAllocSize);
    return add(new (memory) AttributeList(attrName, attrLoc,
                                          scopeName, scopeLoc,
                                          parmName, parmLoc,
                                          introduced, deprecated, obsoleted,
                                          unavailable,
                                          declspec, cxx0x));
  }

  AttributeList *createIntegerAttribute(ASTContext &C, IdentifierInfo *Name,
                                        SourceLocation TokLoc, int Arg);
};

/// addAttributeLists - Add two AttributeLists together
/// The right-hand list is appended to the left-hand list, if any
/// A pointer to the joined list is returned.
/// Note: the lists are not left unmodified.
inline AttributeList *addAttributeLists(AttributeList *Left,
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

/// ParsedAttributes - A collection of parsed attributes.  Currently
/// we don't differentiate between the various attribute syntaxes,
/// which is basically silly.
///
/// Right now this is a very lightweight container, but the expectation
/// is that this will become significantly more serious.
class ParsedAttributes {
public:
  ParsedAttributes(AttributeFactory &factory)
    : pool(factory), list(0) {
  }

  ParsedAttributes(ParsedAttributes &attrs)
    : pool(attrs.pool), list(attrs.list) {
    attrs.list = 0;
  }

  AttributePool &getPool() const { return pool; }

  bool empty() const { return list == 0; }

  void add(AttributeList *newAttr) {
    assert(newAttr);
    assert(newAttr->getNext() == 0);
    newAttr->setNext(list);
    list = newAttr;
  }

  void addAll(AttributeList *newList) {
    if (!newList) return;

    AttributeList *lastInNewList = newList;
    while (AttributeList *next = lastInNewList->getNext())
      lastInNewList = next;

    lastInNewList->setNext(list);
    list = newList;
  }

  void set(AttributeList *newList) {
    list = newList;
  }

  void takeAllFrom(ParsedAttributes &attrs) {
    addAll(attrs.list);
    attrs.list = 0;
    pool.takeAllFrom(attrs.pool);
  }

  void clear() { list = 0; pool.clear(); }
  AttributeList *getList() const { return list; }

  /// Returns a reference to the attribute list.  Try not to introduce
  /// dependencies on this method, it may not be long-lived.
  AttributeList *&getListRef() { return list; }


  AttributeList *addNew(IdentifierInfo *attrName, SourceLocation attrLoc,
                        IdentifierInfo *scopeName, SourceLocation scopeLoc,
                        IdentifierInfo *parmName, SourceLocation parmLoc,
                        Expr **args, unsigned numArgs,
                        bool declspec = false, bool cxx0x = false) {
    AttributeList *attr =
      pool.create(attrName, attrLoc, scopeName, scopeLoc, parmName, parmLoc,
                  args, numArgs, declspec, cxx0x);
    add(attr);
    return attr;
  }

  AttributeList *addNew(IdentifierInfo *attrName, SourceLocation attrLoc,
                        IdentifierInfo *scopeName, SourceLocation scopeLoc,
                        IdentifierInfo *parmName, SourceLocation parmLoc,
                        const AvailabilityChange &introduced,
                        const AvailabilityChange &deprecated,
                        const AvailabilityChange &obsoleted,
                        SourceLocation unavailable,
                        bool declspec = false, bool cxx0x = false) {
    AttributeList *attr =
      pool.create(attrName, attrLoc, scopeName, scopeLoc, parmName, parmLoc,
                  introduced, deprecated, obsoleted, unavailable,
                  declspec, cxx0x);
    add(attr);
    return attr;
  }

  AttributeList *addNewInteger(ASTContext &C, IdentifierInfo *name,
                               SourceLocation loc, int arg) {
    AttributeList *attr =
      pool.createIntegerAttribute(C, name, loc, arg);
    add(attr);
    return attr;
  }


private:
  mutable AttributePool pool;
  AttributeList *list;
};

}  // end namespace clang

#endif
