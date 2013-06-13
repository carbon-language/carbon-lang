//===- Core/DefinedAtom.h - An Atom with content --------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_DEFINED_ATOM_H
#define LLD_CORE_DEFINED_ATOM_H

#include "lld/Core/Atom.h"
#include "lld/Core/Reference.h"

namespace llvm {
  template <typename T>
  class ArrayRef;
  class StringRef;
}

namespace lld {
class File;

/// \brief The fundamental unit of linking.
///
/// A C function or global variable is an atom.  An atom has content and
/// attributes. The content of a function atom is the instructions that
/// implement the function.  The content of a global variable atom is its
/// initial bytes.
///
/// Here are some example attribute sets for common atoms. If a particular
/// attribute is not listed, the default values are:  definition=regular,
/// sectionChoice=basedOnContent, scope=translationUnit, merge=no,
/// deadStrip=normal, interposable=no
///
///  C function:  void foo() {} <br>
///    name=foo, type=code, perm=r_x, scope=global
///
///  C static function:  staic void func() {} <br>
///    name=func, type=code, perm=r_x
///
///  C global variable:  int count = 1; <br>
///    name=count, type=data, perm=rw_, scope=global
///
///  C tentative definition:  int bar; <br>
///    name=bar, type=zerofill, perm=rw_, scope=global,
///    merge=asTentative, interposable=yesAndRuntimeWeak
///
///  Uninitialized C static variable:  static int stuff; <br>
///    name=stuff, type=zerofill, perm=rw_
///
///  Weak C function:  __attribute__((weak)) void foo() {} <br>
///    name=foo, type=code, perm=r_x, scope=global, merge=asWeak
///
///  Hidden C function:  __attribute__((visibility("hidden"))) void foo() {}<br>
///    name=foo, type=code, perm=r_x, scope=linkageUnit
///
///  No-dead-strip function:  __attribute__((used)) void foo() {} <br>
///    name=foo, type=code, perm=r_x, scope=global, deadStrip=never
///
///  Non-inlined C++ inline method:  inline void Foo::doit() {} <br>
///    name=_ZN3Foo4doitEv, type=code, perm=r_x, scope=global,
///    mergeDupes=asWeak
///
///  Non-inlined C++ inline method whose address is taken:
///     inline void Foo::doit() {} <br>
///    name=_ZN3Foo4doitEv, type=code, perm=r_x, scope=global,
///    mergeDupes=asAddressedWeak
///
///  literal c-string:  "hello" <br>
///    name="" type=cstring, perm=r__, scope=linkageUnit
///
///  literal double:  1.234 <br>
///    name="" type=literal8, perm=r__, scope=linkageUnit
///
///  constant:  { 1,2,3 } <br>
///    name="" type=constant, perm=r__, scope=linkageUnit
///
///  Pointer to initializer function:  <br>
///    name="" type=initializer, perm=rw_l,
///    sectionChoice=customRequired
///
///  C function place in custom section:  __attribute__((section("__foo")))
///                                       void foo() {} <br>
///    name=foo, type=code, perm=r_x, scope=global,
///    sectionChoice=customRequired, sectionName=__foo
///
class DefinedAtom : public Atom {
public:
  enum Interposable {
    interposeNo,            // linker can directly bind uses of this atom
    interposeYes,           // linker must indirect (through GOT) uses
    interposeYesAndRuntimeWeak // must indirect and mark symbol weak in final
                               // linked image
  };

  enum Merge {
    mergeNo,                // Another atom with same name is error
    mergeAsTentative,       // Is ANSI C tentative defintion, can be coalesced
    mergeAsWeak,            // is C++ inline definition that was not inlined,
                            // but address was not taken, so atom can be hidden
                            // by linker
    mergeAsWeakAndAddressUsed,// is C++ definition inline definition whose
                              // address was taken.
    mergeByContent          // merge with other constants with same content
  };

  enum ContentType {
    typeUnknown,            // for use with definitionUndefined
    typeCode,               // executable code
    typeResolver,           // function which returns address of target
    typeBranchIsland,       // linker created for large binaries
    typeBranchShim,         // linker created to switch thumb mode
    typeStub,               // linker created for calling external function
    typeStubHelper,         // linker created for initial stub binding
    typeConstant,           // a read-only constant
    typeCString,            // a zero terminated UTF8 C string
    typeUTF16String,        // a zero terminated UTF16 string
    typeCFI,                // a FDE or CIE from dwarf unwind info
    typeLSDA,               // extra unwinding info
    typeLiteral4,           // a four-btye read-only constant
    typeLiteral8,           // an eight-btye read-only constant
    typeLiteral16,          // a sixteen-btye read-only constant
    typeData,               // read-write data
    typeDataFast,           // allow data to be quickly accessed
    typeZeroFill,           // zero-fill data
    typeZeroFillFast,       // allow zero-fill data to be quicky accessed
    typeConstData,          // read-only data after dynamic linker is done
    typeObjC1Class,         // ObjC1 class [Darwin]
    typeLazyPointer,        // pointer through which a stub jumps
    typeLazyDylibPointer,   // pointer through which a stub jumps [Darwin]
    typeCFString,           // NS/CFString object [Darwin]
    typeGOT,                // pointer to external symbol
    typeInitializerPtr,     // pointer to initializer function
    typeTerminatorPtr,      // pointer to terminator function
    typeCStringPtr,         // pointer to UTF8 C string [Darwin]
    typeObjCClassPtr,       // pointer to ObjC class [Darwin]
    typeObjC2CategoryList,  // pointers to ObjC category [Darwin]
    typeDTraceDOF,          // runtime data for Dtrace [Darwin]
    typeTempLTO,            // temporary atom for bitcode reader
    typeCompactUnwindInfo,  // runtime data for unwinder [Darwin]
    typeThunkTLV,           // thunk used to access a TLV [Darwin]
    typeTLVInitialData,     // initial data for a TLV [Darwin]
    typeTLVInitialZeroFill, // TLV initial zero fill data [Darwin]
    typeTLVInitializerPtr,  // pointer to thread local initializer [Darwin]
  };

  // Permission bits for atoms and segments. The order of these values are
  // important, because the layout pass may sort atoms by permission if other
  // attributes are the same.
  enum ContentPermissions {
    perm___  = 0,           // mapped as unaccessible
    permR__  = 8,           // mapped read-only
    permRW_  = 8 + 2,       // mapped readable and writable
    permRW_L = 8 + 2 + 1,   // initially mapped r/w, then made read-only
                            // loader writable
    permR_X  = 8 + 4,       // mapped readable and executable
    permRWX  = 8 + 2 + 4,   // mapped readable and writable and executable
    permUnknown = 16        // unknown or invalid permissions
  };

  enum SectionChoice {
    sectionBasedOnContent,  // linker infers final section based on content
    sectionCustomPreferred, // linker may place in specific section
    sectionCustomRequired   // linker must place in specific section
  };

  enum SectionPosition {
    sectionPositionStart,   // atom must be at start of section (and zero size)
    sectionPositionEarly,   // atom should be near start of section
    sectionPositionAny,     // atom can be anywhere in section
    sectionPositionEnd      // atom must be at end of section (and zero size)
  };

  enum DeadStripKind {
    deadStripNormal,        // linker may dead strip this atom
    deadStripNever,         // linker must never dead strip this atom
    deadStripAlways         // linker must remove this atom if unused
  };

  struct Alignment {
    Alignment(int p2, int m = 0)
      : powerOf2(p2)
      , modulus(m) {}

    uint16_t powerOf2;
    uint16_t modulus;

    bool operator==(const Alignment &rhs) const {
      return (powerOf2 == rhs.powerOf2) && (modulus == rhs.modulus);
    }
  };

  /// \brief returns a value for the order of this Atom within its file.
  ///
  /// This is used by the linker to order the layout of Atoms so that the
  /// resulting image is stable and reproducible.
  ///
  /// Note that this should not be confused with ordinals of exported symbols in
  /// Windows DLLs. In Windows terminology, ordinals are symbols' export table
  /// indices (small integers) which can be used instead of symbol names to
  /// refer items in a DLL.
  virtual uint64_t ordinal() const = 0;

  /// \brief the number of bytes of space this atom's content will occupy in the
  /// final linked image.
  ///
  /// For a function atom, it is the number of bytes of code in the function.
  virtual uint64_t size() const = 0;

  /// \brief The visibility of this atom to other atoms.
  ///
  /// C static functions have scope scopeTranslationUnit.  Regular C functions
  /// have scope scopeGlobal.  Functions compiled with visibility=hidden have
  /// scope scopeLinkageUnit so they can be see by other atoms being linked but
  /// not by the OS loader.
  virtual Scope scope() const = 0;

  /// \brief Whether the linker should use direct or indirect access to this
  /// atom.
  virtual Interposable interposable() const = 0;

  /// \brief how the linker should handle if multiple atoms have the same name.
  virtual Merge merge() const = 0;

  /// \brief The type of this atom, such as code or data.
  virtual ContentType contentType() const = 0;

  /// \brief The alignment constraints on how this atom must be laid out in the
  /// final linked image (e.g. 16-byte aligned).
  virtual Alignment alignment() const = 0;

  /// \brief Whether this atom must be in a specially named section in the final
  /// linked image, or if the linker can infer the section based on the
  /// contentType().
  virtual SectionChoice sectionChoice() const = 0;

  /// \brief If sectionChoice() != sectionBasedOnContent, then this return the
  /// name of the section the atom should be placed into.
  virtual StringRef customSectionName() const = 0;

  /// \brief constraints on whether the linker may dead strip away this atom.
  virtual SectionPosition sectionPosition() const = 0;

  /// \brief constraints on whether the linker may dead strip away this atom.
  virtual DeadStripKind deadStrip() const = 0;

  /// \brief Returns the OS memory protections required for this atom's content
  /// at runtime.
  ///
  /// A function atom is R_X, a global variable is RW_, and a read-only constant
  /// is R__.
  virtual ContentPermissions permissions() const;

  /// \brief means this is a zero size atom that exists to provide an alternate
  /// name for another atom.  Alias atoms must have a special Reference to the
  /// atom they alias which the layout engine recognizes and forces the alias
  /// atom to layout right before the target atom.
  virtual bool isAlias() const = 0;

  /// \brief returns a reference to the raw (unrelocated) bytes of this Atom's
  /// content.
  virtual ArrayRef<uint8_t> rawContent() const = 0;

  /// This class abstracts iterating over the sequence of References
  /// in an Atom.  Concrete instances of DefinedAtom must implement
  /// the derefIterator() and incrementIterator() methods.
  class reference_iterator {
  public:
    reference_iterator(const DefinedAtom &a, const void *it)
      : _atom(a), _it(it) { }

    const Reference *operator*() const {
      return _atom.derefIterator(_it);
    }

    const Reference *operator->() const {
      return _atom.derefIterator(_it);
    }

    bool operator!=(const reference_iterator &other) const {
      return _it != other._it;
    }

    reference_iterator &operator++() {
      _atom.incrementIterator(_it);
      return *this;
    }
  private:
    const DefinedAtom &_atom;
    const void *_it;
  };

  /// \brief Returns an iterator to the beginning of this Atom's References.
  virtual reference_iterator begin() const = 0;

  /// \brief Returns an iterator to the end of this Atom's References.
  virtual reference_iterator end() const = 0;

  static inline bool classof(const Atom *a) {
    return a->definition() == definitionRegular;
  }

  /// Utility for deriving permissions from content type
  static ContentPermissions permissions(ContentType type);

protected:
  // DefinedAtom is an abstract base class. Only subclasses can access
  // constructor.
  DefinedAtom() : Atom(definitionRegular) { }

  // The memory for DefinedAtom objects is always managed by the owning File
  // object.  Therefore, no one but the owning File object should call delete on
  // an Atom.  In fact, some File objects may bulk allocate an array of Atoms,
  // so they cannot be individually deleted by anyone.
  virtual ~DefinedAtom() {}

  /// \brief Returns a pointer to the Reference object that the abstract
  /// iterator "points" to.
  virtual const Reference *derefIterator(const void *iter) const = 0;

  /// \brief Adjusts the abstract iterator to "point" to the next Reference
  /// object for this Atom.
  virtual void incrementIterator(const void *&iter) const = 0;
};
} // end namespace lld

#endif
