//===- Core/Atom.h - The Fundimental Unit of Linking ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_ATOM_H_
#define LLD_CORE_ATOM_H_

#include "lld/Core/Reference.h"

namespace llvm {
  template <typename T>
  class ArrayRef;

  class StringRef;
}

namespace lld {

class File;

/// An atom is the fundamental unit of linking.  A C function or global variable
/// is an atom.  An atom has content and attributes. The content of a function
/// atom is the instructions that implement the function.  The content of a
/// global variable atom is its initial bytes.
class Atom {
public:
  enum Scope {
    scopeTranslationUnit,   // static, private to translation unit
    scopeLinkageUnit,       // hidden, accessible to just atoms being linked
    scopeGlobal             // default
  };

  enum Definition {
    definitionRegular,      // usual C/C++ function or global variable
    definitionTentative,    // C-only pre-ANSI support aka common
    definitionAbsolute,     // asm-only (foo = 10) not tied to any content
    definitionUndefined,    // Only in .o files to model reference to undef
    definitionSharedLibrary // Only in shared libraries to model export
  };

  enum Combine {
    combineNever,            // most symbols
    combineByName,           // weak-definition symbol
    combineByTypeContent,    // simple constant that can be coalesced
    combineByTypeContentDeep // complex coalescable constants
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
    typeZeroFill,           // zero-fill data
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
    typeFirstInSection,     // label for boundary of section [Darwin]
    typeLastInSection,      // label for boundary of section [Darwin]
  };

  enum ContentPermissions {
    perm___  = 0,           // mapped as unacessible
    permR__  = 8,           // mapped read-only
    permR_X  = 8 + 2,       // mapped readable and executable
    permRW_  = 8 + 4,       // mapped readable and writable
    permRW_L = 8 + 4 + 1,   // initially mapped r/w, then made read-only
                            // loader writable
  };

  enum SectionChoice {
    sectionBasedOnContent,  // linker infers final section based on content
    sectionCustomPreferred, // linker may place in specific section
    sectionCustomRequired   // linker must place in specific section
  };

  struct Alignment {
    Alignment(int p2, int m = 0)
      : powerOf2(p2)
      , modulus(m) {}

    uint16_t powerOf2;
    uint16_t modulus;
  };

  // MacOSX specific compact unwind info
  struct UnwindInfo {
    uint32_t startOffset;
    uint32_t unwindInfo;

    typedef UnwindInfo *iterator;
  };

  // link-once (throw away if not used)??
  // dll import/export

  Scope scope() const { return _scope; }
  Definition definition() const { return _definition; }
  Combine combine() const { return _combine; }
  ContentType contentType() const { return _contentType; }
  Alignment alignment() const;
  SectionChoice sectionChoice() const { return _sectionChoice; }
  bool deadStrip() const { return _DeadStrip; }
  bool isThumb() const { return _thumb; }
  bool isAlias() const { return _alias; }
  bool userVisibleName() const { return _userVisibleName; }
  bool autoHide() const;
  void setLive(bool l) { _live = l; }
  bool live() const { return _live; }
  void setOverridesDylibsWeakDef();

  virtual const class File *file() const = 0;
  virtual bool translationUnitSource(llvm::StringRef &path) const;
  virtual llvm::StringRef name() const;
  virtual uint64_t objectAddress() const = 0;
  virtual llvm::StringRef customSectionName() const;
  virtual uint64_t size() const = 0;
  virtual ContentPermissions permissions() const { return perm___; }
  virtual void copyRawContent(uint8_t buffer[]) const = 0;
  virtual llvm::ArrayRef<uint8_t> rawContent() const;
  virtual Reference::iterator referencesBegin() const;
  virtual Reference::iterator referencesEnd() const;
  virtual UnwindInfo::iterator beginUnwind() const;
  virtual UnwindInfo::iterator endUnwind() const;

  Atom( Definition d
      , Combine c
      , Scope s
      , ContentType ct
      , SectionChoice sc
      , bool UserVisibleName
      , bool DeadStrip
      , bool IsThumb
      , bool IsAlias
      , Alignment a)
    : _alignmentModulus(a.modulus)
    , _alignmentPowerOf2(a.powerOf2)
    , _definition(d)
    , _combine(c)
    , _userVisibleName(UserVisibleName)
    , _DeadStrip(DeadStrip)
    , _thumb(IsThumb)
    , _alias(IsAlias)
    , _contentType(ct)
    , _scope(s)
    , _sectionChoice(sc) {}

  virtual ~Atom();

protected:
  uint16_t _alignmentModulus;
  uint8_t _alignmentPowerOf2;
  Definition _definition : 3;
  Combine _combine : 2;
  bool _userVisibleName : 1;
  bool _DeadStrip : 1;
  bool _thumb : 1;
  bool _alias : 1;
  bool _live : 1;
  ContentType _contentType : 8;
  Scope _scope : 2;
  SectionChoice _sectionChoice: 2;
};

} // namespace lld

#endif // LLD_CORE_ATOM_H_
