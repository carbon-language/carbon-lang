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
    definitionWeak,         // can be silently overridden by regular definition
    definitionTentative,    // C-only pre-ANSI support aka common
    definitionAbsolute,     // asm-only (foo = 10) not tied to any content
    definitionUndefined,    // Only in .o files to model reference to undef
    definitionSharedLibrary // Only in shared libraries to model export
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
  };

  // MacOSX specific compact unwind info
  struct UnwindInfo {
    uint32_t startOffset;
    uint32_t unwindInfo;

    typedef UnwindInfo *iterator;
  };

  /// name - The name of the atom. For a function atom, it is the (mangled)
  /// name of the function. 
  virtual llvm::StringRef name() const = 0;
  
  /// internalName - If the name is just a temporary label that should
  /// not show up in the final linked image.
  bool internalName() const { 
    return _internalName; 
  }

  /// size - the number of bytes of space this atom's content will occupy
  /// in the final linked image.  For a function atom, it is the number
  /// of bytes of code in the function.
  virtual uint64_t size() const = 0;

  /// scope - The visibility of this atom to other atoms.  C static functions
  /// have scope scopeTranslationUnit.  Regular C functions have scope 
  /// scopeGlobal.  Functions compiled with visibility=hidden have scope
  /// scopeLinkageUnit so they can be see by other atoms being linked but not
  /// by the OS loader.
  Scope scope() const { 
    return _scope; 
  }
  
  /// definition - Whether this atom is a definition or represents an undefined
  /// or tentative symbol.
  Definition definition() const { 
    return _definition; 
  }
  
  /// mergeDuplicates - For definitionRegular atoms, this means the
  /// atom can be silently coalesced with another atom that has the 
  /// same name or content.
  bool mergeDuplicates() const { 
    return _mergeDuplicates; 
  }
  
  /// contentType - The type of this atom, such as code or data.
  ContentType contentType() const { 
    return _contentType; 
  }
  
  /// alignment - The alignment constraints on how this atom must be laid out 
  /// in the final linked image (e.g. 16-byte aligned).
  Alignment alignment() const {
    return Alignment(_alignmentPowerOf2, _alignmentModulus);
  }

  /// sectionChoice - Whether this atom must be in a specially named section
  /// in the final linked image, or if the linker can infer the section 
  /// based on the contentType().
  SectionChoice sectionChoice() const { 
    return _sectionChoice; 
  }
  
  /// customSectionName - If sectionChoice() != sectionBasedOnContent, then
  /// this return the name of the section the atom should be placed into.
  virtual llvm::StringRef customSectionName() const;
    
  /// deadStrip - constraints on whether the linker may dead strip away 
  /// this atom.
  DeadStripKind deadStrip() const { 
    return _deadStrip; 
  }
  
  /// autoHide - Whether it is ok for the linker to change the scope of this 
  /// atom to hidden as long as all other duplicates are also autoHide.
  bool autoHide() const {
    return _autoHide;
  }

  /// permissions - Returns the OS memory protections required for this atom's
  /// content at runtime.  A function atom is R_X and a global variable is RW_.
  virtual ContentPermissions permissions() const;
    
  /// 
  virtual void copyRawContent(uint8_t buffer[]) const = 0;
  virtual llvm::ArrayRef<uint8_t> rawContent() const;


  bool isThumb() const { return _thumb; }
  bool isAlias() const { return _alias; }
  
  void setLive(bool l) { _live = l; }
  bool live() const { return _live; }

  virtual const class File *file() const = 0;
  virtual bool translationUnitSource(llvm::StringRef &path) const;
  virtual uint64_t objectAddress() const = 0;
  virtual Reference::iterator referencesBegin() const;
  virtual Reference::iterator referencesEnd() const;
  virtual UnwindInfo::iterator beginUnwind() const;
  virtual UnwindInfo::iterator endUnwind() const;

  Atom( Definition d
      , Scope s
      , ContentType ct
      , SectionChoice sc
      , bool internalName
      , DeadStripKind ds
      , bool IsThumb
      , bool IsAlias
      , Alignment a)
    : _alignmentModulus(a.modulus)
    , _alignmentPowerOf2(a.powerOf2)
    , _definition(d)
    , _internalName(internalName)
    , _deadStrip(ds)
    , _thumb(IsThumb)
    , _alias(IsAlias)
    , _contentType(ct)
    , _scope(s)
    , _sectionChoice(sc) {}

  virtual ~Atom();

protected:
  uint16_t      _alignmentModulus;
  uint8_t       _alignmentPowerOf2;
  ContentType   _contentType : 8;
  Definition    _definition : 3;
  Scope         _scope : 2;
  SectionChoice _sectionChoice: 2;
  bool          _internalName : 1;
  DeadStripKind _deadStrip : 2;
  bool          _mergeDuplicates : 1;
  bool          _thumb : 1;
  bool          _autoHide : 1;
  bool          _alias : 1;
  bool          _live : 1;
};

} // namespace lld

#endif // LLD_CORE_ATOM_H_
