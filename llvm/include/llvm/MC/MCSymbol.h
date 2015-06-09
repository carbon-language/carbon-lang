//===- MCSymbol.h - Machine Code Symbols ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCSymbol class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSYMBOL_H
#define LLVM_MC_MCSYMBOL_H

#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class MCAsmInfo;
class MCExpr;
class MCSymbol;
class MCFragment;
class MCSection;
class MCContext;
class raw_ostream;

/// MCSymbol - Instances of this class represent a symbol name in the MC file,
/// and MCSymbols are created and uniqued by the MCContext class.  MCSymbols
/// should only be constructed with valid names for the object file.
///
/// If the symbol is defined/emitted into the current translation unit, the
/// Section member is set to indicate what section it lives in.  Otherwise, if
/// it is a reference to an external entity, it has a null section.
class MCSymbol {
protected:
  /// The kind of the symbol.  If it is any value other than unset then this
  /// class is actually one of the appropriate subclasses of MCSymbol.
  enum SymbolKind {
    SymbolKindUnset,
    SymbolKindCOFF,
    SymbolKindELF,
    SymbolKindMachO,
  };

  // Special sentinal value for the absolute pseudo section.
  //
  // FIXME: Use a PointerInt wrapper for this?
  static MCSection *AbsolutePseudoSection;

  /// Name - The name of the symbol.  The referred-to string data is actually
  /// held by the StringMap that lives in MCContext.
  const StringMapEntry<bool> *Name;

  /// If a symbol has a Fragment, the section is implied, so we only need
  /// one pointer.
  /// FIXME: We might be able to simplify this by having the asm streamer create
  /// dummy fragments.
  /// If this is a section, then it gives the symbol is defined in. This is null
  /// for undefined symbols, and the special AbsolutePseudoSection value for
  /// absolute symbols. If this is a variable symbol, this caches the variable
  /// value's section.
  ///
  /// If this is a fragment, then it gives the fragment this symbol's value is
  /// relative to, if any.
  mutable PointerUnion<MCSection *, MCFragment *> SectionOrFragment;

  /// Value - If non-null, the value for a variable symbol.
  const MCExpr *Value;

  /// IsTemporary - True if this is an assembler temporary label, which
  /// typically does not survive in the .o file's symbol table.  Usually
  /// "Lfoo" or ".foo".
  unsigned IsTemporary : 1;

  /// \brief True if this symbol can be redefined.
  unsigned IsRedefinable : 1;

  /// IsUsed - True if this symbol has been used.
  mutable unsigned IsUsed : 1;

  mutable bool IsRegistered : 1;

  /// This symbol is visible outside this translation unit.
  mutable unsigned IsExternal : 1;

  /// This symbol is private extern.
  mutable unsigned IsPrivateExtern : 1;

  /// LLVM RTTI discriminator. This is actually a SymbolKind enumerator, but is
  /// unsigned to avoid sign extension and achieve better bitpacking with MSVC.
  unsigned Kind : 2;

  /// Index field, for use by the object file implementation.
  mutable uint32_t Index = 0;

  union {
    /// The offset to apply to the fragment address to form this symbol's value.
    uint64_t Offset;

    /// The size of the symbol, if it is 'common'.
    uint64_t CommonSize;
  };

  /// The alignment of the symbol, if it is 'common', or -1.
  //
  // FIXME: Pack this in with other fields?
  unsigned CommonAlign = -1U;

  /// The Flags field is used by object file implementations to store
  /// additional per symbol information which is not easily classified.
  mutable uint32_t Flags = 0;

protected: // MCContext creates and uniques these.
  friend class MCExpr;
  friend class MCContext;
  MCSymbol(SymbolKind Kind, const StringMapEntry<bool> *Name, bool isTemporary)
      : Name(Name), Value(nullptr), IsTemporary(isTemporary),
        IsRedefinable(false), IsUsed(false), IsRegistered(false),
        IsExternal(false), IsPrivateExtern(false),
        Kind(Kind) {
    Offset = 0;
  }

private:
  MCSymbol(const MCSymbol &) = delete;
  void operator=(const MCSymbol &) = delete;
  MCSection *getSectionPtr() const {
    if (MCFragment *F = getFragment())
      return F->getParent();
    assert(!SectionOrFragment.is<MCFragment *>() && "Section or null expected");
    MCSection *Section = SectionOrFragment.dyn_cast<MCSection *>();
    if (Section || !Value)
      return Section;
    return Section = Value->findAssociatedSection();
  }

public:
  /// getName - Get the symbol name.
  StringRef getName() const { return Name ? Name->first() : ""; }

  bool isRegistered() const { return IsRegistered; }
  void setIsRegistered(bool Value) const { IsRegistered = Value; }

  /// \name Accessors
  /// @{

  /// isTemporary - Check if this is an assembler temporary symbol.
  bool isTemporary() const { return IsTemporary; }

  /// isUsed - Check if this is used.
  bool isUsed() const { return IsUsed; }
  void setUsed(bool Value) const { IsUsed = Value; }

  /// \brief Check if this symbol is redefinable.
  bool isRedefinable() const { return IsRedefinable; }
  /// \brief Mark this symbol as redefinable.
  void setRedefinable(bool Value) { IsRedefinable = Value; }
  /// \brief Prepare this symbol to be redefined.
  void redefineIfPossible() {
    if (IsRedefinable) {
      Value = nullptr;
      SectionOrFragment = nullptr;
      IsRedefinable = false;
    }
  }

  /// @}
  /// \name Associated Sections
  /// @{

  /// isDefined - Check if this symbol is defined (i.e., it has an address).
  ///
  /// Defined symbols are either absolute or in some section.
  bool isDefined() const { return getSectionPtr() != nullptr; }

  /// isInSection - Check if this symbol is defined in some section (i.e., it
  /// is defined but not absolute).
  bool isInSection() const { return isDefined() && !isAbsolute(); }

  /// isUndefined - Check if this symbol undefined (i.e., implicitly defined).
  bool isUndefined() const { return !isDefined(); }

  /// isAbsolute - Check if this is an absolute symbol.
  bool isAbsolute() const { return getSectionPtr() == AbsolutePseudoSection; }

  /// Get the section associated with a defined, non-absolute symbol.
  MCSection &getSection() const {
    assert(isInSection() && "Invalid accessor!");
    return *getSectionPtr();
  }

  /// Mark the symbol as defined in the section \p S.
  void setSection(MCSection &S) {
    assert(!isVariable() && "Cannot set section of variable");
    assert(!SectionOrFragment.is<MCFragment *>() && "Section or null expected");
    SectionOrFragment = &S;
  }

  /// Mark the symbol as undefined.
  void setUndefined() {
    SectionOrFragment = nullptr;
  }

  bool isELF() const { return Kind == SymbolKindELF; }

  bool isCOFF() const { return Kind == SymbolKindCOFF; }

  bool isMachO() const { return Kind == SymbolKindMachO; }

  /// @}
  /// \name Variable Symbols
  /// @{

  /// isVariable - Check if this is a variable symbol.
  bool isVariable() const { return Value != nullptr; }

  /// getVariableValue() - Get the value for variable symbols.
  const MCExpr *getVariableValue() const {
    assert(isVariable() && "Invalid accessor!");
    IsUsed = true;
    return Value;
  }

  void setVariableValue(const MCExpr *Value);

  /// @}

  /// Get the (implementation defined) index.
  uint32_t getIndex() const {
    return Index;
  }

  /// Set the (implementation defined) index.
  void setIndex(uint32_t Value) const {
    Index = Value;
  }

  uint64_t getOffset() const {
    assert(!isCommon());
    return Offset;
  }
  void setOffset(uint64_t Value) {
    assert(!isCommon());
    Offset = Value;
  }

  /// Return the size of a 'common' symbol.
  uint64_t getCommonSize() const {
    assert(isCommon() && "Not a 'common' symbol!");
    return CommonSize;
  }

  /// Mark this symbol as being 'common'.
  ///
  /// \param Size - The size of the symbol.
  /// \param Align - The alignment of the symbol.
  void setCommon(uint64_t Size, unsigned Align) {
    assert(getOffset() == 0);
    CommonSize = Size;
    CommonAlign = Align;
  }

  ///  Return the alignment of a 'common' symbol.
  unsigned getCommonAlignment() const {
    assert(isCommon() && "Not a 'common' symbol!");
    return CommonAlign;
  }

  /// Declare this symbol as being 'common'.
  ///
  /// \param Size - The size of the symbol.
  /// \param Align - The alignment of the symbol.
  /// \return True if symbol was already declared as a different type
  bool declareCommon(uint64_t Size, unsigned Align) {
    assert(isCommon() || getOffset() == 0);
    if(isCommon()) {
      if(CommonSize != Size || CommonAlign != Align)
       return true;
    } else
      setCommon(Size, Align);
    return false;
  }

  /// Is this a 'common' symbol.
  bool isCommon() const { return CommonAlign != -1U; }

  MCFragment *getFragment() const {
    return SectionOrFragment.dyn_cast<MCFragment *>();
  }
  void setFragment(MCFragment *Value) const {
    SectionOrFragment = Value;
  }

  bool isExternal() const { return IsExternal; }
  void setExternal(bool Value) const { IsExternal = Value; }

  bool isPrivateExtern() const { return IsPrivateExtern; }
  void setPrivateExtern(bool Value) { IsPrivateExtern = Value; }

  /// print - Print the value to the stream \p OS.
  void print(raw_ostream &OS, const MCAsmInfo *MAI) const;

  /// dump - Print the value to stderr.
  void dump() const;

protected:
  /// Get the (implementation defined) symbol flags.
  uint32_t getFlags() const { return Flags; }

  /// Set the (implementation defined) symbol flags.
  void setFlags(uint32_t Value) const { Flags = Value; }

  /// Modify the flags via a mask
  void modifyFlags(uint32_t Value, uint32_t Mask) const {
    Flags = (Flags & ~Mask) | Value;
  }
};

inline raw_ostream &operator<<(raw_ostream &OS, const MCSymbol &Sym) {
  Sym.print(OS, nullptr);
  return OS;
}
} // end namespace llvm

#endif
