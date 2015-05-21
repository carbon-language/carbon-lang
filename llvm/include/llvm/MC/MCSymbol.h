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

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class MCExpr;
class MCSymbol;
class MCFragment;
class MCSection;
class MCContext;
class raw_ostream;

// TODO: Merge completely with MCSymbol.
class MCSymbolData {
  /// Fragment - The fragment this symbol's value is relative to, if any. Also
  /// stores if this symbol is visible outside this translation unit (bit 0) or
  /// if it is private extern (bit 1).
  PointerIntPair<MCFragment *, 2> Fragment;

  union {
    /// Offset - The offset to apply to the fragment address to form this
    /// symbol's value.
    uint64_t Offset;

    /// CommonSize - The size of the symbol, if it is 'common'.
    uint64_t CommonSize;
  };

  /// SymbolSize - An expression describing how to calculate the size of
  /// a symbol. If a symbol has no size this field will be NULL.
  const MCExpr *SymbolSize = nullptr;

  /// CommonAlign - The alignment of the symbol, if it is 'common', or -1.
  //
  // FIXME: Pack this in with other fields?
  unsigned CommonAlign = -1U;

  /// Flags - The Flags field is used by object file implementations to store
  /// additional per symbol information which is not easily classified.
  uint32_t Flags = 0;

  /// Index - Index field, for use by the object file implementation.
  uint64_t Index = 0;

public:
  MCSymbolData() { Offset = 0; }

  MCFragment *getFragment() const { return Fragment.getPointer(); }
  void setFragment(MCFragment *Value) { Fragment.setPointer(Value); }

  uint64_t getOffset() const {
    assert(!isCommon());
    return Offset;
  }
  void setOffset(uint64_t Value) {
    assert(!isCommon());
    Offset = Value;
  }

  /// @}
  /// \name Symbol Attributes
  /// @{

  bool isExternal() const { return Fragment.getInt() & 1; }
  void setExternal(bool Value) {
    Fragment.setInt((Fragment.getInt() & ~1) | unsigned(Value));
  }

  bool isPrivateExtern() const { return Fragment.getInt() & 2; }
  void setPrivateExtern(bool Value) {
    Fragment.setInt((Fragment.getInt() & ~2) | (unsigned(Value) << 1));
  }

  /// isCommon - Is this a 'common' symbol.
  bool isCommon() const { return CommonAlign != -1U; }

  /// setCommon - Mark this symbol as being 'common'.
  ///
  /// \param Size - The size of the symbol.
  /// \param Align - The alignment of the symbol.
  void setCommon(uint64_t Size, unsigned Align) {
    assert(getOffset() == 0);
    CommonSize = Size;
    CommonAlign = Align;
  }

  /// getCommonSize - Return the size of a 'common' symbol.
  uint64_t getCommonSize() const {
    assert(isCommon() && "Not a 'common' symbol!");
    return CommonSize;
  }

  void setSize(const MCExpr *SS) { SymbolSize = SS; }

  const MCExpr *getSize() const { return SymbolSize; }

  /// getCommonAlignment - Return the alignment of a 'common' symbol.
  unsigned getCommonAlignment() const {
    assert(isCommon() && "Not a 'common' symbol!");
    return CommonAlign;
  }

  /// getFlags - Get the (implementation defined) symbol flags.
  uint32_t getFlags() const { return Flags; }

  /// setFlags - Set the (implementation defined) symbol flags.
  void setFlags(uint32_t Value) { Flags = Value; }

  /// modifyFlags - Modify the flags via a mask
  void modifyFlags(uint32_t Value, uint32_t Mask) {
    Flags = (Flags & ~Mask) | Value;
  }

  /// getIndex - Get the (implementation defined) index.
  uint64_t getIndex() const { return Index; }

  /// setIndex - Set the (implementation defined) index.
  void setIndex(uint64_t Value) { Index = Value; }

  /// @}

  void dump() const;
};

/// MCSymbol - Instances of this class represent a symbol name in the MC file,
/// and MCSymbols are created and uniqued by the MCContext class.  MCSymbols
/// should only be constructed with valid names for the object file.
///
/// If the symbol is defined/emitted into the current translation unit, the
/// Section member is set to indicate what section it lives in.  Otherwise, if
/// it is a reference to an external entity, it has a null section.
class MCSymbol {
  // Special sentinal value for the absolute pseudo section.
  //
  // FIXME: Use a PointerInt wrapper for this?
  static const MCSection *AbsolutePseudoSection;

  /// Name - The name of the symbol.  The referred-to string data is actually
  /// held by the StringMap that lives in MCContext.
  StringRef Name;

  /// Section - The section the symbol is defined in. This is null for
  /// undefined symbols, and the special AbsolutePseudoSection value for
  /// absolute symbols. If this is a variable symbol, this caches the
  /// variable value's section.
  mutable const MCSection *Section;

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

  mutable bool HasData : 1;
  mutable MCSymbolData Data;

private: // MCContext creates and uniques these.
  friend class MCExpr;
  friend class MCContext;
  MCSymbol(StringRef name, bool isTemporary)
      : Name(name), Section(nullptr), Value(nullptr), IsTemporary(isTemporary),
        IsRedefinable(false), IsUsed(false), HasData(false) {}

  MCSymbol(const MCSymbol &) = delete;
  void operator=(const MCSymbol &) = delete;
  const MCSection *getSectionPtr() const {
    if (Section || !Value)
      return Section;
    return Section = Value->FindAssociatedSection();
  }

public:
  /// getName - Get the symbol name.
  StringRef getName() const { return Name; }

  bool hasData() const { return HasData; }

  /// Get associated symbol data.
  MCSymbolData &getData() const {
    assert(HasData && "Missing symbol data!");
    return Data;
  }

  /// Initialize symbol data.
  ///
  /// Nothing really to do here, but this is enables an assertion that \a
  /// MCAssembler::getOrCreateSymbolData() has actually been called before
  /// anyone calls \a getData().
  void initializeData() const { HasData = true; }

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
      Section = nullptr;
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

  /// getSection - Get the section associated with a defined, non-absolute
  /// symbol.
  const MCSection &getSection() const {
    assert(isInSection() && "Invalid accessor!");
    return *getSectionPtr();
  }

  /// setSection - Mark the symbol as defined in the section \p S.
  void setSection(const MCSection &S) {
    assert(!isVariable() && "Cannot set section of variable");
    Section = &S;
  }

  /// setUndefined - Mark the symbol as undefined.
  void setUndefined() { Section = nullptr; }

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

  /// print - Print the value to the stream \p OS.
  void print(raw_ostream &OS) const;

  /// dump - Print the value to stderr.
  void dump() const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const MCSymbol &Sym) {
  Sym.print(OS);
  return OS;
}
} // end namespace llvm

#endif
