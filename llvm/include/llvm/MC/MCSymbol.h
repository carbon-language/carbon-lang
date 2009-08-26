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

#include <string>
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
  class MCSection;
  class MCContext;
  class raw_ostream;

  /// MCSymbol - Instances of this class represent a symbol name in the MC file,
  /// and MCSymbols are created and unique'd by the MCContext class.
  ///
  /// If the symbol is defined/emitted into the current translation unit, the
  /// Section member is set to indicate what section it lives in.  Otherwise, if
  /// it is a reference to an external entity, it has a null section.  
  /// 
  class MCSymbol {
    // Special sentinal value for the absolute pseudo section.
    //
    // FIXME: Use a PointerInt wrapper for this?
    static const MCSection *AbsolutePseudoSection;

    /// Name - The name of the symbol.
    std::string Name;

    /// Section - The section the symbol is defined in. This is null for
    /// undefined symbols, and the special AbsolutePseudoSection value for
    /// absolute symbols.
    const MCSection *Section;

    /// IsTemporary - True if this is an assembler temporary label, which
    /// typically does not survive in the .o file's symbol table.  Usually
    /// "Lfoo" or ".foo".
    unsigned IsTemporary : 1;

  private:  // MCContext creates and uniques these.
    friend class MCContext;
    MCSymbol(const StringRef &_Name, bool _IsTemporary) 
      : Name(_Name), Section(0), IsTemporary(_IsTemporary) {}
    
    MCSymbol(const MCSymbol&);       // DO NOT IMPLEMENT
    void operator=(const MCSymbol&); // DO NOT IMPLEMENT
  public:
    /// getName - Get the symbol name.
    const std::string &getName() const { return Name; }

    /// @name Symbol Type
    /// @{

    /// isDefined - Check if this symbol is defined (i.e., it has an address).
    ///
    /// Defined symbols are either absolute or in some section.
    bool isDefined() const {
      return Section != 0;
    }

    /// isUndefined - Check if this symbol undefined (i.e., implicitly defined).
    bool isUndefined() const {
      return !isDefined();
    }

    /// isAbsolute - Check if this this is an absolute symbol.
    bool isAbsolute() const {
      return Section == AbsolutePseudoSection;
    }

    /// getSection - Get the section associated with a defined, non-absolute
    /// symbol.
    const MCSection &getSection() const {
      assert(!isUndefined() && !isAbsolute() && "Invalid accessor!");
      return *Section;
    }

    /// setSection - Mark the symbol as defined in the section \arg S.
    void setSection(const MCSection &S) { Section = &S; }

    /// setUndefined - Mark the symbol as undefined.
    void setUndefined() {
      Section = 0;
    }

    /// setAbsolute - Mark the symbol as absolute.
    void setAbsolute() { Section = AbsolutePseudoSection; }

    /// @}

    /// print - Print the value to the stream \arg OS.
    void print(raw_ostream &OS) const;

    /// dump - Print the value to stderr.
    void dump() const;
  };

} // end namespace llvm

#endif
