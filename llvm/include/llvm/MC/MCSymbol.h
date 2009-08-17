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
    /// Name - The name of the symbol.
    std::string Name;
    /// Section - The section the symbol is defined in, or null if the symbol
    /// has not been defined in the associated translation unit.
    const MCSection *Section;
    
    /// IsTemporary - True if this is an assembler temporary label, which
    /// typically does not survive in the .o file's symbol table.  Usually
    /// "Lfoo" or ".foo".
    unsigned IsTemporary : 1;
    
    /// IsExternal - True if this symbol has been implicitly defined as an
    /// external, for example by using it in an expression without ever emitting
    /// it as a label. The @var Section for an external symbol is always null.
    unsigned IsExternal : 1;

  private:  // MCContext creates and uniques these.
    friend class MCContext;
    MCSymbol(const StringRef &_Name, bool _IsTemporary) 
      : Name(_Name), Section(0), IsTemporary(_IsTemporary), IsExternal(false) {}
    
    MCSymbol(const MCSymbol&);       // DO NOT IMPLEMENT
    void operator=(const MCSymbol&); // DO NOT IMPLEMENT
  public:
    
    const MCSection *getSection() const { return Section; }
    void setSection(const MCSection *S) { Section = S; }

    bool isExternal() const { return IsExternal; }
    void setExternal(bool Value) { IsExternal = Value; }

    const std::string &getName() const { return Name; }

    /// print - Print the value to the stream \arg OS.
    void print(raw_ostream &OS) const;

    /// dump - Print the value to stderr.
    void dump() const;
  };

} // end namespace llvm

#endif
