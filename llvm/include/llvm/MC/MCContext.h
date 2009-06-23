//===- MCContext.h - Machine Code Context -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCCONTEXT_H
#define LLVM_MC_MCCONTEXT_H

namespace llvm {
  class MCAtom;
  class MCImm;
  class MCSection;
  class MCSymbol;

  /// MCContext - Context object for machine code objects.
  class MCContext {
    MCContext(const MCContext&); // DO NOT IMPLEMENT
    MCContext &operator=(const MCContext&); // DO NOT IMPLEMENT

  public:
    MCContext();
    ~MCContext();

    MCSection *GetSection(const char *Name);
    MCAtom *CreateAtom(MCSection *Section);
    MCSymbol *CreateSymbol(MCAtom *Atom,
                           const char *Name,
                           bool IsTemporary);
    MCSymbol *LookupSymbol(const char *Name) const;

    void SetSymbolValue(MCSymbol *Sym, const MCImm &Value);
    const MCImm &GetSymbolValue(MCSymbol *Sym) const;
  };

} // end namespace llvm

#endif
