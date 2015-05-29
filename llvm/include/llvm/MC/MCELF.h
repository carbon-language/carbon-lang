//===- lib/MC/MCELF.h - ELF MC --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains some support functions used by the ELF Streamer and
// ObjectWriter.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCELF_H
#define LLVM_MC_MCELF_H

namespace llvm {
class MCSymbol;

class MCELF {
 public:
   static void SetBinding(const MCSymbol &Sym, unsigned Binding);
   static unsigned GetBinding(const MCSymbol &Sym);
   static void SetType(const MCSymbol &Sym, unsigned Type);
   static unsigned GetType(const MCSymbol &Sym);
   static void SetVisibility(MCSymbol &Sym, unsigned Visibility);
   static unsigned GetVisibility(const MCSymbol &Sym);
   static void setOther(MCSymbol &Sym, unsigned Other);
   static unsigned getOther(const MCSymbol &Sym);
};

}

#endif
