//===- MCSymbol.h - Machine Code Symbols ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSYMBOL_H
#define LLVM_MC_MCSYMBOL_H

#include <string>

namespace llvm {
  class MCAtom;

  class MCSymbol {
    MCAtom *Atom;
    std::string Name;
    unsigned IsTemporary : 1;

  public:
    MCSymbol(MCAtom *_Atom, const char *_Name, bool _IsTemporary) 
      : Atom(_Atom), Name(_Name), IsTemporary(_IsTemporary) {}
  };

} // end namespace llvm

#endif
