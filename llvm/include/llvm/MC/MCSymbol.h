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
  class MCSymbol {
    MCSection *Section;
    std::string Name;
    unsigned IsTemporary : 1;
    unsigned IsExternal : 1;

  public:
    MCSymbol(const char *_Name, bool _IsTemporary) 
      : Section(0), Name(_Name), IsTemporary(_IsTemporary), IsExternal(false) {}

    MCSection *getSection() const { return Section; }
    void setSection(MCSection *Value) { Section = Value; }

    bool isExternal() const { return IsExternal; }
    void setExternal(bool Value) { IsExternal = Value; }

    const std::string &getName() const { return Name; }
  };

} // end namespace llvm

#endif
