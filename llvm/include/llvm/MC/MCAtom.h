//===- MCAtom.h - Machine Code Atoms ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCATOM_H
#define LLVM_MC_MCATOM_H

namespace llvm {

  class MCAtom {
    MCSection *Section;

  public:
    MCAtom(MCSection *_Section) : Section(_Section) {}

    MCSection *getSection() { return Section; }
  };

} // end namespace llvm

#endif
