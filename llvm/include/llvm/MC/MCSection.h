//===- MCSection.h - Machine Code Sections ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTION_H
#define LLVM_MC_MCSECTION_H

#include <string>

namespace llvm {

  class MCSection {
    std::string Name;

  public:
    MCSection(const char *_Name) : Name(_Name) {}
  };

} // end namespace llvm

#endif
