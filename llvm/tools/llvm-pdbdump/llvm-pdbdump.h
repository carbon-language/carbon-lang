//===- llvm-pdbdump.h ----------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_LLVMPDBDUMP_H
#define LLVM_TOOLS_LLVMPDBDUMP_LLVMPDBDUMP_H

#include "llvm/Support/raw_ostream.h"

namespace llvm {
struct newline {
  newline(int IndentWidth) : Width(IndentWidth) {}
  int Width;
};

inline raw_ostream &operator<<(raw_ostream &OS, const newline &Indent) {
  OS << "\n";
  OS.indent(Indent.Width);
  return OS;
}
}

#endif