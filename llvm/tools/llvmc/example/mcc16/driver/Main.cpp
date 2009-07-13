//===--- Main.cpp - The LLVM Compiler Driver -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Usually this file just includes CompilerDriver/Main.inc, but here we apply
//  some trickery to make the built-in '-save-temps' option hidden and enable
//  '--temp-dir' by default.
//
//===----------------------------------------------------------------------===//

#include "llvm/CompilerDriver/BuiltinOptions.h"
#include "llvm/CompilerDriver/ForceLinkage.h"
#include "llvm/System/Path.h"

namespace llvmc {
  int Main(int argc, char** argv);
}

int main(int argc, char** argv) {

  // HACK
  SaveTemps.setHiddenFlag(llvm::cl::Hidden);
  TempDirname = "tmp-objs";

  // Remove the temp dir if already exists.
  llvm::sys::Path tempDir;
  tempDir = TempDirname;
  tempDir.eraseFromDisk(true);

  llvmc::ForceLinkage();
  return llvmc::Main(argc, argv);
}
