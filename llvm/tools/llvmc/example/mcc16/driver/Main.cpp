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

#include "llvm/Config/config.h"
#include "llvm/CompilerDriver/BuiltinOptions.h"
#include "llvm/CompilerDriver/ForceLinkage.h"
#include "llvm/System/Path.h"
#include <iostream>

namespace llvmc {
  int Main(int argc, char** argv);
}

// Modify the PACKAGE_VERSION to use build number in top level configure file.
void PIC16VersionPrinter(void) {
  std::cout << "MPLAB C16 1.0 " << PACKAGE_VERSION << "\n";
}

int main(int argc, char** argv) {

  // HACK
  SaveTemps.setHiddenFlag(llvm::cl::Hidden);
  TempDirname.setHiddenFlag(llvm::cl::Hidden);
  Languages.setHiddenFlag(llvm::cl::Hidden);
  DryRun.setHiddenFlag(llvm::cl::Hidden);

  llvm::cl::SetVersionPrinter(PIC16VersionPrinter); 

  // Ask for a standard temp dir, but just cache its basename., and delete it.
  llvm::sys::Path tempDir;
  tempDir = llvm::sys::Path::GetTemporaryDirectory();
  TempDirname = tempDir.getBasename();
  tempDir.eraseFromDisk(true);

  // We are creating a temp dir in current dir, with the cached name.
  //  But before that remove if one already exists with that name..
  tempDir = TempDirname;
  tempDir.eraseFromDisk(true);

  llvmc::ForceLinkage();
  return llvmc::Main(argc, argv);
}
