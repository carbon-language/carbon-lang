//===--- Driver.h - Clang GCC Compatible Driver -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_DRIVER_H_
#define CLANG_DRIVER_DRIVER_H_

namespace clang {
  class Compilation;

/// Driver - Encapsulate logic for constructing compilation processes
/// from a set of gcc-driver-like command line arguments.
class Driver {
public:
  Driver();
  ~Driver();

  /// BuildCompilation - Construct a compilation object for a command
  /// line argument vector.
  Compilation *BuildCompilation(int argc, const char **argv);
};

} // end namespace clang

#endif
