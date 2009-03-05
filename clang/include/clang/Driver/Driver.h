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
namespace driver {
  class ArgList;
  class Compilation;
  class OptTable;

/// Driver - Encapsulate logic for constructing compilation processes
/// from a set of gcc-driver-like command line arguments.
class Driver {
  OptTable *Opts;

  /// ParseArgStrings - Parse the given list of strings into an
  /// ArgList.
  ArgList *ParseArgStrings(const char **ArgBegin, const char **ArgEnd);

public:
  Driver();
  ~Driver();

  const OptTable &getOpts() const { return *Opts; }

  /// BuildCompilation - Construct a compilation object for a command
  /// line argument vector.
  Compilation *BuildCompilation(int argc, const char **argv);
};

} // end namespace driver
} // end namespace clang

#endif
