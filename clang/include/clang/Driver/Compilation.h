//===--- Compilation.h - Compilation Task Data Structure --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_COMPILATION_H_
#define CLANG_DRIVER_COMPILATION_H_

namespace clang {

/// Compilation - A set of tasks to perform for a single driver
/// invocation.
class Compilation {
public:
  Compilation();
  ~Compilation();

  /// Execute - Execute the compilation jobs and return an
  /// appropriate exit code.
  int Execute() const;
};

} // end namespace clang

#endif
