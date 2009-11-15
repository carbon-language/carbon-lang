//===--- TargetOptions.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_TARGETOPTIONS_H
#define LLVM_CLANG_FRONTEND_TARGETOPTIONS_H

#include <string>
#include <vector>

namespace clang {

/// TargetOptions - Options for controlling the target.
class TargetOptions {
public:

  /// If given, the name of the target triple to compile for. If not given the
  /// target will be selected to match the host.
  std::string Triple;

  /// If given, the name of the target CPU to generate code for.
  std::string CPU;

  /// If given, the name of the target ABI to use.
  std::string ABI;

  /// The list of target specific features to enable or disable -- this should
  /// be a list of strings starting with by '+' or '-'.
  std::vector<std::string> Features;
};

}  // end namespace clang

#endif
