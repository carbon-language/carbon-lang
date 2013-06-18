//===-- LoopConvert/LoopConvert.h - C++11 for-loop migration ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the definition of the LoopConvertTransform
/// class which is the main interface to the loop-convert transform
/// that tries to make use of range-based for loops where possible.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_LOOP_CONVERT_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_LOOP_CONVERT_H

#include "Core/Transform.h"
#include "llvm/Support/Compiler.h" // For LLVM_OVERRIDE

/// \brief Subclass of Transform that transforms for-loops into range-based
/// for-loops where possible.
class LoopConvertTransform : public Transform {
public:
  LoopConvertTransform(const TransformOptions &Options)
      : Transform("LoopConvert", Options) {}

  /// \see Transform::run().
  virtual int apply(FileOverrides &InputStates,
                    const clang::tooling::CompilationDatabase &Database,
                    const std::vector<std::string> &SourcePaths) LLVM_OVERRIDE;
};

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_LOOP_CONVERT_H
