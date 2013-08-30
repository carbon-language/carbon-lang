//===-- UseNullptr/UseNullptr.h - C++11 nullptr migration -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the definition of the UseNullptrTransform
/// class which is the main interface to the use-nullptr transform that tries to
/// make use of nullptr where possible.
///
//===----------------------------------------------------------------------===//

#ifndef CPP11_MIGRATE_USE_NULLPTR_H
#define CPP11_MIGRATE_USE_NULLPTR_H

#include "Core/Transform.h"
#include "llvm/Support/Compiler.h" // For LLVM_OVERRIDE

/// \brief Subclass of Transform that transforms null pointer constants into
/// C++11's nullptr keyword where possible.
class UseNullptrTransform : public Transform {
public:
  UseNullptrTransform(const TransformOptions &Options)
      : Transform("UseNullptr", Options) {}

  /// \see Transform::run().
  virtual int apply(FileOverrides &InputStates,
                    const clang::tooling::CompilationDatabase &Database,
                    const std::vector<std::string> &SourcePaths) LLVM_OVERRIDE;
};

#endif // CPP11_MIGRATE_USE_NULLPTR_H
