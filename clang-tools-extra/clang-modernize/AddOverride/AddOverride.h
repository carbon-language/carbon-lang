//===-- AddOverride/AddOverride.h - add C++11 override ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the definition of the AddOverrideTransform
/// class which is the main interface to the transform that tries to add the
/// override keyword to declarations of member function that override virtual
/// functions in a base class.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_MODERNIZE_ADD_OVERRIDE_H
#define CLANG_MODERNIZE_ADD_OVERRIDE_H

#include "Core/Transform.h"
#include "llvm/Support/Compiler.h"

class AddOverrideFixer;

/// \brief Subclass of Transform that adds the C++11 override keyword to
/// member functions overriding base class virtual functions.
class AddOverrideTransform : public Transform {
public:
  AddOverrideTransform(const TransformOptions &Options)
      : Transform("AddOverride", Options) {}

  /// \see Transform::run().
  virtual int apply(const clang::tooling::CompilationDatabase &Database,
                    const std::vector<std::string> &SourcePaths) override;

  virtual bool handleBeginSource(clang::CompilerInstance &CI,
                                 llvm::StringRef Filename) override;

private:
  AddOverrideFixer *Fixer;
};

#endif // CLANG_MODERNIZE_ADD_OVERRIDE_H
