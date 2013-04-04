//===-- cpp11-migrate/Transforms.h - class Transforms Def'n -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the definition for class Transforms which is
/// responsible for defining the command-line arguments exposing
/// transformations to the user and applying requested transforms.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_TRANSFORMS_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_TRANSFORMS_H

#include "llvm/Support/CommandLine.h"
#include <vector>

// Forward declarations
namespace llvm {
namespace cl {
class Option;
} // namespace cl
} // namespace llvm
class Transform;

typedef Transform *(*TransformCreator)();

/// \brief Class encapsulating the creation of command line bool options
/// for each transform and instantiating transforms chosen by the user.
class Transforms {
public:
  typedef std::vector<Transform*> TransformVec;
  typedef TransformVec::const_iterator const_iterator;

public:

  ~Transforms();

  /// \brief Create command line options using LLVM's command line library.
  ///
  /// Be sure to call *before* parsing options.
  void createTransformOpts();

  /// \brief Instantiate all transforms that were selected on the command line.
  ///
  /// Call *after* parsing options.
  void createSelectedTransforms();

  /// \brief Return an iterator to the start of a container of instantiated
  /// transforms.
  const_iterator begin() const { return ChosenTransforms.begin(); }

  /// \brief Return an iterator to the end of a container of instantiated
  /// transforms.
  const_iterator end() const { return ChosenTransforms.end(); }

private:
  typedef std::vector<std::pair<llvm::cl::opt<bool>*, TransformCreator> >
    OptionVec;

private:
  TransformVec ChosenTransforms;
  OptionVec Options;
};

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_TRANSFORMS_H
