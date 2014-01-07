//===-- Core/Transforms.h - class Transforms Def'n --------------*- C++ -*-===//
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

#ifndef CLANG_MODERNIZE_TRANSFORMS_H
#define CLANG_MODERNIZE_TRANSFORMS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include <vector>

// Forward declarations
namespace llvm {
namespace cl {
class Option;
} // namespace cl
} // namespace llvm
class Transform;
struct TransformOptions;
struct CompilerVersions;

typedef Transform *(*TransformCreator)(const TransformOptions &);
template <typename T>
Transform *ConstructTransform(const TransformOptions &Options) {
  return new T(Options);
}

/// \brief To group transforms together when printing the help.
extern llvm::cl::OptionCategory TransformCategory;

/// \brief Class encapsulating the creation of command line bool options
/// for each transform and instantiating transforms chosen by the user.
class Transforms {
public:
  typedef std::vector<Transform*> TransformVec;
  typedef TransformVec::const_iterator const_iterator;

public:

  ~Transforms();

  /// \brief Registers all available transforms causing them to be made
  /// available on the command line.
  ///
  /// Be sure to register all transforms *before* parsing command line options.
  void registerTransforms();

  /// \brief Instantiate all transforms that were selected on the command line.
  ///
  /// Call *after* parsing options.
  void createSelectedTransforms(const TransformOptions &Options,
                                const CompilerVersions &RequiredVersions);

  /// \brief Return an iterator to the start of a container of instantiated
  /// transforms.
  const_iterator begin() const { return ChosenTransforms.begin(); }

  /// \brief Return an iterator to the end of a container of instantiated
  /// transforms.
  const_iterator end() const { return ChosenTransforms.end(); }

private:
  bool hasAnyExplicitOption() const;

  typedef llvm::StringMap<llvm::cl::opt<bool> *> OptionMap;

private:
  TransformVec ChosenTransforms;
  OptionMap Options;
};

#endif // CLANG_MODERNIZE_TRANSFORMS_H
