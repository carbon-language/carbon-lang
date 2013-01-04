//===-- cpp11-migrate/Transform.h - Transform Base Class Def'n --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the definition for the base Transform class from
/// which all transforms must subclass.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_TRANSFORM_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_TRANSFORM_H

#include "clang/Tooling/CompilationDatabase.h"
#include <vector>

/// \brief Description of the riskiness of actions that can be taken by
/// transforms.
enum RiskLevel {
  /// Transformations that will not change semantics.
  RL_Safe,

  /// Transformations that might change semantics.
  RL_Reasonable,

  /// Transformations that are likely to change semantics.
  RL_Risky
};

// Forward Declarations
namespace clang {
namespace tooling {
class CompilationDatabase;
} // namespace tooling
} // namespace clang

/// \brief Abstract base class for all C++11 migration transforms.
class Transform {
public:
  Transform() {
    Reset();
  }

  virtual ~Transform() {}

  /// \brief Apply a transform to all files listed in \p SourcePaths.
  ///
  /// \p Database must contain information for how to compile all files in
  /// \p SourcePaths.
  virtual int apply(RiskLevel MaxRiskLevel,
                    const clang::tooling::CompilationDatabase &Database,
                    const std::vector<std::string> &SourcePaths) = 0;

  /// \brief Query if changes were made during the last call to apply().
  bool getChangesMade() const { return ChangesMade; }

  /// \brief Query if changes were not made due to conflicts with other changes
  /// made during the last call to apply() or if changes were too risky for the
  /// requested risk level.
  bool getChangesNotMade() const { return ChangesNotMade; }

  /// \brief Reset internal state of the transform.
  ///
  /// Useful if calling apply() several times with one instantiation of a
  /// transform.
  void Reset() {
    ChangesMade = false;
    ChangesNotMade = false;
  }

protected:
  void setChangesMade() { ChangesMade = true; }
  void setChangesNotMade() { ChangesNotMade = true; }

private:
  bool ChangesMade;
  bool ChangesNotMade;
};

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_TRANSFORM_H
