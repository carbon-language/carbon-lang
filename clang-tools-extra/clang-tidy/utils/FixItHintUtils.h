//===--- FixItHintUtils.h - clang-tidy---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_FIXITHINTUTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_FIXITHINTUTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Sema/DeclSpec.h"

namespace clang {
namespace tidy {
namespace utils {
namespace fixit {

/// Creates fix to make ``VarDecl`` a reference by adding ``&``.
FixItHint changeVarDeclToReference(const VarDecl &Var, ASTContext &Context);

/// This enum defines where the qualifier shall be preferably added.
enum class QualifierPolicy {
  Left,  // Add the qualifier always to the left side, if that is possible.
  Right, // Add the qualifier always to the right side.
};

/// This enum defines which entity is the target for adding the qualifier. This
/// makes only a difference for pointer-types. Other types behave identical
/// for either value of \c ConstTarget.
enum class QualifierTarget {
  Pointee, /// Transforming a pointer attaches to the pointee and not the
           /// pointer itself. For references and normal values this option has
           /// no effect. `int * p = &i;` -> `const int * p = &i` or `int const
           /// * p = &i`.
  Value,   /// Transforming pointers will consider the pointer itself.
           /// `int * p = &i;` -> `int * const = &i`
};

/// \brief Creates fix to qualify ``VarDecl`` with the specified \c Qualifier.
/// Requires that `Var` is isolated in written code like in `int foo = 42;`.
Optional<FixItHint>
addQualifierToVarDecl(const VarDecl &Var, const ASTContext &Context,
                      DeclSpec::TQ Qualifier,
                      QualifierTarget CT = QualifierTarget::Pointee,
                      QualifierPolicy CP = QualifierPolicy::Left);
} // namespace fixit
} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_FIXITHINTUTILS_H
