//===--- TypeTraits.cpp - clang-tidy---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TypeTraits.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"

namespace clang {
namespace tidy {
namespace type_traits {

namespace {
bool classHasTrivialCopyAndDestroy(QualType Type) {
  auto *Record = Type->getAsCXXRecordDecl();
  return Record && Record->hasDefinition() &&
         !Record->hasNonTrivialCopyConstructor() &&
         !Record->hasNonTrivialDestructor();
}
} // namespace

llvm::Optional<bool> isExpensiveToCopy(QualType Type, ASTContext &Context) {
  if (Type->isDependentType())
    return llvm::None;
  return !Type.isTriviallyCopyableType(Context) &&
         !classHasTrivialCopyAndDestroy(Type);
}

} // type_traits
} // namespace tidy
} // namespace clang
