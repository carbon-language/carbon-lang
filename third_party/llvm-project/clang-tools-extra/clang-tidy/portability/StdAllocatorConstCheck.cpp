//===-- StdAllocatorConstCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StdAllocatorConstCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace portability {

void StdAllocatorConstCheck::registerMatchers(MatchFinder *Finder) {
  // Match std::allocator<const T>.
  auto allocatorConst =
      recordType(hasDeclaration(classTemplateSpecializationDecl(
          hasName("::std::allocator"),
          hasTemplateArgument(0, refersToType(qualType(isConstQualified()))))));

  auto hasContainerName =
      hasAnyName("::std::vector", "::std::deque", "::std::list",
                 "::std::multiset", "::std::set", "::std::unordered_multiset",
                 "::std::unordered_set", "::absl::flat_hash_set");

  // Match `std::vector<const T> var;` and other common containers like deque,
  // list, and absl::flat_hash_set. Containers like queue and stack use deque
  // but do not directly use std::allocator as a template argument, so they
  // aren't caught.
  Finder->addMatcher(
      typeLoc(
          templateSpecializationTypeLoc(),
          loc(hasUnqualifiedDesugaredType(anyOf(
              recordType(hasDeclaration(classTemplateSpecializationDecl(
                  hasContainerName,
                  anyOf(
                      hasTemplateArgument(1, refersToType(allocatorConst)),
                      hasTemplateArgument(2, refersToType(allocatorConst)),
                      hasTemplateArgument(3, refersToType(allocatorConst)))))),
              // Match std::vector<const dependent>
              templateSpecializationType(
                  templateArgumentCountIs(1),
                  hasTemplateArgument(
                      0, refersToType(qualType(isConstQualified()))),
                  hasDeclaration(namedDecl(hasContainerName)))))))
          .bind("type_loc"),
      this);
}

void StdAllocatorConstCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *T = Result.Nodes.getNodeAs<TypeLoc>("type_loc");
  if (!T)
    return;
  // Exclude TypeLoc matches in STL headers.
  if (isSystem(Result.Context->getSourceManager().getFileCharacteristic(
          T->getBeginLoc())))
    return;

  diag(T->getBeginLoc(),
       "container using std::allocator<const T> is a deprecated libc++ "
       "extension; remove const for compatibility with other standard "
       "libraries");
}

} // namespace portability
} // namespace tidy
} // namespace clang
