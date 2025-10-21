// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/access.h"

namespace Carbon::Check {

static auto CalculateEffectiveAccess(clang::DeclAccessPair access_pair)
    -> clang::AccessSpecifier {
  // Note that we use `.getAccess()` here, not `->getAccess()`, which is
  // equivalent to `.getDecl()->getAccess()`, because we want to consider the
  // lookup access and not the lexical access.
  switch (access_pair.getAccess()) {
    // Lookup access takes precedence.
    case clang::AS_public:
    case clang::AS_protected:
    case clang::AS_private:
      return access_pair.getAccess();
    case clang::AS_none:
      // No access specified meaning depends on the declaration. For non class
      // members, it means there's no access associated with this function so we
      // treat it as public. For class members it means we lost access along the
      // inheritance path, and the difference between `none` and `private` only
      // matters when the access check is performed within a friend or member of
      // the naming class. Because the naming class is a C++ class, and we don't
      // yet have a mechanism for a C++ class to befriend a Carbon class, we can
      // safely map `none` to `private` for now.
      return access_pair->isCXXClassMember() ? clang::AS_private
                                             : clang::AS_public;
  }
}

auto MapCppAccess(clang::DeclAccessPair access_pair) -> SemIR::AccessKind {
  switch (CalculateEffectiveAccess(access_pair)) {
    case clang::AS_public:
      return SemIR::AccessKind::Public;
    case clang::AS_protected:
      return SemIR::AccessKind::Protected;
    case clang::AS_private:
      return SemIR::AccessKind::Private;
    case clang::AS_none:
      CARBON_FATAL("Couldn't convert access");
  }
}

}  // namespace Carbon::Check
