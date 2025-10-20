// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/access.h"

namespace Carbon::Check {

static auto CalculateEffectiveAccess(clang::AccessSpecifier lookup_access,
                                     clang::AccessSpecifier lexical_access)
    -> clang::AccessSpecifier {
  if (lookup_access != clang::AS_none) {
    // Lookup access takes precedence.
    return lookup_access;
  }

  if (lexical_access != clang::AS_none) {
    // When a base class private member is accessed through a derived class, the
    // lookup access would be set to `AS_none`.
    CARBON_CHECK(lexical_access == clang::AS_private);
    return lexical_access;
  }

  // No access specified means that this is not a record member, so we treat it
  // as public.
  return clang::AS_public;
}

auto ConvertCppAccess(clang::AccessSpecifier lookup_access,
                      clang::AccessSpecifier lexical_access)
    -> SemIR::AccessKind {
  switch (CalculateEffectiveAccess(lookup_access, lexical_access)) {
    case clang::AS_public:
      return SemIR::AccessKind::Public;
    case clang::AS_protected:
      return SemIR::AccessKind::Protected;
    case clang::AS_private:
      return SemIR::AccessKind::Private;
    case clang::AS_none:
      CARBON_FATAL("Couldn't deduce access");
  }
}

}  // namespace Carbon::Check
