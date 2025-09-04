// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_CUSTOM_TYPE_MAPPING_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_CUSTOM_TYPE_MAPPING_H_

#include "clang/AST/DeclCXX.h"

namespace Carbon::Check {

// Carbon types that have a custom mapping from C++.
enum class CustomCppTypeMapping : uint8_t {
  // None.
  None,

  // The Carbon `Str` type, which maps to `std::string_view`.
  Str,
};

// Determines whether record_decl is a C++ class that has a custom mapping into
// Carbon, and if so, returns the corresponding Carbon type. Otherwise returns
// None.
auto GetCustomCppTypeMapping(const clang::CXXRecordDecl* record_decl)
    -> CustomCppTypeMapping;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_CUSTOM_TYPE_MAPPING_H_
