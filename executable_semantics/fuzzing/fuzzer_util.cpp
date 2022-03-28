// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/fuzzing/fuzzer_util.h"

#include <google/protobuf/text_format.h>

#include "common/check.h"

namespace Carbon {

// Appended to fuzzer-generated AST proto when the proto is missing
// `Main()` definition, to prevent early error return in semantic analysis.
static constexpr char EmptyMainDeclaration[] = R"pb(
  function {
    name: "Main"
    param_pattern {}
    return_term {
      kind: Expression
      type { int_type_literal {} }
    }
    body {
      statements {
        return_statement { expression { int_literal { value: 0 } } }
      }
    }
  }
)pb";

auto MaybeAddMain(Fuzzing::CompilationUnit& compilation_unit) -> void {
  const bool has_main = std::any_of(
      compilation_unit.declarations().begin(),
      compilation_unit.declarations().end(),
      [](const Fuzzing::Declaration& decl) {
        return decl.kind_case() == Fuzzing::Declaration::kFunction &&
               decl.function().name() == "Main";
      });
  if (!has_main) {
    Fuzzing::Declaration main_decl;
    CHECK(google::protobuf::TextFormat::ParseFromString(EmptyMainDeclaration,
                                                        &main_decl))
        << "Failed to parse " << EmptyMainDeclaration;
    *compilation_unit.add_declarations() = main_decl;
  }
}

}  // namespace Carbon
