// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/fuzzing/fuzzer_util.h"

#include "common/check.h"
#include "common/fuzzing/proto_to_carbon.h"

namespace Carbon {

// Appended to fuzzer-generated Carbon source when the source is missing
// `Main()` definition, to prevent early error return in semantic analysis.
static constexpr char EmptyMain[] = R"(
fn Main() -> i32 {
  return 0;
}
)";

auto ProtoToCarbonWithMain(const Fuzzing::CompilationUnit& compilation_unit)
    -> std::string {
  const bool has_main = std::any_of(
      compilation_unit.declarations().begin(),
      compilation_unit.declarations().end(),
      [](const Fuzzing::Declaration& decl) {
        return decl.kind_case() == Fuzzing::Declaration::kFunction &&
               decl.function().name() == "Main";
      });
  return Carbon::ProtoToCarbon(compilation_unit) + (has_main ? "" : EmptyMain);
}

}  // namespace Carbon
