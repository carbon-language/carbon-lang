// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/fuzzing/fuzzer_util.h"

#include <google/protobuf/text_format.h>

#include "common/check.h"
#include "common/error.h"
#include "common/fuzzing/proto_to_carbon.h"
#include "explorer/interpreter/exec_program.h"
#include "explorer/syntax/parse.h"
#include "explorer/syntax/prelude.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace Carbon {

// Appended to fuzzer-generated Carbon source when the source is missing
// `Main()` definition, to prevent early error return in semantic analysis.
static constexpr char EmptyMain[] = R"(
fn Main() -> i32 {
  return 0;
}
)";

auto Internal::GetRunfilesFile(const std::string& file)
    -> ErrorOr<std::string> {
  using bazel::tools::cpp::runfiles::Runfiles;
  std::string error;
  // `Runfiles::Create()` fails if passed an empty `argv0`.
  std::unique_ptr<Runfiles> runfiles(Runfiles::Create(
      /*argv0=*/llvm::sys::fs::getMainExecutable(nullptr, nullptr), &error));
  if (runfiles == nullptr) {
    return Error(error);
  }
  std::string full_path = runfiles->Rlocation(file);
  if (!llvm::sys::fs::exists(full_path)) {
    return ErrorBuilder() << full_path << " doesn't exist";
  }
  return full_path;
}

auto ParseCarbonTextProto(const std::string& contents, bool allow_unknown)
    -> ErrorOr<Fuzzing::Carbon> {
  google::protobuf::TextFormat::Parser parser;
  if (allow_unknown) {
    parser.AllowUnknownField(true);
    parser.AllowUnknownExtension(true);
  }
  Fuzzing::Carbon carbon_proto;
  if (!parser.ParseFromString(contents, &carbon_proto)) {
    return ErrorBuilder() << "Couldn't parse Carbon text proto";
  }
  return carbon_proto;
}

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

auto ParseAndExecute(const Fuzzing::CompilationUnit& compilation_unit)
    -> ErrorOr<int> {
  const std::string source = ProtoToCarbonWithMain(compilation_unit);

  Arena arena;
  CARBON_ASSIGN_OR_RETURN(AST ast,
                          ParseFromString(&arena, "Fuzzer.carbon", source,
                                          /*parser_debug=*/false));
  const ErrorOr<std::string> prelude_path =
      Internal::GetRunfilesFile("carbon/explorer/data/prelude.carbon");
  // Can't do anything without a prelude, so it's a fatal error.
  CARBON_CHECK(prelude_path.ok()) << prelude_path.error();

  AddPrelude(*prelude_path, &arena, &ast.declarations);
  CARBON_ASSIGN_OR_RETURN(
      ast, AnalyzeProgram(&arena, ast, /*trace_stream=*/std::nullopt));
  return ExecProgram(&arena, ast, /*trace_stream=*/std::nullopt);
}

}  // namespace Carbon
