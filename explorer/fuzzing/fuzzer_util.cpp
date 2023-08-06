// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/fuzzing/fuzzer_util.h"

#include <google/protobuf/text_format.h>

#include "common/check.h"
#include "common/error.h"
#include "explorer/parse_and_execute/parse_and_execute.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "testing/fuzzing/proto_to_carbon.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace Carbon::Testing {

auto GetRunfilesFile(const std::string& file) -> ErrorOr<std::string> {
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

auto ParseAndExecuteProto(const Fuzzing::Carbon& carbon) -> ErrorOr<int> {
  llvm::vfs::InMemoryFileSystem fs;

  const ErrorOr<std::string> prelude_path =
      GetRunfilesFile("carbon/explorer/data/prelude.carbon");
  // Can't do anything without a prelude, so it's a fatal error.
  CARBON_CHECK(prelude_path.ok()) << prelude_path.error();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> prelude =
      llvm::MemoryBuffer::getFile(*prelude_path);
  CARBON_CHECK(!prelude.getError()) << prelude.getError().message();
  CARBON_CHECK(fs.addFile("prelude.carbon", /*ModificationTime=*/0,
                          std::move(*prelude)));

  const std::string source = ProtoToCarbon(carbon, /*maybe_add_main=*/true);
  CARBON_CHECK(fs.addFile("fuzzer.carbon", /*ModificationTime=*/0,
                          llvm::MemoryBuffer::getMemBuffer(source)));

  TraceStream trace_stream;
  return ParseAndExecute(fs, "prelude.carbon", "fuzzer.carbon",
                         /*parser_debug=*/false, &trace_stream, &llvm::nulls());
}

}  // namespace Carbon::Testing
