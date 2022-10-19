// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/fuzzing/proto_to_carbon.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "explorer/fuzzing/ast_to_proto.h"
#include "explorer/syntax/parse.h"

namespace Carbon::Testing {
namespace {

static std::vector<llvm::StringRef>* carbon_files = nullptr;

// Returns a string representation of `ast`.
auto AstToString(const AST& ast) -> std::string {
  std::string s;
  llvm::raw_string_ostream out(s);
  out << "package " << ast.package.package << (ast.is_api ? "api" : "impl")
      << ";\n";
  for (auto* declaration : ast.declarations) {
    out << *declaration << "\n";
  }
  return s;
}

TEST(ProtoToCarbonTest, Roundtrip) {
  int parsed_ok_count = 0;
  for (const llvm::StringRef f : *carbon_files) {
    Carbon::Arena arena;
    const ErrorOr<AST> ast = Carbon::Parse(&arena, f, /*parser_debug=*/false);
    if (ast.ok()) {
      ++parsed_ok_count;
      const std::string source_from_proto = ProtoToCarbon(AstToProto(*ast));
      SCOPED_TRACE(testing::Message()
                   << "Carbon file: " << f << ", source from proto:\n"
                   << source_from_proto);
      const ErrorOr<AST> ast_from_proto = Carbon::ParseFromString(
          &arena, f, source_from_proto, /*parser_debug=*/false);

      if (ast_from_proto.ok()) {
        EXPECT_EQ(AstToString(*ast), AstToString(*ast_from_proto));
      } else {
        ADD_FAILURE() << "Parse error " << ast_from_proto.error().message();
      }
    }
  }
  // Makes sure files were actually processed.
  EXPECT_GT(parsed_ok_count, 0);
}

}  // namespace
}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  Carbon::Testing::carbon_files =
      new std::vector<llvm::StringRef>(&argv[1], &argv[argc]);
  return RUN_ALL_TESTS();
}
