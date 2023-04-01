// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>

#include "explorer/ast/clone_context.h"
#include "explorer/fuzzing/ast_to_proto.h"
#include "explorer/syntax/parse.h"

namespace Carbon::Testing {
namespace {

static std::vector<llvm::StringRef>* carbon_files = nullptr;

auto CloneAST(Arena& arena, const AST& ast) -> AST {
  CloneContext context(&arena);
  return {
      .package = ast.package,
      .is_api = ast.is_api,
      .imports = ast.imports,
      .declarations = context.Clone(ast.declarations),
      .main_call = context.Clone(ast.main_call),
      .num_prelude_declarations = ast.num_prelude_declarations,
  };
}

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

TEST(CloneTest, SameProtoAfterClone) {
  int parsed_ok_count = 0;
  for (const llvm::StringRef f : *carbon_files) {
    Carbon::Arena arena;
    const ErrorOr<AST> ast = Carbon::Parse(&arena, f, /*parser_debug=*/false);
    if (ast.ok()) {
      ++parsed_ok_count;
      const AST clone = CloneAST(arena, *ast);
      const Fuzzing::CompilationUnit orig_proto = AstToProto(*ast);
      const Fuzzing::CompilationUnit clone_proto = AstToProto(clone);
      // TODO: Use EqualsProto once it's available.
      EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
          orig_proto, clone_proto))
          << "clone produced a different AST. original:\n"
          << AstToString(*ast) << "clone:\n"
          << AstToString(clone);
    }
  }
  // Makes sure files were actually processed.
  EXPECT_GT(parsed_ok_count, 0);
}

}  // namespace
}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  // gtest should remove flags, leaving just input files.
  std::vector<llvm::StringRef> carbon_files(&argv[1], &argv[argc]);
  Carbon::Testing::carbon_files = &carbon_files;
  return RUN_ALL_TESTS();
}
