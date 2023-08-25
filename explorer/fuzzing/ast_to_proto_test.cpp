// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/fuzzing/ast_to_proto.h"

#include <gmock/gmock.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <numeric>
#include <set>
#include <variant>

#include "explorer/syntax/parse.h"
#include "testing/base/test_raw_ostream.h"
#include "testing/fuzzing/proto_to_carbon.h"

namespace Carbon::Testing {
namespace {

using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;

static std::vector<llvm::StringRef>* carbon_files = nullptr;

// Returns a string representation of `ast`.
auto AstToString(const AST& ast) -> std::string {
  TestRawOstream out;
  out << "package " << ast.package.package << (ast.is_api ? "api" : "impl")
      << ";\n";
  for (auto* declaration : ast.declarations) {
    out << *declaration << "\n";
  }
  return out.TakeStr();
}

// Concatenates message and field names.
auto FieldName(const Descriptor& descriptor, const FieldDescriptor& field)
    -> std::string {
  return descriptor.full_name() + "." + field.name();
}

// Traverses the proto to find all unique messages and fields.
auto CollectAllFields(const Descriptor& descriptor,
                      std::set<std::string>& all_messages,
                      std::set<std::string>& all_fields) -> void {
  all_messages.insert(descriptor.full_name());
  for (int i = 0; i < descriptor.field_count(); ++i) {
    const FieldDescriptor* field = descriptor.field(i);
    all_fields.insert(FieldName(descriptor, *field));
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE &&
        all_messages.find(field->message_type()->full_name()) ==
            all_messages.end()) {
      CollectAllFields(*field->message_type(), all_messages, all_fields);
    }
  }
}

// Traverses an instance of the proto to find all used fields.
auto CollectUsedFields(const Message& message,
                       std::set<std::string>& used_fields) -> void {
  const Descriptor* descriptor = message.GetDescriptor();
  const Reflection* reflection = message.GetReflection();
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const FieldDescriptor* field = descriptor->field(i);
    if (!field->is_repeated()) {
      if (reflection->HasField(message, field)) {
        used_fields.insert(FieldName(*descriptor, *field));
      }
    } else {
      if (reflection->FieldSize(message, field) > 0) {
        used_fields.insert(FieldName(*descriptor, *field));
      }
    }
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      if (!field->is_repeated()) {
        if (reflection->HasField(message, field)) {
          CollectUsedFields(reflection->GetMessage(message, field),
                            used_fields);
        }
      } else {
        for (int i = 0; i < reflection->FieldSize(message, field); ++i) {
          CollectUsedFields(reflection->GetRepeatedMessage(message, field, i),
                            used_fields);
        }
      }
    }
  }
}

// A 'smoke' test to check that each field present in `carbon.proto` is set at
// least once after converting all Carbon test sources to proto representation.
TEST(AstToProtoTest, SetsAllProtoFields) {
  Fuzzing::Carbon merged_proto;
  for (const llvm::StringRef f : *carbon_files) {
    Arena arena;
    const ErrorOr<AST> ast = Parse(*llvm::vfs::getRealFileSystem(), &arena, f,
                                   FileKind::Main, /*parser_debug=*/false);
    if (ast.ok()) {
      merged_proto.MergeFrom(AstToProto(*ast));
    }
  }

  std::set<std::string> all_messages;
  std::set<std::string> all_fields;
  CollectAllFields(*Fuzzing::Carbon::GetDescriptor(), all_messages, all_fields);

  std::set<std::string> used_fields;
  CollectUsedFields(merged_proto, used_fields);

  std::set<std::string> unused_fields;
  std::set_difference(all_fields.begin(), all_fields.end(), used_fields.begin(),
                      used_fields.end(),
                      std::inserter(unused_fields, unused_fields.begin()));
  EXPECT_EQ(unused_fields.size(), 0)
      << "Unused fields"
      << std::accumulate(unused_fields.begin(), unused_fields.end(),
                         std::string(),
                         [](const std::string& a, const std::string& b) {
                           return a + '\n' + b;
                         });
}

// Ensures that `carbon.proto` is able to represent ASTs correctly without
// information loss by doing round-trip testing of files:
//
// 1) Converts each parseable Carbon file to a proto representation.
// 2) Converts back to Carbon source.
// 3) Parses the source into a second instance of an AST.
// 4) Compares the second AST with the original.
TEST(AstToProtoTest, Roundtrip) {
  int parsed_ok_count = 0;
  for (const llvm::StringRef f : *carbon_files) {
    Arena arena;
    const ErrorOr<AST> ast = Parse(*llvm::vfs::getRealFileSystem(), &arena, f,
                                   FileKind::Main, /*parser_debug=*/false);
    if (ast.ok()) {
      ++parsed_ok_count;
      const std::string source_from_proto =
          ProtoToCarbon(AstToProto(*ast), /*maybe_add_main=*/false);
      SCOPED_TRACE(testing::Message()
                   << "Carbon file: " << f << ", source from proto:\n"
                   << source_from_proto);
      const ErrorOr<AST> ast_from_proto = ParseFromString(
          &arena, f, FileKind::Main, source_from_proto, /*parser_debug=*/false);

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

// Verifies that an AST and its clone produce identical protos.
TEST(AstToProtoTest, SameProtoAfterClone) {
  int parsed_ok_count = 0;
  for (const llvm::StringRef f : *carbon_files) {
    Arena arena;
    const ErrorOr<AST> ast = Parse(*llvm::vfs::getRealFileSystem(), &arena, f,
                                   FileKind::Main, /*parser_debug=*/false);
    if (ast.ok()) {
      ++parsed_ok_count;
      const AST clone = CloneAST(arena, *ast);
      const Fuzzing::Carbon orig_proto = AstToProto(*ast);
      const Fuzzing::Carbon clone_proto = AstToProto(clone);
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
