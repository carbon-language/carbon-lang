// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/fuzzing/ast_to_proto.h"

#include <gmock/gmock.h>
#include <google/protobuf/descriptor.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <numeric>
#include <set>
#include <variant>

#include "explorer/syntax/parse.h"
#include "llvm/Support/Error.h"

namespace Carbon::Testing {
namespace {

using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;

static std::vector<llvm::StringRef>* carbon_files = nullptr;

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

// Determines which fields in the proto have not been used at all.
auto GetUnusedFields(const Message& message) -> std::set<std::string> {
  std::set<std::string> all_messages;
  std::set<std::string> all_fields;
  CollectAllFields(*message.GetDescriptor(), all_messages, all_fields);

  std::set<std::string> used_fields;
  CollectUsedFields(message, used_fields);

  std::set<std::string> unused_fields;
  std::set_difference(all_fields.begin(), all_fields.end(), used_fields.begin(),
                      used_fields.end(),
                      std::inserter(unused_fields, unused_fields.begin()));
  return unused_fields;
}

// A 'smoke' test to check that each field present in `carbon.proto` is set at
// least once after converting all Carbon test sources to proto representation.
TEST(AstToProtoTest, SetsAllProtoFields) {
  Carbon::Fuzzing::CompilationUnit merged_proto;
  for (const llvm::StringRef f : *carbon_files) {
    Carbon::Arena arena;
    const ErrorOr<AST> ast = Carbon::Parse(&arena, f, /*parser_debug=*/false);
    if (ast.ok()) {
      merged_proto.MergeFrom(AstToProto(*ast));
    }
  }

  std::set<std::string> unused_fields = GetUnusedFields(merged_proto);
  EXPECT_EQ(unused_fields.size(), 0)
      << "Unused fields"
      << std::accumulate(unused_fields.begin(), unused_fields.end(),
                         std::string(),
                         [](const std::string& a, const std::string& b) {
                           return a + '\n' + b;
                         });
}

}  // namespace
}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  // gtest should remove flags, leaving just input files.
  Carbon::Testing::carbon_files =
      new std::vector<llvm::StringRef>(&argv[1], &argv[argc]);
  return RUN_ALL_TESTS();
}
