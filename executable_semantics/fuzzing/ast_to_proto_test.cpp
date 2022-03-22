// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/fuzzing/ast_to_proto.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <numeric>
#include <set>
#include <variant>

#include "executable_semantics/syntax/parse.h"
#include "google/protobuf/descriptor.h"
#include "llvm/Support/Error.h"

namespace Carbon::Testing {
namespace {

using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;

constexpr std::string_view AdditionalSyntax = R"(
  package p api;

  fn f() {
    __intrinsic_print("xyz");
    a __unimplemented_example_infix b;
  }
)";

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

// Finds all `.carbon` files under `root_dir`.
auto GetFiles(std::string_view root_dir, std::string_view extension)
    -> std::vector<std::string> {
  std::vector<std::string> carbon_files;
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::recursive_directory_iterator(root_dir)) {
    if (!std::filesystem::is_directory(entry)) {
      const std::string file = entry.path();
      // Checks that `file` ends with `extension`.
      if (std::equal(extension.rbegin(), extension.rend(), file.rbegin())) {
        carbon_files.push_back(file);
      }
    }
  }
  return carbon_files;
}

TEST(CarbonToProtoTest, SetsAllProtoFields) {
  Carbon::Fuzzing::CompilationUnit merged_proto;
  const std::vector<std::string> carbon_files =
      GetFiles(std::string(getenv("TEST_SRCDIR")) +
                   "/carbon/executable_semantics/testdata",
               ".carbon");
  for (const std::string& f : carbon_files) {
    Carbon::Arena arena;
    const ErrorOr<AST> ast = Carbon::Parse(&arena, f, /*trace=*/false);
    if (ast.ok()) {
      merged_proto.MergeFrom(CarbonToProto(*ast));
    }
  }

  Carbon::Arena arena;
  const ErrorOr<AST> ast =
      Carbon::ParseFromString(&arena, "File.carbon", AdditionalSyntax,
                              /*trace=*/false);
  ASSERT_TRUE(ast.ok());
  merged_proto.MergeFrom(CarbonToProto(*ast));

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
