// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/fuzzing/ast_to_proto.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>
#include <set>

#include "executable_semantics/syntax/parse.h"
#include "google/protobuf/descriptor.h"
#include "llvm/Support/Error.h"

namespace Carbon::Testing {
namespace {

using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;

constexpr std::string_view UseAllSyntax = R"(
  package p library "lib" api;

  import other_package library "lib";

  choice Ints { None, One(i32) }

  class Point {
    var x: i32;
    var x: i32 = 0;
    fn GetX[me: Point]() -> i32 { return me.x; }
  }

  interface Vector {
    fn Add[me: Self](b: Self) -> Self;
  }

  external impl Point as Vector {
    fn Add[me: Point](b: Point) -> Point {
        return {.x = me.x + b.x, .y = me.y + b.y};
    }
  }

  fn Id(t: Type) -> auto { return t; }

  fn Func(b: i32) {
    var x: auto = Ints.None();
    match (x) {
      case Ints.One(x: auto) =>
        return 1;
    }
    var c: auto = b + 1;
    b = b + (-1);
    var d : Bool = true;
    var s : String = "abc";
    var p: {.x: i32, .y: i32} = {.x = 1, .y = 2};
    while (d) {
      if (d) { continue; } else { break; }
    }
    { Id(0); }
    var f: __Fn(i32)->i32 = add1;

    var t: auto = ((1,2),(3,4));
    match (t) {
      case ((a: auto, b: auto), c: auto) =>
        return 0;
    }

    __intrinsic_print("xyz");
    a __unimplemented_example_infix b;

    return b;
  }

  fn swap[T:! Type, U:! Type](tuple: (T, U)) -> (U, T) {
    return (tuple[1], tuple[0]);
  }

  fn Continuation() -> i32 {
    var x: i32 = 0;
    __continuation k {
      x = x + 1;
      __await;
      x = x + 2;
    }
    var k2 : __Continuation = k1;
    __run k;

    var if_expr = if cond then true else false;
    return x;
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

TEST(CarbonToProtoTest, SetsAllProtoFields) {
  Carbon::Arena arena;
  std::variant<Carbon::AST, Carbon::SyntaxErrorCode> ast_or_error =
      Carbon::ParseFromString(&arena, "Test.carbon", UseAllSyntax,
                              /*trace=*/false);
  auto* error = std::get_if<Carbon::SyntaxErrorCode>(&ast_or_error);
  ASSERT_TRUE(error == nullptr) << "Failed to parse: " << *error;

  auto& ast = std::get<Carbon::AST>(ast_or_error);
  const Carbon::Fuzzing::CompilationUnit proto = CarbonToProto(ast);
  std::set<std::string> unused_fields = GetUnusedFields(proto);
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
