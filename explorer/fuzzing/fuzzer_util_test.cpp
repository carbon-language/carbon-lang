// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/fuzzing/fuzzer_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "testing/fuzzing/proto_to_carbon.h"

namespace Carbon::Testing {
namespace {

TEST(FuzzerUtilTest, ParseAndExecuteProto) {
  const ErrorOr<Fuzzing::Carbon> carbon_proto = ParseCarbonTextProto(R"(
    compilation_unit {
      package_statement { package_name: "P" }
      is_api: true
      declarations {
        function {
          name {
            name: "Main"
          }
          param_pattern {}
          return_term {
            kind: Expression
            type { int_type_literal {} }
          }
          body {
            statements {
              return_expression_statement {
                expression { int_literal { value: 0 } }
              }
            }
          }
        }
      }
    })");
  ASSERT_TRUE(carbon_proto.ok());
  const ErrorOr<int> result = ParseAndExecuteProto(*carbon_proto);
  ASSERT_TRUE(result.ok()) << "Execution failed: " << result.error();
  EXPECT_EQ(*result, 0);
}

TEST(FuzzerUtilTest, GetRunfilesFile) {
  EXPECT_THAT(*GetRunfilesFile("carbon/explorer/data/prelude.carbon"),
              testing::EndsWith("/prelude.carbon"));
  EXPECT_THAT(GetRunfilesFile("nonexistent-file").error().message(),
              testing::EndsWith("doesn't exist"));
}

}  // namespace
}  // namespace Carbon::Testing
