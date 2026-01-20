// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/testing/yaml_test_helpers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/error_test_helpers.h"

namespace Carbon::Testing {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Not;

TEST(YamlTestHelpersTest, ValidYaml) {
  EXPECT_THAT(
      Yaml::Value::FromText("[foo, bar]"),
      Yaml::IsYaml(ElementsAre(Yaml::Sequence(ElementsAre("foo", "bar")))));
}

TEST(YamlTestHelpersTest, InvalidYaml) {
  auto result = Yaml::Value::FromText("- foo\nbar");
  // Make sure we've constructed invalid YAML.
  EXPECT_FALSE(result.ok());
  // Make sure the matcher detects the invalid YAML.
  EXPECT_THAT(result, Not(Yaml::IsYaml(_)));
}

TEST(YamlTestHelpersTest, ComposeWithErrorOr) {
  auto helper = []() -> ErrorOr<Yaml::Value> {
    auto result = Yaml::Value::FromText("[foo, bar]");
    if (!result.ok()) {
      return std::move(result).error();
    }
    return {*std::move(result)};
  };

  // Make sure this works correctly with the generic `ErrorOr` test helper as
  // well. Note that `FromText` always produces a sequence of its own, so there
  // are two layers of nested sequence here.
  EXPECT_THAT(helper(), IsSuccess(Yaml::Sequence(ElementsAre(
                            Yaml::Sequence(ElementsAre("foo", "bar"))))));
}

}  // namespace
}  // namespace Carbon::Testing
