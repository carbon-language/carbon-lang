// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/testing/yaml_test_helpers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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

}  // namespace
}  // namespace Carbon::Testing
