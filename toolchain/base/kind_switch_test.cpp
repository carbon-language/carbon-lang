// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/kind_switch.h"

#include <gtest/gtest.h>

#include <string>
#include <variant>

#include "common/raw_string_ostream.h"

namespace Carbon {
namespace {

TEST(KindSwitch, Variant) {
  auto f = [](std::variant<int, float, char> v) -> std::string {
    RawStringOstream str;
    CARBON_KIND_SWITCH(v) {
      case CARBON_KIND(int n):
        str << "int = " << n;
        return str.TakeStr();
      case CARBON_KIND(float f):
        str << "float = " << f;
        return str.TakeStr();
      case CARBON_KIND(char c):
        str << "char = " << c;
        return str.TakeStr();
    }
  };

  EXPECT_EQ(f(int{1}), "int = 1");
  EXPECT_EQ(f(float{2}), "float = 2.000000e+00");
  EXPECT_EQ(f(char{'h'}), "char = h");
}

TEST(KindSwitch, VariantUnusedValue) {
  auto f = [](std::variant<int, float> v) -> std::string {
    RawStringOstream str;
    CARBON_KIND_SWITCH(v) {
      case CARBON_KIND(int n):
        str << "int = " << n;
        return str.TakeStr();
      case CARBON_KIND(float _):
        // The float value is not used, we see that using `_` works.
        str << "float";
        return str.TakeStr();
    }
  };

  EXPECT_EQ(f(int{1}), "int = 1");
  EXPECT_EQ(f(float{2}), "float");
}

}  // namespace
}  // namespace Carbon
