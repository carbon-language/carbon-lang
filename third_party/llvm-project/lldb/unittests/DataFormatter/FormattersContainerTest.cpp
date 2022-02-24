//===-- FormattersContainerTests.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/FormattersContainer.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

// All the prefixes that the exact name matching will strip from the type.
static const std::vector<std::string> exact_name_prefixes = {
    "", // no prefix.
    "class ", "struct ", "union ", "enum ",
};

// TypeMatcher that uses a exact type name string that needs to be matched.
TEST(TypeMatcherTests, ExactName) {
  for (const std::string &prefix : exact_name_prefixes) {
    SCOPED_TRACE("Prefix: " + prefix);

    TypeMatcher matcher(ConstString(prefix + "Name"));
    EXPECT_TRUE(matcher.Matches(ConstString("class Name")));
    EXPECT_TRUE(matcher.Matches(ConstString("struct Name")));
    EXPECT_TRUE(matcher.Matches(ConstString("union Name")));
    EXPECT_TRUE(matcher.Matches(ConstString("enum Name")));
    EXPECT_TRUE(matcher.Matches(ConstString("Name")));

    EXPECT_FALSE(matcher.Matches(ConstString("Name ")));
    EXPECT_FALSE(matcher.Matches(ConstString("ame")));
    EXPECT_FALSE(matcher.Matches(ConstString("Nam")));
    EXPECT_FALSE(matcher.Matches(ConstString("am")));
    EXPECT_FALSE(matcher.Matches(ConstString("a")));
    EXPECT_FALSE(matcher.Matches(ConstString(" ")));
    EXPECT_FALSE(matcher.Matches(ConstString("class N")));
    EXPECT_FALSE(matcher.Matches(ConstString("class ")));
    EXPECT_FALSE(matcher.Matches(ConstString("class")));
  }
}

// TypeMatcher that uses a regex to match a type name.
TEST(TypeMatcherTests, RegexName) {
  TypeMatcher matcher(RegularExpression("^a[a-z]c$"));
  EXPECT_TRUE(matcher.Matches(ConstString("abc")));
  EXPECT_TRUE(matcher.Matches(ConstString("azc")));

  // FIXME: This isn't consistent with the 'exact' type name matches above.
  EXPECT_FALSE(matcher.Matches(ConstString("class abc")));

  EXPECT_FALSE(matcher.Matches(ConstString("abbc")));
  EXPECT_FALSE(matcher.Matches(ConstString(" abc")));
  EXPECT_FALSE(matcher.Matches(ConstString("abc ")));
  EXPECT_FALSE(matcher.Matches(ConstString(" abc ")));
  EXPECT_FALSE(matcher.Matches(ConstString("XabcX")));
  EXPECT_FALSE(matcher.Matches(ConstString("ac")));
  EXPECT_FALSE(matcher.Matches(ConstString("a[a-z]c")));
  EXPECT_FALSE(matcher.Matches(ConstString("aAc")));
  EXPECT_FALSE(matcher.Matches(ConstString("ABC")));
  EXPECT_FALSE(matcher.Matches(ConstString("")));
}

// TypeMatcher that only searches the type name.
TEST(TypeMatcherTests, RegexMatchPart) {
  TypeMatcher matcher(RegularExpression("a[a-z]c"));
  EXPECT_TRUE(matcher.Matches(ConstString("class abc")));
  EXPECT_TRUE(matcher.Matches(ConstString("abc")));
  EXPECT_TRUE(matcher.Matches(ConstString(" abc ")));
  EXPECT_TRUE(matcher.Matches(ConstString("azc")));
  EXPECT_TRUE(matcher.Matches(ConstString("abc ")));
  EXPECT_TRUE(matcher.Matches(ConstString(" abc ")));
  EXPECT_TRUE(matcher.Matches(ConstString(" abc")));
  EXPECT_TRUE(matcher.Matches(ConstString("XabcX")));

  EXPECT_FALSE(matcher.Matches(ConstString("abbc")));
  EXPECT_FALSE(matcher.Matches(ConstString("ac")));
  EXPECT_FALSE(matcher.Matches(ConstString("a[a-z]c")));
  EXPECT_FALSE(matcher.Matches(ConstString("aAc")));
  EXPECT_FALSE(matcher.Matches(ConstString("ABC")));
  EXPECT_FALSE(matcher.Matches(ConstString("")));
}

// GetMatchString for exact type name matchers.
TEST(TypeMatcherTests, GetMatchStringExactName) {
  EXPECT_EQ(TypeMatcher(ConstString("aa")).GetMatchString(), "aa");
  EXPECT_EQ(TypeMatcher(ConstString("")).GetMatchString(), "");
  EXPECT_EQ(TypeMatcher(ConstString("[a]")).GetMatchString(), "[a]");
}

// GetMatchString for regex matchers.
TEST(TypeMatcherTests, GetMatchStringRegex) {
  EXPECT_EQ(TypeMatcher(RegularExpression("aa")).GetMatchString(), "aa");
  EXPECT_EQ(TypeMatcher(RegularExpression("")).GetMatchString(), "");
  EXPECT_EQ(TypeMatcher(RegularExpression("[a]")).GetMatchString(), "[a]");
}

// GetMatchString for regex matchers.
TEST(TypeMatcherTests, CreatedBySameMatchString) {
  TypeMatcher empty_str(ConstString(""));
  TypeMatcher empty_regex(RegularExpression(""));
  EXPECT_TRUE(empty_str.CreatedBySameMatchString(empty_str));
  EXPECT_TRUE(empty_str.CreatedBySameMatchString(empty_regex));

  TypeMatcher a_str(ConstString("a"));
  TypeMatcher a_regex(RegularExpression("a"));
  EXPECT_TRUE(a_str.CreatedBySameMatchString(a_str));
  EXPECT_TRUE(a_str.CreatedBySameMatchString(a_regex));

  TypeMatcher digit_str(ConstString("[0-9]"));
  TypeMatcher digit_regex(RegularExpression("[0-9]"));
  EXPECT_TRUE(digit_str.CreatedBySameMatchString(digit_str));
  EXPECT_TRUE(digit_str.CreatedBySameMatchString(digit_regex));

  EXPECT_FALSE(empty_str.CreatedBySameMatchString(a_str));
  EXPECT_FALSE(empty_str.CreatedBySameMatchString(a_regex));
  EXPECT_FALSE(empty_str.CreatedBySameMatchString(digit_str));
  EXPECT_FALSE(empty_str.CreatedBySameMatchString(digit_regex));

  EXPECT_FALSE(empty_regex.CreatedBySameMatchString(a_str));
  EXPECT_FALSE(empty_regex.CreatedBySameMatchString(a_regex));
  EXPECT_FALSE(empty_regex.CreatedBySameMatchString(digit_str));
  EXPECT_FALSE(empty_regex.CreatedBySameMatchString(digit_regex));

  EXPECT_FALSE(a_str.CreatedBySameMatchString(empty_str));
  EXPECT_FALSE(a_str.CreatedBySameMatchString(empty_regex));
  EXPECT_FALSE(a_str.CreatedBySameMatchString(digit_str));
  EXPECT_FALSE(a_str.CreatedBySameMatchString(digit_regex));

  EXPECT_FALSE(a_regex.CreatedBySameMatchString(empty_str));
  EXPECT_FALSE(a_regex.CreatedBySameMatchString(empty_regex));
  EXPECT_FALSE(a_regex.CreatedBySameMatchString(digit_str));
  EXPECT_FALSE(a_regex.CreatedBySameMatchString(digit_regex));

  EXPECT_FALSE(digit_str.CreatedBySameMatchString(empty_str));
  EXPECT_FALSE(digit_str.CreatedBySameMatchString(empty_regex));
  EXPECT_FALSE(digit_str.CreatedBySameMatchString(a_str));
  EXPECT_FALSE(digit_str.CreatedBySameMatchString(a_regex));

  EXPECT_FALSE(digit_regex.CreatedBySameMatchString(empty_str));
  EXPECT_FALSE(digit_regex.CreatedBySameMatchString(empty_regex));
  EXPECT_FALSE(digit_regex.CreatedBySameMatchString(a_str));
  EXPECT_FALSE(digit_regex.CreatedBySameMatchString(a_regex));
}

// Test CreatedBySameMatchString with stripped exact name prefixes.
TEST(TypeMatcherTests, CreatedBySameMatchStringExactNamePrefixes) {
  for (const std::string &prefix : exact_name_prefixes) {
    SCOPED_TRACE("Prefix: " + prefix);
    TypeMatcher with_prefix(ConstString(prefix + "Name"));
    TypeMatcher without_prefix(RegularExpression(""));

    EXPECT_TRUE(with_prefix.CreatedBySameMatchString(with_prefix));
    EXPECT_TRUE(without_prefix.CreatedBySameMatchString(without_prefix));
  }
}
