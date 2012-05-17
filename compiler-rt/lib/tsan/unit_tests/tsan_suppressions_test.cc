//===-- tsan_suppressions_test.cc -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_suppressions.h"
#include "tsan_rtl.h"
#include "gtest/gtest.h"

#include <string.h>

namespace __tsan {

TEST(Suppressions, Parse) {
  ScopedInRtl in_rtl;
  Suppression *supp0 = SuppressionParse(
    "race:foo\n"
    " 	race:bar\n"  // NOLINT
    "race:baz	 \n"  // NOLINT
    "# a comment\n"
    "race:quz\n"
  );  // NOLINT
  Suppression *supp = supp0;
  EXPECT_EQ(supp->type, SuppressionRace);
  EXPECT_EQ(0, strcmp(supp->func, "quz"));
  supp = supp->next;
  EXPECT_EQ(supp->type, SuppressionRace);
  EXPECT_EQ(0, strcmp(supp->func, "baz"));
  supp = supp->next;
  EXPECT_EQ(supp->type, SuppressionRace);
  EXPECT_EQ(0, strcmp(supp->func, "bar"));
  supp = supp->next;
  EXPECT_EQ(supp->type, SuppressionRace);
  EXPECT_EQ(0, strcmp(supp->func, "foo"));
  supp = supp->next;
  EXPECT_EQ((Suppression*)0, supp);
}

TEST(Suppressions, Parse2) {
  ScopedInRtl in_rtl;
  Suppression *supp0 = SuppressionParse(
    "  	# first line comment\n"  // NOLINT
    " 	race:bar 	\n"  // NOLINT
    "race:baz* *baz\n"
    "# a comment\n"
    "# last line comment\n"
  );  // NOLINT
  Suppression *supp = supp0;
  EXPECT_EQ(supp->type, SuppressionRace);
  EXPECT_EQ(0, strcmp(supp->func, "baz* *baz"));
  supp = supp->next;
  EXPECT_EQ(supp->type, SuppressionRace);
  EXPECT_EQ(0, strcmp(supp->func, "bar"));
  supp = supp->next;
  EXPECT_EQ((Suppression*)0, supp);
}

TEST(Suppressions, Parse3) {
  ScopedInRtl in_rtl;
  Suppression *supp0 = SuppressionParse(
    "# last suppression w/o line-feed\n"
    "race:foo\n"
    "race:bar"
  );  // NOLINT
  Suppression *supp = supp0;
  EXPECT_EQ(supp->type, SuppressionRace);
  EXPECT_EQ(0, strcmp(supp->func, "bar"));
  supp = supp->next;
  EXPECT_EQ(supp->type, SuppressionRace);
  EXPECT_EQ(0, strcmp(supp->func, "foo"));
  supp = supp->next;
  EXPECT_EQ((Suppression*)0, supp);
}

TEST(Suppressions, ParseType) {
  ScopedInRtl in_rtl;
  Suppression *supp0 = SuppressionParse(
    "race:foo\n"
    "thread:bar\n"
    "mutex:baz\n"
    "signal:quz\n"
  );  // NOLINT
  Suppression *supp = supp0;
  EXPECT_EQ(supp->type, SuppressionSignal);
  EXPECT_EQ(0, strcmp(supp->func, "quz"));
  supp = supp->next;
  EXPECT_EQ(supp->type, SuppressionMutex);
  EXPECT_EQ(0, strcmp(supp->func, "baz"));
  supp = supp->next;
  EXPECT_EQ(supp->type, SuppressionThread);
  EXPECT_EQ(0, strcmp(supp->func, "bar"));
  supp = supp->next;
  EXPECT_EQ(supp->type, SuppressionRace);
  EXPECT_EQ(0, strcmp(supp->func, "foo"));
  supp = supp->next;
  EXPECT_EQ((Suppression*)0, supp);
}

static bool MyMatch(const char *templ, const char *func) {
  char tmp[1024];
  strcpy(tmp, templ);  // NOLINT
  return SuppressionMatch(tmp, func);
}

TEST(Suppressions, Match) {
  EXPECT_TRUE(MyMatch("foobar", "foobar"));
  EXPECT_TRUE(MyMatch("foobar", "prefix_foobar_postfix"));
  EXPECT_TRUE(MyMatch("*foobar*", "prefix_foobar_postfix"));
  EXPECT_TRUE(MyMatch("foo*bar", "foo_middle_bar"));
  EXPECT_TRUE(MyMatch("foo*bar", "foobar"));
  EXPECT_TRUE(MyMatch("foo*bar*baz", "foo_middle_bar_another_baz"));
  EXPECT_TRUE(MyMatch("foo*bar*baz", "foo_middle_barbaz"));

  EXPECT_FALSE(MyMatch("foo", "baz"));
  EXPECT_FALSE(MyMatch("foobarbaz", "foobar"));
  EXPECT_FALSE(MyMatch("foobarbaz", "barbaz"));
  EXPECT_FALSE(MyMatch("foo*bar", "foobaz"));
  EXPECT_FALSE(MyMatch("foo*bar", "foo_baz"));
}

}  // namespace __tsan
