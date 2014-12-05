//===-- sanitizer_suppressions_test.cc ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_suppressions.h"
#include "gtest/gtest.h"

#include <string.h>

namespace __sanitizer {

static bool MyMatch(const char *templ, const char *func) {
  char tmp[1024];
  strcpy(tmp, templ);  // NOLINT
  return TemplateMatch(tmp, func);
}

TEST(Suppressions, Match) {
  EXPECT_TRUE(MyMatch("foobar$", "foobar"));

  EXPECT_TRUE(MyMatch("foobar", "foobar"));
  EXPECT_TRUE(MyMatch("*foobar*", "foobar"));
  EXPECT_TRUE(MyMatch("foobar", "prefix_foobar_postfix"));
  EXPECT_TRUE(MyMatch("*foobar*", "prefix_foobar_postfix"));
  EXPECT_TRUE(MyMatch("foo*bar", "foo_middle_bar"));
  EXPECT_TRUE(MyMatch("foo*bar", "foobar"));
  EXPECT_TRUE(MyMatch("foo*bar*baz", "foo_middle_bar_another_baz"));
  EXPECT_TRUE(MyMatch("foo*bar*baz", "foo_middle_barbaz"));
  EXPECT_TRUE(MyMatch("^foobar", "foobar"));
  EXPECT_TRUE(MyMatch("^foobar", "foobar_postfix"));
  EXPECT_TRUE(MyMatch("^*foobar", "foobar"));
  EXPECT_TRUE(MyMatch("^*foobar", "prefix_foobar"));
  EXPECT_TRUE(MyMatch("foobar$", "foobar"));
  EXPECT_TRUE(MyMatch("foobar$", "prefix_foobar"));
  EXPECT_TRUE(MyMatch("*foobar*$", "foobar"));
  EXPECT_TRUE(MyMatch("*foobar*$", "foobar_postfix"));
  EXPECT_TRUE(MyMatch("^foobar$", "foobar"));

  EXPECT_FALSE(MyMatch("foo", "baz"));
  EXPECT_FALSE(MyMatch("foobarbaz", "foobar"));
  EXPECT_FALSE(MyMatch("foobarbaz", "barbaz"));
  EXPECT_FALSE(MyMatch("foo*bar", "foobaz"));
  EXPECT_FALSE(MyMatch("foo*bar", "foo_baz"));
  EXPECT_FALSE(MyMatch("^foobar", "prefix_foobar"));
  EXPECT_FALSE(MyMatch("foobar$", "foobar_postfix"));
  EXPECT_FALSE(MyMatch("^foobar$", "prefix_foobar"));
  EXPECT_FALSE(MyMatch("^foobar$", "foobar_postfix"));
  EXPECT_FALSE(MyMatch("foo^bar", "foobar"));
  EXPECT_FALSE(MyMatch("foo$bar", "foobar"));
  EXPECT_FALSE(MyMatch("foo$^bar", "foobar"));
}

TEST(Suppressions, TypeStrings) {
  CHECK(!internal_strcmp(SuppressionTypeString(SuppressionNone), "none"));
  CHECK(!internal_strcmp(SuppressionTypeString(SuppressionRace), "race"));
  CHECK(!internal_strcmp(SuppressionTypeString(SuppressionMutex), "mutex"));
  CHECK(!internal_strcmp(SuppressionTypeString(SuppressionThread), "thread"));
  CHECK(!internal_strcmp(SuppressionTypeString(SuppressionSignal), "signal"));
  CHECK(!internal_strcmp(SuppressionTypeString(SuppressionLeak), "leak"));
  CHECK(!internal_strcmp(SuppressionTypeString(SuppressionLib),
                         "called_from_lib"));
  CHECK(
      !internal_strcmp(SuppressionTypeString(SuppressionDeadlock), "deadlock"));
  CHECK(!internal_strcmp(SuppressionTypeString(SuppressionVptrCheck),
                         "vptr_check"));
  CHECK(!internal_strcmp(SuppressionTypeString(SuppressionInterceptorName),
                         "interceptor_name"));
  CHECK(
      !internal_strcmp(SuppressionTypeString(SuppressionInterceptorViaFunction),
                       "interceptor_via_fun"));
  CHECK(
      !internal_strcmp(SuppressionTypeString(SuppressionInterceptorViaLibrary),
                       "interceptor_via_lib"));
  // Ensure this test is up-to-date when suppression types are added.
  CHECK_EQ(12, SuppressionTypeCount);
}

class SuppressionContextTest : public ::testing::Test {
 public:
  virtual void SetUp() { ctx_ = new(placeholder_) SuppressionContext; }
  virtual void TearDown() { ctx_->~SuppressionContext(); }

 protected:
  InternalMmapVector<Suppression> *Suppressions() {
    return &ctx_->suppressions_;
  }
  SuppressionContext *ctx_;
  ALIGNED(64) char placeholder_[sizeof(SuppressionContext)];
};

TEST_F(SuppressionContextTest, Parse) {
  ctx_->Parse(
    "race:foo\n"
    " 	race:bar\n"  // NOLINT
    "race:baz	 \n"  // NOLINT
    "# a comment\n"
    "race:quz\n"
  );  // NOLINT
  EXPECT_EQ((unsigned)4, ctx_->SuppressionCount());
  EXPECT_EQ((*Suppressions())[3].type, SuppressionRace);
  EXPECT_EQ(0, strcmp((*Suppressions())[3].templ, "quz"));
  EXPECT_EQ((*Suppressions())[2].type, SuppressionRace);
  EXPECT_EQ(0, strcmp((*Suppressions())[2].templ, "baz"));
  EXPECT_EQ((*Suppressions())[1].type, SuppressionRace);
  EXPECT_EQ(0, strcmp((*Suppressions())[1].templ, "bar"));
  EXPECT_EQ((*Suppressions())[0].type, SuppressionRace);
  EXPECT_EQ(0, strcmp((*Suppressions())[0].templ, "foo"));
}

TEST_F(SuppressionContextTest, Parse2) {
  ctx_->Parse(
    "  	# first line comment\n"  // NOLINT
    " 	race:bar 	\n"  // NOLINT
    "race:baz* *baz\n"
    "# a comment\n"
    "# last line comment\n"
  );  // NOLINT
  EXPECT_EQ((unsigned)2, ctx_->SuppressionCount());
  EXPECT_EQ((*Suppressions())[1].type, SuppressionRace);
  EXPECT_EQ(0, strcmp((*Suppressions())[1].templ, "baz* *baz"));
  EXPECT_EQ((*Suppressions())[0].type, SuppressionRace);
  EXPECT_EQ(0, strcmp((*Suppressions())[0].templ, "bar"));
}

TEST_F(SuppressionContextTest, Parse3) {
  ctx_->Parse(
    "# last suppression w/o line-feed\n"
    "race:foo\n"
    "race:bar"
  );  // NOLINT
  EXPECT_EQ((unsigned)2, ctx_->SuppressionCount());
  EXPECT_EQ((*Suppressions())[1].type, SuppressionRace);
  EXPECT_EQ(0, strcmp((*Suppressions())[1].templ, "bar"));
  EXPECT_EQ((*Suppressions())[0].type, SuppressionRace);
  EXPECT_EQ(0, strcmp((*Suppressions())[0].templ, "foo"));
}

TEST_F(SuppressionContextTest, ParseType) {
  ctx_->Parse(
    "race:foo\n"
    "thread:bar\n"
    "mutex:baz\n"
    "signal:quz\n"
  );  // NOLINT
  EXPECT_EQ((unsigned)4, ctx_->SuppressionCount());
  EXPECT_EQ((*Suppressions())[3].type, SuppressionSignal);
  EXPECT_EQ(0, strcmp((*Suppressions())[3].templ, "quz"));
  EXPECT_EQ((*Suppressions())[2].type, SuppressionMutex);
  EXPECT_EQ(0, strcmp((*Suppressions())[2].templ, "baz"));
  EXPECT_EQ((*Suppressions())[1].type, SuppressionThread);
  EXPECT_EQ(0, strcmp((*Suppressions())[1].templ, "bar"));
  EXPECT_EQ((*Suppressions())[0].type, SuppressionRace);
  EXPECT_EQ(0, strcmp((*Suppressions())[0].templ, "foo"));
}

TEST_F(SuppressionContextTest, HasSuppressionType) {
  ctx_->Parse(
    "race:foo\n"
    "thread:bar\n");
  EXPECT_TRUE(ctx_->HasSuppressionType(SuppressionRace));
  EXPECT_TRUE(ctx_->HasSuppressionType(SuppressionThread));
  EXPECT_FALSE(ctx_->HasSuppressionType(SuppressionMutex));
  EXPECT_FALSE(ctx_->HasSuppressionType(SuppressionSignal));
}

}  // namespace __sanitizer
