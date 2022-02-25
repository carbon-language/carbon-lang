#include "GlobList.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {

TEST(GlobList, Empty) {
  GlobList Filter("");

  EXPECT_TRUE(Filter.contains(""));
  EXPECT_FALSE(Filter.contains("aaa"));
}

TEST(GlobList, Nothing) {
  GlobList Filter("-*");

  EXPECT_FALSE(Filter.contains(""));
  EXPECT_FALSE(Filter.contains("a"));
  EXPECT_FALSE(Filter.contains("-*"));
  EXPECT_FALSE(Filter.contains("-"));
  EXPECT_FALSE(Filter.contains("*"));
}

TEST(GlobList, Everything) {
  GlobList Filter("*");

  EXPECT_TRUE(Filter.contains(""));
  EXPECT_TRUE(Filter.contains("aaaa"));
  EXPECT_TRUE(Filter.contains("-*"));
  EXPECT_TRUE(Filter.contains("-"));
  EXPECT_TRUE(Filter.contains("*"));
}

TEST(GlobList, OneSimplePattern) {
  GlobList Filter("aaa");

  EXPECT_TRUE(Filter.contains("aaa"));
  EXPECT_FALSE(Filter.contains(""));
  EXPECT_FALSE(Filter.contains("aa"));
  EXPECT_FALSE(Filter.contains("aaaa"));
  EXPECT_FALSE(Filter.contains("bbb"));
}

TEST(GlobList, TwoSimplePatterns) {
  GlobList Filter("aaa,bbb");

  EXPECT_TRUE(Filter.contains("aaa"));
  EXPECT_TRUE(Filter.contains("bbb"));
  EXPECT_FALSE(Filter.contains(""));
  EXPECT_FALSE(Filter.contains("aa"));
  EXPECT_FALSE(Filter.contains("aaaa"));
  EXPECT_FALSE(Filter.contains("bbbb"));
}

TEST(GlobList, PatternPriority) {
  // The last glob that matches the string decides whether that string is
  // included or excluded.
  {
    GlobList Filter("a*,-aaa");

    EXPECT_FALSE(Filter.contains(""));
    EXPECT_TRUE(Filter.contains("a"));
    EXPECT_TRUE(Filter.contains("aa"));
    EXPECT_FALSE(Filter.contains("aaa"));
    EXPECT_TRUE(Filter.contains("aaaa"));
  }
  {
    GlobList Filter("-aaa,a*");

    EXPECT_FALSE(Filter.contains(""));
    EXPECT_TRUE(Filter.contains("a"));
    EXPECT_TRUE(Filter.contains("aa"));
    EXPECT_TRUE(Filter.contains("aaa"));
    EXPECT_TRUE(Filter.contains("aaaa"));
  }
}

TEST(GlobList, WhitespacesAtBegin) {
  GlobList Filter("-*,   a.b.*");

  EXPECT_TRUE(Filter.contains("a.b.c"));
  EXPECT_FALSE(Filter.contains("b.c"));
}

TEST(GlobList, Complex) {
  GlobList Filter("*,-a.*, -b.*, \r  \n  a.1.* ,-a.1.A.*,-..,-...,-..+,-*$, -*qwe* ");

  EXPECT_TRUE(Filter.contains("aaa"));
  EXPECT_TRUE(Filter.contains("qqq"));
  EXPECT_FALSE(Filter.contains("a."));
  EXPECT_FALSE(Filter.contains("a.b"));
  EXPECT_FALSE(Filter.contains("b."));
  EXPECT_FALSE(Filter.contains("b.b"));
  EXPECT_TRUE(Filter.contains("a.1.b"));
  EXPECT_FALSE(Filter.contains("a.1.A.a"));
  EXPECT_FALSE(Filter.contains("qwe"));
  EXPECT_FALSE(Filter.contains("asdfqweasdf"));
  EXPECT_TRUE(Filter.contains("asdfqwEasdf"));
}

} // namespace tidy
} // namespace clang
