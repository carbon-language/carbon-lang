#include "ClangTidy.h"
#include "ClangTidyTest.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace test {

class TestCheck : public ClangTidyCheck {
public:
  void registerMatchers(ast_matchers::MatchFinder *Finder) override {
    Finder->addMatcher(ast_matchers::varDecl().bind("var"), this);
  }
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const VarDecl *Var = Result.Nodes.getNodeAs<VarDecl>("var");
    // Add diagnostics in the wrong order.
    diag(Var->getLocation(), "variable");
    diag(Var->getTypeSpecStartLoc(), "type specifier");
  }
};

TEST(ClangTidyDiagnosticConsumer, SortsErrors) {
  std::vector<ClangTidyError> Errors;
  runCheckOnCode<TestCheck>("int a;", &Errors);
  EXPECT_EQ(2ul, Errors.size());
  // FIXME: Remove " []" once the check name is removed from the message text.
  EXPECT_EQ("type specifier []", Errors[0].Message.Message);
  EXPECT_EQ("variable []", Errors[1].Message.Message);
}

TEST(ChecksFilter, Empty) {
  ChecksFilter Filter("");

  EXPECT_TRUE(Filter.isCheckEnabled(""));
  EXPECT_FALSE(Filter.isCheckEnabled("aaa"));
}

TEST(ChecksFilter, Nothing) {
  ChecksFilter Filter("-*");

  EXPECT_FALSE(Filter.isCheckEnabled(""));
  EXPECT_FALSE(Filter.isCheckEnabled("a"));
  EXPECT_FALSE(Filter.isCheckEnabled("-*"));
  EXPECT_FALSE(Filter.isCheckEnabled("-"));
  EXPECT_FALSE(Filter.isCheckEnabled("*"));
}

TEST(ChecksFilter, Everything) {
  ChecksFilter Filter("*");

  EXPECT_TRUE(Filter.isCheckEnabled(""));
  EXPECT_TRUE(Filter.isCheckEnabled("aaaa"));
  EXPECT_TRUE(Filter.isCheckEnabled("-*"));
  EXPECT_TRUE(Filter.isCheckEnabled("-"));
  EXPECT_TRUE(Filter.isCheckEnabled("*"));
}

TEST(ChecksFilter, Simple) {
  ChecksFilter Filter("aaa");

  EXPECT_TRUE(Filter.isCheckEnabled("aaa"));
  EXPECT_FALSE(Filter.isCheckEnabled(""));
  EXPECT_FALSE(Filter.isCheckEnabled("aa"));
  EXPECT_FALSE(Filter.isCheckEnabled("aaaa"));
  EXPECT_FALSE(Filter.isCheckEnabled("bbb"));
}

TEST(ChecksFilter, Complex) {
  ChecksFilter Filter("*,-a.*,-b.*,a.a.*,-a.a.a.*,-..,-...,-..+,-*$,-*qwe*");

  EXPECT_TRUE(Filter.isCheckEnabled("aaa"));
  EXPECT_TRUE(Filter.isCheckEnabled("qqq"));
  EXPECT_FALSE(Filter.isCheckEnabled("a."));
  EXPECT_FALSE(Filter.isCheckEnabled("a.b"));
  EXPECT_FALSE(Filter.isCheckEnabled("b."));
  EXPECT_FALSE(Filter.isCheckEnabled("b.b"));
  EXPECT_TRUE(Filter.isCheckEnabled("a.a.b"));
  EXPECT_FALSE(Filter.isCheckEnabled("a.a.a.a"));
  EXPECT_FALSE(Filter.isCheckEnabled("qwe"));
  EXPECT_FALSE(Filter.isCheckEnabled("asdfqweasdf"));
  EXPECT_TRUE(Filter.isCheckEnabled("asdfqwEasdf"));
}

} // namespace test
} // namespace tidy
} // namespace clang
