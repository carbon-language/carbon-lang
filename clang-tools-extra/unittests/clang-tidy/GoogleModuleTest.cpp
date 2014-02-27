#include "ClangTidyTest.h"
#include "google/GoogleTidyModule.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace test {

#define EXPECT_NO_CHANGES(Check, Code)                                         \
  EXPECT_EQ(Code, runCheckOnCode<Check>(Code))

TEST(ExplicitConstructorCheckTest, SingleArgumentConstructorsOnly) {
  EXPECT_NO_CHANGES(ExplicitConstructorCheck, "class C { C(); };");
  EXPECT_NO_CHANGES(ExplicitConstructorCheck, "class C { C(int i, int j); };");
}

TEST(ExplicitConstructorCheckTest, Basic) {
  EXPECT_EQ("class C { explicit C(int i); };",
            runCheckOnCode<ExplicitConstructorCheck>("class C { C(int i); };"));
}

TEST(ExplicitConstructorCheckTest, DefaultParameters) {
  EXPECT_EQ("class C { explicit C(int i, int j = 0); };",
            runCheckOnCode<ExplicitConstructorCheck>(
                "class C { C(int i, int j = 0); };"));
}

TEST(ExplicitConstructorCheckTest, OutOfLineDefinitions) {
  EXPECT_EQ("class C { explicit C(int i); }; C::C(int i) {}",
            runCheckOnCode<ExplicitConstructorCheck>(
                "class C { C(int i); }; C::C(int i) {}"));
}

} // namespace test
} // namespace tidy
} // namespace clang
