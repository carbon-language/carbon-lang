#include "ClangTidyTest.h"

#include "google/GoogleTidyModule.h"

namespace clang {
namespace tidy {

typedef ClangTidyTest<ExplicitConstructorCheck> ExplicitConstructorCheckTest;

TEST_F(ExplicitConstructorCheckTest, SingleArgumentConstructorsOnly) {
  expectNoChanges("class C { C(); };");
  expectNoChanges("class C { C(int i, int j); };");
}

TEST_F(ExplicitConstructorCheckTest, Basic) {
  EXPECT_EQ("class C { explicit C(int i); };",
            runCheckOn("class C { C(int i); };"));
}

TEST_F(ExplicitConstructorCheckTest, DefaultParameters) {
  EXPECT_EQ("class C { explicit C(int i, int j = 0); };",
            runCheckOn("class C { C(int i, int j = 0); };"));
}

} // namespace tidy
} // namespace clang
