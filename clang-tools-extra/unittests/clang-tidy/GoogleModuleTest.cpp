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
  EXPECT_NO_CHANGES(ExplicitConstructorCheck, "class C { C(const C&); };");
  EXPECT_NO_CHANGES(ExplicitConstructorCheck, "class C { C(C&&); };");
  EXPECT_NO_CHANGES(ExplicitConstructorCheck,
                    "class C { C(const C&) = delete; };");
  EXPECT_NO_CHANGES(ExplicitConstructorCheck,
                    "class C { C(int) = delete; };");
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

TEST(ExplicitConstructorCheckTest, RemoveExplicit) {
  EXPECT_EQ("class A { A(const A&); };\n"
            "class B { /*asdf*/  B(const B&); };\n"
            "class C { /*asdf*/  C(const C&); };",
            runCheckOnCode<ExplicitConstructorCheck>(
                "class A { explicit    A(const A&); };\n"
                "class B { explicit   /*asdf*/  B(const B&); };\n"
                "class C { explicit/*asdf*/  C(const C&); };"));
}

} // namespace test
} // namespace tidy
} // namespace clang
