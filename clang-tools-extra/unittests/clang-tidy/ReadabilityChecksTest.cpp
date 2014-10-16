#include "ClangTidyTest.h"
#include "readability/NamespaceCommentCheck.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace test {

using readability::NamespaceCommentCheck;

TEST(NamespaceCommentCheckTest, Basic) {
  EXPECT_EQ("namespace i {\n} // namespace i",
            runCheckOnCode<NamespaceCommentCheck>("namespace i {\n}"));
  EXPECT_EQ("namespace {\n} // namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n}"));
  EXPECT_EQ(
      "namespace i { namespace j {\n} // namespace j\n } // namespace i",
      runCheckOnCode<NamespaceCommentCheck>("namespace i { namespace j {\n} }"));
}

TEST(NamespaceCommentCheckTest, SingleLineNamespaces) {
  EXPECT_EQ(
      "namespace i { namespace j { } }",
      runCheckOnCode<NamespaceCommentCheck>("namespace i { namespace j { } }"));
}

TEST(NamespaceCommentCheckTest, CheckExistingComments) {
  EXPECT_EQ("namespace i { namespace j {\n"
            "} /* namespace j */ } // namespace i\n"
            " /* random comment */",
            runCheckOnCode<NamespaceCommentCheck>(
                "namespace i { namespace j {\n"
                "} /* namespace j */ } /* random comment */"));
  EXPECT_EQ("namespace {\n"
            "} // namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "} // namespace"));
  EXPECT_EQ("namespace {\n"
            "} //namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "} //namespace"));
  EXPECT_EQ("namespace {\n"
            "} // anonymous namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "} // anonymous namespace"));
  EXPECT_EQ("namespace {\n"
            "} // Anonymous namespace.",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "} // Anonymous namespace."));
  EXPECT_EQ("namespace q {\n"
            "} // namespace q",
            runCheckOnCode<NamespaceCommentCheck>("namespace q {\n"
                                                  "} // anonymous namespace q"));
  EXPECT_EQ(
      "namespace My_NameSpace123 {\n"
      "} // namespace My_NameSpace123",
      runCheckOnCode<NamespaceCommentCheck>("namespace My_NameSpace123 {\n"
                                            "} // namespace My_NameSpace123"));
  EXPECT_EQ(
      "namespace My_NameSpace123 {\n"
      "} //namespace My_NameSpace123",
      runCheckOnCode<NamespaceCommentCheck>("namespace My_NameSpace123 {\n"
                                            "} //namespace My_NameSpace123"));
  EXPECT_EQ("namespace My_NameSpace123 {\n"
            "} //  end namespace   My_NameSpace123",
            runCheckOnCode<NamespaceCommentCheck>(
                "namespace My_NameSpace123 {\n"
                "} //  end namespace   My_NameSpace123"));
  // Understand comments only on the same line.
  EXPECT_EQ("namespace {\n"
            "} // namespace\n"
            "// namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "}\n"
                                                  "// namespace"));
  // Leave unknown comments.
  EXPECT_EQ("namespace {\n"
            "} // namespace // random text",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "} // random text"));
}

TEST(NamespaceCommentCheckTest, FixWrongComments) {
  EXPECT_EQ("namespace i { namespace jJ0_ {\n"
            "} // namespace jJ0_\n"
            " } // namespace i\n"
            " /* random comment */",
            runCheckOnCode<NamespaceCommentCheck>(
                "namespace i { namespace jJ0_ {\n"
                "} /* namespace qqq */ } /* random comment */"));
  EXPECT_EQ("namespace {\n"
            "} // namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "} // namespace asdf"));
}

} // namespace test
} // namespace tidy
} // namespace clang
