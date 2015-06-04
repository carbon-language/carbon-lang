#include "ClangTidyTest.h"
#include "readability/BracesAroundStatementsCheck.h"
#include "readability/NamespaceCommentCheck.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace test {

using readability::BracesAroundStatementsCheck;
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
  // Remove unknown line comments. These are likely to be an unrecognized form
  // of a namespace ending comment.
  EXPECT_EQ("namespace {\n"
            "} // namespace",
            runCheckOnCode<NamespaceCommentCheck>("namespace {\n"
                                                  "} // random text"));
}

TEST(BracesAroundStatementsCheck, IfWithComments) {
  EXPECT_EQ("int main() {\n"
            "  if (false /*dummy token*/) {\n"
            "    // comment\n"
            "    return -1; /**/\n"
            "}\n"
            "  if (false) {\n"
            "    return -1; // comment\n"
            "}\n"
            "  if (false) {\n"
            "    return -1; \n"
            "}/* multi-line \n comment */\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  if (false /*dummy token*/)\n"
                "    // comment\n"
                "    return -1; /**/\n"
                "  if (false)\n"
                "    return -1; // comment\n"
                "  if (false)\n"
                "    return -1; /* multi-line \n comment */\n"
                "}"));
  EXPECT_EQ("int main() {\n"
            "  if (false /*dummy token*/) {\n"
            "    // comment\n"
            "    return -1 /**/ ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  if (false /*dummy token*/)\n"
                "    // comment\n"
                "    return -1 /**/ ;\n"
                "}"));
}

TEST(BracesAroundStatementsCheck, If) {
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  if (false) {\n"
                                                 "    return -1;\n"
                                                 "  }\n"
                                                 "}");
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  if (auto Cond = false) {\n"
                                                 "    return -1;\n"
                                                 "  }\n"
                                                 "}");
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  if (false) {\n"
                                                 "    return -1;\n"
                                                 "  } else {\n"
                                                 "    return -2;\n"
                                                 "  }\n"
                                                 "}");
  EXPECT_EQ("int main() {\n"
            "  if (false) {\n"
            "    return -1;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  if (false)\n"
                                                        "    return -1;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  if (auto Cond = false /**/ ) {\n"
            "    return -1;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  if (auto Cond = false /**/ )\n"
                "    return -1;\n"
                "}"));
  // FIXME: Consider adding braces before EMPTY_MACRO and after the statement.
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck,
                    "#define EMPTY_MACRO\n"
                    "int main() {\n"
                    "  if (auto Cond = false EMPTY_MACRO /**/ ) EMPTY_MACRO\n"
                    "    return -1;\n"
                    "}");
  EXPECT_EQ("int main() {\n"
            "  if (true) { return -1/**/ ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  if (true) return -1/**/ ;\n"
                "}"));
  EXPECT_EQ("int main() {\n"
            "  if (false) {\n"
            "    return -1;\n"
            "  } else {\n"
            "    return -2;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  if (false)\n"
                                                        "    return -1;\n"
                                                        "  else\n"
                                                        "    return -2;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  if (false) {\n"
            "    return -1;\n"
            "  } else if (1 == 2) {\n"
            "    return -2;\n"
            "  } else {\n"
            "    return -3;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  if (false)\n"
                                                        "    return -1;\n"
                                                        "  else if (1 == 2)\n"
                                                        "    return -2;\n"
                                                        "  else\n"
                                                        "    return -3;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  if (false) {\n"
            "    return -1;\n"
            "  } else if (1 == 2) {\n"
            "    return -2;\n"
            "  } else {\n"
            "    return -3;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  if (false)\n"
                                                        "    return -1;\n"
                                                        "  else if (1 == 2) {\n"
                                                        "    return -2;\n"
                                                        "  } else\n"
                                                        "    return -3;\n"
                                                        "}"));
}

TEST(BracesAroundStatementsCheck, IfElseWithShortStatements) {
  ClangTidyOptions Options;
  Options.CheckOptions["test-check.ShortStatementLines"] = "1";

  EXPECT_EQ("int main() {\n"
            "  if (true) return 1;\n"
            "  if (false) { return -1;\n"
            "  } else if (1 == 2) { return -2;\n"
            "  } else { return -3;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  if (true) return 1;\n"
                "  if (false) return -1;\n"
                "  else if (1 == 2) return -2;\n"
                "  else return -3;\n"
                "}",
                nullptr, "input.cc", None, Options));

  // If the last else is an else-if, we also force it.
  EXPECT_EQ("int main() {\n"
            "  if (false) { return -1;\n"
            "  } else if (1 == 2) { return -2;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  if (false) return -1;\n"
                "  else if (1 == 2) return -2;\n"
                "}",
                nullptr, "input.cc", None, Options));
}

TEST(BracesAroundStatementsCheck, For) {
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  for (;;) {\n"
                                                 "    ;\n"
                                                 "  }\n"
                                                 "}");
  EXPECT_EQ("int main() {\n"
            "  for (;;) {\n"
            "    ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  for (;;)\n"
                                                        "    ;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  for (;;) {\n"
            "    /**/ ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  for (;;)\n"
                                                        "    /**/ ;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  for (;;) {\n"
            "    return -1 /**/ ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  for (;;)\n"
                                                        "    return -1 /**/ ;\n"
                                                        "}"));
}

TEST(BracesAroundStatementsCheck, ForRange) {
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  int arr[4];\n"
                                                 "  for (int i : arr) {\n"
                                                 "    ;\n"
                                                 "  }\n"
                                                 "}");
  EXPECT_EQ("int main() {\n"
            "  int arr[4];\n"
            "  for (int i : arr) {\n"
            "    ;\n"
            "}\n"
            "  for (int i : arr) {\n"
            "    return -1 ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  int arr[4];\n"
                                                        "  for (int i : arr)\n"
                                                        "    ;\n"
                                                        "  for (int i : arr)\n"
                                                        "    return -1 ;\n"
                                                        "}"));
}

TEST(BracesAroundStatementsCheck, DoWhile) {
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  do {\n"
                                                 "    ;\n"
                                                 "  } while (false);\n"
                                                 "}");
  EXPECT_EQ("int main() {\n"
            "  do {\n"
            "    ;\n"
            "  } while (false);\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  do\n"
                                                        "    ;\n"
                                                        "  while (false);\n"
                                                        "}"));
}

TEST(BracesAroundStatementsCheck, While) {
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck, "int main() {\n"
                                                 "  while (false) {\n"
                                                 "    ;\n"
                                                 "  }\n"
                                                 "}");
  EXPECT_EQ("int main() {\n"
            "  while (false) {\n"
            "    ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  while (false)\n"
                                                        "    ;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  while (auto Cond = false) {\n"
            "    ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  while (auto Cond = false)\n"
                "    ;\n"
                "}"));
  EXPECT_EQ("int main() {\n"
            "  while (false /*dummy token*/) {\n"
            "    ;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  while (false /*dummy token*/)\n"
                "    ;\n"
                "}"));
  EXPECT_EQ("int main() {\n"
            "  while (false) {\n"
            "    break;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  while (false)\n"
                                                        "    break;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  while (false) {\n"
            "    break /**/;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  while (false)\n"
                                                        "    break /**/;\n"
                                                        "}"));
  EXPECT_EQ("int main() {\n"
            "  while (false) {\n"
            "    /**/;\n"
            "}\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                        "  while (false)\n"
                                                        "    /**/;\n"
                                                        "}"));
}

TEST(BracesAroundStatementsCheck, Nested) {
  EXPECT_EQ("int main() {\n"
            "  do { if (true) {}} while (false);\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  do if (true) {}while (false);\n"
                "}"));
  EXPECT_EQ("int main() {\n"
            "  do { if (true) {}} while (false);\n"
            "}",
            runCheckOnCode<BracesAroundStatementsCheck>(
                "int main() {\n"
                "  do if (true) {}while (false);\n"
                "}"));
  EXPECT_EQ(
      "int main() {\n"
      "  if (true) {\n"
      "    // comment\n"
      "    if (false) {\n"
      "      // comment\n"
      "      /**/ ; // comment\n"
      "    }\n"
      "}\n"
      "}",
      runCheckOnCode<BracesAroundStatementsCheck>("int main() {\n"
                                                  "  if (true)\n"
                                                  "    // comment\n"
                                                  "    if (false) {\n"
                                                  "      // comment\n"
                                                  "      /**/ ; // comment\n"
                                                  "    }\n"
                                                  "}"));
}

TEST(BracesAroundStatementsCheck, Macros) {
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck,
                    "#define IF(COND) if (COND) return -1;\n"
                    "int main() {\n"
                    "  IF(false)\n"
                    "}");
  EXPECT_NO_CHANGES(BracesAroundStatementsCheck,
                    "#define FOR(COND) for (COND) return -1;\n"
                    "int main() {\n"
                    "  FOR(;;)\n"
                    "}");
}

} // namespace test
} // namespace tidy
} // namespace clang
