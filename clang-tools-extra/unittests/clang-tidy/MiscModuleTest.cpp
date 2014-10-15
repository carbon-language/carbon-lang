#include "ClangTidyTest.h"
#include "misc/ArgumentCommentCheck.h"
#include "readability/BracesAroundStatementsCheck.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {

using readability::BracesAroundStatementsCheck;

namespace test {

#define EXPECT_NO_CHANGES(Check, Code)                                         \
  EXPECT_EQ(Code, runCheckOnCode<Check>(Code))

TEST(ArgumentCommentCheckTest, CorrectComments) {
  EXPECT_NO_CHANGES(ArgumentCommentCheck,
                    "void f(int x, int y); void g() { f(/*x=*/0, /*y=*/0); }");
  EXPECT_NO_CHANGES(ArgumentCommentCheck,
                    "struct C { C(int x, int y); }; C c(/*x=*/0, /*y=*/0);");
}

TEST(ArgumentCommentCheckTest, ThisEditDistanceAboveThreshold) {
  EXPECT_NO_CHANGES(ArgumentCommentCheck,
                    "void f(int xxx); void g() { f(/*xyz=*/0); }");
}

TEST(ArgumentCommentCheckTest, OtherEditDistanceAboveThreshold) {
  EXPECT_EQ("void f(int xxx, int yyy); void g() { f(/*xxx=*/0, 0); }",
            runCheckOnCode<ArgumentCommentCheck>(
                "void f(int xxx, int yyy); void g() { f(/*Xxx=*/0, 0); }"));
  EXPECT_EQ("struct C { C(int xxx, int yyy); }; C c(/*xxx=*/0, 0);",
            runCheckOnCode<ArgumentCommentCheck>(
                "struct C { C(int xxx, int yyy); }; C c(/*Xxx=*/0, 0);"));
}

TEST(ArgumentCommentCheckTest, OtherEditDistanceBelowThreshold) {
  EXPECT_NO_CHANGES(ArgumentCommentCheck,
                    "void f(int xxx, int yyy); void g() { f(/*xxy=*/0, 0); }");
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
