#include "ClangTidyTest.h"
#include "llvm/HeaderGuardCheck.h"
#include "llvm/IncludeOrderCheck.h"
#include "llvm/NamespaceCommentCheck.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace test {

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

// FIXME: It seems this might be incompatible to dos path. Investigating.
#if !defined(_WIN32)
static std::string runHeaderGuardCheck(StringRef Code, const Twine &Filename) {
  return test::runCheckOnCode<LLVMHeaderGuardCheck>(
      Code, /*Errors=*/nullptr, Filename, std::string("-xc++-header"));
}

namespace {
struct WithEndifComment : public LLVMHeaderGuardCheck {
  bool shouldSuggestEndifComment(StringRef Filename) override { return true; }
};
} // namespace

static std::string runHeaderGuardCheckWithEndif(StringRef Code,
                                                const Twine &Filename) {
  return test::runCheckOnCode<WithEndifComment>(
      Code, /*Errors=*/nullptr, Filename, std::string("-xc++-header"));
}

TEST(LLVMHeaderGuardCheckTest, FixHeaderGuards) {
  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n#define LLVM_ADT_FOO_H\n#endif\n",
            runHeaderGuardCheck("#ifndef FOO\n#define FOO\n#endif\n",
                                "include/llvm/ADT/foo.h"));

  // Allow trailing underscores.
  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H_\n#define LLVM_ADT_FOO_H_\n#endif\n",
            runHeaderGuardCheck(
                "#ifndef LLVM_ADT_FOO_H_\n#define LLVM_ADT_FOO_H_\n#endif\n",
                "include/llvm/ADT/foo.h"));

  EXPECT_EQ("#ifndef LLVM_CLANG_C_BAR_H\n#define LLVM_CLANG_C_BAR_H\n\n\n#endif\n",
            runHeaderGuardCheck("", "./include/clang-c/bar.h"));

  EXPECT_EQ("#ifndef LLVM_CLANG_LIB_CODEGEN_C_H\n#define "
            "LLVM_CLANG_LIB_CODEGEN_C_H\n\n\n#endif\n",
            runHeaderGuardCheck("", "tools/clang/lib/CodeGen/c.h"));

  EXPECT_EQ("#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_X_H\n#define "
            "LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_X_H\n\n\n#endif\n",
            runHeaderGuardCheck("", "tools/clang/tools/extra/clang-tidy/x.h"));

  EXPECT_EQ(
      "int foo;\n#ifndef LLVM_CLANG_BAR_H\n#define LLVM_CLANG_BAR_H\n#endif\n",
      runHeaderGuardCheck("int foo;\n#ifndef LLVM_CLANG_BAR_H\n"
                          "#define LLVM_CLANG_BAR_H\n#endif\n",
                          "include/clang/bar.h"));

  EXPECT_EQ("#ifndef LLVM_CLANG_BAR_H\n#define LLVM_CLANG_BAR_H\n\n"
            "int foo;\n#ifndef FOOLOLO\n#define FOOLOLO\n#endif\n\n#endif\n",
            runHeaderGuardCheck(
                "int foo;\n#ifndef FOOLOLO\n#define FOOLOLO\n#endif\n",
                "include/clang/bar.h"));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n#define LLVM_ADT_FOO_H\n#endif "
            " // LLVM_ADT_FOO_H\n",
            runHeaderGuardCheckWithEndif("#ifndef FOO\n#define FOO\n#endif\n",
                                         "include/llvm/ADT/foo.h"));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n#define LLVM_ADT_FOO_H\n#endif "
            " // LLVM_ADT_FOO_H\n",
            runHeaderGuardCheckWithEndif("#ifndef LLVM_ADT_FOO_H\n#define "
                                         "LLVM_ADT_FOO_H\n#endif // LLVM_H\n",
                                         "include/llvm/ADT/foo.h"));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n#define LLVM_ADT_FOO_H\n#endif"
            " /* LLVM_ADT_FOO_H */\n",
            runHeaderGuardCheckWithEndif("#ifndef LLVM_ADT_FOO_H\n#define "
                                         "LLVM_ADT_FOO_H\n"
                                         "#endif /* LLVM_ADT_FOO_H */\n",
                                         "include/llvm/ADT/foo.h"));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H_\n#define LLVM_ADT_FOO_H_\n#endif "
            "// LLVM_ADT_FOO_H_\n",
            runHeaderGuardCheckWithEndif(
                "#ifndef LLVM_ADT_FOO_H_\n#define "
                "LLVM_ADT_FOO_H_\n#endif // LLVM_ADT_FOO_H_\n",
                "include/llvm/ADT/foo.h"));

  EXPECT_EQ(
      "#ifndef LLVM_ADT_FOO_H\n#define LLVM_ADT_FOO_H\n#endif  // "
      "LLVM_ADT_FOO_H\n",
      runHeaderGuardCheckWithEndif(
          "#ifndef LLVM_ADT_FOO_H_\n#define LLVM_ADT_FOO_H_\n#endif // LLVM\n",
          "include/llvm/ADT/foo.h"));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n#define LLVM_ADT_FOO_H\n#endif \\ \n// "
            "LLVM_ADT_FOO_H\n",
            runHeaderGuardCheckWithEndif("#ifndef LLVM_ADT_FOO_H\n#define "
                                         "LLVM_ADT_FOO_H\n#endif \\ \n// "
                                         "LLVM_ADT_FOO_H\n",
                                         "include/llvm/ADT/foo.h"));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n#define LLVM_ADT_FOO_H\n#endif  /* "
            "LLVM_ADT_FOO_H\\ \n FOO */",
            runHeaderGuardCheckWithEndif(
                "#ifndef LLVM_ADT_FOO_H\n#define LLVM_ADT_FOO_H\n#endif  /* "
                "LLVM_ADT_FOO_H\\ \n FOO */",
                "include/llvm/ADT/foo.h"));
}
#endif

} // namespace test
} // namespace tidy
} // namespace clang
