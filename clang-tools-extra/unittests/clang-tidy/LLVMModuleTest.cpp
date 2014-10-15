#include "ClangTidyTest.h"
#include "llvm/HeaderGuardCheck.h"
#include "llvm/IncludeOrderCheck.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace test {

// FIXME: It seems this might be incompatible to dos path. Investigating.
#if !defined(_WIN32)
static std::string runHeaderGuardCheck(StringRef Code, const Twine &Filename,
                                       unsigned ExpectedWarnings) {
  std::vector<ClangTidyError> Errors;
  std::string Result = test::runCheckOnCode<LLVMHeaderGuardCheck>(
      Code, &Errors, Filename, std::string("-xc++-header"));
  return Errors.size() == ExpectedWarnings ? Result : "invalid error count";
}

namespace {
struct WithEndifComment : public LLVMHeaderGuardCheck {
  WithEndifComment(StringRef Name, ClangTidyContext *Context)
      : LLVMHeaderGuardCheck(Name, Context) {}
  bool shouldSuggestEndifComment(StringRef Filename) override { return true; }
};
} // namespace

static std::string runHeaderGuardCheckWithEndif(StringRef Code,
                                                const Twine &Filename,
                                                unsigned ExpectedWarnings) {
  std::vector<ClangTidyError> Errors;
  std::string Result = test::runCheckOnCode<WithEndifComment>(
      Code, &Errors, Filename, std::string("-xc++-header"));
  return Errors.size() == ExpectedWarnings ? Result : "invalid error count";
}

TEST(LLVMHeaderGuardCheckTest, FixHeaderGuards) {
  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif\n",
            runHeaderGuardCheck("#ifndef FOO\n"
                                "#define FOO\n"
                                "#endif\n",
                                "include/llvm/ADT/foo.h",
                                /*ExpectedWarnings=*/1));

  // Allow trailing underscores.
  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H_\n"
            "#define LLVM_ADT_FOO_H_\n"
            "#endif\n",
            runHeaderGuardCheck("#ifndef LLVM_ADT_FOO_H_\n"
                                "#define LLVM_ADT_FOO_H_\n"
                                "#endif\n",
                                "include/llvm/ADT/foo.h",
                                /*ExpectedWarnings=*/0));

  EXPECT_EQ("#ifndef LLVM_CLANG_C_BAR_H\n"
            "#define LLVM_CLANG_C_BAR_H\n"
            "\n"
            "\n"
            "#endif\n",
            runHeaderGuardCheck("", "./include/clang-c/bar.h",
                                /*ExpectedWarnings=*/1));

  EXPECT_EQ("#ifndef LLVM_CLANG_LIB_CODEGEN_C_H\n"
            "#define LLVM_CLANG_LIB_CODEGEN_C_H\n"
            "\n"
            "\n"
            "#endif\n",
            runHeaderGuardCheck("", "tools/clang/lib/CodeGen/c.h",
                                /*ExpectedWarnings=*/1));

  EXPECT_EQ("#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_X_H\n"
            "#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_X_H\n"
            "\n"
            "\n"
            "#endif\n",
            runHeaderGuardCheck("", "tools/clang/tools/extra/clang-tidy/x.h",
                                /*ExpectedWarnings=*/1));

  EXPECT_EQ("int foo;\n"
            "#ifndef LLVM_CLANG_BAR_H\n"
            "#define LLVM_CLANG_BAR_H\n"
            "#endif\n",
            runHeaderGuardCheck("int foo;\n"
                                "#ifndef LLVM_CLANG_BAR_H\n"
                                "#define LLVM_CLANG_BAR_H\n"
                                "#endif\n",
                                "include/clang/bar.h", /*ExpectedWarnings=*/1));

  EXPECT_EQ("#ifndef LLVM_CLANG_BAR_H\n"
            "#define LLVM_CLANG_BAR_H\n"
            "\n"
            "int foo;\n"
            "#ifndef FOOLOLO\n"
            "#define FOOLOLO\n"
            "#endif\n"
            "\n"
            "#endif\n",
            runHeaderGuardCheck("int foo;\n"
                                "#ifndef FOOLOLO\n"
                                "#define FOOLOLO\n"
                                "#endif\n",
                                "include/clang/bar.h", /*ExpectedWarnings=*/1));

  // Fix incorrect #endif comments even if we shouldn't add new ones.
  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif // LLVM_ADT_FOO_H\n",
            runHeaderGuardCheck("#ifndef FOO\n"
                                "#define FOO\n"
                                "#endif // FOO\n",
                                "include/llvm/ADT/foo.h",
                                /*ExpectedWarnings=*/1));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif // LLVM_ADT_FOO_H\n",
            runHeaderGuardCheckWithEndif("#ifndef FOO\n"
                                         "#define FOO\n"
                                         "#endif\n",
                                         "include/llvm/ADT/foo.h",
                                         /*ExpectedWarnings=*/1));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif // LLVM_ADT_FOO_H\n",
            runHeaderGuardCheckWithEndif("#ifndef LLVM_ADT_FOO_H\n"
                                         "#define LLVM_ADT_FOO_H\n"
                                         "#endif // LLVM_H\n",
                                         "include/llvm/ADT/foo.h",
                                         /*ExpectedWarnings=*/1));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif /* LLVM_ADT_FOO_H */\n",
            runHeaderGuardCheckWithEndif("#ifndef LLVM_ADT_FOO_H\n"
                                         "#define LLVM_ADT_FOO_H\n"
                                         "#endif /* LLVM_ADT_FOO_H */\n",
                                         "include/llvm/ADT/foo.h",
                                         /*ExpectedWarnings=*/0));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H_\n"
            "#define LLVM_ADT_FOO_H_\n"
            "#endif // LLVM_ADT_FOO_H_\n",
            runHeaderGuardCheckWithEndif("#ifndef LLVM_ADT_FOO_H_\n"
                                         "#define LLVM_ADT_FOO_H_\n"
                                         "#endif // LLVM_ADT_FOO_H_\n",
                                         "include/llvm/ADT/foo.h",
                                         /*ExpectedWarnings=*/0));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif // LLVM_ADT_FOO_H\n",
            runHeaderGuardCheckWithEndif("#ifndef LLVM_ADT_FOO_H_\n"
                                         "#define LLVM_ADT_FOO_H_\n"
                                         "#endif // LLVM\n",
                                         "include/llvm/ADT/foo.h",
                                         /*ExpectedWarnings=*/1));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif \\ \n"
            "// LLVM_ADT_FOO_H\n",
            runHeaderGuardCheckWithEndif("#ifndef LLVM_ADT_FOO_H\n"
                                         "#define LLVM_ADT_FOO_H\n"
                                         "#endif \\ \n"
                                         "// LLVM_ADT_FOO_H\n",
                                         "include/llvm/ADT/foo.h",
                                         /*ExpectedWarnings=*/1));

  EXPECT_EQ("#ifndef LLVM_ADT_FOO_H\n"
            "#define LLVM_ADT_FOO_H\n"
            "#endif  /* LLVM_ADT_FOO_H\\ \n"
            " FOO */",
            runHeaderGuardCheckWithEndif("#ifndef LLVM_ADT_FOO_H\n"
                                         "#define LLVM_ADT_FOO_H\n"
                                         "#endif  /* LLVM_ADT_FOO_H\\ \n"
                                         " FOO */",
                                         "include/llvm/ADT/foo.h",
                                         /*ExpectedWarnings=*/0));
}
#endif

} // namespace test
} // namespace tidy
} // namespace clang
