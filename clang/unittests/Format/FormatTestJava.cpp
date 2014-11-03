//===- unittest/Format/FormatTestJava.cpp - Formatting tests for Java -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FormatTestUtils.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {

class FormatTestJava : public ::testing::Test {
protected:
  static std::string format(llvm::StringRef Code, unsigned Offset,
                            unsigned Length, const FormatStyle &Style) {
    DEBUG(llvm::errs() << "---\n");
    DEBUG(llvm::errs() << Code << "\n\n");
    std::vector<tooling::Range> Ranges(1, tooling::Range(Offset, Length));
    tooling::Replacements Replaces = reformat(Style, Code, Ranges);
    std::string Result = applyAllReplacements(Code, Replaces);
    EXPECT_NE("", Result);
    DEBUG(llvm::errs() << "\n" << Result << "\n\n");
    return Result;
  }

  static std::string format(
      llvm::StringRef Code,
      const FormatStyle &Style = getGoogleStyle(FormatStyle::LK_Java)) {
    return format(Code, 0, Code.size(), Style);
  }

  static FormatStyle getStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getGoogleStyle(FormatStyle::LK_Java);
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  static void verifyFormat(
      llvm::StringRef Code,
      const FormatStyle &Style = getGoogleStyle(FormatStyle::LK_Java)) {
    EXPECT_EQ(Code.str(), format(test::messUp(Code), Style));
  }
};

TEST_F(FormatTestJava, ClassDeclarations) {
  verifyFormat("public class SomeClass {\n"
               "  private int a;\n"
               "  private int b;\n"
               "}");
  verifyFormat("public class A {\n"
               "  class B {\n"
               "    int i;\n"
               "  }\n"
               "  class C {\n"
               "    int j;\n"
               "  }\n"
               "}");
  verifyFormat("public class A extends B.C {}");

  verifyFormat("abstract class SomeClass extends SomeOtherClass\n"
               "    implements SomeInterface {}",
               getStyleWithColumns(60));
  verifyFormat("abstract class SomeClass\n"
               "    extends SomeOtherClass\n"
               "    implements SomeInterface {}",
               getStyleWithColumns(40));
  verifyFormat("abstract class SomeClass\n"
               "    extends SomeOtherClass\n"
               "    implements SomeInterface,\n"
               "               AnotherInterface {}",
               getStyleWithColumns(40));
  verifyFormat("@SomeAnnotation()\n"
               "abstract class aaaaaaaaaaaa extends bbbbbbbbbbbbbbb\n"
               "    implements cccccccccccc {\n"
               "}",
               getStyleWithColumns(76));
}

TEST_F(FormatTestJava, EnumDeclarations) {
  verifyFormat("enum SomeThing { ABC, CDE }");
  verifyFormat("enum SomeThing {\n"
               "  ABC,\n"
               "  CDE,\n"
               "}");
  verifyFormat("public class SomeClass {\n"
               "  enum SomeThing { ABC, CDE }\n"
               "  void f() {\n"
               "  }\n"
               "}");
}

TEST_F(FormatTestJava, ThrowsDeclarations) {
  verifyFormat("public void doSooooooooooooooooooooooooooomething()\n"
               "    throws LooooooooooooooooooooooooooooongException {\n}");
}

TEST_F(FormatTestJava, Annotations) {
  verifyFormat("@Override\n"
               "public String toString() {\n}");
  verifyFormat("@Override\n"
               "@Nullable\n"
               "public String getNameIfPresent() {\n}");

  verifyFormat("@SuppressWarnings(value = \"unchecked\")\n"
               "public void doSomething() {\n}");
  verifyFormat("@SuppressWarnings(value = \"unchecked\")\n"
               "@Author(name = \"abc\")\n"
               "public void doSomething() {\n}");

  verifyFormat("DoSomething(new A() {\n"
               "  @Override\n"
               "  public String toString() {\n"
               "  }\n"
               "});");

  verifyFormat("void SomeFunction(@Nullable String something) {\n"
               "}");

  verifyFormat("@Partial @Mock DataLoader loader;");
  verifyFormat("@SuppressWarnings(value = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\")\n"
               "public static int iiiiiiiiiiiiiiiiiiiiiiii;");

  verifyFormat("@SomeAnnotation(\"With some really looooooooooooooong text\")\n"
               "private static final long something = 0L;");
}

TEST_F(FormatTestJava, Generics) {
  verifyFormat("Iterable<?> a;");
  verifyFormat("Iterable<?> a;");
  verifyFormat("Iterable<? extends SomeObject> a;");

  verifyFormat("A.<B>doSomething();");

  verifyFormat("@Override\n"
               "public Map<String, ?> getAll() {\n"
               "}");
}

TEST_F(FormatTestJava, StringConcatenation) {
  verifyFormat("String someString = \"abc\"\n"
               "                    + \"cde\";");
}

TEST_F(FormatTestJava, TryCatchFinally) {
  verifyFormat("try {\n"
               "  Something();\n"
               "} catch (SomeException e) {\n"
               "  HandleException(e);\n"
               "}");
  verifyFormat("try {\n"
               "  Something();\n"
               "} finally {\n"
               "  AlwaysDoThis();\n"
               "}");
  verifyFormat("try {\n"
               "  Something();\n"
               "} catch (SomeException e) {\n"
               "  HandleException(e);\n"
               "} finally {\n"
               "  AlwaysDoThis();\n"
               "}");

  verifyFormat("try {\n"
               "  Something();\n"
               "} catch (SomeException | OtherException e) {\n"
               "  HandleException(e);\n"
               "}");
}

TEST_F(FormatTestJava, SynchronizedKeyword) {
  verifyFormat("synchronized (mData) {\n"
               "  // ...\n"
               "}");
}

TEST_F(FormatTestJava, ImportDeclarations) {
  verifyFormat("import some.really.loooooooooooooooooooooong.imported.Class;",
               getStyleWithColumns(50));
}

} // end namespace tooling
} // end namespace clang
