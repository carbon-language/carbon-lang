//===- unittests/Interpreter/InterpreterTest.cpp --- Interpreter tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Clang's Interpreter library.
//
//===----------------------------------------------------------------------===//

#include "clang/Interpreter/Interpreter.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"

#include "llvm/ADT/ArrayRef.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;

namespace {
using Args = std::vector<const char *>;
static std::unique_ptr<Interpreter>
createInterpreter(const Args &ExtraArgs = {},
                  DiagnosticConsumer *Client = nullptr) {
  Args ClangArgs = {"-Xclang", "-emit-llvm-only"};
  ClangArgs.insert(ClangArgs.end(), ExtraArgs.begin(), ExtraArgs.end());
  auto CI = cantFail(clang::IncrementalCompilerBuilder::create(ClangArgs));
  if (Client)
    CI->getDiagnostics().setClient(Client, /*ShouldOwnClient=*/false);
  return cantFail(clang::Interpreter::create(std::move(CI)));
}

static size_t DeclsSize(TranslationUnitDecl *PTUDecl) {
  return std::distance(PTUDecl->decls().begin(), PTUDecl->decls().end());
}

TEST(InterpreterTest, Sanity) {
  std::unique_ptr<Interpreter> Interp = createInterpreter();

  using PTU = PartialTranslationUnit;

  PTU &R1(cantFail(Interp->Parse("void g(); void g() {}")));
  EXPECT_EQ(2U, DeclsSize(R1.TUPart));

  PTU &R2(cantFail(Interp->Parse("int i;")));
  EXPECT_EQ(1U, DeclsSize(R2.TUPart));
}

static std::string DeclToString(Decl *D) {
  return llvm::cast<NamedDecl>(D)->getQualifiedNameAsString();
}

TEST(InterpreterTest, IncrementalInputTopLevelDecls) {
  std::unique_ptr<Interpreter> Interp = createInterpreter();
  auto R1 = Interp->Parse("int var1 = 42; int f() { return var1; }");
  // gtest doesn't expand into explicit bool conversions.
  EXPECT_TRUE(!!R1);
  auto R1DeclRange = R1->TUPart->decls();
  EXPECT_EQ(2U, DeclsSize(R1->TUPart));
  EXPECT_EQ("var1", DeclToString(*R1DeclRange.begin()));
  EXPECT_EQ("f", DeclToString(*(++R1DeclRange.begin())));

  auto R2 = Interp->Parse("int var2 = f();");
  EXPECT_TRUE(!!R2);
  auto R2DeclRange = R2->TUPart->decls();
  EXPECT_EQ(1U, DeclsSize(R2->TUPart));
  EXPECT_EQ("var2", DeclToString(*R2DeclRange.begin()));
}

TEST(InterpreterTest, Errors) {
  Args ExtraArgs = {"-Xclang", "-diagnostic-log-file", "-Xclang", "-"};

  // Create the diagnostic engine with unowned consumer.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  auto DiagPrinter = std::make_unique<TextDiagnosticPrinter>(
      DiagnosticsOS, new DiagnosticOptions());

  auto Interp = createInterpreter(ExtraArgs, DiagPrinter.get());
  auto Err = Interp->Parse("intentional_error v1 = 42; ").takeError();
  using ::testing::HasSubstr;
  EXPECT_THAT(DiagnosticsOS.str(),
              HasSubstr("error: unknown type name 'intentional_error'"));
  EXPECT_EQ("Parsing failed.", llvm::toString(std::move(Err)));

  auto RecoverErr = Interp->Parse("int var1 = 42;");
  EXPECT_TRUE(!!RecoverErr);
}

// Here we test whether the user can mix declarations and statements. The
// interpreter should be smart enough to recognize the declarations from the
// statements and wrap the latter into a declaration, producing valid code.
TEST(InterpreterTest, DeclsAndStatements) {
  Args ExtraArgs = {"-Xclang", "-diagnostic-log-file", "-Xclang", "-"};

  // Create the diagnostic engine with unowned consumer.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  auto DiagPrinter = std::make_unique<TextDiagnosticPrinter>(
      DiagnosticsOS, new DiagnosticOptions());

  auto Interp = createInterpreter(ExtraArgs, DiagPrinter.get());
  auto R1 = Interp->Parse(
      "int var1 = 42; extern \"C\" int printf(const char*, ...);");
  // gtest doesn't expand into explicit bool conversions.
  EXPECT_TRUE(!!R1);

  auto *PTU1 = R1->TUPart;
  EXPECT_EQ(2U, DeclsSize(PTU1));

  // FIXME: Add support for wrapping and running statements.
  auto R2 = Interp->Parse("var1++; printf(\"var1 value %d\\n\", var1);");
  EXPECT_FALSE(!!R2);
  using ::testing::HasSubstr;
  EXPECT_THAT(DiagnosticsOS.str(),
              HasSubstr("error: unknown type name 'var1'"));
  auto Err = R2.takeError();
  EXPECT_EQ("Parsing failed.", llvm::toString(std::move(Err)));
}

} // end anonymous namespace
