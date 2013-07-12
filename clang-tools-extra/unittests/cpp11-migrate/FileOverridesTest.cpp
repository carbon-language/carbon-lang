//===- cpp11-migrate/FileOverridesTest.cpp - File overrides unit tests ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Core/FileOverrides.h"
#include "gtest/gtest.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"

using namespace llvm;
using namespace clang;

TEST(SourceOverridesTest, Interface) {
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts(
      new DiagnosticOptions());
  DiagnosticsEngine Diagnostics(
      llvm::IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
      DiagOpts.getPtr());
  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);
  StringRef FileName = "<text>";
  StringRef Code =
      "std::vector<such_a_long_name_for_a_type>::const_iterator long_type =\n"
      "    vec.begin();\n"
      "int   x;"; // to test that it's not the whole file that is reformatted
  llvm::MemoryBuffer *Buf = llvm::MemoryBuffer::getMemBuffer(Code, FileName);
  const clang::FileEntry *Entry =
      Files.getVirtualFile(FileName, Buf->getBufferSize(), 0);
  SM.overrideFileContents(Entry, Buf);

  SourceOverrides Overrides(FileName);

  EXPECT_EQ(FileName, Overrides.getMainFileName());
  EXPECT_FALSE(Overrides.isSourceOverriden());

  tooling::Replacements Replaces;
  unsigned ReplacementLength =
      strlen("std::vector<such_a_long_name_for_a_type>::const_iterator");
  Replaces.insert(
      tooling::Replacement(FileName, 0, ReplacementLength, "auto"));
  Overrides.applyReplacements(Replaces, SM);
  EXPECT_TRUE(Overrides.isSourceOverriden());

  std::string ExpectedContent = "auto long_type =\n"
                                "    vec.begin();\n"
                                "int   x;";

  EXPECT_EQ(ExpectedContent, Overrides.getMainFileContent());
}
