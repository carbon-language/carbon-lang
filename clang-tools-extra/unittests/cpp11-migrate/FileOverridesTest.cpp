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

// Test fixture object that setup some files once for all test cases and remove
// them when the tests are done.
class SourceOverridesTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    DiagOpts =
        new IntrusiveRefCntPtr<DiagnosticOptions>(new DiagnosticOptions());
    Diagnostics = new DiagnosticsEngine(
        IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
        DiagOpts->getPtr());
  }

  static void TearDownTestCase() {
    delete DiagOpts;
    delete Diagnostics;
  }

  virtual void SetUp() {
    Files = new FileManager(FileSystemOptions());
    Sources = NULL;
    FileName = NULL;
    Code = NULL;
  }

  void setFilename(const char *F) { FileName = F; }
  void setCode(const char *C) { Code = C; }

  virtual void TearDown() {
    delete Files;
    delete Sources;
  }

  // Creates a new SourceManager with the virtual file and content
  SourceManager &getNewSourceManager() {
    assert(FileName && Code && "expected Code and FileName to be set.");
    delete Sources;
    Sources = new SourceManager(*Diagnostics, *Files);
    MemoryBuffer *Buf = MemoryBuffer::getMemBuffer(Code, FileName);
    const FileEntry *Entry = Files->getVirtualFile(
        FileName, Buf->getBufferSize(), /*ModificationTime=*/0);
    Sources->overrideFileContents(Entry, Buf);
    return *Sources;
  }

  static SourceManager *Sources;
  static const char *FileName;
  static const char *Code;

private:
  static IntrusiveRefCntPtr<DiagnosticOptions> *DiagOpts;
  static DiagnosticsEngine *Diagnostics;
  static FileManager *Files;
};

IntrusiveRefCntPtr<DiagnosticOptions> *SourceOverridesTest::DiagOpts = NULL;
DiagnosticsEngine *SourceOverridesTest::Diagnostics = NULL;
FileManager *SourceOverridesTest::Files = NULL;
SourceManager *SourceOverridesTest::Sources = NULL;
const char *SourceOverridesTest::FileName;
const char *SourceOverridesTest::Code;

TEST_F(SourceOverridesTest, Interface) {
  setFilename("<test-file>");
  setCode(
      "std::vector<such_a_long_name_for_a_type>::const_iterator long_type =\n"
      "    vec.begin();\n");
  SourceOverrides Overrides(FileName);

  EXPECT_EQ(FileName, Overrides.getMainFileName());
  EXPECT_FALSE(Overrides.isSourceOverriden());

  tooling::Replacements Replaces;
  unsigned ReplacementLength =
      strlen("std::vector<such_a_long_name_for_a_type>::const_iterator");
  Replaces.insert(
      tooling::Replacement(FileName, 0, ReplacementLength, "auto"));
  Overrides.applyReplacements(Replaces, getNewSourceManager());
  EXPECT_TRUE(Overrides.isSourceOverriden());

  std::string ExpectedContent = "auto long_type =\n"
                                "    vec.begin();\n";

  EXPECT_EQ(ExpectedContent, Overrides.getMainFileContent());
}
