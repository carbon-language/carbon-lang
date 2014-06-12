//===- unittests/clang-modernize/UniqueHeaderNameTest.cpp -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Test for the generateReplacementsFileName() in FileOverrides.h
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "Core/ReplacementHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include <system_error>

TEST(UniqueHeaderName, testUniqueHeaderName) {
  using namespace llvm::sys::path;

  llvm::SmallString<32> TmpDir;
  system_temp_directory(true, TmpDir);

  llvm::SmallString<128> SourceFile(TmpDir);
  append(SourceFile, "project/lib/feature.cpp");
  native(SourceFile.str().str(), SourceFile);

  llvm::SmallString<128> DestDir(TmpDir);
  append(DestDir, "replacements");

  llvm::SmallString<128> FullActualPath;
  llvm::SmallString<128> Error;
  bool Result = ReplacementHandling::generateReplacementsFileName(
      DestDir, SourceFile, FullActualPath, Error);

  ASSERT_TRUE(Result);
  EXPECT_TRUE(Error.empty());

  // We need to check the directory name and filename separately since on
  // Windows, the path separator is '\' which is a regex escape character.
  llvm::SmallString<128> ExpectedPath =
      llvm::sys::path::parent_path(SourceFile);
  llvm::SmallString<128> ActualPath =
      llvm::sys::path::parent_path(FullActualPath);
  llvm::SmallString<128> ActualName =
      llvm::sys::path::filename(FullActualPath);

  EXPECT_STREQ(DestDir.c_str(), ActualPath.c_str());

  llvm::StringRef ExpectedName =
      "^feature.cpp_[0-9a-f]{2}_[0-9a-f]{2}_[0-9a-f]{2}_[0-9a-f]{2}_["
      "0-9a-f]{2}_[0-9a-f]{2}.yaml$";
  llvm::Regex R(ExpectedName);
  ASSERT_TRUE(R.match(ActualName))
      << "ExpectedName: " << ExpectedName.data()
      << "\nActualName: " << ActualName.c_str();
  ASSERT_TRUE(Error.empty()) << "Error: " << Error.c_str();
}
