//===- unittests/cpp11-migrate/UniqueHeaderNameTest.cpp -------------------===//
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
#include "Core/FileOverrides.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/system_error.h"

TEST(UniqueHeaderName, testUniqueHeaderName) {
  using namespace llvm::sys::path;

  llvm::SmallString<32> TmpDir;
  system_temp_directory(true, TmpDir);

  llvm::SmallString<128> SourceFile(TmpDir);
  append(SourceFile, "project/lib/feature.cpp");
  native(SourceFile.c_str(), SourceFile);

  llvm::SmallString<128> HeaderFile(TmpDir);
  append(HeaderFile, "project/include/feature.h");
  native(HeaderFile.c_str(), HeaderFile);

  llvm::SmallString<128> ExpectedName("^feature.cpp_feature.h_[0-9a-f]{2}_[0-9a-f]{2}_[0-9a-f]{2}_[0-9a-f]{2}_[0-9a-f]{2}_[0-9a-f]{2}.yaml$");

  llvm::SmallString<128> ActualName;
  llvm::SmallString<128> Error;
  bool Result =
      generateReplacementsFileName(SourceFile, HeaderFile, ActualName, Error);

  ASSERT_TRUE(Result);
  EXPECT_TRUE(Error.empty());

  // We need to check the directory name and filename separately since on
  // Windows, the path separator is '\' which is a regex escape character.
  llvm::SmallString<128> ExpectedHeaderPath =
      llvm::sys::path::parent_path(HeaderFile);
  llvm::SmallString<128> ActualHeaderPath =
      llvm::sys::path::parent_path(ActualName);
  llvm::SmallString<128> ActualHeaderName =
      llvm::sys::path::filename(ActualName);

  EXPECT_STREQ(ExpectedHeaderPath.c_str(), ActualHeaderPath.c_str());

  llvm::Regex R(ExpectedName);
  ASSERT_TRUE(R.match(ActualHeaderName))
      << "ExpectedName: " << ExpectedName.c_str()
      << "\nActualName: " << ActualName.c_str();
  ASSERT_TRUE(Error.empty()) << "Error: " << Error.c_str();
}
