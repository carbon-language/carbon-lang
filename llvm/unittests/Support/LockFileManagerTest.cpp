//===- unittests/LockFileManagerTest.cpp - LockFileManager tests ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {

TEST(LockFileManagerTest, Basic) {
  SmallString<64> TmpDir;
  error_code EC;
  EC = sys::fs::createUniqueDirectory("LockFileManagerTestDir", TmpDir);
  ASSERT_FALSE(EC);

  SmallString<64> LockedFile(TmpDir);
  sys::path::append(LockedFile, "file.lock");

  {
    // The lock file should not exist, so we should successfully acquire it.
    LockFileManager Locked1(LockedFile);
    EXPECT_EQ(LockFileManager::LFS_Owned, Locked1.getState());

    // Attempting to reacquire the lock should fail.  Waiting on it would cause
    // deadlock, so don't try that.
    LockFileManager Locked2(LockedFile);
    EXPECT_NE(LockFileManager::LFS_Owned, Locked2.getState());
  }

  // Now that the lock is out of scope, the file should be gone.
  EXPECT_FALSE(sys::fs::exists(StringRef(LockedFile)));

  EC = sys::fs::remove(StringRef(TmpDir));
  ASSERT_FALSE(EC);
}

} // end anonymous namespace
