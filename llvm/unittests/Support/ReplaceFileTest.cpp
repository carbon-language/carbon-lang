//===- llvm/unittest/Support/ReplaceFileTest.cpp - unit tests -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::sys;

#define ASSERT_NO_ERROR(x)                                                 \
  do {                                                                     \
    if (std::error_code ASSERT_NO_ERROR_ec = x) {                          \
      errs() << #x ": did not return errc::success.\n"                     \
             << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"     \
             << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n"; \
    }                                                                      \
  } while (false)

namespace {
std::error_code CreateFileWithContent(const SmallString<128> &FilePath,
                                      const StringRef &content) {
  int FD = 0;
  if (std::error_code ec = fs::openFileForWrite(FilePath, FD, fs::F_None))
    return ec;

  const bool ShouldClose = true;
  raw_fd_ostream OS(FD, ShouldClose);
  OS << content;

  return std::error_code();
}

class ScopedFD {
  int FD;

  ScopedFD(const ScopedFD &) = delete;
  ScopedFD &operator=(const ScopedFD &) = delete;

 public:
  explicit ScopedFD(int Descriptor) : FD(Descriptor) {}
  ~ScopedFD() { Process::SafelyCloseFileDescriptor(FD); }
};

TEST(rename, FileOpenedForReadingCanBeReplaced) {
  // Create unique temporary directory for this test.
  SmallString<128> TestDirectory;
  ASSERT_NO_ERROR(fs::createUniqueDirectory(
      "FileOpenedForReadingCanBeReplaced-test", TestDirectory));

  // Add a couple of files to the test directory.
  SmallString<128> SourceFileName(TestDirectory);
  path::append(SourceFileName, "source");

  SmallString<128> TargetFileName(TestDirectory);
  path::append(TargetFileName, "target");

  ASSERT_NO_ERROR(CreateFileWithContent(SourceFileName, "!!source!!"));
  ASSERT_NO_ERROR(CreateFileWithContent(TargetFileName, "!!target!!"));

  {
    // Open the target file for reading.
    int ReadFD = 0;
    ASSERT_NO_ERROR(fs::openFileForRead(TargetFileName, ReadFD));
    ScopedFD EventuallyCloseIt(ReadFD);

    // Confirm we can replace the file while it is open.
    EXPECT_TRUE(!fs::rename(SourceFileName, TargetFileName));

    // We should still be able to read the old data through the existing
    // descriptor.
    auto Buffer = MemoryBuffer::getOpenFile(ReadFD, TargetFileName, -1);
    ASSERT_TRUE(static_cast<bool>(Buffer));
    EXPECT_EQ(Buffer.get()->getBuffer(), "!!target!!");

    // The source file should no longer exist
    EXPECT_FALSE(fs::exists(SourceFileName));
  }

  {
    // If we obtain a new descriptor for the target file, we should find that it
    // contains the content that was in the source file.
    int ReadFD = 0;
    ASSERT_NO_ERROR(fs::openFileForRead(TargetFileName, ReadFD));
    ScopedFD EventuallyCloseIt(ReadFD);
    auto Buffer = MemoryBuffer::getOpenFile(ReadFD, TargetFileName, -1);
    ASSERT_TRUE(static_cast<bool>(Buffer));

    EXPECT_EQ(Buffer.get()->getBuffer(), "!!source!!");
  }

  // Rename the target file back to the source file name to confirm that rename
  // still works if the destination does not already exist.
  EXPECT_TRUE(!fs::rename(TargetFileName, SourceFileName));
  EXPECT_FALSE(fs::exists(TargetFileName));
  ASSERT_TRUE(fs::exists(SourceFileName));

  // Clean up.
  ASSERT_NO_ERROR(fs::remove(SourceFileName));
  ASSERT_NO_ERROR(fs::remove(TestDirectory.str()));
}

}  // anonymous namespace
