// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/source/source_buffer.h"

#include <gtest/gtest.h>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon::Testing {
namespace {

TEST(SourceBufferTest, StringRep) {
  auto buffer = SourceBuffer::CreateFromText(llvm::Twine("Hello") + " World");
  ASSERT_TRUE(static_cast<bool>(buffer));

  EXPECT_EQ("/text", buffer->Filename());
  EXPECT_EQ("Hello World", buffer->Text());
}

TEST(SourceBufferTest, CustomFilename) {
  auto buffer = SourceBuffer::CreateFromTextWithFilename("Hello World Again!",
                                                         "/custom/text");
  ASSERT_TRUE(static_cast<bool>(buffer));
  EXPECT_EQ("/custom/text", buffer->Filename());
  EXPECT_EQ("Hello World Again!", buffer->Text());
}

TEST(SourceBufferTest, UnprintableFilename) {
  auto buffer = SourceBuffer::CreateFromTextWithFilename(
      "Unused", llvm::StringRef("\0", 1));
  ASSERT_FALSE(static_cast<bool>(buffer));
}

auto CreateTestFile(llvm::StringRef text) -> std::string {
  int fd = -1;
  llvm::SmallString<1024> path;
  auto error_code =
      llvm::sys::fs::createTemporaryFile("test_file", ".txt", fd, path);
  if (error_code) {
    llvm::report_fatal_error(llvm::Twine("Failed to create temporary file: ") +
                             error_code.message());
  }

  llvm::raw_fd_ostream out_stream(fd, /*shouldClose=*/true);
  out_stream << text;
  out_stream.close();

  return path.str().str();
}

TEST(SourceBufferTest, FileRep) {
  auto test_file_path = CreateTestFile("Hello World");

  auto expected_buffer = SourceBuffer::CreateFromFile(test_file_path);
  ASSERT_TRUE(static_cast<bool>(expected_buffer))
      << "Error message: " << toString(expected_buffer.takeError());

  SourceBuffer& buffer = *expected_buffer;

  EXPECT_EQ(test_file_path, buffer.Filename());
  EXPECT_EQ("Hello World", buffer.Text());
}

}  // namespace
}  // namespace Carbon::Testing
