//===-- FSTests.cpp - File system related tests -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FS.h"
#include "TestFS.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(FSTests, PreambleStatusCache) {
  llvm::StringMap<std::string> Files;
  Files["x"] = "";
  Files["y"] = "";
  Files["main"] = "";
  auto FS = buildTestFS(Files);

  PreambleFileStatusCache StatCache(testPath("main"));
  auto ProduceFS = StatCache.getProducingFS(FS);
  EXPECT_TRUE(ProduceFS->openFileForRead("x"));
  EXPECT_TRUE(ProduceFS->status("y"));
  EXPECT_TRUE(ProduceFS->status("main"));

  EXPECT_TRUE(StatCache.lookup(testPath("x")).hasValue());
  EXPECT_TRUE(StatCache.lookup(testPath("y")).hasValue());
  // Main file is not cached.
  EXPECT_FALSE(StatCache.lookup(testPath("main")).hasValue());

  llvm::vfs::Status S("fake", llvm::sys::fs::UniqueID(123, 456),
                      std::chrono::system_clock::now(), 0, 0, 1024,
                      llvm::sys::fs::file_type::regular_file,
                      llvm::sys::fs::all_all);
  StatCache.update(*FS, S);
  auto ConsumeFS = StatCache.getConsumingFS(FS);
  auto Cached = ConsumeFS->status(testPath("fake"));
  EXPECT_TRUE(Cached);
  EXPECT_EQ(Cached->getName(), testPath("fake"));
  EXPECT_EQ(Cached->getUniqueID(), S.getUniqueID());

  // fake and temp/../fake should hit the same cache entry.
  // However, the Status returned reflects the actual path requested.
  auto CachedDotDot = ConsumeFS->status(testPath("temp/../fake"));
  EXPECT_TRUE(CachedDotDot);
  EXPECT_EQ(CachedDotDot->getName(), testPath("temp/../fake"));
  EXPECT_EQ(CachedDotDot->getUniqueID(), S.getUniqueID());
}

} // namespace
} // namespace clangd
} // namespace clang
