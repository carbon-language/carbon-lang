// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/source_gen.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/set.h"
#include "testing/base/gtest_main.h"
#include "toolchain/driver/driver.h"

namespace Carbon::Testing {
namespace {

using ::testing::AllOf;
using ::testing::ContainerEq;
using ::testing::Contains;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::Le;
using ::testing::MatchesRegex;
using ::testing::SizeIs;

template <typename T>
static auto SumSizes(const T& range) -> ssize_t {
  return std::accumulate(
      range.begin(), range.end(), static_cast<ssize_t>(0),
      [](ssize_t lhs, const auto& rhs) -> ssize_t { return lhs + rhs.size(); });
}

TEST(SourceGenTest, UniqueIds) {
  SourceGen gen;

  auto unique = gen.GetShuffledUniqueIds(1000);
  EXPECT_THAT(unique.size(), Eq(1000));
  Set<llvm::StringRef> set;
  for (llvm::StringRef id : unique) {
    EXPECT_THAT(id, MatchesRegex("[A-Za-z][A-Za-z0-9_]*"));
    EXPECT_TRUE(set.Insert(id).is_inserted()) << "Colliding id: " << id;
  }

  // Check that repeated calls are different in interesting ways, but have the
  // exact same total bytes.
  ssize_t unique_size_sum = SumSizes(unique);
  for ([[maybe_unused]] int i : llvm::seq(1, 10)) {
    auto unique2 = gen.GetShuffledUniqueIds(1000);
    EXPECT_THAT(unique2, SizeIs(1000));
    // Should be (at least) a different shuffle of identifiers.
    EXPECT_THAT(unique2, Not(ContainerEq(unique)));
    // But the sum of lengths should be identical.
    EXPECT_THAT(SumSizes(unique2), Eq(unique_size_sum));
  }

  // Check length constraints have the desired effect.
  unique = gen.GetShuffledUniqueIds(1000, /*min_length=*/10, /*max_length=*/20);
  EXPECT_THAT(unique, Each(SizeIs(AllOf(Ge(10), Le(20)))));

  // Check that uniform id length results in exact coverage of each possible
  // length for an easy case.
  unique = gen.GetShuffledUniqueIds(100, /*min_length=*/10, /*max_length=*/19,
                                    /*uniform=*/true);
  EXPECT_THAT(unique, Contains(SizeIs(10)).Times(10));
  EXPECT_THAT(unique, Contains(SizeIs(11)).Times(10));
  EXPECT_THAT(unique, Contains(SizeIs(12)).Times(10));
  EXPECT_THAT(unique, Contains(SizeIs(13)).Times(10));
  EXPECT_THAT(unique, Contains(SizeIs(14)).Times(10));
  EXPECT_THAT(unique, Contains(SizeIs(15)).Times(10));
  EXPECT_THAT(unique, Contains(SizeIs(16)).Times(10));
  EXPECT_THAT(unique, Contains(SizeIs(17)).Times(10));
  EXPECT_THAT(unique, Contains(SizeIs(18)).Times(10));
  EXPECT_THAT(unique, Contains(SizeIs(19)).Times(10));
}

TEST(SourceGenTest, Ids) {
  SourceGen gen;

  auto ids = gen.GetShuffledIds(1000);
  EXPECT_THAT(ids.size(), Eq(1000));
  for (llvm::StringRef id : ids) {
    EXPECT_THAT(id, MatchesRegex("[A-Za-z][A-Za-z0-9_]*"));
  }

  // Check that repeated calls are different in interesting ways, but have the
  // exact same total bytes.
  ssize_t ids_size_sum = SumSizes(ids);
  for ([[maybe_unused]] int i : llvm::seq(1, 10)) {
    auto ids2 = gen.GetShuffledIds(1000);
    EXPECT_THAT(ids2, SizeIs(1000));
    // Should be (at least) a different shuffle of identifiers.
    EXPECT_THAT(ids2, Not(ContainerEq(ids)));
    // But the sum of lengths should be identical.
    EXPECT_THAT(SumSizes(ids2), Eq(ids_size_sum));
  }

  // Check length constraints have the desired effect.
  ids = gen.GetShuffledIds(1000, /*min_length=*/10, /*max_length=*/20);
  for (llvm::StringRef id : ids) {
    EXPECT_THAT(id.size(), Ge(10)) << "Too short id: " << id;
    EXPECT_THAT(id.size(), Le(20)) << "Too long id: " << id;
  }

  // Check that uniform id length results in exact coverage of each possible
  // length for an easy case.
  ids = gen.GetShuffledIds(100, /*min_length=*/10, /*max_length=*/19,
                           /*uniform=*/true);
  EXPECT_THAT(ids, Contains(SizeIs(10)).Times(10));
  EXPECT_THAT(ids, Contains(SizeIs(11)).Times(10));
  EXPECT_THAT(ids, Contains(SizeIs(12)).Times(10));
  EXPECT_THAT(ids, Contains(SizeIs(13)).Times(10));
  EXPECT_THAT(ids, Contains(SizeIs(14)).Times(10));
  EXPECT_THAT(ids, Contains(SizeIs(15)).Times(10));
  EXPECT_THAT(ids, Contains(SizeIs(16)).Times(10));
  EXPECT_THAT(ids, Contains(SizeIs(17)).Times(10));
  EXPECT_THAT(ids, Contains(SizeIs(18)).Times(10));
  EXPECT_THAT(ids, Contains(SizeIs(19)).Times(10));
}

auto TestCompile(llvm::StringRef source) -> bool {
  llvm::vfs::InMemoryFileSystem fs;
  InstallPaths installation(
      InstallPaths::MakeForBazelRunfiles(Testing::GetTestExePath()));
  Driver driver(fs, &installation, llvm::outs(), llvm::errs());

  // Load the prelude into our VFS.
  //
  // TODO: Factor this and analogous code in file_test into a Driver helper.
  auto prelude =
      Driver::FindPreludeFiles(installation.core_package(), llvm::errs());
  CARBON_CHECK(!prelude.empty());
  for (const auto& path : prelude) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
        llvm::MemoryBuffer::getFile(path);
    CARBON_CHECK(file) << file.getError().message();
    CARBON_CHECK(fs.addFile(path, /*ModificationTime=*/0, std::move(*file)))
        << "Duplicate file: " << path;
  }

  fs.addFile("test.carbon", /*ModificationTime=*/0,
             llvm::MemoryBuffer::getMemBuffer(source));
  return driver.RunCommand({"compile", "--phase=check", "test.carbon"}).success;
}

TEST(SourceGenTest, GenAPIFileDenseDeclsTest) {
  SourceGen gen;

  std::string source =
      gen.GenAPIFileDenseDecls(1000, SourceGen::DenseDeclParams{});
  // Should be within 10% of the requested line count.
  EXPECT_THAT(source, Contains('\n').Times(AllOf(Ge(900), Le(1100))));

  // Make sure we generated valid Carbon code.
  EXPECT_TRUE(TestCompile(source));
}

TEST(SourceGenTest, GenAPIFileDenseDeclsCppTest) {
  SourceGen gen(SourceGen::Language::Cpp);

  std::string source =
      gen.GenAPIFileDenseDecls(1000, SourceGen::DenseDeclParams{});
  // Should be within 10% of the requested line count.
  EXPECT_THAT(source, Contains('\n').Times(AllOf(Ge(900), Le(1100))));

  // TODO: When the driver supports compiling C++ code as easily as Carbon, we
  // should test that the generated C++ code is valid.
}

}  // namespace
}  // namespace Carbon::Testing
