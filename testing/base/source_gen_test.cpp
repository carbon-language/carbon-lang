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
using ::testing::Each;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::Le;
using ::testing::MatchesRegex;
using ::testing::SizeIs;

// Tiny helper to sum the sizes of a range of ranges. Uses a template to avoid
// hard coding any specific types for the two ranges.
template <typename T>
static auto SumSizes(const T& range) -> ssize_t {
  ssize_t sum = 0;
  for (const auto& inner_range : range) {
    sum += inner_range.size();
  }
  return sum;
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
  for ([[maybe_unused]] int i : llvm::seq(10)) {
    auto ids2 = gen.GetShuffledIds(1000);
    EXPECT_THAT(ids2, SizeIs(1000));
    // Should be (at least) a different shuffle of identifiers.
    EXPECT_THAT(ids2, Not(ContainerEq(ids)));
    // But the sum of lengths should be identical.
    EXPECT_THAT(SumSizes(ids2), Eq(ids_size_sum));
  }

  // Check length constraints have the desired effect.
  ids = gen.GetShuffledIds(1000, /*min_length=*/10, /*max_length=*/20);
  EXPECT_THAT(ids, Each(SizeIs(AllOf(Ge(10), Le(20)))));

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

// Largely covered by `Ids`, but need to check for uniqueness specifically.
TEST(SourceGenTest, UniqueIds) {
  SourceGen gen;

  auto unique = gen.GetShuffledUniqueIds(1000);
  EXPECT_THAT(unique.size(), Eq(1000));
  Set<llvm::StringRef> set;
  for (llvm::StringRef id : unique) {
    EXPECT_THAT(id, MatchesRegex("[A-Za-z][A-Za-z0-9_]*"));
    EXPECT_TRUE(set.Insert(id).is_inserted()) << "Colliding id: " << id;
  }

  // Check single length specifically where uniqueness is the most challenging.
  set.Clear();
  unique = gen.GetShuffledUniqueIds(1000, /*min_length=*/4, /*max_length=*/4);
  for (llvm::StringRef id : unique) {
    EXPECT_TRUE(set.Insert(id).is_inserted()) << "Colliding id: " << id;
  }
}

// Check that the source code doesn't have compiler errors.
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

  // Generate a 1000-line file which is enough to have a reasonably accurate
  // line count estimate and have a few classes.
  std::string source =
      gen.GenAPIFileDenseDecls(1000, SourceGen::DenseDeclParams{});
  // Should be within 10% of the requested line count.
  EXPECT_THAT(source, Contains('\n').Times(AllOf(Ge(900), Le(1100))));

  // TODO: When the driver supports compiling C++ code as easily as Carbon, we
  // should test that the generated C++ code is valid.
}

}  // namespace
}  // namespace Carbon::Testing
