// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/source_gen.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/set.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/driver/driver.h"
#include "toolchain/install/install_paths_test_helpers.h"

namespace Carbon::Testing {
namespace {

using ::testing::AllOf;
using ::testing::ContainerEq;
using ::testing::Contains;
using ::testing::Each;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::Gt;
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

TEST(SourceGenTest, Identifiers) {
  SourceGen gen;

  auto idents = gen.GetShuffledIdentifiers(1000);
  EXPECT_THAT(idents.size(), Eq(1000));
  for (llvm::StringRef ident : idents) {
    EXPECT_THAT(ident, MatchesRegex("[A-Za-z][A-Za-z0-9_]*"));
  }

  // We should have at least one identifier of each length [1, 64]. The exact
  // distribution is an implementation detail designed to vaguely match the
  // expected distribution in source code.
  for (int size : llvm::seq_inclusive(1, 64)) {
    EXPECT_THAT(idents, Contains(SizeIs(size)));
  }

  // Check that identifiers 4 characters or shorter are more common than longer
  // lengths. This is a very rough way of double checking that we got the
  // intended distribution.
  for (int short_size : llvm::seq_inclusive(1, 4)) {
    int short_count = llvm::count_if(idents, [&](auto ident) {
      return static_cast<int>(ident.size()) == short_size;
    });
    for (int long_size : llvm::seq_inclusive(5, 64)) {
      EXPECT_THAT(short_count, Gt(llvm::count_if(idents, [&](auto ident) {
                    return static_cast<int>(ident.size()) == long_size;
                  })));
    }
  }

  // Check that repeated calls are different in interesting ways, but have the
  // exact same total bytes.
  ssize_t idents_size_sum = SumSizes(idents);
  for ([[maybe_unused]] auto _ : llvm::seq(10)) {
    auto idents2 = gen.GetShuffledIdentifiers(1000);
    EXPECT_THAT(idents2, SizeIs(1000));
    // Should be (at least) a different shuffle of identifiers.
    EXPECT_THAT(idents2, Not(ContainerEq(idents)));
    // But the sum of lengths should be identical.
    EXPECT_THAT(SumSizes(idents2), Eq(idents_size_sum));
  }

  // Check length constraints have the desired effect.
  idents =
      gen.GetShuffledIdentifiers(1000, /*min_length=*/10, /*max_length=*/20);
  EXPECT_THAT(idents, Each(SizeIs(AllOf(Ge(10), Le(20)))));
}

TEST(SourceGenTest, UniformIdentifiers) {
  SourceGen gen;
  // Check that uniform identifier length results in exact coverage of each
  // possible length for an easy case, both without and with a remainder.
  auto idents =
      gen.GetShuffledIdentifiers(100, /*min_length=*/10, /*max_length=*/19,
                                 /*uniform=*/true);
  EXPECT_THAT(idents, Contains(SizeIs(10)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(11)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(12)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(13)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(14)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(15)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(16)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(17)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(18)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(19)).Times(10));

  idents = gen.GetShuffledIdentifiers(97, /*min_length=*/10, /*max_length=*/19,
                                      /*uniform=*/true);
  EXPECT_THAT(idents, Contains(SizeIs(10)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(11)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(12)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(13)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(14)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(15)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(16)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(17)).Times(9));
  EXPECT_THAT(idents, Contains(SizeIs(18)).Times(9));
  EXPECT_THAT(idents, Contains(SizeIs(19)).Times(9));
}

// Largely covered by `Identifiers` and `UniformIdentifiers`, but need to check
// for uniqueness specifically.
TEST(SourceGenTest, UniqueIdentifiers) {
  SourceGen gen;

  auto unique = gen.GetShuffledUniqueIdentifiers(1000);
  EXPECT_THAT(unique.size(), Eq(1000));
  Set<llvm::StringRef> set;
  for (llvm::StringRef ident : unique) {
    EXPECT_THAT(ident, MatchesRegex("[A-Za-z][A-Za-z0-9_]*"));
    EXPECT_TRUE(set.Insert(ident).is_inserted())
        << "Colliding identifier: " << ident;
  }

  // Check single length specifically where uniqueness is the most challenging.
  set.Clear();
  unique = gen.GetShuffledUniqueIdentifiers(1000, /*min_length=*/4,
                                            /*max_length=*/4);
  for (llvm::StringRef ident : unique) {
    EXPECT_TRUE(set.Insert(ident).is_inserted())
        << "Colliding identifier: " << ident;
  }
}

// Check that the source code doesn't have compiler errors.
auto TestCompile(llvm::StringRef source) -> bool {
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs =
      new llvm::vfs::InMemoryFileSystem;
  InstallPaths installation(
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath()));
  Driver driver(fs, &installation, /*input_stream=*/nullptr, &llvm::outs(),
                &llvm::errs());

  AddPreludeFilesToVfs(installation, fs);

  fs->addFile("test.carbon", /*ModificationTime=*/0,
              llvm::MemoryBuffer::getMemBuffer(source));
  return driver.RunCommand({"compile", "--phase=check", "test.carbon"}).success;
}

TEST(SourceGenTest, GenApiFileDenseDeclsTest) {
  SourceGen gen;

  std::string source =
      gen.GenApiFileDenseDecls(1000, SourceGen::DenseDeclParams{});
  // Should be within 1% of the requested line count.
  EXPECT_THAT(source, Contains('\n').Times(AllOf(Ge(950), Le(1050))));

  // Make sure we generated valid Carbon code.
  EXPECT_TRUE(TestCompile(source));
}

TEST(SourceGenTest, GenApiFileDenseDeclsCppTest) {
  SourceGen gen(SourceGen::Language::Cpp);

  // Generate a 1000-line file which is enough to have a reasonably accurate
  // line count estimate and have a few classes.
  std::string source =
      gen.GenApiFileDenseDecls(1000, SourceGen::DenseDeclParams{});
  // Should be within 10% of the requested line count.
  EXPECT_THAT(source, Contains('\n').Times(AllOf(Ge(900), Le(1100))));

  // TODO: When the driver supports compiling C++ code as easily as Carbon, we
  // should test that the generated C++ code is valid.
}

}  // namespace
}  // namespace Carbon::Testing
