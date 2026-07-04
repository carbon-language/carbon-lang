// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/benchmarking/source_gen.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "common/error.h"
#include "common/raw_string_ostream.h"
#include "common/set.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "testing/base/capture_std_streams.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/base/install_paths_test_helpers.h"
#include "toolchain/driver/clang_runner.h"
#include "toolchain/driver/driver.h"

namespace Carbon::Testing {
namespace {

using ::testing::AllOf;
using ::testing::ContainerEq;
using ::testing::Contains;
using ::testing::Each;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::Le;
using ::testing::MatchesRegex;
using ::testing::SizeIs;
using ::testing::StrEq;

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

// For fixed parameters, the total number of bytes across the returned
// identifiers must not depend on the random seed, even though the specific
// identifiers do. This checks that across a range of parameters and across many
// freshly-seeded generators (each `SourceGen` gets an independent random seed).
TEST(SourceGenTest, IdentifierByteSumStableAcrossSeeds) {
  struct Config {
    int number;
    int min_length;
    int max_length;
    bool uniform;
    bool unique;
  };
  // A spread of parameters including: the default range, narrow ranges, the
  // single-length extreme, uniform distributions, and a uniform range with a
  // `max_length` well beyond the 64 limit that only the uniform path allows.
  Config configs[] = {
      {.number = 1000, .min_length = 1, .max_length = 64, .uniform = false},
      {.number = 1000, .min_length = 4, .max_length = 64, .uniform = false},
      {.number = 999, .min_length = 1, .max_length = 64, .uniform = false},
      {.number = 1000, .min_length = 10, .max_length = 20, .uniform = false},
      {.number = 1000, .min_length = 8, .max_length = 8, .uniform = false},
      {.number = 100, .min_length = 10, .max_length = 19, .uniform = true},
      {.number = 97, .min_length = 10, .max_length = 19, .uniform = true},
      {.number = 500, .min_length = 50, .max_length = 200, .uniform = true},
      {.number = 1000,
       .min_length = 4,
       .max_length = 64,
       .uniform = false,
       .unique = true},
      {.number = 1000,
       .min_length = 4,
       .max_length = 4,
       .uniform = false,
       .unique = true},
      {.number = 200,
       .min_length = 30,
       .max_length = 120,
       .uniform = true,
       .unique = true},
  };

  for (const Config& c : configs) {
    SCOPED_TRACE(llvm::formatv(
        "Config: number={0} min_length={1} max_length={2} uniform={3} "
        "unique={4}",
        c.number, c.min_length, c.max_length, c.uniform, c.unique));
    std::optional<ssize_t> expected_sum;
    bool any_different = false;
    std::optional<llvm::SmallVector<std::string>> first;
    constexpr int NumSeeds = 8;
    for (int seed : llvm::seq(NumSeeds)) {
      // Each iteration constructs a fresh generator with an independent random
      // seed; the traced index identifies which iteration failed.
      SCOPED_TRACE(llvm::formatv("Seed iteration: {0}", seed));
      SourceGen gen;
      auto idents = c.unique
                        ? gen.GetShuffledUniqueIdentifiers(
                              c.number, c.min_length, c.max_length, c.uniform)
                        : gen.GetShuffledIdentifiers(c.number, c.min_length,
                                                     c.max_length, c.uniform);
      EXPECT_THAT(idents, SizeIs(c.number));
      EXPECT_THAT(idents,
                  Each(SizeIs(AllOf(Ge(c.min_length), Le(c.max_length)))));

      ssize_t sum = SumSizes(idents);
      if (!expected_sum) {
        expected_sum = sum;
        first.emplace(idents.begin(), idents.end());
        continue;
      }
      // The byte sum must be identical regardless of the seed.
      EXPECT_THAT(sum, Eq(*expected_sum));
      if (!llvm::equal(idents, *first)) {
        any_different = true;
      }
    }
    // Sanity check that the generators really are producing different content,
    // so that the invariance check above is meaningful rather than trivially
    // passing on identical output.
    EXPECT_TRUE(any_different);
  }
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

// Compiles `source` and returns whether it compiled with no diagnostics at all.
// Generated code should be warning-free: warnings would add diagnostic
// emission to the compile benchmarks.
auto TestCompile(SourceGen::Language language, llvm::StringRef source) -> bool {
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs =
      new llvm::vfs::InMemoryFileSystem;
  InstallPaths installation(
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath()));

  if (language == SourceGen::Language::Carbon) {
    RawStringOstream diagnostics;
    Driver driver(fs, &installation, /*input_stream=*/nullptr, &llvm::outs(),
                  &diagnostics);
    AddPreludeFilesToVfs(installation, fs);
    fs->addFile("test.carbon", /*ModificationTime=*/0,
                llvm::MemoryBuffer::getMemBuffer(source));
    bool success = driver
                       .RunCommand({"compile", "--phase=check",
                                    "--no-include-carbon-core", "test.carbon"})
                       .success;
    std::string output = diagnostics.TakeStr();
    EXPECT_THAT(output, StrEq(""));
    return success && output.empty();
  }

  // Clang reads the system headers from the real filesystem.
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlay_fs =
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem());
  overlay_fs->pushOverlay(fs);
  fs->addFile("test.cpp", /*ModificationTime=*/0,
              llvm::MemoryBuffer::getMemBuffer(source));
  ClangRunner runner(&installation, overlay_fs);
  std::string out;
  std::string err;
  ErrorOr<bool> result = CallWithCapturedOutput(out, err, [&] {
    return runner.RunWithNoRuntimes({"-fsyntax-only", "test.cpp"});
  });
  EXPECT_THAT(err, StrEq(""));
  return result.ok() && *result && err.empty();
}

TEST(SourceGenTest, GenApiFileDenseDeclsTest) {
  SourceGen gen;

  std::string source =
      gen.GenApiFileDenseDecls(1000, SourceGen::DenseDeclParams{});
  // Should be within 1% of the requested line count.
  EXPECT_THAT(source, Contains('\n').Times(AllOf(Ge(950), Le(1050))));

  EXPECT_TRUE(TestCompile(SourceGen::Language::Carbon, source));
}

TEST(SourceGenTest, GenApiFileDenseDeclsCppTest) {
  SourceGen gen(SourceGen::Language::Cpp);

  // Generate a 1000-line file which is enough to have a reasonably accurate
  // line count estimate and have a few classes.
  std::string source =
      gen.GenApiFileDenseDecls(1000, SourceGen::DenseDeclParams{});
  // Should be within 10% of the requested line count.
  EXPECT_THAT(source, Contains('\n').Times(AllOf(Ge(900), Le(1100))));

  EXPECT_TRUE(TestCompile(SourceGen::Language::Cpp, source));
}

static auto CountLines(llvm::StringRef source) -> ssize_t {
  return llvm::count(source, '\n');
}

// Generates a file with each of `num_seeds` independently seeded generators,
// and expects all of the files to have the same byte and line counts. Also
// expects the first file to compile, and the content to vary so that the size
// check isn't vacuous.
static auto ExpectSeedIndependentSize(SourceGen::Language language,
                                      int target_lines,
                                      const SourceGen::DenseDeclParams& params,
                                      int num_seeds) -> void {
  SCOPED_TRACE(
      llvm::formatv("language={0}, target_lines={1}",
                    language == SourceGen::Language::Carbon ? "Carbon" : "C++",
                    target_lines)
          .str());
  std::string first;
  bool any_different = false;
  for (int i : llvm::seq(num_seeds)) {
    SourceGen gen(language);
    std::string source = gen.GenApiFileDenseDecls(target_lines, params);
    if (i == 0) {
      EXPECT_TRUE(TestCompile(language, source));
      first = std::move(source);
      continue;
    }
    EXPECT_THAT(source.size(), Eq(first.size()));
    EXPECT_THAT(CountLines(source), Eq(CountLines(first)));
    any_different = any_different || source != first;
  }
  EXPECT_TRUE(any_different);
}

// Benchmarks are only comparable if the generated source has the same size for
// any seed.
TEST(SourceGenTest, GenApiFileDenseDeclsStableSizeAcrossSeeds) {
  for (SourceGen::Language language :
       {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
    // From barely enough lines for one class up to a large file.
    for (int target_lines : {200, 1000, 5000, 20000}) {
      ExpectSeedIndependentSize(language, target_lines,
                                SourceGen::DenseDeclParams{}, /*num_seeds=*/16);
    }
  }
}

TEST(SourceGenTest, GenApiFileDenseDeclsStableSizeWithVariedParams) {
  llvm::SmallVector<SourceGen::DenseDeclParams, 0> param_set;
  // Function declarations only: no methods and no fields.
  param_set.push_back({.class_params = {.public_function_decls = 20,
                                        .public_method_decls = 0,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 0}});
  // Large parameter counts, which wrap onto several lines.
  param_set.push_back(
      {.class_params = {.public_function_decls = 2,
                        .public_function_decl_params = {.max_params = 16},
                        .public_method_decls = 4,
                        .public_method_decl_params = {.max_params = 16},
                        .private_function_decls = 0,
                        .private_method_decls = 0,
                        .private_field_decls = 0}});
  // The default shape scaled up 2x.
  param_set.push_back({.class_params = {.public_function_decls = 8,
                                        .public_method_decls = 20,
                                        .private_function_decls = 4,
                                        .private_method_decls = 16,
                                        .private_field_decls = 12}});

  for (const SourceGen::DenseDeclParams& params : param_set) {
    for (SourceGen::Language language :
         {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
      ExpectSeedIndependentSize(language, /*target_lines=*/5000, params,
                                /*num_seeds=*/12);
    }
  }
}

// A class's fields can't reference it or any later class, so field-heavy
// classes leave few type uses for references to a class. Whether the valid type
// names run out depends on the shuffle, so use many seeds.
TEST(SourceGenTest, GenApiFileDenseDeclsRobustForFieldHeavyParams) {
  llvm::SmallVector<SourceGen::DenseDeclParams, 0> param_set;
  param_set.push_back({.class_params = {.public_function_decls = 1,
                                        .public_method_decls = 1,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 30}});
  param_set.push_back({.class_params = {.public_function_decls = 0,
                                        .public_method_decls = 1,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 50}});
  // No functions or methods, so every type use is a fixed type.
  param_set.push_back({.class_params = {.public_function_decls = 0,
                                        .public_method_decls = 0,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 16}});

  for (const SourceGen::DenseDeclParams& params : param_set) {
    for (SourceGen::Language language :
         {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
      ExpectSeedIndependentSize(language, /*target_lines=*/3000, params,
                                /*num_seeds=*/32);
    }
  }
}

// Bodies read class-typed parameters through `Checksum`, and return class
// values through `Make`.
TEST(SourceGenTest, GenApiFileDenseDeclsInlineBodies) {
  SourceGen::DenseDeclParams params = {
      .class_params = {.inline_function_defs = 3,
                       .max_body_locals = 4,
                       .inline_getters = 2,
                       .inline_predicates = 2,
                       .inline_forwarders = 2}};
  for (SourceGen::Language language :
       {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
    ExpectSeedIndependentSize(language, /*target_lines=*/2000, params,
                              /*num_seeds=*/16);
  }

  SourceGen gen;
  std::string source = gen.GenApiFileDenseDecls(2000, params);
  EXPECT_THAT(source, HasSubstr("acc = acc + "));
  EXPECT_THAT(source, HasSubstr(".Checksum()"));
  EXPECT_THAT(source, HasSubstr(".Make()"));
  EXPECT_THAT(source, HasSubstr("Impl("));
}

TEST(SourceGenTest, GenApiFileDenseDeclsInlineBodiesWithVariedParams) {
  llvm::SmallVector<SourceGen::DenseDeclParams, 0> param_set;
  // Many large bodies and few other declarations.
  param_set.push_back({.class_params = {.public_function_decls = 1,
                                        .public_method_decls = 1,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 4,
                                        .inline_function_defs = 8,
                                        .max_body_locals = 12}});
  // Many fields, which `Make` initializes, often with an earlier class's
  // `Make`.
  param_set.push_back({.class_params = {.public_function_decls = 1,
                                        .public_method_decls = 1,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 24,
                                        .inline_function_defs = 2,
                                        .max_body_locals = 3}});
  // Many getters and predicates, and no other fields.
  param_set.push_back({.class_params = {.private_field_decls = 0,
                                        .inline_getters = 8,
                                        .inline_predicates = 8}});
  // Many forwarders with many parameters, which can have class types.
  param_set.push_back(
      {.class_params = {.inline_forwarders = 8,
                        .inline_forwarder_params = {.max_params = 8}}});

  for (const SourceGen::DenseDeclParams& params : param_set) {
    for (SourceGen::Language language :
         {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
      ExpectSeedIndependentSize(language, /*target_lines=*/3000, params,
                                /*num_seeds=*/24);
    }
  }
}

// Locals and parameters draw from the same identifiers of each length, and C++
// rejects a local that redeclares a parameter. Whether a local could do so
// depends on the shuffle, so compile many seeds.
TEST(SourceGenTest, GenApiFileDenseDeclsInlineBodiesCppCompiles) {
  SourceGen::DenseDeclParams params = {
      .class_params = {.public_function_decls = 1,
                       .public_method_decls = 1,
                       .private_function_decls = 0,
                       .private_method_decls = 0,
                       .private_field_decls = 4,
                       .inline_function_defs = 8,
                       .max_body_locals = 12}};
  for (int _ : llvm::seq(8)) {
    SourceGen gen(SourceGen::Language::Cpp);
    EXPECT_TRUE(TestCompile(SourceGen::Language::Cpp,
                            gen.GenApiFileDenseDecls(5000, params)));
  }
}

// The line estimates have to track the emitted lines closely, or files miss
// their target size. The target is large so that rounding to a whole number of
// classes is small next to the tolerance. C++ gets a larger tolerance for its
// access specifier lines, which the estimates don't count.
TEST(SourceGenTest, GenApiFileDenseDeclsLineTargetAccuracy) {
  SourceGen::DenseDeclParams params = {
      .class_params = {.inline_function_defs = 1,
                       .max_body_locals = 3,
                       .inline_getters = 1,
                       .inline_predicates = 1,
                       .inline_forwarders = 1}};

  constexpr int TargetLines = 20000;
  for (SourceGen::Language language :
       {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
    SourceGen gen(language);
    std::string source = gen.GenApiFileDenseDecls(TargetLines, params);
    ssize_t lines = CountLines(source);
    if (language == SourceGen::Language::Carbon) {
      // Within 2% of the requested line count.
      EXPECT_THAT(lines, AllOf(Ge(19600), Le(20400)));
    } else {
      // Within 10% of the requested line count.
      EXPECT_THAT(lines, AllOf(Ge(18000), Le(22000)));
    }
  }
}

}  // namespace
}  // namespace Carbon::Testing
