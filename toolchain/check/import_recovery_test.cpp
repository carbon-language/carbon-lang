// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "common/check.h"
#include "common/raw_string_ostream.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/check/check.h"
#include "toolchain/diagnostics/consumer.h"
#include "toolchain/lex/lex.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::Check {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

struct TestUnit {
  SharedValueStores value_stores;
  RawStringOstream diagnostic_output;
  Diagnostics::StreamConsumer consumer =
      Diagnostics::StreamConsumer(&diagnostic_output);
  llvm::LLVMContext llvm_context;
  std::optional<SourceBuffer> source;
  std::optional<Lex::TokenizedBuffer> tokens;
  std::optional<Parse::Tree> tree;
  std::optional<Parse::TreeAndSubtrees> tree_and_subtrees;
  std::optional<SemIR::File> sem_ir;
};

static auto BuildTestUnit(TestUnit& unit, llvm::vfs::InMemoryFileSystem& fs,
                          llvm::StringRef filename, llvm::StringRef source_text,
                          SemIR::CheckIRId check_ir_id) -> void {
  unit.consumer.set_include_diagnostic_kind(true);

  CARBON_CHECK(
      fs.addFile(filename, /*ModificationTime=*/0,
                 llvm::MemoryBuffer::getMemBuffer(
                     source_text, filename, /*RequiresNullTerminator=*/false)));

  auto source = SourceBuffer::MakeFromFile(fs, filename, unit.consumer);
  CARBON_CHECK(source.has_value());
  unit.source = std::move(*source);

  Lex::LexOptions lex_options;
  lex_options.consumer = &unit.consumer;
  unit.tokens.emplace(Lex::Lex(unit.value_stores, *unit.source, lex_options));

  Parse::ParseOptions parse_options;
  parse_options.consumer = &unit.consumer;
  unit.tree.emplace(Parse::Parse(*unit.tokens, parse_options));

  unit.tree_and_subtrees.emplace(*unit.tokens, *unit.tree);
  unit.sem_ir.emplace(&*unit.tree, check_ir_id, unit.tree->packaging_decl(),
                      unit.value_stores, filename.str());
}

TEST(ImportRecoveryTest, CurrentPackageImportByNameStillImportsLibrary) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs =
      new llvm::vfs::InMemoryFileSystem;

  constexpr int TotalIrCount = 2;
  TestUnit unit_a;
  BuildTestUnit(unit_a, *fs, "a.carbon",
                "package Foo library \"a\";\n"
                "\n"
                "class X {}\n",
                SemIR::CheckIRId(0));
  TestUnit unit_b;
  BuildTestUnit(unit_b, *fs, "b.carbon",
                "package Foo library \"b\";\n"
                "\n"
                "import Foo library \"a\";\n"
                "\n"
                "var x: X = {};\n",
                SemIR::CheckIRId(1));

  Parse::GetTreeAndSubtreesStore tree_and_subtrees_getters =
      Parse::GetTreeAndSubtreesStore::MakeForOverwriteWithExplicitSize(2);
  auto get_unit_a = [&]() -> const Parse::TreeAndSubtrees& {
    return *unit_a.tree_and_subtrees;
  };
  auto get_unit_b = [&]() -> const Parse::TreeAndSubtrees& {
    return *unit_b.tree_and_subtrees;
  };
  tree_and_subtrees_getters.Set(SemIR::CheckIRId(0), get_unit_a);
  tree_and_subtrees_getters.Set(SemIR::CheckIRId(1), get_unit_b);

  llvm::SmallVector<Unit> units = {
      {.consumer = &unit_a.consumer,
       .value_stores = &unit_a.value_stores,
       .timings = nullptr,
       .sem_ir = &*unit_a.sem_ir,
       .llvm_context = &unit_a.llvm_context,
       .total_ir_count = TotalIrCount},
      {.consumer = &unit_b.consumer,
       .value_stores = &unit_b.value_stores,
       .timings = nullptr,
       .sem_ir = &*unit_b.sem_ir,
       .llvm_context = &unit_b.llvm_context,
       .total_ir_count = TotalIrCount},
  };

  CheckParseTrees(units, tree_and_subtrees_getters, fs,
                  CheckParseTreesOptions(),
                  /*clang_invocation=*/nullptr);

  auto x_id = unit_b.value_stores.identifiers().Lookup("X");
  ASSERT_TRUE(x_id.has_value());
  EXPECT_TRUE(unit_b.sem_ir->name_scopes()
                  .Get(SemIR::NameScopeId::Package)
                  .Lookup(SemIR::NameId::ForIdentifier(x_id))
                  .has_value());

  auto diagnostics = unit_b.diagnostic_output.TakeStr();
  EXPECT_THAT(diagnostics,
              HasSubstr("imports from the current package must omit the "
                        "package name"));
  EXPECT_THAT(diagnostics, Not(HasSubstr("name `X`")));
}

}  // namespace
}  // namespace Carbon::Check
