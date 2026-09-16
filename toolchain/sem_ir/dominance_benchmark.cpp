// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <algorithm>
#include <optional>

#include "common/check.h"
#include "common/error.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/dominance.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {
namespace {

// The number of value uses in each generated block, in addition to its
// terminators. Real blocks contain more instructions than terminators, so this
// spreads the verifier's per-block costs over several instructions, as they
// would be in practice.
constexpr int InstsPerBlock = 8;

// Builds a file containing synthetic function bodies to verify.
//
// The generated IR only has the properties that `VerifyDominance` looks at: it
// has no locations or real types, and instructions are shared between
// functions, neither of which the verifier examines.
class FileBuilder {
 public:
  FileBuilder()
      : file_(/*parse_tree=*/nullptr, CheckIRId(0),
              /*packaging_decl=*/std::nullopt, value_stores_,
              "dominance_benchmark.carbon"),
        value_id_(AddNonConstInst(BoolLiteral{.type_id = TypeType::TypeId,
                                              .value = BoolValue(false)})),
        cond_id_(AddConstant()) {}

  auto file() -> File& { return file_; }

  // The condition used by every conditional branch. It's constant, so it needs
  // no dominating evaluation.
  auto cond_id() const -> InstId { return cond_id_; }

  // Adds a block whose contents are filled in later.
  auto AddBlock() -> InstBlockId {
    return file_.inst_blocks().AddPlaceholder();
  }

  // Fills the entry block of a function body. It evaluates the value that the
  // rest of the body uses, so that every use in the body is dominated, and
  // branches to `next_id`.
  auto FillEntryBlock(InstBlockId block_id, InstBlockId next_id) -> void {
    file_.inst_blocks().ReplacePlaceholder(block_id,
                                           {value_id_, AddBranch(next_id)});
  }

  // Fills a non-entry block with `num_uses` uses of the value evaluated in the
  // entry block, followed by the block's terminators.
  auto FillBlock(InstBlockId block_id, llvm::ArrayRef<InstId> terminators,
                 int num_uses = InstsPerBlock) -> void {
    llvm::SmallVector<InstId> insts;
    insts.reserve(num_uses + terminators.size());
    for (int i = 0; i != num_uses; ++i) {
      insts.push_back(AddNonConstInst(
          ValueAsRef{.type_id = TypeType::TypeId, .value_id = value_id_}));
    }
    insts.append(terminators.begin(), terminators.end());
    file_.inst_blocks().ReplacePlaceholder(block_id, insts);
  }

  auto AddReturn() -> InstId { return AddInst(Return{}); }

  auto AddBranch(InstBlockId target_id) -> InstId {
    return AddInst(Branch{.target_id = LabelId(target_id)});
  }

  auto AddBranchIf(InstBlockId target_id) -> InstId {
    return AddInst(
        BranchIf{.target_id = LabelId(target_id), .cond_id = cond_id_});
  }

  // Adds a generic, along with a resolved specific for it, that a function can
  // be attached to. `VerifyDominance` verifies a generic function's body once
  // for the generic itself and once for each resolved specific.
  auto AddGeneric() -> GenericId {
    auto decl_id =
        AddInst(FunctionDecl{.type_id = TypeType::TypeId,
                             .function_id = FunctionId(0),
                             .decl_block_id = DeclInstBlockId::None});
    auto generic_id =
        file_.generics().Add(Generic{.decl_id = decl_id,
                                     .bindings_id = InstBlockId::Empty,
                                     .self_specific_id = SpecificId::None});
    // A specific is only verified once it's been resolved, which is indicated
    // by it having a value block for its declaration.
    auto specific_id =
        file_.specifics().GetOrAdd(generic_id, InstBlockId::Empty);
    file_.specifics()
        .Get(specific_id)
        .SetValueBlock(GenericInstIndex::Declaration, InstBlockId::Empty);
    return generic_id;
  }

  auto AddFunction(llvm::ArrayRef<InstBlockId> body_block_ids,
                   GenericId generic_id = GenericId::None) -> FunctionId {
    auto name_id = file_.identifiers().Add("F");
    return file_.functions().Add(
        {{.name_id = NameId::ForIdentifier(name_id),
          .parent_scope_id = NameScopeId::Package,
          .generic_id = generic_id,
          .first_param_node_id = Parse::NodeId::None,
          .last_param_node_id = Parse::NodeId::None,
          .pattern_block_id = InstBlockId::Empty,
          .implicit_param_patterns_id = InstBlockId::None,
          .param_patterns_id = InstBlockId::Empty,
          .is_extern = false,
          .extern_library_id = LibraryNameId::None,
          .non_owning_decl_id = InstId::None,
          .first_owning_decl_id = InstId::None},
         {.call_param_patterns_id = InstBlockId::Empty,
          .call_params_id = InstBlockId::Empty,
          .call_param_default_values_id = InstBlockId::Empty,
          .call_param_ranges = Function::CallParamIndexRanges::Empty,
          .return_type_inst_id = TypeInstId::None,
          .return_form_inst_id = InstId::None,
          .return_pattern_id = InstId::None,
          .body_block_ids = llvm::SmallVector<InstBlockId>(
              body_block_ids.begin(), body_block_ids.end())}});
  }

 private:
  template <typename InstT>
  auto AddInst(InstT inst) -> InstId {
    return file_.insts().AddInNoBlock(LocIdAndInst::NoLoc(inst));
  }

  // Adds an instruction with no constant value, so that uses of it must be
  // dominated by its evaluation.
  template <typename InstT>
  auto AddNonConstInst(InstT inst) -> InstId {
    auto inst_id = AddInst(inst);
    file_.constant_values().Set(inst_id, ConstantId::NotConstant);
    return inst_id;
  }

  auto AddConstant() -> InstId {
    auto inst_id = AddInst(
        BoolLiteral{.type_id = TypeType::TypeId, .value = BoolValue(true)});
    file_.constant_values().Set(inst_id,
                                ConstantId::ForConcreteConstant(inst_id));
    return inst_id;
  }

  SharedValueStores value_stores_;
  File file_;
  // The non-constant value that every generated block uses.
  InstId value_id_;
  InstId cond_id_;
};

// Builds a function body with a given control flow shape and approximate
// number of blocks, and returns its blocks, entry block first.
using BuildBodyFn = auto (*)(FileBuilder& file, int num_blocks)
    -> llvm::SmallVector<InstBlockId>;

// A single large block:
//
//     entry -> body
//
// This is the baseline: it has the same number of instructions as the other
// shapes of the same size, but essentially no control flow, so it measures the
// cost per instruction rather than per block.
auto BuildStraightLine(FileBuilder& file, int num_blocks)
    -> llvm::SmallVector<InstBlockId> {
  auto entry_id = file.AddBlock();
  auto body_id = file.AddBlock();
  file.FillEntryBlock(entry_id, body_id);
  file.FillBlock(body_id, {file.AddReturn()},
                 /*num_uses=*/num_blocks * InstsPerBlock);
  return {entry_id, body_id};
}

// A chain of blocks:
//
//     entry -> b1 -> b2 -> ... -> bn
//
// The dominator tree is a single deep path, so the set of instructions whose
// evaluations dominate the point being verified grows to the size of the body.
auto BuildChain(FileBuilder& file, int num_blocks)
    -> llvm::SmallVector<InstBlockId> {
  int size = std::max(num_blocks, 2);
  llvm::SmallVector<InstBlockId> blocks;
  blocks.reserve(size);
  for (int i = 0; i != size; ++i) {
    blocks.push_back(file.AddBlock());
  }

  file.FillEntryBlock(blocks[0], blocks[1]);
  for (int i = 1; i != size; ++i) {
    file.FillBlock(blocks[i], {i + 1 == size ? file.AddReturn()
                                             : file.AddBranch(blocks[i + 1])});
  }
  return blocks;
}

// A chain of diamonds, as an `if` in a loop body would produce:
//
//     entry -> head0 -> then0 -> head1 -> then1 -> ... -> headn
//                 \ -> else0 -/    \ -> else1 -/
//
// Each join block has two predecessors, so this exercises the dominator
// computation's intersection step, and the dominator tree is both deep and
// branching.
auto BuildDiamonds(FileBuilder& file, int num_blocks)
    -> llvm::SmallVector<InstBlockId> {
  int num_diamonds = std::max((num_blocks - 2) / 3, 1);

  llvm::SmallVector<InstBlockId> blocks = {file.AddBlock()};
  llvm::SmallVector<InstBlockId> heads;
  for (int i = 0; i != num_diamonds + 1; ++i) {
    heads.push_back(file.AddBlock());
    blocks.push_back(heads.back());
  }

  file.FillEntryBlock(blocks[0], heads[0]);
  for (int i = 0; i != num_diamonds; ++i) {
    auto then_id = file.AddBlock();
    auto else_id = file.AddBlock();
    blocks.push_back(then_id);
    blocks.push_back(else_id);

    file.FillBlock(heads[i],
                   {file.AddBranchIf(then_id), file.AddBranch(else_id)});
    file.FillBlock(then_id, {file.AddBranch(heads[i + 1])});
    file.FillBlock(else_id, {file.AddBranch(heads[i + 1])});
  }
  file.FillBlock(heads[num_diamonds], {file.AddReturn()});
  return blocks;
}

// A wide fan-out and fan-in, as a `match` would produce:
//
//     entry -> head -> arm1 -> exit
//                 \ -> ... -/
//                 \ -> armn -/
//
// The join block has a predecessor per arm, so this is the worst case for the
// parts of the verifier that are quadratic in a block's number of edges.
auto BuildFanOutFanIn(FileBuilder& file, int num_blocks)
    -> llvm::SmallVector<InstBlockId> {
  int num_arms = std::max(num_blocks - 3, 1);

  auto entry_id = file.AddBlock();
  auto head_id = file.AddBlock();
  auto exit_id = file.AddBlock();
  llvm::SmallVector<InstBlockId> blocks = {entry_id, head_id, exit_id};

  llvm::SmallVector<InstId> head_terminators;
  head_terminators.reserve(num_arms);
  for (int i = 0; i != num_arms; ++i) {
    auto arm_id = file.AddBlock();
    blocks.push_back(arm_id);
    // The last arm is reached unconditionally, as the `else` of the last
    // condition.
    head_terminators.push_back(i + 1 == num_arms ? file.AddBranch(arm_id)
                                                 : file.AddBranchIf(arm_id));
    file.FillBlock(arm_id, {file.AddBranch(exit_id)});
  }

  file.FillEntryBlock(entry_id, head_id);
  file.FillBlock(head_id, head_terminators);
  file.FillBlock(exit_id, {file.AddReturn()});
  return blocks;
}

// Nested loops:
//
//     entry -> header1 -> body1 -> header2 -> body2 -> ... -> (back edge)
//                 \ -> exit1 <- exit2 <- ...
//
// Each loop's back edge is a predecessor that is visited after the block it
// targets, so the dominator computation needs repeated sweeps to converge.
auto BuildNestedLoops(FileBuilder& file, int num_blocks)
    -> llvm::SmallVector<InstBlockId> {
  int num_loops = std::max((num_blocks - 1) / 3, 1);

  auto entry_id = file.AddBlock();
  llvm::SmallVector<InstBlockId> blocks = {entry_id};
  llvm::SmallVector<InstBlockId> headers;
  llvm::SmallVector<InstBlockId> bodies;
  llvm::SmallVector<InstBlockId> exits;
  for (int i = 0; i != num_loops; ++i) {
    headers.push_back(file.AddBlock());
    bodies.push_back(file.AddBlock());
    exits.push_back(file.AddBlock());
    blocks.append({headers[i], bodies[i], exits[i]});
  }

  file.FillEntryBlock(entry_id, headers[0]);
  for (int i = 0; i != num_loops; ++i) {
    file.FillBlock(headers[i],
                   {file.AddBranchIf(bodies[i]), file.AddBranch(exits[i])});
    // The innermost body closes its own loop; every other body enters the next
    // loop, whose exit branches back to this header.
    file.FillBlock(
        bodies[i],
        {file.AddBranch(i + 1 == num_loops ? headers[i] : headers[i + 1])});
    file.FillBlock(
        exits[i], {i == 0 ? file.AddReturn() : file.AddBranch(headers[i - 1])});
  }
  return blocks;
}

// Verifies `file` repeatedly, and reports the rate at which its instructions
// are verified.
auto RunBenchmark(benchmark::State& state, FileBuilder& file) -> void {
  for (auto _ : state) {
    ErrorOr<Success> result = VerifyDominance(file.file());
    CARBON_CHECK(result.ok(), "{0}", result.error().message());
  }
  state.counters["insts_per_second"] =
      benchmark::Counter(static_cast<double>(file.file().insts().size()),
                         benchmark::Counter::kIsIterationInvariantRate);
}

// The body sizes to benchmark, in blocks. The largest is far bigger than any
// realistic function body, so that superlinear behavior shows up as a falling
// instruction rate.
auto BodySizes(benchmark::Benchmark* bench) -> void {
  bench->RangeMultiplier(8)->Range(8, 32768);
}

// Verifies one function whose body has the shape built by `Build`.
template <BuildBodyFn Build>
auto BM_VerifyDominance(benchmark::State& state) -> void {
  FileBuilder file;
  file.AddFunction(Build(file, state.range(0)));
  RunBenchmark(state, file);
}

BENCHMARK(BM_VerifyDominance<BuildStraightLine>)
    ->Name("BM_VerifyDominance/StraightLine")
    ->Apply(BodySizes);
BENCHMARK(BM_VerifyDominance<BuildChain>)
    ->Name("BM_VerifyDominance/Chain")
    ->Apply(BodySizes);
BENCHMARK(BM_VerifyDominance<BuildDiamonds>)
    ->Name("BM_VerifyDominance/Diamonds")
    ->Apply(BodySizes);
BENCHMARK(BM_VerifyDominance<BuildNestedLoops>)
    ->Name("BM_VerifyDominance/NestedLoops")
    ->Apply(BodySizes);
// This shape has a block with a successor for each arm and a block with a
// predecessor for each arm, so it's the shape to watch for work that's
// quadratic in a block's number of edges.
BENCHMARK(BM_VerifyDominance<BuildFanOutFanIn>)
    ->Name("BM_VerifyDominance/FanOutFanIn")
    ->Apply(BodySizes);

// Verifies many small functions, which is the shape of a real file: this
// measures the verifier's per-function costs rather than its scaling within a
// function body.
auto BM_VerifyDominanceManyFunctions(benchmark::State& state) -> void {
  FileBuilder file;
  for (int i = 0, n = state.range(0); i != n; ++i) {
    file.AddFunction(BuildDiamonds(file, /*num_blocks=*/5));
  }
  RunBenchmark(state, file);
}
BENCHMARK(BM_VerifyDominanceManyFunctions)
    ->Name("BM_VerifyDominance/ManyFunctions")
    ->Apply(BodySizes);

// Verifies many small generic functions, each with one resolved specific.
// Every generic function's body is verified once per specific of its generic,
// so this watches for work that's quadratic in the number of generics in the
// file rather than linear in the number of specifics.
auto BM_VerifyDominanceManyGenericFunctions(benchmark::State& state) -> void {
  FileBuilder file;
  for (int i = 0, n = state.range(0); i != n; ++i) {
    file.AddFunction(BuildDiamonds(file, /*num_blocks=*/5), file.AddGeneric());
  }
  RunBenchmark(state, file);
}
BENCHMARK(BM_VerifyDominanceManyGenericFunctions)
    ->Name("BM_VerifyDominance/ManyGenericFunctions")
    ->Apply(BodySizes);

}  // namespace
}  // namespace Carbon::SemIR
