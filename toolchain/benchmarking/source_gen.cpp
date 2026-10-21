// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/benchmarking/source_gen.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <numeric>
#include <string>
#include <utility>

#include "absl/random/seed_sequences.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Testing {

auto SourceGen::Global() -> SourceGen& {
  static SourceGen global_gen;
  return global_gen;
}

SourceGen::SourceGen(Language language) : language_(language) {}

// Heuristic numbers used in synthesizing various identifier sequences.
static constexpr int MinClassNameLength = 5;
static constexpr int MinMemberNameLength = 4;

// The length of every local variable name. How many times a definition spells
// each local depends on the local's position, so a single length keeps the byte
// count from depending on which name lands where. Locals are shorter than class
// names, so they can't shadow a class that the definition names. This is
// hard-coded rather than in `ClassParams` because seed-independent output sizes
// require a single length `< MinClassNameLength`; it would only need to be
// parameterized to benchmark identifier-length sensitivity specifically for
// local variables.
static constexpr int LocalNameLength = 3;

// The number of function and method declarations in each class.
static auto NumDeclsPerClass(const SourceGen::ClassParams& params) -> int {
  return params.public_function_decls + params.public_method_decls +
         params.private_function_decls + params.private_method_decls;
}

// The number of inline definitions of every kind in each class.
static auto NumInlineDefsPerClass(const SourceGen::ClassParams& params) -> int {
  return params.inline_function_defs + params.inline_getters +
         params.inline_predicates + params.inline_forwarders;
}

// Returns the fixed types that satisfy `eligible`, each repeated `weight` times
// and interleaved with the others.
static auto WeightedFixedTypes(
    const SourceGen::TypeUseParams& params,
    llvm::function_ref<
        auto(const SourceGen::TypeUseParams::FixedTypeWeight&)->bool>
        eligible)
    -> llvm::SmallVector<const SourceGen::TypeUseParams::FixedTypeWeight*> {
  llvm::SmallVector<const SourceGen::TypeUseParams::FixedTypeWeight*> weighted;
  for (int round = 0;; ++round) {
    int size = weighted.size();
    for (const auto& fw : params.fixed_type_weights) {
      if (fw.weight > round && eligible(fw)) {
        weighted.push_back(&fw);
      }
    }
    if (static_cast<int>(weighted.size()) == size) {
      return weighted;
    }
  }
}

// The kinds of step in a definition plan. A plan fixes how many steps of each
// kind a definition has at each depth of nesting, and the definition arranges
// them randomly. A step has the same size wherever it lands at its depth, so
// the arrangement doesn't change the definition's size.
//
// TODO: Plans read each parameter once, read class-typed locals only with
// `Checksum`, and make no calls. Real definitions read parameters and fields
// throughout, and make calls in most statements, many of them to methods of
// their locals. Add those statements and expressions, fixing each one's
// spelling before the shuffle as the type use pools do, so that sizes stay the
// same for every seed.
enum class PlanStepKind : uint8_t {
  // `var <local>: i32 = <expr>;`, whose local is available to be read and
  // updated by subsequent steps.
  ScalarLocal,
  // `var <local>: <class> = <class>.Make();`, whose local is read with
  // `Checksum` by the next step and is available to be read (but not updated)
  // by other steps.
  ClassTypeLocal,
  // A declaration that destructures a tuple into two locals, such as
  // `let (<local>: i32, <local>: i64) = (<expr>, <expr> as i64);`, whose locals
  // are read by the next step and whose `i32` locals are available to be read
  // (and updated, when `var`) by subsequent steps. `form` selects among
  // `DestructureForms`.
  Destructure,
  // `<local> = <expr>;`
  Update,
  // `if (<cond>) {`, a block, and `}`.
  If,
  // `if (<cond>) {`, a block, `} else {`, a block, and `}`.
  IfElse,
  // `if (<cond>) {`, a block, `} else if (<cond>) {`, a block, and `}`.
  ElseIf,
  // `while (<cond>) {`, a block, and `}`.
  While,
  // `if (<cond>) {`, `return <value>;`, and `}`.
  GuardReturn,
  // `if (<cond>) {`, `break;` or `continue;`, and `}`, in a loop's block.
  GuardExit,
  // `<local> = <local> + <reads>;`, where the reads apply a consumer template
  // to each parameter, and read `self` in a method. Definitions add one of
  // these to their plan's steps at depth 0.
  ReadParams,
};

// The number of blocks that a step of `kind` opens.
static auto NumPlanBlocks(PlanStepKind kind) -> int {
  switch (kind) {
    case PlanStepKind::If:
    case PlanStepKind::While:
      return 1;
    case PlanStepKind::IfElse:
    case PlanStepKind::ElseIf:
      return 2;
    default:
      return 0;
  }
}

// Whether a step of `kind` declares locals, which the step after it reads.
static auto IsLocalDeclaration(PlanStepKind kind) -> bool {
  return kind == PlanStepKind::ScalarLocal ||
         kind == PlanStepKind::ClassTypeLocal ||
         kind == PlanStepKind::Destructure;
}

// A step in a definition plan. `form` selects among the kind's spellings.
struct PlanStep {
  PlanStepKind kind;
  int form;
};

// The expressions and conditions in plan steps. `{0}` and `{1}` are locals,
// which all have the same length. Conditions compare a single local, so that
// they can't compare a local with itself.
static constexpr llvm::StringLiteral PlanExprs[] = {
    "{0} + 1", "{0} * 2", "{0} - {1}", "({0} % 7) + {1}"};
static constexpr llvm::StringLiteral PlanConds[] = {"{0} > 3", "{0} < 9",
                                                    "{0} == 2", "{0} != 5"};

// A form of `Destructure` step. `{0}` and `{1}` are operands, and `{a}` and
// `{b}` the two locals, which `read` reads as an `i32` expression. Later steps
// can also read the locals that are `i32`s, and update them when they're
// `var`s.
struct DestructureForm {
  llvm::StringLiteral carbon_decl;
  llvm::StringLiteral cpp_decl;
  llvm::StringLiteral carbon_read;
  llvm::StringLiteral cpp_read;
  bool is_var;
  bool a_is_i32;
  bool b_is_i32;
};
static constexpr DestructureForm DestructureForms[] = {
    {.carbon_decl = "let ({a}: i32, {b}: i32) = ({0} + 1, {1} * 2);",
     .cpp_decl = "const auto [{a}, {b}] = std::pair{{0} + 1, {1} * 2};",
     .carbon_read = "({a} + {b})",
     .cpp_read = "({a} + {b})",
     .is_var = false,
     .a_is_i32 = true,
     .b_is_i32 = true},
    {.carbon_decl = "let ({a}: i32, {b}: i64) = ({0}, {1} as i64);",
     .cpp_decl = "const auto [{a}, {b}] = "
                 "std::pair{{0}, static_cast<std::int64_t>({1})};",
     .carbon_read = "({a} + ({b} as i32))",
     .cpp_read = "({a} + static_cast<int>({b}))",
     .is_var = false,
     .a_is_i32 = true,
     .b_is_i32 = false},
    {.carbon_decl = "var ({a}: bool, {b}: i32) = ({0} > 3, {1});",
     .cpp_decl = "auto [{a}, {b}] = std::pair{{0} > 3, {1}};",
     .carbon_read = "((if {a} then 1 else 0) + {b})",
     .cpp_read = "(({a} ? 1 : 0) + {b})",
     .is_var = true,
     .a_is_i32 = false,
     .b_is_i32 = true},
};

// The steps of a definition plan at each depth. The statement that a guard
// returns or exits with is one depth below the guard, and isn't listed.
struct DefPlan {
  llvm::SmallVector<llvm::SmallVector<PlanStep>> levels;
  // The number of lines in the plan's steps.
  int lines = 0;
  // The number of local variables that the plan's steps declare.
  int locals = 0;
  // The number of class-typed locals among `locals`.
  int class_type_locals = 0;
  // The class names of the plan's class-typed locals, with `""` for `Self`.
  llvm::SmallVector<llvm::StringRef> class_type_local_types;
};

// The internal shape calibration of definition plans, which approximates the
// function definitions in a sample of the Carbon toolchain's, LLVM's, and
// Clang's implementation files, as classified with libclang (excluding
// definitions with a `switch`).
//
// Why these constants are not in `DenseDeclParams` / `ClassParams`:
// `DenseDeclParams` and `ClassParams` parameterize the high-level shape of a
// generated file: what declarations and definitions a class contains, what
// types they use, and the distribution of definition sizes
// (`out_of_line_def_sizes` and `inline_function_def_sizes`). Once a
// definition's statement count is chosen,
// the constants below govern its internal statement mix and control-flow
// nesting (`if`/`else`/`while` blocks, guard returns/exits, and local variable
// density) so that generated definitions of any size have a realistic internal
// shape without requiring callers to configure dozens of coupled internal
// ratios.
//
// Criteria for adding any of these constants to `DenseDeclParams` or
// `ClassParams`: promote a constant to the parameterization when a benchmark or
// test needs to isolate the compiler's sensitivity to a specific structural
// dimension inside definitions (such as nesting depth, branching factor, or
// local variable density) independently of total statement count.
//
// All rates and bucket distributions below are integer percentages (with
// multi-bucket rows summing to 100), so `ComposeDefPlan` can scale counts with
// integer arithmetic (`scale_by_percent`).
//
// By the plan's size, the percentages of plans with each maximum depth, and
// the number of guard returns at a depth as a percentage of the statements at
// the next depth.
struct PlanShapeRow {
  int min_statements;
  std::array<int, 6> max_depth_percents;
  int guard_return_percent;
};
static constexpr PlanShapeRow PlanShapeTable[] = {
    {.min_statements = 0,
     .max_depth_percents = {100, 0, 0, 0, 0, 0},
     .guard_return_percent = 0},
    {.min_statements = 2,
     .max_depth_percents = {25, 62, 13, 0, 0, 0},
     .guard_return_percent = 65},
    {.min_statements = 5,
     .max_depth_percents = {8, 64, 25, 3, 0, 0},
     .guard_return_percent = 47},
    {.min_statements = 8,
     .max_depth_percents = {7, 46, 34, 11, 2, 0},
     .guard_return_percent = 32},
    {.min_statements = 13,
     .max_depth_percents = {2, 36, 39, 18, 4, 1},
     .guard_return_percent = 21},
    {.min_statements = 20,
     .max_depth_percents = {2, 21, 42, 25, 7, 3},
     .guard_return_percent = 16},
    {.min_statements = 29,
     .max_depth_percents = {0, 11, 32, 35, 17, 5},
     .guard_return_percent = 12},
    {.min_statements = 43,
     .max_depth_percents = {0, 2, 23, 42, 21, 12},
     .guard_return_percent = 10},
    {.min_statements = 67,
     .max_depth_percents = {0, 5, 4, 29, 30, 32},
     .guard_return_percent = 12},
};
// By a plan's maximum depth, the percentage of its statements at each depth
// (summing to 100). Every definition also has three statements at depth 0, so
// plans have a lower share there than real definitions.
static constexpr std::array<int, 6> PlanDepthPercents[] = {
    {100, 0, 0, 0, 0, 0},   {60, 40, 0, 0, 0, 0},   {38, 45, 17, 0, 0, 0},
    {25, 38, 27, 10, 0, 0}, {17, 29, 28, 18, 8, 0}, {13, 25, 24, 19, 13, 6},
};
// The percentage of loop blocks that have a guard exit (`break` or `continue`).
static constexpr int LoopExitPercent = 35;
// The percentage of non-guard statements at the next depth that open a new
// `if`, `else`, or `while` block (~2.1 statements per block).
static constexpr int NewBlockPercent = 48;
// The percentages of those blocks that are a loop's, and that are one of the
// two blocks of an `if` with an `else` or `else if`.
static constexpr int LoopBlockPercent = 15;
static constexpr int ElseBlockPercent = 30;
// The percentage of those two-block `if` statements that use `else if` rather
// than `else`.
static constexpr int ElseIfPercent = 42;
// How many times more likely a loop block is than another block of the same
// size to get each further statement when arranging a definition.
static constexpr int LoopBlockWeight = 3;
// The percentages of the statements at depth 0, and deeper, that declare a new
// local variable.
static constexpr int TopLocalPercent = 35;
static constexpr int NestedLocalPercent = 25;
// The percentages of declarations that destructure a tuple, and of the rest
// that declare a local of a class type. Structured bindings are 0.2-1.9% of the
// declarations in the sample, and Carbon's tuples suggest the high end. Class
// types are 29-78% of the types of local variables, 49% on average.
static constexpr int DestructurePercent = 2;
static constexpr int ClassTypeLocalPercent = 50;

// Scales `count` by `percent`%, rounding up with probability equal to the
// remainder (`(count * percent % 100) / 100.0`).
using ScaleByPercentFn = llvm::function_ref<auto(int count, int percent)->int>;

// Returns the maximum depth of the `variant`th plan with `statements`
// statements. The variants of a size follow the distribution in its row of
// `PlanShapeTable`: multiples of the golden ratio (`std::numbers::phi`) mod 1.0
// in `double` spread their choices evenly across `[0, 100)` without needing a
// fixed-point rational approximation.
static auto ChoosePlanMaxDepth(const PlanShapeRow& shape, int statements,
                               int variant) -> int {
  int choice = static_cast<int>(
      std::fmod((variant + 0.5) * std::numbers::phi, 1.0) * 100);
  int max_depth = 0;
  for (int percent : shape.max_depth_percents) {
    choice -= percent;
    if (choice < 0) {
      break;
    }
    ++max_depth;
  }
  return std::min(max_depth, statements - 1);
}

// Splits `statements` statements between depths 0 through `max_depth`,
// following `PlanDepthPercents` and giving each depth at least one.
static auto SplitPlanDepths(int statements, int max_depth,
                            ScaleByPercentFn scale_by_percent)
    -> llvm::SmallVector<int> {
  llvm::SmallVector<int> depth_counts;
  int cumulative_percent = 0;
  int assigned = 0;
  for (int percent :
       llvm::ArrayRef(PlanDepthPercents[max_depth]).take_front(max_depth)) {
    cumulative_percent += percent;
    int upto = scale_by_percent(statements, cumulative_percent);
    depth_counts.push_back(upto - assigned);
    assigned = upto;
  }
  depth_counts.push_back(statements - assigned);
  for (int& count : depth_counts) {
    if (count == 0) {
      ++count;
      --*llvm::max_element(depth_counts);
    }
  }
  return depth_counts;
}

// Composes the plan of a definition with `statements` statements, for the
// `variant`th definition of that size. Percentage-scaled counts round with a
// generator seeded with the size and variant, so that definitions of one size
// vary while their average follows the calibration, and nothing depends on the
// `SourceGen` seed.
static auto ComposeDefPlan(int statements, int variant) -> DefPlan {
  DefPlan plan;
  if (statements == 0) {
    return plan;
  }
  absl::SeedSeq seed_seq = {statements, variant};
  absl::BitGen rng(seed_seq);
  auto scale_by_percent = [&](int count, int percent) {
    return (count * percent + absl::Uniform<int>(rng, 0, 100)) / 100;
  };

  const PlanShapeRow* shape = PlanShapeTable;
  for (const PlanShapeRow& row : PlanShapeTable) {
    if (statements >= row.min_statements) {
      shape = &row;
    }
  }
  llvm::SmallVector<int> depth_counts = SplitPlanDepths(
      statements, ChoosePlanMaxDepth(*shape, statements, variant),
      scale_by_percent);

  plan.levels.resize(depth_counts.size());
  // Each kind's steps take its forms in turn.
  std::array<int, static_cast<int>(PlanStepKind::ReadParams) + 1> next_forms =
      {};
  auto add = [&](int depth, PlanStepKind kind, int count) {
    for (auto _ : llvm::seq(count)) {
      int& next_form = next_forms[static_cast<int>(kind)];
      plan.levels[depth].push_back({.kind = kind, .form = variant + next_form});
      ++next_form;
    }
  };

  // The statements at a depth that guards at the previous depth imply.
  int guard_children = 0;
  // The blocks at a depth, and how many of them are loop blocks.
  int blocks = 1;
  int loop_blocks = 0;
  int num_depths = depth_counts.size();
  for (int depth : llvm::seq(num_depths)) {
    int count = depth_counts[depth];
    int heads = count - guard_children;
    int next = depth + 1 < num_depths ? depth_counts[depth + 1] : 0;
    int guards = 0;
    int exits = 0;
    int ifs = 0;
    int pairs = 0;
    int loops = 0;
    int next_blocks = 0;
    if (next > 0) {
      // Guards' statements can't hold deeper ones, so leave a free statement at
      // the next depth when there are any.
      int min_free = depth + 2 < num_depths ? 1 : 0;
      exits = std::min({scale_by_percent(loop_blocks, LoopExitPercent), heads,
                        next - min_free});
      guards = std::min({scale_by_percent(next, shape->guard_return_percent),
                         heads - exits, next - exits - min_free});
      int free = next - guards - exits;
      // Keep a statement at this depth to hold the next depth's free
      // statements.
      if (free > 0 && heads == guards + exits) {
        if (guards > 0) {
          --guards;
        } else {
          --exits;
        }
        ++free;
      }
      if (free > 0) {
        next_blocks = std::max(1, scale_by_percent(free, NewBlockPercent));
        loops = scale_by_percent(next_blocks, LoopBlockPercent);
        pairs = std::min(scale_by_percent(next_blocks, ElseBlockPercent / 2),
                         (next_blocks - loops) / 2);
        ifs = next_blocks - loops - 2 * pairs;
        // Fit the statements that open blocks into this depth, merging blocks
        // when they don't fit.
        int room = heads - guards - exits;
        while (ifs + pairs + loops > room) {
          if (ifs >= 2) {
            ifs -= 2;
            ++pairs;
          } else if (ifs > 0) {
            --ifs;
            --next_blocks;
          } else if (loops > 0) {
            --loops;
            --next_blocks;
          } else {
            --pairs;
            next_blocks -= 2;
          }
        }
      }
    }

    int fixed = guards + exits + ifs + pairs + loops;
    // Each block at this depth ends with a step that isn't a local declaration.
    int max_locals = heads - fixed - std::max(0, blocks - fixed);
    int local_percent = depth == 0 ? TopLocalPercent : NestedLocalPercent;
    int locals = std::min(scale_by_percent(heads, local_percent), max_locals);
    int else_ifs = scale_by_percent(pairs, ElseIfPercent);
    int updates = heads - fixed - locals;
    int destructures = scale_by_percent(locals, DestructurePercent);
    int class_type_locals =
        scale_by_percent(locals - destructures, ClassTypeLocalPercent);
    add(depth, PlanStepKind::ScalarLocal,
        locals - destructures - class_type_locals);
    add(depth, PlanStepKind::ClassTypeLocal, class_type_locals);
    add(depth, PlanStepKind::Destructure, destructures);
    add(depth, PlanStepKind::Update, updates);
    add(depth, PlanStepKind::If, ifs);
    add(depth, PlanStepKind::IfElse, pairs - else_ifs);
    add(depth, PlanStepKind::ElseIf, else_ifs);
    add(depth, PlanStepKind::While, loops);
    add(depth, PlanStepKind::GuardReturn, guards);
    add(depth, PlanStepKind::GuardExit, exits);

    // Each step takes a line, and so does the `}` that ends each block and
    // guard, and the line between the blocks of a pair.
    plan.lines += count + ifs + loops + 2 * pairs + guards + exits;
    // Each destructuring declares two locals.
    plan.locals += locals + destructures;
    plan.class_type_locals += class_type_locals;
    guard_children = guards + exits;
    blocks = next_blocks;
    loop_blocks = loops;
  }
  return plan;
}

// Returns how many of `num_defs` definitions have each size in `sizes`, in
// proportion to its weight. The definitions left over from rounding down go to
// the sizes with the largest remainders, the earliest first on ties.
static auto ApportionDefSizes(int num_defs,
                              llvm::ArrayRef<SourceGen::DefSizeWeight> sizes)
    -> llvm::SmallVector<int> {
  int64_t total_weight = 0;
  for (const SourceGen::DefSizeWeight& size : sizes) {
    CARBON_CHECK(size.statements >= 0 && size.weight >= 0,
                 "Definition sizes and weights can't be negative.");
    total_weight += size.weight;
  }
  CARBON_CHECK(total_weight > 0,
               "Definition sizes need a positive total weight.");
  llvm::SmallVector<int> counts;
  llvm::SmallVector<int64_t> remainders;
  int left_over = num_defs;
  for (const SourceGen::DefSizeWeight& size : sizes) {
    int64_t share = static_cast<int64_t>(num_defs) * size.weight;
    counts.push_back(share / total_weight);
    remainders.push_back(share % total_weight);
    left_over -= counts.back();
  }
  llvm::SmallVector<int> order(sizes.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::stable_sort(order, [&](int lhs, int rhs) {
    return remainders[lhs] > remainders[rhs];
  });
  for (int index : llvm::ArrayRef(order).take_front(left_over)) {
    ++counts[index];
  }
  return counts;
}

// Builds the plans for `num_defs` definitions, with sizes following `sizes`,
// and appends them to `plans`. Each size's plans spread evenly through the
// order, so that any stretch of the order has a mix of sizes.
static auto BuildDefPlans(int num_defs,
                          llvm::ArrayRef<SourceGen::DefSizeWeight> sizes,
                          llvm::SmallVectorImpl<DefPlan>& plans) -> void {
  // `position` is a `double` in `(0, 1)` so that buckets with different counts
  // can be interleaved by sorting on `(variant + 0.5) / count` directly rather
  // than scaling to a common integer denominator.
  struct Entry {
    double position;
    int statements;
    int variant;
  };
  llvm::SmallVector<Entry> entries;
  entries.reserve(num_defs);
  llvm::SmallVector<int> counts = ApportionDefSizes(num_defs, sizes);
  for (auto [size, count] : llvm::zip_equal(sizes, counts)) {
    for (int variant : llvm::seq(count)) {
      entries.push_back({.position = (variant + 0.5) / count,
                         .statements = size.statements,
                         .variant = variant});
    }
  }
  llvm::stable_sort(entries, [](const Entry& lhs, const Entry& rhs) {
    return lhs.position < rhs.position;
  });
  plans.reserve(plans.size() + num_defs);
  for (const Entry& entry : entries) {
    plans.push_back(ComposeDefPlan(entry.statements, entry.variant));
  }
}

// Estimates the average number of lines in the plans of definitions with sizes
// following `sizes`. Uses `double` (like the other `EstimateAvg*Lines` helpers
// below) to accumulate fractional average line counts across variants and
// constructs before a single final rounding to a whole number of classes.
static auto EstimateAvgDefPlanLines(
    llvm::ArrayRef<SourceGen::DefSizeWeight> sizes) -> double {
  // Average over enough variants that their rounding evens out.
  constexpr int NumVariants = 16;
  double lines = 0.0;
  int64_t total_weight = 0;
  for (const SourceGen::DefSizeWeight& size : sizes) {
    int size_lines = 0;
    for (int variant : llvm::seq(NumVariants)) {
      size_lines += ComposeDefPlan(size.statements, variant).lines;
    }
    lines += static_cast<double>(size_lines) / NumVariants * size.weight;
    total_weight += size.weight;
  }
  CARBON_CHECK(total_weight > 0,
               "Definition sizes need a positive total weight.");
  return lines / total_weight;
}

// The shuffled state used to generate some number of classes.
//
// This state encodes everything used to generate class definitions. The state
// will be consumed until empty.
//
// Detailed comments for out-of-line methods are on their definitions.
class SourceGen::ClassGenState {
 public:
  ClassGenState(SourceGen& gen, int num_classes, const DenseDeclParams& params);

  auto public_function_param_counts() -> llvm::SmallVectorImpl<int>& {
    return public_function_param_counts_;
  }
  auto public_method_param_counts() -> llvm::SmallVectorImpl<int>& {
    return public_method_param_counts_;
  }
  auto private_function_param_counts() -> llvm::SmallVectorImpl<int>& {
    return private_function_param_counts_;
  }
  auto private_method_param_counts() -> llvm::SmallVectorImpl<int>& {
    return private_method_param_counts_;
  }

  auto inline_function_param_counts() -> llvm::SmallVectorImpl<int>& {
    return inline_function_param_counts_;
  }
  auto forwarder_param_counts() -> llvm::SmallVectorImpl<int>& {
    return forwarder_param_counts_;
  }
  // The plans of the out-of-line definitions, followed by those of inline
  // function definitions, starting at `inline_def_plans_start()`. A definition
  // uses the plan at the index of its return type's use among the plans of its
  // kind; see `BuildTypePool`.
  auto def_plans() -> llvm::ArrayRef<DefPlan> { return def_plans_; }
  auto inline_def_plans_start() -> int { return inline_def_plans_start_; }

  auto getter_field_types() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return getter_field_types_;
  }
  // The type of a field that a predicate tests, and the predicate template for
  // that type.
  struct PredicateField {
    llvm::StringRef type;
    llvm::StringRef predicate;
  };
  auto predicate_fields() -> llvm::SmallVectorImpl<PredicateField>& {
    return predicate_fields_;
  }

  auto class_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return class_names_;
  }
  auto decl_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return decl_names_;
  }
  auto field_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return field_names_;
  }
  auto param_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return param_names_;
  }

  auto inline_function_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return inline_function_names_;
  }
  auto inline_param_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return inline_param_names_;
  }
  auto local_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return local_names_;
  }
  auto accessed_field_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return accessed_field_names_;
  }
  auto forwarder_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return forwarder_names_;
  }
  auto forwarder_param_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return forwarder_param_names_;
  }

  auto AddValidTypeName(llvm::StringRef type_name) -> void {
    current_class_ = type_name;
    valid_type_names_.Insert(type_name);
  }

  // The names of all classes in the file. Parameter, member, and field names
  // exclude these so that they can't shadow a class that a definition names.
  auto class_name_set() -> const Set<llvm::StringRef>& {
    return class_name_set_;
  }

  // Emits an expression producing a value of `type`: a call to the class's
  // `Make` function, or the fixed type's value expression.
  auto ProduceValue(llvm::StringRef type, llvm::raw_ostream& os) -> void {
    if (class_name_set_.Contains(type)) {
      os << type << (is_cpp_ ? "()" : ".Make()");
    } else {
      os << fixed_value_.Lookup(type).value();
    }
  }

  // A use of a type drawn from a pool, and its index in the pool before the
  // pool was shuffled. A consumed use also has the consumer template that reads
  // it.
  struct TypeUse {
    llvm::StringRef name;
    llvm::StringRef consumer;
    int index = -1;
  };

  // Each category of type use draws from its own pool so that every slot in a
  // pool emits its type name (and any associated initializer, return value, or
  // parameter-read expression) the same number of times. This keeps the
  // generated file's byte count independent of how shuffling assigns types to
  // slots within a pool.

  // Return and parameter types of declarations without out-of-line
  // definitions (spelled once).
  auto GetDeclType() -> TypeUse { return GetValidTypeUse(decl_type_pool_); }
  // Parameter types of inline function definitions (spelled once in the
  // signature, and read once in the definition using a consumer template).
  auto GetInlineParamType() -> TypeUse {
    return GetValidTypeUse(inline_param_pool_);
  }
  // Return types of inline function definitions (spelled once in the
  // signature, and constructed in `return` statements).
  auto GetInlineReturnType() -> TypeUse {
    return GetValidReturnType(inline_return_pool_, /*is_inline=*/true);
  }
  // Return types of declarations with out-of-line definitions (spelled in both
  // the declaration and the definition, and constructed in `return`
  // statements).
  auto GetOutOfLineReturnType() -> TypeUse {
    return GetValidReturnType(out_of_line_return_pool_, /*is_inline=*/false);
  }
  // Parameter types of declarations with out-of-line definitions (spelled in
  // both the declaration and the definition, and read once in the definition
  // using a consumer template).
  auto GetOutOfLineParamType() -> TypeUse {
    return GetValidTypeUse(out_of_line_param_pool_);
  }
  // Field types (spelled once in the field declaration, and initialized with a
  // value expression in `Make` when the file has definitions).
  auto GetFieldType() -> TypeUse { return GetValidTypeUse(field_type_pool_); }
  // Return and parameter types of forwarders (spelled in both the forwarder
  // and the `<name>Impl` declaration it calls).
  auto GetForwardType() -> TypeUse {
    return GetValidTypeUse(forward_type_pool_);
  }

  auto has_defs() -> bool { return has_defs_; }
  auto define_decls_out_of_line() -> bool { return define_decls_out_of_line_; }
  auto type_pools_empty() -> bool {
    return decl_type_pool_.uses.empty() && inline_param_pool_.uses.empty() &&
           inline_return_pool_.uses.empty() && field_type_pool_.uses.empty() &&
           forward_type_pool_.uses.empty() &&
           out_of_line_return_pool_.uses.empty() &&
           out_of_line_param_pool_.uses.empty();
  }

 private:
  // A pool of type uses, removed as they are emitted. `GetValidTypeUse` resumes
  // its search at `last_index`.
  struct TypePool {
    llvm::SmallVector<TypeUse> uses;
    int last_index = 0;
  };

  auto GetValidTypeUse(
      TypePool& pool, llvm::function_ref<auto(const TypeUse&)->bool> is_valid =
                          [](const TypeUse&) { return true; }) -> TypeUse;
  auto GetValidReturnType(TypePool& pool, bool is_inline) -> TypeUse;

  auto BuildClassAndTypeNames(SourceGen& gen, int num_classes,
                              const ClassParams& class_params,
                              const TypeUseParams& type_use_params) -> void;
  auto BuildTypePool(SourceGen& gen, int num_types, int max_refs_per_class,
                     bool needs_value_expr, bool consumed,
                     bool exclude_last_class,
                     const TypeUseParams& type_use_params) -> TypePool;
  auto AssignClassTypeLocalTypes(SourceGen& gen, int defs_per_class,
                                 llvm::MutableArrayRef<DefPlan> plans) -> void;

  llvm::SmallVector<int> public_function_param_counts_;
  llvm::SmallVector<int> public_method_param_counts_;
  llvm::SmallVector<int> private_function_param_counts_;
  llvm::SmallVector<int> private_method_param_counts_;

  llvm::SmallVector<int> inline_function_param_counts_;
  llvm::SmallVector<int> forwarder_param_counts_;
  llvm::SmallVector<DefPlan> def_plans_;
  int inline_def_plans_start_;

  // Getters return a copy of their field, and `Make` initializes it, so getter
  // and predicate fields have fixed types with a value expression. These types
  // cycle through the eligible types in proportion to their weights, so that
  // their spellings don't depend on the seed.
  llvm::SmallVector<llvm::StringRef> getter_field_types_;
  llvm::SmallVector<PredicateField> predicate_fields_;

  llvm::SmallVector<llvm::StringRef> class_names_;
  // Out-of-line definitions spell declared function and method names a second
  // time, and `Make` does the same for field names, so these have separate
  // pools.
  llvm::SmallVector<llvm::StringRef> decl_names_;
  llvm::SmallVector<llvm::StringRef> field_names_;
  llvm::SmallVector<llvm::StringRef> param_names_;

  // Inline function names are shorter than `MinMemberNameLength`, so they can't
  // collide with class, member, or field names.
  llvm::SmallVector<llvm::StringRef> inline_function_names_;
  llvm::SmallVector<llvm::StringRef> inline_param_names_;
  llvm::SmallVector<llvm::StringRef> local_names_;
  // A getter or predicate spells its field's name again, and a forwarder's
  // call and the declaration it calls spell its name and parameter names
  // again, so these have separate pools.
  llvm::SmallVector<llvm::StringRef> accessed_field_names_;
  llvm::SmallVector<llvm::StringRef> forwarder_names_;
  llvm::SmallVector<llvm::StringRef> forwarder_param_names_;

  bool is_cpp_;
  // Whether the file has any function or method definitions. Then each class
  // also gets a `Make` function (a default constructor in C++), a `Checksum`
  // method, and a `tag` field.
  bool has_defs_;
  bool define_decls_out_of_line_;
  TypePool decl_type_pool_;
  TypePool inline_param_pool_;
  TypePool inline_return_pool_;
  TypePool field_type_pool_;
  TypePool forward_type_pool_;
  TypePool out_of_line_return_pool_;
  TypePool out_of_line_param_pool_;
  llvm::StringRef current_class_;
  Set<llvm::StringRef> valid_type_names_;

  Set<llvm::StringRef> class_name_set_;
  // The value expression and consumer templates of each fixed type, keyed by
  // its spelling. These refer into the `TypeUseParams`, which outlives
  // generation.
  Map<llvm::StringRef, llvm::StringRef> fixed_value_;
  Map<llvm::StringRef, llvm::ArrayRef<llvm::StringRef>> fixed_consumers_;
  llvm::ArrayRef<llvm::StringRef> class_consumers_;
};

// A helper to sum elements of a range.
template <typename T>
static auto Sum(const T& range) -> int {
  return std::accumulate(range.begin(), range.end(), 0);
}

// Given a number of class definitions and the params with which to generate
// them, builds the state that will be used while generating that many classes.
//
// We build the state first and across all the class definitions that will be
// generated so that we can distribute random components across all the
// definitions.
SourceGen::ClassGenState::ClassGenState(SourceGen& gen, int num_classes,
                                        const DenseDeclParams& params)
    : is_cpp_(gen.IsCpp()),
      has_defs_(NumInlineDefsPerClass(params.class_params) > 0 ||
                params.define_decls_out_of_line),
      define_decls_out_of_line_(params.define_decls_out_of_line) {
  const ClassParams& class_params = params.class_params;
  public_function_param_counts_ =
      gen.GetShuffledInts(num_classes * class_params.public_function_decls, 0,
                          class_params.public_function_decl_params.max_params);
  public_method_param_counts_ =
      gen.GetShuffledInts(num_classes * class_params.public_method_decls, 0,
                          class_params.public_method_decl_params.max_params);
  private_function_param_counts_ =
      gen.GetShuffledInts(num_classes * class_params.private_function_decls, 0,
                          class_params.private_function_decl_params.max_params);
  private_method_param_counts_ =
      gen.GetShuffledInts(num_classes * class_params.private_method_decls, 0,
                          class_params.private_method_decl_params.max_params);

  int num_inline_functions = num_classes * class_params.inline_function_defs;
  decl_names_ =
      gen.GetShuffledIdentifiers(num_classes * NumDeclsPerClass(class_params),
                                 /*min_length=*/MinMemberNameLength);
  field_names_ =
      gen.GetShuffledIdentifiers(num_classes * class_params.private_field_decls,
                                 /*min_length=*/MinMemberNameLength);
  int num_out_of_line_defs = define_decls_out_of_line_
                                 ? num_classes * NumDeclsPerClass(class_params)
                                 : 0;
  // Each out-of-line definition has one parameter beyond its random count; see
  // `BuildClassAndTypeNames`.
  int num_params = Sum(public_function_param_counts_) +
                   Sum(public_method_param_counts_) +
                   Sum(private_function_param_counts_) +
                   Sum(private_method_param_counts_) + num_out_of_line_defs;
  param_names_ = gen.GetShuffledIdentifiers(num_params);

  inline_function_param_counts_ =
      gen.GetShuffledInts(num_inline_functions, 0,
                          class_params.inline_function_decl_params.max_params);
  // Each inline definition has one parameter beyond its random count; see
  // `BuildClassAndTypeNames`.
  int num_inline_params =
      Sum(inline_function_param_counts_) + num_inline_functions;
  BuildDefPlans(num_out_of_line_defs, params.out_of_line_def_sizes, def_plans_);
  inline_def_plans_start_ = def_plans_.size();
  BuildDefPlans(num_inline_functions, class_params.inline_function_def_sizes,
                def_plans_);
  int num_locals = 0;
  for (const DefPlan& plan : def_plans_) {
    num_locals += plan.locals;
  }
  int num_getters = num_classes * class_params.inline_getters;
  int num_predicates = num_classes * class_params.inline_predicates;
  int num_forwarders = num_classes * class_params.inline_forwarders;
  inline_function_names_ = gen.GetShuffledIdentifiers(
      num_inline_functions + num_getters + num_predicates, /*min_length=*/2,
      /*max_length=*/MinMemberNameLength - 1);
  inline_param_names_ = gen.GetShuffledIdentifiers(num_inline_params);
  local_names_ = gen.GetShuffledIdentifiers(num_locals,
                                            /*min_length=*/LocalNameLength,
                                            /*max_length=*/LocalNameLength);
  accessed_field_names_ =
      gen.GetShuffledIdentifiers(num_getters + num_predicates,
                                 /*min_length=*/MinMemberNameLength);
  forwarder_param_counts_ = gen.GetShuffledInts(
      num_forwarders, 0, class_params.inline_forwarder_params.max_params);
  forwarder_names_ =
      gen.GetShuffledIdentifiers(num_forwarders, /*min_length=*/2,
                                 /*max_length=*/MinMemberNameLength - 1);
  forwarder_param_names_ =
      gen.GetShuffledIdentifiers(Sum(forwarder_param_counts_));

  BuildClassAndTypeNames(gen, num_classes, class_params,
                         params.type_use_params);
}

auto SourceGen::ClassGenState::GetValidTypeUse(
    TypePool& pool, llvm::function_ref<auto(const TypeUse&)->bool> is_valid)
    -> TypeUse {
  // Check that we don't completely wrap the type names by tracking where we
  // started.
  int initial_last_index = pool.last_index;

  // Now search the type uses, starting from the last used index, to find the
  // first valid one.
  for (;;) {
    if (pool.last_index == 0) {
      pool.last_index = pool.uses.size();
    }
    --pool.last_index;
    TypeUse& use = pool.uses[pool.last_index];
    if (valid_type_names_.Contains(use.name) && is_valid(use)) {
      // Found a valid type use, swap it with the back and pop that off.
      std::swap(pool.uses.back(), use);
      return pool.uses.pop_back_val();
    }

    // `BuildTypePool` caps the references to each class so that a valid type
    // use always remains.
    CARBON_CHECK(pool.last_index != initial_last_index,
                 "Failed to find a valid type name with {0} candidates, an "
                 "initial index of {1}, and with {2} classes left to emit!",
                 pool.uses.size(), initial_last_index, class_names_.size());
  }
}

auto SourceGen::ClassGenState::GetValidReturnType(TypePool& pool,
                                                  bool is_inline) -> TypeUse {
  int plans_start = is_inline ? inline_def_plans_start_ : 0;
  return GetValidTypeUse(pool, [&](const TypeUse& use) {
    // A definition spells its own class `Self` (represented here by the empty
    // string), never by class name. Inline definitions can only name classes
    // defined prior to `current_class_`, whereas out-of-line definitions follow
    // every class and can name any other class.
    const DefPlan& plan = def_plans_[plans_start + use.index];
    for (llvm::StringRef type : plan.class_type_local_types) {
      if (!type.empty() && (type == current_class_ ||
                            (is_inline && !valid_type_names_.Contains(type)))) {
        return false;
      }
    }
    return true;
  });
}

// Builds a shuffled pool of `num_types` type uses, mixing references to the
// classes with the fixed types to roughly match the weights in
// `type_use_params`. `max_refs_per_class` caps the references to each class.
//
// With `needs_value_expr`, the pool only includes fixed types that have a value
// expression (for initializing a field in `Make` or returning a value from a
// function definition). With `consumed`, each use gets one of its type's
// consumer templates, assigned round-robin per type before the shuffle so that
// the mix of templates doesn't depend on the seed. With `exclude_last_class`,
// the pool excludes the last class defined in the file (`class_names_[0]`):
// field types are drawn before the current class is valid, and return types are
// paired with definition plans whose non-`Self` class-typed locals cannot name
// the enclosing class (or, for inline definitions, any later class). Each use
// also records its pre-shuffle index so it can be paired deterministically with
// a definition plan.
//
// `valid_type_names_` must already contain the fixed type spellings.
auto SourceGen::ClassGenState::BuildTypePool(
    SourceGen& gen, int num_types, int max_refs_per_class,
    bool needs_value_expr, bool consumed, bool exclude_last_class,
    const TypeUseParams& type_use_params) -> TypePool {
  TypePool pool;
  if (num_types == 0) {
    return pool;
  }
  pool.uses.reserve(num_types);

  auto fixed_spelling = [&](const TypeUseParams::FixedTypeWeight& fw) {
    return gen.IsCpp() ? fw.cpp_spelling : fw.carbon_spelling;
  };
  auto fixed_usable = [&](const TypeUseParams::FixedTypeWeight& fw) {
    return !needs_value_expr ||
           !(gen.IsCpp() ? fw.cpp_value : fw.carbon_value).empty();
  };

  Map<llvm::StringRef, int> consumer_counters;
  auto append_use = [&](llvm::StringRef name) {
    llvm::StringRef consumer;
    if (consumed) {
      llvm::ArrayRef<llvm::StringRef> consumers =
          class_name_set_.Contains(name)
              ? class_consumers_
              : fixed_consumers_.Lookup(name).value();
      int& counter = consumer_counters.Insert(name, 0).value();
      consumer = consumers[counter++ % consumers.size()];
    }
    pool.uses.push_back({.name = name, .consumer = consumer});
  };

  int type_weight_sum = type_use_params.declared_types_weight;
  for (const auto& fixed_type_weight : type_use_params.fixed_type_weights) {
    if (fixed_usable(fixed_type_weight)) {
      type_weight_sum += fixed_type_weight.weight;
    }
  }

  // Compute the number of declared types used. We expect to have a decent
  // number of repeated names, so we repeatedly append the entire sequence of
  // class names until there is some remainder of names needed. The class
  // defined last has the first unshuffled name; see `BuildClassAndTypeNames`.
  llvm::ArrayRef<llvm::StringRef> classes = class_names_;
  if (exclude_last_class) {
    classes = classes.drop_front();
  }
  int num_classes = classes.size();
  int num_declared_types =
      num_classes == 0
          ? 0
          : num_types * type_use_params.declared_types_weight / type_weight_sum;
  int full_copies = num_classes == 0 ? 0 : num_declared_types / num_classes;
  int remainder = num_classes == 0 ? 0 : num_declared_types % num_classes;

  // Cap the references to each class so that `GetValidTypeUse` always finds a
  // valid type regardless of the shuffle order:
  // - For pools with `!exclude_last_class`, a class is added to
  //   `valid_type_names_` after its fields are chosen, so references to a
  //   class can only be used on its own functions/methods or in later classes.
  //   In particular, the last class defined in the file can only be referenced
  //   by its own functions/methods, so each class can be referenced at most
  //   `max_refs_per_class` times (the number of draws per class from this
  //   pool).
  // - For pools with `exclude_last_class` (`field_type_pool_`,
  //   `inline_return_pool_`, and `out_of_line_return_pool_`), a class cannot
  //   use its own type (or any later class's type, for fields and inline
  //   definitions). Excluding the last class defined in the file ensures every
  //   remaining class is valid during at least the last class's draws, and in
  //   general at most `k * max_refs_per_class` entries are invalid when `k`
  //   classes remain after the current class, so a valid entry is always
  //   available.
  //
  // The fixed types below replace any references removed by the cap, so the
  // pool's spellings and total byte count remain independent of the shuffle.
  if (full_copies >= max_refs_per_class) {
    full_copies = max_refs_per_class;
    remainder = 0;
  }

  for (auto _ : llvm::seq(full_copies)) {
    for (llvm::StringRef name : classes) {
      append_use(name);
    }
  }
  // Now append the remainder number of class names. This is where the class
  // names being un-shuffled is essential. We're going to have one extra
  // reference to some fraction of the class names and we want that to be a
  // stable subset.
  for (llvm::StringRef name : classes.slice(0, remainder)) {
    append_use(name);
  }
  num_declared_types = full_copies * num_classes + remainder;
  CARBON_CHECK(static_cast<int>(pool.uses.size()) == num_declared_types);

  // Use each fixed type weight to append the expected number of copies of that
  // type. This isn't exact however, and is designed to stop short.
  for (const auto& fixed_type_weight : type_use_params.fixed_type_weights) {
    if (!fixed_usable(fixed_type_weight)) {
      continue;
    }
    int num_fixed_type = num_types * fixed_type_weight.weight / type_weight_sum;
    for (auto _ : llvm::seq(num_fixed_type)) {
      append_use(fixed_spelling(fixed_type_weight));
    }
  }

  // If we need a tail of types to hit the exact number, simply round-robin
  // through the usable fixed types without any weighting. With reasonably large
  // numbers of types this won't distort the distribution in an interesting way
  // and is simpler than trying to scale the distribution down.
  while (static_cast<int>(pool.uses.size()) < num_types) {
    for (const auto& fixed_type_weight : type_use_params.fixed_type_weights) {
      if (static_cast<int>(pool.uses.size()) >= num_types) {
        break;
      }
      if (fixed_usable(fixed_type_weight)) {
        append_use(fixed_spelling(fixed_type_weight));
      }
    }
  }
  CARBON_CHECK(static_cast<int>(pool.uses.size()) == num_types);
  pool.last_index = num_types;

  for (auto [index, use] : llvm::enumerate(pool.uses)) {
    use.index = static_cast<int>(index);
  }
  std::shuffle(pool.uses.begin(), pool.uses.end(), gen.rng_);
  return pool;
}

// Assigns the types of class-typed locals for `plans`, which are paired by
// pre-shuffle index with a return-type pool built with
// `exclude_last_class = true` and `max_refs_per_class = defs_per_class`.
//
// In that pool, each index `< (num_classes - 1) * defs_per_class` has return
// type `classes[index % (num_classes - 1)]` (where `classes` excludes the last
// class defined in the file) or a fixed type, and the final `defs_per_class`
// indices have fixed return types. Assigning `classes[index % (num_classes -
// 1)]` (or `""` for `Self`) to the first group and `""` (`Self`) to the final
// `defs_per_class` plans ensures each plan depends on at most one class from
// `classes`, and at least `defs_per_class` plans are valid even in the first
// class defined in the file (where no prior classes exist). Because this runs
// before `class_names_` is shuffled, the multiset of class-local spellings and
// the file's total byte count are independent of the seed.
auto SourceGen::ClassGenState::AssignClassTypeLocalTypes(
    SourceGen& gen, int defs_per_class, llvm::MutableArrayRef<DefPlan> plans)
    -> void {
  int total_class_type_locals = 0;
  for (const DefPlan& plan : plans) {
    total_class_type_locals += plan.class_type_locals;
  }
  if (total_class_type_locals == 0) {
    return;
  }

  llvm::ArrayRef<llvm::StringRef> classes =
      llvm::ArrayRef(class_names_).drop_front();
  int num_classes = class_names_.size();
  int class_plans = static_cast<int>(classes.size()) * defs_per_class;
  int self_locals = 0;
  for (DefPlan& plan : plans.drop_front(class_plans)) {
    plan.class_type_local_types.append(plan.class_type_locals, "");
    self_locals += plan.class_type_locals;
  }

  int target_self_locals =
      (total_class_type_locals + num_classes - 1) / num_classes;
  int extra_self_locals = std::max(0, target_self_locals - self_locals);
  for (auto [index, plan] : llvm::enumerate(plans.take_front(class_plans))) {
    llvm::StringRef class_name = classes[index % classes.size()];
    plan.class_type_local_types.reserve(plan.class_type_locals);
    for (auto _ : llvm::seq(plan.class_type_locals)) {
      if (extra_self_locals > 0) {
        plan.class_type_local_types.push_back("");
        --extra_self_locals;
      } else {
        plan.class_type_local_types.push_back(class_name);
      }
    }
    std::shuffle(plan.class_type_local_types.begin(),
                 plan.class_type_local_types.end(), gen.rng_);
  }
}

// Builds the class names and the type use pools. Each pool caps the references
// to each class at the number of uses each class draws from it after its
// fields; see `BuildTypePool`.
auto SourceGen::ClassGenState::BuildClassAndTypeNames(
    SourceGen& gen, int num_classes, const ClassParams& class_params,
    const TypeUseParams& type_use_params) -> void {
  // Initially get the sequence of class names without shuffling so we can
  // compute our type pools from them prior to any shuffling.
  class_names_ =
      gen.GetUniqueIdentifiers(num_classes, /*min_length=*/MinClassNameLength);
  for (llvm::StringRef name : class_names_) {
    class_name_set_.Insert(name);
  }

  // Every fixed type needs a consumer template, since any of them can be the
  // type of a consumed parameter.
  for (const auto& fw : type_use_params.fixed_type_weights) {
    llvm::StringRef spelling =
        gen.IsCpp() ? fw.cpp_spelling : fw.carbon_spelling;
    valid_type_names_.Insert(spelling);
    fixed_value_.Insert(spelling, gen.IsCpp() ? fw.cpp_value : fw.carbon_value);
    llvm::ArrayRef<llvm::StringRef> consumers =
        gen.IsCpp() ? fw.cpp_consumers : fw.carbon_consumers;
    CARBON_CHECK(!consumers.empty(),
                 "Fixed type `{0}` needs at least one consumer template.",
                 spelling);
    for (llvm::StringRef consumer : consumers) {
      CARBON_CHECK(consumer.contains("{0}"),
                   "Fixed type `{0}` has a consumer template without a name "
                   "placeholder.",
                   spelling);
    }
    fixed_consumers_.Insert(spelling, consumers);
  }
  CARBON_CHECK(!type_use_params.class_consumers.empty(),
               "Class types need at least one consumer template.");
  class_consumers_ = type_use_params.class_consumers;

  int decls_per_class = NumDeclsPerClass(class_params);
  int num_decl_returns = num_classes * decls_per_class;
  int num_decl_params =
      Sum(public_function_param_counts_) + Sum(public_method_param_counts_) +
      Sum(private_function_param_counts_) + Sum(private_method_param_counts_);
  int num_inline_returns = num_classes * class_params.inline_function_defs;
  int num_inline_params = Sum(inline_function_param_counts_);
  int num_fields = num_classes * class_params.private_field_decls;

  // Each class draws `inline_function_defs` return types from this pool after
  // its fields. As for out-of-line definitions, a definition's guards also
  // return values of its return type, so each use's pre-shuffle index pairs it
  // with a definition plan.
  inline_return_pool_ =
      BuildTypePool(gen, num_inline_returns, class_params.inline_function_defs,
                    /*needs_value_expr=*/true, /*consumed=*/false,
                    /*exclude_last_class=*/true, type_use_params);
  AssignClassTypeLocalTypes(
      gen, class_params.inline_function_defs,
      llvm::MutableArrayRef(def_plans_).drop_front(inline_def_plans_start_));

  // Each class draws `private_field_decls` field types from this pool before
  // becoming a valid type itself. When the file has definitions, `Make`
  // initializes every field with a value expression.
  field_type_pool_ =
      BuildTypePool(gen, num_fields, class_params.private_field_decls,
                    /*needs_value_expr=*/has_defs_, /*consumed=*/false,
                    /*exclude_last_class=*/true, type_use_params);

  // Each inline definition has one parameter beyond its random count, so that
  // each class draws at least `inline_function_defs` parameter types from this
  // pool, and the pool can include class types.
  int num_inline_extra_params = num_inline_returns;
  inline_param_pool_ =
      BuildTypePool(gen, num_inline_params + num_inline_extra_params,
                    class_params.inline_function_defs,
                    /*needs_value_expr=*/false, /*consumed=*/true,
                    /*exclude_last_class=*/false, type_use_params);

  if (!define_decls_out_of_line_) {
    // Each class draws `decls_per_class` return types from this pool after its
    // fields.
    decl_type_pool_ =
        BuildTypePool(gen, num_decl_returns + num_decl_params, decls_per_class,
                      /*needs_value_expr=*/false, /*consumed=*/false,
                      /*exclude_last_class=*/false, type_use_params);
  } else {
    // Each class draws `decls_per_class` return types from this pool after its
    // fields. A definition's guards also return values of its return type, so
    // each use's pre-shuffle index pairs it with the definition plan at the
    // same index.
    out_of_line_return_pool_ =
        BuildTypePool(gen, num_decl_returns, decls_per_class,
                      /*needs_value_expr=*/true, /*consumed=*/false,
                      /*exclude_last_class=*/true, type_use_params);
    AssignClassTypeLocalTypes(
        gen, decls_per_class,
        llvm::MutableArrayRef(def_plans_).take_front(inline_def_plans_start_));
    // As for inline definitions, each declaration has one parameter beyond its
    // random count, so that each class draws at least `decls_per_class`
    // parameter types from this pool.
    int num_decl_extra_params = num_decl_returns;
    out_of_line_param_pool_ = BuildTypePool(
        gen, num_decl_params + num_decl_extra_params, decls_per_class,
        /*needs_value_expr=*/false, /*consumed=*/true,
        /*exclude_last_class=*/false, type_use_params);
  }

  // Each class draws `inline_forwarders` return types from this pool after its
  // fields.
  forward_type_pool_ =
      BuildTypePool(gen,
                    num_classes * class_params.inline_forwarders +
                        Sum(forwarder_param_counts_),
                    class_params.inline_forwarders,
                    /*needs_value_expr=*/false, /*consumed=*/false,
                    /*exclude_last_class=*/false, type_use_params);

  auto value = [&](const TypeUseParams::FixedTypeWeight& fw) {
    return gen.IsCpp() ? fw.cpp_value : fw.carbon_value;
  };
  auto predicate = [&](const TypeUseParams::FixedTypeWeight& fw) {
    return gen.IsCpp() ? fw.cpp_predicate : fw.carbon_predicate;
  };
  auto spelling = [&](const TypeUseParams::FixedTypeWeight& fw) {
    return gen.IsCpp() ? fw.cpp_spelling : fw.carbon_spelling;
  };
  int num_getters = num_classes * class_params.inline_getters;
  auto getter_types = WeightedFixedTypes(
      type_use_params, [&](const auto& fw) { return !value(fw).empty(); });
  CARBON_CHECK(num_getters == 0 || !getter_types.empty(),
               "Getters need a fixed type with a value expression.");
  for (int i : llvm::seq(num_getters)) {
    getter_field_types_.push_back(
        spelling(*getter_types[i % getter_types.size()]));
  }
  std::shuffle(getter_field_types_.begin(), getter_field_types_.end(),
               gen.rng_);
  int num_predicates = num_classes * class_params.inline_predicates;
  auto predicate_types =
      WeightedFixedTypes(type_use_params, [&](const auto& fw) {
        return !value(fw).empty() && !predicate(fw).empty();
      });
  CARBON_CHECK(num_predicates == 0 || !predicate_types.empty(),
               "Predicates need a fixed type with a value expression and a "
               "predicate template.");
  for (int i : llvm::seq(num_predicates)) {
    const auto& fw = *predicate_types[i % predicate_types.size()];
    predicate_fields_.push_back(
        {.type = spelling(fw), .predicate = predicate(fw)});
  }
  std::shuffle(predicate_fields_.begin(), predicate_fields_.end(), gen.rng_);

  // Classes are defined from the back, so the first unshuffled name is always
  // the class defined last, which field and return pools leave out; see
  // `BuildTypePool`.
  std::shuffle(class_names_.begin() + 1, class_names_.end(), gen.rng_);
}

// Some heuristic numbers used when formatting generated code. These heuristics
// are loosely based on what we expect to make Carbon code readable, and might
// not fit as well in C++, but we use the same heuristics across languages for
// simplicity and to make the output in different languages more directly
// comparable.
static constexpr int NumSingleLineFunctionParams = 3;
static constexpr int NumSingleLineMethodParams = 2;
static constexpr int MaxParamsPerLine = 4;

// `extra_params` is the number of parameters a function has beyond its random
// count.
static auto EstimateAvgFunctionDeclLines(SourceGen::FunctionDeclParams params,
                                         int extra_params = 0) -> double {
  // Currently model a uniform distribution [0, max] random parameters. Assume
  // a line break before the first parameter for >3 and after every 4th.
  int param_lines = 0;
  for (int num_params :
       llvm::seq_inclusive(extra_params, params.max_params + extra_params)) {
    if (num_params > NumSingleLineFunctionParams) {
      param_lines += (num_params + MaxParamsPerLine - 1) / MaxParamsPerLine;
    }
  }
  return 1.0 + static_cast<double>(param_lines) / (params.max_params + 1);
}

// See `EstimateAvgFunctionDeclLines` for the meaning of `extra_params`.
static auto EstimateAvgMethodDeclLines(SourceGen::MethodDeclParams params,
                                       int extra_params = 0) -> double {
  // Currently model a uniform distribution [0, max] random parameters. Assume
  // a line break before the first parameter for >2 and after every 4th slot,
  // where a Carbon method's `self` takes the first slot. C++ methods have no
  // `self`, so this slightly overestimates their lines.
  int param_lines = 0;
  for (int num_params :
       llvm::seq_inclusive(extra_params, params.max_params + extra_params)) {
    if (num_params > NumSingleLineMethodParams) {
      param_lines += 1 + num_params / MaxParamsPerLine;
    }
  }
  return 1.0 + static_cast<double>(param_lines) / (params.max_params + 1);
}

// Estimates the average number of lines in an inline function definition,
// excluding its comment. The definition has an accumulator line, a line reading
// the parameters, its plan's lines, a return line, and a closing brace line.
static auto EstimateAvgInlineFunctionDefLines(
    const SourceGen::ClassParams& params) -> double {
  constexpr int ExtraParams = 1;
  return EstimateAvgFunctionDeclLines(params.inline_function_decl_params,
                                      ExtraParams) +
         4.0 + EstimateAvgDefPlanLines(params.inline_function_def_sizes);
}

// Note that this should match the heuristics used when formatting.
// TODO: See top-level TODO about line estimates and formatting.
static auto EstimateAvgClassDefLines(
    const SourceGen::DenseDeclParams& dense_params) -> double {
  const SourceGen::ClassParams& params = dense_params.class_params;
  bool define_decls_out_of_line = dense_params.define_decls_out_of_line;
  // Comment line, and class open line.
  double avg = 2.0;

  // Declarations with out-of-line definitions have an extra parameter.
  int decl_extra_params = define_decls_out_of_line ? 1 : 0;

  // One comment line and blank line per function, plus the function lines.
  avg += (2.0 + EstimateAvgFunctionDeclLines(params.public_function_decl_params,
                                             decl_extra_params)) *
         params.public_function_decls;
  avg += (2.0 + EstimateAvgMethodDeclLines(params.public_method_decl_params,
                                           decl_extra_params)) *
         params.public_method_decls;
  avg += (2.0 + EstimateAvgFunctionDeclLines(
                    params.private_function_decl_params, decl_extra_params)) *
         params.private_function_decls;
  avg += (2.0 + EstimateAvgMethodDeclLines(params.private_method_decl_params,
                                           decl_extra_params)) *
         params.private_method_decls;
  avg += (2.0 + EstimateAvgInlineFunctionDefLines(params)) *
         params.inline_function_defs;
  // Getters and predicates are on one line.
  avg += 3.0 * (params.inline_getters + params.inline_predicates);
  // A forwarder's definition has a return line and a closing brace line, and
  // the declaration it calls has the same signature.
  double forwarder_signature_lines =
      EstimateAvgMethodDeclLines(params.inline_forwarder_params);
  avg += (2.0 + forwarder_signature_lines + 2.0 + 2.0 +
          forwarder_signature_lines) *
         params.inline_forwarders;

  bool has_defs = NumInlineDefsPerClass(params) > 0 || define_decls_out_of_line;

  // A blank line and all the fields (if any), including `tag` when the file has
  // definitions.
  double num_fields = params.private_field_decls + params.inline_getters +
                      params.inline_predicates + (has_defs ? 1.0 : 0.0);
  if (num_fields > 0) {
    avg += 1.0 + num_fields;
  }

  // `Make` and `Checksum` each have a blank line, a comment line, a signature
  // line, a return line, and a closing brace line. A C++ constructor puts its
  // initializers and empty definition on one line, and the class begins with a
  // `Self` alias.
  if (has_defs) {
    avg += 10.0;
  }

  // Each out-of-line definition has a blank line, a comment line, a signature
  // line, an accumulator line, a line reading the parameters, its plan's lines,
  // a return line, and a closing brace line.
  if (define_decls_out_of_line) {
    avg += (7.0 + EstimateAvgDefPlanLines(dense_params.out_of_line_def_sizes)) *
           NumDeclsPerClass(params);
  }

  // No need to account for the class close line, we have an extra blank line
  // count for the last of the above.
  return avg;
}

auto SourceGen::GenApiFileDenseDecls(int target_lines,
                                     const DenseDeclParams& params)
    -> std::string {
  RawStringOstream source;

  // Figure out how many classes fit in our target lines, each separated by a
  // blank line. We need to account the comment lines below to start the file.
  // Note that we want a blank line after our file comment block, so every class
  // needs a blank line.
  constexpr int NumFileCommentLines = 4;
  double avg_class_lines = EstimateAvgClassDefLines(params);
  CARBON_CHECK(target_lines > NumFileCommentLines + avg_class_lines,
               "Not enough target lines to generate a single class!");
  // Round to the nearest whole class. Truncating can leave the file nearly a
  // whole class short of the target, which matters when classes with
  // definitions run to hundreds of lines.
  int num_classes =
      std::lround((target_lines - NumFileCommentLines) / (avg_class_lines + 1));
  int expected_lines =
      NumFileCommentLines + num_classes * (avg_class_lines + 1);

  source << "// Generated " << (!IsCpp() ? "Carbon" : "C++")
         << " source file.\n";
  source << llvm::formatv(
                "// {0} target lines: {1} classes, {2} expected lines",
                target_lines, num_classes, expected_lines)
         << "\n";
  source << "//\n// Generating as "
         << (params.define_decls_out_of_line
                 ? "an implementation file with out-of-line definitions"
                 : "an API file with dense declarations")
         << ".\n";

  // Carbon uses an implicitly imported prelude to get builtin types, but C++
  // requires header files so include those.
  if (IsCpp()) {
    source << "\n";
    // Header for specific integer types like `std::int64_t`.
    source << "#include <cstdint>\n";
    // Header for `std::pair`.
    source << "#include <utility>\n";
  }

  auto class_gen_state = ClassGenState(*this, num_classes, params);
  llvm::SmallVector<OutOfLineDef, 0> out_of_line_defs;
  for (auto _ : llvm::seq(num_classes)) {
    source << "\n";
    GenerateClassDef(params.class_params, class_gen_state, source,
                     out_of_line_defs);
  }
  // Out-of-line definitions follow every class, as in an implementation file,
  // where all of the API's declarations are visible.
  for (const OutOfLineDef& def : out_of_line_defs) {
    source << "\n";
    GenerateOutOfLineDef(class_gen_state, def.class_name, def.sig, source);
  }

  // Make sure we consumed all the state.
  CARBON_CHECK(class_gen_state.public_function_param_counts().empty());
  CARBON_CHECK(class_gen_state.public_method_param_counts().empty());
  CARBON_CHECK(class_gen_state.private_function_param_counts().empty());
  CARBON_CHECK(class_gen_state.private_method_param_counts().empty());
  CARBON_CHECK(class_gen_state.inline_function_param_counts().empty());
  CARBON_CHECK(class_gen_state.forwarder_param_counts().empty());
  CARBON_CHECK(class_gen_state.getter_field_types().empty());
  CARBON_CHECK(class_gen_state.predicate_fields().empty());
  CARBON_CHECK(class_gen_state.class_names().empty());
  CARBON_CHECK(class_gen_state.type_pools_empty());
  // The identifier lengths in each name pool don't depend on the seed, so
  // emitting every name keeps the byte count seed-independent.
  CARBON_CHECK(class_gen_state.decl_names().empty());
  CARBON_CHECK(class_gen_state.field_names().empty());
  CARBON_CHECK(class_gen_state.param_names().empty());
  CARBON_CHECK(class_gen_state.inline_function_names().empty());
  CARBON_CHECK(class_gen_state.inline_param_names().empty());
  CARBON_CHECK(class_gen_state.local_names().empty());
  CARBON_CHECK(class_gen_state.accessed_field_names().empty());
  CARBON_CHECK(class_gen_state.forwarder_names().empty());
  CARBON_CHECK(class_gen_state.forwarder_param_names().empty());

  return source.TakeStr();
}

auto SourceGen::GetShuffledIdentifiers(int number, int min_length,
                                       int max_length, bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef> idents =
      GetIdentifiers(number, min_length, max_length, uniform);
  std::shuffle(idents.begin(), idents.end(), rng_);
  return idents;
}

auto SourceGen::GetShuffledUniqueIdentifiers(int number, int min_length,
                                             int max_length, bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  CARBON_CHECK(min_length >= 4,
               "Cannot trivially guarantee enough distinct, unique identifiers "
               "for lengths <= 3");
  llvm::SmallVector<llvm::StringRef> idents =
      GetUniqueIdentifiers(number, min_length, max_length, uniform);
  std::shuffle(idents.begin(), idents.end(), rng_);
  return idents;
}

auto SourceGen::GetIdentifiers(int number, int min_length, int max_length,
                               bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef> idents = GetIdentifiersImpl(
      number, min_length, max_length, uniform,
      [this](int length, int length_count,
             llvm::SmallVectorImpl<llvm::StringRef>& dest) {
        llvm::append_range(dest,
                           GetSingleLengthIdentifiers(length, length_count));
      });

  return idents;
}

auto SourceGen::GetUniqueIdentifiers(int number, int min_length, int max_length,
                                     bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  CARBON_CHECK(min_length >= 4,
               "Cannot trivially guarantee enough distinct, unique identifiers "
               "for lengths <= 3");
  llvm::SmallVector<llvm::StringRef> idents =
      GetIdentifiersImpl(number, min_length, max_length, uniform,
                         [this](int length, int length_count,
                                llvm::SmallVectorImpl<llvm::StringRef>& dest) {
                           AppendUniqueIdentifiers(length, length_count, dest);
                         });

  return idents;
}

auto SourceGen::GetSingleLengthIdentifiers(int length, int number)
    -> llvm::ArrayRef<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef>& idents =
      identifiers_by_length_.Insert(length, {}).value();

  if (static_cast<int>(idents.size()) < number) {
    idents.reserve(number);
    for (auto _ : llvm::seq<int>(idents.size(), number)) {
      auto ident_storage =
          llvm::MutableArrayRef(reinterpret_cast<char*>(storage_.Allocate(
                                    /*Size=*/length, /*Alignment=*/1)),
                                length);
      GenerateRandomIdentifier(ident_storage);
      llvm::StringRef new_id(ident_storage.data(), length);
      idents.push_back(new_id);
    }
    CARBON_CHECK(static_cast<int>(idents.size()) == number);
  }
  return llvm::ArrayRef(idents).slice(0, number);
}

static auto IdentifierStartChars() -> llvm::ArrayRef<char> {
  static llvm::SmallVector<char> chars = [] {
    llvm::SmallVector<char> chars;
    for (char c : llvm::seq_inclusive('A', 'Z')) {
      chars.push_back(c);
    }
    for (char c : llvm::seq_inclusive('a', 'z')) {
      chars.push_back(c);
    }
    return chars;
  }();
  return chars;
}

static auto IdentifierChars() -> llvm::ArrayRef<char> {
  static llvm::SmallVector<char> chars = [] {
    llvm::ArrayRef<char> start_chars = IdentifierStartChars();
    llvm::SmallVector<char> chars(start_chars.begin(), start_chars.end());
    chars.push_back('_');
    for (char c : llvm::seq_inclusive('0', '9')) {
      chars.push_back(c);
    }
    return chars;
  }();
  return chars;
}

static constexpr llvm::StringRef NonCarbonCppKeywords[] = {
    "asm",      "catch",  "do",  "double", "float", "int",  "long",     "new",
    "operator", "signed", "std", "this",   "throw", "try",  "typename", "unix",
    "unsigned", "using",  "xor", "M_E",    "M_El",  "M_PI", "NAN",      "NULL",
};

// Names that generated code declares itself, which random identifiers must
// avoid. For example, an inline function named `Make` would collide with its
// class's `Make` function.
static constexpr llvm::StringRef ReservedGeneratedNames[] = {"Make", "Checksum",
                                                             "acc", "tag"};

// Returns a random identifier string of the specified length.
//
// Ensures this is a valid identifier, avoiding any overlapping syntaxes or
// keywords both in Carbon and C++.
//
// This routine is somewhat expensive and so is useful to cache and reduce the
// frequency of calls. However, each time it is called it computes a completely
// new random identifier and so can be useful to eventually find a distinct
// identifier when needed.
auto SourceGen::GenerateRandomIdentifier(
    llvm::MutableArrayRef<char> dest_storage) -> void {
  llvm::ArrayRef<char> start_chars = IdentifierStartChars();
  llvm::ArrayRef<char> chars = IdentifierChars();

  llvm::StringRef ident(dest_storage.data(), dest_storage.size());
  do {
    dest_storage[0] =
        start_chars[absl::Uniform<int>(rng_, 0, start_chars.size())];
    for (int i : llvm::seq<int>(1, dest_storage.size())) {
      dest_storage[i] = chars[absl::Uniform<int>(rng_, 0, chars.size())];
    }
  } while (
      // TODO: Clean up and simplify this code. With some small refactorings and
      // post-processing we should be able to make this both easier to read and
      // less inefficient.
      llvm::any_of(
          Lex::TokenKind::KeywordTokens,
          [ident](auto token) { return ident == token.fixed_spelling(); }) ||
      llvm::is_contained(NonCarbonCppKeywords, ident) ||
      llvm::is_contained(ReservedGeneratedNames, ident) ||
      ident.ends_with("Impl") || ident.ends_with("_t") ||
      ident.ends_with("_MIN") || ident.ends_with("_MAX") ||
      ident.ends_with("_C") ||
      (llvm::is_contained({'i', 'u', 'f'}, ident[0]) &&
       llvm::all_of(ident.substr(1),
                    [](const char c) { return llvm::isDigit(c); })));
}

// Appends a number of unique, random identifiers with a particular length to
// the provided destination vector.
//
// Uses, and when necessary grows, a cached sequence of random identifiers with
// the specified length. Because these are cached, this is efficient to call
// repeatedly, but will not produce a different sequence of identifiers.
auto SourceGen::AppendUniqueIdentifiers(
    int length, int number, llvm::SmallVectorImpl<llvm::StringRef>& dest)
    -> void {
  auto& [count, unique_idents] =
      unique_identifiers_by_length_.Insert(length, {}).value();

  // See if we need to grow our pool of unique identifiers with the requested
  // length.
  if (count < number) {
    // We'll need to insert exactly the requested new unique identifiers. All
    // our other inserts will find an existing entry.
    unique_idents.GrowForInsertCount(count - number);

    // Generate the needed number of identifiers.
    for (auto _ : llvm::seq<int>(count, number)) {
      // Allocate stable storage for the identifier so we can form stable
      // `StringRef`s to it.
      auto ident_storage =
          llvm::MutableArrayRef(reinterpret_cast<char*>(storage_.Allocate(
                                    /*Size=*/length, /*Alignment=*/1)),
                                length);
      // Repeatedly generate novel identifiers of this length until we find a
      // new unique one.
      for (;;) {
        GenerateRandomIdentifier(ident_storage);
        auto result =
            unique_idents.Insert(llvm::StringRef(ident_storage.data(), length));
        if (result.is_inserted()) {
          break;
        }
      }
    }
    count = number;
  }
  // Append all the identifiers directly out of the set. We make no guarantees
  // about the relative order so we just use the non-deterministic order of the
  // set and avoid additional storage.
  for (llvm::StringRef ident : unique_idents.entries()) {
    if (number == 0) {
      break;
    }
    dest.push_back(ident);
    --number;
  }
  CARBON_CHECK(number == 0);
}

// An array of the counts that should be used for each identifier length to
// produce our desired distribution.
//
// Note that the zero-based index corresponds to a 1-based length, so the count
// for identifiers of length 1 is at index 0.
static constexpr std::array<int, 64> IdentifierLengthCounts = [] {
  std::array<int, 64> ident_length_counts;
  // For non-uniform distribution, we simulate a distribution roughly based on
  // the observed histogram of identifier lengths, but smoothed a bit and
  // reduced to small counts so that we cycle through all the lengths
  // reasonably quickly. We want sampling of even 10% of NumTokens from this
  // in a round-robin form to not be skewed overly much. This still inherently
  // compresses the long tail as we'd rather have coverage even though it
  // distorts the distribution a bit.
  //
  // The distribution here comes from a script that analyzes source code run
  // over a few directories of LLVM. The script renders a visual ascii-art
  // histogram along with the data for each bucket, and that output is
  // included in comments above each bucket size below to help visualize the
  // rough shape we're aiming for.
  //
  // 1 characters   [3976]  ███████████████████████████████▊
  ident_length_counts[0] = 40;
  // 2 characters   [3724]  █████████████████████████████▊
  ident_length_counts[1] = 40;
  // 3 characters   [4173]  █████████████████████████████████▍
  ident_length_counts[2] = 40;
  // 4 characters   [5000]  ████████████████████████████████████████
  ident_length_counts[3] = 50;
  // 5 characters   [1568]  ████████████▌
  ident_length_counts[4] = 20;
  // 6 characters   [2226]  █████████████████▊
  ident_length_counts[5] = 20;
  // 7 characters   [2380]  ███████████████████
  ident_length_counts[6] = 20;
  // 8 characters   [1786]  ██████████████▎
  ident_length_counts[7] = 18;
  // 9 characters   [1397]  ███████████▏
  ident_length_counts[8] = 12;
  // 10 characters  [ 739]  █████▉
  ident_length_counts[9] = 12;
  // 11 characters  [ 779]  ██████▎
  ident_length_counts[10] = 12;
  // 12 characters  [1344]  ██████████▊
  ident_length_counts[11] = 12;
  // 13 characters  [ 498]  ████
  ident_length_counts[12] = 5;
  // 14 characters  [ 284]  ██▎
  ident_length_counts[13] = 3;
  // 15 characters  [ 172]  █▍
  // 16 characters  [ 278]  ██▎
  // 17 characters  [ 191]  █▌
  // 18 characters  [ 207]  █▋
  for (int i = 14; i < 18; ++i) {
    ident_length_counts[i] = 2;
  }
  // 19 - 63 characters are all <100 but non-zero, and we map them to 1 for
  // coverage despite slightly over weighting the tail.
  for (int i = 18; i < 64; ++i) {
    ident_length_counts[i] = 1;
  }
  return ident_length_counts;
}();

// A template function that implements the common logic of `GetIdentifiers` and
// `GetUniqueIdentifiers`. Most parameters correspond to the parameters of those
// functions. Additionally, an `AppendFunc` callable is provided to implement
// the appending operation.
//
// The main functionality provided here is collecting the correct number of
// identifiers from each of the lengths in the range [min_length, max_length]
// and either in our default representative distribution or a uniform
// distribution.
auto SourceGen::GetIdentifiersImpl(int number, int min_length, int max_length,
                                   bool uniform,
                                   llvm::function_ref<AppendFn> append)
    -> llvm::SmallVector<llvm::StringRef> {
  CARBON_CHECK(min_length <= max_length);
  CARBON_CHECK(
      uniform || max_length <= 64,
      "Cannot produce a meaningful non-uniform distribution of lengths longer "
      "than 64 as those are exceedingly rare in our observed data sets.");

  llvm::SmallVector<llvm::StringRef> idents;
  idents.reserve(number);

  // First, compute the total weight of the distribution so we know how many
  // identifiers we'll get each time we collect from it. For a uniform
  // distribution every length has weight one, so the sum is simply the number
  // of lengths; this also avoids indexing the bounded `IdentifierLengthCounts`
  // table, which only covers lengths up to 64 and which uniform callers are
  // allowed to exceed.
  int num_lengths = max_length - min_length + 1;
  int count_sum = uniform ? num_lengths
                          : Sum(llvm::ArrayRef(IdentifierLengthCounts)
                                    .slice(min_length - 1, num_lengths));
  CARBON_CHECK(count_sum >= 1);

  int number_rem = number % count_sum;

  // Finally, walk through each length in the distribution.
  for (int length : llvm::seq_inclusive(min_length, max_length)) {
    // Scale how many identifiers we want of this length if computing a
    // non-uniform distribution. For uniform, we always take one.
    int scale = uniform ? 1 : IdentifierLengthCounts[length - 1];

    // Now we can compute how many identifiers of this length to request.
    int length_count = (number / count_sum) * scale;
    if (number_rem > 0) {
      int rem_adjustment = std::min(scale, number_rem);
      length_count += rem_adjustment;
      number_rem -= rem_adjustment;
    }
    append(length, length_count, idents);
  }
  CARBON_CHECK(number_rem == 0, "Unexpected number remaining: {0}", number_rem);
  CARBON_CHECK(static_cast<int>(idents.size()) == number,
               "Ended up with {0} identifiers instead of the requested {1}",
               idents.size(), number);

  return idents;
}

// Returns a shuffled sequence of integers in the range [min, max].
//
// The order of the returned integers is random, but each integer in the range
// appears the same number of times in the result, with the number of
// appearances rounded up for lower numbers and rounded down for higher numbers
// in order to exactly produce `number` results.
auto SourceGen::GetShuffledInts(int number, int min, int max)
    -> llvm::SmallVector<int> {
  llvm::SmallVector<int> ints;
  ints.reserve(number);

  // Evenly distribute to each value between min and max.
  int num_values = max - min + 1;
  for (int i : llvm::seq_inclusive(min, max)) {
    int i_count = number / num_values;
    i_count += i < (min + (number % num_values));
    ints.append(i_count, i);
  }
  CARBON_CHECK(static_cast<int>(ints.size()) == number);

  std::shuffle(ints.begin(), ints.end(), rng_);
  return ints;
}

// A helper to pop series of unique identifiers off a sequence of random
// identifiers that may have duplicates.
//
// This is particularly designed to work with the sequences of non-unique
// identifiers produced by `GetShuffledIdentifiers` with the important property
// that while popping off unique identifiers found in the shuffled list, we
// don't change the distribution of identifier lengths.
//
// The uniqueness is only per-instance of the class, and so an instance can be
// used to extract a series of names that share a scope.
//
// It works by scanning the sequence to extract each unique identifier found,
// swapping it to the back and popping it off the list. This does shuffle the
// order, but it isn't expected to do so in an interesting way.
//
// It also provides a fallback path in case there are no unique identifiers left
// which computes fresh, random identifiers with the same length as the next one
// in the sequence until a unique one is found.
//
// For simplicity of the fallback path, the lifetime of the identifiers produced
// is bound to the lifetime of the popper instance, and not the generator as a
// whole. If this is ever a problematic constraint, we can start copying
// fallback identifiers into the generator's storage.
class SourceGen::UniqueIdentifierPopper {
 public:
  // The popper never returns an identifier in `excluded`. `excluded` isn't
  // copied, so it must outlive the popper and not change while it's in use.
  explicit UniqueIdentifierPopper(
      SourceGen& gen, llvm::SmallVectorImpl<llvm::StringRef>& data,
      const Set<llvm::StringRef>* excluded = nullptr)
      : gen_(&gen), data_(&data), it_(data_->rbegin()), excluded_(excluded) {}

  // The identifiers this popper has returned.
  auto used() const -> const Set<llvm::StringRef>& { return set_; }

  // Prevents this popper from returning any of `names`, which are copied.
  auto Reserve(const Set<llvm::StringRef>& names) -> void {
    for (llvm::StringRef name : names.entries()) {
      set_.Insert(name);
    }
  }

  // Pop the next unique identifier that can be found in the data, or synthesize
  // one with a valid length. Always consumes exactly one identifier from the
  // data.
  //
  // Note that the lifetime of the underlying identifier is that of the popper
  // and not the underlying data.
  auto Pop() -> llvm::StringRef {
    for (auto end = data_->rend(); it_ != end; ++it_) {
      if (excluded_ && excluded_->Contains(*it_)) {
        continue;
      }
      auto insert = set_.Insert(*it_);
      if (!insert.is_inserted()) {
        continue;
      }

      if (it_ != data_->rbegin()) {
        std::swap(*data_->rbegin(), *it_);
      }
      CARBON_CHECK(insert.key() == data_->back());
      return data_->pop_back_val();
    }

    // Out of unique elements. Overwrite the back, preserving its length,
    // generating a new identifiers until we find a unique one and return that.
    // This ensures we continue to consume the structure and produce the same
    // size identifiers even in the fallback.
    int length = data_->pop_back_val().size();
    auto fallback_ident_storage =
        llvm::MutableArrayRef(reinterpret_cast<char*>(gen_->storage_.Allocate(
                                  /*Size=*/length, /*Alignment=*/1)),
                              length);
    for (;;) {
      gen_->GenerateRandomIdentifier(fallback_ident_storage);
      auto fallback_id = llvm::StringRef(fallback_ident_storage.data(), length);
      if (excluded_ && excluded_->Contains(fallback_id)) {
        continue;
      }
      if (set_.Insert(fallback_id).is_inserted()) {
        return fallback_id;
      }
    }
  }

 private:
  SourceGen* gen_;
  llvm::SmallVectorImpl<llvm::StringRef>* data_;
  llvm::SmallVectorImpl<llvm::StringRef>::reverse_iterator it_;
  const Set<llvm::StringRef>* excluded_;
  Set<llvm::StringRef> set_;
};

// Emits a parameter list, wrapping it as declarations do. Carbon methods begin
// with `self`.
auto SourceGen::EmitParams(bool is_method, llvm::ArrayRef<TypedName> params,
                           llvm::StringRef indent, llvm::raw_ostream& os)
    -> void {
  if (static_cast<int>(params.size()) >
      (is_method ? NumSingleLineMethodParams : NumSingleLineFunctionParams)) {
    os << "\n" << indent << "    ";
  }
  // For Carbon methods, `self` is the first explicit parameter. Its type is
  // omitted, which defaults it to `Self`.
  bool is_carbon_method = is_method && !IsCpp();
  if (is_carbon_method) {
    os << "self";
  }
  for (auto [i, param] : llvm::enumerate(params)) {
    // `self` occupies the first slot for Carbon methods, so shift the index
    // used for separators and line wrapping.
    int slot = static_cast<int>(i) + (is_carbon_method ? 1 : 0);
    if (slot > 0) {
      if ((slot % MaxParamsPerLine) == 0) {
        os << ",\n" << indent << "    ";
      } else {
        os << ", ";
      }
    }
    if (!IsCpp()) {
      os << param.name << ": " << param.type;
    } else {
      os << param.type << " " << param.name;
    }
  }
}

// Emits a function declaration with the given signature.
auto SourceGen::EmitFunctionDecl(llvm::StringRef name, bool is_private,
                                 bool is_method,
                                 llvm::ArrayRef<TypedName> params,
                                 llvm::StringRef return_type,
                                 llvm::StringRef indent, llvm::raw_ostream& os)
    -> void {
  os << indent << "// TODO: make better comment text\n";
  if (!IsCpp()) {
    os << indent << (is_private ? "private " : "") << "fn " << name;
  } else {
    os << indent;
    if (!is_method) {
      os << "static ";
    }
    os << "auto " << name;
  }
  os << "(";
  EmitParams(is_method, params, indent, os);
  os << ") -> " << return_type << ";\n";
}

// Generates a function declaration and writes it to the provided stream.
//
// The declaration can be configured with a function name, private modifier,
// whether it is a method, the parameter count, and how indented it is. Its
// parameter names and types come from `state`. With `captured`, the declaration
// has an out-of-line definition, and records its signature in `captured`.
auto SourceGen::GenerateFunctionDecl(ClassGenState& state, llvm::StringRef name,
                                     bool is_private, bool is_method,
                                     int param_count, llvm::StringRef indent,
                                     llvm::raw_ostream& os,
                                     FunctionSig* captured) -> void {
  bool out_of_line = captured != nullptr;
  // Add the extra parameter; see `BuildClassAndTypeNames`.
  if (out_of_line) {
    param_count += 1;
  }
  // Exclude class names so that a parameter can't shadow a class that an
  // out-of-line definition names.
  UniqueIdentifierPopper unique_param_names(*this, state.param_names(),
                                            &state.class_name_set());
  FunctionSig sig = {.name = name, .is_method = is_method};
  sig.params.reserve(param_count);
  sig.param_consumers.reserve(param_count);
  for (auto _ : llvm::seq(param_count)) {
    llvm::StringRef param = unique_param_names.Pop();
    ClassGenState::TypeUse type =
        out_of_line ? state.GetOutOfLineParamType() : state.GetDeclType();
    sig.params.push_back({.name = param, .type = type.name});
    sig.param_consumers.push_back(type.consumer);
  }
  ClassGenState::TypeUse return_type =
      out_of_line ? state.GetOutOfLineReturnType() : state.GetDeclType();
  sig.return_type = return_type.name;
  sig.def_plan = return_type.index;
  EmitFunctionDecl(name, is_private, is_method, sig.params, sig.return_type,
                   indent, os);
  if (captured) {
    *captured = std::move(sig);
  }
}

// Emits `format` with `name` substituted for its `{0}`.
static auto EmitTemplate(llvm::StringRef format, llvm::StringRef name,
                         llvm::raw_ostream& os) -> void {
  CARBON_CHECK(!format.empty());
  auto [prefix, suffix] = format.split("{0}");
  os << prefix << name << suffix;
}

// A step of an arranged definition. A step that opens blocks opens them at
// `first_block` and, for a second block, the index after it.
struct ArrangedStep {
  PlanStep plan;
  int first_block = -1;
};

// A block of an arranged definition: its steps, and whether it is a loop's
// block.
struct ArrangedBlock {
  llvm::SmallVector<int> steps;
  bool is_loop_block = false;
};

// A definition plan's steps, arranged in blocks. The first block is the
// definition's top-level block.
struct ArrangedDef {
  llvm::SmallVector<ArrangedStep> steps;
  llvm::SmallVector<ArrangedBlock> blocks;
};

// Randomly arranges the steps of `plan`, along with a `ReadParams` step at
// depth 0. Each block gets at least one step, and ends with a step that isn't
// a local declaration, so that the step after each local declaration can read
// its locals. Guard exits go in distinct loop blocks.
//
// The steps beyond the first in each block go in a block chosen with
// probability proportional to its steps so far, counting loop blocks
// `LoopBlockWeight` times. This gives a wider spread of block sizes than
// choosing uniformly, and larger loop blocks, as in real definitions.
static auto ArrangeDef(const DefPlan& plan, absl::BitGen& rng) -> ArrangedDef {
  ArrangedDef def;
  def.blocks.emplace_back();
  llvm::SmallVector<int> level_blocks = {0};
  int num_depths = std::max<int>(plan.levels.size(), 1);
  for (int depth : llvm::seq(num_depths)) {
    llvm::SmallVector<PlanStep> exits;
    llvm::SmallVector<PlanStep> others;
    llvm::SmallVector<PlanStep> locals;
    if (depth < static_cast<int>(plan.levels.size())) {
      for (PlanStep step : plan.levels[depth]) {
        switch (step.kind) {
          case PlanStepKind::GuardExit:
            exits.push_back(step);
            break;
          default:
            (IsLocalDeclaration(step.kind) ? locals : others).push_back(step);
            break;
        }
      }
    }
    if (depth == 0) {
      others.push_back({.kind = PlanStepKind::ReadParams, .form = 0});
    }
    std::shuffle(others.begin(), others.end(), rng);

    // The block of each step placed at this depth.
    llvm::SmallVector<int> placed_blocks;
    auto place = [&](int block, PlanStep step) {
      def.blocks[block].steps.push_back(def.steps.size());
      def.steps.push_back({.plan = step});
      placed_blocks.push_back(block);
    };
    llvm::SmallVector<int> loop_blocks;
    for (int block : level_blocks) {
      if (def.blocks[block].is_loop_block) {
        loop_blocks.push_back(block);
      }
    }
    CARBON_CHECK(exits.size() <= loop_blocks.size());
    std::shuffle(loop_blocks.begin(), loop_blocks.end(), rng);
    for (auto [block, exit] : llvm::zip(loop_blocks, exits)) {
      place(block, exit);
    }
    for (int block : level_blocks) {
      if (def.blocks[block].steps.empty()) {
        CARBON_CHECK(!others.empty(), "Not enough steps to end blocks.");
        place(block, others.pop_back_val());
      }
    }
    for (int block : loop_blocks) {
      placed_blocks.append(LoopBlockWeight - 1, block);
    }
    for (PlanStep step : llvm::concat<PlanStep>(others, locals)) {
      int block =
          placed_blocks[absl::Uniform<int>(rng, 0, placed_blocks.size())];
      place(block, step);
      if (def.blocks[block].is_loop_block) {
        placed_blocks.append(LoopBlockWeight - 1, block);
      }
    }

    for (int block : level_blocks) {
      llvm::SmallVector<int>& steps = def.blocks[block].steps;
      std::shuffle(steps.begin(), steps.end(), rng);
      auto is_local = [&](int step) {
        return IsLocalDeclaration(def.steps[step].plan.kind);
      };
      if (is_local(steps.back())) {
        llvm::SmallVector<int> non_locals;
        for (int i : llvm::seq<int>(steps.size())) {
          if (!is_local(steps[i])) {
            non_locals.push_back(i);
          }
        }
        std::swap(
            steps.back(),
            steps[non_locals[absl::Uniform<int>(rng, 0, non_locals.size())]]);
      }
    }

    // Open the blocks for the next depth.
    llvm::SmallVector<int> next_blocks;
    for (int block : level_blocks) {
      for (int i : llvm::seq<int>(def.blocks[block].steps.size())) {
        ArrangedStep& step = def.steps[def.blocks[block].steps[i]];
        int num_blocks = NumPlanBlocks(step.plan.kind);
        if (num_blocks == 0) {
          continue;
        }
        step.first_block = def.blocks.size();
        for (auto _ : llvm::seq(num_blocks)) {
          next_blocks.push_back(def.blocks.size());
          def.blocks.push_back(
              {.is_loop_block = step.plan.kind == PlanStepKind::While});
        }
      }
    }
    level_blocks = std::move(next_blocks);
  }
  CARBON_CHECK(level_blocks.empty());
  return def;
}

// Emits the steps of `def` after its `acc` declaration, indented beyond
// `indent`. Locals take their names from `locals` in turn, class-typed locals
// take their classes from `class_type_local_types` (with `""` for `Self`), the
// `ReadParams` step adds `reads`, and guards return `return_value`.
//
// The first operand of each step after a local declaration reads the locals
// that it declares, so that every local is read. Every other operand is a
// random `i32` local in scope, and the target of each update a random `var`
// one.
static auto EmitArrangedDef(
    const ArrangedDef& def, llvm::ArrayRef<llvm::StringRef> locals,
    llvm::ArrayRef<llvm::StringRef> class_type_local_types,
    llvm::StringRef reads, llvm::StringRef return_value, bool is_cpp,
    llvm::StringRef indent, absl::BitGen& rng, llvm::raw_ostream& os) -> void {
  // The `i32` locals in scope that later steps can update, and the ones that
  // they can only read.
  llvm::SmallVector<llvm::StringRef> var_scope = {"acc"};
  llvm::SmallVector<llvm::StringRef> let_scope;
  // An expression reading the locals that the last step declared.
  std::string pending = "acc";
  int next_local = 0;
  int next_class_type_local = 0;
  auto pick_local = [&]() {
    int index = absl::Uniform<int>(rng, 0, var_scope.size() + let_scope.size());
    return index < static_cast<int>(var_scope.size())
               ? var_scope[index]
               : let_scope[index - var_scope.size()];
  };
  auto pick_var = [&]() {
    return var_scope[absl::Uniform<int>(rng, 0, var_scope.size())];
  };
  auto emit_expr_to = [&](llvm::StringRef format, llvm::raw_ostream& out) {
    while (!format.empty()) {
      if (format.starts_with("{0}") || format.starts_with("{1}")) {
        out << (pending.empty() ? pick_local() : std::exchange(pending, {}));
        format = format.drop_front(3);
      } else {
        out << format.front();
        format = format.drop_front();
      }
    }
  };
  auto emit_expr = [&](llvm::StringRef format) { emit_expr_to(format, os); };
  auto emit_cond = [&](int form) {
    emit_expr(PlanConds[form % std::size(PlanConds)]);
  };
  auto indent_to = [&](int depth) {
    os << indent;
    os.indent(2 * depth);
  };

  // The blocks being emitted, with the index of the next step in each, and the
  // sizes of the scopes outside it.
  struct Frame {
    int block;
    int next_step;
    int outer_var_scope_size;
    int outer_let_scope_size;
  };
  auto open_block = [&](int block) -> Frame {
    return {.block = block,
            .next_step = 0,
            .outer_var_scope_size = static_cast<int>(var_scope.size()),
            .outer_let_scope_size = static_cast<int>(let_scope.size())};
  };
  llvm::SmallVector<Frame> stack = {open_block(0)};
  while (!stack.empty()) {
    Frame& frame = stack.back();
    int depth = stack.size();
    const ArrangedBlock& block = def.blocks[frame.block];
    if (frame.next_step == static_cast<int>(block.steps.size())) {
      // End the block, and start the second block of an `else` or `else if`.
      var_scope.truncate(frame.outer_var_scope_size);
      let_scope.truncate(frame.outer_let_scope_size);
      int ended = frame.block;
      stack.pop_back();
      if (stack.empty()) {
        break;
      }
      const Frame& owner_frame = stack.back();
      const ArrangedStep& owner =
          def.steps[def.blocks[owner_frame.block]
                        .steps[owner_frame.next_step - 1]];
      indent_to(depth - 1);
      if (NumPlanBlocks(owner.plan.kind) == 1 || ended != owner.first_block) {
        os << "}\n";
        continue;
      }
      if (owner.plan.kind == PlanStepKind::IfElse) {
        os << "} else {\n";
      } else {
        os << "} else if (";
        emit_cond(owner.plan.form + 1);
        os << ") {\n";
      }
      stack.push_back(open_block(ended + 1));
      continue;
    }

    const ArrangedStep& step = def.steps[block.steps[frame.next_step]];
    ++frame.next_step;
    PlanStep planned = step.plan;
    indent_to(depth);
    switch (planned.kind) {
      case PlanStepKind::ScalarLocal: {
        llvm::StringRef name = locals[next_local++];
        os << (is_cpp ? "int " : "var ") << name
           << (is_cpp ? " = " : ": i32 = ");
        emit_expr(PlanExprs[planned.form % std::size(PlanExprs)]);
        os << ";\n";
        var_scope.push_back(name);
        pending = name;
        break;
      }
      case PlanStepKind::ClassTypeLocal: {
        llvm::StringRef name = locals[next_local++];
        llvm::StringRef type = class_type_local_types[next_class_type_local++];
        if (type.empty()) {
          os << (is_cpp ? "Self " : "var ") << name
             << (is_cpp ? " = Self();\n" : ": Self = Make();\n");
        } else if (is_cpp) {
          os << type << " " << name << " = " << type << "();\n";
        } else {
          os << "var " << name << ": " << type << " = " << type << ".Make();\n";
        }
        RawStringOstream read;
        emit_expr_to(("({0} + " + name + ".Checksum())").str(), read);
        pending = read.TakeStr();
        break;
      }
      case PlanStepKind::Destructure: {
        const DestructureForm& form =
            DestructureForms[planned.form % std::size(DestructureForms)];
        llvm::StringRef a = locals[next_local++];
        llvm::StringRef b = locals[next_local++];
        auto substitute = [&](llvm::StringRef format) {
          std::string text = format.str();
          for (auto [placeholder, name] : {std::pair{"{a}", a}, {"{b}", b}}) {
            text.replace(text.find(placeholder), 3, name);
          }
          return text;
        };
        emit_expr(substitute(is_cpp ? form.cpp_decl : form.carbon_decl));
        os << "\n";
        auto& scope = form.is_var ? var_scope : let_scope;
        if (form.a_is_i32) {
          scope.push_back(a);
        }
        if (form.b_is_i32) {
          scope.push_back(b);
        }
        pending = substitute(is_cpp ? form.cpp_read : form.carbon_read);
        break;
      }
      case PlanStepKind::Update:
        os << pick_var() << " = ";
        emit_expr(PlanExprs[planned.form % std::size(PlanExprs)]);
        os << ";\n";
        break;
      case PlanStepKind::ReadParams:
        os << pick_var() << " = ";
        emit_expr("{0}");
        os << " + " << reads << ";\n";
        break;
      case PlanStepKind::If:
      case PlanStepKind::IfElse:
      case PlanStepKind::ElseIf:
      case PlanStepKind::While:
        os << (planned.kind == PlanStepKind::While ? "while (" : "if (");
        emit_cond(planned.form);
        os << ") {\n";
        stack.push_back(open_block(step.first_block));
        break;
      case PlanStepKind::GuardReturn:
      case PlanStepKind::GuardExit:
        os << "if (";
        emit_cond(planned.form);
        os << ") {\n";
        indent_to(depth + 1);
        if (planned.kind == PlanStepKind::GuardReturn) {
          os << "return " << return_value << ";\n";
        } else {
          os << (planned.form % 2 == 0 ? "break;\n" : "continue;\n");
        }
        indent_to(depth);
        os << "}\n";
        break;
    }
  }
}

// Generates the statements of a plan-based function or method definition (used
// for both `inline_function_defs` and out-of-line definitions) with signature
// `sig`, from its `acc` declaration through its `return`, indented beyond
// `indent`.
//
// Unlike the single-expression member idioms (`GenerateGetter`,
// `GeneratePredicate`, and `GenerateForwarder`), which have fixed signatures
// coupled to a class field or private `<name>Impl` declaration, a plan-based
// definition works with any `FunctionSig`: after declaring `acc`, it arranges
// its `DefPlan` steps randomly along with a depth-0 step that reads every
// parameter with its consumer template (and `self` in a method with
// `Checksum`), and ends with `return`. That parameter-read statement is on one
// line however long it is, which the line estimates assume.
//
// Both inline and out-of-line definitions spell their enclosing class `Self`
// (represented as the empty string) in class-typed locals. For other class
// types, inline definitions only use classes defined prior to the enclosing
// class, whereas out-of-line definitions follow every class and can use any
// other class; see `GetValidReturnType`.
auto SourceGen::GenerateFunctionDef(ClassGenState& state,
                                    const FunctionSig& sig,
                                    llvm::StringRef indent,
                                    llvm::raw_ostream& os) -> void {
  os << indent << (IsCpp() ? "  int acc = 0;\n" : "  var acc: i32 = 0;\n");

  RawStringOstream reads;
  llvm::ListSeparator read_sep(" + ");
  if (sig.is_method) {
    reads << read_sep << (IsCpp() ? "Checksum()" : "self.Checksum()");
  }
  for (auto [param, consumer] : llvm::zip(sig.params, sig.param_consumers)) {
    reads << read_sep;
    EmitTemplate(consumer, param.name, reads);
  }

  // Locals and parameters draw from the same identifiers of each length, so
  // exclude this function's parameter names: C++ rejects a local that
  // redeclares a parameter.
  const DefPlan& plan = state.def_plans()[sig.def_plan];
  Set<llvm::StringRef> param_names;
  for (const TypedName& param : sig.params) {
    param_names.Insert(param.name);
  }
  UniqueIdentifierPopper unique_local_names(*this, state.local_names(),
                                            &param_names);
  llvm::SmallVector<llvm::StringRef> locals;
  locals.reserve(plan.locals);
  for (auto _ : llvm::seq(plan.locals)) {
    locals.push_back(unique_local_names.Pop());
  }

  RawStringOstream return_value;
  state.ProduceValue(sig.return_type, return_value);
  std::string return_value_str = return_value.TakeStr();
  EmitArrangedDef(ArrangeDef(plan, rng_), locals, plan.class_type_local_types,
                  reads.TakeStr(), return_value_str, IsCpp(), indent, rng_, os);
  os << indent << "  return " << return_value_str << ";\n";
}

// Generates an out-of-line definition for the declaration `sig`, and writes it
// to the provided stream. The signature is on one line however long it is,
// which the line estimates assume.
auto SourceGen::GenerateOutOfLineDef(ClassGenState& state,
                                     llvm::StringRef class_name,
                                     const FunctionSig& sig,
                                     llvm::raw_ostream& os) -> void {
  os << "// TODO: make better comment text\n";
  if (!IsCpp()) {
    os << "fn " << class_name << "." << sig.name;
  } else {
    os << "auto " << class_name << "::" << sig.name;
  }

  os << "(";
  // For Carbon methods, `self` is the first parameter, as in the declaration.
  bool is_carbon_method = sig.is_method && !IsCpp();
  if (is_carbon_method) {
    os << "self";
  }
  bool first_param = !is_carbon_method;
  for (const TypedName& param : sig.params) {
    if (!first_param) {
      os << ", ";
    }
    first_param = false;
    if (!IsCpp()) {
      os << param.name << ": " << param.type;
    } else {
      os << param.type << " " << param.name;
    }
  }
  os << ") -> " << sig.return_type << " {\n";
  GenerateFunctionDef(state, sig, /*indent=*/"", os);
  os << "}\n";
}

// Generates an inline function definition and writes it to the provided
// stream. Its statements follow the plan paired with its return type; see
// `GenerateFunctionDef`.
auto SourceGen::GenerateInlineFunctionDef(ClassGenState& state,
                                          llvm::StringRef name, int param_count,
                                          llvm::StringRef indent,
                                          llvm::raw_ostream& os) -> void {
  // Add the extra parameter; see `BuildClassAndTypeNames`.
  param_count += 1;

  // Exclude class names so that a parameter can't shadow a class that the
  // definition names.
  UniqueIdentifierPopper unique_param_names(*this, state.inline_param_names(),
                                            &state.class_name_set());
  FunctionSig sig = {.name = name, .is_method = false};
  sig.params.reserve(param_count);
  sig.param_consumers.reserve(param_count);
  for (auto _ : llvm::seq(param_count)) {
    llvm::StringRef param = unique_param_names.Pop();
    ClassGenState::TypeUse type = state.GetInlineParamType();
    sig.params.push_back({.name = param, .type = type.name});
    sig.param_consumers.push_back(type.consumer);
  }
  ClassGenState::TypeUse return_type = state.GetInlineReturnType();
  sig.return_type = return_type.name;
  sig.def_plan = state.inline_def_plans_start() + return_type.index;

  os << indent << "// TODO: make better comment text\n";
  os << indent << (IsCpp() ? "static auto " : "fn ") << name << "(";
  EmitParams(/*is_method=*/false, sig.params, indent, os);
  os << ") -> " << sig.return_type << " {\n";
  GenerateFunctionDef(state, sig, indent, os);
  os << indent << "}\n";
}

// Generates an inline method that returns `field`.
auto SourceGen::GenerateGetter(llvm::StringRef name, TypedName field,
                               llvm::raw_ostream& os) -> void {
  os << "  // TODO: make better comment text\n";
  if (!IsCpp()) {
    os << "  fn " << name << "(self) -> " << field.type << " { return self."
       << field.name << "; }\n";
  } else {
    os << "  auto " << name << "() const -> " << field.type << " { return "
       << field.name << "; }\n";
  }
}

// Generates an inline method that returns the `predicate` template applied to
// `field`.
auto SourceGen::GeneratePredicate(llvm::StringRef name, TypedName field,
                                  llvm::StringRef predicate,
                                  llvm::raw_ostream& os) -> void {
  os << "  // TODO: make better comment text\n";
  if (!IsCpp()) {
    os << "  fn " << name << "(self) -> bool { return ";
    EmitTemplate(predicate, ("self." + field.name).str(), os);
  } else {
    os << "  auto " << name << "() const -> bool { return ";
    EmitTemplate(predicate, field.name, os);
  }
  os << "; }\n";
}

// Generates an inline method that passes its parameters on to `<name>Impl`, a
// private method with the same signature.
auto SourceGen::GenerateForwarder(llvm::StringRef name,
                                  llvm::ArrayRef<TypedName> params,
                                  llvm::StringRef return_type,
                                  llvm::raw_ostream& os) -> void {
  os << "  // TODO: make better comment text\n";
  os << "  " << (IsCpp() ? "auto " : "fn ") << name << "(";
  EmitParams(/*is_method=*/true, params, /*indent=*/"  ", os);
  os << ") -> " << return_type << " {\n";
  os << "    return " << (IsCpp() ? "" : "self.") << name << "Impl(";
  llvm::ListSeparator sep;
  for (const TypedName& param : params) {
    os << sep << param.name;
  }
  os << ");\n  }\n";
}

// Generates a class's `Make` function, which returns a value of the class with
// every field initialized. C++ classes idiomatically initialize with a
// constructor, so C++ gets a default constructor with the same initializers. A
// class-typed field calls an earlier class's `Make`, so these calls never
// recurse.
auto SourceGen::GenerateMakeFunction(ClassGenState& state,
                                     llvm::StringRef class_name,
                                     llvm::ArrayRef<TypedName> fields,
                                     llvm::raw_ostream& os) -> void {
  os << "  // TODO: make better comment text\n";
  if (IsCpp()) {
    os << "  " << class_name << "()\n";
    os << "      : tag(0)";
    for (const TypedName& field : fields) {
      os << ", " << field.name << "(";
      state.ProduceValue(field.type, os);
      os << ")";
    }
    os << " {}\n";
    return;
  }

  os << "  fn Make() -> " << class_name << " {\n";
  os << "    return {.tag = 0";
  for (const TypedName& field : fields) {
    os << ", ." << field.name << " = ";
    state.ProduceValue(field.type, os);
  }
  os << "};\n  }\n";
}

// Generates a class's `Checksum` method, which the class consumer templates
// call to read a value of the class. It reads only `tag`, so it is the same in
// every class without needing consumer expressions for the other fields.
auto SourceGen::GenerateChecksumFunction(llvm::raw_ostream& os) -> void {
  os << "  // TODO: make better comment text\n";
  if (!IsCpp()) {
    os << "  fn Checksum(self) -> i32 {\n";
    os << "    return self.tag + 1;\n";
  } else {
    os << "  auto Checksum() -> int {\n";
    os << "    return tag + 1;\n";
  }
  os << "  }\n";
}

// Generate a class definition and write it to the provided stream.
//
// The structure of the definition is guided by the `params` provided, and it
// consumes the provided state. With out-of-line definitions, it appends the
// declarations to define to `out_of_line_defs`.
auto SourceGen::GenerateClassDef(
    const ClassParams& params, ClassGenState& state, llvm::raw_ostream& os,
    llvm::SmallVectorImpl<OutOfLineDef>& out_of_line_defs) -> void {
  llvm::StringRef name = state.class_names().pop_back_val();
  os << "// TODO: make better comment text\n";
  os << "class " << name << " {\n";
  if (IsCpp()) {
    os << " public:\n";
    if (state.has_defs()) {
      os << "  using Self = " << name << ";\n";
    }
  }

  // Field types can't be the class we're currently declaring. We enforce this
  // by collecting them before inserting that type into the valid set.
  llvm::SmallVector<llvm::StringRef> field_type_names;
  field_type_names.reserve(params.private_field_decls);
  for (auto _ : llvm::seq(params.private_field_decls)) {
    field_type_names.push_back(state.GetFieldType().name);
  }

  // Mark this class as now a valid type now that field type names have been
  // collected. We can reference this class from functions and methods within
  // the definition.
  state.AddValidTypeName(name);

  // Name the getter and predicate fields first, so that member names can avoid
  // them.
  UniqueIdentifierPopper unique_accessed_names(
      *this, state.accessed_field_names(), &state.class_name_set());
  llvm::SmallVector<TypedName> getter_fields;
  getter_fields.reserve(params.inline_getters);
  for (auto _ : llvm::seq(params.inline_getters)) {
    getter_fields.push_back(
        {.name = unique_accessed_names.Pop(),
         .type = state.getter_field_types().pop_back_val()});
  }
  llvm::SmallVector<std::pair<TypedName, llvm::StringRef>> predicate_fields;
  predicate_fields.reserve(params.inline_predicates);
  for (auto _ : llvm::seq(params.inline_predicates)) {
    ClassGenState::PredicateField field =
        state.predicate_fields().pop_back_val();
    predicate_fields.push_back(
        {{.name = unique_accessed_names.Pop(), .type = field.type},
         field.predicate});
  }

  // Definitions in this class can name any class, so exclude class names from
  // member names. Inline function names are too short to be class names.
  UniqueIdentifierPopper unique_member_names(*this, state.decl_names(),
                                             &state.class_name_set());
  unique_member_names.Reserve(unique_accessed_names.used());
  UniqueIdentifierPopper unique_inline_names(*this,
                                             state.inline_function_names());

  // Capture each declaration's signature for its out-of-line definition.
  bool define_out_of_line = state.define_decls_out_of_line();
  auto capture_sig = [&]() -> FunctionSig* {
    if (!define_out_of_line) {
      return nullptr;
    }
    return &out_of_line_defs.emplace_back(OutOfLineDef{.class_name = name}).sig;
  };

  llvm::ListSeparator line_sep("\n");
  for (auto _ : llvm::seq(params.public_function_decls)) {
    os << line_sep;
    GenerateFunctionDecl(state, unique_member_names.Pop(), /*is_private=*/false,
                         /*is_method=*/false,
                         state.public_function_param_counts().pop_back_val(),
                         /*indent=*/"  ", os, capture_sig());
  }
  for (auto _ : llvm::seq(params.inline_function_defs)) {
    os << line_sep;
    GenerateInlineFunctionDef(
        state, unique_inline_names.Pop(),
        state.inline_function_param_counts().pop_back_val(), /*indent=*/"  ",
        os);
  }
  for (auto _ : llvm::seq(params.public_method_decls)) {
    os << line_sep;
    GenerateFunctionDecl(state, unique_member_names.Pop(), /*is_private=*/false,
                         /*is_method=*/true,
                         state.public_method_param_counts().pop_back_val(),
                         /*indent=*/"  ", os, capture_sig());
  }
  for (TypedName field : getter_fields) {
    os << line_sep;
    GenerateGetter(unique_inline_names.Pop(), field, os);
  }
  for (auto [field, predicate] : predicate_fields) {
    os << line_sep;
    GeneratePredicate(unique_inline_names.Pop(), field, predicate, os);
  }

  // Forwarder names are as short as other inline function names, so exclude
  // those.
  UniqueIdentifierPopper unique_forwarder_names(*this, state.forwarder_names());
  unique_forwarder_names.Reserve(unique_inline_names.used());
  struct Forwarder {
    llvm::StringRef name;
    llvm::SmallVector<TypedName> params;
    llvm::StringRef return_type;
  };
  llvm::SmallVector<Forwarder> forwarders;
  forwarders.reserve(params.inline_forwarders);
  for (auto _ : llvm::seq(params.inline_forwarders)) {
    Forwarder& forwarder = forwarders.emplace_back();
    forwarder.name = unique_forwarder_names.Pop();
    // Exclude class names so that a parameter can't shadow a class in the
    // signature.
    UniqueIdentifierPopper unique_param_names(
        *this, state.forwarder_param_names(), &state.class_name_set());
    for (auto _ : llvm::seq(state.forwarder_param_counts().pop_back_val())) {
      forwarder.params.push_back({.name = unique_param_names.Pop(),
                                  .type = state.GetForwardType().name});
    }
    forwarder.return_type = state.GetForwardType().name;
    os << line_sep;
    GenerateForwarder(forwarder.name, forwarder.params, forwarder.return_type,
                      os);
  }

  if (IsCpp()) {
    os << "\n private:\n";
    // Reset the separator.
    line_sep = llvm::ListSeparator("\n");
  }

  for (auto _ : llvm::seq(params.private_function_decls)) {
    os << line_sep;
    GenerateFunctionDecl(state, unique_member_names.Pop(), /*is_private=*/true,
                         /*is_method=*/false,
                         state.private_function_param_counts().pop_back_val(),
                         /*indent=*/"  ", os, capture_sig());
  }
  for (auto _ : llvm::seq(params.private_method_decls)) {
    os << line_sep;
    GenerateFunctionDecl(state, unique_member_names.Pop(), /*is_private=*/true,
                         /*is_method=*/true,
                         state.private_method_param_counts().pop_back_val(),
                         /*indent=*/"  ", os, capture_sig());
  }
  for (const Forwarder& forwarder : forwarders) {
    os << line_sep;
    EmitFunctionDecl((forwarder.name + "Impl").str(), /*is_private=*/true,
                     /*is_method=*/true, forwarder.params,
                     forwarder.return_type, /*indent=*/"  ", os);
  }

  // Field names can't repeat this class's function, method, or field names, or
  // be a class name.
  UniqueIdentifierPopper unique_field_names(*this, state.field_names(),
                                            &state.class_name_set());
  unique_field_names.Reserve(unique_member_names.used());
  llvm::SmallVector<TypedName> fields;
  fields.reserve(field_type_names.size() + getter_fields.size() +
                 predicate_fields.size());
  for (llvm::StringRef type_name : field_type_names) {
    fields.push_back({.name = unique_field_names.Pop(), .type = type_name});
  }
  llvm::append_range(fields, getter_fields);
  for (auto [field, _] : predicate_fields) {
    fields.push_back(field);
  }

  // With definitions, C++ puts `Make` and `Checksum` in a second public
  // section, followed by a private section for the fields.
  bool has_defs = state.has_defs();
  if (IsCpp() && has_defs) {
    os << "\n public:\n";
    line_sep = llvm::ListSeparator("\n");
  }

  // Only definitions call `Make` and `Checksum`.
  if (has_defs) {
    os << line_sep;
    GenerateMakeFunction(state, name, fields, os);
    os << line_sep;
    GenerateChecksumFunction(os);
    if (IsCpp()) {
      os << "\n private:\n";
      line_sep = llvm::ListSeparator("\n");
    }
  }

  os << line_sep;
  // `tag` comes first, matching its position in `Make`.
  if (has_defs) {
    os << (IsCpp() ? "  int tag;\n" : "  private var tag: i32;\n");
  }
  for (const TypedName& field : fields) {
    if (!IsCpp()) {
      os << "  private var " << field.name << ": " << field.type << ";\n";
    } else {
      os << "  " << field.type << " " << field.name << ";\n";
    }
  }
  os << "}" << (IsCpp() ? ";" : "") << "\n";
}

}  // namespace Carbon::Testing
