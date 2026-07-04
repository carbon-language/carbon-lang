// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/benchmarking/source_gen.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>

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

// The length of every local variable name. How many times a body spells each
// local depends on the local's position, so a single length keeps the byte
// count from depending on which name lands where. Locals are shorter than class
// names, so they can't shadow a class that the body names.
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

// The shuffled state used to generate some number of classes.
//
// This state encodes everything used to generate class definitions. The state
// will be consumed until empty.
//
// Detailed comments for out-of-line methods are on their definitions.
class SourceGen::ClassGenState {
 public:
  ClassGenState(SourceGen& gen, int num_classes,
                const ClassParams& class_params,
                const TypeUseParams& type_use_params);

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
  auto local_counts() -> llvm::SmallVectorImpl<int>& { return local_counts_; }
  auto forwarder_param_counts() -> llvm::SmallVectorImpl<int>& {
    return forwarder_param_counts_;
  }

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
    valid_type_names_.Insert(type_name);
  }

  // The names of all classes in the file. Parameter, member, and field names
  // exclude these so that they can't shadow a class that a body names.
  auto class_name_set() -> const Set<llvm::StringRef>& {
    return class_name_set_;
  }

  // Emits an expression producing a value of `type`: a call to the class's
  // `Make` function, or the fixed type's value expression.
  auto ProduceValue(llvm::StringRef type, llvm::raw_ostream& os) -> void {
    if (class_name_set_.Contains(type)) {
      os << type << (is_cpp_ ? "::Make()" : ".Make()");
    } else {
      os << fixed_value_.Lookup(type).value();
    }
  }

  // A use of a type drawn from a pool. A consumed use also has the consumer
  // template that reads it.
  struct TypeUse {
    llvm::StringRef name;
    llvm::StringRef consumer;
  };

  // Each kind of type use draws from its own pool, so that every use in a pool
  // spells its type the same number of times. Then the byte count doesn't
  // depend on which use gets which type. A use is "produced" when a body
  // constructs a value of its type, and "consumed" when a body reads it with a
  // consumer template.

  // Return and parameter types of declarations, and field types when the file
  // has no bodies.
  auto GetDeclType() -> TypeUse { return GetValidTypeUse(decl_type_pool_); }
  // Parameter types of inline definitions, which are consumed.
  auto GetInlineParamType() -> TypeUse {
    return GetValidTypeUse(inline_param_pool_);
  }
  // Return types of inline definitions, and field types when the file has
  // bodies, which are produced.
  auto GetProducedType() -> TypeUse {
    return GetValidTypeUse(produced_type_pool_);
  }
  // `Make` produces every field when the file has bodies. Without bodies,
  // fields share the declaration pool, which can include types that have no
  // value expression.
  auto GetFieldType() -> TypeUse {
    return GetValidTypeUse(has_bodies_ ? produced_type_pool_ : decl_type_pool_);
  }
  // Return and parameter types of forwarders, which the declarations that they
  // call spell again.
  auto GetForwardType() -> TypeUse {
    return GetValidTypeUse(forward_type_pool_);
  }

  auto has_bodies() -> bool { return has_bodies_; }
  auto type_pools_empty() -> bool {
    return decl_type_pool_.uses.empty() && inline_param_pool_.uses.empty() &&
           produced_type_pool_.uses.empty() && forward_type_pool_.uses.empty();
  }

 private:
  // A pool of type uses, removed as they are emitted. `GetValidTypeUse` resumes
  // its search at `last_index`.
  struct TypePool {
    llvm::SmallVector<TypeUse> uses;
    int last_index = 0;
  };

  auto GetValidTypeUse(TypePool& pool) -> TypeUse;

  auto BuildClassAndTypeNames(SourceGen& gen, int num_classes,
                              const ClassParams& class_params,
                              const TypeUseParams& type_use_params) -> void;
  auto BuildTypePool(SourceGen& gen, int num_types, int max_refs_per_class,
                     bool producible_only, bool consumed,
                     const TypeUseParams& type_use_params) -> TypePool;

  llvm::SmallVector<int> public_function_param_counts_;
  llvm::SmallVector<int> public_method_param_counts_;
  llvm::SmallVector<int> private_function_param_counts_;
  llvm::SmallVector<int> private_method_param_counts_;

  llvm::SmallVector<int> inline_function_param_counts_;
  llvm::SmallVector<int> local_counts_;
  llvm::SmallVector<int> forwarder_param_counts_;

  // Getters return a copy of their field, and `Make` initializes it, so getter
  // and predicate fields have fixed types with a value expression. These types
  // cycle through the eligible types in proportion to their weights, so that
  // their spellings don't depend on the seed.
  llvm::SmallVector<llvm::StringRef> getter_field_types_;
  llvm::SmallVector<PredicateField> predicate_fields_;

  llvm::SmallVector<llvm::StringRef> class_names_;
  // Field names have their own pool because Carbon's `Make` spells each field
  // name a second time.
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
  // Whether the file has any bodies. Then each class also gets a `Make`
  // function, a `Checksum` method, and a `tag` field.
  bool has_bodies_;
  TypePool decl_type_pool_;
  TypePool inline_param_pool_;
  TypePool produced_type_pool_;
  TypePool forward_type_pool_;
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
                                        const ClassParams& class_params,
                                        const TypeUseParams& type_use_params)
    : is_cpp_(gen.IsCpp()),
      has_bodies_(NumInlineDefsPerClass(class_params) > 0) {
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
  int num_params =
      Sum(public_function_param_counts_) + Sum(public_method_param_counts_) +
      Sum(private_function_param_counts_) + Sum(private_method_param_counts_);
  param_names_ = gen.GetShuffledIdentifiers(num_params);

  inline_function_param_counts_ =
      gen.GetShuffledInts(num_inline_functions, 0,
                          class_params.inline_function_decl_params.max_params);
  local_counts_ = gen.GetShuffledInts(num_inline_functions, 0,
                                      class_params.max_body_locals);
  // Each inline definition has one parameter beyond its random count; see
  // `BuildClassAndTypeNames`.
  int num_inline_params =
      Sum(inline_function_param_counts_) + num_inline_functions;
  int num_locals = Sum(local_counts_);
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

  BuildClassAndTypeNames(gen, num_classes, class_params, type_use_params);
}

auto SourceGen::ClassGenState::GetValidTypeUse(TypePool& pool) -> TypeUse {
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
    if (valid_type_names_.Contains(use.name)) {
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

// Builds a shuffled pool of `num_types` type uses, mixing references to the
// classes with the fixed types to roughly match the weights in
// `type_use_params`. `max_refs_per_class` caps the references to each class.
//
// With `producible_only`, the pool only uses fixed types that have a value
// expression. With `consumed`, each use gets one of its type's consumer
// templates, round-robin per type before the shuffle, so that the mix of
// templates doesn't depend on the seed.
//
// `valid_type_names_` must already contain the fixed type spellings.
auto SourceGen::ClassGenState::BuildTypePool(
    SourceGen& gen, int num_types, int max_refs_per_class, bool producible_only,
    bool consumed, const TypeUseParams& type_use_params) -> TypePool {
  TypePool pool;
  if (num_types == 0) {
    return pool;
  }
  pool.uses.reserve(num_types);

  auto fixed_spelling = [&](const TypeUseParams::FixedTypeWeight& fw) {
    return gen.IsCpp() ? fw.cpp_spelling : fw.carbon_spelling;
  };
  auto fixed_usable = [&](const TypeUseParams::FixedTypeWeight& fw) {
    return !producible_only ||
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
  // class names until there is some remainder of names needed.
  int num_classes = class_names_.size();
  int num_declared_types =
      num_types * type_use_params.declared_types_weight / type_weight_sum;
  int full_copies = num_declared_types / num_classes;
  int remainder = num_declared_types % num_classes;

  // Cap the references to each class so that `GetValidTypeUse` finds a valid
  // type for any shuffle. A class becomes a valid type after its field types
  // are chosen, so references to it can only go on its own function signatures,
  // or in a later class. The last class defined has only its own, and the cap
  // is the number of uses each class draws from this pool after its fields. The
  // fixed types below replace any references the cap removes, so the pool's
  // spellings, and with them the byte count, don't depend on the shuffle.
  if (full_copies >= max_refs_per_class) {
    full_copies = max_refs_per_class;
    remainder = 0;
  }

  for ([[maybe_unused]] auto _ : llvm::seq(full_copies)) {
    for (llvm::StringRef name : class_names_) {
      append_use(name);
    }
  }
  // Now append the remainder number of class names. This is where the class
  // names being un-shuffled is essential. We're going to have one extra
  // reference to some fraction of the class names and we want that to be a
  // stable subset.
  for (llvm::StringRef name :
       llvm::ArrayRef(class_names_).slice(0, remainder)) {
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
    for ([[maybe_unused]] auto _ : llvm::seq(num_fixed_type)) {
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

  std::shuffle(pool.uses.begin(), pool.uses.end(), gen.rng_);
  return pool;
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
  int num_produced_fields = has_bodies_ ? num_fields : 0;

  // Each class draws `inline_function_defs` return types from this pool after
  // its fields.
  produced_type_pool_ = BuildTypePool(
      gen, num_inline_returns + num_produced_fields,
      class_params.inline_function_defs,
      /*producible_only=*/true, /*consumed=*/false, type_use_params);

  // Each inline definition has one parameter beyond its random count, so that
  // each class draws at least `inline_function_defs` parameter types from this
  // pool, and the pool can include class types.
  int num_inline_extra_params = num_inline_returns;
  inline_param_pool_ = BuildTypePool(
      gen, num_inline_params + num_inline_extra_params,
      class_params.inline_function_defs,
      /*producible_only=*/false, /*consumed=*/true, type_use_params);

  // Each class draws `decls_per_class` return types from this pool after its
  // fields.
  decl_type_pool_ = BuildTypePool(
      gen,
      num_decl_returns + num_decl_params + (num_fields - num_produced_fields),
      decls_per_class,
      /*producible_only=*/false, /*consumed=*/false, type_use_params);

  // Each class draws `inline_forwarders` return types from this pool after its
  // fields.
  forward_type_pool_ = BuildTypePool(
      gen,
      num_classes * class_params.inline_forwarders +
          Sum(forwarder_param_counts_),
      class_params.inline_forwarders,
      /*producible_only=*/false, /*consumed=*/false, type_use_params);

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

  std::shuffle(class_names_.begin(), class_names_.end(), gen.rng_);
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
// excluding its comment. The body has an accumulator line, a line per
// parameter, a line per local plus one more when there are any, a return line,
// and a closing brace line.
static auto EstimateAvgInlineFunctionDefLines(SourceGen::ClassParams params)
    -> double {
  constexpr int ExtraParams = 1;
  double avg_params =
      params.inline_function_decl_params.max_params / 2.0 + ExtraParams;
  double max_locals = params.max_body_locals;
  double avg_locals = max_locals / 2.0;
  double prob_any_local = max_locals / (max_locals + 1.0);
  return EstimateAvgFunctionDeclLines(params.inline_function_decl_params,
                                      ExtraParams) +
         1.0 + avg_params + avg_locals + prob_any_local + 2.0;
}

// Note that this should match the heuristics used when formatting.
// TODO: See top-level TODO about line estimates and formatting.
static auto EstimateAvgClassDefLines(SourceGen::ClassParams params) -> double {
  // Comment line, and class open line.
  double avg = 2.0;

  // One comment line and blank line per function, plus the function lines.
  avg +=
      (2.0 + EstimateAvgFunctionDeclLines(params.public_function_decl_params)) *
      params.public_function_decls;
  avg += (2.0 + EstimateAvgMethodDeclLines(params.public_method_decl_params)) *
         params.public_method_decls;
  avg += (2.0 +
          EstimateAvgFunctionDeclLines(params.private_function_decl_params)) *
         params.private_function_decls;
  avg += (2.0 + EstimateAvgMethodDeclLines(params.private_method_decl_params)) *
         params.private_method_decls;
  avg += (2.0 + EstimateAvgInlineFunctionDefLines(params)) *
         params.inline_function_defs;
  // Getters and predicates are on one line.
  avg += 3.0 * (params.inline_getters + params.inline_predicates);
  // A forwarder's body has a return line and a closing brace line, and the
  // declaration it calls has the same signature.
  double forwarder_signature_lines =
      EstimateAvgMethodDeclLines(params.inline_forwarder_params);
  avg += (2.0 + forwarder_signature_lines + 2.0 + 2.0 +
          forwarder_signature_lines) *
         params.inline_forwarders;

  bool has_bodies = NumInlineDefsPerClass(params) > 0;

  // A blank line and all the fields (if any), including `tag` when the file has
  // bodies.
  double num_fields = params.private_field_decls + params.inline_getters +
                      params.inline_predicates + (has_bodies ? 1.0 : 0.0);
  if (num_fields > 0) {
    avg += 1.0 + num_fields;
  }

  // `Make` and `Checksum` each have a blank line, a comment line, a signature
  // line, a return line, and a closing brace line.
  if (has_bodies) {
    avg += 10.0;
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
  double avg_class_lines = EstimateAvgClassDefLines(params.class_params);
  CARBON_CHECK(target_lines > NumFileCommentLines + avg_class_lines,
               "Not enough target lines to generate a single class!");
  // Round to the nearest whole class. Truncating can leave the file nearly a
  // whole class short of the target, which matters when classes with bodies
  // run to hundreds of lines.
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
  source << "//\n// Generating as an API file with dense declarations.\n";

  // Carbon uses an implicitly imported prelude to get builtin types, but C++
  // requires header files so include those.
  if (IsCpp()) {
    source << "\n";
    // Header for specific integer types like `std::int64_t`.
    source << "#include <cstdint>\n";
    // Header for `std::pair`.
    source << "#include <utility>\n";
  }

  auto class_gen_state = ClassGenState(*this, num_classes, params.class_params,
                                       params.type_use_params);
  for ([[maybe_unused]] auto _ : llvm::seq(num_classes)) {
    source << "\n";
    GenerateClassDef(params.class_params, class_gen_state, source);
  }

  // Make sure we consumed all the state.
  CARBON_CHECK(class_gen_state.public_function_param_counts().empty());
  CARBON_CHECK(class_gen_state.public_method_param_counts().empty());
  CARBON_CHECK(class_gen_state.private_function_param_counts().empty());
  CARBON_CHECK(class_gen_state.private_method_param_counts().empty());
  CARBON_CHECK(class_gen_state.inline_function_param_counts().empty());
  CARBON_CHECK(class_gen_state.local_counts().empty());
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
    for ([[maybe_unused]] auto _ : llvm::seq<int>(idents.size(), number)) {
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
    for ([[maybe_unused]] auto _ : llvm::seq<int>(count, number)) {
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
// parameter names and types come from `state`.
auto SourceGen::GenerateFunctionDecl(ClassGenState& state, llvm::StringRef name,
                                     bool is_private, bool is_method,
                                     int param_count, llvm::StringRef indent,
                                     llvm::raw_ostream& os) -> void {
  UniqueIdentifierPopper unique_param_names(*this, state.param_names());
  llvm::SmallVector<TypedName> params;
  params.reserve(param_count);
  for ([[maybe_unused]] auto _ : llvm::seq(param_count)) {
    params.push_back(
        {.name = unique_param_names.Pop(), .type = state.GetDeclType().name});
  }
  EmitFunctionDecl(name, is_private, is_method, params,
                   state.GetDeclType().name, indent, os);
}

// Emits `format` with `name` substituted for its `{0}`.
static auto EmitTemplate(llvm::StringRef format, llvm::StringRef name,
                         llvm::raw_ostream& os) -> void {
  CARBON_CHECK(!format.empty());
  auto [prefix, suffix] = format.split("{0}");
  os << prefix << name << suffix;
}

// Generates an inline function definition and writes it to the provided
// stream.
//
// The body adds every parameter to an `i32` accumulator, declares a chain of
// `i32` locals, and returns a value of the return type. Reading every parameter
// and local keeps the code free of unused-binding warnings.
auto SourceGen::GenerateInlineFunctionDef(ClassGenState& state,
                                          llvm::StringRef name, int param_count,
                                          int local_count,
                                          llvm::StringRef indent,
                                          llvm::raw_ostream& os) -> void {
  // Add the extra parameter; see `BuildClassAndTypeNames`.
  param_count += 1;

  // Exclude class names so that a parameter can't shadow a class that the body
  // names.
  UniqueIdentifierPopper unique_param_names(*this, state.inline_param_names(),
                                            &state.class_name_set());
  llvm::SmallVector<TypedName> params;
  llvm::SmallVector<llvm::StringRef> consumers;
  params.reserve(param_count);
  consumers.reserve(param_count);
  for ([[maybe_unused]] auto _ : llvm::seq(param_count)) {
    llvm::StringRef param = unique_param_names.Pop();
    ClassGenState::TypeUse type = state.GetInlineParamType();
    params.push_back({.name = param, .type = type.name});
    consumers.push_back(type.consumer);
  }
  llvm::StringRef return_type = state.GetProducedType().name;

  os << indent << "// TODO: make better comment text\n";
  os << indent << (IsCpp() ? "static auto " : "fn ") << name << "(";
  EmitParams(/*is_method=*/false, params, indent, os);
  os << ") -> " << return_type << " {\n";

  std::string body_indent = indent.str() + "  ";

  os << body_indent << (IsCpp() ? "int acc = 0;\n" : "var acc: i32 = 0;\n");
  for (auto [param, consumer] : llvm::zip(params, consumers)) {
    os << body_indent << "acc = acc + ";
    EmitTemplate(consumer, param.name, os);
    os << ";\n";
  }

  // Each local after the first reads the previous one. Locals and parameters
  // draw from the same identifiers of each length, so exclude this function's
  // parameter names: C++ rejects a local that redeclares a parameter.
  llvm::SmallVector<llvm::StringRef> locals;
  locals.reserve(local_count);
  UniqueIdentifierPopper unique_local_names(*this, state.local_names(),
                                            &unique_param_names.used());
  for (int i : llvm::seq(local_count)) {
    llvm::StringRef local = unique_local_names.Pop();
    locals.push_back(local);
    if (!IsCpp()) {
      os << body_indent << "var " << local << ": i32 = ";
    } else {
      os << body_indent << "int " << local << " = ";
    }
    if (i == 0) {
      os << "1";
    } else {
      os << locals[i - 1] << " + 1";
    }
    os << ";\n";
  }
  // Read the last local by assigning it to the first. With one local, this is a
  // self-assignment, which still reads it.
  if (local_count > 0) {
    os << body_indent << locals.front() << " = " << locals.back() << ";\n";
  }

  os << body_indent << "return ";
  state.ProduceValue(return_type, os);
  os << ";\n";

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
// every field initialized. A class-typed field calls an earlier class's `Make`,
// so these calls never recurse.
auto SourceGen::GenerateMakeFunction(ClassGenState& state,
                                     llvm::StringRef class_name,
                                     llvm::ArrayRef<TypedName> fields,
                                     llvm::raw_ostream& os) -> void {
  os << "  // TODO: make better comment text\n";
  os << "  " << (IsCpp() ? "static auto " : "fn ") << "Make() -> " << class_name
     << " {\n";
  os << "    return {";
  llvm::ListSeparator sep;
  os << sep << (IsCpp() ? "0" : ".tag = 0");
  for (const TypedName& field : fields) {
    os << sep;
    // Carbon uses designated initializers; C++ uses positional aggregate init.
    if (!IsCpp()) {
      os << "." << field.name << " = ";
    }
    state.ProduceValue(field.type, os);
  }
  os << "};\n  }\n";
}

// Generates a class's `Checksum` method, which the class consumer templates
// call to read a value of the class. It reads only `tag`, so it is the same in
// every class. Reading the other fields would consume their types, which share
// a pool with inline return types that aren't consumed.
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
// consumes the provided state.
auto SourceGen::GenerateClassDef(const ClassParams& params,
                                 ClassGenState& state, llvm::raw_ostream& os)
    -> void {
  llvm::StringRef name = state.class_names().pop_back_val();
  os << "// TODO: make better comment text\n";
  os << "class " << name << " {\n";
  if (IsCpp()) {
    os << " public:\n";
  }

  // Field types can't be the class we're currently declaring. We enforce this
  // by collecting them before inserting that type into the valid set.
  llvm::SmallVector<llvm::StringRef> field_type_names;
  field_type_names.reserve(params.private_field_decls);
  for ([[maybe_unused]] auto _ : llvm::seq(params.private_field_decls)) {
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
  for ([[maybe_unused]] auto _ : llvm::seq(params.inline_getters)) {
    getter_fields.push_back(
        {.name = unique_accessed_names.Pop(),
         .type = state.getter_field_types().pop_back_val()});
  }
  llvm::SmallVector<std::pair<TypedName, llvm::StringRef>> predicate_fields;
  predicate_fields.reserve(params.inline_predicates);
  for ([[maybe_unused]] auto _ : llvm::seq(params.inline_predicates)) {
    ClassGenState::PredicateField field =
        state.predicate_fields().pop_back_val();
    predicate_fields.push_back(
        {{.name = unique_accessed_names.Pop(), .type = field.type},
         field.predicate});
  }

  // Bodies in this class can name any class, so exclude class names from member
  // names. Inline function names are too short to be class names.
  UniqueIdentifierPopper unique_member_names(*this, state.decl_names(),
                                             &state.class_name_set());
  unique_member_names.Reserve(unique_accessed_names.used());
  UniqueIdentifierPopper unique_inline_names(*this,
                                             state.inline_function_names());

  llvm::ListSeparator line_sep("\n");
  for ([[maybe_unused]] auto _ : llvm::seq(params.public_function_decls)) {
    os << line_sep;
    GenerateFunctionDecl(state, unique_member_names.Pop(), /*is_private=*/false,
                         /*is_method=*/false,
                         state.public_function_param_counts().pop_back_val(),
                         /*indent=*/"  ", os);
  }
  for ([[maybe_unused]] auto _ : llvm::seq(params.inline_function_defs)) {
    os << line_sep;
    GenerateInlineFunctionDef(
        state, unique_inline_names.Pop(),
        state.inline_function_param_counts().pop_back_val(),
        state.local_counts().pop_back_val(), /*indent=*/"  ", os);
  }
  for ([[maybe_unused]] auto _ : llvm::seq(params.public_method_decls)) {
    os << line_sep;
    GenerateFunctionDecl(state, unique_member_names.Pop(), /*is_private=*/false,
                         /*is_method=*/true,
                         state.public_method_param_counts().pop_back_val(),
                         /*indent=*/"  ", os);
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
  for ([[maybe_unused]] auto _ : llvm::seq(params.inline_forwarders)) {
    Forwarder& forwarder = forwarders.emplace_back();
    forwarder.name = unique_forwarder_names.Pop();
    // Exclude class names so that a parameter can't shadow a class in the
    // signature.
    UniqueIdentifierPopper unique_param_names(
        *this, state.forwarder_param_names(), &state.class_name_set());
    for ([[maybe_unused]] auto _ :
         llvm::seq(state.forwarder_param_counts().pop_back_val())) {
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

  for ([[maybe_unused]] auto _ : llvm::seq(params.private_function_decls)) {
    os << line_sep;
    GenerateFunctionDecl(state, unique_member_names.Pop(), /*is_private=*/true,
                         /*is_method=*/false,
                         state.private_function_param_counts().pop_back_val(),
                         /*indent=*/"  ", os);
  }
  for ([[maybe_unused]] auto _ : llvm::seq(params.private_method_decls)) {
    os << line_sep;
    GenerateFunctionDecl(state, unique_member_names.Pop(), /*is_private=*/true,
                         /*is_method=*/true,
                         state.private_method_param_counts().pop_back_val(),
                         /*indent=*/"  ", os);
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

  // With bodies, C++ puts `Make` and the fields in a public section. Public
  // fields make the class an aggregate, which `Make` initializes with a braced
  // list. Without bodies, the fields stay private.
  bool has_bodies = state.has_bodies();
  if (IsCpp() && has_bodies) {
    os << "\n public:\n";
    line_sep = llvm::ListSeparator("\n");
  }

  // Only bodies call `Make` and `Checksum`.
  if (has_bodies) {
    os << line_sep;
    GenerateMakeFunction(state, name, fields, os);
    os << line_sep;
    GenerateChecksumFunction(os);
  }

  os << line_sep;
  // `tag` comes first, matching its position in C++ `Make`.
  if (has_bodies) {
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
