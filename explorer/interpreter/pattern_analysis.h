// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_PATTERN_ANALYSIS_H_
#define CARBON_EXPLORER_INTERPRETER_PATTERN_ANALYSIS_H_

#include <vector>

#include "explorer/ast/pattern.h"
#include "explorer/common/nonnull.h"
#include "explorer/interpreter/value.h"
#include "llvm/ADT/PointerUnion.h"

namespace Carbon {

// An abstracted view of a pattern or constant value (which we view as a
// particular kind of pattern).
class AbstractPattern {
 public:
  enum Kind {
    // A pattern that matches anything.
    Wildcard,
    // A pattern that matches a compound value with sub-patterns to match
    // elements. A compound value is modeled as a discriminator name applied to
    // a sequence of nested values: the alternative `Optional.Element(E)` has
    // discriminator `Element` and nested value `E`, and the tuple `(A,B,C)`
    // has an empty discriminator and nested values `A`, `B`, and `C`.
    Compound,
    // A pattern that matches a particular primitive value.
    Primitive
  };

  // This is intentionally implicit to allow easy conversion from a container
  // of `const Pattern*` to a container of `AbstractPattern`s.
  // NOLINTNEXTLINE(google-explicit-constructor)
  AbstractPattern(Nonnull<const Pattern*> pattern) { Set(pattern); }

  AbstractPattern(Nonnull<const Value*> value, Nonnull<const Value*> type)
      : value_(value), type_(type) {}

  // Make a match-anything wildcard pattern.
  static auto MakeWildcard() -> AbstractPattern {
    return AbstractPattern(WildcardTag());
  }

  // Get the kind for this pattern.
  auto kind() const -> Kind;

  // Get the type, for a non-wildcard pattern.
  auto type() const -> const Value&;

  // Get the discriminator used for a compound pattern.
  auto discriminator() const -> std::string_view;

  // Get the number of nested patterns for a compound pattern.
  auto elements_size() const -> int;

  // Append the nested patterns in this compound pattern to `out`.
  void AppendElementsTo(std::vector<AbstractPattern>& out) const;

  // Get the value for a primitive pattern.
  auto value() const -> const Value&;

 private:
  // This is aligned so that we can use it in the `PointerUnion` below.
  struct alignas(8) WildcardTag {};
  AbstractPattern(WildcardTag)
      : value_(static_cast<const WildcardTag*>(nullptr)), type_(nullptr) {}

  void Set(Nonnull<const Pattern*> pattern);

  // The underlying pattern: either a syntactic pattern, or a constant value,
  // or a placeholder indicating that this is a wildcard pattern.
  llvm::PointerUnion<Nonnull<const Pattern*>, Nonnull<const Value*>,
                     const WildcardTag*>
      value_;
  // Values don't always know their types, so store the type here. We only
  // really need it for the `const Value*` case but also store it for the
  // `const Pattern*` case for convenience.
  const Value* type_;
};

// A matrix of patterns, used for determining exhaustiveness and redundancy of
// patterns in a match statement.
//
// See http://moscova.inria.fr/~maranget/papers/warn/index.html for details on
// the algorithm used here.
class PatternMatrix {
 public:
  // Add a pattern vector row to this collection of pattern vectors.
  void Add(std::vector<AbstractPattern> pattern_vector) {
    CARBON_CHECK(matrix_.empty() || matrix_[0].size() == pattern_vector.size());
    matrix_.push_back(std::move(pattern_vector));
  }

  // Returns true if the given pattern vector is redundant if it appears after
  // the patterns in this matrix. That is, if it will never match following the
  // other patterns because everything it matches is matched by some other
  // pattern.
  auto IsRedundant(llvm::ArrayRef<AbstractPattern> pattern) const -> bool {
    return !IsUseful(pattern, MaxExponentialDepth);
  }

 private:
  // The maximum number of times we will consider all alternatives when
  // recursively expanding the pattern. Allowing this to happen an arbitrary
  // number of times leads to exponential growth in the runtime of the
  // algorithm.
  static constexpr int MaxExponentialDepth = 8;

  // Information about a constructor for a compound type.
  struct DiscriminatorInfo {
    // For an alternative, the name. Otherwise, empty.
    std::string_view discriminator;
    // The number of elements. For a tuple, the size. Always 1 for an
    // alternative.
    int size;
  };

  struct DiscriminatorSet {
    std::vector<DiscriminatorInfo> found;
    bool any_missing;
  };

  // Determine whether the given pattern vector is useful: that is, whether
  // adding it to the matrix would allow any more values to be matched.
  auto IsUseful(llvm::ArrayRef<AbstractPattern> pattern,
                int max_exponential_depth) const -> bool;

  // Find the discriminators used by the first column and check whether we
  // found all of them.
  auto FirstColumnDiscriminators() const -> DiscriminatorSet;

  // Specialize the pattern vector `row` for the case that the first value
  // matched uses `discriminator`.
  static auto SpecializeRow(llvm::ArrayRef<AbstractPattern> row,
                            DiscriminatorInfo discriminator)
      -> std::optional<std::vector<AbstractPattern>>;

  // Specialize the pattern matrix for the case that the first value matched
  // uses `discriminator`, and its elements are matched.
  auto Specialize(DiscriminatorInfo discriminator) const -> PatternMatrix;

  // Specialize the pattern matrix for the case where the first value is known
  // to be `value`, and is not matched.
  auto Specialize(const Value& value) const -> PatternMatrix;

  // Specialize the pattern matrix for the case where the first value uses a
  // discriminator matching none of the non-wildcard patterns.
  auto Default() const -> PatternMatrix;

  // The pattern matrix itself, in row-major order. Each element of `matrix_`
  // is a distinct sequence of patterns that can be matched against a
  // corresponding sequence of values. Each such row has the same length and
  // has elements of the same type.
  std::vector<std::vector<AbstractPattern>> matrix_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_PATTERN_ANALYSIS_H_
