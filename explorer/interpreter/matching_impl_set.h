// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_MATCHING_IMPL_SET_H_
#define CARBON_EXPLORER_INTERPRETER_MATCHING_IMPL_SET_H_

#include <vector>

#include "common/ostream.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/value.h"
#include "explorer/base/nonnull.h"
#include "explorer/interpreter/impl_scope.h"
#include "llvm/ADT/DenseMap.h"

namespace Carbon {

// A set of impl matches that we're currently performing. Each `Match`
// represents an attempt to match a `type as interface` query against an `impl`
// declaration. This is used to detect and reject non-termination when impl
// matching recursively triggers further impl matching.
//
// The language rule we use to detect potential non-termination is to count the
// number of times each "label" appears within the type and interface, where a
// label is the name of a declared entity such as a class or interface, or a
// primitive like `type` or `bool`. For example, `Optional(i32*)` contains the
// labels for `Optional`, `i32`, and `*` (built-in pointer type) once each. If
// we ever try matching the same `impl` twice, where the inner match contains
// at least as many appearances of each label as the outer match, we reject the
// program as invalid. We also reject if a query results in the exact same
// query being performed again.
//
// This class is an implementation detail of `TypeChecker::MatchImpl`.
class MatchingImplSet {
 private:
  class LeafCollector;
  enum class Label : int;
  using Signature = llvm::DenseMap<Label, int>;

 public:
  // An RAII type that tracks an impl match that we're currently performing.
  // One instance of this class will exist for each in-progress call to
  // `MatchImpl`.
  class Match {
   public:
    explicit Match(Nonnull<MatchingImplSet*> parent,
                   Nonnull<const ImplScope::ImplFact*> impl,
                   Nonnull<const Value*> type, Nonnull<const Value*> interface);
    ~Match();

    Match(const Match&) = delete;
    auto operator=(const Match&) -> Match& = delete;

    // Check to see if this match duplicates any prior one within the same set,
    // or if there's a simpler form of this match in the set. If so, returns a
    // suitable error. This should be delayed until we know that the impl
    // structurally matches the type and interface.
    auto DiagnosePotentialCycle(SourceLocation source_loc) -> ErrorOr<Success>;

   private:
    friend class LeafCollector;

    // The set that this match is part of.
    Nonnull<MatchingImplSet*> parent_;
    // The `impl` that is being matched against.
    Nonnull<const ImplScope::ImplFact*> impl_;
    // The type that is being matched against the impl.
    Nonnull<const Value*> type_;
    // The interface that is being matched against the impl.
    Nonnull<const Value*> interface_;
    // The number of times each label appears in the type or interface.
    Signature signature_;
  };

 private:
  friend class llvm::DenseMapInfo<Label>;

  // An opaque integer used to identify a particular label appearing in a type,
  // such as a class name. The named enumerators represent builtins, and values
  // >= `FirstDeclarationLabel` represent declarations from the program.
  enum class Label : int {
    // Label for `type` type constant.
    TypeType,
    // Label for `bool` type constant.
    BoolType,
    // Label for `i32` type constant.
    IntType,
    // Label for `String` type constant.
    StringType,
    // Label for `[_;_]` type constructor.
    ArrayType,
    // Label for `_*` type constructor.
    PointerType,
    // Label for `{.a: _, .b: _, ...}` struct type constructor. We use the same
    // label regardless of the arity of the struct type and any field names.
    StructType,
    // Label for `(_, _, ..., _)` tuple type constructor. We use the same label
    // regardless of the arity of the tuple type.
    TupleType,
    // First Label value corresponding to a Declaration. Must be kept at the
    // end of the enum.
    FirstDeclarationLabel
  };

  // Get the Label that represents a given declaration.
  auto GetLabelForDeclaration(const Declaration& declaration) -> Label;

  // The known declarations and their labels.
  llvm::DenseMap<const Declaration*, Label> declaration_labels_;

  // The matches that are currently being performed, in order from outermost to
  // innermost.
  std::vector<Match*> matches_;
};

}  // namespace Carbon

// Support use of Label as a DenseMap key.
template <>
struct llvm::DenseMapInfo<Carbon::MatchingImplSet::Label> {
  using Base = llvm::DenseMapInfo<int>;
  using Label = Carbon::MatchingImplSet::Label;
  static inline auto getEmptyKey() -> Label {
    return static_cast<Label>(Base::getEmptyKey());
  }
  static inline auto getTombstoneKey() -> Label {
    return static_cast<Label>(Base::getTombstoneKey());
  }
  static inline auto getHashValue(Label label) -> unsigned {
    return Base::getHashValue(static_cast<int>(label));
  }
  static auto isEqual(Label a, Label b) -> bool { return a == b; }
};

#endif  // CARBON_EXPLORER_INTERPRETER_MATCHING_IMPL_SET_H_
