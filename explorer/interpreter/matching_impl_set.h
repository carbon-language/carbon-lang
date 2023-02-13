// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_MATCHING_IMPL_SET_H_
#define CARBON_EXPLORER_INTERPRETER_MATCHING_IMPL_SET_H_

#include <vector>

#include "common/ostream.h"
#include "explorer/ast/declaration.h"
#include "explorer/common/nonnull.h"
#include "explorer/interpreter/impl_scope.h"
#include "explorer/interpreter/value.h"
#include "llvm/ADT/DenseMap.h"

namespace Carbon {

// A set of impls that we're currently matching against. Used to detect and
// reject non-termination.
class MatchingImplSet {
 private:
  class LeafCollector;

 public:
  // An impl match that we're currently performing.
  class Match {
   public:
    explicit Match(Nonnull<MatchingImplSet*> parent,
                   Nonnull<const ImplScope::Impl*> impl,
                   Nonnull<const Value*> type, Nonnull<const Value*> interface);
    ~Match();

    // Check that this match does not duplicate any prior one. Diagnose if it
    // does.
    auto Check(SourceLocation source_loc) -> ErrorOr<Success>;

   private:
    friend class LeafCollector;

    Nonnull<MatchingImplSet*> parent_;
    Nonnull<const ImplScope::Impl*> impl_;
    Nonnull<const Value*> type_;
    Nonnull<const Value*> interface_;
    llvm::DenseMap<int, int> signature_;
  };

 private:
  // An opaque integer used to identify a particular kind of value appearing in
  // a type, such as a class name.
  enum class ValueKey : int {
    TypeType,
    BoolType,
    IntType,
    StringType,
    ArrayType,
    PointerType,
    FirstDeclarationKey
  };

  // Get the ValueKey to use for a given declaration.
  auto GetKeyForDeclaration(const Declaration& declaration) -> ValueKey;

  // The known declarations and their keys.
  llvm::DenseMap<const Declaration*, ValueKey> declaration_keys_;

  // The matches that are currently being performed.
  std::vector<Match*> matches_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_MATCHING_IMPL_SET_H_
