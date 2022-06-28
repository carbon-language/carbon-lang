// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_BINDINGS_H_
#define CARBON_EXPLORER_AST_BINDINGS_H_

#include <map>

#include "explorer/common/nonnull.h"

namespace Carbon {

class ImplBinding;
class GenericBinding;
class Value;

using BindingMap =
    std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>;
using ImplWitnessMap =
    std::map<Nonnull<const ImplBinding*>, Nonnull<const Value*>>;

// A set of evaluated bindings in some context, such as a function or class.
//
// These are shared by a context and all unparameterized entities within that
// context. For example, a class and the name of a method within that class
// will have the same set of bindings.
class Bindings {
 public:
  // Create an instantiated set of bindings for use during evaluation,
  // containing both arguments and witnesses.
  Bindings(BindingMap args, ImplWitnessMap witnesses)
      : args_(args), witnesses_(witnesses) {}

  enum NoWitnessesTag { NoWitnesses };

  // Create a set of bindings for use during type-checking, containing only the
  // arguments but not the corresponding witnesses.
  Bindings(BindingMap args, NoWitnessesTag) : args_(args), witnesses_() {}

  // Argument values corresponding to generic bindings.
  auto args() const -> const BindingMap& { return args_; }

  // Witnesses corresponding to impl bindings.
  auto witnesses() const -> const ImplWitnessMap& { return witnesses_; }

  // An empty set of bindings.
  static auto None() -> Nonnull<const Bindings*>;

 private:
  BindingMap args_;
  ImplWitnessMap witnesses_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_BINDINGS_H_
