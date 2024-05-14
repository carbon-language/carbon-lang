// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/bindings.h"

#include "common/error.h"
#include "explorer/ast/impl_binding.h"
#include "explorer/ast/pattern.h"
#include "explorer/ast/value.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon {

Bindings::Bindings(CloneContext& context, const Bindings& other) {
  for (auto [binding, value] : other.args_) {
    args_.insert({context.Remap(binding), context.Clone(value)});
  }
  for (auto [binding, value] : other.witnesses_) {
    witnesses_.insert({context.Remap(binding), context.Clone(value)});
  }
}

void Bindings::Add(Nonnull<const GenericBinding*> binding,
                   Nonnull<const Value*> value,
                   std::optional<Nonnull<const Value*>> witness) {
  bool added_value = args_.insert({binding, value}).second;
  CARBON_CHECK(added_value) << "Add of already-existing binding";

  if (witness) {
    // TODO: Eventually we should check that we have a witness if and only if
    // the binding has an impl binding.
    auto impl_binding = binding->impl_binding();
    CARBON_CHECK(impl_binding) << "Given witness but have no impl binding";
    bool added_witness = witnesses_.insert({*impl_binding, *witness}).second;
    CARBON_CHECK(added_witness) << "Add of already-existing binding";
  }
}

void Bindings::Print(llvm::raw_ostream& out) const {
  std::vector<std::pair<Nonnull<const GenericBinding*>, Nonnull<const Value*>>>
      args(args_.begin(), args_.end());

  std::vector<std::pair<Nonnull<const ImplBinding*>, Nonnull<const Value*>>>
      witnesses(witnesses_.begin(), witnesses_.end());

  std::stable_sort(args.begin(), args.end(), [](const auto& a, const auto& b) {
    return a.first->index() < b.first->index();
  });

  std::stable_sort(
      witnesses.begin(), witnesses.end(), [](const auto& a, const auto& b) {
        return a.first->type_var()->index() < b.first->type_var()->index();
      });

  llvm::ListSeparator sep;
  out << " >  bindings args: [";
  for (const auto& [binding, value] : args) {
    out << sep << "`" << *binding << "`: `" << *value << "`";
  }
  out << "]\n >  bindings witnesses: [";
  for (const auto& [binding, value] : witnesses) {
    out << sep << "`" << *binding << "`: `" << *value << "`";
  }
  out << "]";
};

auto Bindings::None() -> Nonnull<const Bindings*> {
  static Nonnull<const Bindings*> bindings = new Bindings;
  return bindings;
}

auto Bindings::SymbolicIdentity(
    Nonnull<Arena*> arena,
    llvm::ArrayRef<Nonnull<const GenericBinding*>> bindings)
    -> Nonnull<const Bindings*> {
  auto* result = arena->New<Bindings>();
  for (const auto* binding : bindings) {
    std::optional<Nonnull<const Value*>> witness;
    if (binding->impl_binding()) {
      witness = *binding->impl_binding().value()->symbolic_identity();
    }
    result->Add(binding, *binding->symbolic_identity(), witness);
  }
  return result;
}

}  // namespace Carbon
