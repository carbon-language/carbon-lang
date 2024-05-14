// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/type_structure.h"

#include <limits>

#include "explorer/ast/declaration.h"
#include "explorer/ast/expression_category.h"
#include "explorer/ast/value.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon {

namespace {
struct TypeStructureBuilder {
  // Visit a child of the current value at the given index.
  template <typename T>
  void VisitChild(int index, const T& child) {
    path.push_back(index);
    Visit(child);
    path.pop_back();
  }

  // Visit an instance of a type derived from Value. By default we do this by
  // decomposing the value and walking each piece in turn.
  template <typename T>
  void VisitValue(Nonnull<const T*> value) {
    value->Decompose([&](const auto&... parts) {
      int inner_index = 0;
      (VisitChild(inner_index++, parts), ...);
    });
  }

  // A variable type is a hole.
  void VisitValue(Nonnull<const VariableType*>) { AddHole(); }

  // Visit a value by visiting its derived type.
  void Visit(Nonnull<const Value*> value) {
    value->Visit<void>([&](auto* derived_value) { VisitValue(derived_value); });
  }

  // Visit all of the arguments in a list of bindings.
  void Visit(Nonnull<const Bindings*> bindings) {
    // Reconstruct the lexical ordering of the parameters.
    // TODO: Store bindings as an array indexed by binding index, not as a map.
    std::vector<Nonnull<const GenericBinding*>> params;
    for (auto [param, value] : bindings->args()) {
      params.push_back(param);
    }
    std::sort(params.begin(), params.end(), [](auto* param_1, auto* param_2) {
      return param_1->index() < param_2->index();
    });

    for (int i = 0; i != static_cast<int>(params.size()); ++i) {
      VisitChild(i, bindings->args().find(params[i])->second);
    }
  }

  template <typename T>
  void Visit(const std::optional<T>& opt) {
    if (opt) {
      Visit(*opt);
    }
  }

  template <typename T>
  void Visit(const std::vector<T>& seq) {
    for (int i = 0; i != static_cast<int>(seq.size()); ++i) {
      VisitChild(i, seq[i]);
    }
  }

  void Visit(const NamedValue& value) { Visit(value.value); }

  // Ignore values that can't contain holes.
  void Visit(int) {}
  void Visit(std::string_view) {}
  void Visit(ExpressionCategory) {}
  void Visit(Nonnull<const AstNode*>) {}
  void Visit(const ValueNodeView&) {}
  void Visit(const Address&) {}
  void Visit(const VTable*) {}
  void Visit(const FunctionType::GenericParameter&) {}
  void Visit(const FunctionType::MethodSelf&) {}
  void Visit(const NamedElement*) {}

  // Constraint types can contain mentions of VariableTypes, but they aren't
  // deducible so it's not important to look for them.
  void Visit(const ImplsConstraint&) {}
  void Visit(const IntrinsicConstraint&) {}
  void Visit(const EqualityConstraint&) {}
  void Visit(const RewriteConstraint&) {}
  void Visit(const LookupContext&) {}

  // TODO: Find a way to remove the derived-most pointer from NominalClassValue.
  void Visit(Nonnull<const NominalClassValue**>) {}

  void AddHole() {
    if (!result.empty()) {
      result.push_back(-1);
    }
    result.insert(result.end(), path.begin(), path.end());
  }

  std::vector<int> path;
  std::vector<int> result;
};
}  // namespace

auto TypeStructureSortKey::ForImpl(Nonnull<const Value*> type,
                                   Nonnull<const Value*> interface)
    -> TypeStructureSortKey {
  TypeStructureBuilder builder;
  builder.VisitChild(0, type);
  builder.VisitChild(1, interface);

  TypeStructureSortKey result;
  result.holes_ = std::move(builder.result);
  result.holes_.push_back(std::numeric_limits<int>::max());
  return result;
}

void TypeStructureSortKey::Print(llvm::raw_ostream& out) const {
  out << "[";
  llvm::ListSeparator sep;
  for (int i : holes_) {
    if (i == -1) {
      out << "; ";
      // Reinitialize `sep` to suppress the next separator.
      sep = llvm::ListSeparator();
    } else if (i == std::numeric_limits<int>::max()) {
      out << "]";
    } else {
      out << sep << i;
    }
  }
}

void TypeStructureSortKey::Dump() const { Print(llvm::errs()); }

}  // namespace Carbon
