// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check.h"
#include "common/error.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/value.h"
#include "explorer/common/error_builders.h"
#include "explorer/common/nonnull.h"
#include "explorer/common/source_location.h"
#include "explorer/interpreter/type_checker.h"

namespace Carbon {

// A visitor for type values that collects leaf 'labels', such as class names,
// and adds them to the signature of a `Match` object.
class MatchingImplSet::LeafCollector {
 public:
  explicit LeafCollector(Match* match) : match_(match) {}

  void Collect(const Value* value) {
    value->Visit<void>(
        [&](const auto* derived_value) { VisitValue(derived_value); });
  }

  void Collect(Label label) { ++match_->signature_[label]; }

 private:
  // Most kinds of value don't contribute to the signature.
  void VisitValue(const Value* /*unused*/) {}

  void VisitValue(const TypeType* /*unused*/) { Collect(Label::TypeType); }

  void VisitValue(const BoolType* /*unused*/) { Collect(Label::BoolType); }

  void VisitValue(const IntType* /*unused*/) { Collect(Label::IntType); }

  void VisitValue(const StringType* /*unused*/) { Collect(Label::StringType); }

  void VisitValue(const StaticArrayType* array) {
    Collect(Label::ArrayType);
    Collect(&array->element_type());
  }

  void VisitValue(const PointerType* pointer) {
    Collect(Label::PointerType);
    Collect(&pointer->pointee_type());
  }

  void VisitValue(const StructType* struct_type) {
    Collect(Label::StructType);
    for (auto [name, type] : struct_type->fields()) {
      Collect(type);
    }
  }

  void VisitValue(const TupleType* tuple_type) {
    Collect(Label::TupleType);
    for (const auto* elem_type : tuple_type->elements()) {
      Collect(elem_type);
    }
  }

  void VisitValue(const NominalClassType* class_type) {
    VisitDeclarationAndArgs(class_type->declaration(), class_type->bindings());
  }

  void VisitValue(const MixinPseudoType* mixin_type) {
    VisitDeclarationAndArgs(mixin_type->declaration(), mixin_type->bindings());
  }

  void VisitValue(const InterfaceType* iface_type) {
    VisitDeclarationAndArgs(iface_type->declaration(), iface_type->bindings());
  }

  void VisitValue(const NamedConstraintType* constraint_type) {
    VisitDeclarationAndArgs(constraint_type->declaration(),
                            constraint_type->bindings());
  }

  void VisitValue(const ChoiceType* choice_type) {
    VisitDeclarationAndArgs(choice_type->declaration(),
                            choice_type->bindings());
  }

  void VisitDeclarationAndArgs(const Declaration& declaration,
                               const Bindings& bindings) {
    Collect(match_->parent_->GetLabelForDeclaration(declaration));
    for (auto [key, value] : bindings.args()) {
      Collect(value);
    }
  }

  Match* match_;
};

auto MatchingImplSet::GetLabelForDeclaration(const Declaration& declaration)
    -> Label {
  auto [it, added] = declaration_labels_.insert(
      {&declaration,
       static_cast<Label>(static_cast<int>(Label::FirstDeclarationLabel) +
                          declaration_labels_.size())});
  return it->second;
}

MatchingImplSet::Match::Match(Nonnull<MatchingImplSet*> parent,
                              Nonnull<const ImplScope::ImplFact*> impl,
                              Nonnull<const Value*> type,
                              Nonnull<const Value*> interface)
    : parent_(parent), impl_(impl), type_(type), interface_(interface) {
  // Build our signature.
  LeafCollector collector(this);
  collector.Collect(type);
  collector.Collect(interface);

  parent_->matches_.push_back(this);
}

MatchingImplSet::Match::~Match() {
  CARBON_CHECK(parent_->matches_.back() == this) << "match stack broken";
  parent_->matches_.pop_back();
}

auto MatchingImplSet::Match::DiagnosePotentialCycle(SourceLocation source_loc)
    -> ErrorOr<Success> {
  for (auto* match : parent_->matches_) {
    if (match != this && match->impl_ == impl_) {
      // Whether all labels appear a greater or equal number of times in this
      // match than in `match`.
      bool all_greater_or_equal = true;
      // Whether any label appears strictly more times in this match than in
      // `match`.
      bool any_greater = false;

      for (auto [key, value] : signature_) {
        int other_value = match->signature_.lookup(key);
        if (value < other_value) {
          all_greater_or_equal = false;
          break;
        }
        if (value > other_value) {
          any_greater = true;
        }
      }

      if (all_greater_or_equal) {
        if (any_greater) {
          return ProgramError(source_loc)
                 << "impl matching recursively performed a more complex match "
                    "using the same impl\n"
                 << "  outer match: " << *match->type_ << " as "
                 << *match->interface_ << "\n"
                 << "  inner match: " << *type_ << " as " << *interface_;
        }
        if (ValueEqual(match->type_, type_, std::nullopt) &&
            ValueEqual(match->interface_, interface_, std::nullopt)) {
          return ProgramError(source_loc)
                 << "impl matching for " << *type_ << " as " << *interface_
                 << " recursively performed a match for the same type and "
                    "interface";
        }
      }
    }
  }

  return Success();
}

}  // namespace Carbon
