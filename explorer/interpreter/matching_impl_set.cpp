// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check.h"
#include "common/error.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/value.h"
#include "explorer/base/error_builders.h"
#include "explorer/base/nonnull.h"
#include "explorer/base/source_location.h"
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
    for (const auto& [name, type] : struct_type->fields()) {
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
  CARBON_CHECK(parent_->matches_.back() == this, "match stack broken");
  parent_->matches_.pop_back();
}

auto MatchingImplSet::Match::DiagnosePotentialCycle(SourceLocation source_loc)
    -> ErrorOr<Success> {
  // Determine whether any labels in 'a' have a higher count than in 'b'.
  auto any_labels_with_higher_count = [](const Signature& a,
                                         const Signature& b) {
    if (a.size() > b.size()) {
      // Every label in a signature has a count of at least one.
      return true;
    }
    for (auto [key, a_value] : a) {
      int b_value = b.lookup(key);
      if (a_value > b_value) {
        return true;
      }
    }
    return false;
  };

  for (auto* match : parent_->matches_) {
    if (match != this && match->impl_ == impl_ &&
        !any_labels_with_higher_count(match->signature_, signature_)) {
      // No label in the outer match has a higher count than the same label in
      // the inner match. We might have reached a cycle.
      if (any_labels_with_higher_count(signature_, match->signature_)) {
        // The inner match has a higher count for some label. This query is
        // strictly more complex than the outer one, so reject this potential
        // cycle.
        // TODO: Track which label has a higher count, map it back to a string,
        // and include it in this diagnostic.
        return ProgramError(source_loc)
               << "impl matching recursively performed a more complex match "
                  "using the same impl\n"
               << "  outer match: " << *match->type_ << " as "
               << *match->interface_ << "\n"
               << "  inner match: " << *type_ << " as " << *interface_;
      }

      if (ValueEqual(match->type_, type_, std::nullopt) &&
          ValueEqual(match->interface_, interface_, std::nullopt)) {
        // We hit the same query twice recursively. This is definitely a cycle.
        return ProgramError(source_loc)
               << "impl matching for " << *type_ << " as " << *interface_
               << " recursively performed a match for the same type and "
                  "interface";
      }
    }
  }

  return Success();
}

}  // namespace Carbon
