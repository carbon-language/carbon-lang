// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/action.h"

#include <iterator>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "common/check.h"
#include "common/error.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/value.h"
#include "explorer/common/arena.h"
#include "explorer/common/source_location.h"
#include "explorer/interpreter/stack.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

RuntimeScope::RuntimeScope(RuntimeScope&& other) noexcept
    : locals_(std::move(other.locals_)),
      pinned_values_(std::move(other.pinned_values_)),
      // To transfer ownership of other.allocations_, we have to empty it out.
      allocations_(std::exchange(other.allocations_, {})),
      heap_(other.heap_) {}

auto RuntimeScope::operator=(RuntimeScope&& rhs) noexcept -> RuntimeScope& {
  locals_ = std::move(rhs.locals_);
  pinned_values_ = std::move(rhs.pinned_values_);
  // To transfer ownership of rhs.allocations_, we have to empty it out.
  allocations_ = std::exchange(rhs.allocations_, {});
  heap_ = rhs.heap_;
  return *this;
}

void RuntimeScope::Print(llvm::raw_ostream& out) const {
  out << "{";
  llvm::ListSeparator sep;
  for (const auto& [value_node, value] : locals_) {
    out << sep << value_node.base() << ": " << *value;
  }
  out << "}";
}

void RuntimeScope::Bind(ValueNodeView value_node, Address address) {
  CARBON_CHECK(!value_node.constant_value().has_value());
  bool success =
      locals_.insert({value_node, heap_->arena().New<LocationValue>(address)})
          .second;
  CARBON_CHECK(success) << "Duplicate definition of " << value_node.base();
}

auto RuntimeScope::BindAndPin(ValueNodeView value_node, Address address,
                              SourceLocation source_loc) -> ErrorOr<Success> {
  Bind(value_node, address);
  CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> pinned_value,
                          heap_->Read(address, source_loc));
  bool success = pinned_values_.insert({value_node, pinned_value}).second;
  CARBON_CHECK(success) << "Duplicate pinned node for " << value_node.base();
  return Success();
}

void RuntimeScope::BindLifetimeToScope(Address address) {
  CARBON_CHECK(address.element_path_.IsEmpty())
      << "Cannot extend lifetime of a specific sub-element";
  allocations_.push_back(address.allocation_);
}

void RuntimeScope::BindValue(ValueNodeView value_node,
                             Nonnull<const Value*> value) {
  CARBON_CHECK(!value_node.constant_value().has_value());
  CARBON_CHECK(value->kind() != Value::Kind::LocationValue);
  bool success = locals_.insert({value_node, value}).second;
  CARBON_CHECK(success) << "Duplicate definition of " << value_node.base();
}

auto RuntimeScope::Initialize(ValueNodeView value_node,
                              Nonnull<const Value*> value)
    -> Nonnull<const LocationValue*> {
  CARBON_CHECK(!value_node.constant_value().has_value());
  CARBON_CHECK(value->kind() != Value::Kind::LocationValue);
  allocations_.push_back(heap_->AllocateValue(value));
  const auto* location =
      heap_->arena().New<LocationValue>(Address(allocations_.back()));
  bool success = locals_.insert({value_node, location}).second;
  CARBON_CHECK(success) << "Duplicate definition of " << value_node.base();
  return location;
}

void RuntimeScope::Merge(RuntimeScope other) {
  CARBON_CHECK(heap_ == other.heap_);
  for (auto& element : other.locals_) {
    CARBON_CHECK(locals_.count(element.first) == 0)
        << "Duplicate definition of" << element.first;
    locals_.insert(element);
  }
  for (auto& element : other.pinned_values_) {
    CARBON_CHECK(pinned_values_.count(element.first) == 0)
        << "Duplicate definition of" << element.first;
    pinned_values_.insert(element);
  }
  allocations_.insert(allocations_.end(), other.allocations_.begin(),
                      other.allocations_.end());
  other.allocations_.clear();
}

auto RuntimeScope::Get(ValueNodeView value_node,
                       SourceLocation source_loc) const
    -> ErrorOr<std::optional<Nonnull<const Value*>>> {
  auto it = locals_.find(value_node);
  if (it == locals_.end()) {
    return {std::nullopt};
  }
  auto pinned_it = pinned_values_.find(value_node);
  if (pinned_it != pinned_values_.end()) {
    // Check if pinned value was modified.
    CARBON_CHECK(it->second->kind() == Value::Kind::LocationValue);
    CARBON_ASSIGN_OR_RETURN(
        Nonnull<const Value*> current,
        heap_->Read(cast<LocationValue>(it->second)->address(), source_loc));
    if (current != pinned_it->second) {
      return ProgramError(source_loc)
             << "Value has changed since it was pinned";
    }
  }
  return {it->second};
}

auto RuntimeScope::Capture(
    const std::vector<Nonnull<const RuntimeScope*>>& scopes) -> RuntimeScope {
  CARBON_CHECK(!scopes.empty());
  RuntimeScope result(scopes.front()->heap_);
  for (Nonnull<const RuntimeScope*> scope : scopes) {
    CARBON_CHECK(scope->heap_ == result.heap_);
    for (const auto& entry : scope->locals_) {
      // Intentionally disregards duplicates later in the vector.
      result.locals_.insert(entry);
    }
  }
  return result;
}

void Action::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case Action::Kind::LocationAction:
      out << cast<LocationAction>(*this).expression() << " ";
      break;
    case Action::Kind::ExpressionAction:
      out << cast<ExpressionAction>(*this).expression() << " ";
      break;
    case Action::Kind::ExpressionCategoryAction:
      out << cast<ExpressionCategoryAction>(*this).expression() << " ";
      break;
    case Action::Kind::WitnessAction:
      out << *cast<WitnessAction>(*this).witness() << " ";
      break;
    case Action::Kind::StatementAction:
      cast<StatementAction>(*this).statement().PrintDepth(1, out);
      out << " ";
      break;
    case Action::Kind::DeclarationAction:
      cast<DeclarationAction>(*this).declaration().Print(out);
      out << " ";
      break;
    case Action::Kind::TypeInstantiationAction:
      cast<TypeInstantiationAction>(*this).type()->Print(out);
      out << " ";
      break;
    case Action::Kind::ScopeAction:
      break;
    case Action::Kind::RecursiveAction:
      out << "recursive";
      break;
    case Action::Kind::CleanUpAction:
      out << "clean up";
      break;
    case Action::Kind::DestroyAction:
      out << "destroy";
      break;
  }
  out << "." << pos_ << ".";
  if (!results_.empty()) {
    out << " [[";
    llvm::ListSeparator sep;
    for (const auto& result : results_) {
      out << sep << *result;
    }
    out << "]]";
  }
  if (scope_.has_value()) {
    out << " " << *scope_;
  }
}

}  // namespace Carbon
