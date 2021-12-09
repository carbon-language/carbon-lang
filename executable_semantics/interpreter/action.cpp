// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/action.h"

#include <iterator>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/interpreter/stack.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

DynamicScope::DynamicScope(DynamicScope&& other) noexcept
    : locals_(std::move(other.locals_)),
      allocations_(std::exchange(other.allocations_, {})),
      heap_(other.heap_) {}

auto DynamicScope::operator=(DynamicScope&& rhs) noexcept -> DynamicScope& {
  locals_ = std::move(rhs.locals_);
  allocations_ = std::exchange(rhs.allocations_, {});
  heap_ = rhs.heap_;
  return *this;
}

DynamicScope::~DynamicScope() {
  for (AllocationId allocation : allocations_) {
    heap_->Deallocate(allocation);
  }
}

void DynamicScope::Print(llvm::raw_ostream& out) const {
  out << "{";
  llvm::ListSeparator sep;
  for (const auto& [named_entity, value] : locals_) {
    out << sep << named_entity.name() << ": " << *value;
  }
  out << "}";
}

void DynamicScope::Initialize(NamedEntityView named_entity,
                              Nonnull<const Value*> value) {
  CHECK(!named_entity.constant_value().has_value());
  CHECK(value->kind() != Value::Kind::LValue);
  allocations_.push_back(heap_->AllocateValue(value));
  auto [it, success] = locals_.insert(
      {named_entity, heap_->arena().New<LValue>(Address(allocations_.back()))});
  CHECK(success) << "Duplicate definition of " << named_entity.name();
}

void DynamicScope::Merge(DynamicScope other) {
  locals_.merge(std::move(other.locals_));
  CHECK(other.locals_.size() == 0)
      << "Duplicate definition of " << other.locals_.size()
      << " names, including " << other.locals_.begin()->first.name();
  allocations_.insert(allocations_.end(), other.allocations_.begin(),
                      other.allocations_.end());
  other.allocations_.clear();
  CHECK(heap_ == other.heap_);
}

auto DynamicScope::Get(NamedEntityView named_entity) const
    -> std::optional<Nonnull<const LValue*>> {
  auto it = locals_.find(named_entity);
  if (it != locals_.end()) {
    return it->second;
  } else {
    return std::nullopt;
  }
}

auto DynamicScope::Capture(
    const std::vector<Nonnull<const DynamicScope*>>& scopes) -> DynamicScope {
  DynamicScope result(scopes.front()->heap_);
  for (Nonnull<const DynamicScope*> scope : scopes) {
    for (const auto& entry : scope->locals_) {
      // Intentionally disregards duplicates later in the vector.
      result.locals_.insert(entry);
    }
    CHECK(scope->heap_ == result.heap_);
  }
  return result;
}

void Action::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case Action::Kind::LValAction:
      out << cast<LValAction>(*this).expression();
      break;
    case Action::Kind::ExpressionAction:
      out << cast<ExpressionAction>(*this).expression();
      break;
    case Action::Kind::PatternAction:
      out << cast<PatternAction>(*this).pattern();
      break;
    case Action::Kind::StatementAction:
      cast<StatementAction>(*this).statement().PrintDepth(1, out);
      break;
    case Action::Kind::DeclarationAction:
      cast<DeclarationAction>(*this).declaration().Print(out);
      break;
    case Action::Kind::ScopeAction:
      out << "ScopeAction";
  }
  out << "<" << pos_ << ">";
  if (results_.size() > 0) {
    out << "(";
    llvm::ListSeparator sep;
    for (auto& result : results_) {
      out << sep << *result;
    }
    out << ")";
  }
}

void Action::PrintList(const Stack<Nonnull<Action*>>& ls,
                       llvm::raw_ostream& out) {
  llvm::ListSeparator sep(" :: ");
  for (const auto& action : ls) {
    out << sep << *action;
  }
}

}  // namespace Carbon
