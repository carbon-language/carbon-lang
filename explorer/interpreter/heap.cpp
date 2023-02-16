// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/heap.h"

#include "explorer/common/error_builders.h"
#include "explorer/interpreter/value.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"

namespace Carbon {

auto Heap::AllocateValue(Nonnull<const Value*> v) -> AllocationId {
  // Putting the following two side effects together in this function
  // ensures that we don't do anything else in between, which is really bad!
  // Consider whether to include a copy of the input v in this function
  // or to leave it up to the caller.
  AllocationId a(values_.size());
  values_.push_back(v);
  if (v->kind() == Carbon::Value::Kind::UninitializedValue) {
    states_.push_back(ValueState::Uninitialized);
  } else {
    states_.push_back(ValueState::Alive);
  }
  return a;
}

auto Heap::Read(const Address& a, SourceLocation source_loc) const
    -> ErrorOr<Nonnull<const Value*>> {
  CARBON_RETURN_IF_ERROR(this->CheckInit(a.allocation_, source_loc));
  CARBON_RETURN_IF_ERROR(this->CheckAlive(a.allocation_, source_loc));
  Nonnull<const Value*> value = values_[a.allocation_.index_];
  return value->GetElement(arena_, a.element_path_, source_loc, value);
}

auto Heap::Write(const Address& a, Nonnull<const Value*> v,
                 SourceLocation source_loc) -> ErrorOr<Success> {
  CARBON_RETURN_IF_ERROR(this->CheckAlive(a.allocation_, source_loc));
  if (states_[a.allocation_.index_] == ValueState::Uninitialized) {
    states_[a.allocation_.index_] = ValueState::Alive;
  }
  CARBON_ASSIGN_OR_RETURN(values_[a.allocation_.index_],
                          values_[a.allocation_.index_]->SetField(
                              arena_, a.element_path_, v, source_loc));
  return Success();
}

auto Heap::GetAllocationId(Nonnull<const Value*> v) const
    -> std::optional<AllocationId> {
  auto iter = std::find(values_.begin(), values_.end(), v);
  if (iter != values_.end()) {
    auto index = iter - values_.begin();
    if (states_[index] == ValueState::Alive) {
      return AllocationId(index);
    }
  }
  return std::nullopt;
}

auto Heap::CheckAlive(AllocationId allocation, SourceLocation source_loc) const
    -> ErrorOr<Success> {
  if (states_[allocation.index_] == ValueState::Dead) {
    return ProgramError(source_loc)
           << "undefined behavior: access to dead value "
           << *values_[allocation.index_];
  }
  return Success();
}

auto Heap::CheckInit(AllocationId allocation, SourceLocation source_loc) const
    -> ErrorOr<Success> {
  if (states_[allocation.index_] == ValueState::Uninitialized) {
    return ProgramError(source_loc)
           << "undefined behavior: access to uninitialized value "
           << *values_[allocation.index_];
  }
  return Success();
}

void Heap::Deallocate(AllocationId allocation) {
  if (states_[allocation.index_] != ValueState::Dead) {
    states_[allocation.index_] = ValueState::Dead;
  } else {
    CARBON_FATAL() << "deallocating an already dead value: "
                   << *values_[allocation.index_];
  }
}

void Heap::Deallocate(const Address& a) { Deallocate(a.allocation_); }

void Heap::Print(llvm::raw_ostream& out) const {
  llvm::ListSeparator sep;
  for (size_t i = 0; i < values_.size(); ++i) {
    out << sep;
    out << i << ": ";
    if (states_[i] == ValueState::Uninitialized) {
      out << "!";
    } else if (states_[i] == ValueState::Dead) {
      out << "!!";
    }
    out << *values_[i];
  }
}

}  // namespace Carbon
