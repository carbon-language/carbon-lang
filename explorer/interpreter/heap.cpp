// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/heap.h"

#include "common/check.h"
#include "common/error.h"
#include "explorer/ast/value.h"
#include "explorer/base/error_builders.h"
#include "explorer/base/source_location.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"

namespace Carbon {

auto Heap::AllocateValue(Nonnull<const Value*> v) -> AllocationId {
  // Putting the following two side effects together in this function
  // ensures that we don't do anything else in between, which would be really
  // bad! Consider whether to include a copy of the input v in this function or
  // to leave it up to the caller.
  AllocationId a(values_.size());
  values_.push_back(v);
  bool is_uninitialized = false;

  if (v->kind() == Carbon::Value::Kind::UninitializedValue) {
    states_.push_back(ValueState::Uninitialized);
    is_uninitialized = true;
  } else {
    states_.push_back(ValueState::Alive);
  }
  bound_values_.push_back(llvm::DenseMap<const AstNode*, Address>{});

  if (trace_stream_->is_enabled()) {
    trace_stream_->Allocate()
        << "memory-alloc: #" << a.index_ << " `" << *v << "`"
        << (is_uninitialized ? " uninitialized" : "") << "\n";
  }

  return a;
}

auto Heap::Read(const Address& a, SourceLocation source_loc) const
    -> ErrorOr<Nonnull<const Value*>> {
  CARBON_RETURN_IF_ERROR(this->CheckInit(a.allocation_, source_loc));
  CARBON_RETURN_IF_ERROR(this->CheckAlive(a.allocation_, source_loc));
  Nonnull<const Value*> value = values_[a.allocation_.index_];
  ErrorOr<Nonnull<const Value*>> read_value =
      value->GetElement(arena_, a.element_path_, source_loc, value);

  if (trace_stream_->is_enabled()) {
    trace_stream_->Read() << "memory-read: #" << a.allocation_.index_ << " `"
                          << **read_value << "`\n";
  }

  return read_value;
}

auto Heap::Write(const Address& a, Nonnull<const Value*> v,
                 SourceLocation source_loc) -> ErrorOr<Success> {
  CARBON_RETURN_IF_ERROR(this->CheckAlive(a.allocation_, source_loc));
  if (states_[a.allocation_.index_] == ValueState::Uninitialized) {
    if (!a.element_path_.IsEmpty()) {
      return ProgramError(source_loc)
             << "undefined behavior: store to subobject of uninitialized value "
             << *values_[a.allocation_.index_];
    }
    states_[a.allocation_.index_] = ValueState::Alive;
  }
  CARBON_ASSIGN_OR_RETURN(values_[a.allocation_.index_],
                          values_[a.allocation_.index_]->SetField(
                              arena_, a.element_path_, v, source_loc));
  auto& bound_values_map = bound_values_[a.allocation_.index_];
  // End lifetime of all values bound to this address and its subobjects.
  if (a.element_path_.IsEmpty()) {
    bound_values_map.clear();
  } else {
    for (auto value_it = bound_values_map.begin();
         value_it != bound_values_map.end(); ++value_it) {
      if (AddressesAreStrictlyNested(a, value_it->second)) {
        bound_values_map.erase(value_it);
      }
    }
  }

  if (trace_stream_->is_enabled()) {
    trace_stream_->Write() << "memory-write: #" << a.allocation_.index_ << " `"
                           << *values_[a.allocation_.index_] << "`\n";
  }

  return Success();
}

auto Heap::CheckAlive(AllocationId allocation, SourceLocation source_loc) const
    -> ErrorOr<Success> {
  const auto state = states_[allocation.index_];
  if (state == ValueState::Dead || state == ValueState::Discarded) {
    return ProgramError(source_loc)
           << "undefined behavior: access to dead or discarded value "
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

auto Heap::Deallocate(AllocationId allocation) -> ErrorOr<Success> {
  if (states_[allocation.index_] != ValueState::Dead) {
    states_[allocation.index_] = ValueState::Dead;
  } else {
    CARBON_FATAL() << "deallocating an already dead value: "
                   << *values_[allocation.index_];
  }

  if (trace_stream_->is_enabled()) {
    trace_stream_->Deallocate() << "memory-dealloc: #" << allocation.index_
                                << " `" << *values_[allocation.index_] << "`\n";
  }

  return Success();
}

auto Heap::Deallocate(const Address& a) -> ErrorOr<Success> {
  return Deallocate(a.allocation_);
}

auto Heap::is_initialized(AllocationId allocation) const -> bool {
  return states_[allocation.index_] != ValueState::Uninitialized;
}

auto Heap::is_discarded(AllocationId allocation) const -> bool {
  return states_[allocation.index_] == ValueState::Discarded;
}

void Heap::Discard(AllocationId allocation) {
  CARBON_CHECK(states_[allocation.index_] == ValueState::Uninitialized);
  states_[allocation.index_] = ValueState::Discarded;
}

void Heap::BindValueToReference(const ValueNodeView& node, const Address& a) {
  // Update mapped node ignoring any previous mapping.
  bound_values_[a.allocation_.index_].insert({&node.base(), a});
}

auto Heap::is_bound_value_alive(const ValueNodeView& node,
                                const Address& a) const -> bool {
  return bound_values_[a.allocation_.index_].contains(&node.base());
}

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

auto Heap::AddressesAreStrictlyNested(const Address& first,
                                      const Address& second) -> bool {
  if (first.allocation_.index_ != second.allocation_.index_) {
    return false;
  }
  return PathsAreStrictlyNested(first.element_path_, second.element_path_);
}

auto Heap::PathsAreStrictlyNested(const ElementPath& first,
                                  const ElementPath& second) -> bool {
  for (size_t i = 0;
       i < std::min(first.components_.size(), second.components_.size()); ++i) {
    Nonnull<const Element*> element = first.components_[i].element();
    Nonnull<const Element*> other_element = second.components_[i].element();
    if (element->kind() != other_element->kind()) {
      return false;
    }
    switch (element->kind()) {
      case Carbon::ElementKind::NamedElement:
        if (!element->IsNamed(
                llvm::cast<NamedElement>(other_element)->name())) {
          return false;
        }
        break;
      case Carbon::ElementKind::PositionalElement:
        if (llvm::cast<PositionalElement>(element)->index() !=
            llvm::cast<PositionalElement>(other_element)->index()) {
          return false;
        }
        break;
      case Carbon::ElementKind::BaseElement:
        // Nothing to test.
        break;
    }
  }
  return true;
}
}  // namespace Carbon
