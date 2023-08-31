// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_HEAP_H_
#define CARBON_EXPLORER_INTERPRETER_HEAP_H_

#include <vector>

#include "common/ostream.h"
#include "explorer/ast/address.h"
#include "explorer/ast/value.h"
#include "explorer/ast/value_node.h"
#include "explorer/base/nonnull.h"
#include "explorer/base/source_location.h"
#include "explorer/base/trace_stream.h"
#include "explorer/interpreter/heap_allocation_interface.h"

namespace Carbon {

// A Heap represents the abstract machine's dynamically allocated memory.
class Heap : public HeapAllocationInterface, public Printable<Heap> {
 public:
  enum class ValueState {
    Uninitialized,
    Discarded,
    Alive,
    Dead,
  };

  // Constructs an empty Heap.
  explicit Heap(Nonnull<TraceStream*> trace_stream, Nonnull<Arena*> arena)
      : arena_(arena), trace_stream_(trace_stream){};

  Heap(const Heap&) = delete;
  auto operator=(const Heap&) -> Heap& = delete;

  // Returns the value at the given address in the heap after
  // checking that it is alive.
  auto Read(const Address& a, SourceLocation source_loc) const
      -> ErrorOr<Nonnull<const Value*>> override;

  // Writes the given value at the address in the heap after
  // checking that the address is alive.
  auto Write(const Address& a, Nonnull<const Value*> v,
             SourceLocation source_loc) -> ErrorOr<Success> override;

  // Returns whether the value bound at the given node is still alive.
  auto is_bound_value_alive(const ValueNodeView& node, const Address& a) const
      -> bool override;

  void BindValueToReference(const ValueNodeView& node,
                            const Address& a) override;

  // Put the given value on the heap and mark its state.
  // Mark UninitializedValue as uninitialized and other values as alive.
  auto AllocateValue(Nonnull<const Value*> v) -> AllocationId override;

  // Marks this allocation, and all of its sub-objects, as dead.
  auto Deallocate(AllocationId allocation) -> ErrorOr<Success> override;
  auto Deallocate(const Address& a) -> ErrorOr<Success>;

  // Marks this allocation, and all its sub-objects, as discarded.
  void Discard(AllocationId allocation);
  // Returns whether the given allocation was unused and discarded.
  auto is_discarded(AllocationId allocation) const -> bool;
  // Returns whether the given allocation was initialized.
  auto is_initialized(AllocationId allocation) const -> bool;

  // Print all the values on the heap to the stream `out`.
  void Print(llvm::raw_ostream& out) const;

  auto arena() const -> Arena& override { return *arena_; }

 private:
  // Returns whether the address have the same AllocationdId and their path
  // are strictly nested.
  static auto AddressesAreStrictlyNested(const Address& first,
                                         const Address& second) -> bool;

  // Returns whether the provided paths are strictly nested. This checks the
  // name, index, and base element only, and might not valid if used to
  // compare paths based on a different AllocationId.
  static auto PathsAreStrictlyNested(const ElementPath& first,
                                     const ElementPath& second) -> bool;

  // Signal an error if the allocation is no longer alive.
  auto CheckAlive(AllocationId allocation, SourceLocation source_loc) const
      -> ErrorOr<Success>;

  // Signal an error if the allocation has not been initialized.
  auto CheckInit(AllocationId allocation, SourceLocation source_loc) const
      -> ErrorOr<Success>;

  Nonnull<Arena*> arena_;
  std::vector<Nonnull<const Value*>> values_;
  std::vector<ValueState> states_;
  std::vector<llvm::DenseMap<const AstNode*, Address>> bound_values_;
  Nonnull<TraceStream*> trace_stream_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_HEAP_H_
