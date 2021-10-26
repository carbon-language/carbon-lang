// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_IMPL_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_IMPL_H_

#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/nonnull.h"
#include "executable_semantics/interpreter/address.h"
#include "executable_semantics/interpreter/heap.h"
#include "executable_semantics/interpreter/value.h"

namespace Carbon {

// HeapImpl is the concrete implementation of Heap.
class HeapImpl : public Heap {
 public:
  // Constructs an empty HeapImpl.
  explicit HeapImpl(Nonnull<Arena*> arena) : Heap(), arena_(arena){};

  HeapImpl(const HeapImpl&) = delete;
  auto operator=(const HeapImpl&) -> HeapImpl& = delete;

  auto Read(const Address& a, SourceLocation source_loc)
      -> Nonnull<const Value*> override;
  void Write(const Address& a, Nonnull<const Value*> v,
             SourceLocation source_loc) override;
  auto AllocateValue(Nonnull<const Value*> v) -> AllocationId override;
  void Deallocate(AllocationId allocation) override;
  void PrintAllocation(AllocationId allocation,
                       llvm::raw_ostream& out) const override;
  void Print(llvm::raw_ostream& out) const override;

 private:
  // Signal an error if the address is no longer alive.
  void CheckAlive(AllocationId allocation, SourceLocation source_loc);

  Nonnull<Arena*> arena_;
  std::vector<Nonnull<const Value*>> values_;
  std::vector<bool> alive_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_INTERFACE_H_
