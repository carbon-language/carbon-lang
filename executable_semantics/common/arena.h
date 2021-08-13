// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_ARENA_H_
#define EXECUTABLE_SEMANTICS_COMMON_ARENA_H_

#include <memory>
#include <vector>

#include "executable_semantics/common/ptr.h"
#include "llvm/Support/ManagedStatic.h"

namespace Carbon {

class Arena {
 public:
  // Allocates an object in the arena, returning a pointer to it.
  template <typename T, typename... Args>
  auto New(Args&&... args) -> Ptr<T> {
    auto smart_ptr =
        std::make_unique<ArenaEntryTyped<T>>(std::forward<Args>(args)...);
    T* raw_ptr = smart_ptr->Instance();
    arena.push_back(std::move(smart_ptr));
    return Ptr<T>(raw_ptr);
  }

  // TODO: Remove. This is only to help findability during migration.
  template <typename T, typename... Args>
  auto RawNew(Args&&... args) -> T* {
    auto smart_ptr =
        std::make_unique<ArenaEntryTyped<T>>(std::forward<Args>(args)...);
    T* raw_ptr = smart_ptr->Instance();
    arena.push_back(std::move(smart_ptr));
    return raw_ptr;
  }

 private:
  // Virtualizes arena entries so that a single vector can contain many types,
  // avoiding templated statics.
  class ArenaEntry {
   public:
    virtual ~ArenaEntry() = default;
  };

  // Templated destruction of a pointer.
  template <typename T>
  class ArenaEntryTyped : public ArenaEntry {
   public:
    template <typename... Args>
    explicit ArenaEntryTyped(Args&&... args)
        : instance(std::forward<Args>(args)...) {}

    auto Instance() -> T* { return &instance; }

   private:
    T instance;
  };

  // Manages allocations in an arena for destruction at shutdown.
  std::vector<std::unique_ptr<ArenaEntry>> arena;
};

extern llvm::ManagedStatic<Arena> global_arena;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_ARENA_H_
