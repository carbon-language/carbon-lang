// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_ARENA_H_
#define EXECUTABLE_SEMANTICS_COMMON_ARENA_H_

#include <memory>
#include <vector>

#include "executable_semantics/common/nonnull.h"

namespace Carbon {

class Arena {
 public:
  // Allocates an object in the arena, returning a pointer to it.
  template <typename T, typename... Args>
  auto New(Args&&... args) -> Nonnull<T*> {
    auto smart_ptr =
        std::make_unique<ArenaEntryTyped<T>>(std::forward<Args>(args)...);
    Nonnull<T*> ptr = smart_ptr->Instance();
    arena_.push_back(std::move(smart_ptr));
    return ptr;
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
        : instance_(std::forward<Args>(args)...) {}

    auto Instance() -> Nonnull<T*> { return Nonnull<T*>(&instance_); }

   private:
    T instance_;
  };

  // Manages allocations in an arena for destruction at shutdown.
  std::vector<std::unique_ptr<ArenaEntry>> arena_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_ARENA_H_
