// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_ARENA_H_
#define EXECUTABLE_SEMANTICS_COMMON_ARENA_H_

#include <memory>
#include <vector>

#include "llvm/Support/ManagedStatic.h"

namespace Carbon {

namespace ArenaInternal {

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
  explicit ArenaEntryTyped(Args&... args)
      : instance(std::forward<Args>(args)...) {}

  auto Instance() -> T* { return &instance; }

 private:
  T instance;
};

// Manages allocations in an arena for destruction at shutdown.
extern llvm::ManagedStatic<std::vector<std::unique_ptr<ArenaEntry>>> arena;

}  // namespace ArenaInternal

// Allocates an object in the arena, returning a pointer to it.
template <typename T, typename... Args>
static auto ArenaNew(Args&... args) -> T* {
  auto smart_ptr = std::make_unique<ArenaInternal::ArenaEntryTyped<T>>(
      std::forward<Args>(args)...);
  T* raw_ptr = smart_ptr->Instance();
  ArenaInternal::arena->push_back(std::move(smart_ptr));
  return raw_ptr;
}

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_ARENA_H_
