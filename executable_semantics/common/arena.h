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
class ArenaPtr {
 public:
  virtual ~ArenaPtr() = default;
};

// Templated destruction of a pointer through the smart pointer. This avoids
// per-type arenas.
template <typename T>
class ArenaPtrTyped : public ArenaPtr {
 public:
  ArenaPtrTyped(T* raw_ptr) : ptr(raw_ptr) {}

 private:
  std::unique_ptr<T> ptr;
};

// Manages allocations in an arena for destruction at shutdown.
extern llvm::ManagedStatic<std::vector<std::unique_ptr<ArenaPtr>>> arena;

}  // namespace ArenaInternal

template <typename T, typename... Args>
static T* ArenaNew(Args&... args) {
  T* ptr = new T(std::forward<Args>(args)...);
  ArenaInternal::arena->push_back(
      std::make_unique<ArenaInternal::ArenaPtrTyped<T>>(ptr));
  return ptr;
}

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_ARENA_H_
