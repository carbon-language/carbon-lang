// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_BASE_BENCHMARK_HELPERS_H_
#define CARBON_TESTING_BASE_BENCHMARK_HELPERS_H_

#include <type_traits>

namespace Carbon::Testing {

namespace Internal {

// A value that can go directly into a `"+r"` inline assembly constraint.
template <typename T>
concept RegisterValue =
    !std::is_const_v<T> &&
    (std::is_integral_v<T> || std::is_enum_v<T> || std::is_pointer_v<T>) &&
    sizeof(T) <= sizeof(void*);

// A container whose `data()` returns a pointer to its contents.
template <typename T>
concept DataContainer = requires(T& container) {
  requires std::is_pointer_v<decltype(container.data())>;
};

}  // namespace Internal

// Keeps the compiler from optimizing based on `value`, without forcing it out
// of registers.
//
// Use this instead of `benchmark::DoNotOptimize`, whose `"+r,m"` constraint
// Clang implements by storing the value to the stack and loading it back. For
// a value carried between loop iterations, that puts a store and a reload on
// the loop's critical path.
//
// Only types that can be handled entirely in registers are accepted, and any
// other type is a compile error:
//
// - Integers, enums, and pointers go directly into a `"+r"` constraint, so the
//   compiler must assume the call reads and may change `value`.
// - Containers pass their `data()` pointer to the first form.
//
// Both forms also clobber `"memory"`, so the compiler must assume the call
// reads and writes any memory reachable through a pointer passed to it. That
// holds even for a copy of a pointer, such as the result of `data()`: the copy
// itself is discarded, but the memory it points to must be computed before the
// call and can't be assumed unchanged after it.
//
// The container form doesn't keep the compiler from optimizing based on the
// container's size. To block that too, pass the container's address instead,
// which makes the whole container reachable.
//
// TODO: Add forms for more types, such as floating-point values, when
// benchmarks need them, keeping each one in registers.
template <typename T>
  requires Internal::RegisterValue<std::remove_reference_t<T>>
[[clang::always_inline]] inline auto DoNotOptimize(T&& value) -> void {
  __asm__ volatile("" : "+r"(value) : : "memory");
}

template <typename T>
  requires Internal::DataContainer<std::remove_reference_t<T>>
[[clang::always_inline]] inline auto DoNotOptimize(T&& container) -> void {
  DoNotOptimize(container.data());
}

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_BASE_BENCHMARK_HELPERS_H_
