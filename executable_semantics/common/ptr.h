// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_PTR_H_
#define EXECUTABLE_SEMANTICS_COMMON_PTR_H_

namespace Carbon {

// A non-nullable pointer. Written as `Ptr<T>` instead of `T*`.
template <typename T>
using Ptr = T* _Nonnull __attribute__((nonnull));

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_PTR_H_
