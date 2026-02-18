// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CPP_INITIALIZER_LIST_H_
#define CARBON_TOOLCHAIN_SEM_IR_CPP_INITIALIZER_LIST_H_

#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

class File;

// The layout of `std::initializer_list` that we are dealing with.
struct StdInitializerListLayout {
  enum Kind : int8_t {
    // Not a recognized layout.
    None,
    // `struct { T* begin; T* end; }`
    PointerPointer,
    // `struct { T* begin; size_t size; }`
    PointerInt,
  };
  Kind kind = Kind::None;
  // If the kind is PointerInt, the type of the size.
  TypeId size_type_id = TypeId::None;
};

// Returns the kind of `std::initializer_list` that `type_id` represents, or
// `None` if it is not a `std::initializer_list`. This does not verify that
// `type_id` is actually a type named `std::initializer_list`, only that it has
// a recognized set of fields that allows us to treat it as one.
auto GetStdInitializerListLayout(const File& sem_ir, TypeId type_id)
    -> StdInitializerListLayout;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CPP_INITIALIZER_LIST_H_
