// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_BASE_PRINT_AS_ID_H_
#define CARBON_EXPLORER_BASE_PRINT_AS_ID_H_

#include "common/ostream.h"

namespace Carbon {

// Helper to support printing the ID for a type that has a method
// `void PrintID(llvm::raw_ostream& out) const`. Usage:
//
//     out << PrintAsID(obj);
template <typename T>
class PrintAsID {
 public:
  explicit PrintAsID(const T& object) : object_(&object) {}

  friend auto operator<<(llvm::raw_ostream& out, const PrintAsID& self)
      -> llvm::raw_ostream& {
    self.object_->PrintID(out);
    return out;
  }

 private:
  const T* object_;
};

template <typename T>
PrintAsID(const T&) -> PrintAsID<T>;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_BASE_PRINT_AS_ID_H_
