// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_OSTREAM_H_
#define CARBON_COMMON_OSTREAM_H_

#include <ostream>

#include "llvm/Support/raw_os_ostream.h"
// Libraries should include this header instead of raw_ostream.
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"  // IWYU pragma: export

namespace Carbon {

template <typename DerivedT>
class Printable {
  // Provides simple printing for debuggers.
  LLVM_DUMP_METHOD void Dump() const {
    static_cast<const DerivedT*>(this)->Print(llvm::errs());
  }

  // Supports printing to llvm::raw_ostream.
  friend auto operator<<(llvm::raw_ostream& out, const DerivedT& obj)
      -> llvm::raw_ostream& {
    obj.Print(out);
    return out;
  }

  // Supports printing to std::ostream.
  friend auto operator<<(std::ostream& out, const DerivedT& obj)
      -> std::ostream& {
    llvm::raw_os_ostream raw_os(out);
    obj.Print(raw_os);
    return out;
  }

  // Allows GoogleTest and GoogleMock to print pointers by dereferencing them.
  // This is important to allow automatic printing of arguments of mocked
  // APIs.
  friend auto PrintTo(DerivedT* p, std::ostream* out) -> void {
    *out << static_cast<const void*>(p);
    // Also print the object if non-null.
    if (p) {
      *out << " pointing to " << *p;
    }
  }
};

}  // namespace Carbon

namespace llvm {

// Injects an `operator<<` overload into the `llvm` namespace which detects LLVM
// types with `raw_ostream` overloads and uses that to map to a `std::ostream`
// overload. This allows LLVM types to be printed to `std::ostream` via their
// `raw_ostream` operator overloads, which is needed both for logging and
// testing.
//
// To make this overload be unusually low priority, it is designed to take even
// the `std::ostream` parameter as a template, and SFINAE disable itself unless
// that template parameter matches `std::ostream`. This ensures that an
// *explicit* operator will be preferred when provided. Some LLVM types may have
// this, and so we want to prioritize accordingly.
//
// It would be slightly cleaner for LLVM itself to provide this overload in
// `raw_os_ostream.h` so that we wouldn't need to inject into LLVM's namespace,
// but supporting `std::ostream` isn't a priority for LLVM so we handle it
// locally instead.
template <typename StreamT, typename ClassT,
          typename = std::enable_if_t<
              std::is_base_of_v<std::ostream, std::decay_t<StreamT>>>,
          typename = std::enable_if_t<
              !std::is_same_v<std::decay_t<ClassT>, raw_ostream>>>
auto operator<<(StreamT& standard_out, const ClassT& value) -> StreamT& {
  raw_os_ostream(standard_out) << value;
  return standard_out;
}

}  // namespace llvm

#endif  // CARBON_COMMON_OSTREAM_H_
