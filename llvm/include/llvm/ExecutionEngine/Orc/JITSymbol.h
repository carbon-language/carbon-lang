//===----------- JITSymbol.h - JIT symbol abstraction -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Abstraction for target process addresses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_JITSYMBOL_H
#define LLVM_EXECUTIONENGINE_ORC_JITSYMBOL_H

#include "llvm/Support/Compiler.h"
#include <cassert>
#include <functional>

namespace llvm {

/// @brief Represents an address in the target process's address space.
typedef uint64_t TargetAddress;

/// @brief Represents a symbol in the JIT.
class JITSymbol {
public:
  typedef std::function<TargetAddress()> GetAddressFtor;

  JITSymbol(std::nullptr_t) : CachedAddr(0) {}

  JITSymbol(GetAddressFtor GetAddress)
      : CachedAddr(0), GetAddress(std::move(GetAddress)) {}

  /// @brief Returns true if the symbol exists, false otherwise.
  explicit operator bool() const { return CachedAddr || GetAddress; }

  /// @brief Get the address of the symbol in the target address space. Returns
  ///        '0' if the symbol does not exist.
  TargetAddress getAddress() {
    if (GetAddress) {
      CachedAddr = GetAddress();
      assert(CachedAddr && "Symbol could not be materialized.");
      GetAddress = nullptr;
    }
    return CachedAddr;
  }

private:
  TargetAddress CachedAddr;
  GetAddressFtor GetAddress;
};

}

#endif // LLVM_EXECUTIONENGINE_ORC_JITSYMBOL_H
