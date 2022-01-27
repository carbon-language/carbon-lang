//===- EPCGenericDylibManager.h -- Generic EPC Dylib management -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements dylib loading and searching by making calls to
// ExecutorProcessControl::callWrapper.
//
// This simplifies the implementaton of new ExecutorProcessControl instances,
// as this implementation will always work (at the cost of some performance
// overhead for the calls).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCGENERICDYLIBMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_EPCGENERICDYLIBMANAGER_H

#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"

namespace llvm {
namespace orc {

class SymbolLookupSet;

class EPCGenericDylibManager {
public:
  /// Function addresses for memory access.
  struct SymbolAddrs {
    ExecutorAddr Instance;
    ExecutorAddr Open;
    ExecutorAddr Lookup;
  };

  /// Create an EPCGenericMemoryAccess instance from a given set of
  /// function addrs.
  static Expected<EPCGenericDylibManager>
  CreateWithDefaultBootstrapSymbols(ExecutorProcessControl &EPC);

  /// Create an EPCGenericMemoryAccess instance from a given set of
  /// function addrs.
  EPCGenericDylibManager(ExecutorProcessControl &EPC, SymbolAddrs SAs)
      : EPC(EPC), SAs(SAs) {}

  /// Loads the dylib with the given name.
  Expected<tpctypes::DylibHandle> open(StringRef Path, uint64_t Mode);

  /// Looks up symbols within the given dylib.
  Expected<std::vector<ExecutorAddr>> lookup(tpctypes::DylibHandle H,
                                             const SymbolLookupSet &Lookup);

  /// Looks up symbols within the given dylib.
  Expected<std::vector<ExecutorAddr>>
  lookup(tpctypes::DylibHandle H, const RemoteSymbolLookupSet &Lookup);

private:
  ExecutorProcessControl &EPC;
  SymbolAddrs SAs;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICDYLIBMANAGER_H
