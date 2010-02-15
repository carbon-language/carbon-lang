//===-- llvm/CodeGen/MachineModuleInfoImpls.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements object-file format specific implementations of
// MachineModuleInfoImpl.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/MC/MCSymbol.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// MachineModuleInfoMachO
//===----------------------------------------------------------------------===//

// Out of line virtual method.
void MachineModuleInfoMachO::Anchor() {}
void MachineModuleInfoELF::Anchor() {}

static int SortSymbolPair(const void *LHS, const void *RHS) {
  const MCSymbol *LHSS =
    ((const std::pair<MCSymbol*, MCSymbol*>*)LHS)->first;
  const MCSymbol *RHSS =
    ((const std::pair<MCSymbol*, MCSymbol*>*)RHS)->first;
  return LHSS->getName().compare(RHSS->getName());
}

/// GetSortedStubs - Return the entries from a DenseMap in a deterministic
/// sorted orer.
MachineModuleInfoImpl::SymbolListTy
MachineModuleInfoImpl::GetSortedStubs(const DenseMap<MCSymbol*,
                                                     MCSymbol*> &Map) {
  MachineModuleInfoImpl::SymbolListTy List(Map.begin(), Map.end());

  if (!List.empty())
    qsort(&List[0], List.size(), sizeof(List[0]), SortSymbolPair);
  return List;
}

