//===-- TextStubHelpers.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TextAPI/InterfaceFile.h"
#include <algorithm>
#include <string>

#ifndef TEXT_STUB_HELPERS_H
#define TEXT_STUB_HELPERS_H

namespace llvm {
struct ExportedSymbol {
  llvm::MachO::SymbolKind Kind;
  std::string Name;
  bool WeakDefined;
  bool ThreadLocalValue;
};

using ExportedSymbolSeq = std::vector<ExportedSymbol>;
using UUIDs = std::vector<std::pair<llvm::MachO::Target, std::string>>;
using TBDFile = std::unique_ptr<MachO::InterfaceFile>;
using TBDReexportFile = std::shared_ptr<MachO::InterfaceFile>;

inline bool operator<(const ExportedSymbol &LHS, const ExportedSymbol &RHS) {
  return std::tie(LHS.Kind, LHS.Name) < std::tie(RHS.Kind, RHS.Name);
}

inline bool operator==(const ExportedSymbol &LHS, const ExportedSymbol &RHS) {
  return std::tie(LHS.Kind, LHS.Name, LHS.WeakDefined, LHS.ThreadLocalValue) ==
         std::tie(RHS.Kind, RHS.Name, RHS.WeakDefined, RHS.ThreadLocalValue);
}

inline std::string stripWhitespace(std::string S) {
  S.erase(std::remove_if(S.begin(), S.end(), ::isspace), S.end());
  return S;
}

// This will transform a single InterfaceFile then compare against the other
// InterfaceFile then transform the second InterfaceFile in the same way to
// regain equality.
inline bool
checkEqualityOnTransform(MachO::InterfaceFile &FileA,
                         MachO::InterfaceFile &FileB,
                         void (*Transform)(MachO::InterfaceFile *)) {
  Transform(&FileA);
  // Files should not be equal.
  if (FileA == FileB)
    return false;
  Transform(&FileB);
  // Files should be equal.
  if (FileA != FileB)
    return false;
  return true;
}

} // namespace llvm
#endif
