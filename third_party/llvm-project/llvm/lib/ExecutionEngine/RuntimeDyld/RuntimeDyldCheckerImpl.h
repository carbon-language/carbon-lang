//===-- RuntimeDyldCheckerImpl.h -- RuntimeDyld test framework --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_RUNTIMEDYLD_RUNTIMEDYLDCHECKERIMPL_H
#define LLVM_LIB_EXECUTIONENGINE_RUNTIMEDYLD_RUNTIMEDYLDCHECKERIMPL_H

#include "RuntimeDyldImpl.h"

namespace llvm {

class RuntimeDyldCheckerImpl {
  friend class RuntimeDyldChecker;
  friend class RuntimeDyldCheckerExprEval;

  using IsSymbolValidFunction =
    RuntimeDyldChecker::IsSymbolValidFunction;
  using GetSymbolInfoFunction = RuntimeDyldChecker::GetSymbolInfoFunction;
  using GetSectionInfoFunction = RuntimeDyldChecker::GetSectionInfoFunction;
  using GetStubInfoFunction = RuntimeDyldChecker::GetStubInfoFunction;
  using GetGOTInfoFunction = RuntimeDyldChecker::GetGOTInfoFunction;

public:
  RuntimeDyldCheckerImpl(
      IsSymbolValidFunction IsSymbolValid, GetSymbolInfoFunction GetSymbolInfo,
      GetSectionInfoFunction GetSectionInfo, GetStubInfoFunction GetStubInfo,
      GetGOTInfoFunction GetGOTInfo, support::endianness Endianness,
      MCDisassembler *Disassembler, MCInstPrinter *InstPrinter,
      llvm::raw_ostream &ErrStream);

  bool check(StringRef CheckExpr) const;
  bool checkAllRulesInBuffer(StringRef RulePrefix, MemoryBuffer *MemBuf) const;

private:

  // StubMap typedefs.

  Expected<JITSymbolResolver::LookupResult>
  lookup(const JITSymbolResolver::LookupSet &Symbols) const;

  bool isSymbolValid(StringRef Symbol) const;
  uint64_t getSymbolLocalAddr(StringRef Symbol) const;
  uint64_t getSymbolRemoteAddr(StringRef Symbol) const;
  uint64_t readMemoryAtAddr(uint64_t Addr, unsigned Size) const;

  StringRef getSymbolContent(StringRef Symbol) const;

  std::pair<uint64_t, std::string> getSectionAddr(StringRef FileName,
                                                  StringRef SectionName,
                                                  bool IsInsideLoad) const;

  std::pair<uint64_t, std::string>
  getStubOrGOTAddrFor(StringRef StubContainerName, StringRef Symbol,
                      bool IsInsideLoad, bool IsStubAddr) const;

  Optional<uint64_t> getSectionLoadAddress(void *LocalAddr) const;

  IsSymbolValidFunction IsSymbolValid;
  GetSymbolInfoFunction GetSymbolInfo;
  GetSectionInfoFunction GetSectionInfo;
  GetStubInfoFunction GetStubInfo;
  GetGOTInfoFunction GetGOTInfo;
  support::endianness Endianness;
  MCDisassembler *Disassembler;
  MCInstPrinter *InstPrinter;
  llvm::raw_ostream &ErrStream;
};
}

#endif
