//===-- RuntimeDyldCheckerImpl.h -- RuntimeDyld test framework --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIMEDYLDCHECKERIMPL_H
#define LLVM_RUNTIMEDYLDCHECKERIMPL_H

#include "RuntimeDyldImpl.h"
#include <set>

namespace llvm {

class RuntimeDyldCheckerImpl {
  friend class RuntimeDyldImpl;
  friend class RuntimeDyldCheckerExprEval;

public:
  RuntimeDyldCheckerImpl(RuntimeDyld &RTDyld, MCDisassembler *Disassembler,
                         MCInstPrinter *InstPrinter,
                         llvm::raw_ostream &ErrStream);

  bool check(StringRef CheckExpr) const;
  bool checkAllRulesInBuffer(StringRef RulePrefix, MemoryBuffer *MemBuf) const;

private:
  RuntimeDyldImpl &getRTDyld() const { return *RTDyld.Dyld; }

  bool isSymbolValid(StringRef Symbol) const;
  uint64_t getSymbolLinkerAddr(StringRef Symbol) const;
  uint64_t getSymbolRemoteAddr(StringRef Symbol) const;
  uint64_t readMemoryAtAddr(uint64_t Addr, unsigned Size) const;
  std::pair<uint64_t, std::string> getStubAddrFor(StringRef FilePath,
                                                  StringRef SectionName,
                                                  StringRef Symbol,
                                                  bool IsInsideLoad) const;
  StringRef getSubsectionStartingAt(StringRef Name) const;

  void registerStubMap(StringRef FileName, unsigned SectionID,
                       const RuntimeDyldImpl::StubMap &RTDyldStubs);

  RuntimeDyld &RTDyld;
  MCDisassembler *Disassembler;
  MCInstPrinter *InstPrinter;
  llvm::raw_ostream &ErrStream;

  // StubMap typedefs.
  typedef std::pair<unsigned, uint64_t> StubLoc;
  typedef std::map<std::string, StubLoc> SymbolStubMap;
  typedef std::map<std::string, SymbolStubMap> SectionStubMap;
  typedef std::map<std::string, SectionStubMap> StubMap;
  StubMap Stubs;
};
}

#endif // LLVM_RUNTIMEDYLDCHECKERIMPL_H
