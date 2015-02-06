//===- IPDBSession.h - base interface for a PDB symbol context --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_IPDBSESSION_H
#define LLVM_DEBUGINFO_PDB_IPDBSESSION_H

#include <memory>

#include "PDBTypes.h"

namespace llvm {

class PDBSymbolExe;

/// IPDBSession defines an interface used to provide a context for querying
/// debug information from a debug data source (for example, a PDB).
class IPDBSession {
public:
  virtual ~IPDBSession();

  virtual uint64_t getLoadAddress() const = 0;
  virtual void setLoadAddress(uint64_t Address) = 0;
  virtual std::unique_ptr<PDBSymbolExe> getGlobalScope() const = 0;
  virtual std::unique_ptr<PDBSymbol> getSymbolById() const = 0;
  virtual std::unique_ptr<IPDBSourceFile> getSourceFileById() const = 0;

  virtual std::unique_ptr<IPDBEnumDataStreams> getDebugStreams() const = 0;
};
}

#endif
