//===- PDBSymbolExe.cpp - ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#if defined(_WIN32)
#include <windows.h>
#endif

using namespace llvm;

namespace {
std::string GuidToString(PDB_UniqueId *Id) {
#if defined(_WIN32)
  GUID *Guid = reinterpret_cast<GUID *>(Id);
  OLECHAR GuidBuf[40];
  int Result = StringFromGUID2(*Guid, GuidBuf, 39);
  const char *InputBytes = reinterpret_cast<const char *>(GuidBuf);
  std::string ResultString;
  convertUTF16ToUTF8String(ArrayRef<char>(InputBytes, Result * 2),
                           ResultString);
  return ResultString;
#else
  return std::string();
#endif
}
}

PDBSymbolExe::PDBSymbolExe(std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(std::move(Symbol)) {}

void PDBSymbolExe::dump(llvm::raw_ostream &OS) const {
}
