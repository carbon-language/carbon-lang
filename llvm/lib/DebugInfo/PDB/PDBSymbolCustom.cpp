//===- PDBSymbolCustom.cpp - compiler-specific types ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/IPDBRawSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCustom.h"

using namespace llvm;

void PDBSymbolCustom::getDataBytes(llvm::SmallVector<uint8_t, 32> &bytes) {
  RawSymbol->getDataBytes(bytes);
}
