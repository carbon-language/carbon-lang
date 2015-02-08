//===- PDBSymbolData.cpp - PDB data (e.g. variable) accessors ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"

using namespace llvm;

PDBSymbolData::PDBSymbolData(IPDBSession &PDBSession,
                             std::unique_ptr<IPDBRawSymbol> DataSymbol)
    : PDBSymbol(PDBSession, std::move(DataSymbol)) {}

void PDBSymbolData::dump(llvm::raw_ostream &OS) const {}