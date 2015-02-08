//===- PDBSymbolThunk.cpp - -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolThunk.h"

using namespace llvm;

PDBSymbolThunk::PDBSymbolThunk(std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(std::move(Symbol)) {}

void PDBSymbolThunk::dump(llvm::raw_ostream &OS) const {}
