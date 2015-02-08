//===- PDBSymbolPublicSymbol.cpp - ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolPublicSymbol.h"

using namespace llvm;

PDBSymbolPublicSymbol::PDBSymbolPublicSymbol(
    std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(std::move(Symbol)) {}

void PDBSymbolPublicSymbol::dump(llvm::raw_ostream &OS) const {}
