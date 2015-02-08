//===- PDBSymbolTypeCustom.cpp - --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeCustom.h"

using namespace llvm;

PDBSymbolTypeCustom::PDBSymbolTypeCustom(std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(std::move(Symbol)) {}

void PDBSymbolTypeCustom::dump(llvm::raw_ostream &OS) const {}
