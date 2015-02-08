//===- PDBSymbolFunc.cpp - --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"

using namespace llvm;

PDBSymbolFunc::PDBSymbolFunc(IPDBSession &PDBSession,
                             std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolFunc::dump(llvm::raw_ostream &OS) const {}
