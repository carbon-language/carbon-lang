//===- PDBSymbolLabel.cpp - -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolLabel.h"

using namespace llvm;

PDBSymbolLabel::PDBSymbolLabel(const IPDBSession &PDBSession,
                               std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(Session, std::move(Symbol)) {}

void PDBSymbolLabel::dump(llvm::raw_ostream &OS) const {}
