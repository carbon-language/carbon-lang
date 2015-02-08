//===- PDBSymbolFuncDebugStart.cpp - ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugStart.h"

using namespace llvm;

PDBSymbolFuncDebugStart::PDBSymbolFuncDebugStart(
    IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(Session, std::move(Symbol)) {}

void PDBSymbolFuncDebugStart::dump(llvm::raw_ostream &OS) const {}
