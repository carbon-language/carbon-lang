//===- PDBSymbolFunc.cpp - --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"
#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

#include <utility>

using namespace llvm;
PDBSymbolFunc::PDBSymbolFunc(const IPDBSession &PDBSession,
                             std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

std::unique_ptr<PDBSymbolTypeFunctionSig> PDBSymbolFunc::getSignature() const {
  return Session.getConcreteSymbolById<PDBSymbolTypeFunctionSig>(getTypeId());
}

void PDBSymbolFunc::dump(raw_ostream &OS, int Indent,
                         PDBSymDumper &Dumper) const {
  Dumper.dump(*this, OS, Indent);
}
