//===- PDBSymbolTypePointer.cpp -----------------------------------*- C++ -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"

#include <utility>

using namespace llvm;

PDBSymbolTypePointer::PDBSymbolTypePointer(
    const IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolTypePointer::dump(raw_ostream &OS, int Indent,
                                PDB_DumpLevel Level) const {
  OS << stream_indent(Indent);
  if (isConstType())
    OS << "const ";
  if (isVolatileType())
    OS << "volatile ";
  uint32_t PointeeId = getTypeId();
  if (auto PointeeType = Session.getSymbolById(PointeeId)) {
    // Function pointers get special treatment, since we need to print the * in
    // the middle of the signature.
    if (auto FuncSig = dyn_cast<PDBSymbolTypeFunctionSig>(PointeeType.get())) {
      if (auto ReturnType = FuncSig->getReturnType())
        ReturnType->dump(OS, 0, PDB_DumpLevel::Compact);
      OS << " (" << FuncSig->getCallingConvention() << " ";
      OS << ((isReference()) ? "&" : "*") << ")";
      FuncSig->dumpArgList(OS);
    } else {
      PointeeType->dump(OS, 0, PDB_DumpLevel::Compact);
      OS << ((isReference()) ? "&" : "*");
    }
  }
}
