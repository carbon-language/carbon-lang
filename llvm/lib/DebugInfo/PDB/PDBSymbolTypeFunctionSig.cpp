//===- PDBSymbolTypeFunctionSig.cpp - --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/DebugInfo/PDB/ConcreteSymbolEnumerator.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionArg.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"

using namespace llvm;

PDBSymbolTypeFunctionSig::PDBSymbolTypeFunctionSig(
    const IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {}

void PDBSymbolTypeFunctionSig::dump(raw_ostream &OS, int Indent,
                                    PDB_DumpLevel Level) const {
  OS << stream_indent(Indent);

  uint32_t ReturnTypeId = getTypeId();
  if (auto ReturnType = Session.getSymbolById(ReturnTypeId)) {
    ReturnType->dump(OS, 0, PDB_DumpLevel::Compact);
    OS << " ";
  }
  // TODO: We need a way to detect if this is a pointer to function so that we
  // can print the * between the return type and the argument list.  The only
  // way to do this is to pass the parent into this function, but that will
  // require a larger interface change.
  OS << getCallingConvention() << " ";
  uint32_t ClassId = getClassParentId();
  if (ClassId != 0) {
    if (auto ClassParent = Session.getSymbolById(ClassId)) {
      OS << "(";
      ClassParent->dump(OS, 0, PDB_DumpLevel::Compact);
      OS << "::*)";
    }
  }
  OS.flush();
  OS << "(";
  if (auto ChildEnum = findAllChildren<PDBSymbolTypeFunctionArg>()) {
    uint32_t Index = 0;
    while (auto Arg = ChildEnum->getNext()) {
      Arg->dump(OS, 0, PDB_DumpLevel::Compact);
      if (++Index < ChildEnum->getChildCount())
        OS << ", ";
    }
  }
  OS << ")";
}
