//===- TypedefDumper.cpp - PDBSymDumper impl for typedefs -------- * C++ *-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TypedefDumper.h"

#include "FunctionDumper.h"
#include "llvm-pdbdump.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

using namespace llvm;

TypedefDumper::TypedefDumper() : PDBSymDumper(true) {}

void TypedefDumper::start(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS,
                          int Indent) {
  OS << "typedef ";
  uint32_t TargetId = Symbol.getTypeId();
  if (auto TypeSymbol = Symbol.getSession().getSymbolById(TargetId))
    TypeSymbol->dump(OS, 0, *this);
  OS << " " << Symbol.getName();
}

void TypedefDumper::dump(const PDBSymbolTypeArray &Symbol, raw_ostream &OS,
                         int Indent) {}

void TypedefDumper::dump(const PDBSymbolTypeBuiltin &Symbol, raw_ostream &OS,
                         int Indent) {
  PDB_BuiltinType Type = Symbol.getBuiltinType();
  OS << Type;
  if (Type == PDB_BuiltinType::UInt || Type == PDB_BuiltinType::Int)
    OS << (8 * Symbol.getLength()) << "_t";
}

void TypedefDumper::dump(const PDBSymbolTypeEnum &Symbol, raw_ostream &OS,
                         int Indent) {
  OS << "enum " << Symbol.getName();
}

void TypedefDumper::dump(const PDBSymbolTypePointer &Symbol, raw_ostream &OS,
                         int Indent) {
  if (Symbol.isConstType())
    OS << "const ";
  if (Symbol.isVolatileType())
    OS << "volatile ";
  uint32_t PointeeId = Symbol.getTypeId();
  auto PointeeType = Symbol.getSession().getSymbolById(PointeeId);
  if (!PointeeType)
    return;
  if (auto FuncSig = dyn_cast<PDBSymbolTypeFunctionSig>(PointeeType.get())) {
    FunctionDumper::PointerType Pointer = FunctionDumper::PointerType::Pointer;
    if (Symbol.isReference())
      Pointer = FunctionDumper::PointerType::Reference;
    FunctionDumper NestedDumper;
    NestedDumper.start(*FuncSig, Pointer, OS);
    OS.flush();
  } else {
    PointeeType->dump(OS, Indent, *this);
    OS << ((Symbol.isReference()) ? "&" : "*");
  }
}

void TypedefDumper::dump(const PDBSymbolTypeFunctionSig &Symbol,
                         raw_ostream &OS, int Indent) {
  FunctionDumper Dumper;
  Dumper.start(Symbol, FunctionDumper::PointerType::None, OS);
}

void TypedefDumper::dump(const PDBSymbolTypeUDT &Symbol, raw_ostream &OS,
                         int Indent) {
  OS << "class " << Symbol.getName();
}
