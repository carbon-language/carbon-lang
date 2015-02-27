//===- TypedefDumper.cpp - PDBSymDumper impl for typedefs -------- * C++ *-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TypedefDumper.h"

#include "BuiltinDumper.h"
#include "FunctionDumper.h"
#include "LinePrinter.h"
#include "llvm-pdbdump.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

using namespace llvm;

TypedefDumper::TypedefDumper(LinePrinter &P) : PDBSymDumper(true), Printer(P) {}

void TypedefDumper::start(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS,
                          int Indent) {
  WithColor(Printer, PDB_ColorItem::Keyword).get() << "typedef ";
  uint32_t TargetId = Symbol.getTypeId();
  if (auto TypeSymbol = Symbol.getSession().getSymbolById(TargetId))
    TypeSymbol->dump(OS, 0, *this);
  WithColor(Printer, PDB_ColorItem::Type).get() << " " << Symbol.getName();
}

void TypedefDumper::dump(const PDBSymbolTypeArray &Symbol, raw_ostream &OS,
                         int Indent) {}

void TypedefDumper::dump(const PDBSymbolTypeBuiltin &Symbol, raw_ostream &OS,
                         int Indent) {
  BuiltinDumper Dumper(Printer);
  Dumper.start(Symbol, OS);
}

void TypedefDumper::dump(const PDBSymbolTypeEnum &Symbol, raw_ostream &OS,
                         int Indent) {
  WithColor(Printer, PDB_ColorItem::Keyword).get() << "enum ";
  WithColor(Printer, PDB_ColorItem::Type).get() << " " << Symbol.getName();
}

void TypedefDumper::dump(const PDBSymbolTypePointer &Symbol, raw_ostream &OS,
                         int Indent) {
  if (Symbol.isConstType())
    WithColor(Printer, PDB_ColorItem::Keyword).get() << "const ";
  if (Symbol.isVolatileType())
    WithColor(Printer, PDB_ColorItem::Keyword).get() << "volatile ";
  uint32_t PointeeId = Symbol.getTypeId();
  auto PointeeType = Symbol.getSession().getSymbolById(PointeeId);
  if (!PointeeType)
    return;
  if (auto FuncSig = dyn_cast<PDBSymbolTypeFunctionSig>(PointeeType.get())) {
    FunctionDumper::PointerType Pointer = FunctionDumper::PointerType::Pointer;
    if (Symbol.isReference())
      Pointer = FunctionDumper::PointerType::Reference;
    FunctionDumper NestedDumper(Printer);
    NestedDumper.start(*FuncSig, nullptr, Pointer, OS);
  } else {
    PointeeType->dump(OS, Indent, *this);
    OS << ((Symbol.isReference()) ? "&" : "*");
  }
}

void TypedefDumper::dump(const PDBSymbolTypeFunctionSig &Symbol,
                         raw_ostream &OS, int Indent) {
  FunctionDumper Dumper(Printer);
  Dumper.start(Symbol, nullptr, FunctionDumper::PointerType::None, OS);
}

void TypedefDumper::dump(const PDBSymbolTypeUDT &Symbol, raw_ostream &OS,
                         int Indent) {
  WithColor(Printer, PDB_ColorItem::Keyword).get() << "class ";
  WithColor(Printer, PDB_ColorItem::Type).get() << " " << Symbol.getName();
}
