//===- VariableDumper.cpp - -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "VariableDumper.h"

#include "llvm-pdbdump.h"
#include "FunctionDumper.h"

#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeArray.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

#include "llvm/Support/Format.h"

using namespace llvm;

VariableDumper::VariableDumper() : PDBSymDumper(true) {}

void VariableDumper::start(const PDBSymbolData &Var, raw_ostream &OS,
                           int Indent) {
  OS << newline(Indent);
  OS << "data ";

  auto VarType = Var.getType();

  switch (auto LocType = Var.getLocationType()) {
  case PDB_LocType::Static:
    OS << "[" << format_hex(Var.getRelativeVirtualAddress(), 10) << "] ";
    OS << "static ";
    dumpSymbolTypeAndName(*VarType, Var.getName(), OS);
    break;
  case PDB_LocType::Constant:
    OS << "const ";
    dumpSymbolTypeAndName(*VarType, Var.getName(), OS);
    OS << "[" << Var.getValue() << "]";
    break;
  case PDB_LocType::ThisRel: {
    int Offset = Var.getOffset();
    OS << "+" << format_hex(Var.getOffset(), 4) << " ";
    OS.flush();
    dumpSymbolTypeAndName(*VarType, Var.getName(), OS);
    break;
  }
  default:
    break;
    OS << "unknown(" << LocType << ") " << Var.getName();
  }
}

void VariableDumper::dump(const PDBSymbolTypeBuiltin &Symbol, raw_ostream &OS,
                          int Indent) {
  OS << Symbol.getBuiltinType();
}

void VariableDumper::dump(const PDBSymbolTypeEnum &Symbol, raw_ostream &OS,
                          int Indent) {
  OS << Symbol.getName();
}

void VariableDumper::dump(const PDBSymbolTypeFunctionSig &Symbol,
                          raw_ostream &OS, int Indent) {}

void VariableDumper::dump(const PDBSymbolTypePointer &Symbol, raw_ostream &OS,
                          int Indent) {
  uint32_t PointeeId = Symbol.getTypeId();
  auto PointeeType = Symbol.getPointeeType();
  if (!PointeeType)
    return;

  if (auto Func = dyn_cast<PDBSymbolFunc>(PointeeType.get())) {
    FunctionDumper NestedDumper;
    FunctionDumper::PointerType Pointer =
        Symbol.isReference() ? FunctionDumper::PointerType::Reference
                             : FunctionDumper::PointerType::Pointer;
    NestedDumper.start(*Func, Pointer, OS, Indent);
  } else {
    if (Symbol.isConstType())
      OS << "const ";
    if (Symbol.isVolatileType())
      OS << "volatile ";
    PointeeType->dump(OS, Indent, *this);
    OS << (Symbol.isReference() ? "&" : "*");
  }
}

void VariableDumper::dump(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS,
                          int Indent) {
  OS << "typedef " << Symbol.getName();
}

void VariableDumper::dump(const PDBSymbolTypeUDT &Symbol, raw_ostream &OS,
                          int Indent) {
  OS << Symbol.getName();
}

void VariableDumper::dumpSymbolTypeAndName(const PDBSymbol &Type,
                                           StringRef Name, raw_ostream &OS) {
  if (auto *ArrayType = dyn_cast<PDBSymbolTypeArray>(&Type)) {
    bool Done = false;
    std::string IndexSpec;
    raw_string_ostream IndexStream(IndexSpec);
    std::unique_ptr<PDBSymbol> ElementType = ArrayType->getElementType();
    while (auto NestedArray = dyn_cast<PDBSymbolTypeArray>(ElementType.get())) {
      IndexStream << "[" << NestedArray->getCount() << "]";
      ElementType = NestedArray->getElementType();
    }
    IndexStream << "[" << ArrayType->getCount() << "]";
    ElementType->dump(OS, 0, *this);
    OS << " " << Name << IndexStream.str();
  } else {
    Type.dump(OS, 0, *this);
    OS << " " << Name;
  }
}
