//===- VariableDumper.cpp - -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "VariableDumper.h"

#include "BuiltinDumper.h"
#include "LinePrinter.h"
#include "llvm-pdbdump.h"
#include "FunctionDumper.h"

#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeArray.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

#include "llvm/Support/Format.h"

using namespace llvm;

VariableDumper::VariableDumper(LinePrinter &P)
    : PDBSymDumper(true), Printer(P) {}

void VariableDumper::start(const PDBSymbolData &Var, raw_ostream &OS,
                           int Indent) {
  Printer.NewLine();
  Printer << "data ";

  auto VarType = Var.getType();

  switch (auto LocType = Var.getLocationType()) {
  case PDB_LocType::Static:
    WithColor(Printer, PDB_ColorItem::Address).get()
        << "[" << format_hex(Var.getRelativeVirtualAddress(), 10) << "] ";
    WithColor(Printer, PDB_ColorItem::Keyword).get() << "static ";
    dumpSymbolTypeAndName(*VarType, Var.getName(), OS);
    break;
  case PDB_LocType::Constant:
    WithColor(Printer, PDB_ColorItem::Keyword).get() << "const ";
    dumpSymbolTypeAndName(*VarType, Var.getName(), OS);
    Printer << "[";
    WithColor(Printer, PDB_ColorItem::LiteralValue).get() << Var.getValue();
    Printer << "]";
    break;
  case PDB_LocType::ThisRel:
    WithColor(Printer, PDB_ColorItem::Offset).get()
        << "+" << format_hex(Var.getOffset(), 4) << " ";
    dumpSymbolTypeAndName(*VarType, Var.getName(), OS);
    break;
  default:
    OS << "unknown(" << LocType << ") ";
    WithColor(Printer, PDB_ColorItem::Identifier).get() << Var.getName();
    break;
  }
}

void VariableDumper::dump(const PDBSymbolTypeBuiltin &Symbol, raw_ostream &OS,
                          int Indent) {
  BuiltinDumper Dumper(Printer);
  Dumper.start(Symbol, OS);
}

void VariableDumper::dump(const PDBSymbolTypeEnum &Symbol, raw_ostream &OS,
                          int Indent) {
  WithColor(Printer, PDB_ColorItem::Type).get() << Symbol.getName();
}

void VariableDumper::dump(const PDBSymbolTypeFunctionSig &Symbol,
                          raw_ostream &OS, int Indent) {}

void VariableDumper::dump(const PDBSymbolTypePointer &Symbol, raw_ostream &OS,
                          int Indent) {
  auto PointeeType = Symbol.getPointeeType();
  if (!PointeeType)
    return;

  if (auto Func = dyn_cast<PDBSymbolFunc>(PointeeType.get())) {
    FunctionDumper NestedDumper(Printer);
    FunctionDumper::PointerType Pointer =
        Symbol.isReference() ? FunctionDumper::PointerType::Reference
                             : FunctionDumper::PointerType::Pointer;
    NestedDumper.start(*Func, Pointer, OS, Indent);
  } else {
    if (Symbol.isConstType())
      WithColor(Printer, PDB_ColorItem::Keyword).get() << "const ";
    if (Symbol.isVolatileType())
      WithColor(Printer, PDB_ColorItem::Keyword).get() << "volatile ";
    PointeeType->dump(OS, Indent, *this);
    Printer << (Symbol.isReference() ? "&" : "*");
  }
}

void VariableDumper::dump(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS,
                          int Indent) {
  WithColor(Printer, PDB_ColorItem::Keyword).get() << "typedef ";
  WithColor(Printer, PDB_ColorItem::Type).get() << Symbol.getName();
}

void VariableDumper::dump(const PDBSymbolTypeUDT &Symbol, raw_ostream &OS,
                          int Indent) {
  WithColor(Printer, PDB_ColorItem::Type).get() << Symbol.getName();
}

void VariableDumper::dumpSymbolTypeAndName(const PDBSymbol &Type,
                                           StringRef Name, raw_ostream &OS) {
  if (auto *ArrayType = dyn_cast<PDBSymbolTypeArray>(&Type)) {
    std::string IndexSpec;
    raw_string_ostream IndexStream(IndexSpec);
    std::unique_ptr<PDBSymbol> ElementType = ArrayType->getElementType();
    while (auto NestedArray = dyn_cast<PDBSymbolTypeArray>(ElementType.get())) {
      IndexStream << "[";
      IndexStream << NestedArray->getCount();
      IndexStream << "]";
      ElementType = NestedArray->getElementType();
    }
    IndexStream << "[" << ArrayType->getCount() << "]";
    ElementType->dump(OS, 0, *this);
    WithColor(Printer, PDB_ColorItem::Identifier).get() << " " << Name;
    Printer << IndexStream.str();
  } else {
    if (!tryDumpFunctionPointer(Type, Name, OS)) {
      Type.dump(OS, 0, *this);
      WithColor(Printer, PDB_ColorItem::Identifier).get() << " " << Name;
    }
  }
}

bool VariableDumper::tryDumpFunctionPointer(const PDBSymbol &Type,
                                            StringRef Name, raw_ostream &OS) {
  // Function pointers come across as pointers to function signatures.  But the
  // signature carries no name, so we have to handle this case separately.
  if (auto *PointerType = dyn_cast<PDBSymbolTypePointer>(&Type)) {
    auto PointeeType = PointerType->getPointeeType();
    if (auto *FunctionSig =
            dyn_cast<PDBSymbolTypeFunctionSig>(PointeeType.get())) {
      FunctionDumper Dumper(Printer);
      FunctionDumper::PointerType PT = FunctionDumper::PointerType::Pointer;
      if (PointerType->isReference())
        PT = FunctionDumper::PointerType::Reference;
      std::string NameStr(Name.begin(), Name.end());
      Dumper.start(*FunctionSig, NameStr.c_str(), PT, OS);
      return true;
    }
  }
  return false;
}
