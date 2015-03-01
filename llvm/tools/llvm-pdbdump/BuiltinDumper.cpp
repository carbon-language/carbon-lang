//===- BuiltinDumper.cpp ---------------------------------------- *- C++ *-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "BuiltinDumper.h"
#include "LinePrinter.h"
#include "llvm-pdbdump.h"

#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"

using namespace llvm;

BuiltinDumper::BuiltinDumper(LinePrinter &P)
    : PDBSymDumper(false), Printer(P) {}

void BuiltinDumper::start(const PDBSymbolTypeBuiltin &Symbol) {
  PDB_BuiltinType Type = Symbol.getBuiltinType();
  switch (Type) {
  case PDB_BuiltinType::Float:
    if (Symbol.getLength() == 4)
      WithColor(Printer, PDB_ColorItem::Type).get() << "float";
    else
      WithColor(Printer, PDB_ColorItem::Type).get() << "double";
    break;
  case PDB_BuiltinType::UInt:
    WithColor(Printer, PDB_ColorItem::Type).get() << "unsigned";
    if (Symbol.getLength() == 8)
      WithColor(Printer, PDB_ColorItem::Type).get() << " __int64";
    break;
  case PDB_BuiltinType::Int:
    if (Symbol.getLength() == 4)
      WithColor(Printer, PDB_ColorItem::Type).get() << "int";
    else
      WithColor(Printer, PDB_ColorItem::Type).get() << "__int64";
    break;
  case PDB_BuiltinType::Char:
    WithColor(Printer, PDB_ColorItem::Type).get() << "char";
    break;
  case PDB_BuiltinType::WCharT:
    WithColor(Printer, PDB_ColorItem::Type).get() << "wchar_t";
    break;
  case PDB_BuiltinType::Void:
    WithColor(Printer, PDB_ColorItem::Type).get() << "void";
    break;
  case PDB_BuiltinType::Long:
    WithColor(Printer, PDB_ColorItem::Type).get() << "long";
    break;
  case PDB_BuiltinType::ULong:
    WithColor(Printer, PDB_ColorItem::Type).get() << "unsigned long";
    break;
  case PDB_BuiltinType::Bool:
    WithColor(Printer, PDB_ColorItem::Type).get() << "bool";
    break;
  case PDB_BuiltinType::Currency:
    WithColor(Printer, PDB_ColorItem::Type).get() << "CURRENCY";
    break;
  case PDB_BuiltinType::Date:
    WithColor(Printer, PDB_ColorItem::Type).get() << "DATE";
    break;
  case PDB_BuiltinType::Variant:
    WithColor(Printer, PDB_ColorItem::Type).get() << "VARIANT";
    break;
  case PDB_BuiltinType::Complex:
    WithColor(Printer, PDB_ColorItem::Type).get() << "complex";
    break;
  case PDB_BuiltinType::Bitfield:
    WithColor(Printer, PDB_ColorItem::Type).get() << "bitfield";
    break;
  case PDB_BuiltinType::BSTR:
    WithColor(Printer, PDB_ColorItem::Type).get() << "BSTR";
    break;
  case PDB_BuiltinType::HResult:
    WithColor(Printer, PDB_ColorItem::Type).get() << "HRESULT";
    break;
  case PDB_BuiltinType::BCD:
    WithColor(Printer, PDB_ColorItem::Type).get() << "HRESULT";
    break;
  default:
    WithColor(Printer, PDB_ColorItem::Type).get() << "(unknown)";
    break;
  }
}
