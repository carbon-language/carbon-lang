//===- BuiltinDumper.cpp ---------------------------------------- *- C++ *-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "BuiltinDumper.h"
#include "llvm-pdbdump.h"

#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"

using namespace llvm;

BuiltinDumper::BuiltinDumper() : PDBSymDumper(false) {}

void BuiltinDumper::start(const PDBSymbolTypeBuiltin &Symbol,
                          llvm::raw_ostream &OS) {
  PDB_BuiltinType Type = Symbol.getBuiltinType();
  switch (Type) {
  case PDB_BuiltinType::Float:
    OS << ((Symbol.getLength() == 4) ? "float" : "double");
    break;
  case PDB_BuiltinType::UInt:
    OS << "unsigned";
    if (Symbol.getLength() == 8)
      OS << " __int64";
    break;
  case PDB_BuiltinType::Int:
    OS << ((Symbol.getLength() == 4) ? "int" : "__int64");
    break;
  case PDB_BuiltinType::Char:
    OS << "char";
    break;
  case PDB_BuiltinType::WCharT:
    OS << "wchar_t";
    break;
  case PDB_BuiltinType::Void:
    OS << "void";
    break;
  case PDB_BuiltinType::Long:
    OS << "long";
    break;
  case PDB_BuiltinType::ULong:
    OS << "unsigned long";
    break;
  case PDB_BuiltinType::Bool:
    OS << "bool";
    break;
  case PDB_BuiltinType::Currency:
    OS << "CURRENCY";
    break;
  case PDB_BuiltinType::Date:
    OS << "DATE";
    break;
  case PDB_BuiltinType::Variant:
    OS << "VARIANT";
    break;
  case PDB_BuiltinType::Complex:
    OS << "complex";
    break;
  case PDB_BuiltinType::Bitfield:
    OS << "bitfield";
    break;
  case PDB_BuiltinType::BSTR:
    OS << "BSTR";
    break;
  case PDB_BuiltinType::HResult:
    OS << "HRESULT";
    break;
  case PDB_BuiltinType::BCD:
    OS << "HRESULT";
    break;
  default:
    OS << "(unknown builtin type)";
    break;
  }
}
