//===- TypeDumper.cpp - PDBSymDumper implementation for types *----- C++ *-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TypeDumper.h"

#include "FunctionDumper.h"
#include "llvm-pdbdump.h"
#include "TypedefDumper.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"

using namespace llvm;

TypeDumper::TypeDumper() : PDBSymDumper(true) {}

void TypeDumper::start(const PDBSymbolExe &Exe, raw_ostream &OS, int Indent) {
  auto Enums = Exe.findAllChildren<PDBSymbolTypeEnum>();
  OS << newline(Indent) << "Enums: (" << Enums->getChildCount() << " items)";
  while (auto Enum = Enums->getNext())
    Enum->dump(OS, Indent + 2, *this);

  auto FuncSigs = Exe.findAllChildren<PDBSymbolTypeFunctionSig>();
  OS << newline(Indent);
  OS << "Function Signatures: (" << FuncSigs->getChildCount() << " items)";
  while (auto Sig = FuncSigs->getNext())
    Sig->dump(OS, Indent + 2, *this);

  auto Typedefs = Exe.findAllChildren<PDBSymbolTypeTypedef>();
  OS << newline(Indent) << "Typedefs: (" << Typedefs->getChildCount()
     << " items)";
  while (auto Typedef = Typedefs->getNext())
    Typedef->dump(OS, Indent + 2, *this);
}

void TypeDumper::dump(const PDBSymbolTypeEnum &Symbol, raw_ostream &OS,
                      int Indent) {
  OS << newline(Indent) << "enum " << Symbol.getName();
}

void TypeDumper::dump(const PDBSymbolTypeFunctionSig &Symbol, raw_ostream &OS,
                      int Indent) {
  OS << newline(Indent);
  FunctionDumper Dumper;
  Dumper.start(Symbol, FunctionDumper::PointerType::None, OS);
}

void TypeDumper::dump(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS,
                      int Indent) {
  OS << newline(Indent);
  TypedefDumper Dumper;
  Dumper.start(Symbol, OS, Indent);
  OS.flush();
}
