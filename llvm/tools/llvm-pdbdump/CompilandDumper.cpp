//===- CompilandDumper.cpp - llvm-pdbdump compiland symbol dumper *- C++ *-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CompilandDumper.h"
#include "llvm-pdbdump.h"

#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugEnd.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugStart.h"
#include "llvm/DebugInfo/PDB/PDBSymbolLabel.h"
#include "llvm/DebugInfo/PDB/PDBSymbolThunk.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"
#include "llvm/DebugInfo/PDB/PDBSymbolUnknown.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "FunctionDumper.h"

#include <utility>
#include <vector>

using namespace llvm;

CompilandDumper::CompilandDumper() : PDBSymDumper(true) {}

void CompilandDumper::dump(const PDBSymbolCompilandDetails &Symbol,
                           raw_ostream &OS, int Indent) {}

void CompilandDumper::dump(const PDBSymbolCompilandEnv &Symbol, raw_ostream &OS,
                           int Indent) {}

void CompilandDumper::start(const PDBSymbolCompiland &Symbol, raw_ostream &OS,
                            int Indent, bool Children) {
  std::string FullName = Symbol.getName();
  OS << newline(Indent) << FullName;
  if (!Children)
    return;

  auto ChildrenEnum = Symbol.findAllChildren();
  while (auto Child = ChildrenEnum->getNext())
    Child->dump(OS, Indent + 2, *this);
}

void CompilandDumper::dump(const PDBSymbolData &Symbol, raw_ostream &OS,
                           int Indent) {
  OS << newline(Indent);
  switch (auto LocType = Symbol.getLocationType()) {
  case PDB_LocType::Static:
    OS << "data: [";
    OS << format_hex(Symbol.getRelativeVirtualAddress(), 10);
    OS << "]";
    break;
  case PDB_LocType::Constant:
    OS << "constant: [" << Symbol.getValue() << "]";
    break;
  default:
    OS << "data(unexpected type=" << LocType << ")";
  }

  OS << " " << Symbol.getName();
}

void CompilandDumper::dump(const PDBSymbolFunc &Symbol, raw_ostream &OS,
                           int Indent) {
  uint32_t FuncStart = Symbol.getRelativeVirtualAddress();
  uint32_t FuncEnd = FuncStart + Symbol.getLength();
  OS << newline(Indent) << "func [" << format_hex(FuncStart, 8);
  if (auto DebugStart = Symbol.findOneChild<PDBSymbolFuncDebugStart>())
    OS << "+" << DebugStart->getRelativeVirtualAddress() - FuncStart;
  OS << " - " << format_hex(FuncEnd, 8);
  if (auto DebugEnd = Symbol.findOneChild<PDBSymbolFuncDebugEnd>())
    OS << "-" << FuncEnd - DebugEnd->getRelativeVirtualAddress();
  OS << "] ";

  if (Symbol.hasFramePointer())
    OS << "(" << Symbol.getLocalBasePointerRegisterId() << ")";
  else
    OS << "(FPO)";

  OS << " ";

  FunctionDumper Dumper;
  Dumper.start(Symbol, OS);
  OS.flush();
}

void CompilandDumper::dump(const PDBSymbolLabel &Symbol, raw_ostream &OS,
                           int Indent) {
  OS << newline(Indent);
  OS << "label [" << format_hex(Symbol.getRelativeVirtualAddress(), 10) << "] "
     << Symbol.getName();
}

void CompilandDumper::dump(const PDBSymbolThunk &Symbol, raw_ostream &OS,
                           int Indent) {
  OS << newline(Indent) << "thunk ";
  PDB_ThunkOrdinal Ordinal = Symbol.getThunkOrdinal();
  uint32_t RVA = Symbol.getRelativeVirtualAddress();
  if (Ordinal == PDB_ThunkOrdinal::TrampIncremental) {
    OS << format_hex(RVA, 10);
    OS << " -> " << format_hex(Symbol.getTargetRelativeVirtualAddress(), 10);
  } else {
    OS << "[" << format_hex(RVA, 10);
    OS << " - " << format_hex(RVA + Symbol.getLength(), 10) << "]";
  }
  OS << " (" << Ordinal << ") ";
  std::string Name = Symbol.getName();
  if (!Name.empty())
    OS << Name;
}

void CompilandDumper::dump(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS,
                           int Indent) {}

void CompilandDumper::dump(const PDBSymbolUnknown &Symbol, raw_ostream &OS,
                           int Indent) {
  OS << newline(Indent);
  OS << "unknown (" << Symbol.getSymTag() << ")";
}
