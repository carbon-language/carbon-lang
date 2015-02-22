//===- PDBSymDumper.cpp - ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymDumper.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define PDB_SYMDUMP_UNREACHABLE(Type)                                          \
  if (RequireImpl)                                                             \
    llvm_unreachable("Attempt to dump " #Type " with no dump implementation");

PDBSymDumper::PDBSymDumper(bool ShouldRequireImpl)
    : RequireImpl(ShouldRequireImpl) {}

PDBSymDumper::~PDBSymDumper() {}

void PDBSymDumper::dump(const PDBSymbolAnnotation &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolAnnotation)
}

void PDBSymDumper::dump(const PDBSymbolBlock &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolBlock)
}

void PDBSymDumper::dump(const PDBSymbolCompiland &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolCompiland)
}

void PDBSymDumper::dump(const PDBSymbolCompilandDetails &Symbol,
                        raw_ostream &OS, int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolCompilandDetails)
}

void PDBSymDumper::dump(const PDBSymbolCompilandEnv &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolCompilandEnv)
}

void PDBSymDumper::dump(const PDBSymbolCustom &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolCustom)
}

void PDBSymDumper::dump(const PDBSymbolData &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolData)
}

void PDBSymDumper::dump(const PDBSymbolExe &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolExe)
}

void PDBSymDumper::dump(const PDBSymbolFunc &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolFunc)
}

void PDBSymDumper::dump(const PDBSymbolFuncDebugEnd &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolFuncDebugEnd)
}

void PDBSymDumper::dump(const PDBSymbolFuncDebugStart &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolFuncDebugStart)
}

void PDBSymDumper::dump(const PDBSymbolLabel &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolLabel)
}

void PDBSymDumper::dump(const PDBSymbolPublicSymbol &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolPublicSymbol)
}

void PDBSymDumper::dump(const PDBSymbolThunk &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolThunk)
}

void PDBSymDumper::dump(const PDBSymbolTypeArray &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeArray)
}

void PDBSymDumper::dump(const PDBSymbolTypeBaseClass &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeBaseClass)
}

void PDBSymDumper::dump(const PDBSymbolTypeBuiltin &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeBuiltin)
}

void PDBSymDumper::dump(const PDBSymbolTypeCustom &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeCustom)
}

void PDBSymDumper::dump(const PDBSymbolTypeDimension &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeDimension)
}

void PDBSymDumper::dump(const PDBSymbolTypeEnum &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeEnum)
}

void PDBSymDumper::dump(const PDBSymbolTypeFriend &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeFriend)
}

void PDBSymDumper::dump(const PDBSymbolTypeFunctionArg &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeFunctionArg)
}

void PDBSymDumper::dump(const PDBSymbolTypeFunctionSig &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeFunctionSig)
}

void PDBSymDumper::dump(const PDBSymbolTypeManaged &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeManaged)
}

void PDBSymDumper::dump(const PDBSymbolTypePointer &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypePointer)
}

void PDBSymDumper::dump(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeTypedef)
}

void PDBSymDumper::dump(const PDBSymbolTypeUDT &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeUDT)
}

void PDBSymDumper::dump(const PDBSymbolTypeVTable &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeVTable)
}

void PDBSymDumper::dump(const PDBSymbolTypeVTableShape &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolTypeVTableShape)
}

void PDBSymDumper::dump(const PDBSymbolUnknown &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolUnknown)
}

void PDBSymDumper::dump(const PDBSymbolUsingNamespace &Symbol, raw_ostream &OS,
                        int Indent) {
  PDB_SYMDUMP_UNREACHABLE(PDBSymbolUsingNamespace)
}
