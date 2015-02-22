//===- PDBSymDumper.h - base interface for PDB symbol dumper *- C++ -----*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMDUMPER_H
#define LLVM_DEBUGINFO_PDB_PDBSYMDUMPER_H

#include "PDBTypes.h"

namespace llvm {

class raw_ostream;

class PDBSymDumper {
public:
  PDBSymDumper(bool ShouldRequireImpl);
  virtual ~PDBSymDumper();

  virtual void dump(const PDBSymbolAnnotation &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolBlock &Symbol, raw_ostream &OS, int Indent);
  virtual void dump(const PDBSymbolCompiland &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolCompilandDetails &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolCompilandEnv &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolCustom &Symbol, raw_ostream &OS, int Indent);
  virtual void dump(const PDBSymbolData &Symbol, raw_ostream &OS, int Indent);
  virtual void dump(const PDBSymbolExe &Symbol, raw_ostream &OS, int Indent);
  virtual void dump(const PDBSymbolFunc &Symbol, raw_ostream &OS, int Indent);
  virtual void dump(const PDBSymbolFuncDebugEnd &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolFuncDebugStart &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolLabel &Symbol, raw_ostream &OS, int Indent);
  virtual void dump(const PDBSymbolPublicSymbol &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolThunk &Symbol, raw_ostream &OS, int Indent);
  virtual void dump(const PDBSymbolTypeArray &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypeBaseClass &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypeBuiltin &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypeCustom &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypeDimension &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypeEnum &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypeFriend &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypeFunctionArg &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypeFunctionSig &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypeManaged &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypePointer &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypeUDT &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypeVTable &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolTypeVTableShape &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolUnknown &Symbol, raw_ostream &OS,
                    int Indent);
  virtual void dump(const PDBSymbolUsingNamespace &Symbol, raw_ostream &OS,
                    int Indent);

private:
  bool RequireImpl;
};
}

#endif
