//===- VariableDumper.h - PDBSymDumper implementation for types -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_VARIABLEDUMPER_H
#define LLVM_TOOLS_LLVMPDBDUMP_VARIABLEDUMPER_H

#include "llvm/DebugInfo/PDB/PDBSymDumper.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class VariableDumper : public PDBSymDumper {
public:
  VariableDumper();

  void start(const PDBSymbolData &Var, raw_ostream &OS, int Indent);

  void dump(const PDBSymbolTypeBuiltin &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypeEnum &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypeFunctionSig &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypePointer &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypeUDT &Symbol, raw_ostream &OS,
            int Indent) override;

private:
  void dumpSymbolTypeAndName(const PDBSymbol &Type, StringRef Name,
                             raw_ostream &OS);
  bool tryDumpFunctionPointer(const PDBSymbol &Type, StringRef Name,
                              raw_ostream &OS);
};
}

#endif
