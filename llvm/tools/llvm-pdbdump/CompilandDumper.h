//===- CompilandDumper.h - llvm-pdbdump compiland symbol dumper *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_COMPILANDDUMPER_H
#define LLVM_TOOLS_LLVMPDBDUMP_COMPILANDDUMPER_H

#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

namespace llvm {

class LinePrinter;

class CompilandDumper : public PDBSymDumper {
public:
  CompilandDumper(LinePrinter &P);

  void start(const PDBSymbolCompiland &Symbol, bool Children);

  void dump(const PDBSymbolCompilandDetails &Symbol) override;
  void dump(const PDBSymbolCompilandEnv &Symbol) override;
  void dump(const PDBSymbolData &Symbol) override;
  void dump(const PDBSymbolFunc &Symbol) override;
  void dump(const PDBSymbolLabel &Symbol) override;
  void dump(const PDBSymbolThunk &Symbol) override;
  void dump(const PDBSymbolTypeTypedef &Symbol) override;
  void dump(const PDBSymbolUnknown &Symbol) override;

private:
  LinePrinter &Printer;
};
}

#endif
