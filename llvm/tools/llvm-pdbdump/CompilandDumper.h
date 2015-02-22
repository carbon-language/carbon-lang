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

class CompilandDumper : public PDBSymDumper {
public:
  CompilandDumper();

  void start(const PDBSymbolCompiland &Symbol, raw_ostream &OS, int Indent,
             bool Children);

  void dump(const PDBSymbolCompilandDetails &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolCompilandEnv &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolData &Symbol, raw_ostream &OS, int Indent) override;
  void dump(const PDBSymbolFunc &Symbol, raw_ostream &OS, int Indent) override;
  void dump(const PDBSymbolLabel &Symbol, raw_ostream &OS, int Indent) override;
  void dump(const PDBSymbolThunk &Symbol, raw_ostream &OS, int Indent) override;
  void dump(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolUnknown &Symbol, raw_ostream &OS,
            int Indent) override;
};
}

#endif
