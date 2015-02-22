//===- TypedefDumper.h - llvm-pdbdump typedef dumper ---------*- C++ ----*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_TYPEDEFDUMPER_H
#define LLVM_TOOLS_LLVMPDBDUMP_TYPEDEFDUMPER_H

#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

namespace llvm {

class TypedefDumper : public PDBSymDumper {
public:
  TypedefDumper();

  void start(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS, int Indent);

  void dump(const PDBSymbolTypeArray &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypeBuiltin &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypeEnum &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypeFunctionSig &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypePointer &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypeUDT &Symbol, raw_ostream &OS,
            int Indent) override;
};
}

#endif
