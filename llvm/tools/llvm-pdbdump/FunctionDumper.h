//===- FunctionDumper.h --------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_FUNCTIONDUMPER_H
#define LLVM_TOOLS_LLVMPDBDUMP_FUNCTIONDUMPER_H

#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

namespace llvm {

class FunctionDumper : public PDBSymDumper {
public:
  FunctionDumper();

  enum class PointerType { None, Pointer, Reference };

  void start(const PDBSymbolTypeFunctionSig &Symbol, PointerType Pointer,
             raw_ostream &OS);
  void start(const PDBSymbolFunc &Symbol, raw_ostream &OS);

  void dump(const PDBSymbolTypeArray &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypeBuiltin &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypeEnum &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypeFunctionArg &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypePointer &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypeTypedef &Symbol, raw_ostream &OS,
            int Indent) override;
  void dump(const PDBSymbolTypeUDT &Symbol, raw_ostream &OS,
            int Indent) override;
};
}

#endif
