//===- ARM64ExternalSymbolizer.h - Symbolizer for ARM64 ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Symbolize ARM64 assembly code during disassembly using callbacks.
//
//===----------------------------------------------------------------------===//

#ifndef ARM64EXTERNALSYMBOLIZER_H
#define ARM64EXTERNALSYMBOLIZER_H

#include "llvm/MC/MCExternalSymbolizer.h"

namespace llvm {

class ARM64ExternalSymbolizer : public MCExternalSymbolizer {
public:
  ARM64ExternalSymbolizer(MCContext &Ctx,
                          std::unique_ptr<MCRelocationInfo> RelInfo,
                          LLVMOpInfoCallback GetOpInfo,
                          LLVMSymbolLookupCallback SymbolLookUp, void *DisInfo)
    : MCExternalSymbolizer(Ctx, std::move(RelInfo), GetOpInfo, SymbolLookUp,
                           DisInfo) {}

  bool tryAddingSymbolicOperand(MCInst &MI, raw_ostream &CommentStream,
                                int64_t Value, uint64_t Address, bool IsBranch,
                                uint64_t Offset, uint64_t InstSize) override;
};

} // namespace llvm

#endif
