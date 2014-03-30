//===-- lib/MC/MCDisassembler.cpp - Disassembler interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCExternalSymbolizer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

MCDisassembler::~MCDisassembler() {
}

void MCDisassembler::setupForSymbolicDisassembly(
    LLVMOpInfoCallback GetOpInfo, LLVMSymbolLookupCallback SymbolLookUp,
    void *DisInfo, MCContext *Ctx, std::unique_ptr<MCRelocationInfo> &RelInfo) {
  this->GetOpInfo = GetOpInfo;
  this->SymbolLookUp = SymbolLookUp;
  this->DisInfo = DisInfo;
  this->Ctx = Ctx;
  assert(Ctx != 0 && "No MCContext given for symbolic disassembly");
  if (!Symbolizer)
    Symbolizer.reset(new MCExternalSymbolizer(*Ctx, std::move(RelInfo),
                                              GetOpInfo, SymbolLookUp,
                                              DisInfo));
}

bool MCDisassembler::tryAddingSymbolicOperand(MCInst &Inst, int64_t Value,
                                              uint64_t Address, bool IsBranch,
                                              uint64_t Offset,
                                              uint64_t InstSize) const {
  raw_ostream &cStream = CommentStream ? *CommentStream : nulls();
  if (Symbolizer)
    return Symbolizer->tryAddingSymbolicOperand(Inst, cStream, Value, Address,
                                                IsBranch, Offset, InstSize);
  return false;
}

void MCDisassembler::tryAddingPcLoadReferenceComment(int64_t Value,
                                                     uint64_t Address) const {
  raw_ostream &cStream = CommentStream ? *CommentStream : nulls();
  if (Symbolizer)
    Symbolizer->tryAddingPcLoadReferenceComment(cStream, Value, Address);
}

void MCDisassembler::setSymbolizer(std::unique_ptr<MCSymbolizer> Symzer) {
  Symbolizer = std::move(Symzer);
}
