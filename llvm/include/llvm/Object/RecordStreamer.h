//===-- RecordStreamer.h - Record asm defined and used symbols ---*- C++ -*===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_RECORD_STREAMER
#define LLVM_OBJECT_RECORD_STREAMER

#include "llvm/MC/MCStreamer.h"

namespace llvm {
class RecordStreamer : public MCStreamer {
public:
  enum State { NeverSeen, Global, Defined, DefinedGlobal, Used };

private:
  StringMap<State> Symbols;
  void markDefined(const MCSymbol &Symbol);
  void markGlobal(const MCSymbol &Symbol);
  void markUsed(const MCSymbol &Symbol);
  void visitUsedSymbol(const MCSymbol &Sym) override;

public:
  typedef StringMap<State>::const_iterator const_iterator;
  const_iterator begin();
  const_iterator end();
  RecordStreamer(MCContext &Context);
  void EmitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override;
  void EmitLabel(MCSymbol *Symbol) override;
  void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) override;
  bool EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override;
  void EmitZerofill(const MCSection *Section, MCSymbol *Symbol, uint64_t Size,
                    unsigned ByteAlignment) override;
  void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        unsigned ByteAlignment) override;
};
}
#endif
