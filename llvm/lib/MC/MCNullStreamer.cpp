//===- lib/MC/MCNullStreamer.cpp - Dummy Streamer Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"

using namespace llvm;

namespace {

  class MCNullStreamer : public MCStreamer {
  public:
    MCNullStreamer(MCContext &Context) : MCStreamer(Context) {}

    /// @name MCStreamer Interface
    /// @{

    virtual void SwitchSection(const MCSection *Section) {}

    virtual void EmitLabel(MCSymbol *Symbol) {}

    virtual void EmitAssemblerFlag(AssemblerFlag Flag) {}

    virtual void EmitAssignment(MCSymbol *Symbol, const MCValue &Value,
                                bool MakeAbsolute = false) {}

    virtual void EmitSymbolAttribute(MCSymbol *Symbol, SymbolAttr Attribute) {}

    virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {}

    virtual void EmitLocalSymbol(MCSymbol *Symbol, const MCValue &Value) {}

    virtual void EmitCommonSymbol(MCSymbol *Symbol, unsigned Size,
                                  unsigned Pow2Alignment, bool IsLocal) {}

    virtual void EmitZerofill(MCSection *Section, MCSymbol *Symbol = NULL,
                              unsigned Size = 0, unsigned Pow2Alignment = 0) {}

    virtual void EmitBytes(const StringRef &Data) {}

    virtual void EmitValue(const MCValue &Value, unsigned Size) {}

    virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                      unsigned ValueSize = 1,
                                      unsigned MaxBytesToEmit = 0) {}

    virtual void EmitValueToOffset(const MCValue &Offset, 
                                   unsigned char Value = 0) {}
    
    virtual void EmitInstruction(const MCInst &Inst) {}

    virtual void Finish() {}
    
    /// @}
  };

}
    
MCStreamer *llvm::createNullStreamer(MCContext &Context) {
  return new MCNullStreamer(Context);
}
