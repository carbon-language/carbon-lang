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

using namespace llvm;

namespace {

  class MCNullStreamer : public MCStreamer {
  public:
    MCNullStreamer(MCContext &Context) : MCStreamer(Context) {}

    /// @name MCStreamer Interface
    /// @{

    virtual void InitSections() {
    }

    virtual void SwitchSection(const MCSection *Section) {
      PrevSection = CurSection;
      CurSection = Section;
    }

    virtual void EmitLabel(MCSymbol *Symbol) {
      assert(Symbol->isUndefined() && "Cannot define a symbol twice!");
      assert(CurSection && "Cannot emit before setting section!");
      Symbol->setSection(*CurSection);
    }

    virtual void EmitAssemblerFlag(MCAssemblerFlag Flag) {}
    virtual void EmitThumbFunc(MCSymbol *Func) {}

    virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {}
    virtual void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol){}

    virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute){}

    virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {}

    virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol) {}
    virtual void EmitCOFFSymbolStorageClass(int StorageClass) {}
    virtual void EmitCOFFSymbolType(int Type) {}
    virtual void EndCOFFSymbolDef() {}

    virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {}
    virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                  unsigned ByteAlignment) {}
    virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size) {}

    virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                              unsigned Size = 0, unsigned ByteAlignment = 0) {}
    virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                                uint64_t Size, unsigned ByteAlignment) {}
    virtual void EmitBytes(StringRef Data, unsigned AddrSpace) {}

    virtual void EmitValue(const MCExpr *Value, unsigned Size,
                           unsigned AddrSpace) {}
    virtual void EmitULEB128Value(const MCExpr *Value,
                                  unsigned AddrSpace = 0) {}
    virtual void EmitSLEB128Value(const MCExpr *Value,
                                  unsigned AddrSpace = 0) {}
    virtual void EmitGPRel32Value(const MCExpr *Value) {}
    virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                      unsigned ValueSize = 1,
                                      unsigned MaxBytesToEmit = 0) {}

    virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                   unsigned MaxBytesToEmit = 0) {}

    virtual void EmitValueToOffset(const MCExpr *Offset,
                                   unsigned char Value = 0) {}
    
    virtual void EmitFileDirective(StringRef Filename) {}
    virtual void EmitDwarfFileDirective(unsigned FileNo,StringRef Filename) {}
    virtual void EmitInstruction(const MCInst &Inst) {}

    virtual void Finish() {}
    
    /// @}
  };

}
    
MCStreamer *llvm::createNullStreamer(MCContext &Context) {
  return new MCNullStreamer(Context);
}
