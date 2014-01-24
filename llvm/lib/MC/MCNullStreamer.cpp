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
    MCNullStreamer(MCContext &Context) : MCStreamer(Context, 0) {}

    /// @name MCStreamer Interface
    /// @{

    virtual void ChangeSection(const MCSection *Section,
                               const MCExpr *Subsection) {
    }

    virtual void EmitLabel(MCSymbol *Symbol) {
      assert(Symbol->isUndefined() && "Cannot define a symbol twice!");
      assert(getCurrentSection().first &&"Cannot emit before setting section!");
      AssignSection(Symbol, getCurrentSection().first);
    }
    virtual void EmitDebugLabel(MCSymbol *Symbol) {
      EmitLabel(Symbol);
    }
    virtual void EmitAssemblerFlag(MCAssemblerFlag Flag) {}
    virtual void EmitThumbFunc(MCSymbol *Func) {}

    virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {}
    virtual void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol){}
    virtual void EmitDwarfAdvanceLineAddr(int64_t LineDelta,
                                          const MCSymbol *LastLabel,
                                          const MCSymbol *Label,
                                          unsigned PointerSize) {}

    virtual bool EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute){
      return true;
    }

    virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {}

    virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol) {}
    virtual void EmitCOFFSymbolStorageClass(int StorageClass) {}
    virtual void EmitCOFFSymbolType(int Type) {}
    virtual void EndCOFFSymbolDef() {}
    virtual void EmitCOFFSecRel32(MCSymbol const *Symbol) {}

    virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {}
    virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                  unsigned ByteAlignment) {}
    virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {}
    virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                              uint64_t Size = 0, unsigned ByteAlignment = 0) {}
    virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                                uint64_t Size, unsigned ByteAlignment) {}
    virtual void EmitBytes(StringRef Data) {}

    virtual void EmitValueImpl(const MCExpr *Value, unsigned Size) {}
    virtual void EmitULEB128Value(const MCExpr *Value) {}
    virtual void EmitSLEB128Value(const MCExpr *Value) {}
    virtual void EmitGPRel32Value(const MCExpr *Value) {}
    virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                      unsigned ValueSize = 1,
                                      unsigned MaxBytesToEmit = 0) {}

    virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                   unsigned MaxBytesToEmit = 0) {}

    virtual bool EmitValueToOffset(const MCExpr *Offset,
                                   unsigned char Value = 0) { return false; }

    virtual void EmitFileDirective(StringRef Filename) {}
    virtual bool EmitDwarfFileDirective(unsigned FileNo, StringRef Directory,
                                        StringRef Filename, unsigned CUID = 0) {
      return false;
    }
    virtual void EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                                       unsigned Column, unsigned Flags,
                                       unsigned Isa, unsigned Discriminator,
                                       StringRef FileName) {}
    virtual void EmitInstruction(const MCInst &Inst) {}

    virtual void EmitBundleAlignMode(unsigned AlignPow2) {}
    virtual void EmitBundleLock(bool AlignToEnd) {}
    virtual void EmitBundleUnlock() {}

    virtual void FinishImpl() {}

    virtual void EmitCFIEndProcImpl(MCDwarfFrameInfo &Frame) {
      RecordProcEnd(Frame);
    }
  };

}

MCStreamer *llvm::createNullStreamer(MCContext &Context) {
  return new MCNullStreamer(Context);
}
