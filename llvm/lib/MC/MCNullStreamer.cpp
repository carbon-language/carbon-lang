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

    void ChangeSection(const MCSection *Section,
                       const MCExpr *Subsection) override {
    }

    void EmitLabel(MCSymbol *Symbol) override {
      assert(Symbol->isUndefined() && "Cannot define a symbol twice!");
      assert(getCurrentSection().first &&"Cannot emit before setting section!");
      AssignSection(Symbol, getCurrentSection().first);
    }
    void EmitDebugLabel(MCSymbol *Symbol) override {
      EmitLabel(Symbol);
    }
    void EmitAssemblerFlag(MCAssemblerFlag Flag) override {}
    void EmitThumbFunc(MCSymbol *Func) override {}

    void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) override {}
    void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) override {}
    void EmitDwarfAdvanceLineAddr(int64_t LineDelta,
                                  const MCSymbol *LastLabel,
                                  const MCSymbol *Label,
                                  unsigned PointerSize) override {}

    bool EmitSymbolAttribute(MCSymbol *Symbol,
                             MCSymbolAttr Attribute) override {
      return true;
    }

    void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) override {}

    void BeginCOFFSymbolDef(const MCSymbol *Symbol) override {}
    void EmitCOFFSymbolStorageClass(int StorageClass) override {}
    void EmitCOFFSymbolType(int Type) override {}
    void EndCOFFSymbolDef() override {}
    void EmitCOFFSecRel32(MCSymbol const *Symbol) override {}

    void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) override {}
    void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                          unsigned ByteAlignment) override {}
    void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                               unsigned ByteAlignment) override {}
    void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = nullptr,
                      uint64_t Size = 0, unsigned ByteAlignment = 0) override {}
    void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                        uint64_t Size, unsigned ByteAlignment) override {}
    void EmitBytes(StringRef Data) override {}

    void EmitValueImpl(const MCExpr *Value, unsigned Size,
                       const SMLoc &Loc = SMLoc()) override {}
    void EmitULEB128Value(const MCExpr *Value) override {}
    void EmitSLEB128Value(const MCExpr *Value) override {}
    void EmitGPRel32Value(const MCExpr *Value) override {}
    void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                              unsigned ValueSize = 1,
                              unsigned MaxBytesToEmit = 0) override {}

    void EmitCodeAlignment(unsigned ByteAlignment,
                           unsigned MaxBytesToEmit = 0) override {}

    bool EmitValueToOffset(const MCExpr *Offset,
                           unsigned char Value = 0) override { return false; }

    void EmitFileDirective(StringRef Filename) override {}
    unsigned EmitDwarfFileDirective(unsigned FileNo, StringRef Directory,
                                    StringRef Filename,
                                    unsigned CUID = 0) override {
      return 0;
    }
    void EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                               unsigned Column, unsigned Flags,
                               unsigned Isa, unsigned Discriminator,
                               StringRef FileName) override {}
    void EmitInstruction(const MCInst &Inst, const MCSubtargetInfo&) override {}

    void EmitBundleAlignMode(unsigned AlignPow2) override {}
    void EmitBundleLock(bool AlignToEnd) override {}
    void EmitBundleUnlock() override {}

    void FinishImpl() override {}

    void EmitCFIEndProcImpl(MCDwarfFrameInfo &Frame) override {
      RecordProcEnd(Frame);
    }
  };

}

MCStreamer *llvm::createNullStreamer(MCContext &Context) {
  return new MCNullStreamer(Context);
}
