//===-- llvm/MC/WinCOFFStreamer.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains an implementation of a Win32 COFF object file streamer.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "WinCOFFStreamer"

#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/Target/TargetAsmBackend.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define dbg_notimpl(x) \
  do { dbgs() << "not implemented, " << __FUNCTION__  << " (" << x << ")"; \
    abort(); } while (false);

namespace {
class WinCOFFStreamer : public MCObjectStreamer {
public:
  WinCOFFStreamer(MCContext &Context,
                  TargetAsmBackend &TAB,
                  MCCodeEmitter &CE,
                  raw_ostream &OS);

  // MCStreamer interface

  virtual void EmitLabel(MCSymbol *Symbol);
  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag);
  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value);
  virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute);
  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue);
  virtual void BeginCOFFSymbolDef(MCSymbol const *Symbol);
  virtual void EmitCOFFSymbolStorageClass(int StorageClass);
  virtual void EmitCOFFSymbolType(int Type);
  virtual void EndCOFFSymbolDef();
  virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value);
  virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        unsigned ByteAlignment);
  virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size);
  virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol,
                    unsigned Size,unsigned ByteAlignment);
  virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                      uint64_t Size, unsigned ByteAlignment);
  virtual void EmitBytes(StringRef Data, unsigned AddrSpace);
  virtual void EmitValue(const MCExpr *Value, unsigned Size, 
                         unsigned AddrSpace);
  virtual void EmitGPRel32Value(const MCExpr *Value);
  virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value,
                            unsigned ValueSize, unsigned MaxBytesToEmit);
  virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                 unsigned MaxBytesToEmit);
  virtual void EmitValueToOffset(const MCExpr *Offset, unsigned char Value);
  virtual void EmitFileDirective(StringRef Filename);
  virtual void EmitDwarfFileDirective(unsigned FileNo,StringRef Filename);
  virtual void EmitInstruction(const MCInst &Instruction);
  virtual void Finish();
};
} // end anonymous namespace.

WinCOFFStreamer::WinCOFFStreamer(MCContext &Context,
                                 TargetAsmBackend &TAB,
                                 MCCodeEmitter &CE,
                                 raw_ostream &OS)
    : MCObjectStreamer(Context, TAB, OS, &CE) {
}

// MCStreamer interface

void WinCOFFStreamer::EmitLabel(MCSymbol *Symbol) {
}

void WinCOFFStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {
  dbg_notimpl("Flag = " << Flag);
}

void WinCOFFStreamer::EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
}

void WinCOFFStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                          MCSymbolAttr Attribute) {
}

void WinCOFFStreamer::EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
  dbg_notimpl("Symbol = " << Symbol->getName() << ", DescValue = "<< DescValue);
}

void WinCOFFStreamer::BeginCOFFSymbolDef(MCSymbol const *Symbol) {
}

void WinCOFFStreamer::EmitCOFFSymbolStorageClass(int StorageClass) {
}

void WinCOFFStreamer::EmitCOFFSymbolType(int Type) {
}

void WinCOFFStreamer::EndCOFFSymbolDef() {
}

void WinCOFFStreamer::EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
  dbg_notimpl("Symbol = " << Symbol->getName() << ", Value = " << *Value);
}

void WinCOFFStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                      unsigned ByteAlignment) {
}

void WinCOFFStreamer::EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size) {
}

void WinCOFFStreamer::EmitZerofill(const MCSection *Section, MCSymbol *Symbol,
                                  unsigned Size,unsigned ByteAlignment) {
  MCSectionCOFF const *SectionCOFF =
    static_cast<MCSectionCOFF const *>(Section);

  dbg_notimpl("Section = " << SectionCOFF->getSectionName() << ", Symbol = " <<
              Symbol->getName() << ", Size = " << Size << ", ByteAlignment = "
              << ByteAlignment);
}

void WinCOFFStreamer::EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                                     uint64_t Size, unsigned ByteAlignment) {
  MCSectionCOFF const *SectionCOFF =
    static_cast<MCSectionCOFF const *>(Section);

  dbg_notimpl("Section = " << SectionCOFF->getSectionName() << ", Symbol = " <<
              Symbol->getName() << ", Size = " << Size << ", ByteAlignment = "
              << ByteAlignment);
}

void WinCOFFStreamer::EmitBytes(StringRef Data, unsigned AddrSpace) {
}

void WinCOFFStreamer::EmitValue(const MCExpr *Value, unsigned Size,
                               unsigned AddrSpace) {
}

void WinCOFFStreamer::EmitGPRel32Value(const MCExpr *Value) {
  dbg_notimpl("Value = '" << *Value);
}

void WinCOFFStreamer::EmitValueToAlignment(unsigned ByteAlignment,
                                          int64_t Value,
                                          unsigned ValueSize,
                                          unsigned MaxBytesToEmit) {
}

void WinCOFFStreamer::EmitCodeAlignment(unsigned ByteAlignment,
                                       unsigned MaxBytesToEmit = 0) {
}

void WinCOFFStreamer::EmitValueToOffset(const MCExpr *Offset,
                                       unsigned char Value = 0) {
  dbg_notimpl("Offset = '" << *Offset << "', Value = " << Value);
}

void WinCOFFStreamer::EmitFileDirective(StringRef Filename) {
  // Ignore for now, linkers don't care, and proper debug
  // info will be a much large effort.
}

void WinCOFFStreamer::EmitDwarfFileDirective(unsigned FileNo,
                                            StringRef Filename) {
  dbg_notimpl("FileNo = " << FileNo << ", Filename = '" << Filename << "'");
}

void WinCOFFStreamer::EmitInstruction(const MCInst &Instruction) {
}

void WinCOFFStreamer::Finish() {
  MCObjectStreamer::Finish();
}

namespace llvm
{
  MCStreamer *createWinCOFFStreamer(MCContext &Context,
                                    TargetAsmBackend &TAB,
                                    MCCodeEmitter &CE,
                                    raw_ostream &OS) {
    return new WinCOFFStreamer(Context, TAB, CE, OS);
  }
}
