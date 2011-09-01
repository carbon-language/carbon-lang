//===- lib/MC/MCELFStreamer.h - ELF Object Output -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file assembles .s files and emits ELF .o object files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCELFSTREAMER_H
#define LLVM_MC_MCELFSTREAMER_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSectionELF.h"

namespace llvm {

class MCELFStreamer : public MCObjectStreamer {
public:
  MCELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                  raw_ostream &OS, MCCodeEmitter *Emitter)
    : MCObjectStreamer(Context, TAB, OS, Emitter) {}

  MCELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                raw_ostream &OS, MCCodeEmitter *Emitter,
                MCAssembler *Assembler)
    : MCObjectStreamer(Context, TAB, OS, Emitter, Assembler) {}


  ~MCELFStreamer() {}

  /// @name MCStreamer Interface
  /// @{

  virtual void InitSections();
  virtual void ChangeSection(const MCSection *Section);
  virtual void EmitLabel(MCSymbol *Symbol);
  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag);
  virtual void EmitThumbFunc(MCSymbol *Func);
  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value);
  virtual void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol);
  virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute);
  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
    assert(0 && "ELF doesn't support this directive");
  }
  virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                unsigned ByteAlignment);
  virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol) {
    assert(0 && "ELF doesn't support this directive");
  }

  virtual void EmitCOFFSymbolStorageClass(int StorageClass) {
    assert(0 && "ELF doesn't support this directive");
  }

  virtual void EmitCOFFSymbolType(int Type) {
    assert(0 && "ELF doesn't support this directive");
  }

  virtual void EndCOFFSymbolDef() {
    assert(0 && "ELF doesn't support this directive");
  }

  virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
     MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);
     SD.setSize(Value);
  }

  virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                     unsigned ByteAlignment);

  virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                            unsigned Size = 0, unsigned ByteAlignment = 0) {
    assert(0 && "ELF doesn't support this directive");
  }
  virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                              uint64_t Size, unsigned ByteAlignment = 0) {
    assert(0 && "ELF doesn't support this directive");
  }
  virtual void EmitBytes(StringRef Data, unsigned AddrSpace);
  virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                    unsigned ValueSize = 1,
                                    unsigned MaxBytesToEmit = 0);
  virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                 unsigned MaxBytesToEmit = 0);

  virtual void EmitFileDirective(StringRef Filename);

  virtual void Finish();

private:
  virtual void EmitInstToFragment(const MCInst &Inst);
  virtual void EmitInstToData(const MCInst &Inst);

  void fixSymbolsInTLSFixups(const MCExpr *expr);

  struct LocalCommon {
    MCSymbolData *SD;
    uint64_t Size;
    unsigned ByteAlignment;
  };
  std::vector<LocalCommon> LocalCommons;

  SmallPtrSet<MCSymbol *, 16> BindingExplicitlySet;
  /// @}
  void SetSection(StringRef Section, unsigned Type, unsigned Flags,
                  SectionKind Kind) {
    SwitchSection(getContext().getELFSection(Section, Type, Flags, Kind));
  }

  void SetSectionData() {
    SetSection(".data", ELF::SHT_PROGBITS,
               ELF::SHF_WRITE |ELF::SHF_ALLOC,
               SectionKind::getDataRel());
    EmitCodeAlignment(4, 0);
  }
  void SetSectionText() {
    SetSection(".text", ELF::SHT_PROGBITS,
               ELF::SHF_EXECINSTR |
               ELF::SHF_ALLOC, SectionKind::getText());
    EmitCodeAlignment(4, 0);
  }
  void SetSectionBss() {
    SetSection(".bss", ELF::SHT_NOBITS,
               ELF::SHF_WRITE |
               ELF::SHF_ALLOC, SectionKind::getBSS());
    EmitCodeAlignment(4, 0);
  }
};

} // end llvm namespace

#endif
