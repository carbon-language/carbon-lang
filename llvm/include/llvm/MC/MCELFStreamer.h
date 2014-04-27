//===- MCELFStreamer.h - MCStreamer ELF Object File Interface ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCELFSTREAMER_H
#define LLVM_MC_MCELFSTREAMER_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {
class MCAsmBackend;
class MCAssembler;
class MCCodeEmitter;
class MCExpr;
class MCInst;
class MCSymbol;
class MCSymbolData;
class raw_ostream;

class MCELFStreamer : public MCObjectStreamer {
public:
  MCELFStreamer(MCContext &Context, MCAsmBackend &TAB, raw_ostream &OS,
                MCCodeEmitter *Emitter)
      : MCObjectStreamer(Context, TAB, OS, Emitter),
        SeenIdent(false) {}

  MCELFStreamer(MCContext &Context, MCAsmBackend &TAB, raw_ostream &OS,
                MCCodeEmitter *Emitter, MCAssembler *Assembler)
      : MCObjectStreamer(Context, TAB, OS, Emitter, Assembler),
        SeenIdent(false) {}

  virtual ~MCELFStreamer();

  /// @name MCStreamer Interface
  /// @{

  void InitSections() override;
  void ChangeSection(const MCSection *Section,
                     const MCExpr *Subsection) override;
  void EmitLabel(MCSymbol *Symbol) override;
  void EmitDebugLabel(MCSymbol *Symbol) override;
  void EmitAssemblerFlag(MCAssemblerFlag Flag) override;
  void EmitThumbFunc(MCSymbol *Func) override;
  void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) override;
  bool EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override;
  void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) override;
  void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        unsigned ByteAlignment) override;
  void BeginCOFFSymbolDef(const MCSymbol *Symbol) override;
  void EmitCOFFSymbolStorageClass(int StorageClass) override;
  void EmitCOFFSymbolType(int Type) override;
  void EndCOFFSymbolDef() override;

  void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) override;

  void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                             unsigned ByteAlignment) override;

  void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = nullptr,
                    uint64_t Size = 0, unsigned ByteAlignment = 0) override;
  void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                      uint64_t Size, unsigned ByteAlignment = 0) override;
  void EmitValueImpl(const MCExpr *Value, unsigned Size,
                     const SMLoc &Loc = SMLoc()) override;

  void EmitFileDirective(StringRef Filename) override;

  void EmitIdent(StringRef IdentString) override;

  void EmitValueToAlignment(unsigned, int64_t, unsigned, unsigned) override;

  void Flush() override;

  void FinishImpl() override;

  void EmitBundleAlignMode(unsigned AlignPow2) override;
  void EmitBundleLock(bool AlignToEnd) override;
  void EmitBundleUnlock() override;

private:
  void EmitInstToFragment(const MCInst &Inst, const MCSubtargetInfo &) override;
  void EmitInstToData(const MCInst &Inst, const MCSubtargetInfo &) override;

  void fixSymbolsInTLSFixups(const MCExpr *expr);

  bool SeenIdent;

  struct LocalCommon {
    MCSymbolData *SD;
    uint64_t Size;
    unsigned ByteAlignment;
  };

  std::vector<LocalCommon> LocalCommons;

  SmallPtrSet<MCSymbol *, 16> BindingExplicitlySet;
};

MCELFStreamer *createARMELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                                    raw_ostream &OS, MCCodeEmitter *Emitter,
                                    bool RelaxAll, bool NoExecStack,
                                    bool IsThumb);

} // end namespace llvm

#endif
