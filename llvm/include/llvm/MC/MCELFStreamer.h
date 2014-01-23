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
  MCELFStreamer(MCContext &Context, MCTargetStreamer *TargetStreamer,
                MCAsmBackend &TAB, raw_ostream &OS, MCCodeEmitter *Emitter)
      : MCObjectStreamer(Context, TargetStreamer, TAB, OS, Emitter), 
                         SeenIdent(false) {}

  MCELFStreamer(MCContext &Context, MCTargetStreamer *TargetStreamer,
                MCAsmBackend &TAB, raw_ostream &OS, MCCodeEmitter *Emitter,
                MCAssembler *Assembler)
      : MCObjectStreamer(Context, TargetStreamer, TAB, OS, Emitter, Assembler), 
                         SeenIdent(false) {}

  virtual ~MCELFStreamer();

  /// @name MCStreamer Interface
  /// @{

  virtual void InitSections();
  virtual void InitToTextSection();
  virtual void ChangeSection(const MCSection *Section,
                             const MCExpr *Subsection);
  virtual void EmitLabel(MCSymbol *Symbol);
  virtual void EmitDebugLabel(MCSymbol *Symbol);
  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag);
  virtual void EmitThumbFunc(MCSymbol *Func);
  virtual void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol);
  virtual bool EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute);
  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue);
  virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                unsigned ByteAlignment);
  virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol);
  virtual void EmitCOFFSymbolStorageClass(int StorageClass);
  virtual void EmitCOFFSymbolType(int Type);
  virtual void EndCOFFSymbolDef();

  virtual MCSymbolData &getOrCreateSymbolData(MCSymbol *Symbol);

  virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value);

  virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                     unsigned ByteAlignment);

  virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                            uint64_t Size = 0, unsigned ByteAlignment = 0);
  virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                              uint64_t Size, unsigned ByteAlignment = 0);
  virtual void EmitValueImpl(const MCExpr *Value, unsigned Size);

  virtual void EmitFileDirective(StringRef Filename);

  virtual void EmitIdent(StringRef IdentString);

  virtual void EmitValueToAlignment(unsigned, int64_t, unsigned, unsigned);

  virtual void Flush();

  virtual void FinishImpl();

private:
  virtual void EmitInstToFragment(const MCInst &Inst);
  virtual void EmitInstToData(const MCInst &Inst);

  virtual void EmitBundleAlignMode(unsigned AlignPow2);
  virtual void EmitBundleLock(bool AlignToEnd);
  virtual void EmitBundleUnlock();

  void fixSymbolsInTLSFixups(const MCExpr *expr);

  bool SeenIdent;

  struct LocalCommon {
    MCSymbolData *SD;
    uint64_t Size;
    unsigned ByteAlignment;
  };

  std::vector<LocalCommon> LocalCommons;

  SmallPtrSet<MCSymbol *, 16> BindingExplicitlySet;

  void SetSectionData();
  void SetSectionText();
  void SetSectionBss();
};

MCELFStreamer *createARMELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                                    raw_ostream &OS, MCCodeEmitter *Emitter,
                                    bool RelaxAll, bool NoExecStack,
                                    bool IsThumb);

} // end namespace llvm

#endif
