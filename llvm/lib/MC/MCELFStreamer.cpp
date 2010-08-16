//===- lib/MC/MCELFStreamer.cpp - ELF Object Output ------------===//
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

#include "llvm/MC/MCStreamer.h"

#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCELFSymbolFlags.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmBackend.h"

using namespace llvm;

namespace {

class MCELFStreamer : public MCObjectStreamer {
public:
  MCELFStreamer(MCContext &Context, TargetAsmBackend &TAB,
                  raw_ostream &OS, MCCodeEmitter *Emitter)
    : MCObjectStreamer(Context, TAB, OS, Emitter) {}

  ~MCELFStreamer() {}

  /// @name MCStreamer Interface
  /// @{

  virtual void EmitLabel(MCSymbol *Symbol);
  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag);
  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value);
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

  virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size) {
    assert(0 && "ELF doesn't support this directive");
  }
  virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                            unsigned Size = 0, unsigned ByteAlignment = 0) {
    assert(0 && "ELF doesn't support this directive");
  }
  virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                              uint64_t Size, unsigned ByteAlignment = 0) {
    assert(0 && "ELF doesn't support this directive");
  }
  virtual void EmitBytes(StringRef Data, unsigned AddrSpace);
  virtual void EmitValue(const MCExpr *Value, unsigned Size,unsigned AddrSpace);
  virtual void EmitGPRel32Value(const MCExpr *Value) {
    assert(0 && "ELF doesn't support this directive");
  }
  virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                    unsigned ValueSize = 1,
                                    unsigned MaxBytesToEmit = 0);
  virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                 unsigned MaxBytesToEmit = 0);
  virtual void EmitValueToOffset(const MCExpr *Offset,
                                 unsigned char Value = 0);

  virtual void EmitFileDirective(StringRef Filename);
  virtual void EmitDwarfFileDirective(unsigned FileNo, StringRef Filename) {
    DEBUG(dbgs() << "FIXME: MCELFStreamer:EmitDwarfFileDirective not implemented\n");
  }

  virtual void EmitInstruction(const MCInst &Inst);
  virtual void Finish();

  /// @}
};

} // end anonymous namespace.

void MCELFStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");

  // FIXME: This is wasteful, we don't necessarily need to create a data
  // fragment. Instead, we should mark the symbol as pointing into the data
  // fragment if it exists, otherwise we should just queue the label and set its
  // fragment pointer when we emit the next fragment.
  MCDataFragment *F = getOrCreateDataFragment();
  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);
  assert(!SD.getFragment() && "Unexpected fragment on symbol data!");
  SD.setFragment(F);
  SD.setOffset(F->getContents().size());

  Symbol->setSection(*CurSection);
}

void MCELFStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {
  switch (Flag) {
  case MCAF_SubsectionsViaSymbols:
    getAssembler().setSubsectionsViaSymbols(true);
    return;
  }

  assert(0 && "invalid assembler flag!");
}

void MCELFStreamer::EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  // FIXME: Lift context changes into super class.
  getAssembler().getOrCreateSymbolData(*Symbol);
  Symbol->setVariableValue(AddValueSymbols(Value));
}

void MCELFStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                          MCSymbolAttr Attribute) {
  // Indirect symbols are handled differently, to match how 'as' handles
  // them. This makes writing matching .o files easier.
  if (Attribute == MCSA_IndirectSymbol) {
    // Note that we intentionally cannot use the symbol data here; this is
    // important for matching the string table that 'as' generates.
    IndirectSymbolData ISD;
    ISD.Symbol = Symbol;
    ISD.SectionData = getCurrentSectionData();
    getAssembler().getIndirectSymbols().push_back(ISD);
    return;
  }

  // Adding a symbol attribute always introduces the symbol, note that an
  // important side effect of calling getOrCreateSymbolData here is to register
  // the symbol with the assembler.
  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  // The implementation of symbol attributes is designed to match 'as', but it
  // leaves much to desired. It doesn't really make sense to arbitrarily add and
  // remove flags, but 'as' allows this (in particular, see .desc).
  //
  // In the future it might be worth trying to make these operations more well
  // defined.
  switch (Attribute) {
  case MCSA_LazyReference:
  case MCSA_Reference:
  case MCSA_NoDeadStrip:
  case MCSA_PrivateExtern:
  case MCSA_WeakReference:
  case MCSA_WeakDefinition:
  case MCSA_WeakDefAutoPrivate:
  case MCSA_Invalid:
  case MCSA_ELF_TypeIndFunction:
  case MCSA_IndirectSymbol:
    assert(0 && "Invalid symbol attribute for ELF!");
    break;

  case MCSA_Global:
    SD.setFlags(SD.getFlags() | ELF_STB_Global);
    SD.setExternal(true);
    break;

  case MCSA_Weak:
    SD.setFlags(SD.getFlags() | ELF_STB_Weak);
    break;

  case MCSA_Local:
    SD.setFlags(SD.getFlags() | ELF_STB_Local);
    break;

  case MCSA_ELF_TypeFunction:
    SD.setFlags(SD.getFlags() | ELF_STT_Func);
    break;

  case MCSA_ELF_TypeObject:
    SD.setFlags(SD.getFlags() | ELF_STT_Object);
    break;

  case MCSA_ELF_TypeTLS:
    SD.setFlags(SD.getFlags() | ELF_STT_Tls);
    break;

  case MCSA_ELF_TypeCommon:
    SD.setFlags(SD.getFlags() | ELF_STT_Common);
    break;

  case MCSA_ELF_TypeNoType:
    SD.setFlags(SD.getFlags() | ELF_STT_Notype);
    break;

  case MCSA_Protected:
    SD.setFlags(SD.getFlags() | ELF_STV_Protected);
    break;

  case MCSA_Hidden:
    SD.setFlags(SD.getFlags() | ELF_STV_Hidden);
    break;

  case MCSA_Internal:
    SD.setFlags(SD.getFlags() | ELF_STV_Internal);
    break;
  }
}

void MCELFStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {
  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  if ((SD.getFlags() & (0xf << ELF_STB_Shift)) == ELF_STB_Local) {
    const MCSection *Section = getAssembler().getContext().getELFSection(".bss",
                                                                    MCSectionELF::SHT_NOBITS,
                                                                    MCSectionELF::SHF_WRITE |
                                                                    MCSectionELF::SHF_ALLOC,
                                                                    SectionKind::getBSS());

    MCSectionData &SectData = getAssembler().getOrCreateSectionData(*Section);
    MCFragment *F = new MCFillFragment(0, 0, Size, &SectData);
    SD.setFragment(F);
    Symbol->setSection(*Section);
    SD.setSize(MCConstantExpr::Create(Size, getContext()));
  } else {
    SD.setExternal(true);
  }

  SD.setCommon(Size, ByteAlignment);
}

void MCELFStreamer::EmitBytes(StringRef Data, unsigned AddrSpace) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  getOrCreateDataFragment()->getContents().append(Data.begin(), Data.end());
}

void MCELFStreamer::EmitValue(const MCExpr *Value, unsigned Size,
                                unsigned AddrSpace) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  MCDataFragment *DF = getOrCreateDataFragment();

  // Avoid fixups when possible.
  int64_t AbsValue;
  if (AddValueSymbols(Value)->EvaluateAsAbsolute(AbsValue)) {
    // FIXME: Endianness assumption.
    for (unsigned i = 0; i != Size; ++i)
      DF->getContents().push_back(uint8_t(AbsValue >> (i * 8)));
  } else {
    DF->addFixup(MCFixup::Create(DF->getContents().size(), AddValueSymbols(Value),
                                 MCFixup::getKindForSize(Size)));
    DF->getContents().resize(DF->getContents().size() + Size, 0);
  }
}

void MCELFStreamer::EmitValueToAlignment(unsigned ByteAlignment,
                                           int64_t Value, unsigned ValueSize,
                                           unsigned MaxBytesToEmit) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  if (MaxBytesToEmit == 0)
    MaxBytesToEmit = ByteAlignment;
  new MCAlignFragment(ByteAlignment, Value, ValueSize, MaxBytesToEmit,
                      getCurrentSectionData());

  // Update the maximum alignment on the current section if necessary.
  if (ByteAlignment > getCurrentSectionData()->getAlignment())
    getCurrentSectionData()->setAlignment(ByteAlignment);
}

void MCELFStreamer::EmitCodeAlignment(unsigned ByteAlignment,
                                        unsigned MaxBytesToEmit) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  if (MaxBytesToEmit == 0)
    MaxBytesToEmit = ByteAlignment;
  MCAlignFragment *F = new MCAlignFragment(ByteAlignment, 0, 1, MaxBytesToEmit,
                                           getCurrentSectionData());
  F->setEmitNops(true);

  // Update the maximum alignment on the current section if necessary.
  if (ByteAlignment > getCurrentSectionData()->getAlignment())
    getCurrentSectionData()->setAlignment(ByteAlignment);
}

void MCELFStreamer::EmitValueToOffset(const MCExpr *Offset,
                                        unsigned char Value) {
  // TODO: This is exactly the same as MCMachOStreamer. Consider merging into
  // MCObjectStreamer.
  new MCOrgFragment(*Offset, Value, getCurrentSectionData());
}

// Add a symbol for the file name of this module. This is the second
// entry in the module's symbol table (the first being the null symbol).
void MCELFStreamer::EmitFileDirective(StringRef Filename) {
  MCSymbol *Symbol = getAssembler().getContext().GetOrCreateSymbol(Filename);
  Symbol->setSection(*CurSection);
  Symbol->setAbsolute();

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  SD.setFlags(ELF_STT_File | ELF_STB_Local | ELF_STV_Default);
}

void MCELFStreamer::EmitInstruction(const MCInst &Inst) {
  // Scan for values.
  for (unsigned i = 0; i != Inst.getNumOperands(); ++i)
    if (Inst.getOperand(i).isExpr())
      AddValueSymbols(Inst.getOperand(i).getExpr());

  getCurrentSectionData()->setHasInstructions(true);

  // FIXME-PERF: Common case is that we don't need to relax, encode directly
  // onto the data fragments buffers.

  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  getAssembler().getEmitter().EncodeInstruction(Inst, VecOS, Fixups);
  VecOS.flush();

  // FIXME: Eliminate this copy.
  SmallVector<MCFixup, 4> AsmFixups;
  for (unsigned i = 0, e = Fixups.size(); i != e; ++i) {
    MCFixup &F = Fixups[i];
    AsmFixups.push_back(MCFixup::Create(F.getOffset(), F.getValue(),
                                        F.getKind()));
  }

  // See if we might need to relax this instruction, if so it needs its own
  // fragment.
  //
  // FIXME-PERF: Support target hook to do a fast path that avoids the encoder,
  // when we can immediately tell that we will get something which might need
  // relaxation (and compute its size).
  //
  // FIXME-PERF: We should also be smart about immediately relaxing instructions
  // which we can already show will never possibly fit (we can also do a very
  // good job of this before we do the first relaxation pass, because we have
  // total knowledge about undefined symbols at that point). Even now, though,
  // we can do a decent job, especially on Darwin where scattering means that we
  // are going to often know that we can never fully resolve a fixup.
  if (getAssembler().getBackend().MayNeedRelaxation(Inst)) {
    MCInstFragment *IF = new MCInstFragment(Inst, getCurrentSectionData());

    // Add the fixups and data.
    //
    // FIXME: Revisit this design decision when relaxation is done, we may be
    // able to get away with not storing any extra data in the MCInst.
    IF->getCode() = Code;
    IF->getFixups() = AsmFixups;

    return;
  }

  // Add the fixups and data.
  MCDataFragment *DF = getOrCreateDataFragment();
  for (unsigned i = 0, e = AsmFixups.size(); i != e; ++i) {
    AsmFixups[i].setOffset(AsmFixups[i].getOffset() + DF->getContents().size());
    DF->addFixup(AsmFixups[i]);
  }
  DF->getContents().append(Code.begin(), Code.end());
}

void MCELFStreamer::Finish() {
  getAssembler().Finish();
}

MCStreamer *llvm::createELFStreamer(MCContext &Context, TargetAsmBackend &TAB,
                                      raw_ostream &OS, MCCodeEmitter *CE,
                                      bool RelaxAll) {
  MCELFStreamer *S = new MCELFStreamer(Context, TAB, OS, CE);
  if (RelaxAll)
    S->getAssembler().setRelaxAll(true);
  return S;
}
