//===-- llvm/MC/WinCOFFStreamer.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains an implementation of a Windows COFF object file streamer.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCWin64EH.h"
#include "llvm/MC/MCWinCOFFStreamer.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "WinCOFFStreamer"

namespace llvm {
MCWinCOFFStreamer::MCWinCOFFStreamer(MCContext &Context, MCAsmBackend &MAB,
                                     MCCodeEmitter &CE, raw_ostream &OS)
    : MCObjectStreamer(Context, MAB, OS, &CE), CurSymbol(nullptr) {}

void MCWinCOFFStreamer::EmitInstToData(const MCInst &Inst,
                                       const MCSubtargetInfo &STI) {
  MCDataFragment *DF = getOrCreateDataFragment();

  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  getAssembler().getEmitter().EncodeInstruction(Inst, VecOS, Fixups, STI);
  VecOS.flush();

  // Add the fixups and data.
  for (unsigned i = 0, e = Fixups.size(); i != e; ++i) {
    Fixups[i].setOffset(Fixups[i].getOffset() + DF->getContents().size());
    DF->getFixups().push_back(Fixups[i]);
  }

  DF->getContents().append(Code.begin(), Code.end());
}

void MCWinCOFFStreamer::InitSections() {
  // FIXME: this is identical to the ELF one.
  // This emulates the same behavior of GNU as. This makes it easier
  // to compare the output as the major sections are in the same order.
  SwitchSection(getContext().getObjectFileInfo()->getTextSection());
  EmitCodeAlignment(4);

  SwitchSection(getContext().getObjectFileInfo()->getDataSection());
  EmitCodeAlignment(4);

  SwitchSection(getContext().getObjectFileInfo()->getBSSSection());
  EmitCodeAlignment(4);

  SwitchSection(getContext().getObjectFileInfo()->getTextSection());
}

void MCWinCOFFStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");
  MCObjectStreamer::EmitLabel(Symbol);
}

void MCWinCOFFStreamer::EmitDebugLabel(MCSymbol *Symbol) {
  EmitLabel(Symbol);
}

void MCWinCOFFStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {
  llvm_unreachable("not implemented");
}

void MCWinCOFFStreamer::EmitThumbFunc(MCSymbol *Func) {
  llvm_unreachable("not implemented");
}

bool MCWinCOFFStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                            MCSymbolAttr Attribute) {
  assert(Symbol && "Symbol must be non-null!");
  assert((!Symbol->isInSection() ||
          Symbol->getSection().getVariant() == MCSection::SV_COFF) &&
         "Got non-COFF section in the COFF backend!");

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  switch (Attribute) {
  default: return false;
  case MCSA_WeakReference:
  case MCSA_Weak:
    SD.modifyFlags(COFF::SF_WeakExternal, COFF::SF_WeakExternal);
    SD.setExternal(true);
    break;
  case MCSA_Global:
    SD.setExternal(true);
    break;
  }

  return true;
}

void MCWinCOFFStreamer::EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
  llvm_unreachable("not implemented");
}

void MCWinCOFFStreamer::BeginCOFFSymbolDef(MCSymbol const *Symbol) {
  assert((!Symbol->isInSection() ||
          Symbol->getSection().getVariant() == MCSection::SV_COFF) &&
         "Got non-COFF section in the COFF backend!");
  assert(!CurSymbol && "starting new symbol definition in a symbol definition");
  CurSymbol = Symbol;
}

void MCWinCOFFStreamer::EmitCOFFSymbolStorageClass(int StorageClass) {
  assert(CurSymbol && "StorageClass specified outside of symbol definition");
  assert((StorageClass & ~0xFF) == 0 &&
         "StorageClass must only have data in the first byte!");

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*CurSymbol);
  SD.modifyFlags(StorageClass << COFF::SF_ClassShift, COFF::SF_ClassMask);
}

void MCWinCOFFStreamer::EmitCOFFSymbolType(int Type) {
  assert(CurSymbol && "SymbolType specified outside of a symbol definition");
  assert((Type & ~0xFFFF) == 0 &&
         "Type must only have data in the first 2 bytes");

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*CurSymbol);
  SD.modifyFlags(Type << COFF::SF_TypeShift, COFF::SF_TypeMask);
}

void MCWinCOFFStreamer::EndCOFFSymbolDef() {
  assert(CurSymbol && "ending symbol definition without beginning one");
  CurSymbol = nullptr;
}

void MCWinCOFFStreamer::EmitCOFFSectionIndex(MCSymbol const *Symbol) {
  MCDataFragment *DF = getOrCreateDataFragment();
  const MCSymbolRefExpr *SRE = MCSymbolRefExpr::Create(Symbol, getContext());
  MCFixup Fixup = MCFixup::Create(DF->getContents().size(), SRE, FK_SecRel_2);
  DF->getFixups().push_back(Fixup);
  DF->getContents().resize(DF->getContents().size() + 4, 0);
}

void MCWinCOFFStreamer::EmitCOFFSecRel32(MCSymbol const *Symbol) {
  MCDataFragment *DF = getOrCreateDataFragment();
  const MCSymbolRefExpr *SRE = MCSymbolRefExpr::Create(Symbol, getContext());
  MCFixup Fixup = MCFixup::Create(DF->getContents().size(), SRE, FK_SecRel_4);
  DF->getFixups().push_back(Fixup);
  DF->getContents().resize(DF->getContents().size() + 4, 0);
}

void MCWinCOFFStreamer::EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
  llvm_unreachable("not supported");
}

void MCWinCOFFStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                         unsigned ByteAlignment) {
  assert((!Symbol->isInSection() ||
          Symbol->getSection().getVariant() == MCSection::SV_COFF) &&
         "Got non-COFF section in the COFF backend!");

  if (ByteAlignment > 32)
    report_fatal_error("alignment is limited to 32-bytes");

  AssignSection(Symbol, nullptr);

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);
  SD.setExternal(true);
  SD.setCommon(Size, ByteAlignment);
}

void MCWinCOFFStreamer::EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                              unsigned ByteAlignment) {
  assert(!Symbol->isInSection() && "Symbol must not already have a section!");

  const MCSection *Section = getContext().getObjectFileInfo()->getBSSSection();
  MCSectionData &SectionData = getAssembler().getOrCreateSectionData(*Section);
  if (SectionData.getAlignment() < ByteAlignment)
    SectionData.setAlignment(ByteAlignment);

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);
  SD.setExternal(false);

  AssignSection(Symbol, Section);

  if (ByteAlignment != 1)
    new MCAlignFragment(ByteAlignment, /*_Value=*/0, /*_ValueSize=*/0,
                        ByteAlignment, &SectionData);

  MCFillFragment *Fragment =
      new MCFillFragment(/*_Value=*/0, /*_ValueSize=*/0, Size, &SectionData);
  SD.setFragment(Fragment);
}

void MCWinCOFFStreamer::EmitZerofill(const MCSection *Section,
                                     MCSymbol *Symbol, uint64_t Size,
                                     unsigned ByteAlignment) {
  llvm_unreachable("not implemented");
}

void MCWinCOFFStreamer::EmitTBSSSymbol(const MCSection *Section,
                                       MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {
  llvm_unreachable("not implemented");
}

void MCWinCOFFStreamer::EmitFileDirective(StringRef Filename) {
  getAssembler().addFileName(Filename);
}

// TODO: Implement this if you want to emit .comment section in COFF obj files.
void MCWinCOFFStreamer::EmitIdent(StringRef IdentString) {
  llvm_unreachable("not implemented");
}

void MCWinCOFFStreamer::EmitWin64EHHandlerData() {
  MCStreamer::EmitWin64EHHandlerData();

  // We have to emit the unwind info now, because this directive
  // actually switches to the .xdata section!
  MCWin64EHUnwindEmitter::EmitUnwindInfo(*this, getCurrentW64UnwindInfo());
}

void MCWinCOFFStreamer::FinishImpl() {
  EmitFrames(nullptr, true);
  EmitW64Tables();
  MCObjectStreamer::FinishImpl();
}

MCSymbolData &MCWinCOFFStreamer::getOrCreateSymbolData(const MCSymbol *Symbol) {
  return getAssembler().getOrCreateSymbolData(*Symbol);
}

MCStreamer *createWinCOFFStreamer(MCContext &Context, MCAsmBackend &MAB,
                                  MCCodeEmitter &CE, raw_ostream &OS,
                                  bool RelaxAll) {
  MCWinCOFFStreamer *S = new MCWinCOFFStreamer(Context, MAB, CE, OS);
  S->getAssembler().setRelaxAll(RelaxAll);
  return S;
}
}

