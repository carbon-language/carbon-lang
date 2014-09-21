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

#include "llvm/ADT/StringExtras.h"
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

  if (CurSymbol)
    FatalError("starting a new symbol definition without completing the "
               "previous one");
  CurSymbol = Symbol;
}

void MCWinCOFFStreamer::EmitCOFFSymbolStorageClass(int StorageClass) {
  if (!CurSymbol)
    FatalError("storage class specified outside of symbol definition");

  if (StorageClass & ~0xff)
    FatalError(Twine("storage class value '") + itostr(StorageClass) +
               "' out of range");

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*CurSymbol);
  SD.modifyFlags(StorageClass << COFF::SF_ClassShift, COFF::SF_ClassMask);
}

void MCWinCOFFStreamer::EmitCOFFSymbolType(int Type) {
  if (!CurSymbol)
    FatalError("symbol type specified outside of a symbol definition");

  if (Type & ~0xffff)
    FatalError(Twine("type value '") + itostr(Type) + "' out of range");

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*CurSymbol);
  SD.modifyFlags(Type << COFF::SF_TypeShift, COFF::SF_TypeMask);
}

void MCWinCOFFStreamer::EndCOFFSymbolDef() {
  if (!CurSymbol)
    FatalError("ending symbol definition without starting one");
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

  const Triple &T = getContext().getObjectFileInfo()->getTargetTriple();
  if (T.isKnownWindowsMSVCEnvironment()) {
    if (ByteAlignment > 32)
      report_fatal_error("alignment is limited to 32-bytes");
  } else {
    // The bfd linker from binutils only supports alignments less than 4 bytes
    // without inserting -aligncomm arguments into the .drectve section.
    // TODO: Support inserting -aligncomm into the .drectve section.
    if (ByteAlignment > 4)
      report_fatal_error("alignment is limited to 4-bytes");
  }
  // Round size up to alignment so that we will honor the alignment request.
  // TODO: We don't need to do this if we are targeting the bfd linker once we
  // add support for adding -aligncomm into the .drectve section.
  Size = std::max(Size, static_cast<uint64_t>(ByteAlignment));

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

void MCWinCOFFStreamer::EmitWinEHHandlerData() {
  llvm_unreachable("not implemented");
}

void MCWinCOFFStreamer::FinishImpl() {
  MCObjectStreamer::FinishImpl();
}

LLVM_ATTRIBUTE_NORETURN
void MCWinCOFFStreamer::FatalError(const Twine &Msg) const {
  getContext().FatalError(SMLoc(), Msg);
}
}

