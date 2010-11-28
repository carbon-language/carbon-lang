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
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmBackend.h"
#include "llvm/ADT/StringMap.h"

#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
class WinCOFFStreamer : public MCObjectStreamer {
public:
  MCSymbol const *CurSymbol;

  WinCOFFStreamer(MCContext &Context,
                  TargetAsmBackend &TAB,
                  MCCodeEmitter &CE,
                  raw_ostream &OS);

  void AddCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                       unsigned ByteAlignment, bool External);

  // MCStreamer interface

  virtual void InitSections();
  virtual void EmitLabel(MCSymbol *Symbol);
  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag);
  virtual void EmitThumbFunc(MCSymbol *Func);
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
  virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value,
                                   unsigned ValueSize, unsigned MaxBytesToEmit);
  virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                 unsigned MaxBytesToEmit);
  virtual void EmitValueToOffset(const MCExpr *Offset, unsigned char Value);
  virtual void EmitFileDirective(StringRef Filename);
  virtual void EmitInstruction(const MCInst &Instruction);
  virtual void Finish();

private:
  virtual void EmitInstToFragment(const MCInst &Inst) {
    llvm_unreachable("Not used by WinCOFF.");
  }
  virtual void EmitInstToData(const MCInst &Inst) {
    llvm_unreachable("Not used by WinCOFF.");
  }

  void SetSection(StringRef Section,
                  unsigned Characteristics,
                  SectionKind Kind) {
    SwitchSection(getContext().getCOFFSection(Section, Characteristics, Kind));
  }

  void SetSectionText() {
    SetSection(".text",
               COFF::IMAGE_SCN_CNT_CODE
             | COFF::IMAGE_SCN_MEM_EXECUTE
             | COFF::IMAGE_SCN_MEM_READ,
               SectionKind::getText());
    EmitCodeAlignment(4, 0);
  }

  void SetSectionData() {
    SetSection(".data",
               COFF::IMAGE_SCN_CNT_INITIALIZED_DATA
             | COFF::IMAGE_SCN_MEM_READ
             | COFF::IMAGE_SCN_MEM_WRITE,
               SectionKind::getDataRel());
    EmitCodeAlignment(4, 0);
  }

  void SetSectionBSS() {
    SetSection(".bss",
               COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA
             | COFF::IMAGE_SCN_MEM_READ
             | COFF::IMAGE_SCN_MEM_WRITE,
               SectionKind::getBSS());
    EmitCodeAlignment(4, 0);
  }

};
} // end anonymous namespace.

WinCOFFStreamer::WinCOFFStreamer(MCContext &Context,
                                 TargetAsmBackend &TAB,
                                 MCCodeEmitter &CE,
                                 raw_ostream &OS)
    : MCObjectStreamer(Context, TAB, OS, &CE, true)
    , CurSymbol(NULL) {
}

void WinCOFFStreamer::AddCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                      unsigned ByteAlignment, bool External) {
  assert(!Symbol->isInSection() && "Symbol must not already have a section!");

  std::string SectionName(".bss$linkonce");
  SectionName.append(Symbol->getName().begin(), Symbol->getName().end());

  MCSymbolData &SymbolData = getAssembler().getOrCreateSymbolData(*Symbol);

  unsigned Characteristics =
    COFF::IMAGE_SCN_LNK_COMDAT |
    COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA |
    COFF::IMAGE_SCN_MEM_READ |
    COFF::IMAGE_SCN_MEM_WRITE;

  int Selection = COFF::IMAGE_COMDAT_SELECT_LARGEST;

  const MCSection *Section = MCStreamer::getContext().getCOFFSection(
    SectionName, Characteristics, Selection, SectionKind::getBSS());

  MCSectionData &SectionData = getAssembler().getOrCreateSectionData(*Section);

  if (SectionData.getAlignment() < ByteAlignment)
    SectionData.setAlignment(ByteAlignment);

  SymbolData.setExternal(External);

  Symbol->setSection(*Section);

  if (ByteAlignment != 1)
      new MCAlignFragment(ByteAlignment, 0, 0, ByteAlignment, &SectionData);

  SymbolData.setFragment(new MCFillFragment(0, 0, Size, &SectionData));
}

// MCStreamer interface

void WinCOFFStreamer::InitSections() {
  SetSectionText();
  SetSectionData();
  SetSectionBSS();
  SetSectionText();
}

void WinCOFFStreamer::EmitLabel(MCSymbol *Symbol) {
  // TODO: This is copied almost exactly from the MachOStreamer. Consider
  // merging into MCObjectStreamer?
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");
  assert(!Symbol->isVariable() && "Cannot emit a variable symbol!");
  assert(CurSection && "Cannot emit before setting section!");

  Symbol->setSection(*CurSection);

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  // FIXME: This is wasteful, we don't necessarily need to create a data
  // fragment. Instead, we should mark the symbol as pointing into the data
  // fragment if it exists, otherwise we should just queue the label and set its
  // fragment pointer when we emit the next fragment.
  MCDataFragment *DF = getOrCreateDataFragment();

  assert(!SD.getFragment() && "Unexpected fragment on symbol data!");
  SD.setFragment(DF);
  SD.setOffset(DF->getContents().size());
}

void WinCOFFStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {
  llvm_unreachable("not implemented");
}

void WinCOFFStreamer::EmitThumbFunc(MCSymbol *Func) {
  llvm_unreachable("not implemented");
}

void WinCOFFStreamer::EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
  assert((Symbol->isInSection()
         ? Symbol->getSection().getVariant() == MCSection::SV_COFF
         : true) && "Got non COFF section in the COFF backend!");
  // FIXME: This is all very ugly and depressing. What needs to happen here
  // depends on quite a few things that are all part of relaxation, which we
  // don't really even do.

  if (Value->getKind() != MCExpr::SymbolRef) {
    // TODO: This is exactly the same as MachOStreamer. Consider merging into
    // MCObjectStreamer.
    getAssembler().getOrCreateSymbolData(*Symbol);
    AddValueSymbols(Value);
    Symbol->setVariableValue(Value);
  } else {
    // FIXME: This is a horrible way to do this :(. This should really be
    // handled after we are done with the MC* objects and immediately before
    // writing out the object file when we know exactly what the symbol should
    // look like in the coff symbol table. I'm not doing that now because the
    // COFF object writer doesn't have a clearly defined separation between MC
    // data structures, the object writers data structures, and the raw, POD,
    // data structures that get written to disk.

    // Copy over the aliased data.
    MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);
    const MCSymbolData &RealSD = getAssembler().getOrCreateSymbolData(
      dyn_cast<const MCSymbolRefExpr>(Value)->getSymbol());

    // FIXME: This is particularly nasty because it breaks as soon as any data
    // members of MCSymbolData change.
    SD.CommonAlign     = RealSD.CommonAlign;
    SD.CommonSize      = RealSD.CommonSize;
    SD.Flags           = RealSD.Flags;
    SD.Fragment        = RealSD.Fragment;
    SD.Index           = RealSD.Index;
    SD.IsExternal      = RealSD.IsExternal;
    SD.IsPrivateExtern = RealSD.IsPrivateExtern;
    SD.Offset          = RealSD.Offset;
    SD.SymbolSize      = RealSD.SymbolSize;
  }
}

void WinCOFFStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                          MCSymbolAttr Attribute) {
  assert(Symbol && "Symbol must be non-null!");
  assert((Symbol->isInSection()
         ? Symbol->getSection().getVariant() == MCSection::SV_COFF
         : true) && "Got non COFF section in the COFF backend!");
  switch (Attribute) {
  case MCSA_WeakReference:
  case MCSA_Weak: {
      MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);
      SD.modifyFlags(COFF::SF_WeakExternal, COFF::SF_WeakExternal);
      SD.setExternal(true);
    }
    break;

  case MCSA_Global:
    getAssembler().getOrCreateSymbolData(*Symbol).setExternal(true);
    break;

  default:
    llvm_unreachable("unsupported attribute");
    break;
  }
}

void WinCOFFStreamer::EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
  llvm_unreachable("not implemented");
}

void WinCOFFStreamer::BeginCOFFSymbolDef(MCSymbol const *Symbol) {
  assert((Symbol->isInSection()
         ? Symbol->getSection().getVariant() == MCSection::SV_COFF
         : true) && "Got non COFF section in the COFF backend!");
  assert(CurSymbol == NULL && "EndCOFFSymbolDef must be called between calls "
                              "to BeginCOFFSymbolDef!");
  CurSymbol = Symbol;
}

void WinCOFFStreamer::EmitCOFFSymbolStorageClass(int StorageClass) {
  assert(CurSymbol != NULL && "BeginCOFFSymbolDef must be called first!");
  assert((StorageClass & ~0xFF) == 0 && "StorageClass must only have data in "
                                        "the first byte!");

  getAssembler().getOrCreateSymbolData(*CurSymbol).modifyFlags(
    StorageClass << COFF::SF_ClassShift,
    COFF::SF_ClassMask);
}

void WinCOFFStreamer::EmitCOFFSymbolType(int Type) {
  assert(CurSymbol != NULL && "BeginCOFFSymbolDef must be called first!");
  assert((Type & ~0xFFFF) == 0 && "Type must only have data in the first 2 "
                                  "bytes");

  getAssembler().getOrCreateSymbolData(*CurSymbol).modifyFlags(
    Type << COFF::SF_TypeShift,
    COFF::SF_TypeMask);
}

void WinCOFFStreamer::EndCOFFSymbolDef() {
  assert(CurSymbol != NULL && "BeginCOFFSymbolDef must be called first!");
  CurSymbol = NULL;
}

void WinCOFFStreamer::EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
  llvm_unreachable("not implemented");
}

void WinCOFFStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {
  assert((Symbol->isInSection()
         ? Symbol->getSection().getVariant() == MCSection::SV_COFF
         : true) && "Got non COFF section in the COFF backend!");
  AddCommonSymbol(Symbol, Size, ByteAlignment, true);
}

void WinCOFFStreamer::EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size) {
  assert((Symbol->isInSection()
         ? Symbol->getSection().getVariant() == MCSection::SV_COFF
         : true) && "Got non COFF section in the COFF backend!");
  AddCommonSymbol(Symbol, Size, 1, false);
}

void WinCOFFStreamer::EmitZerofill(const MCSection *Section, MCSymbol *Symbol,
                                   unsigned Size,unsigned ByteAlignment) {
  llvm_unreachable("not implemented");
}

void WinCOFFStreamer::EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                                     uint64_t Size, unsigned ByteAlignment) {
  llvm_unreachable("not implemented");
}

void WinCOFFStreamer::EmitBytes(StringRef Data, unsigned AddrSpace) {
  // TODO: This is copied exactly from the MachOStreamer. Consider merging into
  // MCObjectStreamer?
  getOrCreateDataFragment()->getContents().append(Data.begin(), Data.end());
}

void WinCOFFStreamer::EmitValue(const MCExpr *Value, unsigned Size,
                                unsigned AddrSpace) {
  assert(AddrSpace == 0 && "Address space must be 0!");

  // TODO: This is copied exactly from the MachOStreamer. Consider merging into
  // MCObjectStreamer?
  MCDataFragment *DF = getOrCreateDataFragment();

  // Avoid fixups when possible.
  int64_t AbsValue;
  if (AddValueSymbols(Value)->EvaluateAsAbsolute(AbsValue)) {
    // FIXME: Endianness assumption.
    for (unsigned i = 0; i != Size; ++i)
      DF->getContents().push_back(uint8_t(AbsValue >> (i * 8)));
  } else {
    DF->addFixup(MCFixup::Create(DF->getContents().size(),
                                 AddValueSymbols(Value),
                                 MCFixup::getKindForSize(Size, false)));
    DF->getContents().resize(DF->getContents().size() + Size, 0);
  }
}

void WinCOFFStreamer::EmitValueToAlignment(unsigned ByteAlignment,
                                           int64_t Value,
                                           unsigned ValueSize,
                                           unsigned MaxBytesToEmit) {
  // TODO: This is copied exactly from the MachOStreamer. Consider merging into
  // MCObjectStreamer?
  if (MaxBytesToEmit == 0)
    MaxBytesToEmit = ByteAlignment;
  new MCAlignFragment(ByteAlignment, Value, ValueSize, MaxBytesToEmit,
                      getCurrentSectionData());

  // Update the maximum alignment on the current section if necessary.
  if (ByteAlignment > getCurrentSectionData()->getAlignment())
    getCurrentSectionData()->setAlignment(ByteAlignment);
}

void WinCOFFStreamer::EmitCodeAlignment(unsigned ByteAlignment,
                                        unsigned MaxBytesToEmit) {
  // TODO: This is copied exactly from the MachOStreamer. Consider merging into
  // MCObjectStreamer?
  if (MaxBytesToEmit == 0)
    MaxBytesToEmit = ByteAlignment;
  MCAlignFragment *F = new MCAlignFragment(ByteAlignment, 0, 1, MaxBytesToEmit,
                                           getCurrentSectionData());
  F->setEmitNops(true);

  // Update the maximum alignment on the current section if necessary.
  if (ByteAlignment > getCurrentSectionData()->getAlignment())
    getCurrentSectionData()->setAlignment(ByteAlignment);
}

void WinCOFFStreamer::EmitValueToOffset(const MCExpr *Offset,
                                        unsigned char Value) {
  llvm_unreachable("not implemented");
}

void WinCOFFStreamer::EmitFileDirective(StringRef Filename) {
  // Ignore for now, linkers don't care, and proper debug
  // info will be a much large effort.
}

void WinCOFFStreamer::EmitInstruction(const MCInst &Instruction) {
  for (unsigned i = 0, e = Instruction.getNumOperands(); i != e; ++i)
    if (Instruction.getOperand(i).isExpr())
      AddValueSymbols(Instruction.getOperand(i).getExpr());

  getCurrentSectionData()->setHasInstructions(true);

  MCInstFragment *Fragment =
    new MCInstFragment(Instruction, getCurrentSectionData());

  raw_svector_ostream VecOS(Fragment->getCode());

  getAssembler().getEmitter().EncodeInstruction(Instruction, VecOS,
                                                Fragment->getFixups());
}

void WinCOFFStreamer::Finish() {
  MCObjectStreamer::Finish();
}

namespace llvm
{
  MCStreamer *createWinCOFFStreamer(MCContext &Context,
                                    TargetAsmBackend &TAB,
                                    MCCodeEmitter &CE,
                                    raw_ostream &OS,
                                    bool RelaxAll) {
    WinCOFFStreamer *S = new WinCOFFStreamer(Context, TAB, CE, OS);
    S->getAssembler().setRelaxAll(RelaxAll);
    return S;
  }
}
