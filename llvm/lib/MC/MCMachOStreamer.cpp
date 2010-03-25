//===- lib/MC/MCMachOStreamer.cpp - Mach-O Object Output ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"

#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmBackend.h"

using namespace llvm;

namespace {

class MCMachOStreamer : public MCStreamer {
  /// SymbolFlags - We store the value for the 'desc' symbol field in the lowest
  /// 16 bits of the implementation defined flags.
  enum SymbolFlags { // See <mach-o/nlist.h>.
    SF_DescFlagsMask                        = 0xFFFF,

    // Reference type flags.
    SF_ReferenceTypeMask                    = 0x0007,
    SF_ReferenceTypeUndefinedNonLazy        = 0x0000,
    SF_ReferenceTypeUndefinedLazy           = 0x0001,
    SF_ReferenceTypeDefined                 = 0x0002,
    SF_ReferenceTypePrivateDefined          = 0x0003,
    SF_ReferenceTypePrivateUndefinedNonLazy = 0x0004,
    SF_ReferenceTypePrivateUndefinedLazy    = 0x0005,

    // Other 'desc' flags.
    SF_NoDeadStrip                          = 0x0020,
    SF_WeakReference                        = 0x0040,
    SF_WeakDefinition                       = 0x0080
  };

private:
  MCAssembler Assembler;
  MCSectionData *CurSectionData;

private:
  MCFragment *getCurrentFragment() const {
    assert(CurSectionData && "No current section!");

    if (!CurSectionData->empty())
      return &CurSectionData->getFragmentList().back();

    return 0;
  }

  /// Get a data fragment to write into, creating a new one if the current
  /// fragment is not a data fragment.
  MCDataFragment *getOrCreateDataFragment() const {
    MCDataFragment *F = dyn_cast_or_null<MCDataFragment>(getCurrentFragment());
    if (!F)
      F = new MCDataFragment(CurSectionData);
    return F;
  }

public:
  MCMachOStreamer(MCContext &Context, TargetAsmBackend &TAB,
                  raw_ostream &_OS, MCCodeEmitter *_Emitter)
    : MCStreamer(Context), Assembler(Context, TAB, *_Emitter, _OS),
      CurSectionData(0) {}
  ~MCMachOStreamer() {}

  MCAssembler &getAssembler() { return Assembler; }

  const MCExpr *AddValueSymbols(const MCExpr *Value) {
    switch (Value->getKind()) {
    case MCExpr::Target: assert(0 && "Can't handle target exprs yet!");
    case MCExpr::Constant:
      break;

    case MCExpr::Binary: {
      const MCBinaryExpr *BE = cast<MCBinaryExpr>(Value);
      AddValueSymbols(BE->getLHS());
      AddValueSymbols(BE->getRHS());
      break;
    }

    case MCExpr::SymbolRef:
      Assembler.getOrCreateSymbolData(
        cast<MCSymbolRefExpr>(Value)->getSymbol());
      break;

    case MCExpr::Unary:
      AddValueSymbols(cast<MCUnaryExpr>(Value)->getSubExpr());
      break;
    }

    return Value;
  }

  /// @name MCStreamer Interface
  /// @{

  virtual void SwitchSection(const MCSection *Section);
  virtual void EmitLabel(MCSymbol *Symbol);
  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag);
  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value);
  virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute);
  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue);
  virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                unsigned ByteAlignment);
  virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
    assert(0 && "macho doesn't support this directive");
  }
  virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size) {
    assert(0 && "macho doesn't support this directive");
  }
  virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                            unsigned Size = 0, unsigned ByteAlignment = 0);
  virtual void EmitBytes(StringRef Data, unsigned AddrSpace);
  virtual void EmitValue(const MCExpr *Value, unsigned Size,unsigned AddrSpace);
  virtual void EmitGPRel32Value(const MCExpr *Value) {
    assert(0 && "macho doesn't support this directive");
  }
  virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                    unsigned ValueSize = 1,
                                    unsigned MaxBytesToEmit = 0);
  virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                 unsigned MaxBytesToEmit = 0);
  virtual void EmitValueToOffset(const MCExpr *Offset,
                                 unsigned char Value = 0);
  
  virtual void EmitFileDirective(StringRef Filename) {
    errs() << "FIXME: MCMachoStreamer:EmitFileDirective not implemented\n";
  }
  virtual void EmitDwarfFileDirective(unsigned FileNo, StringRef Filename) {
    errs() << "FIXME: MCMachoStreamer:EmitDwarfFileDirective not implemented\n";
  }
  
  virtual void EmitInstruction(const MCInst &Inst);
  virtual void Finish();

  /// @}
};

} // end anonymous namespace.

void MCMachOStreamer::SwitchSection(const MCSection *Section) {
  assert(Section && "Cannot switch to a null section!");
  
  // If already in this section, then this is a noop.
  if (Section == CurSection) return;

  CurSection = Section;
  CurSectionData = &Assembler.getOrCreateSectionData(*Section);
}

void MCMachOStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");

  // FIXME: This is wasteful, we don't necessarily need to create a data
  // fragment. Instead, we should mark the symbol as pointing into the data
  // fragment if it exists, otherwise we should just queue the label and set its
  // fragment pointer when we emit the next fragment.
  MCDataFragment *F = getOrCreateDataFragment();
  MCSymbolData &SD = Assembler.getOrCreateSymbolData(*Symbol);
  assert(!SD.getFragment() && "Unexpected fragment on symbol data!");
  SD.setFragment(F);
  SD.setOffset(F->getContents().size());

  // This causes the reference type and weak reference flags to be cleared.
  SD.setFlags(SD.getFlags() & ~(SF_WeakReference | SF_ReferenceTypeMask));
  
  Symbol->setSection(*CurSection);
}

void MCMachOStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {
  switch (Flag) {
  case MCAF_SubsectionsViaSymbols:
    Assembler.setSubsectionsViaSymbols(true);
    return;
  }

  assert(0 && "invalid assembler flag!");
}

void MCMachOStreamer::EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
  // Only absolute symbols can be redefined.
  assert((Symbol->isUndefined() || Symbol->isAbsolute()) &&
         "Cannot define a symbol twice!");

  // FIXME: Lift context changes into super class.
  // FIXME: Set associated section.
  Symbol->setValue(AddValueSymbols(Value));
}

void MCMachOStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                          MCSymbolAttr Attribute) {
  // Indirect symbols are handled differently, to match how 'as' handles
  // them. This makes writing matching .o files easier.
  if (Attribute == MCSA_IndirectSymbol) {
    // Note that we intentionally cannot use the symbol data here; this is
    // important for matching the string table that 'as' generates.
    IndirectSymbolData ISD;
    ISD.Symbol = Symbol;
    ISD.SectionData = CurSectionData;
    Assembler.getIndirectSymbols().push_back(ISD);
    return;
  }

  // Adding a symbol attribute always introduces the symbol, note that an
  // important side effect of calling getOrCreateSymbolData here is to register
  // the symbol with the assembler.
  MCSymbolData &SD = Assembler.getOrCreateSymbolData(*Symbol);

  // The implementation of symbol attributes is designed to match 'as', but it
  // leaves much to desired. It doesn't really make sense to arbitrarily add and
  // remove flags, but 'as' allows this (in particular, see .desc).
  //
  // In the future it might be worth trying to make these operations more well
  // defined.
  switch (Attribute) {
  case MCSA_Invalid:
  case MCSA_ELF_TypeFunction:
  case MCSA_ELF_TypeIndFunction:
  case MCSA_ELF_TypeObject:
  case MCSA_ELF_TypeTLS:
  case MCSA_ELF_TypeCommon:
  case MCSA_ELF_TypeNoType:
  case MCSA_IndirectSymbol:
  case MCSA_Hidden:
  case MCSA_Internal:
  case MCSA_Protected:
  case MCSA_Weak:
  case MCSA_Local:
    assert(0 && "Invalid symbol attribute for Mach-O!");
    break;

  case MCSA_Global:
    SD.setExternal(true);
    break;

  case MCSA_LazyReference:
    // FIXME: This requires -dynamic.
    SD.setFlags(SD.getFlags() | SF_NoDeadStrip);
    if (Symbol->isUndefined())
      SD.setFlags(SD.getFlags() | SF_ReferenceTypeUndefinedLazy);
    break;

    // Since .reference sets the no dead strip bit, it is equivalent to
    // .no_dead_strip in practice.
  case MCSA_Reference:
  case MCSA_NoDeadStrip:
    SD.setFlags(SD.getFlags() | SF_NoDeadStrip);
    break;

  case MCSA_PrivateExtern:
    SD.setExternal(true);
    SD.setPrivateExtern(true);
    break;

  case MCSA_WeakReference:
    // FIXME: This requires -dynamic.
    if (Symbol->isUndefined())
      SD.setFlags(SD.getFlags() | SF_WeakReference);
    break;

  case MCSA_WeakDefinition:
    // FIXME: 'as' enforces that this is defined and global. The manual claims
    // it has to be in a coalesced section, but this isn't enforced.
    SD.setFlags(SD.getFlags() | SF_WeakDefinition);
    break;
  }
}

void MCMachOStreamer::EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
  // Encode the 'desc' value into the lowest implementation defined bits.
  assert(DescValue == (DescValue & SF_DescFlagsMask) && 
         "Invalid .desc value!");
  Assembler.getOrCreateSymbolData(*Symbol).setFlags(DescValue&SF_DescFlagsMask);
}

void MCMachOStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {
  // FIXME: Darwin 'as' does appear to allow redef of a .comm by itself.
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");

  MCSymbolData &SD = Assembler.getOrCreateSymbolData(*Symbol);
  SD.setExternal(true);
  SD.setCommon(Size, ByteAlignment);
}

void MCMachOStreamer::EmitZerofill(const MCSection *Section, MCSymbol *Symbol,
                                   unsigned Size, unsigned ByteAlignment) {
  MCSectionData &SectData = Assembler.getOrCreateSectionData(*Section);

  // The symbol may not be present, which only creates the section.
  if (!Symbol)
    return;

  // FIXME: Assert that this section has the zerofill type.

  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");

  MCSymbolData &SD = Assembler.getOrCreateSymbolData(*Symbol);

  MCFragment *F = new MCZeroFillFragment(Size, ByteAlignment, &SectData);
  SD.setFragment(F);

  Symbol->setSection(*Section);

  // Update the maximum alignment on the zero fill section if necessary.
  if (ByteAlignment > SectData.getAlignment())
    SectData.setAlignment(ByteAlignment);
}

void MCMachOStreamer::EmitBytes(StringRef Data, unsigned AddrSpace) {
  getOrCreateDataFragment()->getContents().append(Data.begin(), Data.end());
}

void MCMachOStreamer::EmitValue(const MCExpr *Value, unsigned Size,
                                unsigned AddrSpace) {
  MCDataFragment *DF = getOrCreateDataFragment();

  // Avoid fixups when possible.
  int64_t AbsValue;
  if (AddValueSymbols(Value)->EvaluateAsAbsolute(AbsValue)) {
    // FIXME: Endianness assumption.
    for (unsigned i = 0; i != Size; ++i)
      DF->getContents().push_back(uint8_t(AbsValue >> (i * 8)));
  } else {
    DF->addFixup(MCAsmFixup(DF->getContents().size(), *AddValueSymbols(Value),
                            MCFixup::getKindForSize(Size)));
    DF->getContents().resize(DF->getContents().size() + Size, 0);
  }
}

void MCMachOStreamer::EmitValueToAlignment(unsigned ByteAlignment,
                                           int64_t Value, unsigned ValueSize,
                                           unsigned MaxBytesToEmit) {
  if (MaxBytesToEmit == 0)
    MaxBytesToEmit = ByteAlignment;
  new MCAlignFragment(ByteAlignment, Value, ValueSize, MaxBytesToEmit,
                      false /* EmitNops */, CurSectionData);

  // Update the maximum alignment on the current section if necessary.
  if (ByteAlignment > CurSectionData->getAlignment())
    CurSectionData->setAlignment(ByteAlignment);
}

void MCMachOStreamer::EmitCodeAlignment(unsigned ByteAlignment,
                                        unsigned MaxBytesToEmit) {
  if (MaxBytesToEmit == 0)
    MaxBytesToEmit = ByteAlignment;
  new MCAlignFragment(ByteAlignment, 0, 1, MaxBytesToEmit,
                      true /* EmitNops */, CurSectionData);

  // Update the maximum alignment on the current section if necessary.
  if (ByteAlignment > CurSectionData->getAlignment())
    CurSectionData->setAlignment(ByteAlignment);
}

void MCMachOStreamer::EmitValueToOffset(const MCExpr *Offset,
                                        unsigned char Value) {
  new MCOrgFragment(*Offset, Value, CurSectionData);
}

void MCMachOStreamer::EmitInstruction(const MCInst &Inst) {
  // Scan for values.
  for (unsigned i = 0; i != Inst.getNumOperands(); ++i)
    if (Inst.getOperand(i).isExpr())
      AddValueSymbols(Inst.getOperand(i).getExpr());

  CurSectionData->setHasInstructions(true);

  // FIXME-PERF: Common case is that we don't need to relax, encode directly
  // onto the data fragments buffers.

  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  Assembler.getEmitter().EncodeInstruction(Inst, VecOS, Fixups);
  VecOS.flush();

  // FIXME: Eliminate this copy.
  SmallVector<MCAsmFixup, 4> AsmFixups;
  for (unsigned i = 0, e = Fixups.size(); i != e; ++i) {
    MCFixup &F = Fixups[i];
    AsmFixups.push_back(MCAsmFixup(F.getOffset(), *F.getValue(),
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
  if (Assembler.getBackend().MayNeedRelaxation(Inst, AsmFixups)) {
    MCInstFragment *IF = new MCInstFragment(Inst, CurSectionData);

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
    AsmFixups[i].Offset += DF->getContents().size();
    DF->addFixup(AsmFixups[i]);
  }
  DF->getContents().append(Code.begin(), Code.end());
}

void MCMachOStreamer::Finish() {
  Assembler.Finish();
}

MCStreamer *llvm::createMachOStreamer(MCContext &Context, TargetAsmBackend &TAB,
                                      raw_ostream &OS, MCCodeEmitter *CE,
                                      bool RelaxAll) {
  MCMachOStreamer *S = new MCMachOStreamer(Context, TAB, OS, CE);
  if (RelaxAll)
    S->getAssembler().setRelaxAll(true);
  return S;
}
