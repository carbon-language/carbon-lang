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
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
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

  DenseMap<const MCSection*, MCSectionData*> SectionMap;
  
  DenseMap<const MCSymbol*, MCSymbolData*> SymbolMap;

private:
  MCFragment *getCurrentFragment() const {
    assert(CurSectionData && "No current section!");

    if (!CurSectionData->empty())
      return &CurSectionData->getFragmentList().back();

    return 0;
  }

  MCSymbolData &getSymbolData(MCSymbol &Symbol) {
    MCSymbolData *&Entry = SymbolMap[&Symbol];

    if (!Entry)
      Entry = new MCSymbolData(Symbol, 0, 0, &Assembler);

    return *Entry;
  }

public:
  MCMachOStreamer(MCContext &Context, raw_ostream &_OS)
    : MCStreamer(Context), Assembler(_OS), CurSectionData(0) {}
  ~MCMachOStreamer() {}

  /// @name MCStreamer Interface
  /// @{

  virtual void SwitchSection(const MCSection *Section);

  virtual void EmitLabel(MCSymbol *Symbol);

  virtual void EmitAssemblerFlag(AssemblerFlag Flag);

  virtual void EmitAssignment(MCSymbol *Symbol, const MCValue &Value,
                              bool MakeAbsolute = false);

  virtual void EmitSymbolAttribute(MCSymbol *Symbol, SymbolAttr Attribute);

  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue);

  virtual void EmitLocalSymbol(MCSymbol *Symbol, const MCValue &Value);

  virtual void EmitCommonSymbol(MCSymbol *Symbol, unsigned Size,
                                unsigned Pow2Alignment, bool IsLocal);

  virtual void EmitZerofill(MCSection *Section, MCSymbol *Symbol = NULL,
                            unsigned Size = 0, unsigned Pow2Alignment = 0);

  virtual void EmitBytes(const StringRef &Data);

  virtual void EmitValue(const MCValue &Value, unsigned Size);

  virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                    unsigned ValueSize = 1,
                                    unsigned MaxBytesToEmit = 0);

  virtual void EmitValueToOffset(const MCValue &Offset,
                                 unsigned char Value = 0);

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
  MCSectionData *&Entry = SectionMap[Section];

  if (!Entry)
    Entry = new MCSectionData(*Section, &Assembler);

  CurSectionData = Entry;
}

void MCMachOStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");

  MCDataFragment *F = dyn_cast_or_null<MCDataFragment>(getCurrentFragment());
  if (!F)
    F = new MCDataFragment(CurSectionData);

  MCSymbolData &SD = getSymbolData(*Symbol);
  assert(!SD.getFragment() && "Unexpected fragment on symbol data!");
  SD.setFragment(F);
  SD.setOffset(F->getContents().size());

  // This causes the reference type and weak reference flags to be cleared.
  SD.setFlags(SD.getFlags() & ~(SF_WeakReference | SF_ReferenceTypeMask));
  
  Symbol->setSection(*CurSection);
}

void MCMachOStreamer::EmitAssemblerFlag(AssemblerFlag Flag) {
  llvm_unreachable("FIXME: Not yet implemented!");
}

void MCMachOStreamer::EmitAssignment(MCSymbol *Symbol,
                                     const MCValue &Value,
                                     bool MakeAbsolute) {
  // Only absolute symbols can be redefined.
  assert((Symbol->isUndefined() || Symbol->isAbsolute()) &&
         "Cannot define a symbol twice!");

  llvm_unreachable("FIXME: Not yet implemented!");
}

void MCMachOStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                          SymbolAttr Attribute) {
  // Indirect symbols are handled differently, to match how 'as' handles
  // them. This makes writing matching .o files easier.
  if (Attribute == MCStreamer::IndirectSymbol) {
    // Note that we intentionally cannot use the symbol data here; this is
    // important for matching the string table that 'as' generates.
    IndirectSymbolData ISD;
    ISD.Symbol = Symbol;
    ISD.SectionData = CurSectionData;
    Assembler.getIndirectSymbols().push_back(ISD);
    return;
  }

  // Adding a symbol attribute always introduces the symbol, note that an
  // important side effect of calling getSymbolData here is to register the
  // symbol with the assembler.
  MCSymbolData &SD = getSymbolData(*Symbol);

  // The implementation of symbol attributes is designed to match 'as', but it
  // leaves much to desired. It doesn't really make sense to arbitrarily add and
  // remove flags, but 'as' allows this (in particular, see .desc).
  //
  // In the future it might be worth trying to make these operations more well
  // defined.
  switch (Attribute) {
  case MCStreamer::IndirectSymbol:
  case MCStreamer::Hidden:
  case MCStreamer::Internal:
  case MCStreamer::Protected:
  case MCStreamer::Weak:
    assert(0 && "Invalid symbol attribute for Mach-O!");
    break;

  case MCStreamer::Global:
    getSymbolData(*Symbol).setExternal(true);
    break;

  case MCStreamer::LazyReference:
    // FIXME: This requires -dynamic.
    SD.setFlags(SD.getFlags() | SF_NoDeadStrip);
    if (Symbol->isUndefined())
      SD.setFlags(SD.getFlags() | SF_ReferenceTypeUndefinedLazy);
    break;

    // Since .reference sets the no dead strip bit, it is equivalent to
    // .no_dead_strip in practice.
  case MCStreamer::Reference:
  case MCStreamer::NoDeadStrip:
    SD.setFlags(SD.getFlags() | SF_NoDeadStrip);
    break;

  case MCStreamer::PrivateExtern:
    SD.setExternal(true);
    SD.setPrivateExtern(true);
    break;

  case MCStreamer::WeakReference:
    // FIXME: This requires -dynamic.
    if (Symbol->isUndefined())
      SD.setFlags(SD.getFlags() | SF_WeakReference);
    break;

  case MCStreamer::WeakDefinition:
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
  getSymbolData(*Symbol).setFlags(DescValue & SF_DescFlagsMask);
}

void MCMachOStreamer::EmitLocalSymbol(MCSymbol *Symbol, const MCValue &Value) {
  llvm_unreachable("FIXME: Not yet implemented!");
}

void MCMachOStreamer::EmitCommonSymbol(MCSymbol *Symbol, unsigned Size,
                                       unsigned Pow2Alignment,
                                       bool IsLocal) {
  llvm_unreachable("FIXME: Not yet implemented!");
}

void MCMachOStreamer::EmitZerofill(MCSection *Section, MCSymbol *Symbol,
                                   unsigned Size, unsigned Pow2Alignment) {
  llvm_unreachable("FIXME: Not yet implemented!");
}

void MCMachOStreamer::EmitBytes(const StringRef &Data) {
  MCDataFragment *DF = dyn_cast_or_null<MCDataFragment>(getCurrentFragment());
  if (!DF)
    DF = new MCDataFragment(CurSectionData);
  DF->getContents().append(Data.begin(), Data.end());
}

void MCMachOStreamer::EmitValue(const MCValue &Value, unsigned Size) {
  new MCFillFragment(Value, Size, 1, CurSectionData);
}

void MCMachOStreamer::EmitValueToAlignment(unsigned ByteAlignment,
                                           int64_t Value, unsigned ValueSize,
                                           unsigned MaxBytesToEmit) {
  if (MaxBytesToEmit == 0)
    MaxBytesToEmit = ByteAlignment;
  new MCAlignFragment(ByteAlignment, Value, ValueSize, MaxBytesToEmit,
                      CurSectionData);

  // Update the maximum alignment on the current section if necessary.
  if (ByteAlignment > CurSectionData->getAlignment())
    CurSectionData->setAlignment(ByteAlignment);
}

void MCMachOStreamer::EmitValueToOffset(const MCValue &Offset,
                                        unsigned char Value) {
  new MCOrgFragment(Offset, Value, CurSectionData);
}

void MCMachOStreamer::EmitInstruction(const MCInst &Inst) {
  llvm_unreachable("FIXME: Not yet implemented!");
}

void MCMachOStreamer::Finish() {
  Assembler.Finish();
}

MCStreamer *llvm::createMachOStreamer(MCContext &Context, raw_ostream &OS) {
  return new MCMachOStreamer(Context, OS);
}
