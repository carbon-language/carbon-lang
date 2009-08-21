//===- lib/MC/MCMachOStreamer.cpp - Mach-O Object Output ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

namespace {

class MCMachOStreamer : public MCStreamer {
  MCAssembler Assembler;

  MCSectionData *CurSectionData;

  DenseMap<const MCSection*, MCSectionData*> SectionMap;

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

  if (Section != CurSection) {
    CurSection = Section;
    MCSectionData *&Entry = SectionMap[Section];

    if (!Entry)
      Entry = new MCSectionData(*Section, &Assembler);

    CurSectionData = Entry;
  }
}

void MCMachOStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->getSection() == 0 && "Cannot emit a symbol twice!");
  assert(CurSection && "Cannot emit before setting section!");
  assert(!getContext().GetSymbolValue(Symbol) &&
         "Cannot emit symbol which was directly assigned to!");

  llvm_unreachable("FIXME: Not yet implemented!");

  Symbol->setSection(CurSection);
  Symbol->setExternal(false);
}

void MCMachOStreamer::EmitAssemblerFlag(AssemblerFlag Flag) {
  llvm_unreachable("FIXME: Not yet implemented!");
}

void MCMachOStreamer::EmitAssignment(MCSymbol *Symbol,
                                     const MCValue &Value,
                                     bool MakeAbsolute) {
  assert(!Symbol->getSection() && "Cannot assign to a label!");

  llvm_unreachable("FIXME: Not yet implemented!");
}

void MCMachOStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                          SymbolAttr Attribute) {
  llvm_unreachable("FIXME: Not yet implemented!");
}

void MCMachOStreamer::EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
  llvm_unreachable("FIXME: Not yet implemented!");
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
  MCDataFragment *DF = new MCDataFragment(CurSectionData);
  DF->getContents().append(Data.begin(), Data.end());
}

void MCMachOStreamer::EmitValue(const MCValue &Value, unsigned Size) {
  new MCFillFragment(Value, Size, 1, CurSectionData);
}

void MCMachOStreamer::EmitValueToAlignment(unsigned ByteAlignment,
                                           int64_t Value, unsigned ValueSize,
                                           unsigned MaxBytesToEmit) {
  new MCAlignFragment(ByteAlignment, Value, ValueSize, MaxBytesToEmit,
                      CurSectionData);

  // Update the maximum alignment on the current section if necessary
  if (ByteAlignment > CurSectionData->getAlignment())
    CurSectionData->setAlignment(ByteAlignment);
}

void MCMachOStreamer::EmitValueToOffset(const MCValue &Offset,
                                        unsigned char Value) {
  new MCOrgFragment(Offset, Value, 1, CurSectionData);
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
