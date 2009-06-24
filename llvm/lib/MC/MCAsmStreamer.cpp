//===- lib/MC/MCAsmStreamer.cpp - Text Assembly Output --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {

  class MCAsmStreamer : public MCStreamer {
    raw_ostream &OS;

    MCSection *CurSection;

  public:
    MCAsmStreamer(MCContext &Context, raw_ostream &_OS)
      : MCStreamer(Context), OS(_OS) {}
    ~MCAsmStreamer() {}

    /// @name MCStreamer Interface
    /// @{

    virtual void SwitchSection(MCSection *Section);

    virtual void EmitLabel(MCSymbol *Symbol);

    virtual void EmitAssignment(MCSymbol *Symbol, const MCValue &Value,
                                bool MakeAbsolute = false);

    virtual void EmitSymbolAttribute(MCSymbol *Symbol, SymbolAttr Attribute);

    virtual void EmitBytes(const char *Data, unsigned Length);

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

}

/// Allow printing values directly to a raw_ostream.
inline raw_ostream &operator<<(raw_ostream &os, const MCValue &Value) {
  if (Value.getSymA()) {
    os << Value.getSymA()->getName();
    if (Value.getSymB())
      os << " - " << Value.getSymB()->getName();
    if (Value.getCst())
      os << " + " << Value.getCst();
  } else {
    assert(!Value.getSymB() && "Invalid machine code value!");
    os << Value.getCst();
  }

  return os;
}

void MCAsmStreamer::SwitchSection(MCSection *Section) {
  if (Section != CurSection) {
    CurSection = Section;

    // FIXME: Really we would like the segment, flags, etc. to be separate
    // values instead of embedded in the name. Not all assemblers understand all
    // this stuff though.
    OS << ".section " << Section->getName() << "\n";
  }
}

void MCAsmStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->getSection() == 0 && "Cannot emit a symbol twice!");
  assert(CurSection && "Cannot emit before setting section!");
  assert(!getContext().GetSymbolValue(Symbol) && 
         "Cannot emit symbol which was directly assigned to!");

  OS << Symbol->getName() << ":\n";
  Symbol->setSection(CurSection);
}

void MCAsmStreamer::EmitAssignment(MCSymbol *Symbol, const MCValue &Value,
                                   bool MakeAbsolute) {
  assert(!Symbol->getSection() && "Cannot assign to a label!");

  if (MakeAbsolute) {
    OS << ".set " << Symbol->getName() << ", " << Value << '\n';
  } else {
    OS << Symbol->getName() << " = " << Value << '\n';
  }

  getContext().SetSymbolValue(Symbol, Value);
}

void MCAsmStreamer::EmitSymbolAttribute(MCSymbol *Symbol, 
                                        SymbolAttr Attribute) {
  switch (Attribute) {
  case Global: OS << ".globl"; break;
  case Hidden: OS << ".hidden"; break;
  case IndirectSymbol: OS << ".indirect_symbol"; break;
  case Internal: OS << ".internal"; break;
  case LazyReference: OS << ".lazy_reference"; break;
  case NoDeadStrip: OS << ".no_dead_strip"; break;
  case PrivateExtern: OS << ".private_extern"; break;
  case Protected: OS << ".protected"; break;
  case Reference: OS << ".reference"; break;
  case Weak: OS << ".weak"; break;
  case WeakDefinition: OS << ".weak_definition"; break;
  case WeakReference: OS << ".weak_reference"; break;
  }

  OS << ' ' << Symbol->getName() << '\n';
}

void MCAsmStreamer::EmitBytes(const char *Data, unsigned Length) {
  assert(CurSection && "Cannot emit contents before setting section!");
  for (unsigned i = 0; i != Length; ++i)
    OS << ".byte " << (unsigned) Data[i] << '\n';
}

void MCAsmStreamer::EmitValue(const MCValue &Value, unsigned Size) {
  assert(CurSection && "Cannot emit contents before setting section!");
  // Need target hooks to know how to print this.
  switch (Size) {
  default:
    assert(0 && "Invalid size for machine code value!");
  case 1: OS << ".byte"; break;
  case 2: OS << ".short"; break;
  case 4: OS << ".long"; break;
  case 8: OS << ".quad"; break;
  }

  OS << ' ' << Value << '\n';
}

void MCAsmStreamer::EmitValueToAlignment(unsigned ByteAlignment, int64_t Value,
                                         unsigned ValueSize,
                                         unsigned MaxBytesToEmit) {
  unsigned Pow2 = Log2_32(ByteAlignment);
  assert((1U << Pow2) == ByteAlignment && "Invalid alignment!");

  switch (ValueSize) {
  default:
    assert(0 && "Invalid size for machine code value!");
  case 8:
    assert(0 && "Unsupported alignment size!");
  case 1: OS << ".p2align"; break;
  case 2: OS << ".p2alignw"; break;
  case 4: OS << ".p2alignl"; break;
  }

  OS << ' ' << Pow2;

  OS << ", " << Value;
  if (MaxBytesToEmit) 
    OS << ", " << MaxBytesToEmit;
  OS << '\n';
}

void MCAsmStreamer::EmitValueToOffset(const MCValue &Offset, 
                                      unsigned char Value) {
  // FIXME: Verify that Offset is associated with the current section.
  OS << ".org " << Offset << ", " << (unsigned) Value << '\n';
}

void MCAsmStreamer::EmitInstruction(const MCInst &Inst) {
  assert(CurSection && "Cannot emit contents before setting section!");
  // FIXME: Implement.
  OS << "# FIXME: Implement instruction printing!\n";
}

void MCAsmStreamer::Finish() {
  OS.flush();
}
    
MCStreamer *llvm::createAsmStreamer(MCContext &Context, raw_ostream &OS) {
  return new MCAsmStreamer(Context, OS);
}
