//===- lib/MC/MCAsmStreamer.cpp - Text Assembly Output --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"

#include "llvm/MC/MCAtom.h"
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
  // FIXME: We need to enforce that we aren't printing atoms which are more
  // complicated than the assembler understands.
  //assert(Symbol->getAtom()->getSection() == CurSection && 
  //       "The label for a symbol must match its section!");
  OS << Symbol->getName() << ":\n";
}

void MCAsmStreamer::EmitAssignment(MCSymbol *Symbol, const MCValue &Value,
                                   bool MakeAbsolute) {
  if (MakeAbsolute) {
    OS << ".set " << Symbol->getName() << ", " << Value << '\n';
  } else {
    OS << Symbol->getName() << " = " << Value << '\n';
  }
}

void MCAsmStreamer::EmitSymbolAttribute(MCSymbol *Symbol, 
                                        SymbolAttr Attribute) {
  switch (Attribute) {
  case Global: OS << ".globl"; break;
  case Weak: OS << ".weak"; break;
  case PrivateExtern: OS << ".private_extern"; break;
  }

  OS << ' ' << Symbol->getName() << '\n';
}

void MCAsmStreamer::EmitBytes(const char *Data, unsigned Length) {
  for (unsigned i = 0; i != Length; ++i) {
    OS << ".byte " << (unsigned) Data[i] << '\n';
  }
}

void MCAsmStreamer::EmitValue(const MCValue &Value, unsigned Size) {
  // Need target hooks to know how to print this.
  switch (Size) {
  default:
    assert(0 && "Invalid size for machine code value!");
  case 1: OS << ".byte"; break;
  case 2: OS << ".hword"; break;
  case 4: OS << ".long"; break;
  case 8: OS << ".quad"; break;
  }

  OS << ' ' << Value << '\n';
}

void MCAsmStreamer::EmitInstruction(const MCInst &Inst) {
  // FIXME: Implement.
  OS << "# FIXME: Implement instruction printing!\n";
}

void MCAsmStreamer::Finish() {
  OS.flush();
}
    
MCStreamer *llvm::createAsmStreamer(MCContext &Context, raw_ostream &OS) {
  return new MCAsmStreamer(Context, OS);
}
