//===-- MipsTargetStreamer.cpp - Mips Target Streamer Methods -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Mips specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetStreamer.h"
#include "MipsMCTargetDesc.h"
#include "llvm/MC/MCELF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

// Pin vtable to this file.
void MipsTargetStreamer::anchor() {}

MipsTargetStreamer::MipsTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

MipsTargetAsmStreamer::MipsTargetAsmStreamer(MCStreamer &S,
                                             formatted_raw_ostream &OS)
    : MipsTargetStreamer(S), OS(OS) {}

void MipsTargetAsmStreamer::emitDirectiveSetMicroMips() {
  OS << "\t.set\tmicromips\n";
}

void MipsTargetAsmStreamer::emitDirectiveSetNoMicroMips() {
  OS << "\t.set\tnomicromips\n";
}

void MipsTargetAsmStreamer::emitDirectiveSetMips16() {
  OS << "\t.set\tmips16\n";
}

void MipsTargetAsmStreamer::emitDirectiveSetNoMips16() {
  OS << "\t.set\tnomips16\n";
}

void MipsTargetAsmStreamer::emitDirectiveSetReorder() {
  OS << "\t.set\treorder\n";
}

void MipsTargetAsmStreamer::emitDirectiveSetNoReorder() {
  OS << "\t.set\tnoreorder\n";
}

void MipsTargetAsmStreamer::emitDirectiveSetMacro() {
  OS << "\t.set\tmacro\n";
}

void MipsTargetAsmStreamer::emitDirectiveSetNoMacro() {
  OS << "\t.set\tnomacro\n";
}

void MipsTargetAsmStreamer::emitDirectiveSetAt() {
  OS << "\t.set\tat\n";
}

void MipsTargetAsmStreamer::emitDirectiveSetNoAt() {
  OS << "\t.set\tnoat\n";
}

void MipsTargetAsmStreamer::emitDirectiveEnd(StringRef Name) {
  OS << "\t.end\t" << Name << '\n';
}

void MipsTargetAsmStreamer::emitDirectiveEnt(const MCSymbol &Symbol) {
  OS << "\t.ent\t" << Symbol.getName() << '\n';
}

void MipsTargetAsmStreamer::emitDirectiveAbiCalls() { OS << "\t.abicalls\n"; }
void MipsTargetAsmStreamer::emitDirectiveOptionPic0() {
  OS << "\t.option\tpic0\n";
}

// This part is for ELF object output.
MipsTargetELFStreamer::MipsTargetELFStreamer(MCStreamer &S,
                                             const MCSubtargetInfo &STI)
    : MipsTargetStreamer(S), MicroMipsEnabled(false) {
  MCAssembler &MCA = getStreamer().getAssembler();
  uint64_t Features = STI.getFeatureBits();
  Triple T(STI.getTargetTriple());

  // Update e_header flags
  unsigned EFlags = 0;

  // Architecture
  if (Features & Mips::FeatureMips64r2)
    EFlags |= ELF::EF_MIPS_ARCH_64R2;
  else if (Features & Mips::FeatureMips64)
    EFlags |= ELF::EF_MIPS_ARCH_64;
  else if (Features & Mips::FeatureMips32r2)
    EFlags |= ELF::EF_MIPS_ARCH_32R2;
  else if (Features & Mips::FeatureMips32)
    EFlags |= ELF::EF_MIPS_ARCH_32;

  if (T.isArch64Bit()) {
    EFlags |= ELF::EF_MIPS_ABI2;
  } else {
    if (Features & Mips::FeatureMips64r2 || Features & Mips::FeatureMips64)
      EFlags |= ELF::EF_MIPS_32BITMODE;

    // ABI
    EFlags |= ELF::EF_MIPS_ABI_O32;
  }

  MCA.setELFHeaderEFlags(EFlags);
}

void MipsTargetELFStreamer::emitLabel(MCSymbol *Symbol) {
  if (!isMicroMipsEnabled())
    return;
  MCSymbolData &Data = getStreamer().getOrCreateSymbolData(Symbol);
  uint8_t Type = MCELF::GetType(Data);
  if (Type != ELF::STT_FUNC)
    return;

  // The "other" values are stored in the last 6 bits of the second byte
  // The traditional defines for STO values assume the full byte and thus
  // the shift to pack it.
  MCELF::setOther(Data, ELF::STO_MIPS_MICROMIPS >> 2);
}

MCELFStreamer &MipsTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

void MipsTargetELFStreamer::emitDirectiveSetMicroMips() {
  MicroMipsEnabled = true;

  MCAssembler &MCA = getStreamer().getAssembler();
  unsigned Flags = MCA.getELFHeaderEFlags();
  Flags |= ELF::EF_MIPS_MICROMIPS;
  MCA.setELFHeaderEFlags(Flags);
}

void MipsTargetELFStreamer::emitDirectiveSetNoMicroMips() {
  MicroMipsEnabled = false;
}

void MipsTargetELFStreamer::emitDirectiveSetMips16() {
  MCAssembler &MCA = getStreamer().getAssembler();
  unsigned Flags = MCA.getELFHeaderEFlags();
  Flags |= ELF::EF_MIPS_ARCH_ASE_M16;
  MCA.setELFHeaderEFlags(Flags);
}

void MipsTargetELFStreamer::emitDirectiveSetNoMips16() {
  // FIXME: implement.
}

void MipsTargetELFStreamer::emitDirectiveSetReorder() {
  // FIXME: implement.
}

void MipsTargetELFStreamer::emitDirectiveSetNoReorder() {
  MCAssembler &MCA = getStreamer().getAssembler();
  unsigned Flags = MCA.getELFHeaderEFlags();
  Flags |= ELF::EF_MIPS_NOREORDER;
  MCA.setELFHeaderEFlags(Flags);
}

void MipsTargetELFStreamer::emitDirectiveSetMacro() {
  // FIXME: implement.
}

void MipsTargetELFStreamer::emitDirectiveSetNoMacro() {
  // FIXME: implement.
}

void MipsTargetELFStreamer::emitDirectiveSetAt() {
  // FIXME: implement.
}

void MipsTargetELFStreamer::emitDirectiveSetNoAt() {
  // FIXME: implement.
}

void MipsTargetELFStreamer::emitDirectiveEnd(StringRef Name) {
  // FIXME: implement.
}

void MipsTargetELFStreamer::emitDirectiveEnt(const MCSymbol &Symbol) {
  // FIXME: implement.
}

void MipsTargetELFStreamer::emitDirectiveAbiCalls() {
  MCAssembler &MCA = getStreamer().getAssembler();
  unsigned Flags = MCA.getELFHeaderEFlags();
  Flags |= ELF::EF_MIPS_CPIC | ELF::EF_MIPS_PIC;
  MCA.setELFHeaderEFlags(Flags);
}
void MipsTargetELFStreamer::emitDirectiveOptionPic0() {
  MCAssembler &MCA = getStreamer().getAssembler();
  unsigned Flags = MCA.getELFHeaderEFlags();
  Flags &= ~ELF::EF_MIPS_PIC;
  MCA.setELFHeaderEFlags(Flags);
}
