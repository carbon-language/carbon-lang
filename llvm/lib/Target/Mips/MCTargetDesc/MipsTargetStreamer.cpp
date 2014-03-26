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

#include "InstPrinter/MipsInstPrinter.h"
#include "MipsMCTargetDesc.h"
#include "MipsTargetObjectFile.h"
#include "MipsTargetStreamer.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELF.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
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

void MipsTargetAsmStreamer::emitDirectiveOptionPic2() {
  OS << "\t.option\tpic2\n";
}

void MipsTargetAsmStreamer::emitFrame(unsigned StackReg, unsigned StackSize,
                                      unsigned ReturnReg) {
  OS << "\t.frame\t$"
     << StringRef(MipsInstPrinter::getRegisterName(StackReg)).lower() << ","
     << StackSize << ",$"
     << StringRef(MipsInstPrinter::getRegisterName(ReturnReg)).lower() << '\n';
}

void MipsTargetAsmStreamer::emitDirectiveSetMips32R2() {
  OS << "\t.set\tmips32r2\n";
}

void MipsTargetAsmStreamer::emitDirectiveSetMips64R2() {
  OS << "\t.set\tmips64r2\n";
}

void MipsTargetAsmStreamer::emitDirectiveSetDsp() {
  OS << "\t.set\tdsp\n";
}
// Print a 32 bit hex number with all numbers.
static void printHex32(unsigned Value, raw_ostream &OS) {
  OS << "0x";
  for (int i = 7; i >= 0; i--)
    OS.write_hex((Value & (0xF << (i*4))) >> (i*4));
}

void MipsTargetAsmStreamer::emitMask(unsigned CPUBitmask,
                                     int CPUTopSavedRegOff) {
  OS << "\t.mask \t";
  printHex32(CPUBitmask, OS);
  OS << ',' << CPUTopSavedRegOff << '\n';
}

void MipsTargetAsmStreamer::emitFMask(unsigned FPUBitmask,
                                      int FPUTopSavedRegOff) {
  OS << "\t.fmask\t";
  printHex32(FPUBitmask, OS);
  OS << "," << FPUTopSavedRegOff << '\n';
}

// This part is for ELF object output.
MipsTargetELFStreamer::MipsTargetELFStreamer(MCStreamer &S,
                                             const MCSubtargetInfo &STI)
    : MipsTargetStreamer(S), MicroMipsEnabled(false), STI(STI) {
  MCAssembler &MCA = getStreamer().getAssembler();
  uint64_t Features = STI.getFeatureBits();
  Triple T(STI.getTargetTriple());
  Pic = (MCA.getContext().getObjectFileInfo()->getRelocM() ==  Reloc::PIC_)
            ? true
            : false;

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
    if (Features & Mips::FeatureN32)
      EFlags |= ELF::EF_MIPS_ABI2;
    else if (Features & Mips::FeatureO32) {
      EFlags |= ELF::EF_MIPS_ABI_O32;
      EFlags |= ELF::EF_MIPS_32BITMODE; /* Compatibility Mode */
    }
    // No need to set any bit for N64 which is the default ABI at the moment
    // for 64-bit Mips architectures.
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

void MipsTargetELFStreamer::finish() {
  MCAssembler &MCA = getStreamer().getAssembler();
  MCContext &Context = MCA.getContext();
  MCStreamer &OS = getStreamer();
  Triple T(STI.getTargetTriple());
  uint64_t Features = STI.getFeatureBits();

  if (T.isArch64Bit() && (Features & Mips::FeatureN64)) {
    const MCSectionELF *Sec = Context.getELFSection(
        ".MIPS.options", ELF::SHT_MIPS_OPTIONS,
        ELF::SHF_ALLOC | ELF::SHF_MIPS_NOSTRIP, SectionKind::getMetadata());
    OS.SwitchSection(Sec);

    OS.EmitIntValue(1, 1); // kind
    OS.EmitIntValue(40, 1); // size
    OS.EmitIntValue(0, 2); // section
    OS.EmitIntValue(0, 4); // info
    OS.EmitIntValue(0, 4); // ri_gprmask
    OS.EmitIntValue(0, 4); // pad
    OS.EmitIntValue(0, 4); // ri_cpr[0]mask
    OS.EmitIntValue(0, 4); // ri_cpr[1]mask
    OS.EmitIntValue(0, 4); // ri_cpr[2]mask
    OS.EmitIntValue(0, 4); // ri_cpr[3]mask
    OS.EmitIntValue(0, 8); // ri_gp_value
  } else {
    const MCSectionELF *Sec =
        Context.getELFSection(".reginfo", ELF::SHT_MIPS_REGINFO, ELF::SHF_ALLOC,
                              SectionKind::getMetadata());
    OS.SwitchSection(Sec);

    OS.EmitIntValue(0, 4); // ri_gprmask
    OS.EmitIntValue(0, 4); // ri_cpr[0]mask
    OS.EmitIntValue(0, 4); // ri_cpr[1]mask
    OS.EmitIntValue(0, 4); // ri_cpr[2]mask
    OS.EmitIntValue(0, 4); // ri_cpr[3]mask
    OS.EmitIntValue(0, 4); // ri_gp_value
  }
}

void MipsTargetELFStreamer::emitAssignment(MCSymbol *Symbol,
                                           const MCExpr *Value) {
  // If on rhs is micromips symbol then mark Symbol as microMips.
  if (Value->getKind() != MCExpr::SymbolRef)
    return;
  const MCSymbol &RhsSym =
    static_cast<const MCSymbolRefExpr *>(Value)->getSymbol();
  MCSymbolData &Data = getStreamer().getOrCreateSymbolData(&RhsSym);
  uint8_t Type = MCELF::GetType(Data);
  if ((Type != ELF::STT_FUNC)
      || !(MCELF::getOther(Data) & (ELF::STO_MIPS_MICROMIPS >> 2)))
    return;

  MCSymbolData &SymbolData = getStreamer().getOrCreateSymbolData(Symbol);
  // The "other" values are stored in the last 6 bits of the second byte.
  // The traditional defines for STO values assume the full byte and thus
  // the shift to pack it.
  MCELF::setOther(SymbolData, ELF::STO_MIPS_MICROMIPS >> 2);
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
  // This option overrides other PIC options like -KPIC.
  Pic = false;
  Flags &= ~ELF::EF_MIPS_PIC;
  MCA.setELFHeaderEFlags(Flags);
}

void MipsTargetELFStreamer::emitDirectiveOptionPic2() {
  MCAssembler &MCA = getStreamer().getAssembler();
  unsigned Flags = MCA.getELFHeaderEFlags();
  Pic = true;
  // NOTE: We are following the GAS behaviour here which means the directive
  // 'pic2' also sets the CPIC bit in the ELF header. This is different from
  // what is stated in the SYSV ABI which consider the bits EF_MIPS_PIC and
  // EF_MIPS_CPIC to be mutually exclusive.
  Flags |= ELF::EF_MIPS_PIC | ELF::EF_MIPS_CPIC;
  MCA.setELFHeaderEFlags(Flags);
}

void MipsTargetELFStreamer::emitFrame(unsigned StackReg, unsigned StackSize,
                                      unsigned ReturnReg) {
  // FIXME: implement.
}

void MipsTargetELFStreamer::emitMask(unsigned CPUBitmask,
                                     int CPUTopSavedRegOff) {
  // FIXME: implement.
}

void MipsTargetELFStreamer::emitFMask(unsigned FPUBitmask,
                                      int FPUTopSavedRegOff) {
  // FIXME: implement.
}

void MipsTargetELFStreamer::emitDirectiveSetMips32R2() {
  // No action required for ELF output.
}

void MipsTargetELFStreamer::emitDirectiveSetMips64R2() {
  // No action required for ELF output.
}

void MipsTargetELFStreamer::emitDirectiveSetDsp() {
  // No action required for ELF output.
}
