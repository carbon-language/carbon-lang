//===-- MipsTargetStreamer.h - Mips Target Streamer ------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSTARGETSTREAMER_H
#define MIPSTARGETSTREAMER_H

#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {
struct Elf_Internal_ABIFlags_v0 {
  // Version of flags structure.
  uint16_t version;
  // The level of the ISA: 1-5, 32, 64.
  uint8_t isa_level;
  // The revision of ISA: 0 for MIPS V and below, 1-n otherwise.
  uint8_t isa_rev;
  // The size of general purpose registers.
  uint8_t gpr_size;
  // The size of co-processor 1 registers.
  uint8_t cpr1_size;
  // The size of co-processor 2 registers.
  uint8_t cpr2_size;
  // The floating-point ABI.
  uint8_t fp_abi;
  // Processor-specific extension.
  uint32_t isa_ext;
  // Mask of ASEs used.
  uint32_t ases;
  // Mask of general flags.
  uint32_t flags1;
  uint32_t flags2;

  Elf_Internal_ABIFlags_v0()
      : version(0), isa_level(0), isa_rev(0), gpr_size(0), cpr1_size(0),
        cpr2_size(0), fp_abi(0), isa_ext(0), ases(0), flags1(0), flags2(0) {}
};

// Values for the xxx_size bytes of an ABI flags structure.
enum {
  AFL_REG_NONE = 0x00, // No registers.
  AFL_REG_32 = 0x01,   // 32-bit registers.
  AFL_REG_64 = 0x02,   // 64-bit registers.
  AFL_REG_128 = 0x03   // 128-bit registers.
};

// Masks for the ases word of an ABI flags structure.
enum {
  AFL_ASE_DSP = 0x00000001,       // DSP ASE.
  AFL_ASE_DSPR2 = 0x00000002,     // DSP R2 ASE.
  AFL_ASE_EVA = 0x00000004,       // Enhanced VA Scheme.
  AFL_ASE_MCU = 0x00000008,       // MCU (MicroController) ASE.
  AFL_ASE_MDMX = 0x00000010,      // MDMX ASE.
  AFL_ASE_MIPS3D = 0x00000020,    // MIPS-3D ASE.
  AFL_ASE_MT = 0x00000040,        // MT ASE.
  AFL_ASE_SMARTMIPS = 0x00000080, // SmartMIPS ASE.
  AFL_ASE_VIRT = 0x00000100,      // VZ ASE.
  AFL_ASE_MSA = 0x00000200,       // MSA ASE.
  AFL_ASE_MIPS16 = 0x00000400,    // MIPS16 ASE.
  AFL_ASE_MICROMIPS = 0x00000800, // MICROMIPS ASE.
  AFL_ASE_XPA = 0x00001000        // XPA ASE.
};

// Values for the isa_ext word of an ABI flags structure.
enum {
  AFL_EXT_XLR = 1,          // RMI Xlr instruction.
  AFL_EXT_OCTEON2 = 2,      // Cavium Networks Octeon2.
  AFL_EXT_OCTEONP = 3,      // Cavium Networks OcteonP.
  AFL_EXT_LOONGSON_3A = 4,  // Loongson 3A.
  AFL_EXT_OCTEON = 5,       // Cavium Networks Octeon.
  AFL_EXT_5900 = 6,         // MIPS R5900 instruction.
  AFL_EXT_4650 = 7,         // MIPS R4650 instruction.
  AFL_EXT_4010 = 8,         // LSI R4010 instruction.
  AFL_EXT_4100 = 9,         // NEC VR4100 instruction.
  AFL_EXT_3900 = 10,        // Toshiba R3900 instruction.
  AFL_EXT_10000 = 11,       // MIPS R10000 instruction.
  AFL_EXT_SB1 = 12,         // Broadcom SB-1 instruction.
  AFL_EXT_4111 = 13,        // NEC VR4111/VR4181 instruction.
  AFL_EXT_4120 = 14,        // NEC VR4120 instruction.
  AFL_EXT_5400 = 15,        // NEC VR5400 instruction.
  AFL_EXT_5500 = 16,        // NEC VR5500 instruction.
  AFL_EXT_LOONGSON_2E = 17, // ST Microelectronics Loongson 2E.
  AFL_EXT_LOONGSON_2F = 18  // ST Microelectronics Loongson 2F.
};

// Values for the fp_abi word of an ABI flags structure.
enum {
  Val_GNU_MIPS_ABI_FP_DOUBLE = 1,
  Val_GNU_MIPS_ABI_FP_XX = 5,
  Val_GNU_MIPS_ABI_FP_64 = 6
};

class MipsTargetStreamer : public MCTargetStreamer {
public:
  MipsTargetStreamer(MCStreamer &S);
  virtual void emitDirectiveSetMicroMips();
  virtual void emitDirectiveSetNoMicroMips();
  virtual void emitDirectiveSetMips16();
  virtual void emitDirectiveSetNoMips16();

  virtual void emitDirectiveSetReorder();
  virtual void emitDirectiveSetNoReorder();
  virtual void emitDirectiveSetMacro();
  virtual void emitDirectiveSetNoMacro();
  virtual void emitDirectiveSetAt();
  virtual void emitDirectiveSetNoAt();
  virtual void emitDirectiveEnd(StringRef Name);

  virtual void emitDirectiveEnt(const MCSymbol &Symbol);
  virtual void emitDirectiveAbiCalls();
  virtual void emitDirectiveNaN2008();
  virtual void emitDirectiveNaNLegacy();
  virtual void emitDirectiveOptionPic0();
  virtual void emitDirectiveOptionPic2();
  virtual void emitFrame(unsigned StackReg, unsigned StackSize,
                         unsigned ReturnReg);
  virtual void emitMask(unsigned CPUBitmask, int CPUTopSavedRegOff);
  virtual void emitFMask(unsigned FPUBitmask, int FPUTopSavedRegOff);

  virtual void emitDirectiveSetMips32R2();
  virtual void emitDirectiveSetMips64();
  virtual void emitDirectiveSetMips64R2();
  virtual void emitDirectiveSetDsp();

  // PIC support
  virtual void emitDirectiveCpload(unsigned RegNo);
  virtual void emitDirectiveCpsetup(unsigned RegNo, int RegOrOffset,
                                    const MCSymbol &Sym, bool IsReg);
  // ABI Flags
  virtual void emitDirectiveModule(unsigned Value, bool is32BitAbi){};
  virtual void emitDirectiveSetFp(unsigned Value, bool is32BitAbi){};
  virtual void emitMipsAbiFlags(){};
  void setCanHaveModuleDir(bool Can) { canHaveModuleDirective = Can; }
  bool getCanHaveModuleDir() { return canHaveModuleDirective; }

  void setVersion(uint16_t Version) { MipsABIFlags.version = Version; }
  void setISALevel(uint8_t Level) { MipsABIFlags.isa_level = Level; }
  void setISARev(uint8_t Rev) { MipsABIFlags.isa_rev = Rev; }
  void setGprSize(uint8_t Size) { MipsABIFlags.gpr_size = Size; }
  void setCpr1Size(uint8_t Size) { MipsABIFlags.cpr1_size = Size; }
  void setCpr2Size(uint8_t Size) { MipsABIFlags.cpr2_size = Size; }
  void setFpABI(uint8_t Abi) { MipsABIFlags.fp_abi = Abi; }
  void setIsaExt(uint32_t IsaExt) { MipsABIFlags.isa_ext = IsaExt; }
  void setASEs(uint32_t Ases) { MipsABIFlags.ases = Ases; }
  void setFlags1(uint32_t Flags) { MipsABIFlags.flags1 = Flags; }
  void setFlags2(uint32_t Flags) { MipsABIFlags.flags2 = Flags; }

  uint8_t getFPAbi() { return MipsABIFlags.fp_abi; }
  // This method enables template classes to set internal abi flags
  // structure values.
  template <class PredicateLibrary>
  void updateABIInfo(const PredicateLibrary &P) {
    setVersion(0); // Version, default value is 0.

    if (P.hasMips64()) { // isa_level
      setISALevel(64);
      if (P.hasMips64r6())
        setISARev(6);
      else if (P.hasMips64r2())
        setISARev(2);
      else
        setISARev(1);
    } else if (P.hasMips32()) {
      setISALevel(32);
      if (P.hasMips32r6())
        setISARev(6);
      else if (P.hasMips32r2())
        setISARev(2);
      else
        setISARev(1);
    } else {
      setISARev(0);
      if (P.hasMips5())
        setISALevel(5);
      else if (P.hasMips4())
        setISALevel(4);
      else if (P.hasMips3())
        setISALevel(3);
      else if (P.hasMips2())
        setISALevel(2);
      else if (P.hasMips1())
        setISALevel(1);
      else
        llvm_unreachable("Unknown ISA level!");
    }

    if (P.isGP64bit()) // GPR size.
      setGprSize(AFL_REG_64);
    else
      setGprSize(AFL_REG_32);

    // TODO: check for MSA128 value.
    if (P.mipsSEUsesSoftFloat())
      setCpr1Size(AFL_REG_NONE);
    else if (P.isFP64bit())
      setCpr1Size(AFL_REG_64);
    else
      setCpr1Size(AFL_REG_32);
    setCpr2Size(AFL_REG_NONE); // Default value.

    // Set ASE.
    unsigned AseFlags = 0;
    if (P.hasDSP())
      AseFlags |= AFL_ASE_DSP;
    if (P.hasDSPR2())
      AseFlags |= AFL_ASE_DSPR2;
    if (P.hasMSA())
      AseFlags |= AFL_ASE_MSA;
    if (P.inMicroMipsMode())
      AseFlags |= AFL_ASE_MICROMIPS;
    if (P.inMips16Mode())
      AseFlags |= AFL_ASE_MIPS16;

    if (P.isABI_N32() || P.isABI_N64())
      setFpABI(Val_GNU_MIPS_ABI_FP_DOUBLE);
    else if (P.isABI_O32()) {
      if (P.isFP64bit())
        setFpABI(Val_GNU_MIPS_ABI_FP_64);
      else if (P.isABI_FPXX())
        setFpABI(Val_GNU_MIPS_ABI_FP_XX);
      else
        setFpABI(Val_GNU_MIPS_ABI_FP_DOUBLE);
    } else
      setFpABI(0); // Default value.
  }

protected:
  Elf_Internal_ABIFlags_v0 MipsABIFlags;

private:
  bool canHaveModuleDirective;
};

// This part is for ascii assembly output
class MipsTargetAsmStreamer : public MipsTargetStreamer {
  formatted_raw_ostream &OS;

public:
  MipsTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);
  void emitDirectiveSetMicroMips() override;
  void emitDirectiveSetNoMicroMips() override;
  void emitDirectiveSetMips16() override;
  void emitDirectiveSetNoMips16() override;

  void emitDirectiveSetReorder() override;
  void emitDirectiveSetNoReorder() override;
  void emitDirectiveSetMacro() override;
  void emitDirectiveSetNoMacro() override;
  void emitDirectiveSetAt() override;
  void emitDirectiveSetNoAt() override;
  void emitDirectiveEnd(StringRef Name) override;

  void emitDirectiveEnt(const MCSymbol &Symbol) override;
  void emitDirectiveAbiCalls() override;
  void emitDirectiveNaN2008() override;
  void emitDirectiveNaNLegacy() override;
  void emitDirectiveOptionPic0() override;
  void emitDirectiveOptionPic2() override;
  void emitFrame(unsigned StackReg, unsigned StackSize,
                 unsigned ReturnReg) override;
  void emitMask(unsigned CPUBitmask, int CPUTopSavedRegOff) override;
  void emitFMask(unsigned FPUBitmask, int FPUTopSavedRegOff) override;

  void emitDirectiveSetMips32R2() override;
  void emitDirectiveSetMips64() override;
  void emitDirectiveSetMips64R2() override;
  void emitDirectiveSetDsp() override;

  // PIC support
  virtual void emitDirectiveCpload(unsigned RegNo);
  void emitDirectiveCpsetup(unsigned RegNo, int RegOrOffset,
                            const MCSymbol &Sym, bool IsReg) override;

  // ABI Flags
  void emitDirectiveModule(unsigned Value, bool is32BitAbi) override;
  void emitDirectiveSetFp(unsigned Value, bool is32BitAbi) override;
  void emitMipsAbiFlags() override;
};

// This part is for ELF object output
class MipsTargetELFStreamer : public MipsTargetStreamer {
  bool MicroMipsEnabled;
  const MCSubtargetInfo &STI;
  bool Pic;

public:
  bool isMicroMipsEnabled() const { return MicroMipsEnabled; }
  MCELFStreamer &getStreamer();
  MipsTargetELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI);

  void emitLabel(MCSymbol *Symbol) override;
  void emitAssignment(MCSymbol *Symbol, const MCExpr *Value) override;
  void finish() override;

  void emitDirectiveSetMicroMips() override;
  void emitDirectiveSetNoMicroMips() override;
  void emitDirectiveSetMips16() override;
  void emitDirectiveSetNoMips16() override;

  void emitDirectiveSetReorder() override;
  void emitDirectiveSetNoReorder() override;
  void emitDirectiveSetMacro() override;
  void emitDirectiveSetNoMacro() override;
  void emitDirectiveSetAt() override;
  void emitDirectiveSetNoAt() override;
  void emitDirectiveEnd(StringRef Name) override;

  void emitDirectiveEnt(const MCSymbol &Symbol) override;
  void emitDirectiveAbiCalls() override;
  void emitDirectiveNaN2008() override;
  void emitDirectiveNaNLegacy() override;
  void emitDirectiveOptionPic0() override;
  void emitDirectiveOptionPic2() override;
  void emitFrame(unsigned StackReg, unsigned StackSize,
                 unsigned ReturnReg) override;
  void emitMask(unsigned CPUBitmask, int CPUTopSavedRegOff) override;
  void emitFMask(unsigned FPUBitmask, int FPUTopSavedRegOff) override;

  void emitDirectiveSetMips32R2() override;
  void emitDirectiveSetMips64() override;
  void emitDirectiveSetMips64R2() override;
  void emitDirectiveSetDsp() override;

  // PIC support
  virtual void emitDirectiveCpload(unsigned RegNo);
  void emitDirectiveCpsetup(unsigned RegNo, int RegOrOffset,
                            const MCSymbol &Sym, bool IsReg) override;

  // ABI Flags
  void emitMipsAbiFlags() override;

protected:
  bool isO32() const { return STI.getFeatureBits() & Mips::FeatureO32; }
  bool isN32() const { return STI.getFeatureBits() & Mips::FeatureN32; }
  bool isN64() const { return STI.getFeatureBits() & Mips::FeatureN64; }
};
}
#endif
