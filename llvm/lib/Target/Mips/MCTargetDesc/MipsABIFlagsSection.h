//===-- MipsABIFlagsSection.h - Mips ELF ABI Flags Section -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSABIFLAGSSECTION_H
#define MIPSABIFLAGSSECTION_H

#include "llvm/MC/MCStreamer.h"

namespace llvm {

class MCStreamer;

struct MipsABIFlagsSection {
  // Values for the xxx_size bytes of an ABI flags structure.
  enum AFL_REG {
    AFL_REG_NONE = 0x00, // No registers.
    AFL_REG_32 = 0x01,   // 32-bit registers.
    AFL_REG_64 = 0x02,   // 64-bit registers.
    AFL_REG_128 = 0x03   // 128-bit registers.
  };

  // Masks for the ases word of an ABI flags structure.
  enum AFL_ASE {
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
  enum AFL_EXT {
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
  enum Val_GNU_MIPS_ABI {
    Val_GNU_MIPS_ABI_FP_ANY = 0,
    Val_GNU_MIPS_ABI_FP_DOUBLE = 1,
    Val_GNU_MIPS_ABI_FP_XX = 5,
    Val_GNU_MIPS_ABI_FP_64 = 6,
    Val_GNU_MIPS_ABI_FP_64A = 7
  };

  enum AFL_FLAGS1 {
    AFL_FLAGS1_ODDSPREG = 1
  };

  // Internal representation of the values used in .module fp=value
  enum class FpABIKind { ANY, XX, S32, S64 };

  // Version of flags structure.
  uint16_t Version;
  // The level of the ISA: 1-5, 32, 64.
  uint8_t ISALevel;
  // The revision of ISA: 0 for MIPS V and below, 1-n otherwise.
  uint8_t ISARevision;
  // The size of general purpose registers.
  AFL_REG GPRSize;
  // The size of co-processor 1 registers.
  AFL_REG CPR1Size;
  // The size of co-processor 2 registers.
  AFL_REG CPR2Size;
  // Processor-specific extension.
  uint32_t ISAExtensionSet;
  // Mask of ASEs used.
  uint32_t ASESet;

  bool OddSPReg;

  bool Is32BitABI;

protected:
  // The floating-point ABI.
  FpABIKind FpABI;

public:
  MipsABIFlagsSection()
      : Version(0), ISALevel(0), ISARevision(0), GPRSize(AFL_REG_NONE),
        CPR1Size(AFL_REG_NONE), CPR2Size(AFL_REG_NONE), ISAExtensionSet(0),
        ASESet(0), OddSPReg(false), Is32BitABI(false), FpABI(FpABIKind::ANY) {}

  uint16_t getVersionValue() { return (uint16_t)Version; }
  uint8_t getISALevelValue() { return (uint8_t)ISALevel; }
  uint8_t getISARevisionValue() { return (uint8_t)ISARevision; }
  uint8_t getGPRSizeValue() { return (uint8_t)GPRSize; }
  uint8_t getCPR1SizeValue();
  uint8_t getCPR2SizeValue() { return (uint8_t)CPR2Size; }
  uint8_t getFpABIValue();
  uint32_t getISAExtensionSetValue() { return (uint32_t)ISAExtensionSet; }
  uint32_t getASESetValue() { return (uint32_t)ASESet; }

  uint32_t getFlags1Value() {
    uint32_t Value = 0;

    if (OddSPReg)
      Value |= (uint32_t)AFL_FLAGS1_ODDSPREG;

    return Value;
  }

  uint32_t getFlags2Value() { return 0; }

  FpABIKind getFpABI() { return FpABI; }
  void setFpABI(FpABIKind Value, bool IsABI32Bit) {
    FpABI = Value;
    Is32BitABI = IsABI32Bit;
  }
  StringRef getFpABIString(FpABIKind Value);

  template <class PredicateLibrary>
  void setISALevelAndRevisionFromPredicates(const PredicateLibrary &P) {
    if (P.hasMips64()) {
      ISALevel = 64;
      if (P.hasMips64r6())
        ISARevision = 6;
      else if (P.hasMips64r2())
        ISARevision = 2;
      else
        ISARevision = 1;
    } else if (P.hasMips32()) {
      ISALevel = 32;
      if (P.hasMips32r6())
        ISARevision = 6;
      else if (P.hasMips32r2())
        ISARevision = 2;
      else
        ISARevision = 1;
    } else {
      ISARevision = 0;
      if (P.hasMips5())
        ISALevel = 5;
      else if (P.hasMips4())
        ISALevel = 4;
      else if (P.hasMips3())
        ISALevel = 3;
      else if (P.hasMips2())
        ISALevel = 2;
      else if (P.hasMips1())
        ISALevel = 1;
      else
        llvm_unreachable("Unknown ISA level!");
    }
  }

  template <class PredicateLibrary>
  void setGPRSizeFromPredicates(const PredicateLibrary &P) {
    GPRSize = P.isGP64bit() ? AFL_REG_64 : AFL_REG_32;
  }

  template <class PredicateLibrary>
  void setCPR1SizeFromPredicates(const PredicateLibrary &P) {
    if (P.abiUsesSoftFloat())
      CPR1Size = AFL_REG_NONE;
    else if (P.hasMSA())
      CPR1Size = AFL_REG_128;
    else
      CPR1Size = P.isFP64bit() ? AFL_REG_64 : AFL_REG_32;
  }

  template <class PredicateLibrary>
  void setASESetFromPredicates(const PredicateLibrary &P) {
    ASESet = 0;
    if (P.hasDSP())
      ASESet |= AFL_ASE_DSP;
    if (P.hasDSPR2())
      ASESet |= AFL_ASE_DSPR2;
    if (P.hasMSA())
      ASESet |= AFL_ASE_MSA;
    if (P.inMicroMipsMode())
      ASESet |= AFL_ASE_MICROMIPS;
    if (P.inMips16Mode())
      ASESet |= AFL_ASE_MIPS16;
  }

  template <class PredicateLibrary>
  void setFpAbiFromPredicates(const PredicateLibrary &P) {
    Is32BitABI = P.isABI_O32();

    FpABI = FpABIKind::ANY;
    if (P.isABI_N32() || P.isABI_N64())
      FpABI = FpABIKind::S64;
    else if (P.isABI_O32()) {
      if (P.isABI_FPXX())
        FpABI = FpABIKind::XX;
      else if (P.isFP64bit())
        FpABI = FpABIKind::S64;
      else
        FpABI = FpABIKind::S32;
    }
  }

  template <class PredicateLibrary>
  void setAllFromPredicates(const PredicateLibrary &P) {
    setISALevelAndRevisionFromPredicates(P);
    setGPRSizeFromPredicates(P);
    setCPR1SizeFromPredicates(P);
    setASESetFromPredicates(P);
    setFpAbiFromPredicates(P);
    OddSPReg = P.useOddSPReg();
  }
};

MCStreamer &operator<<(MCStreamer &OS, MipsABIFlagsSection &ABIFlagsSection);
}

#endif
