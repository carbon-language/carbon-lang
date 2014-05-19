//===-- ARM64InstPrinter.cpp - Convert ARM64 MCInst to assembly syntax ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an ARM64 MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "ARM64InstPrinter.h"
#include "MCTargetDesc/ARM64AddressingModes.h"
#include "Utils/ARM64BaseInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#define GET_INSTRUCTION_NAME
#define PRINT_ALIAS_INSTR
#include "ARM64GenAsmWriter.inc"
#define GET_INSTRUCTION_NAME
#define PRINT_ALIAS_INSTR
#include "ARM64GenAsmWriter1.inc"

ARM64InstPrinter::ARM64InstPrinter(const MCAsmInfo &MAI, const MCInstrInfo &MII,
                                   const MCRegisterInfo &MRI,
                                   const MCSubtargetInfo &STI)
    : MCInstPrinter(MAI, MII, MRI) {
  // Initialize the set of available features.
  setAvailableFeatures(STI.getFeatureBits());
}

ARM64AppleInstPrinter::ARM64AppleInstPrinter(const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI,
                                             const MCSubtargetInfo &STI)
    : ARM64InstPrinter(MAI, MII, MRI, STI) {}

void ARM64InstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  // This is for .cfi directives.
  OS << getRegisterName(RegNo);
}

void ARM64InstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                                 StringRef Annot) {
  // Check for special encodings and print the canonical alias instead.

  unsigned Opcode = MI->getOpcode();

  if (Opcode == ARM64::SYSxt)
    if (printSysAlias(MI, O)) {
      printAnnotation(O, Annot);
      return;
    }

  // SBFM/UBFM should print to a nicer aliased form if possible.
  if (Opcode == ARM64::SBFMXri || Opcode == ARM64::SBFMWri ||
      Opcode == ARM64::UBFMXri || Opcode == ARM64::UBFMWri) {
    const MCOperand &Op0 = MI->getOperand(0);
    const MCOperand &Op1 = MI->getOperand(1);
    const MCOperand &Op2 = MI->getOperand(2);
    const MCOperand &Op3 = MI->getOperand(3);

    bool IsSigned = (Opcode == ARM64::SBFMXri || Opcode == ARM64::SBFMWri);
    bool Is64Bit = (Opcode == ARM64::SBFMXri || Opcode == ARM64::UBFMXri);
    if (Op2.isImm() && Op2.getImm() == 0 && Op3.isImm()) {
      const char *AsmMnemonic = nullptr;

      switch (Op3.getImm()) {
      default:
        break;
      case 7:
        if (IsSigned)
          AsmMnemonic = "sxtb";
        else if (!Is64Bit)
          AsmMnemonic = "uxtb";
        break;
      case 15:
        if (IsSigned)
          AsmMnemonic = "sxth";
        else if (!Is64Bit)
          AsmMnemonic = "uxth";
        break;
      case 31:
        // *xtw is only valid for signed 64-bit operations.
        if (Is64Bit && IsSigned)
          AsmMnemonic = "sxtw";
        break;
      }

      if (AsmMnemonic) {
        O << '\t' << AsmMnemonic << '\t' << getRegisterName(Op0.getReg())
          << ", " << getRegisterName(getWRegFromXReg(Op1.getReg()));
        printAnnotation(O, Annot);
        return;
      }
    }

    // All immediate shifts are aliases, implemented using the Bitfield
    // instruction. In all cases the immediate shift amount shift must be in
    // the range 0 to (reg.size -1).
    if (Op2.isImm() && Op3.isImm()) {
      const char *AsmMnemonic = nullptr;
      int shift = 0;
      int64_t immr = Op2.getImm();
      int64_t imms = Op3.getImm();
      if (Opcode == ARM64::UBFMWri && imms != 0x1F && ((imms + 1) == immr)) {
        AsmMnemonic = "lsl";
        shift = 31 - imms;
      } else if (Opcode == ARM64::UBFMXri && imms != 0x3f &&
                 ((imms + 1 == immr))) {
        AsmMnemonic = "lsl";
        shift = 63 - imms;
      } else if (Opcode == ARM64::UBFMWri && imms == 0x1f) {
        AsmMnemonic = "lsr";
        shift = immr;
      } else if (Opcode == ARM64::UBFMXri && imms == 0x3f) {
        AsmMnemonic = "lsr";
        shift = immr;
      } else if (Opcode == ARM64::SBFMWri && imms == 0x1f) {
        AsmMnemonic = "asr";
        shift = immr;
      } else if (Opcode == ARM64::SBFMXri && imms == 0x3f) {
        AsmMnemonic = "asr";
        shift = immr;
      }
      if (AsmMnemonic) {
        O << '\t' << AsmMnemonic << '\t' << getRegisterName(Op0.getReg())
          << ", " << getRegisterName(Op1.getReg()) << ", #" << shift;
        printAnnotation(O, Annot);
        return;
      }
    }

    // SBFIZ/UBFIZ aliases
    if (Op2.getImm() > Op3.getImm()) {
      O << '\t' << (IsSigned ? "sbfiz" : "ubfiz") << '\t'
        << getRegisterName(Op0.getReg()) << ", " << getRegisterName(Op1.getReg())
        << ", #" << (Is64Bit ? 64 : 32) - Op2.getImm() << ", #" << Op3.getImm() + 1;
      printAnnotation(O, Annot);
      return;
    }

    // Otherwise SBFX/UBFX is the preferred form
    O << '\t' << (IsSigned ? "sbfx" : "ubfx") << '\t'
      << getRegisterName(Op0.getReg()) << ", " << getRegisterName(Op1.getReg())
      << ", #" << Op2.getImm() << ", #" << Op3.getImm() - Op2.getImm() + 1;
    printAnnotation(O, Annot);
    return;
  }

  if (Opcode == ARM64::BFMXri || Opcode == ARM64::BFMWri) {
    const MCOperand &Op0 = MI->getOperand(0); // Op1 == Op0
    const MCOperand &Op2 = MI->getOperand(2);
    int ImmR = MI->getOperand(3).getImm();
    int ImmS = MI->getOperand(4).getImm();

    // BFI alias
    if (ImmS < ImmR) {
      int BitWidth = Opcode == ARM64::BFMXri ? 64 : 32;
      int LSB = (BitWidth - ImmR) % BitWidth;
      int Width = ImmS + 1;
      O << "\tbfi\t" << getRegisterName(Op0.getReg()) << ", "
        << getRegisterName(Op2.getReg()) << ", #" << LSB << ", #" << Width;
      printAnnotation(O, Annot);
      return;
    }

    int LSB = ImmR;
    int Width = ImmS - ImmR + 1;
    // Otherwise BFXIL the preferred form
    O << "\tbfxil\t"
      << getRegisterName(Op0.getReg()) << ", " << getRegisterName(Op2.getReg())
      << ", #" << LSB << ", #" << Width;
    printAnnotation(O, Annot);
    return;
  }

  // Symbolic operands for MOVZ, MOVN and MOVK already imply a shift
  // (e.g. :gottprel_g1: is always going to be "lsl #16") so it should not be
  // printed.
  if ((Opcode == ARM64::MOVZXi || Opcode == ARM64::MOVZWi ||
       Opcode == ARM64::MOVNXi || Opcode == ARM64::MOVNWi) &&
      MI->getOperand(1).isExpr()) {
    if (Opcode == ARM64::MOVZXi || Opcode == ARM64::MOVZWi)
      O << "\tmovz\t";
    else
      O << "\tmovn\t";

    O << getRegisterName(MI->getOperand(0).getReg()) << ", #"
      << *MI->getOperand(1).getExpr();
    return;
  }

  if ((Opcode == ARM64::MOVKXi || Opcode == ARM64::MOVKWi) &&
      MI->getOperand(2).isExpr()) {
    O << "\tmovk\t" << getRegisterName(MI->getOperand(0).getReg()) << ", #"
      << *MI->getOperand(2).getExpr();
    return;
  }

  if (!printAliasInstr(MI, O))
    printInstruction(MI, O);

  printAnnotation(O, Annot);
}

static bool isTblTbxInstruction(unsigned Opcode, StringRef &Layout,
                                bool &IsTbx) {
  switch (Opcode) {
  case ARM64::TBXv8i8One:
  case ARM64::TBXv8i8Two:
  case ARM64::TBXv8i8Three:
  case ARM64::TBXv8i8Four:
    IsTbx = true;
    Layout = ".8b";
    return true;
  case ARM64::TBLv8i8One:
  case ARM64::TBLv8i8Two:
  case ARM64::TBLv8i8Three:
  case ARM64::TBLv8i8Four:
    IsTbx = false;
    Layout = ".8b";
    return true;
  case ARM64::TBXv16i8One:
  case ARM64::TBXv16i8Two:
  case ARM64::TBXv16i8Three:
  case ARM64::TBXv16i8Four:
    IsTbx = true;
    Layout = ".16b";
    return true;
  case ARM64::TBLv16i8One:
  case ARM64::TBLv16i8Two:
  case ARM64::TBLv16i8Three:
  case ARM64::TBLv16i8Four:
    IsTbx = false;
    Layout = ".16b";
    return true;
  default:
    return false;
  }
}

struct LdStNInstrDesc {
  unsigned Opcode;
  const char *Mnemonic;
  const char *Layout;
  int ListOperand;
  bool HasLane;
  int NaturalOffset;
};

static LdStNInstrDesc LdStNInstInfo[] = {
  { ARM64::LD1i8,             "ld1",  ".b",     1, true,  0  },
  { ARM64::LD1i16,            "ld1",  ".h",     1, true,  0  },
  { ARM64::LD1i32,            "ld1",  ".s",     1, true,  0  },
  { ARM64::LD1i64,            "ld1",  ".d",     1, true,  0  },
  { ARM64::LD1i8_POST,        "ld1",  ".b",     2, true,  1  },
  { ARM64::LD1i16_POST,       "ld1",  ".h",     2, true,  2  },
  { ARM64::LD1i32_POST,       "ld1",  ".s",     2, true,  4  },
  { ARM64::LD1i64_POST,       "ld1",  ".d",     2, true,  8  },
  { ARM64::LD1Rv16b,          "ld1r", ".16b",   0, false, 0  },
  { ARM64::LD1Rv8h,           "ld1r", ".8h",    0, false, 0  },
  { ARM64::LD1Rv4s,           "ld1r", ".4s",    0, false, 0  },
  { ARM64::LD1Rv2d,           "ld1r", ".2d",    0, false, 0  },
  { ARM64::LD1Rv8b,           "ld1r", ".8b",    0, false, 0  },
  { ARM64::LD1Rv4h,           "ld1r", ".4h",    0, false, 0  },
  { ARM64::LD1Rv2s,           "ld1r", ".2s",    0, false, 0  },
  { ARM64::LD1Rv1d,           "ld1r", ".1d",    0, false, 0  },
  { ARM64::LD1Rv16b_POST,     "ld1r", ".16b",   1, false, 1  },
  { ARM64::LD1Rv8h_POST,      "ld1r", ".8h",    1, false, 2  },
  { ARM64::LD1Rv4s_POST,      "ld1r", ".4s",    1, false, 4  },
  { ARM64::LD1Rv2d_POST,      "ld1r", ".2d",    1, false, 8  },
  { ARM64::LD1Rv8b_POST,      "ld1r", ".8b",    1, false, 1  },
  { ARM64::LD1Rv4h_POST,      "ld1r", ".4h",    1, false, 2  },
  { ARM64::LD1Rv2s_POST,      "ld1r", ".2s",    1, false, 4  },
  { ARM64::LD1Rv1d_POST,      "ld1r", ".1d",    1, false, 8  },
  { ARM64::LD1Onev16b,        "ld1",  ".16b",   0, false, 0  },
  { ARM64::LD1Onev8h,         "ld1",  ".8h",    0, false, 0  },
  { ARM64::LD1Onev4s,         "ld1",  ".4s",    0, false, 0  },
  { ARM64::LD1Onev2d,         "ld1",  ".2d",    0, false, 0  },
  { ARM64::LD1Onev8b,         "ld1",  ".8b",    0, false, 0  },
  { ARM64::LD1Onev4h,         "ld1",  ".4h",    0, false, 0  },
  { ARM64::LD1Onev2s,         "ld1",  ".2s",    0, false, 0  },
  { ARM64::LD1Onev1d,         "ld1",  ".1d",    0, false, 0  },
  { ARM64::LD1Onev16b_POST,   "ld1",  ".16b",   1, false, 16 },
  { ARM64::LD1Onev8h_POST,    "ld1",  ".8h",    1, false, 16 },
  { ARM64::LD1Onev4s_POST,    "ld1",  ".4s",    1, false, 16 },
  { ARM64::LD1Onev2d_POST,    "ld1",  ".2d",    1, false, 16 },
  { ARM64::LD1Onev8b_POST,    "ld1",  ".8b",    1, false, 8  },
  { ARM64::LD1Onev4h_POST,    "ld1",  ".4h",    1, false, 8  },
  { ARM64::LD1Onev2s_POST,    "ld1",  ".2s",    1, false, 8  },
  { ARM64::LD1Onev1d_POST,    "ld1",  ".1d",    1, false, 8  },
  { ARM64::LD1Twov16b,        "ld1",  ".16b",   0, false, 0  },
  { ARM64::LD1Twov8h,         "ld1",  ".8h",    0, false, 0  },
  { ARM64::LD1Twov4s,         "ld1",  ".4s",    0, false, 0  },
  { ARM64::LD1Twov2d,         "ld1",  ".2d",    0, false, 0  },
  { ARM64::LD1Twov8b,         "ld1",  ".8b",    0, false, 0  },
  { ARM64::LD1Twov4h,         "ld1",  ".4h",    0, false, 0  },
  { ARM64::LD1Twov2s,         "ld1",  ".2s",    0, false, 0  },
  { ARM64::LD1Twov1d,         "ld1",  ".1d",    0, false, 0  },
  { ARM64::LD1Twov16b_POST,   "ld1",  ".16b",   1, false, 32 },
  { ARM64::LD1Twov8h_POST,    "ld1",  ".8h",    1, false, 32 },
  { ARM64::LD1Twov4s_POST,    "ld1",  ".4s",    1, false, 32 },
  { ARM64::LD1Twov2d_POST,    "ld1",  ".2d",    1, false, 32 },
  { ARM64::LD1Twov8b_POST,    "ld1",  ".8b",    1, false, 16 },
  { ARM64::LD1Twov4h_POST,    "ld1",  ".4h",    1, false, 16 },
  { ARM64::LD1Twov2s_POST,    "ld1",  ".2s",    1, false, 16 },
  { ARM64::LD1Twov1d_POST,    "ld1",  ".1d",    1, false, 16 },
  { ARM64::LD1Threev16b,      "ld1",  ".16b",   0, false, 0  },
  { ARM64::LD1Threev8h,       "ld1",  ".8h",    0, false, 0  },
  { ARM64::LD1Threev4s,       "ld1",  ".4s",    0, false, 0  },
  { ARM64::LD1Threev2d,       "ld1",  ".2d",    0, false, 0  },
  { ARM64::LD1Threev8b,       "ld1",  ".8b",    0, false, 0  },
  { ARM64::LD1Threev4h,       "ld1",  ".4h",    0, false, 0  },
  { ARM64::LD1Threev2s,       "ld1",  ".2s",    0, false, 0  },
  { ARM64::LD1Threev1d,       "ld1",  ".1d",    0, false, 0  },
  { ARM64::LD1Threev16b_POST, "ld1",  ".16b",   1, false, 48 },
  { ARM64::LD1Threev8h_POST,  "ld1",  ".8h",    1, false, 48 },
  { ARM64::LD1Threev4s_POST,  "ld1",  ".4s",    1, false, 48 },
  { ARM64::LD1Threev2d_POST,  "ld1",  ".2d",    1, false, 48 },
  { ARM64::LD1Threev8b_POST,  "ld1",  ".8b",    1, false, 24 },
  { ARM64::LD1Threev4h_POST,  "ld1",  ".4h",    1, false, 24 },
  { ARM64::LD1Threev2s_POST,  "ld1",  ".2s",    1, false, 24 },
  { ARM64::LD1Threev1d_POST,  "ld1",  ".1d",    1, false, 24 },
  { ARM64::LD1Fourv16b,       "ld1",  ".16b",   0, false, 0  },
  { ARM64::LD1Fourv8h,        "ld1",  ".8h",    0, false, 0  },
  { ARM64::LD1Fourv4s,        "ld1",  ".4s",    0, false, 0  },
  { ARM64::LD1Fourv2d,        "ld1",  ".2d",    0, false, 0  },
  { ARM64::LD1Fourv8b,        "ld1",  ".8b",    0, false, 0  },
  { ARM64::LD1Fourv4h,        "ld1",  ".4h",    0, false, 0  },
  { ARM64::LD1Fourv2s,        "ld1",  ".2s",    0, false, 0  },
  { ARM64::LD1Fourv1d,        "ld1",  ".1d",    0, false, 0  },
  { ARM64::LD1Fourv16b_POST,  "ld1",  ".16b",   1, false, 64 },
  { ARM64::LD1Fourv8h_POST,   "ld1",  ".8h",    1, false, 64 },
  { ARM64::LD1Fourv4s_POST,   "ld1",  ".4s",    1, false, 64 },
  { ARM64::LD1Fourv2d_POST,   "ld1",  ".2d",    1, false, 64 },
  { ARM64::LD1Fourv8b_POST,   "ld1",  ".8b",    1, false, 32 },
  { ARM64::LD1Fourv4h_POST,   "ld1",  ".4h",    1, false, 32 },
  { ARM64::LD1Fourv2s_POST,   "ld1",  ".2s",    1, false, 32 },
  { ARM64::LD1Fourv1d_POST,   "ld1",  ".1d",    1, false, 32 },
  { ARM64::LD2i8,             "ld2",  ".b",     1, true,  0  },
  { ARM64::LD2i16,            "ld2",  ".h",     1, true,  0  },
  { ARM64::LD2i32,            "ld2",  ".s",     1, true,  0  },
  { ARM64::LD2i64,            "ld2",  ".d",     1, true,  0  },
  { ARM64::LD2i8_POST,        "ld2",  ".b",     2, true,  2  },
  { ARM64::LD2i16_POST,       "ld2",  ".h",     2, true,  4  },
  { ARM64::LD2i32_POST,       "ld2",  ".s",     2, true,  8  },
  { ARM64::LD2i64_POST,       "ld2",  ".d",     2, true,  16  },
  { ARM64::LD2Rv16b,          "ld2r", ".16b",   0, false, 0  },
  { ARM64::LD2Rv8h,           "ld2r", ".8h",    0, false, 0  },
  { ARM64::LD2Rv4s,           "ld2r", ".4s",    0, false, 0  },
  { ARM64::LD2Rv2d,           "ld2r", ".2d",    0, false, 0  },
  { ARM64::LD2Rv8b,           "ld2r", ".8b",    0, false, 0  },
  { ARM64::LD2Rv4h,           "ld2r", ".4h",    0, false, 0  },
  { ARM64::LD2Rv2s,           "ld2r", ".2s",    0, false, 0  },
  { ARM64::LD2Rv1d,           "ld2r", ".1d",    0, false, 0  },
  { ARM64::LD2Rv16b_POST,     "ld2r", ".16b",   1, false, 2  },
  { ARM64::LD2Rv8h_POST,      "ld2r", ".8h",    1, false, 4  },
  { ARM64::LD2Rv4s_POST,      "ld2r", ".4s",    1, false, 8  },
  { ARM64::LD2Rv2d_POST,      "ld2r", ".2d",    1, false, 16 },
  { ARM64::LD2Rv8b_POST,      "ld2r", ".8b",    1, false, 2  },
  { ARM64::LD2Rv4h_POST,      "ld2r", ".4h",    1, false, 4  },
  { ARM64::LD2Rv2s_POST,      "ld2r", ".2s",    1, false, 8  },
  { ARM64::LD2Rv1d_POST,      "ld2r", ".1d",    1, false, 16 },
  { ARM64::LD2Twov16b,        "ld2",  ".16b",   0, false, 0  },
  { ARM64::LD2Twov8h,         "ld2",  ".8h",    0, false, 0  },
  { ARM64::LD2Twov4s,         "ld2",  ".4s",    0, false, 0  },
  { ARM64::LD2Twov2d,         "ld2",  ".2d",    0, false, 0  },
  { ARM64::LD2Twov8b,         "ld2",  ".8b",    0, false, 0  },
  { ARM64::LD2Twov4h,         "ld2",  ".4h",    0, false, 0  },
  { ARM64::LD2Twov2s,         "ld2",  ".2s",    0, false, 0  },
  { ARM64::LD2Twov16b_POST,   "ld2",  ".16b",   1, false, 32 },
  { ARM64::LD2Twov8h_POST,    "ld2",  ".8h",    1, false, 32 },
  { ARM64::LD2Twov4s_POST,    "ld2",  ".4s",    1, false, 32 },
  { ARM64::LD2Twov2d_POST,    "ld2",  ".2d",    1, false, 32 },
  { ARM64::LD2Twov8b_POST,    "ld2",  ".8b",    1, false, 16 },
  { ARM64::LD2Twov4h_POST,    "ld2",  ".4h",    1, false, 16 },
  { ARM64::LD2Twov2s_POST,    "ld2",  ".2s",    1, false, 16 },
  { ARM64::LD3i8,             "ld3",  ".b",     1, true,  0  },
  { ARM64::LD3i16,            "ld3",  ".h",     1, true,  0  },
  { ARM64::LD3i32,            "ld3",  ".s",     1, true,  0  },
  { ARM64::LD3i64,            "ld3",  ".d",     1, true,  0  },
  { ARM64::LD3i8_POST,        "ld3",  ".b",     2, true,  3  },
  { ARM64::LD3i16_POST,       "ld3",  ".h",     2, true,  6  },
  { ARM64::LD3i32_POST,       "ld3",  ".s",     2, true,  12  },
  { ARM64::LD3i64_POST,       "ld3",  ".d",     2, true,  24  },
  { ARM64::LD3Rv16b,          "ld3r", ".16b",   0, false, 0  },
  { ARM64::LD3Rv8h,           "ld3r", ".8h",    0, false, 0  },
  { ARM64::LD3Rv4s,           "ld3r", ".4s",    0, false, 0  },
  { ARM64::LD3Rv2d,           "ld3r", ".2d",    0, false, 0  },
  { ARM64::LD3Rv8b,           "ld3r", ".8b",    0, false, 0  },
  { ARM64::LD3Rv4h,           "ld3r", ".4h",    0, false, 0  },
  { ARM64::LD3Rv2s,           "ld3r", ".2s",    0, false, 0  },
  { ARM64::LD3Rv1d,           "ld3r", ".1d",    0, false, 0  },
  { ARM64::LD3Rv16b_POST,     "ld3r", ".16b",   1, false, 3  },
  { ARM64::LD3Rv8h_POST,      "ld3r", ".8h",    1, false, 6  },
  { ARM64::LD3Rv4s_POST,      "ld3r", ".4s",    1, false, 12 },
  { ARM64::LD3Rv2d_POST,      "ld3r", ".2d",    1, false, 24 },
  { ARM64::LD3Rv8b_POST,      "ld3r", ".8b",    1, false, 3  },
  { ARM64::LD3Rv4h_POST,      "ld3r", ".4h",    1, false, 6  },
  { ARM64::LD3Rv2s_POST,      "ld3r", ".2s",    1, false, 12 },
  { ARM64::LD3Rv1d_POST,      "ld3r", ".1d",    1, false, 24 },
  { ARM64::LD3Threev16b,      "ld3",  ".16b",   0, false, 0  },
  { ARM64::LD3Threev8h,       "ld3",  ".8h",    0, false, 0  },
  { ARM64::LD3Threev4s,       "ld3",  ".4s",    0, false, 0  },
  { ARM64::LD3Threev2d,       "ld3",  ".2d",    0, false, 0  },
  { ARM64::LD3Threev8b,       "ld3",  ".8b",    0, false, 0  },
  { ARM64::LD3Threev4h,       "ld3",  ".4h",    0, false, 0  },
  { ARM64::LD3Threev2s,       "ld3",  ".2s",    0, false, 0  },
  { ARM64::LD3Threev16b_POST, "ld3",  ".16b",   1, false, 48 },
  { ARM64::LD3Threev8h_POST,  "ld3",  ".8h",    1, false, 48 },
  { ARM64::LD3Threev4s_POST,  "ld3",  ".4s",    1, false, 48 },
  { ARM64::LD3Threev2d_POST,  "ld3",  ".2d",    1, false, 48 },
  { ARM64::LD3Threev8b_POST,  "ld3",  ".8b",    1, false, 24 },
  { ARM64::LD3Threev4h_POST,  "ld3",  ".4h",    1, false, 24 },
  { ARM64::LD3Threev2s_POST,  "ld3",  ".2s",    1, false, 24 },
  { ARM64::LD4i8,             "ld4",  ".b",     1, true,  0  },
  { ARM64::LD4i16,            "ld4",  ".h",     1, true,  0  },
  { ARM64::LD4i32,            "ld4",  ".s",     1, true,  0  },
  { ARM64::LD4i64,            "ld4",  ".d",     1, true,  0  },
  { ARM64::LD4i8_POST,        "ld4",  ".b",     2, true,  4  },
  { ARM64::LD4i16_POST,       "ld4",  ".h",     2, true,  8  },
  { ARM64::LD4i32_POST,       "ld4",  ".s",     2, true,  16 },
  { ARM64::LD4i64_POST,       "ld4",  ".d",     2, true,  32 },
  { ARM64::LD4Rv16b,          "ld4r", ".16b",   0, false, 0  },
  { ARM64::LD4Rv8h,           "ld4r", ".8h",    0, false, 0  },
  { ARM64::LD4Rv4s,           "ld4r", ".4s",    0, false, 0  },
  { ARM64::LD4Rv2d,           "ld4r", ".2d",    0, false, 0  },
  { ARM64::LD4Rv8b,           "ld4r", ".8b",    0, false, 0  },
  { ARM64::LD4Rv4h,           "ld4r", ".4h",    0, false, 0  },
  { ARM64::LD4Rv2s,           "ld4r", ".2s",    0, false, 0  },
  { ARM64::LD4Rv1d,           "ld4r", ".1d",    0, false, 0  },
  { ARM64::LD4Rv16b_POST,     "ld4r", ".16b",   1, false, 4  },
  { ARM64::LD4Rv8h_POST,      "ld4r", ".8h",    1, false, 8  },
  { ARM64::LD4Rv4s_POST,      "ld4r", ".4s",    1, false, 16 },
  { ARM64::LD4Rv2d_POST,      "ld4r", ".2d",    1, false, 32 },
  { ARM64::LD4Rv8b_POST,      "ld4r", ".8b",    1, false, 4  },
  { ARM64::LD4Rv4h_POST,      "ld4r", ".4h",    1, false, 8  },
  { ARM64::LD4Rv2s_POST,      "ld4r", ".2s",    1, false, 16 },
  { ARM64::LD4Rv1d_POST,      "ld4r", ".1d",    1, false, 32 },
  { ARM64::LD4Fourv16b,       "ld4",  ".16b",   0, false, 0  },
  { ARM64::LD4Fourv8h,        "ld4",  ".8h",    0, false, 0  },
  { ARM64::LD4Fourv4s,        "ld4",  ".4s",    0, false, 0  },
  { ARM64::LD4Fourv2d,        "ld4",  ".2d",    0, false, 0  },
  { ARM64::LD4Fourv8b,        "ld4",  ".8b",    0, false, 0  },
  { ARM64::LD4Fourv4h,        "ld4",  ".4h",    0, false, 0  },
  { ARM64::LD4Fourv2s,        "ld4",  ".2s",    0, false, 0  },
  { ARM64::LD4Fourv16b_POST,  "ld4",  ".16b",   1, false, 64 },
  { ARM64::LD4Fourv8h_POST,   "ld4",  ".8h",    1, false, 64 },
  { ARM64::LD4Fourv4s_POST,   "ld4",  ".4s",    1, false, 64 },
  { ARM64::LD4Fourv2d_POST,   "ld4",  ".2d",    1, false, 64 },
  { ARM64::LD4Fourv8b_POST,   "ld4",  ".8b",    1, false, 32 },
  { ARM64::LD4Fourv4h_POST,   "ld4",  ".4h",    1, false, 32 },
  { ARM64::LD4Fourv2s_POST,   "ld4",  ".2s",    1, false, 32 },
  { ARM64::ST1i8,             "st1",  ".b",     0, true,  0  },
  { ARM64::ST1i16,            "st1",  ".h",     0, true,  0  },
  { ARM64::ST1i32,            "st1",  ".s",     0, true,  0  },
  { ARM64::ST1i64,            "st1",  ".d",     0, true,  0  },
  { ARM64::ST1i8_POST,        "st1",  ".b",     1, true,  1  },
  { ARM64::ST1i16_POST,       "st1",  ".h",     1, true,  2  },
  { ARM64::ST1i32_POST,       "st1",  ".s",     1, true,  4  },
  { ARM64::ST1i64_POST,       "st1",  ".d",     1, true,  8  },
  { ARM64::ST1Onev16b,        "st1",  ".16b",   0, false, 0  },
  { ARM64::ST1Onev8h,         "st1",  ".8h",    0, false, 0  },
  { ARM64::ST1Onev4s,         "st1",  ".4s",    0, false, 0  },
  { ARM64::ST1Onev2d,         "st1",  ".2d",    0, false, 0  },
  { ARM64::ST1Onev8b,         "st1",  ".8b",    0, false, 0  },
  { ARM64::ST1Onev4h,         "st1",  ".4h",    0, false, 0  },
  { ARM64::ST1Onev2s,         "st1",  ".2s",    0, false, 0  },
  { ARM64::ST1Onev1d,         "st1",  ".1d",    0, false, 0  },
  { ARM64::ST1Onev16b_POST,   "st1",  ".16b",   1, false, 16 },
  { ARM64::ST1Onev8h_POST,    "st1",  ".8h",    1, false, 16 },
  { ARM64::ST1Onev4s_POST,    "st1",  ".4s",    1, false, 16 },
  { ARM64::ST1Onev2d_POST,    "st1",  ".2d",    1, false, 16 },
  { ARM64::ST1Onev8b_POST,    "st1",  ".8b",    1, false, 8  },
  { ARM64::ST1Onev4h_POST,    "st1",  ".4h",    1, false, 8  },
  { ARM64::ST1Onev2s_POST,    "st1",  ".2s",    1, false, 8  },
  { ARM64::ST1Onev1d_POST,    "st1",  ".1d",    1, false, 8  },
  { ARM64::ST1Twov16b,        "st1",  ".16b",   0, false, 0  },
  { ARM64::ST1Twov8h,         "st1",  ".8h",    0, false, 0  },
  { ARM64::ST1Twov4s,         "st1",  ".4s",    0, false, 0  },
  { ARM64::ST1Twov2d,         "st1",  ".2d",    0, false, 0  },
  { ARM64::ST1Twov8b,         "st1",  ".8b",    0, false, 0  },
  { ARM64::ST1Twov4h,         "st1",  ".4h",    0, false, 0  },
  { ARM64::ST1Twov2s,         "st1",  ".2s",    0, false, 0  },
  { ARM64::ST1Twov1d,         "st1",  ".1d",    0, false, 0  },
  { ARM64::ST1Twov16b_POST,   "st1",  ".16b",   1, false, 32 },
  { ARM64::ST1Twov8h_POST,    "st1",  ".8h",    1, false, 32 },
  { ARM64::ST1Twov4s_POST,    "st1",  ".4s",    1, false, 32 },
  { ARM64::ST1Twov2d_POST,    "st1",  ".2d",    1, false, 32 },
  { ARM64::ST1Twov8b_POST,    "st1",  ".8b",    1, false, 16 },
  { ARM64::ST1Twov4h_POST,    "st1",  ".4h",    1, false, 16 },
  { ARM64::ST1Twov2s_POST,    "st1",  ".2s",    1, false, 16 },
  { ARM64::ST1Twov1d_POST,    "st1",  ".1d",    1, false, 16 },
  { ARM64::ST1Threev16b,      "st1",  ".16b",   0, false, 0  },
  { ARM64::ST1Threev8h,       "st1",  ".8h",    0, false, 0  },
  { ARM64::ST1Threev4s,       "st1",  ".4s",    0, false, 0  },
  { ARM64::ST1Threev2d,       "st1",  ".2d",    0, false, 0  },
  { ARM64::ST1Threev8b,       "st1",  ".8b",    0, false, 0  },
  { ARM64::ST1Threev4h,       "st1",  ".4h",    0, false, 0  },
  { ARM64::ST1Threev2s,       "st1",  ".2s",    0, false, 0  },
  { ARM64::ST1Threev1d,       "st1",  ".1d",    0, false, 0  },
  { ARM64::ST1Threev16b_POST, "st1",  ".16b",   1, false, 48 },
  { ARM64::ST1Threev8h_POST,  "st1",  ".8h",    1, false, 48 },
  { ARM64::ST1Threev4s_POST,  "st1",  ".4s",    1, false, 48 },
  { ARM64::ST1Threev2d_POST,  "st1",  ".2d",    1, false, 48 },
  { ARM64::ST1Threev8b_POST,  "st1",  ".8b",    1, false, 24 },
  { ARM64::ST1Threev4h_POST,  "st1",  ".4h",    1, false, 24 },
  { ARM64::ST1Threev2s_POST,  "st1",  ".2s",    1, false, 24 },
  { ARM64::ST1Threev1d_POST,  "st1",  ".1d",    1, false, 24 },
  { ARM64::ST1Fourv16b,       "st1",  ".16b",   0, false, 0  },
  { ARM64::ST1Fourv8h,        "st1",  ".8h",    0, false, 0  },
  { ARM64::ST1Fourv4s,        "st1",  ".4s",    0, false, 0  },
  { ARM64::ST1Fourv2d,        "st1",  ".2d",    0, false, 0  },
  { ARM64::ST1Fourv8b,        "st1",  ".8b",    0, false, 0  },
  { ARM64::ST1Fourv4h,        "st1",  ".4h",    0, false, 0  },
  { ARM64::ST1Fourv2s,        "st1",  ".2s",    0, false, 0  },
  { ARM64::ST1Fourv1d,        "st1",  ".1d",    0, false, 0  },
  { ARM64::ST1Fourv16b_POST,  "st1",  ".16b",   1, false, 64 },
  { ARM64::ST1Fourv8h_POST,   "st1",  ".8h",    1, false, 64 },
  { ARM64::ST1Fourv4s_POST,   "st1",  ".4s",    1, false, 64 },
  { ARM64::ST1Fourv2d_POST,   "st1",  ".2d",    1, false, 64 },
  { ARM64::ST1Fourv8b_POST,   "st1",  ".8b",    1, false, 32 },
  { ARM64::ST1Fourv4h_POST,   "st1",  ".4h",    1, false, 32 },
  { ARM64::ST1Fourv2s_POST,   "st1",  ".2s",    1, false, 32 },
  { ARM64::ST1Fourv1d_POST,   "st1",  ".1d",    1, false, 32 },
  { ARM64::ST2i8,             "st2",  ".b",     0, true,  0  },
  { ARM64::ST2i16,            "st2",  ".h",     0, true,  0  },
  { ARM64::ST2i32,            "st2",  ".s",     0, true,  0  },
  { ARM64::ST2i64,            "st2",  ".d",     0, true,  0  },
  { ARM64::ST2i8_POST,        "st2",  ".b",     1, true,  2  },
  { ARM64::ST2i16_POST,       "st2",  ".h",     1, true,  4  },
  { ARM64::ST2i32_POST,       "st2",  ".s",     1, true,  8  },
  { ARM64::ST2i64_POST,       "st2",  ".d",     1, true,  16 },
  { ARM64::ST2Twov16b,        "st2",  ".16b",   0, false, 0  },
  { ARM64::ST2Twov8h,         "st2",  ".8h",    0, false, 0  },
  { ARM64::ST2Twov4s,         "st2",  ".4s",    0, false, 0  },
  { ARM64::ST2Twov2d,         "st2",  ".2d",    0, false, 0  },
  { ARM64::ST2Twov8b,         "st2",  ".8b",    0, false, 0  },
  { ARM64::ST2Twov4h,         "st2",  ".4h",    0, false, 0  },
  { ARM64::ST2Twov2s,         "st2",  ".2s",    0, false, 0  },
  { ARM64::ST2Twov16b_POST,   "st2",  ".16b",   1, false, 32 },
  { ARM64::ST2Twov8h_POST,    "st2",  ".8h",    1, false, 32 },
  { ARM64::ST2Twov4s_POST,    "st2",  ".4s",    1, false, 32 },
  { ARM64::ST2Twov2d_POST,    "st2",  ".2d",    1, false, 32 },
  { ARM64::ST2Twov8b_POST,    "st2",  ".8b",    1, false, 16 },
  { ARM64::ST2Twov4h_POST,    "st2",  ".4h",    1, false, 16 },
  { ARM64::ST2Twov2s_POST,    "st2",  ".2s",    1, false, 16 },
  { ARM64::ST3i8,             "st3",  ".b",     0, true,  0  },
  { ARM64::ST3i16,            "st3",  ".h",     0, true,  0  },
  { ARM64::ST3i32,            "st3",  ".s",     0, true,  0  },
  { ARM64::ST3i64,            "st3",  ".d",     0, true,  0  },
  { ARM64::ST3i8_POST,        "st3",  ".b",     1, true,  3  },
  { ARM64::ST3i16_POST,       "st3",  ".h",     1, true,  6  },
  { ARM64::ST3i32_POST,       "st3",  ".s",     1, true,  12 },
  { ARM64::ST3i64_POST,       "st3",  ".d",     1, true,  24 },
  { ARM64::ST3Threev16b,      "st3",  ".16b",   0, false, 0  },
  { ARM64::ST3Threev8h,       "st3",  ".8h",    0, false, 0  },
  { ARM64::ST3Threev4s,       "st3",  ".4s",    0, false, 0  },
  { ARM64::ST3Threev2d,       "st3",  ".2d",    0, false, 0  },
  { ARM64::ST3Threev8b,       "st3",  ".8b",    0, false, 0  },
  { ARM64::ST3Threev4h,       "st3",  ".4h",    0, false, 0  },
  { ARM64::ST3Threev2s,       "st3",  ".2s",    0, false, 0  },
  { ARM64::ST3Threev16b_POST, "st3",  ".16b",   1, false, 48 },
  { ARM64::ST3Threev8h_POST,  "st3",  ".8h",    1, false, 48 },
  { ARM64::ST3Threev4s_POST,  "st3",  ".4s",    1, false, 48 },
  { ARM64::ST3Threev2d_POST,  "st3",  ".2d",    1, false, 48 },
  { ARM64::ST3Threev8b_POST,  "st3",  ".8b",    1, false, 24 },
  { ARM64::ST3Threev4h_POST,  "st3",  ".4h",    1, false, 24 },
  { ARM64::ST3Threev2s_POST,  "st3",  ".2s",    1, false, 24 },
  { ARM64::ST4i8,             "st4",  ".b",     0, true,  0  },
  { ARM64::ST4i16,            "st4",  ".h",     0, true,  0  },
  { ARM64::ST4i32,            "st4",  ".s",     0, true,  0  },
  { ARM64::ST4i64,            "st4",  ".d",     0, true,  0  },
  { ARM64::ST4i8_POST,        "st4",  ".b",     1, true,  4  },
  { ARM64::ST4i16_POST,       "st4",  ".h",     1, true,  8  },
  { ARM64::ST4i32_POST,       "st4",  ".s",     1, true,  16 },
  { ARM64::ST4i64_POST,       "st4",  ".d",     1, true,  32 },
  { ARM64::ST4Fourv16b,       "st4",  ".16b",   0, false, 0  },
  { ARM64::ST4Fourv8h,        "st4",  ".8h",    0, false, 0  },
  { ARM64::ST4Fourv4s,        "st4",  ".4s",    0, false, 0  },
  { ARM64::ST4Fourv2d,        "st4",  ".2d",    0, false, 0  },
  { ARM64::ST4Fourv8b,        "st4",  ".8b",    0, false, 0  },
  { ARM64::ST4Fourv4h,        "st4",  ".4h",    0, false, 0  },
  { ARM64::ST4Fourv2s,        "st4",  ".2s",    0, false, 0  },
  { ARM64::ST4Fourv16b_POST,  "st4",  ".16b",   1, false, 64 },
  { ARM64::ST4Fourv8h_POST,   "st4",  ".8h",    1, false, 64 },
  { ARM64::ST4Fourv4s_POST,   "st4",  ".4s",    1, false, 64 },
  { ARM64::ST4Fourv2d_POST,   "st4",  ".2d",    1, false, 64 },
  { ARM64::ST4Fourv8b_POST,   "st4",  ".8b",    1, false, 32 },
  { ARM64::ST4Fourv4h_POST,   "st4",  ".4h",    1, false, 32 },
  { ARM64::ST4Fourv2s_POST,   "st4",  ".2s",    1, false, 32 },
};

static LdStNInstrDesc *getLdStNInstrDesc(unsigned Opcode) {
  unsigned Idx;
  for (Idx = 0; Idx != array_lengthof(LdStNInstInfo); ++Idx)
    if (LdStNInstInfo[Idx].Opcode == Opcode)
      return &LdStNInstInfo[Idx];

  return nullptr;
}

void ARM64AppleInstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                                      StringRef Annot) {
  unsigned Opcode = MI->getOpcode();
  StringRef Layout, Mnemonic;

  bool IsTbx;
  if (isTblTbxInstruction(MI->getOpcode(), Layout, IsTbx)) {
    O << "\t" << (IsTbx ? "tbx" : "tbl") << Layout << '\t'
      << getRegisterName(MI->getOperand(0).getReg(), ARM64::vreg) << ", ";

    unsigned ListOpNum = IsTbx ? 2 : 1;
    printVectorList(MI, ListOpNum, O, "");

    O << ", "
      << getRegisterName(MI->getOperand(ListOpNum + 1).getReg(), ARM64::vreg);
    printAnnotation(O, Annot);
    return;
  }

  if (LdStNInstrDesc *LdStDesc = getLdStNInstrDesc(Opcode)) {
    O << "\t" << LdStDesc->Mnemonic << LdStDesc->Layout << '\t';

    // Now onto the operands: first a vector list with possible lane
    // specifier. E.g. { v0 }[2]
    int OpNum = LdStDesc->ListOperand;
    printVectorList(MI, OpNum++, O, "");

    if (LdStDesc->HasLane)
      O << '[' << MI->getOperand(OpNum++).getImm() << ']';

    // Next the address: [xN]
    unsigned AddrReg = MI->getOperand(OpNum++).getReg();
    O << ", [" << getRegisterName(AddrReg) << ']';

    // Finally, there might be a post-indexed offset.
    if (LdStDesc->NaturalOffset != 0) {
      unsigned Reg = MI->getOperand(OpNum++).getReg();
      if (Reg != ARM64::XZR)
        O << ", " << getRegisterName(Reg);
      else {
        assert(LdStDesc->NaturalOffset && "no offset on post-inc instruction?");
        O << ", #" << LdStDesc->NaturalOffset;
      }
    }

    printAnnotation(O, Annot);
    return;
  }

  ARM64InstPrinter::printInst(MI, O, Annot);
}

bool ARM64InstPrinter::printSysAlias(const MCInst *MI, raw_ostream &O) {
#ifndef NDEBUG
  unsigned Opcode = MI->getOpcode();
  assert(Opcode == ARM64::SYSxt && "Invalid opcode for SYS alias!");
#endif

  const char *Asm = nullptr;
  const MCOperand &Op1 = MI->getOperand(0);
  const MCOperand &Cn = MI->getOperand(1);
  const MCOperand &Cm = MI->getOperand(2);
  const MCOperand &Op2 = MI->getOperand(3);

  unsigned Op1Val = Op1.getImm();
  unsigned CnVal = Cn.getImm();
  unsigned CmVal = Cm.getImm();
  unsigned Op2Val = Op2.getImm();

  if (CnVal == 7) {
    switch (CmVal) {
    default:
      break;

    // IC aliases
    case 1:
      if (Op1Val == 0 && Op2Val == 0)
        Asm = "ic\tialluis";
      break;
    case 5:
      if (Op1Val == 0 && Op2Val == 0)
        Asm = "ic\tiallu";
      else if (Op1Val == 3 && Op2Val == 1)
        Asm = "ic\tivau";
      break;

    // DC aliases
    case 4:
      if (Op1Val == 3 && Op2Val == 1)
        Asm = "dc\tzva";
      break;
    case 6:
      if (Op1Val == 0 && Op2Val == 1)
        Asm = "dc\tivac";
      if (Op1Val == 0 && Op2Val == 2)
        Asm = "dc\tisw";
      break;
    case 10:
      if (Op1Val == 3 && Op2Val == 1)
        Asm = "dc\tcvac";
      else if (Op1Val == 0 && Op2Val == 2)
        Asm = "dc\tcsw";
      break;
    case 11:
      if (Op1Val == 3 && Op2Val == 1)
        Asm = "dc\tcvau";
      break;
    case 14:
      if (Op1Val == 3 && Op2Val == 1)
        Asm = "dc\tcivac";
      else if (Op1Val == 0 && Op2Val == 2)
        Asm = "dc\tcisw";
      break;

    // AT aliases
    case 8:
      switch (Op1Val) {
      default:
        break;
      case 0:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "at\ts1e1r"; break;
        case 1: Asm = "at\ts1e1w"; break;
        case 2: Asm = "at\ts1e0r"; break;
        case 3: Asm = "at\ts1e0w"; break;
        }
        break;
      case 4:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "at\ts1e2r"; break;
        case 1: Asm = "at\ts1e2w"; break;
        case 4: Asm = "at\ts12e1r"; break;
        case 5: Asm = "at\ts12e1w"; break;
        case 6: Asm = "at\ts12e0r"; break;
        case 7: Asm = "at\ts12e0w"; break;
        }
        break;
      case 6:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "at\ts1e3r"; break;
        case 1: Asm = "at\ts1e3w"; break;
        }
        break;
      }
      break;
    }
  } else if (CnVal == 8) {
    // TLBI aliases
    switch (CmVal) {
    default:
      break;
    case 3:
      switch (Op1Val) {
      default:
        break;
      case 0:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "tlbi\tvmalle1is"; break;
        case 1: Asm = "tlbi\tvae1is"; break;
        case 2: Asm = "tlbi\taside1is"; break;
        case 3: Asm = "tlbi\tvaae1is"; break;
        case 5: Asm = "tlbi\tvale1is"; break;
        case 7: Asm = "tlbi\tvaale1is"; break;
        }
        break;
      case 4:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "tlbi\talle2is"; break;
        case 1: Asm = "tlbi\tvae2is"; break;
        case 4: Asm = "tlbi\talle1is"; break;
        case 5: Asm = "tlbi\tvale2is"; break;
        case 6: Asm = "tlbi\tvmalls12e1is"; break;
        }
        break;
      case 6:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "tlbi\talle3is"; break;
        case 1: Asm = "tlbi\tvae3is"; break;
        case 5: Asm = "tlbi\tvale3is"; break;
        }
        break;
      }
      break;
    case 0:
      switch (Op1Val) {
      default:
        break;
      case 4:
        switch (Op2Val) {
        default:
          break;
        case 1: Asm = "tlbi\tipas2e1is"; break;
        case 5: Asm = "tlbi\tipas2le1is"; break;
        }
        break;
      }
      break;
    case 4:
      switch (Op1Val) {
      default:
        break;
      case 4:
        switch (Op2Val) {
        default:
          break;
        case 1: Asm = "tlbi\tipas2e1"; break;
        case 5: Asm = "tlbi\tipas2le1"; break;
        }
        break;
      }
      break;
    case 7:
      switch (Op1Val) {
      default:
        break;
      case 0:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "tlbi\tvmalle1"; break;
        case 1: Asm = "tlbi\tvae1"; break;
        case 2: Asm = "tlbi\taside1"; break;
        case 3: Asm = "tlbi\tvaae1"; break;
        case 5: Asm = "tlbi\tvale1"; break;
        case 7: Asm = "tlbi\tvaale1"; break;
        }
        break;
      case 4:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "tlbi\talle2"; break;
        case 1: Asm = "tlbi\tvae2"; break;
        case 4: Asm = "tlbi\talle1"; break;
        case 5: Asm = "tlbi\tvale2"; break;
        case 6: Asm = "tlbi\tvmalls12e1"; break;
        }
        break;
      case 6:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "tlbi\talle3"; break;
        case 1: Asm = "tlbi\tvae3";  break;
        case 5: Asm = "tlbi\tvale3"; break;
        }
        break;
      }
      break;
    }
  }

  if (Asm) {
    unsigned Reg = MI->getOperand(4).getReg();

    O << '\t' << Asm;
    if (StringRef(Asm).lower().find("all") == StringRef::npos)
      O << ", " << getRegisterName(Reg);
  }

  return Asm != nullptr;
}

void ARM64InstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    unsigned Reg = Op.getReg();
    O << getRegisterName(Reg);
  } else if (Op.isImm()) {
    O << '#' << Op.getImm();
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    O << *Op.getExpr();
  }
}

void ARM64InstPrinter::printHexImm(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  O << format("#%#llx", Op.getImm());
}

void ARM64InstPrinter::printPostIncOperand(const MCInst *MI, unsigned OpNo,
                                           unsigned Imm, raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    unsigned Reg = Op.getReg();
    if (Reg == ARM64::XZR)
      O << "#" << Imm;
    else
      O << getRegisterName(Reg);
  } else
    assert(0 && "unknown operand kind in printPostIncOperand64");
}

void ARM64InstPrinter::printVRegOperand(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isReg() && "Non-register vreg operand!");
  unsigned Reg = Op.getReg();
  O << getRegisterName(Reg, ARM64::vreg);
}

void ARM64InstPrinter::printSysCROperand(const MCInst *MI, unsigned OpNo,
                                         raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isImm() && "System instruction C[nm] operands must be immediates!");
  O << "c" << Op.getImm();
}

void ARM64InstPrinter::printAddSubImm(const MCInst *MI, unsigned OpNum,
                                      raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.isImm()) {
    unsigned Val = (MO.getImm() & 0xfff);
    assert(Val == MO.getImm() && "Add/sub immediate out of range!");
    unsigned Shift =
        ARM64_AM::getShiftValue(MI->getOperand(OpNum + 1).getImm());
    O << '#' << Val;
    if (Shift != 0)
      printShifter(MI, OpNum + 1, O);

    if (CommentStream)
      *CommentStream << "=#" << (Val << Shift) << '\n';
  } else {
    assert(MO.isExpr() && "Unexpected operand type!");
    O << *MO.getExpr();
    printShifter(MI, OpNum + 1, O);
  }
}

void ARM64InstPrinter::printLogicalImm32(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  uint64_t Val = MI->getOperand(OpNum).getImm();
  O << "#0x";
  O.write_hex(ARM64_AM::decodeLogicalImmediate(Val, 32));
}

void ARM64InstPrinter::printLogicalImm64(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  uint64_t Val = MI->getOperand(OpNum).getImm();
  O << "#0x";
  O.write_hex(ARM64_AM::decodeLogicalImmediate(Val, 64));
}

void ARM64InstPrinter::printShifter(const MCInst *MI, unsigned OpNum,
                                    raw_ostream &O) {
  unsigned Val = MI->getOperand(OpNum).getImm();
  // LSL #0 should not be printed.
  if (ARM64_AM::getShiftType(Val) == ARM64_AM::LSL &&
      ARM64_AM::getShiftValue(Val) == 0)
    return;
  O << ", " << ARM64_AM::getShiftExtendName(ARM64_AM::getShiftType(Val)) << " #"
    << ARM64_AM::getShiftValue(Val);
}

void ARM64InstPrinter::printShiftedRegister(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  O << getRegisterName(MI->getOperand(OpNum).getReg());
  printShifter(MI, OpNum + 1, O);
}

void ARM64InstPrinter::printExtendedRegister(const MCInst *MI, unsigned OpNum,
                                             raw_ostream &O) {
  O << getRegisterName(MI->getOperand(OpNum).getReg());
  printExtend(MI, OpNum + 1, O);
}

void ARM64InstPrinter::printExtend(const MCInst *MI, unsigned OpNum,
                                   raw_ostream &O) {
  unsigned Val = MI->getOperand(OpNum).getImm();
  ARM64_AM::ShiftExtendType ExtType = ARM64_AM::getArithExtendType(Val);
  unsigned ShiftVal = ARM64_AM::getArithShiftValue(Val);

  // If the destination or first source register operand is [W]SP, print
  // UXTW/UXTX as LSL, and if the shift amount is also zero, print nothing at
  // all.
  if (ExtType == ARM64_AM::UXTW || ExtType == ARM64_AM::UXTX) {
    unsigned Dest = MI->getOperand(0).getReg();
    unsigned Src1 = MI->getOperand(1).getReg();
    if ( ((Dest == ARM64::SP || Src1 == ARM64::SP) &&
          ExtType == ARM64_AM::UXTX) ||
         ((Dest == ARM64::WSP || Src1 == ARM64::WSP) &&
          ExtType == ARM64_AM::UXTW) ) {
      if (ShiftVal != 0)
        O << ", lsl #" << ShiftVal;
      return;
    }
  }
  O << ", " << ARM64_AM::getShiftExtendName(ExtType);
  if (ShiftVal != 0)
    O << " #" << ShiftVal;
}

void ARM64InstPrinter::printCondCode(const MCInst *MI, unsigned OpNum,
                                     raw_ostream &O) {
  ARM64CC::CondCode CC = (ARM64CC::CondCode)MI->getOperand(OpNum).getImm();
  O << ARM64CC::getCondCodeName(CC);
}

void ARM64InstPrinter::printInverseCondCode(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  ARM64CC::CondCode CC = (ARM64CC::CondCode)MI->getOperand(OpNum).getImm();
  O << ARM64CC::getCondCodeName(ARM64CC::getInvertedCondCode(CC));
}

void ARM64InstPrinter::printAMNoIndex(const MCInst *MI, unsigned OpNum,
                                      raw_ostream &O) {
  O << '[' << getRegisterName(MI->getOperand(OpNum).getReg()) << ']';
}

template<int Scale>
void ARM64InstPrinter::printImmScale(const MCInst *MI, unsigned OpNum,
                                     raw_ostream &O) {
  O << '#' << Scale * MI->getOperand(OpNum).getImm();
}

void ARM64InstPrinter::printAMIndexed(const MCInst *MI, unsigned OpNum,
                                      unsigned Scale, raw_ostream &O) {
  const MCOperand MO1 = MI->getOperand(OpNum + 1);
  O << '[' << getRegisterName(MI->getOperand(OpNum).getReg());
  if (MO1.isImm()) {
    if (MO1.getImm() != 0)
      O << ", #" << (MO1.getImm() * Scale);
  } else {
    assert(MO1.isExpr() && "Unexpected operand type!");
    O << ", " << *MO1.getExpr();
  }
  O << ']';
}

void ARM64InstPrinter::printAMIndexedWB(const MCInst *MI, unsigned OpNum,
                                        unsigned Scale, raw_ostream &O) {
  const MCOperand MO1 = MI->getOperand(OpNum + 1);
  O << '[' << getRegisterName(MI->getOperand(OpNum).getReg());
  if (MO1.isImm()) {
      O << ", #" << (MO1.getImm() * Scale);
  } else {
    assert(MO1.isExpr() && "Unexpected operand type!");
    O << ", " << *MO1.getExpr();
  }
  O << ']';
}

void ARM64InstPrinter::printPrefetchOp(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  unsigned prfop = MI->getOperand(OpNum).getImm();
  bool Valid;
  StringRef Name = ARM64PRFM::PRFMMapper().toString(prfop, Valid);
  if (Valid)
    O << Name;
  else
    O << '#' << prfop;
}

void ARM64InstPrinter::printMemoryPostIndexed(const MCInst *MI, unsigned OpNum,
                                              raw_ostream &O, unsigned Scale) {
  O << '[' << getRegisterName(MI->getOperand(OpNum).getReg()) << ']' << ", #"
    << Scale * MI->getOperand(OpNum + 1).getImm();
}

void ARM64InstPrinter::printMemoryRegOffset(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O, int Scale) {
  unsigned Val = MI->getOperand(OpNum + 2).getImm();
  ARM64_AM::ShiftExtendType ExtType = ARM64_AM::getMemExtendType(Val);

  O << '[' << getRegisterName(MI->getOperand(OpNum).getReg()) << ", ";
  if (ExtType == ARM64_AM::UXTW || ExtType == ARM64_AM::SXTW)
    O << getRegisterName(getWRegFromXReg(MI->getOperand(OpNum + 1).getReg()));
  else
    O << getRegisterName(MI->getOperand(OpNum + 1).getReg());

  bool DoShift = ARM64_AM::getMemDoShift(Val);

  if (ExtType == ARM64_AM::UXTX) {
    if (DoShift)
      O << ", lsl";
  } else
    O << ", " << ARM64_AM::getShiftExtendName(ExtType);

  if (DoShift)
    O << " #" << Log2_32(Scale);

  O << "]";
}

void ARM64InstPrinter::printFPImmOperand(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  float FPImm = MO.isFPImm() ? MO.getFPImm() : ARM64_AM::getFPImmFloat(MO.getImm());

  // 8 decimal places are enough to perfectly represent permitted floats.
  O << format("#%.8f", FPImm);
}

static unsigned getNextVectorRegister(unsigned Reg, unsigned Stride = 1) {
  while (Stride--) {
    switch (Reg) {
    default:
      assert(0 && "Vector register expected!");
    case ARM64::Q0:  Reg = ARM64::Q1;  break;
    case ARM64::Q1:  Reg = ARM64::Q2;  break;
    case ARM64::Q2:  Reg = ARM64::Q3;  break;
    case ARM64::Q3:  Reg = ARM64::Q4;  break;
    case ARM64::Q4:  Reg = ARM64::Q5;  break;
    case ARM64::Q5:  Reg = ARM64::Q6;  break;
    case ARM64::Q6:  Reg = ARM64::Q7;  break;
    case ARM64::Q7:  Reg = ARM64::Q8;  break;
    case ARM64::Q8:  Reg = ARM64::Q9;  break;
    case ARM64::Q9:  Reg = ARM64::Q10; break;
    case ARM64::Q10: Reg = ARM64::Q11; break;
    case ARM64::Q11: Reg = ARM64::Q12; break;
    case ARM64::Q12: Reg = ARM64::Q13; break;
    case ARM64::Q13: Reg = ARM64::Q14; break;
    case ARM64::Q14: Reg = ARM64::Q15; break;
    case ARM64::Q15: Reg = ARM64::Q16; break;
    case ARM64::Q16: Reg = ARM64::Q17; break;
    case ARM64::Q17: Reg = ARM64::Q18; break;
    case ARM64::Q18: Reg = ARM64::Q19; break;
    case ARM64::Q19: Reg = ARM64::Q20; break;
    case ARM64::Q20: Reg = ARM64::Q21; break;
    case ARM64::Q21: Reg = ARM64::Q22; break;
    case ARM64::Q22: Reg = ARM64::Q23; break;
    case ARM64::Q23: Reg = ARM64::Q24; break;
    case ARM64::Q24: Reg = ARM64::Q25; break;
    case ARM64::Q25: Reg = ARM64::Q26; break;
    case ARM64::Q26: Reg = ARM64::Q27; break;
    case ARM64::Q27: Reg = ARM64::Q28; break;
    case ARM64::Q28: Reg = ARM64::Q29; break;
    case ARM64::Q29: Reg = ARM64::Q30; break;
    case ARM64::Q30: Reg = ARM64::Q31; break;
    // Vector lists can wrap around.
    case ARM64::Q31:
      Reg = ARM64::Q0;
      break;
    }
  }
  return Reg;
}

void ARM64InstPrinter::printVectorList(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O, StringRef LayoutSuffix) {
  unsigned Reg = MI->getOperand(OpNum).getReg();

  O << "{ ";

  // Work out how many registers there are in the list (if there is an actual
  // list).
  unsigned NumRegs = 1;
  if (MRI.getRegClass(ARM64::DDRegClassID).contains(Reg) ||
      MRI.getRegClass(ARM64::QQRegClassID).contains(Reg))
    NumRegs = 2;
  else if (MRI.getRegClass(ARM64::DDDRegClassID).contains(Reg) ||
           MRI.getRegClass(ARM64::QQQRegClassID).contains(Reg))
    NumRegs = 3;
  else if (MRI.getRegClass(ARM64::DDDDRegClassID).contains(Reg) ||
           MRI.getRegClass(ARM64::QQQQRegClassID).contains(Reg))
    NumRegs = 4;

  // Now forget about the list and find out what the first register is.
  if (unsigned FirstReg = MRI.getSubReg(Reg, ARM64::dsub0))
    Reg = FirstReg;
  else if (unsigned FirstReg = MRI.getSubReg(Reg, ARM64::qsub0))
    Reg = FirstReg;

  // If it's a D-reg, we need to promote it to the equivalent Q-reg before
  // printing (otherwise getRegisterName fails).
  if (MRI.getRegClass(ARM64::FPR64RegClassID).contains(Reg)) {
    const MCRegisterClass &FPR128RC = MRI.getRegClass(ARM64::FPR128RegClassID);
    Reg = MRI.getMatchingSuperReg(Reg, ARM64::dsub, &FPR128RC);
  }

  for (unsigned i = 0; i < NumRegs; ++i, Reg = getNextVectorRegister(Reg)) {
    O << getRegisterName(Reg, ARM64::vreg) << LayoutSuffix;
    if (i + 1 != NumRegs)
      O << ", ";
  }

  O << " }";
}

void ARM64InstPrinter::printImplicitlyTypedVectorList(const MCInst *MI,
                                                      unsigned OpNum,
                                                      raw_ostream &O) {
  printVectorList(MI, OpNum, O, "");
}

template <unsigned NumLanes, char LaneKind>
void ARM64InstPrinter::printTypedVectorList(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  std::string Suffix(".");
  if (NumLanes)
    Suffix += itostr(NumLanes) + LaneKind;
  else
    Suffix += LaneKind;

  printVectorList(MI, OpNum, O, Suffix);
}

void ARM64InstPrinter::printVectorIndex(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  O << "[" << MI->getOperand(OpNum).getImm() << "]";
}

void ARM64InstPrinter::printAlignedLabel(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);

  // If the label has already been resolved to an immediate offset (say, when
  // we're running the disassembler), just print the immediate.
  if (Op.isImm()) {
    O << "#" << (Op.getImm() << 2);
    return;
  }

  // If the branch target is simply an address then print it in hex.
  const MCConstantExpr *BranchTarget =
      dyn_cast<MCConstantExpr>(MI->getOperand(OpNum).getExpr());
  int64_t Address;
  if (BranchTarget && BranchTarget->EvaluateAsAbsolute(Address)) {
    O << "0x";
    O.write_hex(Address);
  } else {
    // Otherwise, just print the expression.
    O << *MI->getOperand(OpNum).getExpr();
  }
}

void ARM64InstPrinter::printAdrpLabel(const MCInst *MI, unsigned OpNum,
                                      raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);

  // If the label has already been resolved to an immediate offset (say, when
  // we're running the disassembler), just print the immediate.
  if (Op.isImm()) {
    O << "#" << (Op.getImm() << 12);
    return;
  }

  // Otherwise, just print the expression.
  O << *MI->getOperand(OpNum).getExpr();
}

void ARM64InstPrinter::printBarrierOption(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  unsigned Val = MI->getOperand(OpNo).getImm();
  unsigned Opcode = MI->getOpcode();

  bool Valid;
  StringRef Name;
  if (Opcode == ARM64::ISB)
    Name = ARM64ISB::ISBMapper().toString(Val, Valid);
  else
    Name = ARM64DB::DBarrierMapper().toString(Val, Valid);
  if (Valid)
    O << Name;
  else
    O << "#" << Val;
}

void ARM64InstPrinter::printMRSSystemRegister(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  unsigned Val = MI->getOperand(OpNo).getImm();

  bool Valid;
  auto Mapper = ARM64SysReg::MRSMapper(getAvailableFeatures());
  std::string Name = Mapper.toString(Val, Valid);

  if (Valid)
    O << StringRef(Name).upper();
}

void ARM64InstPrinter::printMSRSystemRegister(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  unsigned Val = MI->getOperand(OpNo).getImm();

  bool Valid;
  auto Mapper = ARM64SysReg::MSRMapper(getAvailableFeatures());
  std::string Name = Mapper.toString(Val, Valid);

  if (Valid)
    O << StringRef(Name).upper();
}

void ARM64InstPrinter::printSystemPStateField(const MCInst *MI, unsigned OpNo,
                                              raw_ostream &O) {
  unsigned Val = MI->getOperand(OpNo).getImm();

  bool Valid;
  StringRef Name = ARM64PState::PStateMapper().toString(Val, Valid);
  if (Valid)
    O << StringRef(Name.str()).upper();
  else
    O << "#" << Val;
}

void ARM64InstPrinter::printSIMDType10Operand(const MCInst *MI, unsigned OpNo,
                                              raw_ostream &O) {
  unsigned RawVal = MI->getOperand(OpNo).getImm();
  uint64_t Val = ARM64_AM::decodeAdvSIMDModImmType10(RawVal);
  O << format("#%#016llx", Val);
}
