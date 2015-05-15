//===-- HexagonMCCodeEmitter.cpp - Hexagon Target Descriptions ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonFixupKinds.h"
#include "MCTargetDesc/HexagonMCCodeEmitter.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "mccodeemitter"

using namespace llvm;
using namespace Hexagon;

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");

namespace {
/// \brief 10.6 Instruction Packets
/// Possible values for instruction packet parse field.
enum class ParseField { duplex = 0x0, last0 = 0x1, last1 = 0x2, end = 0x3 };
/// \brief Returns the packet bits based on instruction position.
uint32_t getPacketBits(MCInst const &HMI) {
  unsigned const ParseFieldOffset = 14;
  ParseField Field = HexagonMCInstrInfo::isPacketEnd(HMI) ? ParseField::end
                                                          : ParseField::last0;
  return static_cast<uint32_t>(Field) << ParseFieldOffset;
}
void emitLittleEndian(uint64_t Binary, raw_ostream &OS) {
  OS << static_cast<uint8_t>((Binary >> 0x00) & 0xff);
  OS << static_cast<uint8_t>((Binary >> 0x08) & 0xff);
  OS << static_cast<uint8_t>((Binary >> 0x10) & 0xff);
  OS << static_cast<uint8_t>((Binary >> 0x18) & 0xff);
}
}

HexagonMCCodeEmitter::HexagonMCCodeEmitter(MCInstrInfo const &aMII,
                                           MCContext &aMCT)
    : MCT(aMCT), MCII(aMII), Addend(new unsigned(0)),
      Extended(new bool(false)) {}

void HexagonMCCodeEmitter::encodeInstruction(MCInst const &MI, raw_ostream &OS,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             MCSubtargetInfo const &STI) const {
  uint64_t Binary = getBinaryCodeForInstr(MI, Fixups, STI) | getPacketBits(MI);
  assert(HexagonMCInstrInfo::getDesc(MCII, MI).getSize() == 4 &&
         "All instructions should be 32bit");
  (void)&MCII;
  emitLittleEndian(Binary, OS);
  ++MCNumEmitted;
}

static Hexagon::Fixups getFixupNoBits(MCInstrInfo const &MCII, const MCInst &MI,
                                      const MCOperand &MO,
                                      const MCSymbolRefExpr::VariantKind kind) {
  const MCInstrDesc &MCID = HexagonMCInstrInfo::getDesc(MCII, MI);
  unsigned insnType = llvm::HexagonMCInstrInfo::getType(MCII, MI);

  if (insnType == HexagonII::TypePREFIX) {
    switch (kind) {
    case llvm::MCSymbolRefExpr::VK_GOTOFF:
      return Hexagon::fixup_Hexagon_GOTREL_32_6_X;
    case llvm::MCSymbolRefExpr::VK_GOT:
      return Hexagon::fixup_Hexagon_GOT_32_6_X;
    case llvm::MCSymbolRefExpr::VK_TPREL:
      return Hexagon::fixup_Hexagon_TPREL_32_6_X;
    case llvm::MCSymbolRefExpr::VK_DTPREL:
      return Hexagon::fixup_Hexagon_DTPREL_32_6_X;
    case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
      return Hexagon::fixup_Hexagon_GD_GOT_32_6_X;
    case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
      return Hexagon::fixup_Hexagon_LD_GOT_32_6_X;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE:
      return Hexagon::fixup_Hexagon_IE_32_6_X;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
      return Hexagon::fixup_Hexagon_IE_GOT_32_6_X;
    default:
      if (MCID.isBranch())
        return Hexagon::fixup_Hexagon_B32_PCREL_X;
      else
        return Hexagon::fixup_Hexagon_32_6_X;
    }
  } else if (MCID.isBranch())
    return (Hexagon::fixup_Hexagon_B13_PCREL);

  switch (MCID.getOpcode()) {
  case Hexagon::HI:
  case Hexagon::A2_tfrih:
    switch (kind) {
    case llvm::MCSymbolRefExpr::VK_GOT:
      return Hexagon::fixup_Hexagon_GOT_HI16;
    case llvm::MCSymbolRefExpr::VK_GOTOFF:
      return Hexagon::fixup_Hexagon_GOTREL_HI16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
      return Hexagon::fixup_Hexagon_GD_GOT_HI16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
      return Hexagon::fixup_Hexagon_LD_GOT_HI16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE:
      return Hexagon::fixup_Hexagon_IE_HI16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
      return Hexagon::fixup_Hexagon_IE_GOT_HI16;
    case llvm::MCSymbolRefExpr::VK_TPREL:
      return Hexagon::fixup_Hexagon_TPREL_HI16;
    case llvm::MCSymbolRefExpr::VK_DTPREL:
      return Hexagon::fixup_Hexagon_DTPREL_HI16;
    default:
      return Hexagon::fixup_Hexagon_HI16;
    }

  case Hexagon::LO:
  case Hexagon::A2_tfril:
    switch (kind) {
    case llvm::MCSymbolRefExpr::VK_GOT:
      return Hexagon::fixup_Hexagon_GOT_LO16;
    case llvm::MCSymbolRefExpr::VK_GOTOFF:
      return Hexagon::fixup_Hexagon_GOTREL_LO16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
      return Hexagon::fixup_Hexagon_GD_GOT_LO16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
      return Hexagon::fixup_Hexagon_LD_GOT_LO16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE:
      return Hexagon::fixup_Hexagon_IE_LO16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
      return Hexagon::fixup_Hexagon_IE_GOT_LO16;
    case llvm::MCSymbolRefExpr::VK_TPREL:
      return Hexagon::fixup_Hexagon_TPREL_LO16;
    case llvm::MCSymbolRefExpr::VK_DTPREL:
      return Hexagon::fixup_Hexagon_DTPREL_LO16;
    default:
      return Hexagon::fixup_Hexagon_LO16;
    }

  // The only relocs left should be GP relative:
  default:
    if (MCID.mayStore() || MCID.mayLoad()) {
      for (const uint16_t *ImpUses = MCID.getImplicitUses(); *ImpUses;
           ++ImpUses) {
        if (*ImpUses == Hexagon::GP) {
          switch (HexagonMCInstrInfo::getAccessSize(MCII, MI)) {
          case HexagonII::MemAccessSize::ByteAccess:
            return fixup_Hexagon_GPREL16_0;
          case HexagonII::MemAccessSize::HalfWordAccess:
            return fixup_Hexagon_GPREL16_1;
          case HexagonII::MemAccessSize::WordAccess:
            return fixup_Hexagon_GPREL16_2;
          case HexagonII::MemAccessSize::DoubleWordAccess:
            return fixup_Hexagon_GPREL16_3;
          default:
            llvm_unreachable("unhandled fixup");
          }
        }
      }
    } else
      llvm_unreachable("unhandled fixup");
  }

  return LastTargetFixupKind;
}

unsigned HexagonMCCodeEmitter::getExprOpValue(const MCInst &MI,
                                              const MCOperand &MO,
                                              const MCExpr *ME,
                                              SmallVectorImpl<MCFixup> &Fixups,
                                              const MCSubtargetInfo &STI) const

{
  int64_t Res;

  if (ME->EvaluateAsAbsolute(Res))
    return Res;

  MCExpr::ExprKind MK = ME->getKind();
  if (MK == MCExpr::Constant) {
    return cast<MCConstantExpr>(ME)->getValue();
  }
  if (MK == MCExpr::Binary) {
    unsigned Res;
    Res = getExprOpValue(MI, MO, cast<MCBinaryExpr>(ME)->getLHS(), Fixups, STI);
    Res +=
        getExprOpValue(MI, MO, cast<MCBinaryExpr>(ME)->getRHS(), Fixups, STI);
    return Res;
  }

  assert(MK == MCExpr::SymbolRef);

  Hexagon::Fixups FixupKind =
      Hexagon::Fixups(Hexagon::fixup_Hexagon_TPREL_LO16);
  const MCSymbolRefExpr *MCSRE = static_cast<const MCSymbolRefExpr *>(ME);
  const MCInstrDesc &MCID = HexagonMCInstrInfo::getDesc(MCII, MI);
  unsigned bits = HexagonMCInstrInfo::getExtentBits(MCII, MI) -
                  HexagonMCInstrInfo::getExtentAlignment(MCII, MI);
  const MCSymbolRefExpr::VariantKind kind = MCSRE->getKind();

  DEBUG(dbgs() << "----------------------------------------\n");
  DEBUG(dbgs() << "Opcode Name: " << HexagonMCInstrInfo::getName(MCII, MI)
               << "\n");
  DEBUG(dbgs() << "Opcode: " << MCID.getOpcode() << "\n");
  DEBUG(dbgs() << "Relocation bits: " << bits << "\n");
  DEBUG(dbgs() << "Addend: " << *Addend << "\n");
  DEBUG(dbgs() << "----------------------------------------\n");

  switch (bits) {
  default:
    DEBUG(dbgs() << "unrecognized bit count of " << bits << '\n');
    break;

  case 32:
    switch (kind) {
    case llvm::MCSymbolRefExpr::VK_Hexagon_PCREL:
      FixupKind = Hexagon::fixup_Hexagon_32_PCREL;
      break;
    case llvm::MCSymbolRefExpr::VK_GOT:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_GOT_32_6_X
                            : Hexagon::fixup_Hexagon_GOT_32;
      break;
    case llvm::MCSymbolRefExpr::VK_GOTOFF:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_GOTREL_32_6_X
                            : Hexagon::fixup_Hexagon_GOTREL_32;
      break;
    case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_GD_GOT_32_6_X
                            : Hexagon::fixup_Hexagon_GD_GOT_32;
      break;
    case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_LD_GOT_32_6_X
                            : Hexagon::fixup_Hexagon_LD_GOT_32;
      break;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_IE_32_6_X
                            : Hexagon::fixup_Hexagon_IE_32;
      break;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_IE_GOT_32_6_X
                            : Hexagon::fixup_Hexagon_IE_GOT_32;
      break;
    case llvm::MCSymbolRefExpr::VK_TPREL:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_TPREL_32_6_X
                            : Hexagon::fixup_Hexagon_TPREL_32;
      break;
    case llvm::MCSymbolRefExpr::VK_DTPREL:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_DTPREL_32_6_X
                            : Hexagon::fixup_Hexagon_DTPREL_32;
      break;
    default:
      FixupKind =
          *Extended ? Hexagon::fixup_Hexagon_32_6_X : Hexagon::fixup_Hexagon_32;
      break;
    }
    break;

  case 22:
    switch (kind) {
    case llvm::MCSymbolRefExpr::VK_Hexagon_GD_PLT:
      FixupKind = Hexagon::fixup_Hexagon_GD_PLT_B22_PCREL;
      break;
    case llvm::MCSymbolRefExpr::VK_Hexagon_LD_PLT:
      FixupKind = Hexagon::fixup_Hexagon_LD_PLT_B22_PCREL;
      break;
    default:
      if (MCID.isBranch() || MCID.isCall()) {
        FixupKind = *Extended ? Hexagon::fixup_Hexagon_B22_PCREL_X
                              : Hexagon::fixup_Hexagon_B22_PCREL;
      } else {
        errs() << "unrecognized relocation, bits: " << bits << "\n";
        errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
      }
      break;
    }
    break;

  case 16:
    if (*Extended) {
      switch (kind) {
      default:
        FixupKind = Hexagon::fixup_Hexagon_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GOT_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_GOTOFF:
        FixupKind = Hexagon::fixup_Hexagon_GOTREL_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GD_GOT_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
        FixupKind = Hexagon::fixup_Hexagon_LD_GOT_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_IE:
        FixupKind = Hexagon::fixup_Hexagon_IE_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
        FixupKind = Hexagon::fixup_Hexagon_IE_GOT_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_TPREL:
        FixupKind = Hexagon::fixup_Hexagon_TPREL_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_DTPREL:
        FixupKind = Hexagon::fixup_Hexagon_DTPREL_16_X;
        break;
      }
    } else
      switch (kind) {
      default:
        errs() << "unrecognized relocation, bits " << bits << "\n";
        errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
        break;
      case llvm::MCSymbolRefExpr::VK_GOTOFF:
        if ((MCID.getOpcode() == Hexagon::HI) ||
            (MCID.getOpcode() == Hexagon::LO_H))
          FixupKind = Hexagon::fixup_Hexagon_GOTREL_HI16;
        else
          FixupKind = Hexagon::fixup_Hexagon_GOTREL_LO16;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_GPREL:
        FixupKind = Hexagon::fixup_Hexagon_GPREL16_0;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_LO16:
        FixupKind = Hexagon::fixup_Hexagon_LO16;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_HI16:
        FixupKind = Hexagon::fixup_Hexagon_HI16;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GD_GOT_16;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
        FixupKind = Hexagon::fixup_Hexagon_LD_GOT_16;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
        FixupKind = Hexagon::fixup_Hexagon_IE_GOT_16;
        break;
      case llvm::MCSymbolRefExpr::VK_TPREL:
        FixupKind = Hexagon::fixup_Hexagon_TPREL_16;
        break;
      case llvm::MCSymbolRefExpr::VK_DTPREL:
        FixupKind = Hexagon::fixup_Hexagon_DTPREL_16;
        break;
      }
    break;

  case 15:
    if (MCID.isBranch() || MCID.isCall())
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_B15_PCREL_X
                            : Hexagon::fixup_Hexagon_B15_PCREL;
    break;

  case 13:
    if (MCID.isBranch())
      FixupKind = Hexagon::fixup_Hexagon_B13_PCREL;
    else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 12:
    if (*Extended)
      switch (kind) {
      default:
        FixupKind = Hexagon::fixup_Hexagon_12_X;
        break;
      // There isn't a GOT_12_X, both 11_X and 16_X resolve to 6/26
      case llvm::MCSymbolRefExpr::VK_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GOT_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_GOTOFF:
        FixupKind = Hexagon::fixup_Hexagon_GOTREL_16_X;
        break;
      }
    else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 11:
    if (*Extended)
      switch (kind) {
      default:
        FixupKind = Hexagon::fixup_Hexagon_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GOT_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_GOTOFF:
        FixupKind = Hexagon::fixup_Hexagon_GOTREL_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GD_GOT_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
        FixupKind = Hexagon::fixup_Hexagon_LD_GOT_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
        FixupKind = Hexagon::fixup_Hexagon_IE_GOT_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_TPREL:
        FixupKind = Hexagon::fixup_Hexagon_TPREL_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_DTPREL:
        FixupKind = Hexagon::fixup_Hexagon_DTPREL_11_X;
        break;
      }
    else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 10:
    if (*Extended)
      FixupKind = Hexagon::fixup_Hexagon_10_X;
    break;

  case 9:
    if (MCID.isBranch() ||
        (llvm::HexagonMCInstrInfo::getType(MCII, MI) == HexagonII::TypeCR))
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_B9_PCREL_X
                            : Hexagon::fixup_Hexagon_B9_PCREL;
    else if (*Extended)
      FixupKind = Hexagon::fixup_Hexagon_9_X;
    else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 8:
    if (*Extended)
      FixupKind = Hexagon::fixup_Hexagon_8_X;
    else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 7:
    if (MCID.isBranch() ||
        (llvm::HexagonMCInstrInfo::getType(MCII, MI) == HexagonII::TypeCR))
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_B7_PCREL_X
                            : Hexagon::fixup_Hexagon_B7_PCREL;
    else if (*Extended)
      FixupKind = Hexagon::fixup_Hexagon_7_X;
    else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 6:
    if (*Extended) {
      switch (kind) {
      default:
        FixupKind = Hexagon::fixup_Hexagon_6_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_PCREL:
        FixupKind = Hexagon::fixup_Hexagon_6_PCREL_X;
        break;
      // This is part of an extender, GOT_11 is a
      // Word32_U6 unsigned/truncated reloc.
      case llvm::MCSymbolRefExpr::VK_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GOT_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_GOTOFF:
        FixupKind = Hexagon::fixup_Hexagon_GOTREL_11_X;
        break;
      }
    } else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 0:
    FixupKind = getFixupNoBits(MCII, MI, MO, kind);
    break;
  }

  MCFixup fixup =
      MCFixup::create(*Addend, MO.getExpr(), MCFixupKind(FixupKind));
  Fixups.push_back(fixup);
  // All of the information is in the fixup.
  return (0);
}

unsigned
HexagonMCCodeEmitter::getMachineOpValue(MCInst const &MI, MCOperand const &MO,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        MCSubtargetInfo const &STI) const {
  if (MO.isReg())
    return MCT.getRegisterInfo()->getEncodingValue(MO.getReg());
  if (MO.isImm())
    return static_cast<unsigned>(MO.getImm());

  // MO must be an ME.
  assert(MO.isExpr());
  return getExprOpValue(MI, MO, MO.getExpr(), Fixups, STI);
}

MCCodeEmitter *llvm::createHexagonMCCodeEmitter(MCInstrInfo const &MII,
                                                MCRegisterInfo const &MRI,
                                                MCContext &MCT) {
  return new HexagonMCCodeEmitter(MII, MCT);
}

#include "HexagonGenMCCodeEmitter.inc"
