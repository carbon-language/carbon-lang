//===-- HexagonDisassembler.cpp - Disassembler for Hexagon ISA ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonMCInst.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "hexagon-disassembler"

// Pull DecodeStatus and its enum values into the global namespace.
typedef llvm::MCDisassembler::DecodeStatus DecodeStatus;

namespace {
/// \brief Hexagon disassembler for all Hexagon platforms.
class HexagonDisassembler : public MCDisassembler {
public:
  HexagonDisassembler(MCSubtargetInfo const &STI, MCContext &Ctx)
      : MCDisassembler(STI, Ctx) {}

  DecodeStatus getInstruction(MCInst &Instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &VStream,
                              raw_ostream &CStream) const override;
};
}

static DecodeStatus DecodeModRegsRegisterClass(MCInst &Inst, unsigned RegNo,
  uint64_t Address, const void *Decoder);
static DecodeStatus DecodeCtrRegsRegisterClass(MCInst &Inst, unsigned RegNo,
  uint64_t Address, const void *Decoder);

static const uint16_t IntRegDecoderTable[] = {
  Hexagon::R0, Hexagon::R1, Hexagon::R2, Hexagon::R3, Hexagon::R4,
  Hexagon::R5, Hexagon::R6, Hexagon::R7, Hexagon::R8, Hexagon::R9,
  Hexagon::R10, Hexagon::R11, Hexagon::R12, Hexagon::R13, Hexagon::R14,
  Hexagon::R15, Hexagon::R16, Hexagon::R17, Hexagon::R18, Hexagon::R19,
  Hexagon::R20, Hexagon::R21, Hexagon::R22, Hexagon::R23, Hexagon::R24,
  Hexagon::R25, Hexagon::R26, Hexagon::R27, Hexagon::R28, Hexagon::R29,
  Hexagon::R30, Hexagon::R31 };

static const uint16_t PredRegDecoderTable[] = { Hexagon::P0, Hexagon::P1,
Hexagon::P2, Hexagon::P3 };

static DecodeStatus DecodeRegisterClass(MCInst &Inst, unsigned RegNo,
  const uint16_t Table[], size_t Size) {
  if (RegNo < Size) {
    Inst.addOperand(MCOperand::CreateReg(Table[RegNo]));
    return MCDisassembler::Success;
  }
  else
    return MCDisassembler::Fail;
}

static DecodeStatus DecodeIntRegsRegisterClass(MCInst &Inst, unsigned RegNo,
  uint64_t /*Address*/,
  void const *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  unsigned Register = IntRegDecoderTable[RegNo];
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeCtrRegsRegisterClass(MCInst &Inst, unsigned RegNo,
  uint64_t /*Address*/, const void *Decoder) {
  static const uint16_t CtrlRegDecoderTable[] = {
    Hexagon::SA0, Hexagon::LC0, Hexagon::SA1, Hexagon::LC1,
    Hexagon::P3_0, Hexagon::NoRegister, Hexagon::C6, Hexagon::C7,
    Hexagon::USR, Hexagon::PC, Hexagon::UGP, Hexagon::GP,
    Hexagon::CS0, Hexagon::CS1, Hexagon::UPCL, Hexagon::UPCH
  };

  if (RegNo >= sizeof(CtrlRegDecoderTable) / sizeof(CtrlRegDecoderTable[0]))
    return MCDisassembler::Fail;

  if (CtrlRegDecoderTable[RegNo] == Hexagon::NoRegister)
    return MCDisassembler::Fail;

  unsigned Register = CtrlRegDecoderTable[RegNo];
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeModRegsRegisterClass(MCInst &Inst, unsigned RegNo,
  uint64_t /*Address*/, const void *Decoder) {
  unsigned Register = 0;
  switch (RegNo) {
  case 0:
    Register = Hexagon::M0;
    break;
  case 1:
    Register = Hexagon::M1;
    break;
  default:
    return MCDisassembler::Fail;
  }
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeDoubleRegsRegisterClass(MCInst &Inst, unsigned RegNo,
  uint64_t /*Address*/, const void *Decoder) {
  static const uint16_t DoubleRegDecoderTable[] = {
    Hexagon::D0, Hexagon::D1, Hexagon::D2, Hexagon::D3,
    Hexagon::D4, Hexagon::D5, Hexagon::D6, Hexagon::D7,
    Hexagon::D8, Hexagon::D9, Hexagon::D10, Hexagon::D11,
    Hexagon::D12, Hexagon::D13, Hexagon::D14, Hexagon::D15
  };

  return (DecodeRegisterClass(Inst, RegNo >> 1,
    DoubleRegDecoderTable,
    sizeof (DoubleRegDecoderTable)));
}

static DecodeStatus DecodePredRegsRegisterClass(MCInst &Inst, unsigned RegNo,
  uint64_t /*Address*/,
  void const *Decoder) {
  if (RegNo > 3)
    return MCDisassembler::Fail;

  unsigned Register = PredRegDecoderTable[RegNo];
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
}

#include "HexagonGenDisassemblerTables.inc"

static MCDisassembler *createHexagonDisassembler(Target const &T,
                                                 MCSubtargetInfo const &STI,
                                                 MCContext &Ctx) {
  return new HexagonDisassembler(STI, Ctx);
}

extern "C" void LLVMInitializeHexagonDisassembler() {
  TargetRegistry::RegisterMCDisassembler(TheHexagonTarget,
                                         createHexagonDisassembler);
}

DecodeStatus HexagonDisassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                                 ArrayRef<uint8_t> Bytes,
                                                 uint64_t Address,
                                                 raw_ostream &os,
                                                 raw_ostream &cs) const {
  Size = 4;
  if (Bytes.size() < 4)
    return MCDisassembler::Fail;

  uint32_t insn =
      llvm::support::endian::read<uint32_t, llvm::support::little,
                                  llvm::support::unaligned>(Bytes.data());

  // Remove parse bits.
  insn &= ~static_cast<uint32_t>(HexagonII::InstParseBits::INST_PARSE_MASK);
  DecodeStatus Result = decodeInstruction(DecoderTable32, MI, insn, Address, this, STI);
  HexagonMCInst::AppendImplicitOperands(MI);
  return Result;
}
