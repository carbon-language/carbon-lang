//===-- HexagonDisassembler.cpp - Disassembler for Hexagon ISA ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/Endian.h"

#include <vector>
#include <array>

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

  DecodeStatus getInstruction(MCInst &instr, uint64_t &size,
                              MemoryObject const &region, uint64_t address,
                              raw_ostream &vStream, raw_ostream &cStream) const override;
};
}

static const uint16_t IntRegDecoderTable[] = {
    Hexagon::R0,  Hexagon::R1,  Hexagon::R2,  Hexagon::R3,  Hexagon::R4,
    Hexagon::R5,  Hexagon::R6,  Hexagon::R7,  Hexagon::R8,  Hexagon::R9,
    Hexagon::R10, Hexagon::R11, Hexagon::R12, Hexagon::R13, Hexagon::R14,
    Hexagon::R15, Hexagon::R16, Hexagon::R17, Hexagon::R18, Hexagon::R19,
    Hexagon::R20, Hexagon::R21, Hexagon::R22, Hexagon::R23, Hexagon::R24,
    Hexagon::R25, Hexagon::R26, Hexagon::R27, Hexagon::R28, Hexagon::R29,
    Hexagon::R30, Hexagon::R31};

static const uint16_t PredRegDecoderTable[] = {Hexagon::P0, Hexagon::P1,
                                               Hexagon::P2, Hexagon::P3};

static DecodeStatus DecodeIntRegsRegisterClass(MCInst &Inst, unsigned RegNo,
                                               uint64_t /*Address*/,
                                               void const *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  unsigned Register = IntRegDecoderTable[RegNo];
  Inst.addOperand(MCOperand::CreateReg(Register));
  return MCDisassembler::Success;
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
                                                 MemoryObject const &Region,
                                                 uint64_t Address,
                                                 raw_ostream &os,
                                                 raw_ostream &cs) const {
  std::array<uint8_t, 4> Bytes;
  Size = 4;
  if (Region.readBytes(Address, Bytes.size(), Bytes.data()) == -1) {
    return MCDisassembler::Fail;
  }
  uint32_t insn =
      llvm::support::endian::read<uint32_t, llvm::support::little,
                                  llvm::support::unaligned>(Bytes.data());

  // Remove parse bits.
  insn &= ~static_cast<uint32_t>(HexagonII::InstParseBits::INST_PARSE_MASK);
  return decodeInstruction(DecoderTable32, MI, insn, Address, this, STI);
}
