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
