//===-- SystemZDisassembler.cpp - Disassembler for SystemZ ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SystemZ.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

typedef MCDisassembler::DecodeStatus DecodeStatus;

namespace {
class SystemZDisassembler : public MCDisassembler {
public:
  SystemZDisassembler(const MCSubtargetInfo &STI)
    : MCDisassembler(STI) {}
  virtual ~SystemZDisassembler() {}

  // Override MCDisassembler.
  virtual DecodeStatus getInstruction(MCInst &instr,
                                      uint64_t &size,
                                      const MemoryObject &region,
                                      uint64_t address,
                                      raw_ostream &vStream,
                                      raw_ostream &cStream) const override;
};
} // end anonymous namespace

static MCDisassembler *createSystemZDisassembler(const Target &T,
                                                 const MCSubtargetInfo &STI) {
  return new SystemZDisassembler(STI);
}

extern "C" void LLVMInitializeSystemZDisassembler() {
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(TheSystemZTarget,
                                         createSystemZDisassembler);
}

static DecodeStatus decodeRegisterClass(MCInst &Inst, uint64_t RegNo,
                                        const unsigned *Regs) {
  assert(RegNo < 16 && "Invalid register");
  RegNo = Regs[RegNo];
  if (RegNo == 0)
    return MCDisassembler::Fail;
  Inst.addOperand(MCOperand::CreateReg(RegNo));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGR32BitRegisterClass(MCInst &Inst, uint64_t RegNo,
                                               uint64_t Address,
                                               const void *Decoder) {
  return decodeRegisterClass(Inst, RegNo, SystemZMC::GR32Regs);
}

static DecodeStatus DecodeGRH32BitRegisterClass(MCInst &Inst, uint64_t RegNo,
                                                uint64_t Address,
                                                const void *Decoder) {
  return decodeRegisterClass(Inst, RegNo, SystemZMC::GRH32Regs);
}

static DecodeStatus DecodeGR64BitRegisterClass(MCInst &Inst, uint64_t RegNo,
                                               uint64_t Address,
                                               const void *Decoder) {
  return decodeRegisterClass(Inst, RegNo, SystemZMC::GR64Regs);
}

static DecodeStatus DecodeGR128BitRegisterClass(MCInst &Inst, uint64_t RegNo,
                                                uint64_t Address,
                                                const void *Decoder) {
  return decodeRegisterClass(Inst, RegNo, SystemZMC::GR128Regs);
}

static DecodeStatus DecodeADDR64BitRegisterClass(MCInst &Inst, uint64_t RegNo,
                                                 uint64_t Address,
                                                 const void *Decoder) {
  return decodeRegisterClass(Inst, RegNo, SystemZMC::GR64Regs);
}

static DecodeStatus DecodeFP32BitRegisterClass(MCInst &Inst, uint64_t RegNo,
                                               uint64_t Address,
                                               const void *Decoder) {
  return decodeRegisterClass(Inst, RegNo, SystemZMC::FP32Regs);
}

static DecodeStatus DecodeFP64BitRegisterClass(MCInst &Inst, uint64_t RegNo,
                                               uint64_t Address,
                                               const void *Decoder) {
  return decodeRegisterClass(Inst, RegNo, SystemZMC::FP64Regs);
}

static DecodeStatus DecodeFP128BitRegisterClass(MCInst &Inst, uint64_t RegNo,
                                                uint64_t Address,
                                                const void *Decoder) {
  return decodeRegisterClass(Inst, RegNo, SystemZMC::FP128Regs);
}

template<unsigned N>
static DecodeStatus decodeUImmOperand(MCInst &Inst, uint64_t Imm) {
  assert(isUInt<N>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::CreateImm(Imm));
  return MCDisassembler::Success;
}

template<unsigned N>
static DecodeStatus decodeSImmOperand(MCInst &Inst, uint64_t Imm) {
  assert(isUInt<N>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::CreateImm(SignExtend64<N>(Imm)));
  return MCDisassembler::Success;
}

static DecodeStatus decodeAccessRegOperand(MCInst &Inst, uint64_t Imm,
                                           uint64_t Address,
                                           const void *Decoder) {
  return decodeUImmOperand<4>(Inst, Imm);
}

static DecodeStatus decodeU4ImmOperand(MCInst &Inst, uint64_t Imm,
                                       uint64_t Address, const void *Decoder) {
  return decodeUImmOperand<4>(Inst, Imm);
}

static DecodeStatus decodeU6ImmOperand(MCInst &Inst, uint64_t Imm,
                                       uint64_t Address, const void *Decoder) {
  return decodeUImmOperand<6>(Inst, Imm);
}

static DecodeStatus decodeU8ImmOperand(MCInst &Inst, uint64_t Imm,
                                       uint64_t Address, const void *Decoder) {
  return decodeUImmOperand<8>(Inst, Imm);
}

static DecodeStatus decodeU16ImmOperand(MCInst &Inst, uint64_t Imm,
                                        uint64_t Address, const void *Decoder) {
  return decodeUImmOperand<16>(Inst, Imm);
}

static DecodeStatus decodeU32ImmOperand(MCInst &Inst, uint64_t Imm,
                                        uint64_t Address, const void *Decoder) {
  return decodeUImmOperand<32>(Inst, Imm);
}

static DecodeStatus decodeS8ImmOperand(MCInst &Inst, uint64_t Imm,
                                       uint64_t Address, const void *Decoder) {
  return decodeSImmOperand<8>(Inst, Imm);
}

static DecodeStatus decodeS16ImmOperand(MCInst &Inst, uint64_t Imm,
                                        uint64_t Address, const void *Decoder) {
  return decodeSImmOperand<16>(Inst, Imm);
}

static DecodeStatus decodeS32ImmOperand(MCInst &Inst, uint64_t Imm,
                                        uint64_t Address, const void *Decoder) {
  return decodeSImmOperand<32>(Inst, Imm);
}

template<unsigned N>
static DecodeStatus decodePCDBLOperand(MCInst &Inst, uint64_t Imm,
                                       uint64_t Address) {
  assert(isUInt<N>(Imm) && "Invalid PC-relative offset");
  Inst.addOperand(MCOperand::CreateImm(SignExtend64<N>(Imm) * 2 + Address));
  return MCDisassembler::Success;
}

static DecodeStatus decodePC16DBLOperand(MCInst &Inst, uint64_t Imm,
                                         uint64_t Address,
                                         const void *Decoder) {
  return decodePCDBLOperand<16>(Inst, Imm, Address);
}

static DecodeStatus decodePC32DBLOperand(MCInst &Inst, uint64_t Imm,
                                         uint64_t Address,
                                         const void *Decoder) {
  return decodePCDBLOperand<32>(Inst, Imm, Address);
}

static DecodeStatus decodeBDAddr12Operand(MCInst &Inst, uint64_t Field,
                                          const unsigned *Regs) {
  uint64_t Base = Field >> 12;
  uint64_t Disp = Field & 0xfff;
  assert(Base < 16 && "Invalid BDAddr12");
  Inst.addOperand(MCOperand::CreateReg(Base == 0 ? 0 : Regs[Base]));
  Inst.addOperand(MCOperand::CreateImm(Disp));
  return MCDisassembler::Success;
}

static DecodeStatus decodeBDAddr20Operand(MCInst &Inst, uint64_t Field,
                                          const unsigned *Regs) {
  uint64_t Base = Field >> 20;
  uint64_t Disp = ((Field << 12) & 0xff000) | ((Field >> 8) & 0xfff);
  assert(Base < 16 && "Invalid BDAddr20");
  Inst.addOperand(MCOperand::CreateReg(Base == 0 ? 0 : Regs[Base]));
  Inst.addOperand(MCOperand::CreateImm(SignExtend64<20>(Disp)));
  return MCDisassembler::Success;
}

static DecodeStatus decodeBDXAddr12Operand(MCInst &Inst, uint64_t Field,
                                           const unsigned *Regs) {
  uint64_t Index = Field >> 16;
  uint64_t Base = (Field >> 12) & 0xf;
  uint64_t Disp = Field & 0xfff;
  assert(Index < 16 && "Invalid BDXAddr12");
  Inst.addOperand(MCOperand::CreateReg(Base == 0 ? 0 : Regs[Base]));
  Inst.addOperand(MCOperand::CreateImm(Disp));
  Inst.addOperand(MCOperand::CreateReg(Index == 0 ? 0 : Regs[Index]));
  return MCDisassembler::Success;
}

static DecodeStatus decodeBDXAddr20Operand(MCInst &Inst, uint64_t Field,
                                           const unsigned *Regs) {
  uint64_t Index = Field >> 24;
  uint64_t Base = (Field >> 20) & 0xf;
  uint64_t Disp = ((Field & 0xfff00) >> 8) | ((Field & 0xff) << 12);
  assert(Index < 16 && "Invalid BDXAddr20");
  Inst.addOperand(MCOperand::CreateReg(Base == 0 ? 0 : Regs[Base]));
  Inst.addOperand(MCOperand::CreateImm(SignExtend64<20>(Disp)));
  Inst.addOperand(MCOperand::CreateReg(Index == 0 ? 0 : Regs[Index]));
  return MCDisassembler::Success;
}

static DecodeStatus decodeBDLAddr12Len8Operand(MCInst &Inst, uint64_t Field,
                                               const unsigned *Regs) {
  uint64_t Length = Field >> 16;
  uint64_t Base = (Field >> 12) & 0xf;
  uint64_t Disp = Field & 0xfff;
  assert(Length < 256 && "Invalid BDLAddr12Len8");
  Inst.addOperand(MCOperand::CreateReg(Base == 0 ? 0 : Regs[Base]));
  Inst.addOperand(MCOperand::CreateImm(Disp));
  Inst.addOperand(MCOperand::CreateImm(Length + 1));
  return MCDisassembler::Success;
}

static DecodeStatus decodeBDAddr32Disp12Operand(MCInst &Inst, uint64_t Field,
                                                uint64_t Address,
                                                const void *Decoder) {
  return decodeBDAddr12Operand(Inst, Field, SystemZMC::GR32Regs);
}

static DecodeStatus decodeBDAddr32Disp20Operand(MCInst &Inst, uint64_t Field,
                                                uint64_t Address,
                                                const void *Decoder) {
  return decodeBDAddr20Operand(Inst, Field, SystemZMC::GR32Regs);
}

static DecodeStatus decodeBDAddr64Disp12Operand(MCInst &Inst, uint64_t Field,
                                                uint64_t Address,
                                                const void *Decoder) {
  return decodeBDAddr12Operand(Inst, Field, SystemZMC::GR64Regs);
}

static DecodeStatus decodeBDAddr64Disp20Operand(MCInst &Inst, uint64_t Field,
                                                uint64_t Address,
                                                const void *Decoder) {
  return decodeBDAddr20Operand(Inst, Field, SystemZMC::GR64Regs);
}

static DecodeStatus decodeBDXAddr64Disp12Operand(MCInst &Inst, uint64_t Field,
                                                 uint64_t Address,
                                                 const void *Decoder) {
  return decodeBDXAddr12Operand(Inst, Field, SystemZMC::GR64Regs);
}

static DecodeStatus decodeBDXAddr64Disp20Operand(MCInst &Inst, uint64_t Field,
                                                 uint64_t Address,
                                                 const void *Decoder) {
  return decodeBDXAddr20Operand(Inst, Field, SystemZMC::GR64Regs);
}

static DecodeStatus decodeBDLAddr64Disp12Len8Operand(MCInst &Inst,
                                                     uint64_t Field,
                                                     uint64_t Address,
                                                     const void *Decoder) {
  return decodeBDLAddr12Len8Operand(Inst, Field, SystemZMC::GR64Regs);
}

#include "SystemZGenDisassemblerTables.inc"

DecodeStatus SystemZDisassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                                 const MemoryObject &Region,
                                                 uint64_t Address,
                                                 raw_ostream &os,
                                                 raw_ostream &cs) const {
  // Get the first two bytes of the instruction.
  uint8_t Bytes[6];
  Size = 0;
  if (Region.readBytes(Address, 2, Bytes) == -1)
    return MCDisassembler::Fail;

  // The top 2 bits of the first byte specify the size.
  const uint8_t *Table;
  if (Bytes[0] < 0x40) {
    Size = 2;
    Table = DecoderTable16;
  } else if (Bytes[0] < 0xc0) {
    Size = 4;
    Table = DecoderTable32;
  } else {
    Size = 6;
    Table = DecoderTable48;
  }

  // Read any remaining bytes.
  if (Size > 2 && Region.readBytes(Address + 2, Size - 2, Bytes + 2) == -1)
    return MCDisassembler::Fail;

  // Construct the instruction.
  uint64_t Inst = 0;
  for (uint64_t I = 0; I < Size; ++I)
    Inst = (Inst << 8) | Bytes[I];

  return decodeInstruction(Table, MI, Inst, Address, this, STI);
}
