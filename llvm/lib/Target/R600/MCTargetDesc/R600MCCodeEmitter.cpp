//===- R600MCCodeEmitter.cpp - Code Emitter for R600->Cayman GPU families -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// This code emitter outputs bytecode that is understood by the r600g driver
/// in the Mesa [1] project.  The bytecode is very similar to the hardware's ISA,
/// but it still needs to be run through a finalizer in order to be executed
/// by the GPU.
///
/// [1] http://www.mesa3d.org/
//
//===----------------------------------------------------------------------===//

#include "R600Defines.h"
#include "MCTargetDesc/AMDGPUMCCodeEmitter.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"
#include <stdio.h>

#define SRC_BYTE_COUNT 11
#define DST_BYTE_COUNT 5

using namespace llvm;

namespace {

class R600MCCodeEmitter : public AMDGPUMCCodeEmitter {
  R600MCCodeEmitter(const R600MCCodeEmitter &) LLVM_DELETED_FUNCTION;
  void operator=(const R600MCCodeEmitter &) LLVM_DELETED_FUNCTION;
  const MCInstrInfo &MCII;
  const MCRegisterInfo &MRI;
  const MCSubtargetInfo &STI;
  MCContext &Ctx;

public:

  R600MCCodeEmitter(const MCInstrInfo &mcii, const MCRegisterInfo &mri,
                    const MCSubtargetInfo &sti, MCContext &ctx)
    : MCII(mcii), MRI(mri), STI(sti), Ctx(ctx) { }

  /// \brief Encode the instruction and write it to the OS.
  virtual void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;

  /// \returns the encoding for an MCOperand.
  virtual uint64_t getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                     SmallVectorImpl<MCFixup> &Fixups) const;
private:

  void EmitALUInstr(const MCInst &MI, SmallVectorImpl<MCFixup> &Fixups,
                    raw_ostream &OS) const;
  void EmitSrc(const MCInst &MI, unsigned OpIdx, raw_ostream &OS) const;
  void EmitSrcISA(const MCInst &MI, unsigned RegOpIdx, unsigned SelOpIdx,
                    raw_ostream &OS) const;
  void EmitDst(const MCInst &MI, raw_ostream &OS) const;
  void EmitFCInstr(const MCInst &MI, raw_ostream &OS) const;

  void EmitNullBytes(unsigned int byteCount, raw_ostream &OS) const;

  void EmitByte(unsigned int byte, raw_ostream &OS) const;

  void EmitTwoBytes(uint32_t bytes, raw_ostream &OS) const;

  void Emit(uint32_t value, raw_ostream &OS) const;
  void Emit(uint64_t value, raw_ostream &OS) const;

  unsigned getHWRegChan(unsigned reg) const;
  unsigned getHWReg(unsigned regNo) const;

  bool isFCOp(unsigned opcode) const;
  bool isTexOp(unsigned opcode) const;
  bool isFlagSet(const MCInst &MI, unsigned Operand, unsigned Flag) const;

};

} // End anonymous namespace

enum RegElement {
  ELEMENT_X = 0,
  ELEMENT_Y,
  ELEMENT_Z,
  ELEMENT_W
};

enum InstrTypes {
  INSTR_ALU = 0,
  INSTR_TEX,
  INSTR_FC,
  INSTR_NATIVE,
  INSTR_VTX,
  INSTR_EXPORT,
  INSTR_CFALU
};

enum FCInstr {
  FC_IF_PREDICATE = 0,
  FC_ELSE,
  FC_ENDIF,
  FC_BGNLOOP,
  FC_ENDLOOP,
  FC_BREAK_PREDICATE,
  FC_CONTINUE
};

enum TextureTypes {
  TEXTURE_1D = 1,
  TEXTURE_2D,
  TEXTURE_3D,
  TEXTURE_CUBE,
  TEXTURE_RECT,
  TEXTURE_SHADOW1D,
  TEXTURE_SHADOW2D,
  TEXTURE_SHADOWRECT,
  TEXTURE_1D_ARRAY,
  TEXTURE_2D_ARRAY,
  TEXTURE_SHADOW1D_ARRAY,
  TEXTURE_SHADOW2D_ARRAY
};

MCCodeEmitter *llvm::createR600MCCodeEmitter(const MCInstrInfo &MCII,
                                           const MCRegisterInfo &MRI,
                                           const MCSubtargetInfo &STI,
                                           MCContext &Ctx) {
  return new R600MCCodeEmitter(MCII, MRI, STI, Ctx);
}

void R600MCCodeEmitter::EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                                       SmallVectorImpl<MCFixup> &Fixups) const {
  if (isFCOp(MI.getOpcode())){
    EmitFCInstr(MI, OS);
  } else if (MI.getOpcode() == AMDGPU::RETURN ||
    MI.getOpcode() == AMDGPU::BUNDLE ||
    MI.getOpcode() == AMDGPU::KILL) {
    return;
  } else {
    switch(MI.getOpcode()) {
    case AMDGPU::RAT_WRITE_CACHELESS_32_eg:
    case AMDGPU::RAT_WRITE_CACHELESS_128_eg: {
      uint64_t inst = getBinaryCodeForInstr(MI, Fixups);
      EmitByte(INSTR_NATIVE, OS);
      Emit(inst, OS);
      break;
    }
    case AMDGPU::CONSTANT_LOAD_eg:
    case AMDGPU::VTX_READ_PARAM_8_eg:
    case AMDGPU::VTX_READ_PARAM_16_eg:
    case AMDGPU::VTX_READ_PARAM_32_eg:
    case AMDGPU::VTX_READ_PARAM_128_eg:
    case AMDGPU::VTX_READ_GLOBAL_8_eg:
    case AMDGPU::VTX_READ_GLOBAL_32_eg:
    case AMDGPU::VTX_READ_GLOBAL_128_eg:
    case AMDGPU::TEX_VTX_CONSTBUF:
    case AMDGPU::TEX_VTX_TEXBUF : {
      uint64_t InstWord01 = getBinaryCodeForInstr(MI, Fixups);
      uint32_t InstWord2 = MI.getOperand(2).getImm(); // Offset

      EmitByte(INSTR_VTX, OS);
      Emit(InstWord01, OS);
      Emit(InstWord2, OS);
      break;
    }
    case AMDGPU::TEX_LD:
    case AMDGPU::TEX_GET_TEXTURE_RESINFO:
    case AMDGPU::TEX_SAMPLE:
    case AMDGPU::TEX_SAMPLE_C:
    case AMDGPU::TEX_SAMPLE_L:
    case AMDGPU::TEX_SAMPLE_C_L:
    case AMDGPU::TEX_SAMPLE_LB:
    case AMDGPU::TEX_SAMPLE_C_LB:
    case AMDGPU::TEX_SAMPLE_G:
    case AMDGPU::TEX_SAMPLE_C_G:
    case AMDGPU::TEX_GET_GRADIENTS_H:
    case AMDGPU::TEX_GET_GRADIENTS_V:
    case AMDGPU::TEX_SET_GRADIENTS_H:
    case AMDGPU::TEX_SET_GRADIENTS_V: {
      unsigned Opcode = MI.getOpcode();
      bool HasOffsets = (Opcode == AMDGPU::TEX_LD);
      unsigned OpOffset = HasOffsets ? 3 : 0;
      int64_t Sampler = MI.getOperand(OpOffset + 3).getImm();
      int64_t TextureType = MI.getOperand(OpOffset + 4).getImm();

      uint32_t SrcSelect[4] = {0, 1, 2, 3};
      uint32_t Offsets[3] = {0, 0, 0};
      uint64_t CoordType[4] = {1, 1, 1, 1};

      if (HasOffsets)
        for (unsigned i = 0; i < 3; i++) {
          int SignedOffset = MI.getOperand(i + 2).getImm();
          Offsets[i] = (SignedOffset & 0x1F);
        }
          

      if (TextureType == TEXTURE_RECT ||
          TextureType == TEXTURE_SHADOWRECT) {
        CoordType[ELEMENT_X] = 0;
        CoordType[ELEMENT_Y] = 0;
      }

      if (TextureType == TEXTURE_1D_ARRAY ||
          TextureType == TEXTURE_SHADOW1D_ARRAY) {
        if (Opcode == AMDGPU::TEX_SAMPLE_C_L ||
            Opcode == AMDGPU::TEX_SAMPLE_C_LB) {
          CoordType[ELEMENT_Y] = 0;
        } else {
          CoordType[ELEMENT_Z] = 0;
          SrcSelect[ELEMENT_Z] = ELEMENT_Y;
        }
      } else if (TextureType == TEXTURE_2D_ARRAY ||
          TextureType == TEXTURE_SHADOW2D_ARRAY) {
        CoordType[ELEMENT_Z] = 0;
      }


      if ((TextureType == TEXTURE_SHADOW1D ||
          TextureType == TEXTURE_SHADOW2D ||
          TextureType == TEXTURE_SHADOWRECT ||
          TextureType == TEXTURE_SHADOW1D_ARRAY) &&
          Opcode != AMDGPU::TEX_SAMPLE_C_L &&
          Opcode != AMDGPU::TEX_SAMPLE_C_LB) {
        SrcSelect[ELEMENT_W] = ELEMENT_Z;
      }

      uint64_t Word01 = getBinaryCodeForInstr(MI, Fixups) |
          CoordType[ELEMENT_X] << 60 | CoordType[ELEMENT_Y] << 61 |
          CoordType[ELEMENT_Z] << 62 | CoordType[ELEMENT_W] << 63;
      uint32_t Word2 = Sampler << 15 | SrcSelect[ELEMENT_X] << 20 |
          SrcSelect[ELEMENT_Y] << 23 | SrcSelect[ELEMENT_Z] << 26 |
          SrcSelect[ELEMENT_W] << 29 | Offsets[0] << 0 | Offsets[1] << 5 |
          Offsets[2] << 10;

      EmitByte(INSTR_TEX, OS);
      Emit(Word01, OS);
      Emit(Word2, OS);
      break;
    }
    case AMDGPU::CF_ALU:
    case AMDGPU::CF_ALU_PUSH_BEFORE: {
      uint64_t Inst = getBinaryCodeForInstr(MI, Fixups);
      EmitByte(INSTR_CFALU, OS);
      Emit(Inst, OS);
      break;
    }
    case AMDGPU::CF_TC_EG:
    case AMDGPU::CF_VC_EG:
    case AMDGPU::CF_CALL_FS_EG:
    case AMDGPU::CF_TC_R600:
    case AMDGPU::CF_VC_R600:
    case AMDGPU::CF_CALL_FS_R600:
      return;
    case AMDGPU::WHILE_LOOP_EG:
    case AMDGPU::END_LOOP_EG:
    case AMDGPU::LOOP_BREAK_EG:
    case AMDGPU::CF_CONTINUE_EG:
    case AMDGPU::CF_JUMP_EG:
    case AMDGPU::CF_ELSE_EG:
    case AMDGPU::POP_EG:
    case AMDGPU::WHILE_LOOP_R600:
    case AMDGPU::END_LOOP_R600:
    case AMDGPU::LOOP_BREAK_R600:
    case AMDGPU::CF_CONTINUE_R600:
    case AMDGPU::CF_JUMP_R600:
    case AMDGPU::CF_ELSE_R600:
    case AMDGPU::POP_R600:
    case AMDGPU::EG_ExportSwz:
    case AMDGPU::R600_ExportSwz:
    case AMDGPU::EG_ExportBuf:
    case AMDGPU::R600_ExportBuf:
    case AMDGPU::PAD:
    case AMDGPU::CF_END_R600:
    case AMDGPU::CF_END_EG:
    case AMDGPU::CF_END_CM: {
      uint64_t Inst = getBinaryCodeForInstr(MI, Fixups);
      EmitByte(INSTR_NATIVE, OS);
      Emit(Inst, OS);
      break;
    }
    default:
      EmitALUInstr(MI, Fixups, OS);
      break;
    }
  }
}

void R600MCCodeEmitter::EmitALUInstr(const MCInst &MI,
                                     SmallVectorImpl<MCFixup> &Fixups,
                                     raw_ostream &OS) const {
  const MCInstrDesc &MCDesc = MCII.get(MI.getOpcode());

  // Emit instruction type
  EmitByte(INSTR_ALU, OS);

  uint64_t InstWord01 = getBinaryCodeForInstr(MI, Fixups);

  //older alu have different encoding for instructions with one or two src
  //parameters.
  if ((STI.getFeatureBits() & AMDGPU::FeatureR600ALUInst) &&
      !(MCDesc.TSFlags & R600_InstFlag::OP3)) {
    uint64_t ISAOpCode = InstWord01 & (0x3FFULL << 39);
    InstWord01 &= ~(0x3FFULL << 39);
    InstWord01 |= ISAOpCode << 1;
  }

  unsigned SrcNum = MCDesc.TSFlags & R600_InstFlag::OP3 ? 3 :
      MCDesc.TSFlags & R600_InstFlag::OP2 ? 2 : 1;

  EmitByte(SrcNum, OS);

  const unsigned SrcOps[3][2] = {
      {R600Operands::SRC0, R600Operands::SRC0_SEL},
      {R600Operands::SRC1, R600Operands::SRC1_SEL},
      {R600Operands::SRC2, R600Operands::SRC2_SEL}
  };

  for (unsigned SrcIdx = 0; SrcIdx < SrcNum; ++SrcIdx) {
    unsigned RegOpIdx = R600Operands::ALUOpTable[SrcNum-1][SrcOps[SrcIdx][0]];
    unsigned SelOpIdx = R600Operands::ALUOpTable[SrcNum-1][SrcOps[SrcIdx][1]];
    EmitSrcISA(MI, RegOpIdx, SelOpIdx, OS);
  }

  Emit(InstWord01, OS);
  return;
}

void R600MCCodeEmitter::EmitSrc(const MCInst &MI, unsigned OpIdx,
                                raw_ostream &OS) const {
  const MCOperand &MO = MI.getOperand(OpIdx);
  union {
    float f;
    uint32_t i;
  } Value;
  Value.i = 0;
  // Emit the source select (2 bytes).  For GPRs, this is the register index.
  // For other potential instruction operands, (e.g. constant registers) the
  // value of the source select is defined in the r600isa docs.
  if (MO.isReg()) {
    unsigned reg = MO.getReg();
    EmitTwoBytes(getHWReg(reg), OS);
    if (reg == AMDGPU::ALU_LITERAL_X) {
      unsigned ImmOpIndex = MI.getNumOperands() - 1;
      MCOperand ImmOp = MI.getOperand(ImmOpIndex);
      if (ImmOp.isFPImm()) {
        Value.f = ImmOp.getFPImm();
      } else {
        assert(ImmOp.isImm());
        Value.i = ImmOp.getImm();
      }
    }
  } else {
    // XXX: Handle other operand types.
    EmitTwoBytes(0, OS);
  }

  // Emit the source channel (1 byte)
  if (MO.isReg()) {
    EmitByte(getHWRegChan(MO.getReg()), OS);
  } else {
    EmitByte(0, OS);
  }

  // XXX: Emit isNegated (1 byte)
  if ((!(isFlagSet(MI, OpIdx, MO_FLAG_ABS)))
      && (isFlagSet(MI, OpIdx, MO_FLAG_NEG) ||
     (MO.isReg() &&
      (MO.getReg() == AMDGPU::NEG_ONE || MO.getReg() == AMDGPU::NEG_HALF)))){
    EmitByte(1, OS);
  } else {
    EmitByte(0, OS);
  }

  // Emit isAbsolute (1 byte)
  if (isFlagSet(MI, OpIdx, MO_FLAG_ABS)) {
    EmitByte(1, OS);
  } else {
    EmitByte(0, OS);
  }

  // XXX: Emit relative addressing mode (1 byte)
  EmitByte(0, OS);

  // Emit kc_bank, This will be adjusted later by r600_asm
  EmitByte(0, OS);

  // Emit the literal value, if applicable (4 bytes).
  Emit(Value.i, OS);

}

void R600MCCodeEmitter::EmitSrcISA(const MCInst &MI, unsigned RegOpIdx,
                                   unsigned SelOpIdx, raw_ostream &OS) const {
  const MCOperand &RegMO = MI.getOperand(RegOpIdx);
  const MCOperand &SelMO = MI.getOperand(SelOpIdx);

  union {
    float f;
    uint32_t i;
  } InlineConstant;
  InlineConstant.i = 0;
  // Emit source type (1 byte) and source select (4 bytes). For GPRs type is 0
  // and select is 0 (GPR index is encoded in the instr encoding. For constants
  // type is 1 and select is the original const select passed from the driver.
  unsigned Reg = RegMO.getReg();
  if (Reg == AMDGPU::ALU_CONST) {
    EmitByte(1, OS);
    uint32_t Sel = SelMO.getImm();
    Emit(Sel, OS);
  } else {
    EmitByte(0, OS);
    Emit((uint32_t)0, OS);
  }

  if (Reg == AMDGPU::ALU_LITERAL_X) {
    unsigned ImmOpIndex = MI.getNumOperands() - 1;
    MCOperand ImmOp = MI.getOperand(ImmOpIndex);
    if (ImmOp.isFPImm()) {
      InlineConstant.f = ImmOp.getFPImm();
    } else {
      assert(ImmOp.isImm());
      InlineConstant.i = ImmOp.getImm();
    }
  }

  // Emit the literal value, if applicable (4 bytes).
  Emit(InlineConstant.i, OS);
}

void R600MCCodeEmitter::EmitFCInstr(const MCInst &MI, raw_ostream &OS) const {

  // Emit instruction type
  EmitByte(INSTR_FC, OS);

  // Emit SRC
  unsigned NumOperands = MI.getNumOperands();
  if (NumOperands > 0) {
    assert(NumOperands == 1);
    EmitSrc(MI, 0, OS);
  } else {
    EmitNullBytes(SRC_BYTE_COUNT, OS);
  }

  // Emit FC Instruction
  enum FCInstr instr;
  switch (MI.getOpcode()) {
  case AMDGPU::PREDICATED_BREAK:
    instr = FC_BREAK_PREDICATE;
    break;
  case AMDGPU::CONTINUE:
    instr = FC_CONTINUE;
    break;
  case AMDGPU::IF_PREDICATE_SET:
    instr = FC_IF_PREDICATE;
    break;
  case AMDGPU::ELSE:
    instr = FC_ELSE;
    break;
  case AMDGPU::ENDIF:
    instr = FC_ENDIF;
    break;
  case AMDGPU::ENDLOOP:
    instr = FC_ENDLOOP;
    break;
  case AMDGPU::WHILELOOP:
    instr = FC_BGNLOOP;
    break;
  default:
    abort();
    break;
  }
  EmitByte(instr, OS);
}

void R600MCCodeEmitter::EmitNullBytes(unsigned int ByteCount,
                                      raw_ostream &OS) const {

  for (unsigned int i = 0; i < ByteCount; i++) {
    EmitByte(0, OS);
  }
}

void R600MCCodeEmitter::EmitByte(unsigned int Byte, raw_ostream &OS) const {
  OS.write((uint8_t) Byte & 0xff);
}

void R600MCCodeEmitter::EmitTwoBytes(unsigned int Bytes,
                                     raw_ostream &OS) const {
  OS.write((uint8_t) (Bytes & 0xff));
  OS.write((uint8_t) ((Bytes >> 8) & 0xff));
}

void R600MCCodeEmitter::Emit(uint32_t Value, raw_ostream &OS) const {
  for (unsigned i = 0; i < 4; i++) {
    OS.write((uint8_t) ((Value >> (8 * i)) & 0xff));
  }
}

void R600MCCodeEmitter::Emit(uint64_t Value, raw_ostream &OS) const {
  for (unsigned i = 0; i < 8; i++) {
    EmitByte((Value >> (8 * i)) & 0xff, OS);
  }
}

unsigned R600MCCodeEmitter::getHWRegChan(unsigned reg) const {
  return MRI.getEncodingValue(reg) >> HW_CHAN_SHIFT;
}

unsigned R600MCCodeEmitter::getHWReg(unsigned RegNo) const {
  return MRI.getEncodingValue(RegNo) & HW_REG_MASK;
}

uint64_t R600MCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                              const MCOperand &MO,
                                        SmallVectorImpl<MCFixup> &Fixup) const {
  if (MO.isReg()) {
    if (HAS_NATIVE_OPERANDS(MCII.get(MI.getOpcode()).TSFlags)) {
      return MRI.getEncodingValue(MO.getReg());
    } else {
      return getHWReg(MO.getReg());
    }
  } else if (MO.isImm()) {
    return MO.getImm();
  } else {
    assert(0);
    return 0;
  }
}

//===----------------------------------------------------------------------===//
// Encoding helper functions
//===----------------------------------------------------------------------===//

bool R600MCCodeEmitter::isFCOp(unsigned opcode) const {
  switch(opcode) {
  default: return false;
  case AMDGPU::PREDICATED_BREAK:
  case AMDGPU::CONTINUE:
  case AMDGPU::IF_PREDICATE_SET:
  case AMDGPU::ELSE:
  case AMDGPU::ENDIF:
  case AMDGPU::ENDLOOP:
  case AMDGPU::WHILELOOP:
    return true;
  }
}

bool R600MCCodeEmitter::isTexOp(unsigned opcode) const {
  switch(opcode) {
  default: return false;
  case AMDGPU::TEX_LD:
  case AMDGPU::TEX_GET_TEXTURE_RESINFO:
  case AMDGPU::TEX_SAMPLE:
  case AMDGPU::TEX_SAMPLE_C:
  case AMDGPU::TEX_SAMPLE_L:
  case AMDGPU::TEX_SAMPLE_C_L:
  case AMDGPU::TEX_SAMPLE_LB:
  case AMDGPU::TEX_SAMPLE_C_LB:
  case AMDGPU::TEX_SAMPLE_G:
  case AMDGPU::TEX_SAMPLE_C_G:
  case AMDGPU::TEX_GET_GRADIENTS_H:
  case AMDGPU::TEX_GET_GRADIENTS_V:
  case AMDGPU::TEX_SET_GRADIENTS_H:
  case AMDGPU::TEX_SET_GRADIENTS_V:
    return true;
  }
}

bool R600MCCodeEmitter::isFlagSet(const MCInst &MI, unsigned Operand,
                                  unsigned Flag) const {
  const MCInstrDesc &MCDesc = MCII.get(MI.getOpcode());
  unsigned FlagIndex = GET_FLAG_OPERAND_IDX(MCDesc.TSFlags);
  if (FlagIndex == 0) {
    return false;
  }
  assert(MI.getOperand(FlagIndex).isImm());
  return !!((MI.getOperand(FlagIndex).getImm() >>
            (NUM_MO_FLAGS * Operand)) & Flag);
}

#include "AMDGPUGenMCCodeEmitter.inc"
