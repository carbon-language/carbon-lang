//===-- R600CodeEmitter.cpp - Code Emitter for R600->Cayman GPU families --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This code emitters outputs bytecode that is understood by the r600g driver
// in the Mesa [1] project.  The bytecode is very similar to the hardware's ISA,
// except that the size of the instruction fields are rounded up to the
// nearest byte.
//
// [1] http://www.mesa3d.org/
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUUtil.h"
#include "AMDILCodeEmitter.h"
#include "AMDILInstrInfo.h"
#include "AMDILUtilityFunctions.h"
#include "R600InstrInfo.h"
#include "R600RegisterInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetMachine.h"

#include <stdio.h>

#define SRC_BYTE_COUNT 11
#define DST_BYTE_COUNT 5

using namespace llvm;

namespace {

class R600CodeEmitter : public MachineFunctionPass, public AMDILCodeEmitter {

private:

  static char ID;
  formatted_raw_ostream &_OS;
  const TargetMachine * TM;
  const MachineRegisterInfo * MRI;
  const R600RegisterInfo * TRI;

  bool IsCube;
  bool IsReduction;
  bool IsVector;
  unsigned currentElement;
  bool IsLast;

public:

  R600CodeEmitter(formatted_raw_ostream &OS) : MachineFunctionPass(ID),
      _OS(OS), TM(NULL), IsCube(false), IsReduction(false), IsVector(false),
      IsLast(true) { }

  const char *getPassName() const { return "AMDGPU Machine Code Emitter"; }

  bool runOnMachineFunction(MachineFunction &MF);
  virtual uint64_t getMachineOpValue(const MachineInstr &MI,
                                     const MachineOperand &MO) const;

private:

  void EmitALUInstr(MachineInstr  &MI);
  void EmitSrc(const MachineOperand & MO, int chan_override  = -1);
  void EmitDst(const MachineOperand & MO);
  void EmitALU(MachineInstr &MI, unsigned numSrc);
  void EmitTexInstr(MachineInstr &MI);
  void EmitFCInstr(MachineInstr &MI);

  void EmitNullBytes(unsigned int byteCount);

  void EmitByte(unsigned int byte);

  void EmitTwoBytes(uint32_t bytes);

  void Emit(uint32_t value);
  void Emit(uint64_t value);

  unsigned getHWReg(unsigned regNo) const;

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
  INSTR_VTX
};

enum FCInstr {
  FC_IF = 0,
  FC_ELSE,
  FC_ENDIF,
  FC_BGNLOOP,
  FC_ENDLOOP,
  FC_BREAK,
  FC_BREAK_NZ_INT,
  FC_CONTINUE,
  FC_BREAK_Z_INT
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

char R600CodeEmitter::ID = 0;

FunctionPass *llvm::createR600CodeEmitterPass(formatted_raw_ostream &OS) {
  return new R600CodeEmitter(OS);
}

bool R600CodeEmitter::runOnMachineFunction(MachineFunction &MF) {

  TM = &MF.getTarget();
  MRI = &MF.getRegInfo();
  TRI = static_cast<const R600RegisterInfo *>(TM->getRegisterInfo());
  const R600InstrInfo * TII = static_cast<const R600InstrInfo *>(TM->getInstrInfo());
  const AMDILSubtarget &STM = TM->getSubtarget<AMDILSubtarget>();
  std::string gpu = STM.getDeviceName();

  if (STM.dumpCode()) {
    MF.dump();
  }

  for (MachineFunction::iterator BB = MF.begin(), BB_E = MF.end();
                                                  BB != BB_E; ++BB) {
     MachineBasicBlock &MBB = *BB;
     for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
                                                       I != E; ++I) {
          MachineInstr &MI = *I;
	  IsReduction = AMDGPU::isReductionOp(MI.getOpcode());
	  IsVector = TII->isVector(MI);
	  IsCube = AMDGPU::isCubeOp(MI.getOpcode());
          if (MI.getNumOperands() > 1 && MI.getOperand(0).isReg() && MI.getOperand(0).isDead()) {
            continue;
          }
          if (AMDGPU::isTexOp(MI.getOpcode())) {
            EmitTexInstr(MI);
          } else if (AMDGPU::isFCOp(MI.getOpcode())){
            EmitFCInstr(MI);
          } else if (IsReduction || IsVector || IsCube) {
            IsLast = false;
            // XXX: On Cayman, some (all?) of the vector instructions only need
            // to fill the first three slots.
            for (currentElement = 0; currentElement < 4; currentElement++) {
              IsLast = (currentElement == 3);
              EmitALUInstr(MI);
            }
            IsReduction = false;
	    IsVector = false;
	    IsCube = false;
          } else if (MI.getOpcode() == AMDGPU::RETURN ||
                     MI.getOpcode() == AMDGPU::BUNDLE ||
                     MI.getOpcode() == AMDGPU::KILL) {
            continue;
          } else {
            switch(MI.getOpcode()) {
            case AMDGPU::RAT_WRITE_CACHELESS_eg:
              {
                  uint64_t inst = getBinaryCodeForInstr(MI);
                // Set End Of Program bit
                // XXX: Need better check of end of program.  EOP should be
                // encoded in one of the operands of the MI, and it should be
                // set in a prior pass.
                MachineBasicBlock::iterator NextI = llvm::next(I);
                MachineInstr &NextMI = *NextI;
                if (NextMI.getOpcode() == AMDGPU::RETURN) {
                  inst |= (((uint64_t)1) << 53);
                }
                EmitByte(INSTR_NATIVE);
                Emit(inst);
                break;
              }
            case AMDGPU::VTX_READ_PARAM_i32_eg:
            case AMDGPU::VTX_READ_PARAM_f32_eg:
            case AMDGPU::VTX_READ_GLOBAL_i32_eg:
            case AMDGPU::VTX_READ_GLOBAL_f32_eg:
            case AMDGPU::VTX_READ_GLOBAL_v4i32_eg:
            case AMDGPU::VTX_READ_GLOBAL_v4f32_eg:
              {
                uint64_t InstWord01 = getBinaryCodeForInstr(MI);
                uint32_t InstWord2 = MI.getOperand(2).getImm(); // Offset

                EmitByte(INSTR_VTX);
                Emit(InstWord01);
                Emit(InstWord2);
                break;
              }

            default:
              EmitALUInstr(MI);
              break;
          }
        }
    }
  }
  return false;
}

void R600CodeEmitter::EmitALUInstr(MachineInstr &MI)
{

  unsigned numOperands = MI.getNumExplicitOperands();

   // Some instructions are just place holder instructions that represent
   // operations that the GPU does automatically.  They should be ignored.
  if (AMDGPU::isPlaceHolderOpcode(MI.getOpcode())) {
    return;
  }

  // XXX Check if instruction writes a result
  if (numOperands < 1) {
    return;
  }
  const MachineOperand dstOp = MI.getOperand(0);

  // Emit instruction type
  EmitByte(0);

  if (IsCube) {
    static const int cube_src_swz[] = {2, 2, 0, 1};
    EmitSrc(MI.getOperand(1), cube_src_swz[currentElement]);
    EmitSrc(MI.getOperand(1), cube_src_swz[3-currentElement]);
    EmitNullBytes(SRC_BYTE_COUNT);
  } else {
    unsigned int opIndex;
    for (opIndex = 1; opIndex < numOperands; opIndex++) {
      // Literal constants are always stored as the last operand.
      if (MI.getOperand(opIndex).isImm() || MI.getOperand(opIndex).isFPImm()) {
        break;
      }
      EmitSrc(MI.getOperand(opIndex));
    }

    // Emit zeros for unused sources
    for ( ; opIndex < 4; opIndex++) {
      EmitNullBytes(SRC_BYTE_COUNT);
    }
  }

  EmitDst(dstOp);

  EmitALU(MI, numOperands - 1);
}

void R600CodeEmitter::EmitSrc(const MachineOperand & MO, int chan_override)
{
  uint32_t value = 0;
  // Emit the source select (2 bytes).  For GPRs, this is the register index.
  // For other potential instruction operands, (e.g. constant registers) the
  // value of the source select is defined in the r600isa docs.
  if (MO.isReg()) {
    unsigned reg = MO.getReg();
    EmitTwoBytes(getHWReg(reg));
    if (reg == AMDGPU::ALU_LITERAL_X) {
      const MachineInstr * parent = MO.getParent();
      unsigned immOpIndex = parent->getNumExplicitOperands() - 1;
      MachineOperand immOp = parent->getOperand(immOpIndex);
      if (immOp.isFPImm()) {
        value = immOp.getFPImm()->getValueAPF().bitcastToAPInt().getZExtValue();
      } else {
        assert(immOp.isImm());
        value = immOp.getImm();
      }
    }
  } else {
    // XXX: Handle other operand types.
    EmitTwoBytes(0);
  }

  // Emit the source channel (1 byte)
  if (chan_override != -1) {
    EmitByte(chan_override);
  } else if (IsReduction) {
    EmitByte(currentElement);
  } else if (MO.isReg()) {
    EmitByte(TRI->getHWRegChan(MO.getReg()));
  } else {
    EmitByte(0);
  }

  // XXX: Emit isNegated (1 byte)
  if ((!(MO.getTargetFlags() & MO_FLAG_ABS))
      && (MO.getTargetFlags() & MO_FLAG_NEG ||
     (MO.isReg() &&
      (MO.getReg() == AMDGPU::NEG_ONE || MO.getReg() == AMDGPU::NEG_HALF)))){
    EmitByte(1);
  } else {
    EmitByte(0);
  }

  // Emit isAbsolute (1 byte)
  if (MO.getTargetFlags() & MO_FLAG_ABS) {
    EmitByte(1);
  } else {
    EmitByte(0);
  }

  // XXX: Emit relative addressing mode (1 byte)
  EmitByte(0);

  // Emit kc_bank, This will be adjusted later by r600_asm
  EmitByte(0);

  // Emit the literal value, if applicable (4 bytes).
  Emit(value);

}

void R600CodeEmitter::EmitDst(const MachineOperand & MO)
{
  if (MO.isReg()) {
    // Emit the destination register index (1 byte)
    EmitByte(getHWReg(MO.getReg()));

    // Emit the element of the destination register (1 byte)
    if (IsReduction || IsCube || IsVector) {
      EmitByte(currentElement);
    } else {
      EmitByte(TRI->getHWRegChan(MO.getReg()));
    }

    // Emit isClamped (1 byte)
    if (MO.getTargetFlags() & MO_FLAG_CLAMP) {
      EmitByte(1);
    } else {
      EmitByte(0);
    }

    // Emit writemask (1 byte).
    if (((IsReduction || IsVector) &&
          currentElement != TRI->getHWRegChan(MO.getReg()))
       || MO.getTargetFlags() & MO_FLAG_MASK) {
      EmitByte(0);
    } else {
      EmitByte(1);
    }

    // XXX: Emit relative addressing mode
    EmitByte(0);
  } else {
    // XXX: Handle other operand types.  Are there any for destination regs?
    EmitNullBytes(DST_BYTE_COUNT);
  }
}

void R600CodeEmitter::EmitALU(MachineInstr &MI, unsigned numSrc)
{
  // Emit the instruction (2 bytes)
  EmitTwoBytes(getBinaryCodeForInstr(MI));

  // Emit IsLast (for this instruction group) (1 byte)
  if (IsLast) {
    EmitByte(1);
  } else {
    EmitByte(0);
  }
  // Emit isOp3 (1 byte)
  if (numSrc == 3) {
    EmitByte(1);
  } else {
    EmitByte(0);
  }

  // XXX: Emit predicate (1 byte)
  EmitByte(0);

  // XXX: Emit bank swizzle. (1 byte)  Do we need this?  It looks like
  // r600_asm.c sets it.
  EmitByte(0);

  // XXX: Emit bank_swizzle_force (1 byte) Not sure what this is for.
  EmitByte(0);

  // XXX: Emit OMOD (1 byte) Not implemented.
  EmitByte(0);

  // XXX: Emit index_mode.  I think this is for indirect addressing, so we
  // don't need to worry about it.
  EmitByte(0);
}

void R600CodeEmitter::EmitTexInstr(MachineInstr &MI)
{

  unsigned opcode = MI.getOpcode();
  bool hasOffsets = (opcode == AMDGPU::TEX_LD);
  unsigned op_offset = hasOffsets ? 3 : 0;
  int64_t sampler = MI.getOperand(op_offset+2).getImm();
  int64_t textureType = MI.getOperand(op_offset+3).getImm();
  unsigned srcSelect[4] = {0, 1, 2, 3};

  // Emit instruction type
  EmitByte(1);

  // Emit instruction
  EmitByte(getBinaryCodeForInstr(MI));

  // XXX: Emit resource id r600_shader.c uses sampler + 1.  Why?
  EmitByte(sampler + 1 + 1);

  // Emit source register
  EmitByte(getHWReg(MI.getOperand(1).getReg()));

  // XXX: Emit src isRelativeAddress
  EmitByte(0);

  // Emit destination register
  EmitByte(getHWReg(MI.getOperand(0).getReg()));

  // XXX: Emit dst isRealtiveAddress
  EmitByte(0);

  // XXX: Emit dst select
  EmitByte(0); // X
  EmitByte(1); // Y
  EmitByte(2); // Z
  EmitByte(3); // W

  // XXX: Emit lod bias
  EmitByte(0);

  // XXX: Emit coord types
  unsigned coordType[4] = {1, 1, 1, 1};

  if (textureType == TEXTURE_RECT
      || textureType == TEXTURE_SHADOWRECT) {
    coordType[ELEMENT_X] = 0;
    coordType[ELEMENT_Y] = 0;
  }

  if (textureType == TEXTURE_1D_ARRAY
      || textureType == TEXTURE_SHADOW1D_ARRAY) {
    if (opcode == AMDGPU::TEX_SAMPLE_C_L || opcode == AMDGPU::TEX_SAMPLE_C_LB) {
      coordType[ELEMENT_Y] = 0;
    } else {
      coordType[ELEMENT_Z] = 0;
      srcSelect[ELEMENT_Z] = ELEMENT_Y;
    }
  } else if (textureType == TEXTURE_2D_ARRAY
             || textureType == TEXTURE_SHADOW2D_ARRAY) {
    coordType[ELEMENT_Z] = 0;
  }

  for (unsigned i = 0; i < 4; i++) {
    EmitByte(coordType[i]);
  }

  // XXX: Emit offsets
  if (hasOffsets)
	  for (unsigned i = 2; i < 5; i++)
		  EmitByte(MI.getOperand(i).getImm()<<1);
  else
	  EmitNullBytes(3);

  // Emit sampler id
  EmitByte(sampler);

  // XXX:Emit source select
  if ((textureType == TEXTURE_SHADOW1D
      || textureType == TEXTURE_SHADOW2D
      || textureType == TEXTURE_SHADOWRECT
      || textureType == TEXTURE_SHADOW1D_ARRAY)
      && opcode != AMDGPU::TEX_SAMPLE_C_L
      && opcode != AMDGPU::TEX_SAMPLE_C_LB) {
    srcSelect[ELEMENT_W] = ELEMENT_Z;
  }

  for (unsigned i = 0; i < 4; i++) {
    EmitByte(srcSelect[i]);
  }
}

void R600CodeEmitter::EmitFCInstr(MachineInstr &MI)
{
  // Emit instruction type
  EmitByte(INSTR_FC);

  // Emit SRC
  unsigned numOperands = MI.getNumOperands();
  if (numOperands > 0) {
    assert(numOperands == 1);
    EmitSrc(MI.getOperand(0));
  } else {
    EmitNullBytes(SRC_BYTE_COUNT);
  }

  // Emit FC Instruction
  enum FCInstr instr;
  switch (MI.getOpcode()) {
  case AMDGPU::BREAK_LOGICALZ_f32:
    instr = FC_BREAK;
    break;
  case AMDGPU::BREAK_LOGICALNZ_f32:
  case AMDGPU::BREAK_LOGICALNZ_i32:
    instr = FC_BREAK_NZ_INT;
    break;
  case AMDGPU::BREAK_LOGICALZ_i32:
    instr = FC_BREAK_Z_INT;
    break;
  case AMDGPU::CONTINUE_LOGICALNZ_f32:
  case AMDGPU::CONTINUE_LOGICALNZ_i32:
    instr = FC_CONTINUE;
    break;
  case AMDGPU::IF_LOGICALNZ_f32:
  case AMDGPU::IF_LOGICALNZ_i32:
    instr = FC_IF;
    break;
  case AMDGPU::IF_LOGICALZ_f32:
    abort();
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
  EmitByte(instr);
}

void R600CodeEmitter::EmitNullBytes(unsigned int byteCount)
{
  for (unsigned int i = 0; i < byteCount; i++) {
    EmitByte(0);
  }
}

void R600CodeEmitter::EmitByte(unsigned int byte)
{
  _OS.write((uint8_t) byte & 0xff);
}
void R600CodeEmitter::EmitTwoBytes(unsigned int bytes)
{
  _OS.write((uint8_t) (bytes & 0xff));
  _OS.write((uint8_t) ((bytes >> 8) & 0xff));
}

void R600CodeEmitter::Emit(uint32_t value)
{
  for (unsigned i = 0; i < 4; i++) {
    _OS.write((uint8_t) ((value >> (8 * i)) & 0xff));
  }
}

void R600CodeEmitter::Emit(uint64_t value)
{
  for (unsigned i = 0; i < 8; i++) {
    EmitByte((value >> (8 * i)) & 0xff);
  }
}

unsigned R600CodeEmitter::getHWReg(unsigned regNo) const
{
  unsigned HWReg;

  HWReg = TRI->getEncodingValue(regNo);
  if (AMDGPU::R600_CReg32RegClass.contains(regNo)) {
    HWReg += 512;
  }
  return HWReg;
}

uint64_t R600CodeEmitter::getMachineOpValue(const MachineInstr &MI,
                                            const MachineOperand &MO) const
{
  if (MO.isReg()) {
    return getHWReg(MO.getReg());
  } else {
    return MO.getImm();
  }
}

#include "AMDGPUGenCodeEmitter.inc"

