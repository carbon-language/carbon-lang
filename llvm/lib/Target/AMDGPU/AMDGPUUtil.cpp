//===-- AMDGPUUtil.cpp - AMDGPU Utility functions -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common utility functions used by hw codegen targets
//
//===----------------------------------------------------------------------===//

#include "AMDGPUUtil.h"
#include "AMDGPURegisterInfo.h"
#include "AMDIL.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

// Some instructions act as place holders to emulate operations that the GPU
// hardware does automatically. This function can be used to check if
// an opcode falls into this category.
bool AMDGPU::isPlaceHolderOpcode(unsigned opcode)
{
  switch (opcode) {
  default: return false;
  case AMDGPU::RETURN:
  case AMDGPU::LOAD_INPUT:
  case AMDGPU::LAST:
  case AMDGPU::MASK_WRITE:
  case AMDGPU::RESERVE_REG:
    return true;
  }
}

bool AMDGPU::isTransOp(unsigned opcode)
{
  switch(opcode) {
    default: return false;

    case AMDGPU::COS_r600:
    case AMDGPU::COS_eg:
    case AMDGPU::MULLIT:
    case AMDGPU::MUL_LIT_r600:
    case AMDGPU::MUL_LIT_eg:
    case AMDGPU::EXP_IEEE_r600:
    case AMDGPU::EXP_IEEE_eg:
    case AMDGPU::LOG_CLAMPED_r600:
    case AMDGPU::LOG_IEEE_r600:
    case AMDGPU::LOG_CLAMPED_eg:
    case AMDGPU::LOG_IEEE_eg:
      return true;
  }
}

bool AMDGPU::isTexOp(unsigned opcode)
{
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

bool AMDGPU::isReductionOp(unsigned opcode)
{
  switch(opcode) {
    default: return false;
    case AMDGPU::DOT4_r600:
    case AMDGPU::DOT4_eg:
      return true;
  }
}

bool AMDGPU::isCubeOp(unsigned opcode)
{
  switch(opcode) {
    default: return false;
    case AMDGPU::CUBE_r600:
    case AMDGPU::CUBE_eg:
      return true;
  }
}


bool AMDGPU::isFCOp(unsigned opcode)
{
  switch(opcode) {
  default: return false;
  case AMDGPU::BREAK_LOGICALZ_f32:
  case AMDGPU::BREAK_LOGICALNZ_i32:
  case AMDGPU::BREAK_LOGICALZ_i32:
  case AMDGPU::BREAK_LOGICALNZ_f32:
  case AMDGPU::CONTINUE_LOGICALNZ_f32:
  case AMDGPU::IF_LOGICALNZ_i32:
  case AMDGPU::IF_LOGICALZ_f32:
  case AMDGPU::ELSE:
  case AMDGPU::ENDIF:
  case AMDGPU::ENDLOOP:
  case AMDGPU::IF_LOGICALNZ_f32:
  case AMDGPU::WHILELOOP:
    return true;
  }
}

void AMDGPU::utilAddLiveIn(MachineFunction * MF,
                           MachineRegisterInfo & MRI,
                           const TargetInstrInfo * TII,
                           unsigned physReg, unsigned virtReg)
{
    if (!MRI.isLiveIn(physReg)) {
      MRI.addLiveIn(physReg, virtReg);
      MF->front().addLiveIn(physReg);
      BuildMI(MF->front(), MF->front().begin(), DebugLoc(),
              TII->get(TargetOpcode::COPY), virtReg)
                .addReg(physReg);
    } else {
      MRI.replaceRegWith(virtReg, MRI.getLiveInVirtReg(physReg));
    }
}
