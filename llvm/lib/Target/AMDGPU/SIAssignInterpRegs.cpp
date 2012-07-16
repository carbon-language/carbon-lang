//===-- SIAssignInterpRegs.cpp - Assign interpolation registers -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass maps the pseudo interpolation registers to the correct physical
// registers.  Prior to executing a fragment shader, the GPU loads interpolation
// parameters into physical registers.  The specific physical register that each
// interpolation parameter ends up in depends on the type of the interpolation
// parameter as well as how many interpolation parameters are used by the
// shader.
//
//===----------------------------------------------------------------------===//



#include "AMDGPU.h"
#include "AMDGPUUtil.h"
#include "AMDIL.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

namespace {

class SIAssignInterpRegsPass : public MachineFunctionPass {

private:
  static char ID;
  TargetMachine &TM;

public:
  SIAssignInterpRegsPass(TargetMachine &tm) :
    MachineFunctionPass(ID), TM(tm) { }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  const char *getPassName() const { return "SI Assign intrpolation registers"; }
};

} // End anonymous namespace

char SIAssignInterpRegsPass::ID = 0;

#define INTERP_VALUES 16

struct interp_info {
  bool enabled;
  unsigned regs[3];
  unsigned reg_count;
};


FunctionPass *llvm::createSIAssignInterpRegsPass(TargetMachine &tm) {
  return new SIAssignInterpRegsPass(tm);
}

bool SIAssignInterpRegsPass::runOnMachineFunction(MachineFunction &MF)
{

  struct interp_info InterpUse[INTERP_VALUES] = {
    {false, {AMDGPU::PERSP_SAMPLE_I, AMDGPU::PERSP_SAMPLE_J}, 2},
    {false, {AMDGPU::PERSP_CENTER_I, AMDGPU::PERSP_CENTER_J}, 2},
    {false, {AMDGPU::PERSP_CENTROID_I, AMDGPU::PERSP_CENTROID_J}, 2},
    {false, {AMDGPU::PERSP_I_W, AMDGPU::PERSP_J_W, AMDGPU::PERSP_1_W}, 3},
    {false, {AMDGPU::LINEAR_SAMPLE_I, AMDGPU::LINEAR_SAMPLE_J}, 2},
    {false, {AMDGPU::LINEAR_CENTER_I, AMDGPU::LINEAR_CENTER_J}, 2},
    {false, {AMDGPU::LINEAR_CENTROID_I, AMDGPU::LINEAR_CENTROID_J}, 2},
    {false, {AMDGPU::LINE_STIPPLE_TEX_COORD}, 1},
    {false, {AMDGPU::POS_X_FLOAT}, 1},
    {false, {AMDGPU::POS_Y_FLOAT}, 1},
    {false, {AMDGPU::POS_Z_FLOAT}, 1},
    {false, {AMDGPU::POS_W_FLOAT}, 1},
    {false, {AMDGPU::FRONT_FACE}, 1},
    {false, {AMDGPU::ANCILLARY}, 1},
    {false, {AMDGPU::SAMPLE_COVERAGE}, 1},
    {false, {AMDGPU::POS_FIXED_PT}, 1}
  };

  SIMachineFunctionInfo * MFI = MF.getInfo<SIMachineFunctionInfo>();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  /* First pass, mark the interpolation values that are used. */
  for (unsigned interp_idx = 0; interp_idx < INTERP_VALUES; interp_idx++) {
    for (unsigned reg_idx = 0; reg_idx < InterpUse[interp_idx].reg_count;
                                                               reg_idx++) {
      InterpUse[interp_idx].enabled =
                            !MRI.use_empty(InterpUse[interp_idx].regs[reg_idx]);
    }
  }

  unsigned used_vgprs = 0;

  /* Second pass, replace with VGPRs. */
  for (unsigned interp_idx = 0; interp_idx < INTERP_VALUES; interp_idx++) {
    if (!InterpUse[interp_idx].enabled) {
      continue;
    }
    MFI->spi_ps_input_addr |= (1 << interp_idx);

    for (unsigned reg_idx = 0; reg_idx < InterpUse[interp_idx].reg_count;
                                                  reg_idx++, used_vgprs++) {
      unsigned new_reg = AMDGPU::VReg_32RegClass.getRegister(used_vgprs);
      unsigned virt_reg = MRI.createVirtualRegister(&AMDGPU::VReg_32RegClass);
      MRI.replaceRegWith(InterpUse[interp_idx].regs[reg_idx], virt_reg);
      AMDGPU::utilAddLiveIn(&MF, MRI, TM.getInstrInfo(), new_reg, virt_reg);
    }
  }

  return false;
}
