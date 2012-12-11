//===-- SIAssignInterpRegs.cpp - Assign interpolation registers -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief This pass maps the pseudo interpolation registers to the correct physical
/// registers.
//
/// Prior to executing a fragment shader, the GPU loads interpolation
/// parameters into physical registers.  The specific physical register that each
/// interpolation parameter ends up in depends on the type of the interpolation
/// parameter as well as how many interpolation parameters are used by the
/// shader.
//
//===----------------------------------------------------------------------===//



#include "AMDGPU.h"
#include "AMDIL.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

namespace {

class SIAssignInterpRegsPass : public MachineFunctionPass {

private:
  static char ID;
  TargetMachine &TM;

  void addLiveIn(MachineFunction * MF,  MachineRegisterInfo & MRI,
                 unsigned physReg, unsigned virtReg);

public:
  SIAssignInterpRegsPass(TargetMachine &tm) :
    MachineFunctionPass(ID), TM(tm) { }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  const char *getPassName() const { return "SI Assign intrpolation registers"; }
};

} // End anonymous namespace

char SIAssignInterpRegsPass::ID = 0;

#define INTERP_VALUES 16
#define REQUIRED_VALUE_MAX_INDEX 7

struct InterpInfo {
  bool Enabled;
  unsigned Regs[3];
  unsigned RegCount;
};


FunctionPass *llvm::createSIAssignInterpRegsPass(TargetMachine &tm) {
  return new SIAssignInterpRegsPass(tm);
}

bool SIAssignInterpRegsPass::runOnMachineFunction(MachineFunction &MF) {

  struct InterpInfo InterpUse[INTERP_VALUES] = {
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
  // This pass is only needed for pixel shaders.
  if (MFI->ShaderType != ShaderType::PIXEL) {
    return false;
  }
  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool ForceEnable = true;

  // First pass, mark the interpolation values that are used.
  for (unsigned InterpIdx = 0; InterpIdx < INTERP_VALUES; InterpIdx++) {
    for (unsigned RegIdx = 0; RegIdx < InterpUse[InterpIdx].RegCount;
                                                               RegIdx++) {
      InterpUse[InterpIdx].Enabled = InterpUse[InterpIdx].Enabled ||
                            !MRI.use_empty(InterpUse[InterpIdx].Regs[RegIdx]);
      if (InterpUse[InterpIdx].Enabled &&
          InterpIdx <= REQUIRED_VALUE_MAX_INDEX) {
        ForceEnable = false;
      }
    }
  }

  // At least one interpolation mode must be enabled or else the GPU will hang.
  if (ForceEnable) {
    InterpUse[0].Enabled = true;
  }

  unsigned UsedVgprs = 0;

  // Second pass, replace with VGPRs.
  for (unsigned InterpIdx = 0; InterpIdx < INTERP_VALUES; InterpIdx++) {
    if (!InterpUse[InterpIdx].Enabled) {
      continue;
    }
    MFI->SPIPSInputAddr |= (1 << InterpIdx);

    for (unsigned RegIdx = 0; RegIdx < InterpUse[InterpIdx].RegCount;
                                                  RegIdx++, UsedVgprs++) {
      unsigned NewReg = AMDGPU::VReg_32RegClass.getRegister(UsedVgprs);
      unsigned VirtReg = MRI.createVirtualRegister(&AMDGPU::VReg_32RegClass);
      MRI.replaceRegWith(InterpUse[InterpIdx].Regs[RegIdx], VirtReg);
      addLiveIn(&MF, MRI, NewReg, VirtReg);
    }
  }

  return false;
}

void SIAssignInterpRegsPass::addLiveIn(MachineFunction * MF,
                           MachineRegisterInfo & MRI,
                           unsigned physReg, unsigned virtReg) {
    const TargetInstrInfo * TII = TM.getInstrInfo();
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
