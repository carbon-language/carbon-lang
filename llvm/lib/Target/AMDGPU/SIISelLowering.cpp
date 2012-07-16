//===-- SIISelLowering.cpp - SI DAG Lowering Implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Most of the DAG lowering is handled in AMDGPUISelLowering.cpp.  This file is
// mostly EmitInstrWithCustomInserter().
//
//===----------------------------------------------------------------------===//

#include "SIISelLowering.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

SITargetLowering::SITargetLowering(TargetMachine &TM) :
    AMDGPUTargetLowering(TM),
    TII(static_cast<const SIInstrInfo*>(TM.getInstrInfo()))
{
  addRegisterClass(MVT::v4f32, &AMDGPU::VReg_128RegClass);
  addRegisterClass(MVT::f32, &AMDGPU::VReg_32RegClass);
  addRegisterClass(MVT::i32, &AMDGPU::VReg_32RegClass);
  addRegisterClass(MVT::i64, &AMDGPU::VReg_64RegClass);

  addRegisterClass(MVT::v4i32, &AMDGPU::SReg_128RegClass);
  addRegisterClass(MVT::v8i32, &AMDGPU::SReg_256RegClass);

  computeRegisterProperties();

  setOperationAction(ISD::ADD, MVT::i64, Legal);
  setOperationAction(ISD::ADD, MVT::i32, Legal);

}

MachineBasicBlock * SITargetLowering::EmitInstrWithCustomInserter(
    MachineInstr * MI, MachineBasicBlock * BB) const
{
  const TargetInstrInfo * TII = getTargetMachine().getInstrInfo();
  MachineRegisterInfo & MRI = BB->getParent()->getRegInfo();
  MachineBasicBlock::iterator I = MI;

  if (TII->get(MI->getOpcode()).TSFlags & SIInstrFlags::NEED_WAIT) {
    AppendS_WAITCNT(MI, *BB, llvm::next(I));
    return BB;
  }

  switch (MI->getOpcode()) {
  default:
    return AMDGPUTargetLowering::EmitInstrWithCustomInserter(MI, BB);

  case AMDGPU::CLAMP_SI:
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::V_MOV_B32_e64))
           .addOperand(MI->getOperand(0))
           .addOperand(MI->getOperand(1))
           // VSRC1-2 are unused, but we still need to fill all the
           // operand slots, so we just reuse the VSRC0 operand
           .addOperand(MI->getOperand(1))
           .addOperand(MI->getOperand(1))
           .addImm(0) // ABS
           .addImm(1) // CLAMP
           .addImm(0) // OMOD
           .addImm(0); // NEG
    MI->eraseFromParent();
    break;

  case AMDGPU::FABS_SI:
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::V_MOV_B32_e64))
                 .addOperand(MI->getOperand(0))
                 .addOperand(MI->getOperand(1))
                 // VSRC1-2 are unused, but we still need to fill all the
                 // operand slots, so we just reuse the VSRC0 operand
                 .addOperand(MI->getOperand(1))
                 .addOperand(MI->getOperand(1))
                 .addImm(1) // ABS
                 .addImm(0) // CLAMP
                 .addImm(0) // OMOD
                 .addImm(0); // NEG
    MI->eraseFromParent();
    break;

  case AMDGPU::SI_INTERP:
    LowerSI_INTERP(MI, *BB, I, MRI);
    break;
  case AMDGPU::SI_INTERP_CONST:
    LowerSI_INTERP_CONST(MI, *BB, I);
    break;
  case AMDGPU::SI_V_CNDLT:
    LowerSI_V_CNDLT(MI, *BB, I, MRI);
    break;
  case AMDGPU::USE_SGPR_32:
  case AMDGPU::USE_SGPR_64:
    lowerUSE_SGPR(MI, BB->getParent(), MRI);
    MI->eraseFromParent();
    break;
  case AMDGPU::VS_LOAD_BUFFER_INDEX:
    addLiveIn(MI, BB->getParent(), MRI, TII, AMDGPU::VGPR0);
    MI->eraseFromParent();
    break;
  }
  return BB;
}

void SITargetLowering::AppendS_WAITCNT(MachineInstr *MI, MachineBasicBlock &BB,
    MachineBasicBlock::iterator I) const
{
  BuildMI(BB, I, BB.findDebugLoc(I), TII->get(AMDGPU::S_WAITCNT))
          .addImm(0);
}

void SITargetLowering::LowerSI_INTERP(MachineInstr *MI, MachineBasicBlock &BB,
    MachineBasicBlock::iterator I, MachineRegisterInfo & MRI) const
{
  unsigned tmp = MRI.createVirtualRegister(&AMDGPU::VReg_32RegClass);
  MachineOperand dst = MI->getOperand(0);
  MachineOperand iReg = MI->getOperand(1);
  MachineOperand jReg = MI->getOperand(2);
  MachineOperand attr_chan = MI->getOperand(3);
  MachineOperand attr = MI->getOperand(4);
  MachineOperand params = MI->getOperand(5);

  BuildMI(BB, I, BB.findDebugLoc(I), TII->get(AMDGPU::S_MOV_B32))
          .addReg(AMDGPU::M0)
          .addOperand(params);

  BuildMI(BB, I, BB.findDebugLoc(I), TII->get(AMDGPU::V_INTERP_P1_F32), tmp)
          .addOperand(iReg)
          .addOperand(attr_chan)
          .addOperand(attr);

  BuildMI(BB, I, BB.findDebugLoc(I), TII->get(AMDGPU::V_INTERP_P2_F32))
          .addOperand(dst)
          .addReg(tmp)
          .addOperand(jReg)
          .addOperand(attr_chan)
          .addOperand(attr);

  MI->eraseFromParent();
}

void SITargetLowering::LowerSI_INTERP_CONST(MachineInstr *MI,
    MachineBasicBlock &BB, MachineBasicBlock::iterator I) const
{
  MachineOperand dst = MI->getOperand(0);
  MachineOperand attr_chan = MI->getOperand(1);
  MachineOperand attr = MI->getOperand(2);
  MachineOperand params = MI->getOperand(3);

  BuildMI(BB, I, BB.findDebugLoc(I), TII->get(AMDGPU::S_MOV_B32))
          .addReg(AMDGPU::M0)
          .addOperand(params);

  BuildMI(BB, I, BB.findDebugLoc(I), TII->get(AMDGPU::V_INTERP_MOV_F32))
          .addOperand(dst)
          .addOperand(attr_chan)
          .addOperand(attr);

  MI->eraseFromParent();
}

void SITargetLowering::LowerSI_V_CNDLT(MachineInstr *MI, MachineBasicBlock &BB,
    MachineBasicBlock::iterator I, MachineRegisterInfo & MRI) const
{
  BuildMI(BB, I, BB.findDebugLoc(I), TII->get(AMDGPU::V_CMP_LT_F32_e32))
          .addOperand(MI->getOperand(1))
          .addReg(AMDGPU::SREG_LIT_0);

  BuildMI(BB, I, BB.findDebugLoc(I), TII->get(AMDGPU::V_CNDMASK_B32))
          .addOperand(MI->getOperand(0))
          .addOperand(MI->getOperand(2))
          .addOperand(MI->getOperand(3));

  MI->eraseFromParent();
}

void SITargetLowering::lowerUSE_SGPR(MachineInstr *MI,
    MachineFunction * MF, MachineRegisterInfo & MRI) const
{
  const TargetInstrInfo * TII = getTargetMachine().getInstrInfo();
  unsigned dstReg = MI->getOperand(0).getReg();
  int64_t newIndex = MI->getOperand(1).getImm();
  const TargetRegisterClass * dstClass = MRI.getRegClass(dstReg);
  unsigned DwordWidth = dstClass->getSize() / 4;
  assert(newIndex % DwordWidth == 0 && "USER_SGPR not properly aligned");
  newIndex = newIndex / DwordWidth;

  unsigned newReg = dstClass->getRegister(newIndex);
  addLiveIn(MI, MF, MRI, TII, newReg); 
}

