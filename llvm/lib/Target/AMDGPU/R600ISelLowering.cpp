//===-- R600ISelLowering.cpp - R600 DAG Lowering Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Most of the DAG lowering is handled in AMDGPUISelLowering.cpp.  This file
// is mostly EmitInstrWithCustomInserter().
//
//===----------------------------------------------------------------------===//

#include "R600ISelLowering.h"
#include "AMDGPUUtil.h"
#include "R600InstrInfo.h"
#include "R600MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

R600TargetLowering::R600TargetLowering(TargetMachine &TM) :
    AMDGPUTargetLowering(TM),
    TII(static_cast<const R600InstrInfo*>(TM.getInstrInfo()))
{
  setOperationAction(ISD::MUL, MVT::i64, Expand);
  addRegisterClass(MVT::v4f32, &AMDGPU::R600_Reg128RegClass);
  addRegisterClass(MVT::f32, &AMDGPU::R600_Reg32RegClass);
  addRegisterClass(MVT::v4i32, &AMDGPU::R600_Reg128RegClass);
  addRegisterClass(MVT::i32, &AMDGPU::R600_Reg32RegClass);
  computeRegisterProperties();

  setOperationAction(ISD::FSUB, MVT::f32, Expand);

  setOperationAction(ISD::ROTL, MVT::i32, Custom);

  setSchedulingPreference(Sched::VLIW);
}

MachineBasicBlock * R600TargetLowering::EmitInstrWithCustomInserter(
    MachineInstr * MI, MachineBasicBlock * BB) const
{
  MachineFunction * MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  MachineBasicBlock::iterator I = *MI;

  switch (MI->getOpcode()) {
  default: return AMDGPUTargetLowering::EmitInstrWithCustomInserter(MI, BB);
  case AMDGPU::TGID_X:
    addLiveIn(MI, MF, MRI, TII, AMDGPU::T1_X);
    break;
  case AMDGPU::TGID_Y:
    addLiveIn(MI, MF, MRI, TII, AMDGPU::T1_Y);
    break;
  case AMDGPU::TGID_Z:
    addLiveIn(MI, MF, MRI, TII, AMDGPU::T1_Z);
    break;
  case AMDGPU::TIDIG_X:
    addLiveIn(MI, MF, MRI, TII, AMDGPU::T0_X);
    break;
  case AMDGPU::TIDIG_Y:
    addLiveIn(MI, MF, MRI, TII, AMDGPU::T0_Y);
    break;
  case AMDGPU::TIDIG_Z:
    addLiveIn(MI, MF, MRI, TII, AMDGPU::T0_Z);
    break;
  case AMDGPU::NGROUPS_X:
    lowerImplicitParameter(MI, *BB, MRI, 0);
    break;
  case AMDGPU::NGROUPS_Y:
    lowerImplicitParameter(MI, *BB, MRI, 1);
    break;
  case AMDGPU::NGROUPS_Z:
    lowerImplicitParameter(MI, *BB, MRI, 2);
    break;
  case AMDGPU::GLOBAL_SIZE_X:
    lowerImplicitParameter(MI, *BB, MRI, 3);
    break;
  case AMDGPU::GLOBAL_SIZE_Y:
    lowerImplicitParameter(MI, *BB, MRI, 4);
    break;
  case AMDGPU::GLOBAL_SIZE_Z:
    lowerImplicitParameter(MI, *BB, MRI, 5);
    break;
  case AMDGPU::LOCAL_SIZE_X:
    lowerImplicitParameter(MI, *BB, MRI, 6);
    break;
  case AMDGPU::LOCAL_SIZE_Y:
    lowerImplicitParameter(MI, *BB, MRI, 7);
    break;
  case AMDGPU::LOCAL_SIZE_Z:
    lowerImplicitParameter(MI, *BB, MRI, 8);
    break;

  case AMDGPU::CLAMP_R600:
    MI->getOperand(0).addTargetFlag(MO_FLAG_CLAMP);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::MOV))
           .addOperand(MI->getOperand(0))
           .addOperand(MI->getOperand(1));
    break;

  case AMDGPU::FABS_R600:
    MI->getOperand(1).addTargetFlag(MO_FLAG_ABS);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::MOV))
           .addOperand(MI->getOperand(0))
           .addOperand(MI->getOperand(1));
    break;

  case AMDGPU::FNEG_R600:
    MI->getOperand(1).addTargetFlag(MO_FLAG_NEG);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::MOV))
            .addOperand(MI->getOperand(0))
            .addOperand(MI->getOperand(1));
    break;

  case AMDGPU::R600_LOAD_CONST:
    {
      int64_t RegIndex = MI->getOperand(1).getImm();
      unsigned ConstantReg = AMDGPU::R600_CReg32RegClass.getRegister(RegIndex);
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::COPY))
                  .addOperand(MI->getOperand(0))
                  .addReg(ConstantReg);
      break;
    }

  case AMDGPU::LOAD_INPUT:
    {
      int64_t RegIndex = MI->getOperand(1).getImm();
      addLiveIn(MI, MF, MRI, TII,
                AMDGPU::R600_TReg32RegClass.getRegister(RegIndex));
      break;
    }

  case AMDGPU::MASK_WRITE:
    {
      unsigned maskedRegister = MI->getOperand(0).getReg();
      assert(TargetRegisterInfo::isVirtualRegister(maskedRegister));
      MachineInstr * defInstr = MRI.getVRegDef(maskedRegister);
      MachineOperand * def = defInstr->findRegisterDefOperand(maskedRegister);
      def->addTargetFlag(MO_FLAG_MASK);
      // Return early so the instruction is not erased
      return BB;
    }

  case AMDGPU::RAT_WRITE_CACHELESS_eg:
    {
      // Convert to DWORD address
      unsigned NewAddr = MRI.createVirtualRegister(
                                             &AMDGPU::R600_TReg32_XRegClass);
      unsigned ShiftValue = MRI.createVirtualRegister(
                                              &AMDGPU::R600_TReg32RegClass);

      // XXX In theory, we should be able to pass ShiftValue directly to
      // the LSHR_eg instruction as an inline literal, but I tried doing it
      // this way and it didn't produce the correct results.
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::MOV), ShiftValue)
              .addReg(AMDGPU::ALU_LITERAL_X)
              .addImm(2);
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::LSHR_eg), NewAddr)
              .addOperand(MI->getOperand(1))
              .addReg(ShiftValue);
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(MI->getOpcode()))
              .addOperand(MI->getOperand(0))
              .addReg(NewAddr);
      break;
    }

  case AMDGPU::STORE_OUTPUT:
    {
      int64_t OutputIndex = MI->getOperand(1).getImm();
      unsigned OutputReg = AMDGPU::R600_TReg32RegClass.getRegister(OutputIndex);

      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::COPY), OutputReg)
                  .addOperand(MI->getOperand(0));

      if (!MRI.isLiveOut(OutputReg)) {
        MRI.addLiveOut(OutputReg);
      }
      break;
    }

  case AMDGPU::RESERVE_REG:
    {
      R600MachineFunctionInfo * MFI = MF->getInfo<R600MachineFunctionInfo>();
      int64_t ReservedIndex = MI->getOperand(0).getImm();
      unsigned ReservedReg =
                          AMDGPU::R600_TReg32RegClass.getRegister(ReservedIndex);
      MFI->ReservedRegs.push_back(ReservedReg);
      break;
    }

  case AMDGPU::TXD:
    {
      unsigned t0 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);
      unsigned t1 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);

      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_H), t0)
              .addOperand(MI->getOperand(3))
              .addOperand(MI->getOperand(4))
              .addOperand(MI->getOperand(5));
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_V), t1)
              .addOperand(MI->getOperand(2))
              .addOperand(MI->getOperand(4))
              .addOperand(MI->getOperand(5));
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SAMPLE_G))
              .addOperand(MI->getOperand(0))
              .addOperand(MI->getOperand(1))
              .addOperand(MI->getOperand(4))
              .addOperand(MI->getOperand(5))
              .addReg(t0, RegState::Implicit)
              .addReg(t1, RegState::Implicit);
      break;
    }
  case AMDGPU::TXD_SHADOW:
    {
      unsigned t0 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);
      unsigned t1 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);

      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_H), t0)
              .addOperand(MI->getOperand(3))
              .addOperand(MI->getOperand(4))
              .addOperand(MI->getOperand(5));
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_V), t1)
              .addOperand(MI->getOperand(2))
              .addOperand(MI->getOperand(4))
              .addOperand(MI->getOperand(5));
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SAMPLE_C_G))
              .addOperand(MI->getOperand(0))
              .addOperand(MI->getOperand(1))
              .addOperand(MI->getOperand(4))
              .addOperand(MI->getOperand(5))
              .addReg(t0, RegState::Implicit)
              .addReg(t1, RegState::Implicit);
      break;
    }


  }

  MI->eraseFromParent();
  return BB;
}

void R600TargetLowering::lowerImplicitParameter(MachineInstr *MI, MachineBasicBlock &BB,
    MachineRegisterInfo & MRI, unsigned dword_offset) const
{
  MachineBasicBlock::iterator I = *MI;
  unsigned PtrReg = MRI.createVirtualRegister(&AMDGPU::R600_TReg32_XRegClass);
  MRI.setRegClass(MI->getOperand(0).getReg(), &AMDGPU::R600_TReg32_XRegClass);

  BuildMI(BB, I, BB.findDebugLoc(I), TII->get(AMDGPU::MOV), PtrReg)
          .addReg(AMDGPU::ALU_LITERAL_X)
          .addImm(dword_offset * 4);

  BuildMI(BB, I, BB.findDebugLoc(I), TII->get(AMDGPU::VTX_READ_PARAM_i32_eg))
          .addOperand(MI->getOperand(0))
          .addReg(PtrReg)
          .addImm(0);
}

//===----------------------------------------------------------------------===//
// Custom DAG Lowering Operations
//===----------------------------------------------------------------------===//


SDValue R600TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const
{
  switch (Op.getOpcode()) {
  default: return AMDGPUTargetLowering::LowerOperation(Op, DAG);
  case ISD::ROTL: return LowerROTL(Op, DAG);
  }
}

SDValue R600TargetLowering::LowerROTL(SDValue Op, SelectionDAG &DAG) const
{
  DebugLoc DL = Op.getDebugLoc();
  EVT VT = Op.getValueType();

  return DAG.getNode(AMDGPUISD::BITALIGN, DL, VT,
                     Op.getOperand(0),
                     Op.getOperand(0),
                     DAG.getNode(ISD::SUB, DL, VT,
                                 DAG.getConstant(32, MVT::i32),
                                 Op.getOperand(1)));
}
