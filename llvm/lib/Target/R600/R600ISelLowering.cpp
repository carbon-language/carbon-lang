//===-- R600ISelLowering.cpp - R600 DAG Lowering Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Custom DAG lowering for R600
//
//===----------------------------------------------------------------------===//

#include "R600ISelLowering.h"
#include "R600Defines.h"
#include "R600InstrInfo.h"
#include "R600MachineFunctionInfo.h"
#include "llvm/Argument.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"

using namespace llvm;

R600TargetLowering::R600TargetLowering(TargetMachine &TM) :
    AMDGPUTargetLowering(TM),
    TII(static_cast<const R600InstrInfo*>(TM.getInstrInfo())) {
  setOperationAction(ISD::MUL, MVT::i64, Expand);
  addRegisterClass(MVT::v4f32, &AMDGPU::R600_Reg128RegClass);
  addRegisterClass(MVT::f32, &AMDGPU::R600_Reg32RegClass);
  addRegisterClass(MVT::v4i32, &AMDGPU::R600_Reg128RegClass);
  addRegisterClass(MVT::i32, &AMDGPU::R600_Reg32RegClass);
  computeRegisterProperties();

  setOperationAction(ISD::FADD, MVT::v4f32, Expand);
  setOperationAction(ISD::FMUL, MVT::v4f32, Expand);
  setOperationAction(ISD::FDIV, MVT::v4f32, Expand);
  setOperationAction(ISD::FSUB, MVT::v4f32, Expand);

  setOperationAction(ISD::ADD,  MVT::v4i32, Expand);
  setOperationAction(ISD::AND,  MVT::v4i32, Expand);
  setOperationAction(ISD::UDIV, MVT::v4i32, Expand);
  setOperationAction(ISD::UREM, MVT::v4i32, Expand);
  setOperationAction(ISD::SETCC, MVT::v4i32, Expand);

  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f32, Custom);

  setOperationAction(ISD::FSUB, MVT::f32, Expand);

  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i1, Custom);
  setOperationAction(ISD::FPOW, MVT::f32, Custom);

  setOperationAction(ISD::ROTL, MVT::i32, Custom);

  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);

  setOperationAction(ISD::SETCC, MVT::i32, Custom);
  setOperationAction(ISD::SETCC, MVT::f32, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i1, Custom);

  setOperationAction(ISD::SELECT, MVT::i32, Custom);
  setOperationAction(ISD::SELECT, MVT::f32, Custom);

  setOperationAction(ISD::STORE, MVT::i32, Custom);
  setOperationAction(ISD::STORE, MVT::v4i32, Custom);

  setTargetDAGCombine(ISD::FP_ROUND);

  setSchedulingPreference(Sched::VLIW);
}

MachineBasicBlock * R600TargetLowering::EmitInstrWithCustomInserter(
    MachineInstr * MI, MachineBasicBlock * BB) const {
  MachineFunction * MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  MachineBasicBlock::iterator I = *MI;

  switch (MI->getOpcode()) {
  default: return AMDGPUTargetLowering::EmitInstrWithCustomInserter(MI, BB);
  case AMDGPU::SHADER_TYPE: break;
  case AMDGPU::CLAMP_R600: {
    MachineInstr *NewMI = TII->buildDefaultInstruction(*BB, I,
                                                   AMDGPU::MOV,
                                                   MI->getOperand(0).getReg(),
                                                   MI->getOperand(1).getReg());
    TII->addFlag(NewMI, 0, MO_FLAG_CLAMP);
    break;
  }

  case AMDGPU::FABS_R600: {
    MachineInstr *NewMI = TII->buildDefaultInstruction(*BB, I,
                                                    AMDGPU::MOV,
                                                    MI->getOperand(0).getReg(),
                                                    MI->getOperand(1).getReg());
    TII->addFlag(NewMI, 0, MO_FLAG_ABS);
    break;
  }

  case AMDGPU::FNEG_R600: {
    MachineInstr *NewMI = TII->buildDefaultInstruction(*BB, I,
                                                    AMDGPU::MOV,
                                                    MI->getOperand(0).getReg(),
                                                    MI->getOperand(1).getReg());
    TII->addFlag(NewMI, 0, MO_FLAG_NEG);
    break;
  }

  case AMDGPU::R600_LOAD_CONST: {
    int64_t RegIndex = MI->getOperand(1).getImm();
    unsigned ConstantReg = AMDGPU::R600_CReg32RegClass.getRegister(RegIndex);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::COPY))
                .addOperand(MI->getOperand(0))
                .addReg(ConstantReg);
    break;
  }

  case AMDGPU::MASK_WRITE: {
    unsigned maskedRegister = MI->getOperand(0).getReg();
    assert(TargetRegisterInfo::isVirtualRegister(maskedRegister));
    MachineInstr * defInstr = MRI.getVRegDef(maskedRegister);
    TII->addFlag(defInstr, 0, MO_FLAG_MASK);
    break;
  }

  case AMDGPU::MOV_IMM_F32:
    TII->buildMovImm(*BB, I, MI->getOperand(0).getReg(),
                     MI->getOperand(1).getFPImm()->getValueAPF()
                         .bitcastToAPInt().getZExtValue());
    break;
  case AMDGPU::MOV_IMM_I32:
    TII->buildMovImm(*BB, I, MI->getOperand(0).getReg(),
                     MI->getOperand(1).getImm());
    break;


  case AMDGPU::RAT_WRITE_CACHELESS_32_eg:
  case AMDGPU::RAT_WRITE_CACHELESS_128_eg: {
    unsigned EOP = (llvm::next(I)->getOpcode() == AMDGPU::RETURN) ? 1 : 0;

    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(MI->getOpcode()))
            .addOperand(MI->getOperand(0))
            .addOperand(MI->getOperand(1))
            .addImm(EOP); // Set End of program bit
    break;
  }

  case AMDGPU::RESERVE_REG: {
    R600MachineFunctionInfo * MFI = MF->getInfo<R600MachineFunctionInfo>();
    int64_t ReservedIndex = MI->getOperand(0).getImm();
    unsigned ReservedReg =
                         AMDGPU::R600_TReg32RegClass.getRegister(ReservedIndex);
    MFI->ReservedRegs.push_back(ReservedReg);
    unsigned SuperReg =
          AMDGPU::R600_Reg128RegClass.getRegister(ReservedIndex / 4);
    MFI->ReservedRegs.push_back(SuperReg);
    break;
  }

  case AMDGPU::TXD: {
    unsigned T0 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);
    unsigned T1 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);

    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_H), T0)
            .addOperand(MI->getOperand(3))
            .addOperand(MI->getOperand(4))
            .addOperand(MI->getOperand(5))
            .addOperand(MI->getOperand(6));
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_V), T1)
            .addOperand(MI->getOperand(2))
            .addOperand(MI->getOperand(4))
            .addOperand(MI->getOperand(5))
            .addOperand(MI->getOperand(6));
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SAMPLE_G))
            .addOperand(MI->getOperand(0))
            .addOperand(MI->getOperand(1))
            .addOperand(MI->getOperand(4))
            .addOperand(MI->getOperand(5))
            .addOperand(MI->getOperand(6))
            .addReg(T0, RegState::Implicit)
            .addReg(T1, RegState::Implicit);
    break;
  }

  case AMDGPU::TXD_SHADOW: {
    unsigned T0 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);
    unsigned T1 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);

    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_H), T0)
            .addOperand(MI->getOperand(3))
            .addOperand(MI->getOperand(4))
            .addOperand(MI->getOperand(5))
            .addOperand(MI->getOperand(6));
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_V), T1)
            .addOperand(MI->getOperand(2))
            .addOperand(MI->getOperand(4))
            .addOperand(MI->getOperand(5))
            .addOperand(MI->getOperand(6));
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SAMPLE_C_G))
            .addOperand(MI->getOperand(0))
            .addOperand(MI->getOperand(1))
            .addOperand(MI->getOperand(4))
            .addOperand(MI->getOperand(5))
            .addOperand(MI->getOperand(6))
            .addReg(T0, RegState::Implicit)
            .addReg(T1, RegState::Implicit);
    break;
  }

  case AMDGPU::BRANCH:
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::JUMP))
              .addOperand(MI->getOperand(0))
              .addReg(0);
      break;

  case AMDGPU::BRANCH_COND_f32: {
    MachineInstr *NewMI =
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::PRED_X),
              AMDGPU::PREDICATE_BIT)
              .addOperand(MI->getOperand(1))
              .addImm(OPCODE_IS_NOT_ZERO)
              .addImm(0); // Flags
    TII->addFlag(NewMI, 0, MO_FLAG_PUSH);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::JUMP))
            .addOperand(MI->getOperand(0))
            .addReg(AMDGPU::PREDICATE_BIT, RegState::Kill);
    break;
  }

  case AMDGPU::BRANCH_COND_i32: {
    MachineInstr *NewMI =
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::PRED_X),
            AMDGPU::PREDICATE_BIT)
            .addOperand(MI->getOperand(1))
            .addImm(OPCODE_IS_NOT_ZERO_INT)
            .addImm(0); // Flags
    TII->addFlag(NewMI, 0, MO_FLAG_PUSH);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::JUMP))
           .addOperand(MI->getOperand(0))
            .addReg(AMDGPU::PREDICATE_BIT, RegState::Kill);
    break;
  }

  case AMDGPU::input_perspective: {
    R600MachineFunctionInfo *MFI = MF->getInfo<R600MachineFunctionInfo>();

    // XXX Be more fine about register reservation
    for (unsigned i = 0; i < 4; i ++) {
      unsigned ReservedReg = AMDGPU::R600_TReg32RegClass.getRegister(i);
      MFI->ReservedRegs.push_back(ReservedReg);
    }

    switch (MI->getOperand(1).getImm()) {
    case 0:// Perspective
      MFI->HasPerspectiveInterpolation = true;
      break;
    case 1:// Linear
      MFI->HasLinearInterpolation = true;
      break;
    default:
      assert(0 && "Unknow ij index");
    }

    return BB;
  }

  case AMDGPU::EG_ExportSwz:
  case AMDGPU::R600_ExportSwz: {
    bool EOP = (llvm::next(I)->getOpcode() == AMDGPU::RETURN)? 1 : 0;
    if (!EOP)
      return BB;
    unsigned CfInst = (MI->getOpcode() == AMDGPU::EG_ExportSwz)? 84 : 40;
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(MI->getOpcode()))
            .addOperand(MI->getOperand(0))
            .addOperand(MI->getOperand(1))
            .addOperand(MI->getOperand(2))
            .addOperand(MI->getOperand(3))
            .addOperand(MI->getOperand(4))
            .addOperand(MI->getOperand(5))
            .addOperand(MI->getOperand(6))
            .addImm(CfInst)
            .addImm(1);
    break;
  }
  }

  MI->eraseFromParent();
  return BB;
}

//===----------------------------------------------------------------------===//
// Custom DAG Lowering Operations
//===----------------------------------------------------------------------===//

using namespace llvm::Intrinsic;
using namespace llvm::AMDGPUIntrinsic;

static SDValue
InsertScalarToRegisterExport(SelectionDAG &DAG, DebugLoc DL, SDNode **ExportMap,
    unsigned Slot, unsigned Channel, unsigned Inst, unsigned Type,
    SDValue Scalar, SDValue Chain) {
  if (!ExportMap[Slot]) {
    SDValue Vector = DAG.getNode(ISD::INSERT_VECTOR_ELT,
      DL, MVT::v4f32,
      DAG.getUNDEF(MVT::v4f32),
      Scalar,
      DAG.getConstant(Channel, MVT::i32));

    unsigned Mask = 1 << Channel;

    const SDValue Ops[] = {Chain, Vector, DAG.getConstant(Inst, MVT::i32),
        DAG.getConstant(Type, MVT::i32), DAG.getConstant(Slot, MVT::i32),
        DAG.getConstant(Mask, MVT::i32)};

    SDValue Res =  DAG.getNode(
        AMDGPUISD::EXPORT,
        DL,
        MVT::Other,
        Ops, 6);
     ExportMap[Slot] = Res.getNode();
     return Res;
  }

  SDNode *ExportInstruction = (SDNode *) ExportMap[Slot] ;
  SDValue PreviousVector = ExportInstruction->getOperand(1);
  SDValue Vector = DAG.getNode(ISD::INSERT_VECTOR_ELT,
      DL, MVT::v4f32,
      PreviousVector,
      Scalar,
      DAG.getConstant(Channel, MVT::i32));

  unsigned Mask = dyn_cast<ConstantSDNode>(ExportInstruction->getOperand(5))
      ->getZExtValue();
  Mask |= (1 << Channel);

  const SDValue Ops[] = {ExportInstruction->getOperand(0), Vector,
      DAG.getConstant(Inst, MVT::i32),
      DAG.getConstant(Type, MVT::i32),
      DAG.getConstant(Slot, MVT::i32),
      DAG.getConstant(Mask, MVT::i32)};

  DAG.UpdateNodeOperands(ExportInstruction,
      Ops, 6);

  return Chain;

}

SDValue R600TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default: return AMDGPUTargetLowering::LowerOperation(Op, DAG);
  case ISD::BR_CC: return LowerBR_CC(Op, DAG);
  case ISD::ROTL: return LowerROTL(Op, DAG);
  case ISD::SELECT_CC: return LowerSELECT_CC(Op, DAG);
  case ISD::SELECT: return LowerSELECT(Op, DAG);
  case ISD::SETCC: return LowerSETCC(Op, DAG);
  case ISD::STORE: return LowerSTORE(Op, DAG);
  case ISD::FPOW: return LowerFPOW(Op, DAG);
  case ISD::INTRINSIC_VOID: {
    SDValue Chain = Op.getOperand(0);
    unsigned IntrinsicID =
                         cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
    switch (IntrinsicID) {
    case AMDGPUIntrinsic::AMDGPU_store_output: {
      MachineFunction &MF = DAG.getMachineFunction();
      MachineRegisterInfo &MRI = MF.getRegInfo();
      int64_t RegIndex = cast<ConstantSDNode>(Op.getOperand(3))->getZExtValue();
      unsigned Reg = AMDGPU::R600_TReg32RegClass.getRegister(RegIndex);
      if (!MRI.isLiveOut(Reg)) {
        MRI.addLiveOut(Reg);
      }
      return DAG.getCopyToReg(Chain, Op.getDebugLoc(), Reg, Op.getOperand(2));
    }
    case AMDGPUIntrinsic::R600_store_pixel_color: {
      MachineFunction &MF = DAG.getMachineFunction();
      R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();
      int64_t RegIndex = cast<ConstantSDNode>(Op.getOperand(3))->getZExtValue();

      SDNode **OutputsMap = MFI->Outputs;
      return InsertScalarToRegisterExport(DAG, Op.getDebugLoc(), OutputsMap,
          RegIndex / 4, RegIndex % 4, 0, 0, Op.getOperand(2),
          Chain);

    }
    case AMDGPUIntrinsic::R600_store_stream_output : {
      MachineFunction &MF = DAG.getMachineFunction();
      R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();
      int64_t RegIndex = cast<ConstantSDNode>(Op.getOperand(3))->getZExtValue();
      int64_t BufIndex = cast<ConstantSDNode>(Op.getOperand(4))->getZExtValue();

      SDNode **OutputsMap = MFI->StreamOutputs[BufIndex];
      unsigned Inst;
      switch (cast<ConstantSDNode>(Op.getOperand(4))->getZExtValue()  ) {
      // STREAM3
      case 3:
        Inst = 4;
        break;
      // STREAM2
      case 2:
        Inst = 3;
        break;
      // STREAM1
      case 1:
        Inst = 2;
        break;
      // STREAM0
      case 0:
        Inst = 1;
        break;
      default:
        llvm_unreachable("Wrong buffer id for stream outputs !");
      }

      return InsertScalarToRegisterExport(DAG, Op.getDebugLoc(), OutputsMap,
          RegIndex / 4, RegIndex % 4, Inst, 0, Op.getOperand(2),
          Chain);
    }
    // default for switch(IntrinsicID)
    default: break;
    }
    // break out of case ISD::INTRINSIC_VOID in switch(Op.getOpcode())
    break;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntrinsicID =
                         cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
    EVT VT = Op.getValueType();
    DebugLoc DL = Op.getDebugLoc();
    switch(IntrinsicID) {
    default: return AMDGPUTargetLowering::LowerOperation(Op, DAG);
    case AMDGPUIntrinsic::R600_load_input: {
      int64_t RegIndex = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
      unsigned Reg = AMDGPU::R600_TReg32RegClass.getRegister(RegIndex);
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass, Reg, VT);
    }
    case AMDGPUIntrinsic::R600_load_input_perspective: {
      int slot = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
      if (slot < 0)
        return DAG.getUNDEF(MVT::f32);
      SDValue FullVector = DAG.getNode(
          AMDGPUISD::INTERP,
          DL, MVT::v4f32,
          DAG.getConstant(0, MVT::i32), DAG.getConstant(slot / 4 , MVT::i32));
      return DAG.getNode(ISD::EXTRACT_VECTOR_ELT,
        DL, VT, FullVector, DAG.getConstant(slot % 4, MVT::i32));
    }
    case AMDGPUIntrinsic::R600_load_input_linear: {
      int slot = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
      if (slot < 0)
        return DAG.getUNDEF(MVT::f32);
      SDValue FullVector = DAG.getNode(
        AMDGPUISD::INTERP,
        DL, MVT::v4f32,
        DAG.getConstant(1, MVT::i32), DAG.getConstant(slot / 4 , MVT::i32));
      return DAG.getNode(ISD::EXTRACT_VECTOR_ELT,
        DL, VT, FullVector, DAG.getConstant(slot % 4, MVT::i32));
    }
    case AMDGPUIntrinsic::R600_load_input_constant: {
      int slot = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
      if (slot < 0)
        return DAG.getUNDEF(MVT::f32);
      SDValue FullVector = DAG.getNode(
        AMDGPUISD::INTERP_P0,
        DL, MVT::v4f32,
        DAG.getConstant(slot / 4 , MVT::i32));
      return DAG.getNode(ISD::EXTRACT_VECTOR_ELT,
          DL, VT, FullVector, DAG.getConstant(slot % 4, MVT::i32));
    }

    case r600_read_ngroups_x:
      return LowerImplicitParameter(DAG, VT, DL, 0);
    case r600_read_ngroups_y:
      return LowerImplicitParameter(DAG, VT, DL, 1);
    case r600_read_ngroups_z:
      return LowerImplicitParameter(DAG, VT, DL, 2);
    case r600_read_global_size_x:
      return LowerImplicitParameter(DAG, VT, DL, 3);
    case r600_read_global_size_y:
      return LowerImplicitParameter(DAG, VT, DL, 4);
    case r600_read_global_size_z:
      return LowerImplicitParameter(DAG, VT, DL, 5);
    case r600_read_local_size_x:
      return LowerImplicitParameter(DAG, VT, DL, 6);
    case r600_read_local_size_y:
      return LowerImplicitParameter(DAG, VT, DL, 7);
    case r600_read_local_size_z:
      return LowerImplicitParameter(DAG, VT, DL, 8);

    case r600_read_tgid_x:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T1_X, VT);
    case r600_read_tgid_y:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T1_Y, VT);
    case r600_read_tgid_z:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T1_Z, VT);
    case r600_read_tidig_x:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T0_X, VT);
    case r600_read_tidig_y:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T0_Y, VT);
    case r600_read_tidig_z:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T0_Z, VT);
    }
    // break out of case ISD::INTRINSIC_WO_CHAIN in switch(Op.getOpcode())
    break;
  }
  } // end switch(Op.getOpcode())
  return SDValue();
}

void R600TargetLowering::ReplaceNodeResults(SDNode *N,
                                            SmallVectorImpl<SDValue> &Results,
                                            SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  default: return;
  case ISD::FP_TO_UINT: Results.push_back(LowerFPTOUINT(N->getOperand(0), DAG));
  }
}

SDValue R600TargetLowering::LowerFPTOUINT(SDValue Op, SelectionDAG &DAG) const {
  return DAG.getNode(
      ISD::SETCC,
      Op.getDebugLoc(),
      MVT::i1,
      Op, DAG.getConstantFP(0.0f, MVT::f32),
      DAG.getCondCode(ISD::SETNE)
      );
}

SDValue R600TargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue CC = Op.getOperand(1);
  SDValue LHS   = Op.getOperand(2);
  SDValue RHS   = Op.getOperand(3);
  SDValue JumpT  = Op.getOperand(4);
  SDValue CmpValue;
  SDValue Result;

  if (LHS.getValueType() == MVT::i32) {
    CmpValue = DAG.getNode(
        ISD::SELECT_CC,
        Op.getDebugLoc(),
        MVT::i32,
        LHS, RHS,
        DAG.getConstant(-1, MVT::i32),
        DAG.getConstant(0, MVT::i32),
        CC);
  } else if (LHS.getValueType() == MVT::f32) {
    CmpValue = DAG.getNode(
        ISD::SELECT_CC,
        Op.getDebugLoc(),
        MVT::f32,
        LHS, RHS,
        DAG.getConstantFP(1.0f, MVT::f32),
        DAG.getConstantFP(0.0f, MVT::f32),
        CC);
  } else {
    assert(0 && "Not valid type for br_cc");
  }
  Result = DAG.getNode(
      AMDGPUISD::BRANCH_COND,
      CmpValue.getDebugLoc(),
      MVT::Other, Chain,
      JumpT, CmpValue);
  return Result;
}

SDValue R600TargetLowering::LowerImplicitParameter(SelectionDAG &DAG, EVT VT,
                                                   DebugLoc DL,
                                                   unsigned DwordOffset) const {
  unsigned ByteOffset = DwordOffset * 4;
  PointerType * PtrType = PointerType::get(VT.getTypeForEVT(*DAG.getContext()),
                                      AMDGPUAS::PARAM_I_ADDRESS);

  // We shouldn't be using an offset wider than 16-bits for implicit parameters.
  assert(isInt<16>(ByteOffset));

  return DAG.getLoad(VT, DL, DAG.getEntryNode(),
                     DAG.getConstant(ByteOffset, MVT::i32), // PTR
                     MachinePointerInfo(ConstantPointerNull::get(PtrType)),
                     false, false, false, 0);
}

SDValue R600TargetLowering::LowerROTL(SDValue Op, SelectionDAG &DAG) const {
  DebugLoc DL = Op.getDebugLoc();
  EVT VT = Op.getValueType();

  return DAG.getNode(AMDGPUISD::BITALIGN, DL, VT,
                     Op.getOperand(0),
                     Op.getOperand(0),
                     DAG.getNode(ISD::SUB, DL, VT,
                                 DAG.getConstant(32, MVT::i32),
                                 Op.getOperand(1)));
}

bool R600TargetLowering::isZero(SDValue Op) const {
  if(ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(Op)) {
    return Cst->isNullValue();
  } else if(ConstantFPSDNode *CstFP = dyn_cast<ConstantFPSDNode>(Op)){
    return CstFP->isZero();
  } else {
    return false;
  }
}

SDValue R600TargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const {
  DebugLoc DL = Op.getDebugLoc();
  EVT VT = Op.getValueType();

  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue True = Op.getOperand(2);
  SDValue False = Op.getOperand(3);
  SDValue CC = Op.getOperand(4);
  SDValue Temp;

  // LHS and RHS are guaranteed to be the same value type
  EVT CompareVT = LHS.getValueType();

  // Check if we can lower this to a native operation.

  // Try to lower to a CND* instruction:
  // CND* instructions requires RHS to be zero.  Some SELECT_CC nodes that
  // can be lowered to CND* instructions can also be lowered to SET*
  // instructions.  CND* instructions are cheaper, because they dont't
  // require additional instructions to convert their result to the correct
  // value type, so this check should be first.
  if (isZero(LHS) || isZero(RHS)) {
    SDValue Cond = (isZero(LHS) ? RHS : LHS);
    SDValue Zero = (isZero(LHS) ? LHS : RHS);
    ISD::CondCode CCOpcode = cast<CondCodeSDNode>(CC)->get();
    if (CompareVT != VT) {
      // Bitcast True / False to the correct types.  This will end up being
      // a nop, but it allows us to define only a single pattern in the
      // .TD files for each CND* instruction rather than having to have
      // one pattern for integer True/False and one for fp True/False
      True = DAG.getNode(ISD::BITCAST, DL, CompareVT, True);
      False = DAG.getNode(ISD::BITCAST, DL, CompareVT, False);
    }
    if (isZero(LHS)) {
      CCOpcode = ISD::getSetCCSwappedOperands(CCOpcode);
    }

    switch (CCOpcode) {
    case ISD::SETONE:
    case ISD::SETUNE:
    case ISD::SETNE:
    case ISD::SETULE:
    case ISD::SETULT:
    case ISD::SETOLE:
    case ISD::SETOLT:
    case ISD::SETLE:
    case ISD::SETLT:
      CCOpcode = ISD::getSetCCInverse(CCOpcode, CompareVT == MVT::i32);
      Temp = True;
      True = False;
      False = Temp;
      break;
    default:
      break;
    }
    SDValue SelectNode = DAG.getNode(ISD::SELECT_CC, DL, CompareVT,
        Cond, Zero,
        True, False,
        DAG.getCondCode(CCOpcode));
    return DAG.getNode(ISD::BITCAST, DL, VT, SelectNode);
  }

  // Try to lower to a SET* instruction:
  // We need all the operands of SELECT_CC to have the same value type, so if
  // necessary we need to change True and False to be the same type as LHS and
  // RHS, and then convert the result of the select_cc back to the correct type.

  // Move hardware True/False values to the correct operand.
  if (isHWTrueValue(False) && isHWFalseValue(True)) {
    ISD::CondCode CCOpcode = cast<CondCodeSDNode>(CC)->get();
    std::swap(False, True);
    CC = DAG.getCondCode(ISD::getSetCCInverse(CCOpcode, CompareVT == MVT::i32));
  }

  if (isHWTrueValue(True) && isHWFalseValue(False)) {
    if (CompareVT !=  VT) {
      if (VT == MVT::f32 && CompareVT == MVT::i32) {
        SDValue Boolean = DAG.getNode(ISD::SELECT_CC, DL, CompareVT,
            LHS, RHS,
            DAG.getConstant(-1, MVT::i32),
            DAG.getConstant(0, MVT::i32),
            CC);
        // Convert integer values of true (-1) and false (0) to fp values of
        // true (1.0f) and false (0.0f).
        SDValue LSB = DAG.getNode(ISD::AND, DL, MVT::i32, Boolean,
                                                  DAG.getConstant(1, MVT::i32));
        return DAG.getNode(ISD::UINT_TO_FP, DL, VT, LSB);
      } else if (VT == MVT::i32 && CompareVT == MVT::f32) {
        SDValue BoolAsFlt = DAG.getNode(ISD::SELECT_CC, DL, CompareVT,
            LHS, RHS,
            DAG.getConstantFP(1.0f, MVT::f32),
            DAG.getConstantFP(0.0f, MVT::f32),
            CC);
        // Convert fp values of true (1.0f) and false (0.0f) to integer values
        // of true (-1) and false (0).
        SDValue Neg = DAG.getNode(ISD::FNEG, DL, MVT::f32, BoolAsFlt);
        return DAG.getNode(ISD::FP_TO_SINT, DL, VT, Neg);
      } else {
        // I don't think there will be any other type pairings.
        assert(!"Unhandled operand type parings in SELECT_CC");
      }
    } else {
      // This SELECT_CC is already legal.
      return DAG.getNode(ISD::SELECT_CC, DL, VT, LHS, RHS, True, False, CC);
    }
  }

  // Possible Min/Max pattern
  SDValue MinMax = LowerMinMax(Op, DAG);
  if (MinMax.getNode()) {
    return MinMax;
  }

  // If we make it this for it means we have no native instructions to handle
  // this SELECT_CC, so we must lower it.
  SDValue HWTrue, HWFalse;

  if (CompareVT == MVT::f32) {
    HWTrue = DAG.getConstantFP(1.0f, CompareVT);
    HWFalse = DAG.getConstantFP(0.0f, CompareVT);
  } else if (CompareVT == MVT::i32) {
    HWTrue = DAG.getConstant(-1, CompareVT);
    HWFalse = DAG.getConstant(0, CompareVT);
  }
  else {
    assert(!"Unhandled value type in LowerSELECT_CC");
  }

  // Lower this unsupported SELECT_CC into a combination of two supported
  // SELECT_CC operations.
  SDValue Cond = DAG.getNode(ISD::SELECT_CC, DL, CompareVT, LHS, RHS, HWTrue, HWFalse, CC);

  return DAG.getNode(ISD::SELECT_CC, DL, VT,
      Cond, HWFalse,
      True, False,
      DAG.getCondCode(ISD::SETNE));
}

SDValue R600TargetLowering::LowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  return DAG.getNode(ISD::SELECT_CC,
      Op.getDebugLoc(),
      Op.getValueType(),
      Op.getOperand(0),
      DAG.getConstant(0, MVT::i32),
      Op.getOperand(1),
      Op.getOperand(2),
      DAG.getCondCode(ISD::SETNE));
}

SDValue R600TargetLowering::LowerSETCC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Cond;
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue CC  = Op.getOperand(2);
  DebugLoc DL = Op.getDebugLoc();
  assert(Op.getValueType() == MVT::i32);
  if (LHS.getValueType() == MVT::i32) {
    Cond = DAG.getNode(
        ISD::SELECT_CC,
        Op.getDebugLoc(),
        MVT::i32,
        LHS, RHS,
        DAG.getConstant(-1, MVT::i32),
        DAG.getConstant(0, MVT::i32),
        CC);
  } else if (LHS.getValueType() == MVT::f32) {
    Cond = DAG.getNode(
        ISD::SELECT_CC,
        Op.getDebugLoc(),
        MVT::f32,
        LHS, RHS,
        DAG.getConstantFP(1.0f, MVT::f32),
        DAG.getConstantFP(0.0f, MVT::f32),
        CC);
    Cond = DAG.getNode(
        ISD::FP_TO_SINT,
        DL,
        MVT::i32,
        Cond);
  } else {
    assert(0 && "Not valid type for set_cc");
  }
  Cond = DAG.getNode(
      ISD::AND,
      DL,
      MVT::i32,
      DAG.getConstant(1, MVT::i32),
      Cond);
  return Cond;
}

SDValue R600TargetLowering::LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  DebugLoc DL = Op.getDebugLoc();
  StoreSDNode *StoreNode = cast<StoreSDNode>(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue Value = Op.getOperand(1);
  SDValue Ptr = Op.getOperand(2);

  if (StoreNode->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS &&
      Ptr->getOpcode() != AMDGPUISD::DWORDADDR) {
    // Convert pointer from byte address to dword address.
    Ptr = DAG.getNode(AMDGPUISD::DWORDADDR, DL, Ptr.getValueType(),
                      DAG.getNode(ISD::SRL, DL, Ptr.getValueType(),
                                  Ptr, DAG.getConstant(2, MVT::i32)));

    if (StoreNode->isTruncatingStore() || StoreNode->isIndexed()) {
      assert(!"Truncated and indexed stores not supported yet");
    } else {
      Chain = DAG.getStore(Chain, DL, Value, Ptr, StoreNode->getMemOperand());
    }
    return Chain;
  }
  return SDValue();
}


SDValue R600TargetLowering::LowerFPOW(SDValue Op,
    SelectionDAG &DAG) const {
  DebugLoc DL = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  SDValue LogBase = DAG.getNode(ISD::FLOG2, DL, VT, Op.getOperand(0));
  SDValue MulLogBase = DAG.getNode(ISD::FMUL, DL, VT, Op.getOperand(1), LogBase);
  return DAG.getNode(ISD::FEXP2, DL, VT, MulLogBase);
}

/// XXX Only kernel functions are supported, so we can assume for now that
/// every function is a kernel function, but in the future we should use
/// separate calling conventions for kernel and non-kernel functions.
SDValue R600TargetLowering::LowerFormalArguments(
                                      SDValue Chain,
                                      CallingConv::ID CallConv,
                                      bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                      DebugLoc DL, SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &InVals) const {
  unsigned ParamOffsetBytes = 36;
  Function::const_arg_iterator FuncArg =
                            DAG.getMachineFunction().getFunction()->arg_begin();
  for (unsigned i = 0, e = Ins.size(); i < e; ++i, ++FuncArg) {
    EVT VT = Ins[i].VT;
    Type *ArgType = FuncArg->getType();
    unsigned ArgSizeInBits = ArgType->isPointerTy() ?
                             32 : ArgType->getPrimitiveSizeInBits();
    unsigned ArgBytes = ArgSizeInBits >> 3;
    EVT ArgVT;
    if (ArgSizeInBits < VT.getSizeInBits()) {
      assert(!ArgType->isFloatTy() &&
             "Extending floating point arguments not supported yet");
      ArgVT = MVT::getIntegerVT(ArgSizeInBits);
    } else {
      ArgVT = VT;
    }
    PointerType *PtrTy = PointerType::get(VT.getTypeForEVT(*DAG.getContext()),
                                                    AMDGPUAS::PARAM_I_ADDRESS);
    SDValue Arg = DAG.getExtLoad(ISD::ZEXTLOAD, DL, VT, DAG.getRoot(),
                                DAG.getConstant(ParamOffsetBytes, MVT::i32),
                                       MachinePointerInfo(new Argument(PtrTy)),
                                       ArgVT, false, false, ArgBytes);
    InVals.push_back(Arg);
    ParamOffsetBytes += ArgBytes;
  }
  return Chain;
}

EVT R600TargetLowering::getSetCCResultType(EVT VT) const {
   if (!VT.isVector()) return MVT::i32;
   return VT.changeVectorElementTypeToInteger();
}

//===----------------------------------------------------------------------===//
// Custom DAG Optimizations
//===----------------------------------------------------------------------===//

SDValue R600TargetLowering::PerformDAGCombine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

  switch (N->getOpcode()) {
  // (f32 fp_round (f64 uint_to_fp a)) -> (f32 uint_to_fp a)
  case ISD::FP_ROUND: {
      SDValue Arg = N->getOperand(0);
      if (Arg.getOpcode() == ISD::UINT_TO_FP && Arg.getValueType() == MVT::f64) {
        return DAG.getNode(ISD::UINT_TO_FP, N->getDebugLoc(), N->getValueType(0),
                           Arg.getOperand(0));
      }
      break;
    }
  }
  return SDValue();
}
