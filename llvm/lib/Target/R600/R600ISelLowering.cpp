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
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"

using namespace llvm;

R600TargetLowering::R600TargetLowering(TargetMachine &TM) :
    AMDGPUTargetLowering(TM),
    Gen(TM.getSubtarget<AMDGPUSubtarget>().getGeneration()) {
  addRegisterClass(MVT::v4f32, &AMDGPU::R600_Reg128RegClass);
  addRegisterClass(MVT::f32, &AMDGPU::R600_Reg32RegClass);
  addRegisterClass(MVT::v4i32, &AMDGPU::R600_Reg128RegClass);
  addRegisterClass(MVT::i32, &AMDGPU::R600_Reg32RegClass);
  computeRegisterProperties();

  setOperationAction(ISD::FADD, MVT::v4f32, Expand);
  setOperationAction(ISD::FMUL, MVT::v4f32, Expand);
  setOperationAction(ISD::FDIV, MVT::v4f32, Expand);
  setOperationAction(ISD::FSUB, MVT::v4f32, Expand);

  setOperationAction(ISD::FCOS, MVT::f32, Custom);
  setOperationAction(ISD::FSIN, MVT::f32, Custom);

  setOperationAction(ISD::FP_TO_SINT, MVT::v4i32, Expand);
  setOperationAction(ISD::FP_TO_UINT, MVT::v4i32, Expand);
  setOperationAction(ISD::SINT_TO_FP, MVT::v4i32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::v4i32, Expand);
  setOperationAction(ISD::SETCC, MVT::v4i32, Expand);

  setOperationAction(ISD::BR_CC, MVT::i32, Expand);
  setOperationAction(ISD::BR_CC, MVT::f32, Expand);

  setOperationAction(ISD::FSUB, MVT::f32, Expand);

  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i1, Custom);

  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);

  setOperationAction(ISD::SETCC, MVT::i32, Expand);
  setOperationAction(ISD::SETCC, MVT::f32, Expand);
  setOperationAction(ISD::FP_TO_UINT, MVT::i1, Custom);

  setOperationAction(ISD::SELECT, MVT::i32, Custom);
  setOperationAction(ISD::SELECT, MVT::f32, Custom);

  setOperationAction(ISD::VSELECT, MVT::v4i32, Expand);
  setOperationAction(ISD::VSELECT, MVT::v2i32, Expand);

  // Legalize loads and stores to the private address space.
  setOperationAction(ISD::LOAD, MVT::i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v2i32, Expand);
  setOperationAction(ISD::LOAD, MVT::v4i32, Custom);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4i8, Custom);
  setLoadExtAction(ISD::EXTLOAD, MVT::i8, Custom);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i8, Custom);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::v4i8, Custom);
  setOperationAction(ISD::STORE, MVT::i8, Custom);
  setOperationAction(ISD::STORE, MVT::i32, Custom);
  setOperationAction(ISD::STORE, MVT::v2i32, Expand);
  setOperationAction(ISD::STORE, MVT::v4i32, Custom);

  setOperationAction(ISD::LOAD, MVT::i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v4i32, Custom);
  setOperationAction(ISD::FrameIndex, MVT::i32, Custom);

  setTargetDAGCombine(ISD::FP_ROUND);
  setTargetDAGCombine(ISD::FP_TO_SINT);
  setTargetDAGCombine(ISD::EXTRACT_VECTOR_ELT);
  setTargetDAGCombine(ISD::SELECT_CC);

  setBooleanContents(ZeroOrNegativeOneBooleanContent);
  setBooleanVectorContents(ZeroOrNegativeOneBooleanContent);
  setSchedulingPreference(Sched::VLIW);
}

MachineBasicBlock * R600TargetLowering::EmitInstrWithCustomInserter(
    MachineInstr * MI, MachineBasicBlock * BB) const {
  MachineFunction * MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  MachineBasicBlock::iterator I = *MI;
  const R600InstrInfo *TII =
    static_cast<const R600InstrInfo*>(MF->getTarget().getInstrInfo());

  switch (MI->getOpcode()) {
  default: return AMDGPUTargetLowering::EmitInstrWithCustomInserter(MI, BB);
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

  case AMDGPU::MASK_WRITE: {
    unsigned maskedRegister = MI->getOperand(0).getReg();
    assert(TargetRegisterInfo::isVirtualRegister(maskedRegister));
    MachineInstr * defInstr = MRI.getVRegDef(maskedRegister);
    TII->addFlag(defInstr, 0, MO_FLAG_MASK);
    break;
  }

  case AMDGPU::LDS_READ_RET: {
    MachineInstrBuilder NewMI = BuildMI(*BB, I, BB->findDebugLoc(I),
                                        TII->get(MI->getOpcode()),
                                        AMDGPU::OQAP);
    for (unsigned i = 1, e = MI->getNumOperands(); i < e; ++i) {
      NewMI.addOperand(MI->getOperand(i));
    }
    TII->buildDefaultInstruction(*BB, I, AMDGPU::MOV,
                                 MI->getOperand(0).getReg(),
                                 AMDGPU::OQAP);
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
  case AMDGPU::CONST_COPY: {
    MachineInstr *NewMI = TII->buildDefaultInstruction(*BB, MI, AMDGPU::MOV,
        MI->getOperand(0).getReg(), AMDGPU::ALU_CONST);
    TII->setImmOperand(NewMI, AMDGPU::OpName::src0_sel,
        MI->getOperand(1).getImm());
    break;
  }

  case AMDGPU::RAT_WRITE_CACHELESS_32_eg:
  case AMDGPU::RAT_WRITE_CACHELESS_128_eg: {
    unsigned EOP = (llvm::next(I)->getOpcode() == AMDGPU::RETURN) ? 1 : 0;

    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(MI->getOpcode()))
            .addOperand(MI->getOperand(0))
            .addOperand(MI->getOperand(1))
            .addImm(EOP); // Set End of program bit
    break;
  }

  case AMDGPU::TXD: {
    unsigned T0 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);
    unsigned T1 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);
    MachineOperand &RID = MI->getOperand(4);
    MachineOperand &SID = MI->getOperand(5);
    unsigned TextureId = MI->getOperand(6).getImm();
    unsigned SrcX = 0, SrcY = 1, SrcZ = 2, SrcW = 3;
    unsigned CTX = 1, CTY = 1, CTZ = 1, CTW = 1;

    switch (TextureId) {
    case 5: // Rect
      CTX = CTY = 0;
      break;
    case 6: // Shadow1D
      SrcW = SrcZ;
      break;
    case 7: // Shadow2D
      SrcW = SrcZ;
      break;
    case 8: // ShadowRect
      CTX = CTY = 0;
      SrcW = SrcZ;
      break;
    case 9: // 1DArray
      SrcZ = SrcY;
      CTZ = 0;
      break;
    case 10: // 2DArray
      CTZ = 0;
      break;
    case 11: // Shadow1DArray
      SrcZ = SrcY;
      CTZ = 0;
      break;
    case 12: // Shadow2DArray
      CTZ = 0;
      break;
    }
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_H), T0)
            .addOperand(MI->getOperand(3))
            .addImm(SrcX)
            .addImm(SrcY)
            .addImm(SrcZ)
            .addImm(SrcW)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(1)
            .addImm(2)
            .addImm(3)
            .addOperand(RID)
            .addOperand(SID)
            .addImm(CTX)
            .addImm(CTY)
            .addImm(CTZ)
            .addImm(CTW);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_V), T1)
            .addOperand(MI->getOperand(2))
            .addImm(SrcX)
            .addImm(SrcY)
            .addImm(SrcZ)
            .addImm(SrcW)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(1)
            .addImm(2)
            .addImm(3)
            .addOperand(RID)
            .addOperand(SID)
            .addImm(CTX)
            .addImm(CTY)
            .addImm(CTZ)
            .addImm(CTW);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SAMPLE_G))
            .addOperand(MI->getOperand(0))
            .addOperand(MI->getOperand(1))
            .addImm(SrcX)
            .addImm(SrcY)
            .addImm(SrcZ)
            .addImm(SrcW)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(1)
            .addImm(2)
            .addImm(3)
            .addOperand(RID)
            .addOperand(SID)
            .addImm(CTX)
            .addImm(CTY)
            .addImm(CTZ)
            .addImm(CTW)
            .addReg(T0, RegState::Implicit)
            .addReg(T1, RegState::Implicit);
    break;
  }

  case AMDGPU::TXD_SHADOW: {
    unsigned T0 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);
    unsigned T1 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);
    MachineOperand &RID = MI->getOperand(4);
    MachineOperand &SID = MI->getOperand(5);
    unsigned TextureId = MI->getOperand(6).getImm();
    unsigned SrcX = 0, SrcY = 1, SrcZ = 2, SrcW = 3;
    unsigned CTX = 1, CTY = 1, CTZ = 1, CTW = 1;

    switch (TextureId) {
    case 5: // Rect
      CTX = CTY = 0;
      break;
    case 6: // Shadow1D
      SrcW = SrcZ;
      break;
    case 7: // Shadow2D
      SrcW = SrcZ;
      break;
    case 8: // ShadowRect
      CTX = CTY = 0;
      SrcW = SrcZ;
      break;
    case 9: // 1DArray
      SrcZ = SrcY;
      CTZ = 0;
      break;
    case 10: // 2DArray
      CTZ = 0;
      break;
    case 11: // Shadow1DArray
      SrcZ = SrcY;
      CTZ = 0;
      break;
    case 12: // Shadow2DArray
      CTZ = 0;
      break;
    }

    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_H), T0)
            .addOperand(MI->getOperand(3))
            .addImm(SrcX)
            .addImm(SrcY)
            .addImm(SrcZ)
            .addImm(SrcW)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(1)
            .addImm(2)
            .addImm(3)
            .addOperand(RID)
            .addOperand(SID)
            .addImm(CTX)
            .addImm(CTY)
            .addImm(CTZ)
            .addImm(CTW);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_V), T1)
            .addOperand(MI->getOperand(2))
            .addImm(SrcX)
            .addImm(SrcY)
            .addImm(SrcZ)
            .addImm(SrcW)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(1)
            .addImm(2)
            .addImm(3)
            .addOperand(RID)
            .addOperand(SID)
            .addImm(CTX)
            .addImm(CTY)
            .addImm(CTZ)
            .addImm(CTW);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SAMPLE_C_G))
            .addOperand(MI->getOperand(0))
            .addOperand(MI->getOperand(1))
            .addImm(SrcX)
            .addImm(SrcY)
            .addImm(SrcZ)
            .addImm(SrcW)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(1)
            .addImm(2)
            .addImm(3)
            .addOperand(RID)
            .addOperand(SID)
            .addImm(CTX)
            .addImm(CTY)
            .addImm(CTZ)
            .addImm(CTW)
            .addReg(T0, RegState::Implicit)
            .addReg(T1, RegState::Implicit);
    break;
  }

  case AMDGPU::BRANCH:
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::JUMP))
              .addOperand(MI->getOperand(0));
      break;

  case AMDGPU::BRANCH_COND_f32: {
    MachineInstr *NewMI =
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::PRED_X),
              AMDGPU::PREDICATE_BIT)
              .addOperand(MI->getOperand(1))
              .addImm(OPCODE_IS_NOT_ZERO)
              .addImm(0); // Flags
    TII->addFlag(NewMI, 0, MO_FLAG_PUSH);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::JUMP_COND))
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
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::JUMP_COND))
           .addOperand(MI->getOperand(0))
            .addReg(AMDGPU::PREDICATE_BIT, RegState::Kill);
    break;
  }

  case AMDGPU::EG_ExportSwz:
  case AMDGPU::R600_ExportSwz: {
    // Instruction is left unmodified if its not the last one of its type
    bool isLastInstructionOfItsType = true;
    unsigned InstExportType = MI->getOperand(1).getImm();
    for (MachineBasicBlock::iterator NextExportInst = llvm::next(I),
         EndBlock = BB->end(); NextExportInst != EndBlock;
         NextExportInst = llvm::next(NextExportInst)) {
      if (NextExportInst->getOpcode() == AMDGPU::EG_ExportSwz ||
          NextExportInst->getOpcode() == AMDGPU::R600_ExportSwz) {
        unsigned CurrentInstExportType = NextExportInst->getOperand(1)
            .getImm();
        if (CurrentInstExportType == InstExportType) {
          isLastInstructionOfItsType = false;
          break;
        }
      }
    }
    bool EOP = (llvm::next(I)->getOpcode() == AMDGPU::RETURN)? 1 : 0;
    if (!EOP && !isLastInstructionOfItsType)
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
            .addImm(EOP);
    break;
  }
  case AMDGPU::RETURN: {
    // RETURN instructions must have the live-out registers as implicit uses,
    // otherwise they appear dead.
    R600MachineFunctionInfo *MFI = MF->getInfo<R600MachineFunctionInfo>();
    MachineInstrBuilder MIB(*MF, MI);
    for (unsigned i = 0, e = MFI->LiveOuts.size(); i != e; ++i)
      MIB.addReg(MFI->LiveOuts[i], RegState::Implicit);
    return BB;
  }
  }

  MI->eraseFromParent();
  return BB;
}

//===----------------------------------------------------------------------===//
// Custom DAG Lowering Operations
//===----------------------------------------------------------------------===//

SDValue R600TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();
  switch (Op.getOpcode()) {
  default: return AMDGPUTargetLowering::LowerOperation(Op, DAG);
  case ISD::FCOS:
  case ISD::FSIN: return LowerTrig(Op, DAG);
  case ISD::SELECT_CC: return LowerSELECT_CC(Op, DAG);
  case ISD::SELECT: return LowerSELECT(Op, DAG);
  case ISD::STORE: return LowerSTORE(Op, DAG);
  case ISD::LOAD: return LowerLOAD(Op, DAG);
  case ISD::FrameIndex: return LowerFrameIndex(Op, DAG);
  case ISD::GlobalAddress: return LowerGlobalAddress(MFI, Op, DAG);
  case ISD::INTRINSIC_VOID: {
    SDValue Chain = Op.getOperand(0);
    unsigned IntrinsicID =
                         cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
    switch (IntrinsicID) {
    case AMDGPUIntrinsic::AMDGPU_store_output: {
      int64_t RegIndex = cast<ConstantSDNode>(Op.getOperand(3))->getZExtValue();
      unsigned Reg = AMDGPU::R600_TReg32RegClass.getRegister(RegIndex);
      MFI->LiveOuts.push_back(Reg);
      return DAG.getCopyToReg(Chain, SDLoc(Op), Reg, Op.getOperand(2));
    }
    case AMDGPUIntrinsic::R600_store_swizzle: {
      const SDValue Args[8] = {
        Chain,
        Op.getOperand(2), // Export Value
        Op.getOperand(3), // ArrayBase
        Op.getOperand(4), // Type
        DAG.getConstant(0, MVT::i32), // SWZ_X
        DAG.getConstant(1, MVT::i32), // SWZ_Y
        DAG.getConstant(2, MVT::i32), // SWZ_Z
        DAG.getConstant(3, MVT::i32) // SWZ_W
      };
      return DAG.getNode(AMDGPUISD::EXPORT, SDLoc(Op), Op.getValueType(),
          Args, 8);
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
    SDLoc DL(Op);
    switch(IntrinsicID) {
    default: return AMDGPUTargetLowering::LowerOperation(Op, DAG);
    case AMDGPUIntrinsic::R600_load_input: {
      int64_t RegIndex = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
      unsigned Reg = AMDGPU::R600_TReg32RegClass.getRegister(RegIndex);
      MachineFunction &MF = DAG.getMachineFunction();
      MachineRegisterInfo &MRI = MF.getRegInfo();
      MRI.addLiveIn(Reg);
      return DAG.getCopyFromReg(DAG.getEntryNode(),
          SDLoc(DAG.getEntryNode()), Reg, VT);
    }

    case AMDGPUIntrinsic::R600_interp_input: {
      int slot = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
      int ijb = cast<ConstantSDNode>(Op.getOperand(2))->getSExtValue();
      MachineSDNode *interp;
      if (ijb < 0) {
        const MachineFunction &MF = DAG.getMachineFunction();
        const R600InstrInfo *TII =
          static_cast<const R600InstrInfo*>(MF.getTarget().getInstrInfo());
        interp = DAG.getMachineNode(AMDGPU::INTERP_VEC_LOAD, DL,
            MVT::v4f32, DAG.getTargetConstant(slot / 4 , MVT::i32));
        return DAG.getTargetExtractSubreg(
            TII->getRegisterInfo().getSubRegFromChannel(slot % 4),
            DL, MVT::f32, SDValue(interp, 0));
      }

      MachineFunction &MF = DAG.getMachineFunction();
      MachineRegisterInfo &MRI = MF.getRegInfo();
      unsigned RegisterI = AMDGPU::R600_TReg32RegClass.getRegister(2 * ijb);
      unsigned RegisterJ = AMDGPU::R600_TReg32RegClass.getRegister(2 * ijb + 1);
      MRI.addLiveIn(RegisterI);
      MRI.addLiveIn(RegisterJ);
      SDValue RegisterINode = DAG.getCopyFromReg(DAG.getEntryNode(),
          SDLoc(DAG.getEntryNode()), RegisterI, MVT::f32);
      SDValue RegisterJNode = DAG.getCopyFromReg(DAG.getEntryNode(),
          SDLoc(DAG.getEntryNode()), RegisterJ, MVT::f32);

      if (slot % 4 < 2)
        interp = DAG.getMachineNode(AMDGPU::INTERP_PAIR_XY, DL,
            MVT::f32, MVT::f32, DAG.getTargetConstant(slot / 4 , MVT::i32),
            RegisterJNode, RegisterINode);
      else
        interp = DAG.getMachineNode(AMDGPU::INTERP_PAIR_ZW, DL,
            MVT::f32, MVT::f32, DAG.getTargetConstant(slot / 4 , MVT::i32),
            RegisterJNode, RegisterINode);
      return SDValue(interp, slot % 2);
    }
    case AMDGPUIntrinsic::R600_tex:
    case AMDGPUIntrinsic::R600_texc:
    case AMDGPUIntrinsic::R600_txl:
    case AMDGPUIntrinsic::R600_txlc:
    case AMDGPUIntrinsic::R600_txb:
    case AMDGPUIntrinsic::R600_txbc:
    case AMDGPUIntrinsic::R600_txf:
    case AMDGPUIntrinsic::R600_txq:
    case AMDGPUIntrinsic::R600_ddx:
    case AMDGPUIntrinsic::R600_ddy: {
      unsigned TextureOp;
      switch (IntrinsicID) {
      case AMDGPUIntrinsic::R600_tex:
        TextureOp = 0;
        break;
      case AMDGPUIntrinsic::R600_texc:
        TextureOp = 1;
        break;
      case AMDGPUIntrinsic::R600_txl:
        TextureOp = 2;
        break;
      case AMDGPUIntrinsic::R600_txlc:
        TextureOp = 3;
        break;
      case AMDGPUIntrinsic::R600_txb:
        TextureOp = 4;
        break;
      case AMDGPUIntrinsic::R600_txbc:
        TextureOp = 5;
        break;
      case AMDGPUIntrinsic::R600_txf:
        TextureOp = 6;
        break;
      case AMDGPUIntrinsic::R600_txq:
        TextureOp = 7;
        break;
      case AMDGPUIntrinsic::R600_ddx:
        TextureOp = 8;
        break;
      case AMDGPUIntrinsic::R600_ddy:
        TextureOp = 9;
        break;
      default:
        llvm_unreachable("Unknow Texture Operation");
      }

      SDValue TexArgs[19] = {
        DAG.getConstant(TextureOp, MVT::i32),
        Op.getOperand(1),
        DAG.getConstant(0, MVT::i32),
        DAG.getConstant(1, MVT::i32),
        DAG.getConstant(2, MVT::i32),
        DAG.getConstant(3, MVT::i32),
        Op.getOperand(2),
        Op.getOperand(3),
        Op.getOperand(4),
        DAG.getConstant(0, MVT::i32),
        DAG.getConstant(1, MVT::i32),
        DAG.getConstant(2, MVT::i32),
        DAG.getConstant(3, MVT::i32),
        Op.getOperand(5),
        Op.getOperand(6),
        Op.getOperand(7),
        Op.getOperand(8),
        Op.getOperand(9),
        Op.getOperand(10)
      };
      return DAG.getNode(AMDGPUISD::TEXTURE_FETCH, DL, MVT::v4f32, TexArgs, 19);
    }
    case AMDGPUIntrinsic::AMDGPU_dp4: {
      SDValue Args[8] = {
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(1),
          DAG.getConstant(0, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(2),
          DAG.getConstant(0, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(1),
          DAG.getConstant(1, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(2),
          DAG.getConstant(1, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(1),
          DAG.getConstant(2, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(2),
          DAG.getConstant(2, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(1),
          DAG.getConstant(3, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(2),
          DAG.getConstant(3, MVT::i32))
      };
      return DAG.getNode(AMDGPUISD::DOT4, DL, MVT::f32, Args, 8);
    }

    case Intrinsic::r600_read_ngroups_x:
      return LowerImplicitParameter(DAG, VT, DL, 0);
    case Intrinsic::r600_read_ngroups_y:
      return LowerImplicitParameter(DAG, VT, DL, 1);
    case Intrinsic::r600_read_ngroups_z:
      return LowerImplicitParameter(DAG, VT, DL, 2);
    case Intrinsic::r600_read_global_size_x:
      return LowerImplicitParameter(DAG, VT, DL, 3);
    case Intrinsic::r600_read_global_size_y:
      return LowerImplicitParameter(DAG, VT, DL, 4);
    case Intrinsic::r600_read_global_size_z:
      return LowerImplicitParameter(DAG, VT, DL, 5);
    case Intrinsic::r600_read_local_size_x:
      return LowerImplicitParameter(DAG, VT, DL, 6);
    case Intrinsic::r600_read_local_size_y:
      return LowerImplicitParameter(DAG, VT, DL, 7);
    case Intrinsic::r600_read_local_size_z:
      return LowerImplicitParameter(DAG, VT, DL, 8);

    case Intrinsic::r600_read_tgid_x:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T1_X, VT);
    case Intrinsic::r600_read_tgid_y:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T1_Y, VT);
    case Intrinsic::r600_read_tgid_z:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T1_Z, VT);
    case Intrinsic::r600_read_tidig_x:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T0_X, VT);
    case Intrinsic::r600_read_tidig_y:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T0_Y, VT);
    case Intrinsic::r600_read_tidig_z:
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
    return;
  case ISD::LOAD: {
    SDNode *Node = LowerLOAD(SDValue(N, 0), DAG).getNode();
    Results.push_back(SDValue(Node, 0));
    Results.push_back(SDValue(Node, 1));
    // XXX: LLVM seems not to replace Chain Value inside CustomWidenLowerNode
    // function
    DAG.ReplaceAllUsesOfValueWith(SDValue(N,1), SDValue(Node, 1));
    return;
  }
  case ISD::STORE:
    SDNode *Node = LowerSTORE(SDValue(N, 0), DAG).getNode();
    Results.push_back(SDValue(Node, 0));
    return;
  }
}

SDValue R600TargetLowering::LowerTrig(SDValue Op, SelectionDAG &DAG) const {
  // On hw >= R700, COS/SIN input must be between -1. and 1.
  // Thus we lower them to TRIG ( FRACT ( x / 2Pi + 0.5) - 0.5)
  EVT VT = Op.getValueType();
  SDValue Arg = Op.getOperand(0);
  SDValue FractPart = DAG.getNode(AMDGPUISD::FRACT, SDLoc(Op), VT,
      DAG.getNode(ISD::FADD, SDLoc(Op), VT,
        DAG.getNode(ISD::FMUL, SDLoc(Op), VT, Arg,
          DAG.getConstantFP(0.15915494309, MVT::f32)),
        DAG.getConstantFP(0.5, MVT::f32)));
  unsigned TrigNode;
  switch (Op.getOpcode()) {
  case ISD::FCOS:
    TrigNode = AMDGPUISD::COS_HW;
    break;
  case ISD::FSIN:
    TrigNode = AMDGPUISD::SIN_HW;
    break;
  default:
    llvm_unreachable("Wrong trig opcode");
  }
  SDValue TrigVal = DAG.getNode(TrigNode, SDLoc(Op), VT,
      DAG.getNode(ISD::FADD, SDLoc(Op), VT, FractPart,
        DAG.getConstantFP(-0.5, MVT::f32)));
  if (Gen >= AMDGPUSubtarget::R700)
    return TrigVal;
  // On R600 hw, COS/SIN input must be between -Pi and Pi.
  return DAG.getNode(ISD::FMUL, SDLoc(Op), VT, TrigVal,
      DAG.getConstantFP(3.14159265359, MVT::f32));
}

SDValue R600TargetLowering::LowerFPTOUINT(SDValue Op, SelectionDAG &DAG) const {
  return DAG.getNode(
      ISD::SETCC,
      SDLoc(Op),
      MVT::i1,
      Op, DAG.getConstantFP(0.0f, MVT::f32),
      DAG.getCondCode(ISD::SETNE)
      );
}

SDValue R600TargetLowering::LowerImplicitParameter(SelectionDAG &DAG, EVT VT,
                                                   SDLoc DL,
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

SDValue R600TargetLowering::LowerFrameIndex(SDValue Op, SelectionDAG &DAG) const {

  MachineFunction &MF = DAG.getMachineFunction();
  const AMDGPUFrameLowering *TFL =
   static_cast<const AMDGPUFrameLowering*>(getTargetMachine().getFrameLowering());

  FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Op);
  assert(FIN);

  unsigned FrameIndex = FIN->getIndex();
  unsigned Offset = TFL->getFrameIndexOffset(MF, FrameIndex);
  return DAG.getConstant(Offset * 4 * TFL->getStackWidth(MF), MVT::i32);
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
  SDLoc DL(Op);
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

  // Try to lower to a SET* instruction:
  //
  // SET* can match the following patterns:
  //
  // select_cc f32, f32, -1,  0, cc_any
  // select_cc f32, f32, 1.0f, 0.0f, cc_any
  // select_cc i32, i32, -1,  0, cc_any
  //

  // Move hardware True/False values to the correct operand.
  if (isHWTrueValue(False) && isHWFalseValue(True)) {
    ISD::CondCode CCOpcode = cast<CondCodeSDNode>(CC)->get();
    std::swap(False, True);
    CC = DAG.getCondCode(ISD::getSetCCInverse(CCOpcode, CompareVT == MVT::i32));
  }

  if (isHWTrueValue(True) && isHWFalseValue(False) &&
      (CompareVT == VT || VT == MVT::i32)) {
    // This can be matched by a SET* instruction.
    return DAG.getNode(ISD::SELECT_CC, DL, VT, LHS, RHS, True, False, CC);
  }

  // Try to lower to a CND* instruction:
  //
  // CND* can match the following patterns:
  //
  // select_cc f32, 0.0, f32, f32, cc_any
  // select_cc f32, 0.0, i32, i32, cc_any
  // select_cc i32, 0,   f32, f32, cc_any
  // select_cc i32, 0,   i32, i32, cc_any
  //
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
      SDLoc(Op),
      Op.getValueType(),
      Op.getOperand(0),
      DAG.getConstant(0, MVT::i32),
      Op.getOperand(1),
      Op.getOperand(2),
      DAG.getCondCode(ISD::SETNE));
}

/// LLVM generates byte-addresed pointers.  For indirect addressing, we need to
/// convert these pointers to a register index.  Each register holds
/// 16 bytes, (4 x 32bit sub-register), but we need to take into account the
/// \p StackWidth, which tells us how many of the 4 sub-registrers will be used
/// for indirect addressing.
SDValue R600TargetLowering::stackPtrToRegIndex(SDValue Ptr,
                                               unsigned StackWidth,
                                               SelectionDAG &DAG) const {
  unsigned SRLPad;
  switch(StackWidth) {
  case 1:
    SRLPad = 2;
    break;
  case 2:
    SRLPad = 3;
    break;
  case 4:
    SRLPad = 4;
    break;
  default: llvm_unreachable("Invalid stack width");
  }

  return DAG.getNode(ISD::SRL, SDLoc(Ptr), Ptr.getValueType(), Ptr,
                     DAG.getConstant(SRLPad, MVT::i32));
}

void R600TargetLowering::getStackAddress(unsigned StackWidth,
                                         unsigned ElemIdx,
                                         unsigned &Channel,
                                         unsigned &PtrIncr) const {
  switch (StackWidth) {
  default:
  case 1:
    Channel = 0;
    if (ElemIdx > 0) {
      PtrIncr = 1;
    } else {
      PtrIncr = 0;
    }
    break;
  case 2:
    Channel = ElemIdx % 2;
    if (ElemIdx == 2) {
      PtrIncr = 1;
    } else {
      PtrIncr = 0;
    }
    break;
  case 4:
    Channel = ElemIdx;
    PtrIncr = 0;
    break;
  }
}

SDValue R600TargetLowering::LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
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

  EVT ValueVT = Value.getValueType();

  if (StoreNode->getAddressSpace() != AMDGPUAS::PRIVATE_ADDRESS) {
    return SDValue();
  }

  // Lowering for indirect addressing

  const MachineFunction &MF = DAG.getMachineFunction();
  const AMDGPUFrameLowering *TFL = static_cast<const AMDGPUFrameLowering*>(
                                         getTargetMachine().getFrameLowering());
  unsigned StackWidth = TFL->getStackWidth(MF);

  Ptr = stackPtrToRegIndex(Ptr, StackWidth, DAG);

  if (ValueVT.isVector()) {
    unsigned NumElemVT = ValueVT.getVectorNumElements();
    EVT ElemVT = ValueVT.getVectorElementType();
    SDValue Stores[4];

    assert(NumElemVT >= StackWidth && "Stack width cannot be greater than "
                                      "vector width in load");

    for (unsigned i = 0; i < NumElemVT; ++i) {
      unsigned Channel, PtrIncr;
      getStackAddress(StackWidth, i, Channel, PtrIncr);
      Ptr = DAG.getNode(ISD::ADD, DL, MVT::i32, Ptr,
                        DAG.getConstant(PtrIncr, MVT::i32));
      SDValue Elem = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, ElemVT,
                                 Value, DAG.getConstant(i, MVT::i32));

      Stores[i] = DAG.getNode(AMDGPUISD::REGISTER_STORE, DL, MVT::Other,
                              Chain, Elem, Ptr,
                              DAG.getTargetConstant(Channel, MVT::i32));
    }
     Chain =  DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Stores, NumElemVT);
   } else {
    if (ValueVT == MVT::i8) {
      Value = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i32, Value);
    }
    Chain = DAG.getNode(AMDGPUISD::REGISTER_STORE, DL, MVT::Other, Chain, Value, Ptr,
    DAG.getTargetConstant(0, MVT::i32)); // Channel
  }

  return Chain;
}

// return (512 + (kc_bank << 12)
static int
ConstantAddressBlock(unsigned AddressSpace) {
  switch (AddressSpace) {
  case AMDGPUAS::CONSTANT_BUFFER_0:
    return 512;
  case AMDGPUAS::CONSTANT_BUFFER_1:
    return 512 + 4096;
  case AMDGPUAS::CONSTANT_BUFFER_2:
    return 512 + 4096 * 2;
  case AMDGPUAS::CONSTANT_BUFFER_3:
    return 512 + 4096 * 3;
  case AMDGPUAS::CONSTANT_BUFFER_4:
    return 512 + 4096 * 4;
  case AMDGPUAS::CONSTANT_BUFFER_5:
    return 512 + 4096 * 5;
  case AMDGPUAS::CONSTANT_BUFFER_6:
    return 512 + 4096 * 6;
  case AMDGPUAS::CONSTANT_BUFFER_7:
    return 512 + 4096 * 7;
  case AMDGPUAS::CONSTANT_BUFFER_8:
    return 512 + 4096 * 8;
  case AMDGPUAS::CONSTANT_BUFFER_9:
    return 512 + 4096 * 9;
  case AMDGPUAS::CONSTANT_BUFFER_10:
    return 512 + 4096 * 10;
  case AMDGPUAS::CONSTANT_BUFFER_11:
    return 512 + 4096 * 11;
  case AMDGPUAS::CONSTANT_BUFFER_12:
    return 512 + 4096 * 12;
  case AMDGPUAS::CONSTANT_BUFFER_13:
    return 512 + 4096 * 13;
  case AMDGPUAS::CONSTANT_BUFFER_14:
    return 512 + 4096 * 14;
  case AMDGPUAS::CONSTANT_BUFFER_15:
    return 512 + 4096 * 15;
  default:
    return -1;
  }
}

SDValue R600TargetLowering::LowerLOAD(SDValue Op, SelectionDAG &DAG) const
{
  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  LoadSDNode *LoadNode = cast<LoadSDNode>(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue Ptr = Op.getOperand(1);
  SDValue LoweredLoad;

  int ConstantBlock = ConstantAddressBlock(LoadNode->getAddressSpace());
  if (ConstantBlock > -1) {
    SDValue Result;
    if (dyn_cast<ConstantExpr>(LoadNode->getSrcValue()) ||
        dyn_cast<Constant>(LoadNode->getSrcValue()) ||
        dyn_cast<ConstantSDNode>(Ptr)) {
      SDValue Slots[4];
      for (unsigned i = 0; i < 4; i++) {
        // We want Const position encoded with the following formula :
        // (((512 + (kc_bank << 12) + const_index) << 2) + chan)
        // const_index is Ptr computed by llvm using an alignment of 16.
        // Thus we add (((512 + (kc_bank << 12)) + chan ) * 4 here and
        // then div by 4 at the ISel step
        SDValue NewPtr = DAG.getNode(ISD::ADD, DL, Ptr.getValueType(), Ptr,
            DAG.getConstant(4 * i + ConstantBlock * 16, MVT::i32));
        Slots[i] = DAG.getNode(AMDGPUISD::CONST_ADDRESS, DL, MVT::i32, NewPtr);
      }
      Result = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v4i32, Slots, 4);
    } else {
      // non constant ptr cant be folded, keeps it as a v4f32 load
      Result = DAG.getNode(AMDGPUISD::CONST_ADDRESS, DL, MVT::v4i32,
          DAG.getNode(ISD::SRL, DL, MVT::i32, Ptr, DAG.getConstant(4, MVT::i32)),
          DAG.getConstant(LoadNode->getAddressSpace() -
                          AMDGPUAS::CONSTANT_BUFFER_0, MVT::i32)
          );
    }

    if (!VT.isVector()) {
      Result = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, Result,
          DAG.getConstant(0, MVT::i32));
    }

    SDValue MergedValues[2] = {
        Result,
        Chain
    };
    return DAG.getMergeValues(MergedValues, 2, DL);
  }

  if (LoadNode->getAddressSpace() != AMDGPUAS::PRIVATE_ADDRESS) {
    return SDValue();
  }

  // Lowering for indirect addressing
  const MachineFunction &MF = DAG.getMachineFunction();
  const AMDGPUFrameLowering *TFL = static_cast<const AMDGPUFrameLowering*>(
                                         getTargetMachine().getFrameLowering());
  unsigned StackWidth = TFL->getStackWidth(MF);

  Ptr = stackPtrToRegIndex(Ptr, StackWidth, DAG);

  if (VT.isVector()) {
    unsigned NumElemVT = VT.getVectorNumElements();
    EVT ElemVT = VT.getVectorElementType();
    SDValue Loads[4];

    assert(NumElemVT >= StackWidth && "Stack width cannot be greater than "
                                      "vector width in load");

    for (unsigned i = 0; i < NumElemVT; ++i) {
      unsigned Channel, PtrIncr;
      getStackAddress(StackWidth, i, Channel, PtrIncr);
      Ptr = DAG.getNode(ISD::ADD, DL, MVT::i32, Ptr,
                        DAG.getConstant(PtrIncr, MVT::i32));
      Loads[i] = DAG.getNode(AMDGPUISD::REGISTER_LOAD, DL, ElemVT,
                             Chain, Ptr,
                             DAG.getTargetConstant(Channel, MVT::i32),
                             Op.getOperand(2));
    }
    for (unsigned i = NumElemVT; i < 4; ++i) {
      Loads[i] = DAG.getUNDEF(ElemVT);
    }
    EVT TargetVT = EVT::getVectorVT(*DAG.getContext(), ElemVT, 4);
    LoweredLoad = DAG.getNode(ISD::BUILD_VECTOR, DL, TargetVT, Loads, 4);
  } else {
    LoweredLoad = DAG.getNode(AMDGPUISD::REGISTER_LOAD, DL, VT,
                              Chain, Ptr,
                              DAG.getTargetConstant(0, MVT::i32), // Channel
                              Op.getOperand(2));
  }

  SDValue Ops[2];
  Ops[0] = LoweredLoad;
  Ops[1] = Chain;

  return DAG.getMergeValues(Ops, 2, DL);
}

/// XXX Only kernel functions are supported, so we can assume for now that
/// every function is a kernel function, but in the future we should use
/// separate calling conventions for kernel and non-kernel functions.
SDValue R600TargetLowering::LowerFormalArguments(
                                      SDValue Chain,
                                      CallingConv::ID CallConv,
                                      bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                      SDLoc DL, SelectionDAG &DAG,
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
                                       MachinePointerInfo(UndefValue::get(PtrTy)),
                                       ArgVT, false, false, ArgBytes);
    InVals.push_back(Arg);
    ParamOffsetBytes += ArgBytes;
  }
  return Chain;
}

EVT R600TargetLowering::getSetCCResultType(LLVMContext &, EVT VT) const {
   if (!VT.isVector()) return MVT::i32;
   return VT.changeVectorElementTypeToInteger();
}

static SDValue
CompactSwizzlableVector(SelectionDAG &DAG, SDValue VectorEntry,
                        DenseMap<unsigned, unsigned> &RemapSwizzle) {
  assert(VectorEntry.getOpcode() == ISD::BUILD_VECTOR);
  assert(RemapSwizzle.empty());
  SDValue NewBldVec[4] = {
      VectorEntry.getOperand(0),
      VectorEntry.getOperand(1),
      VectorEntry.getOperand(2),
      VectorEntry.getOperand(3)
  };

  for (unsigned i = 0; i < 4; i++) {
    if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(NewBldVec[i])) {
      if (C->isZero()) {
        RemapSwizzle[i] = 4; // SEL_0
        NewBldVec[i] = DAG.getUNDEF(MVT::f32);
      } else if (C->isExactlyValue(1.0)) {
        RemapSwizzle[i] = 5; // SEL_1
        NewBldVec[i] = DAG.getUNDEF(MVT::f32);
      }
    }

    if (NewBldVec[i].getOpcode() == ISD::UNDEF)
      continue;
    for (unsigned j = 0; j < i; j++) {
      if (NewBldVec[i] == NewBldVec[j]) {
        NewBldVec[i] = DAG.getUNDEF(NewBldVec[i].getValueType());
        RemapSwizzle[i] = j;
        break;
      }
    }
  }

  return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(VectorEntry),
      VectorEntry.getValueType(), NewBldVec, 4);
}

static SDValue ReorganizeVector(SelectionDAG &DAG, SDValue VectorEntry,
                                DenseMap<unsigned, unsigned> &RemapSwizzle) {
  assert(VectorEntry.getOpcode() == ISD::BUILD_VECTOR);
  assert(RemapSwizzle.empty());
  SDValue NewBldVec[4] = {
      VectorEntry.getOperand(0),
      VectorEntry.getOperand(1),
      VectorEntry.getOperand(2),
      VectorEntry.getOperand(3)
  };
  bool isUnmovable[4] = { false, false, false, false };

  for (unsigned i = 0; i < 4; i++) {
    if (NewBldVec[i].getOpcode() == ISD::EXTRACT_VECTOR_ELT) {
      unsigned Idx = dyn_cast<ConstantSDNode>(NewBldVec[i].getOperand(1))
          ->getZExtValue();
      if (!isUnmovable[Idx]) {
        // Swap i and Idx
        std::swap(NewBldVec[Idx], NewBldVec[i]);
        RemapSwizzle[Idx] = i;
        RemapSwizzle[i] = Idx;
      }
      isUnmovable[Idx] = true;
    }
  }

  return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(VectorEntry),
      VectorEntry.getValueType(), NewBldVec, 4);
}


SDValue R600TargetLowering::OptimizeSwizzle(SDValue BuildVector,
SDValue Swz[4], SelectionDAG &DAG) const {
  assert(BuildVector.getOpcode() == ISD::BUILD_VECTOR);
  // Old -> New swizzle values
  DenseMap<unsigned, unsigned> SwizzleRemap;

  BuildVector = CompactSwizzlableVector(DAG, BuildVector, SwizzleRemap);
  for (unsigned i = 0; i < 4; i++) {
    unsigned Idx = dyn_cast<ConstantSDNode>(Swz[i])->getZExtValue();
    if (SwizzleRemap.find(Idx) != SwizzleRemap.end())
      Swz[i] = DAG.getConstant(SwizzleRemap[Idx], MVT::i32);
  }

  SwizzleRemap.clear();
  BuildVector = ReorganizeVector(DAG, BuildVector, SwizzleRemap);
  for (unsigned i = 0; i < 4; i++) {
    unsigned Idx = dyn_cast<ConstantSDNode>(Swz[i])->getZExtValue();
    if (SwizzleRemap.find(Idx) != SwizzleRemap.end())
      Swz[i] = DAG.getConstant(SwizzleRemap[Idx], MVT::i32);
  }

  return BuildVector;
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
        return DAG.getNode(ISD::UINT_TO_FP, SDLoc(N), N->getValueType(0),
                           Arg.getOperand(0));
      }
      break;
    }

  // (i32 fp_to_sint (fneg (select_cc f32, f32, 1.0, 0.0 cc))) ->
  // (i32 select_cc f32, f32, -1, 0 cc)
  //
  // Mesa's GLSL frontend generates the above pattern a lot and we can lower
  // this to one of the SET*_DX10 instructions.
  case ISD::FP_TO_SINT: {
    SDValue FNeg = N->getOperand(0);
    if (FNeg.getOpcode() != ISD::FNEG) {
      return SDValue();
    }
    SDValue SelectCC = FNeg.getOperand(0);
    if (SelectCC.getOpcode() != ISD::SELECT_CC ||
        SelectCC.getOperand(0).getValueType() != MVT::f32 || // LHS
        SelectCC.getOperand(2).getValueType() != MVT::f32 || // True
        !isHWTrueValue(SelectCC.getOperand(2)) ||
        !isHWFalseValue(SelectCC.getOperand(3))) {
      return SDValue();
    }

    return DAG.getNode(ISD::SELECT_CC, SDLoc(N), N->getValueType(0),
                           SelectCC.getOperand(0), // LHS
                           SelectCC.getOperand(1), // RHS
                           DAG.getConstant(-1, MVT::i32), // True
                           DAG.getConstant(0, MVT::i32),  // Flase
                           SelectCC.getOperand(4)); // CC

    break;
  }
  // Extract_vec (Build_vector) generated by custom lowering
  // also needs to be customly combined
  case ISD::EXTRACT_VECTOR_ELT: {
    SDValue Arg = N->getOperand(0);
    if (Arg.getOpcode() == ISD::BUILD_VECTOR) {
      if (ConstantSDNode *Const = dyn_cast<ConstantSDNode>(N->getOperand(1))) {
        unsigned Element = Const->getZExtValue();
        return Arg->getOperand(Element);
      }
    }
    if (Arg.getOpcode() == ISD::BITCAST &&
        Arg.getOperand(0).getOpcode() == ISD::BUILD_VECTOR) {
      if (ConstantSDNode *Const = dyn_cast<ConstantSDNode>(N->getOperand(1))) {
        unsigned Element = Const->getZExtValue();
        return DAG.getNode(ISD::BITCAST, SDLoc(N), N->getVTList(),
            Arg->getOperand(0).getOperand(Element));
      }
    }
  }

  case ISD::SELECT_CC: {
    // fold selectcc (selectcc x, y, a, b, cc), b, a, b, seteq ->
    //      selectcc x, y, a, b, inv(cc)
    //
    // fold selectcc (selectcc x, y, a, b, cc), b, a, b, setne ->
    //      selectcc x, y, a, b, cc
    SDValue LHS = N->getOperand(0);
    if (LHS.getOpcode() != ISD::SELECT_CC) {
      return SDValue();
    }

    SDValue RHS = N->getOperand(1);
    SDValue True = N->getOperand(2);
    SDValue False = N->getOperand(3);
    ISD::CondCode NCC = cast<CondCodeSDNode>(N->getOperand(4))->get();

    if (LHS.getOperand(2).getNode() != True.getNode() ||
        LHS.getOperand(3).getNode() != False.getNode() ||
        RHS.getNode() != False.getNode()) {
      return SDValue();
    }

    switch (NCC) {
    default: return SDValue();
    case ISD::SETNE: return LHS;
    case ISD::SETEQ: {
      ISD::CondCode LHSCC = cast<CondCodeSDNode>(LHS.getOperand(4))->get();
      LHSCC = ISD::getSetCCInverse(LHSCC,
                                  LHS.getOperand(0).getValueType().isInteger());
      return DAG.getSelectCC(SDLoc(N),
                             LHS.getOperand(0),
                             LHS.getOperand(1),
                             LHS.getOperand(2),
                             LHS.getOperand(3),
                             LHSCC);
    }
    }
  }
  case AMDGPUISD::EXPORT: {
    SDValue Arg = N->getOperand(1);
    if (Arg.getOpcode() != ISD::BUILD_VECTOR)
      break;

    SDValue NewArgs[8] = {
      N->getOperand(0), // Chain
      SDValue(),
      N->getOperand(2), // ArrayBase
      N->getOperand(3), // Type
      N->getOperand(4), // SWZ_X
      N->getOperand(5), // SWZ_Y
      N->getOperand(6), // SWZ_Z
      N->getOperand(7) // SWZ_W
    };
    SDLoc DL(N);
    NewArgs[1] = OptimizeSwizzle(N->getOperand(1), &NewArgs[4], DAG);
    return DAG.getNode(AMDGPUISD::EXPORT, DL, N->getVTList(), NewArgs, 8);
  }
  case AMDGPUISD::TEXTURE_FETCH: {
    SDValue Arg = N->getOperand(1);
    if (Arg.getOpcode() != ISD::BUILD_VECTOR)
      break;

    SDValue NewArgs[19] = {
      N->getOperand(0),
      N->getOperand(1),
      N->getOperand(2),
      N->getOperand(3),
      N->getOperand(4),
      N->getOperand(5),
      N->getOperand(6),
      N->getOperand(7),
      N->getOperand(8),
      N->getOperand(9),
      N->getOperand(10),
      N->getOperand(11),
      N->getOperand(12),
      N->getOperand(13),
      N->getOperand(14),
      N->getOperand(15),
      N->getOperand(16),
      N->getOperand(17),
      N->getOperand(18),
    };
    NewArgs[1] = OptimizeSwizzle(N->getOperand(1), &NewArgs[2], DAG);
    return DAG.getNode(AMDGPUISD::TEXTURE_FETCH, SDLoc(N), N->getVTList(),
        NewArgs, 19);
  }
  }
  return SDValue();
}
