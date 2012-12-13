//===-- R600ExpandSpecialInstrs.cpp - Expand special instructions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Vector, Reduction, and Cube instructions need to fill the entire instruction
/// group to work correctly.  This pass expands these individual instructions
/// into several instructions that will completely fill the instruction group.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "R600Defines.h"
#include "R600InstrInfo.h"
#include "R600RegisterInfo.h"
#include "R600MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

namespace {

class R600ExpandSpecialInstrsPass : public MachineFunctionPass {

private:
  static char ID;
  const R600InstrInfo *TII;

  bool ExpandInputPerspective(MachineInstr& MI);
  bool ExpandInputConstant(MachineInstr& MI);

public:
  R600ExpandSpecialInstrsPass(TargetMachine &tm) : MachineFunctionPass(ID),
    TII (static_cast<const R600InstrInfo *>(tm.getInstrInfo())) { }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  const char *getPassName() const {
    return "R600 Expand special instructions pass";
  }
};

} // End anonymous namespace

char R600ExpandSpecialInstrsPass::ID = 0;

FunctionPass *llvm::createR600ExpandSpecialInstrsPass(TargetMachine &TM) {
  return new R600ExpandSpecialInstrsPass(TM);
}

bool R600ExpandSpecialInstrsPass::ExpandInputPerspective(MachineInstr &MI) {
  const R600RegisterInfo &TRI = TII->getRegisterInfo();
  if (MI.getOpcode() != AMDGPU::input_perspective)
    return false;

  MachineBasicBlock::iterator I = &MI;
  unsigned DstReg = MI.getOperand(0).getReg();
  R600MachineFunctionInfo *MFI = MI.getParent()->getParent()
      ->getInfo<R600MachineFunctionInfo>();
  unsigned IJIndexBase;

  // In Evergreen ISA doc section 8.3.2 :
  // We need to interpolate XY and ZW in two different instruction groups.
  // An INTERP_* must occupy all 4 slots of an instruction group.
  // Output of INTERP_XY is written in X,Y slots
  // Output of INTERP_ZW is written in Z,W slots
  //
  // Thus interpolation requires the following sequences :
  //
  // AnyGPR.x = INTERP_ZW; (Write Masked Out)
  // AnyGPR.y = INTERP_ZW; (Write Masked Out)
  // DstGPR.z = INTERP_ZW;
  // DstGPR.w = INTERP_ZW; (End of first IG)
  // DstGPR.x = INTERP_XY;
  // DstGPR.y = INTERP_XY;
  // AnyGPR.z = INTERP_XY; (Write Masked Out)
  // AnyGPR.w = INTERP_XY; (Write Masked Out) (End of second IG)
  //
  switch (MI.getOperand(1).getImm()) {
  case 0:
    IJIndexBase = MFI->GetIJPerspectiveIndex();
    break;
  case 1:
    IJIndexBase = MFI->GetIJLinearIndex();
    break;
  default:
    assert(0 && "Unknow ij index");
  }

  for (unsigned i = 0; i < 8; i++) {
    unsigned IJIndex = AMDGPU::R600_TReg32RegClass.getRegister(
        2 * IJIndexBase + ((i + 1) % 2));
    unsigned ReadReg = AMDGPU::R600_ArrayBaseRegClass.getRegister(
        MI.getOperand(2).getImm());


    unsigned Sel = AMDGPU::sel_x;
    switch (i % 4) {
    case 0:Sel = AMDGPU::sel_x;break;
    case 1:Sel = AMDGPU::sel_y;break;
    case 2:Sel = AMDGPU::sel_z;break;
    case 3:Sel = AMDGPU::sel_w;break;
    default:break;
    }

    unsigned Res = TRI.getSubReg(DstReg, Sel);

    unsigned Opcode = (i < 4)?AMDGPU::INTERP_ZW:AMDGPU::INTERP_XY;

    MachineBasicBlock &MBB = *(MI.getParent());
    MachineInstr *NewMI =
        TII->buildDefaultInstruction(MBB, I, Opcode, Res, IJIndex, ReadReg);

    if (!(i> 1 && i < 6)) {
      TII->addFlag(NewMI, 0, MO_FLAG_MASK);
    }

    if (i % 4 !=  3)
      TII->addFlag(NewMI, 0, MO_FLAG_NOT_LAST);
  }

  MI.eraseFromParent();

  return true;
}

bool R600ExpandSpecialInstrsPass::ExpandInputConstant(MachineInstr &MI) {
  const R600RegisterInfo &TRI = TII->getRegisterInfo();
  if (MI.getOpcode() != AMDGPU::input_constant)
    return false;

  MachineBasicBlock::iterator I = &MI;
  unsigned DstReg = MI.getOperand(0).getReg();

  for (unsigned i = 0; i < 4; i++) {
    unsigned ReadReg = AMDGPU::R600_ArrayBaseRegClass.getRegister(
        MI.getOperand(1).getImm());

    unsigned Sel = AMDGPU::sel_x;
    switch (i % 4) {
    case 0:Sel = AMDGPU::sel_x;break;
    case 1:Sel = AMDGPU::sel_y;break;
    case 2:Sel = AMDGPU::sel_z;break;
    case 3:Sel = AMDGPU::sel_w;break;
    default:break;
    }

    unsigned Res = TRI.getSubReg(DstReg, Sel);

    MachineBasicBlock &MBB = *(MI.getParent());
    MachineInstr *NewMI = TII->buildDefaultInstruction(
        MBB, I, AMDGPU::INTERP_LOAD_P0, Res, ReadReg);

    if (i % 4 !=  3)
      TII->addFlag(NewMI, 0, MO_FLAG_NOT_LAST);
  }

  MI.eraseFromParent();

  return true;
}

bool R600ExpandSpecialInstrsPass::runOnMachineFunction(MachineFunction &MF) {

  const R600RegisterInfo &TRI = TII->getRegisterInfo();

  for (MachineFunction::iterator BB = MF.begin(), BB_E = MF.end();
                                                  BB != BB_E; ++BB) {
    MachineBasicBlock &MBB = *BB;
    MachineBasicBlock::iterator I = MBB.begin();
    while (I != MBB.end()) {
      MachineInstr &MI = *I;
      I = llvm::next(I);

      switch (MI.getOpcode()) {
      default: break;
      // Expand PRED_X to one of the PRED_SET instructions.
      case AMDGPU::PRED_X: {
        uint64_t Flags = MI.getOperand(3).getImm();
        // The native opcode used by PRED_X is stored as an immediate in the
        // third operand.
        MachineInstr *PredSet = TII->buildDefaultInstruction(MBB, I,
                                            MI.getOperand(2).getImm(), // opcode
                                            MI.getOperand(0).getReg(), // dst
                                            MI.getOperand(1).getReg(), // src0
                                            AMDGPU::ZERO);             // src1
        TII->addFlag(PredSet, 0, MO_FLAG_MASK);
        if (Flags & MO_FLAG_PUSH) {
          TII->setImmOperand(PredSet, R600Operands::UPDATE_EXEC_MASK, 1);
        } else {
          TII->setImmOperand(PredSet, R600Operands::UPDATE_PREDICATE, 1);
        }
        MI.eraseFromParent();
        continue;
        }
      case AMDGPU::BREAK:
        MachineInstr *PredSet = TII->buildDefaultInstruction(MBB, I,
                                          AMDGPU::PRED_SETE_INT,
                                          AMDGPU::PREDICATE_BIT,
                                          AMDGPU::ZERO,
                                          AMDGPU::ZERO);
        TII->addFlag(PredSet, 0, MO_FLAG_MASK);
        TII->setImmOperand(PredSet, R600Operands::UPDATE_EXEC_MASK, 1);

        BuildMI(MBB, I, MBB.findDebugLoc(I),
                TII->get(AMDGPU::PREDICATED_BREAK))
                .addReg(AMDGPU::PREDICATE_BIT);
        MI.eraseFromParent();
        continue;
    }

    if (ExpandInputPerspective(MI))
      continue;
    if (ExpandInputConstant(MI))
      continue;

      bool IsReduction = TII->isReductionOp(MI.getOpcode());
      bool IsVector = TII->isVector(MI);
      bool IsCube = TII->isCubeOp(MI.getOpcode());
      if (!IsReduction && !IsVector && !IsCube) {
        continue;
      }

      // Expand the instruction
      //
      // Reduction instructions:
      // T0_X = DP4 T1_XYZW, T2_XYZW
      // becomes:
      // TO_X = DP4 T1_X, T2_X
      // TO_Y (write masked) = DP4 T1_Y, T2_Y
      // TO_Z (write masked) = DP4 T1_Z, T2_Z
      // TO_W (write masked) = DP4 T1_W, T2_W
      //
      // Vector instructions:
      // T0_X = MULLO_INT T1_X, T2_X
      // becomes:
      // T0_X = MULLO_INT T1_X, T2_X
      // T0_Y (write masked) = MULLO_INT T1_X, T2_X
      // T0_Z (write masked) = MULLO_INT T1_X, T2_X
      // T0_W (write masked) = MULLO_INT T1_X, T2_X
      //
      // Cube instructions:
      // T0_XYZW = CUBE T1_XYZW
      // becomes:
      // TO_X = CUBE T1_Z, T1_Y
      // T0_Y = CUBE T1_Z, T1_X
      // T0_Z = CUBE T1_X, T1_Z
      // T0_W = CUBE T1_Y, T1_Z
      for (unsigned Chan = 0; Chan < 4; Chan++) {
        unsigned DstReg = MI.getOperand(
                            TII->getOperandIdx(MI, R600Operands::DST)).getReg();
        unsigned Src0 = MI.getOperand(
                           TII->getOperandIdx(MI, R600Operands::SRC0)).getReg();
        unsigned Src1 = 0;

        // Determine the correct source registers
        if (!IsCube) {
          int Src1Idx = TII->getOperandIdx(MI, R600Operands::SRC1);
          if (Src1Idx != -1) {
            Src1 = MI.getOperand(Src1Idx).getReg();
          }
        }
        if (IsReduction) {
          unsigned SubRegIndex = TRI.getSubRegFromChannel(Chan);
          Src0 = TRI.getSubReg(Src0, SubRegIndex);
          Src1 = TRI.getSubReg(Src1, SubRegIndex);
        } else if (IsCube) {
          static const int CubeSrcSwz[] = {2, 2, 0, 1};
          unsigned SubRegIndex0 = TRI.getSubRegFromChannel(CubeSrcSwz[Chan]);
          unsigned SubRegIndex1 = TRI.getSubRegFromChannel(CubeSrcSwz[3 - Chan]);
          Src1 = TRI.getSubReg(Src0, SubRegIndex1);
          Src0 = TRI.getSubReg(Src0, SubRegIndex0);
        }

        // Determine the correct destination registers;
        bool Mask = false;
        bool NotLast = true;
        if (IsCube) {
          unsigned SubRegIndex = TRI.getSubRegFromChannel(Chan);
          DstReg = TRI.getSubReg(DstReg, SubRegIndex);
        } else {
          // Mask the write if the original instruction does not write to
          // the current Channel.
          Mask = (Chan != TRI.getHWRegChan(DstReg));
          unsigned DstBase = TRI.getEncodingValue(DstReg) & HW_REG_MASK;
          DstReg = AMDGPU::R600_TReg32RegClass.getRegister((DstBase * 4) + Chan);
        }

        // Set the IsLast bit
        NotLast = (Chan != 3 );

        // Add the new instruction
        unsigned Opcode = MI.getOpcode();
        switch (Opcode) {
        case AMDGPU::CUBE_r600_pseudo:
          Opcode = AMDGPU::CUBE_r600_real;
          break;
        case AMDGPU::CUBE_eg_pseudo:
          Opcode = AMDGPU::CUBE_eg_real;
          break;
        case AMDGPU::DOT4_r600_pseudo:
          Opcode = AMDGPU::DOT4_r600_real;
          break;
        case AMDGPU::DOT4_eg_pseudo:
          Opcode = AMDGPU::DOT4_eg_real;
          break;
        default:
          break;
        }

        MachineInstr *NewMI =
          TII->buildDefaultInstruction(MBB, I, Opcode, DstReg, Src0, Src1);

        if (Chan != 0)
          NewMI->bundleWithPred();
        if (Mask) {
          TII->addFlag(NewMI, 0, MO_FLAG_MASK);
        }
        if (NotLast) {
          TII->addFlag(NewMI, 0, MO_FLAG_NOT_LAST);
        }
      }
      MI.eraseFromParent();
    }
  }
  return false;
}
