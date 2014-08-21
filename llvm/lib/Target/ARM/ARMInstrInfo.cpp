//===-- ARMInstrInfo.cpp - ARM Instruction Information --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMInstrInfo.h"
#include "ARM.h"
#include "ARMConstantPoolValue.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMTargetMachine.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
using namespace llvm;

ARMInstrInfo::ARMInstrInfo(const ARMSubtarget &STI)
  : ARMBaseInstrInfo(STI), RI(STI) {
}

/// getNoopForMachoTarget - Return the noop instruction to use for a noop.
void ARMInstrInfo::getNoopForMachoTarget(MCInst &NopInst) const {
  if (hasNOP()) {
    NopInst.setOpcode(ARM::HINT);
    NopInst.addOperand(MCOperand::CreateImm(0));
    NopInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    NopInst.addOperand(MCOperand::CreateReg(0));
  } else {
    NopInst.setOpcode(ARM::MOVr);
    NopInst.addOperand(MCOperand::CreateReg(ARM::R0));
    NopInst.addOperand(MCOperand::CreateReg(ARM::R0));
    NopInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    NopInst.addOperand(MCOperand::CreateReg(0));
    NopInst.addOperand(MCOperand::CreateReg(0));
  }
}

unsigned ARMInstrInfo::getUnindexedOpcode(unsigned Opc) const {
  switch (Opc) {
  default: break;
  case ARM::LDR_PRE_IMM:
  case ARM::LDR_PRE_REG:
  case ARM::LDR_POST_IMM:
  case ARM::LDR_POST_REG:
    return ARM::LDRi12;
  case ARM::LDRH_PRE:
  case ARM::LDRH_POST:
    return ARM::LDRH;
  case ARM::LDRB_PRE_IMM:
  case ARM::LDRB_PRE_REG:
  case ARM::LDRB_POST_IMM:
  case ARM::LDRB_POST_REG:
    return ARM::LDRBi12;
  case ARM::LDRSH_PRE:
  case ARM::LDRSH_POST:
    return ARM::LDRSH;
  case ARM::LDRSB_PRE:
  case ARM::LDRSB_POST:
    return ARM::LDRSB;
  case ARM::STR_PRE_IMM:
  case ARM::STR_PRE_REG:
  case ARM::STR_POST_IMM:
  case ARM::STR_POST_REG:
    return ARM::STRi12;
  case ARM::STRH_PRE:
  case ARM::STRH_POST:
    return ARM::STRH;
  case ARM::STRB_PRE_IMM:
  case ARM::STRB_PRE_REG:
  case ARM::STRB_POST_IMM:
  case ARM::STRB_POST_REG:
    return ARM::STRBi12;
  }

  return 0;
}

void ARMInstrInfo::expandLoadStackGuard(MachineBasicBlock::iterator MI,
                                        Reloc::Model RM) const {
  if (RM == Reloc::PIC_)
    expandLoadStackGuardBase(MI, ARM::LDRLIT_ga_pcrel, ARM::LDRi12, RM);
  else
    expandLoadStackGuardBase(MI, ARM::LDRLIT_ga_abs, ARM::LDRi12, RM);
}

bool ARMInstrInfo::getRegSequenceLikeInputs(
    const MachineInstr &MI, unsigned DefIdx,
    SmallVectorImpl<RegSubRegPairAndIdx> &InputRegs) const {
  assert(DefIdx < MI.getDesc().getNumDefs() && "Invalid definition index");
  assert(MI.isRegSequenceLike() && "Invalid kind of instruction");

  switch (MI.getOpcode()) {
  case ARM::VMOVDRR:
    // dX = VMOVDRR rY, rZ
    // is the same as:
    // dX = REG_SEQUENCE rY, ssub_0, rZ, ssub_1
    // Populate the InputRegs accordingly.
    // rY
    const MachineOperand *MOReg = &MI.getOperand(1);
    InputRegs.push_back(
        RegSubRegPairAndIdx(MOReg->getReg(), MOReg->getSubReg(), ARM::ssub_0));
    // rZ
    MOReg = &MI.getOperand(2);
    InputRegs.push_back(
        RegSubRegPairAndIdx(MOReg->getReg(), MOReg->getSubReg(), ARM::ssub_1));
    return true;
  }
  llvm_unreachable("Target dependent opcode missing");
}

bool ARMInstrInfo::getExtractSubregLikeInputs(
    const MachineInstr &MI, unsigned DefIdx,
    RegSubRegPairAndIdx &InputReg) const {
  assert(DefIdx < MI.getDesc().getNumDefs() && "Invalid definition index");
  assert(MI.isExtractSubregLike() && "Invalid kind of instruction");

  switch (MI.getOpcode()) {
  case ARM::VMOVRRD:
    // rX, rY = VMOVRRD dZ
    // is the same as:
    // rX = EXTRACT_SUBREG dZ, ssub_0
    // rY = EXTRACT_SUBREG dZ, ssub_1
    const MachineOperand &MOReg = MI.getOperand(2);
    InputReg.Reg = MOReg.getReg();
    InputReg.SubReg = MOReg.getSubReg();
    InputReg.SubIdx = DefIdx == 0 ? ARM::ssub_0 : ARM::ssub_1;
    return true;
  }
  llvm_unreachable("Target dependent opcode missing");
}

bool ARMInstrInfo::getInsertSubregLikeInputs(
    const MachineInstr &MI, unsigned DefIdx, RegSubRegPair &BaseReg,
    RegSubRegPairAndIdx &InsertedReg) const {
  assert(DefIdx < MI.getDesc().getNumDefs() && "Invalid definition index");
  assert(MI.isInsertSubregLike() && "Invalid kind of instruction");

  switch (MI.getOpcode()) {
  case ARM::VSETLNi32:
    // dX = VSETLNi32 dY, rZ, imm
    const MachineOperand &MOBaseReg = MI.getOperand(1);
    const MachineOperand &MOInsertedReg = MI.getOperand(2);
    const MachineOperand &MOIndex = MI.getOperand(3);
    BaseReg.Reg = MOBaseReg.getReg();
    BaseReg.SubReg = MOBaseReg.getSubReg();

    InsertedReg.Reg = MOInsertedReg.getReg();
    InsertedReg.SubReg = MOInsertedReg.getSubReg();
    InsertedReg.SubIdx = MOIndex.getImm() == 0 ? ARM::ssub_0 : ARM::ssub_1;
    return true;
  }
  llvm_unreachable("Target dependent opcode missing");
}

namespace {
  /// ARMCGBR - Create Global Base Reg pass. This initializes the PIC
  /// global base register for ARM ELF.
  struct ARMCGBR : public MachineFunctionPass {
    static char ID;
    ARMCGBR() : MachineFunctionPass(ID) {}

    bool runOnMachineFunction(MachineFunction &MF) override {
      ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
      if (AFI->getGlobalBaseReg() == 0)
        return false;

      const ARMTargetMachine *TM =
        static_cast<const ARMTargetMachine *>(&MF.getTarget());
      if (TM->getRelocationModel() != Reloc::PIC_)
        return false;

      LLVMContext *Context = &MF.getFunction()->getContext();
      unsigned ARMPCLabelIndex = AFI->createPICLabelUId();
      unsigned PCAdj = TM->getSubtarget<ARMSubtarget>().isThumb() ? 4 : 8;
      ARMConstantPoolValue *CPV = ARMConstantPoolSymbol::Create(
          *Context, "_GLOBAL_OFFSET_TABLE_", ARMPCLabelIndex, PCAdj);

      unsigned Align =
          TM->getSubtargetImpl()->getDataLayout()->getPrefTypeAlignment(
              Type::getInt32PtrTy(*Context));
      unsigned Idx = MF.getConstantPool()->getConstantPoolIndex(CPV, Align);

      MachineBasicBlock &FirstMBB = MF.front();
      MachineBasicBlock::iterator MBBI = FirstMBB.begin();
      DebugLoc DL = FirstMBB.findDebugLoc(MBBI);
      unsigned TempReg =
          MF.getRegInfo().createVirtualRegister(&ARM::rGPRRegClass);
      unsigned Opc = TM->getSubtarget<ARMSubtarget>().isThumb2() ?
                     ARM::t2LDRpci : ARM::LDRcp;
      const TargetInstrInfo &TII = *TM->getSubtargetImpl()->getInstrInfo();
      MachineInstrBuilder MIB = BuildMI(FirstMBB, MBBI, DL,
                                        TII.get(Opc), TempReg)
                                .addConstantPoolIndex(Idx);
      if (Opc == ARM::LDRcp)
        MIB.addImm(0);
      AddDefaultPred(MIB);

      // Fix the GOT address by adding pc.
      unsigned GlobalBaseReg = AFI->getGlobalBaseReg();
      Opc = TM->getSubtarget<ARMSubtarget>().isThumb2() ? ARM::tPICADD
                                                        : ARM::PICADD;
      MIB = BuildMI(FirstMBB, MBBI, DL, TII.get(Opc), GlobalBaseReg)
                .addReg(TempReg)
                .addImm(ARMPCLabelIndex);
      if (Opc == ARM::PICADD)
        AddDefaultPred(MIB);


      return true;
    }

    const char *getPassName() const override {
      return "ARM PIC Global Base Reg Initialization";
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
}

char ARMCGBR::ID = 0;
FunctionPass*
llvm::createARMGlobalBaseRegPass() { return new ARMCGBR(); }
