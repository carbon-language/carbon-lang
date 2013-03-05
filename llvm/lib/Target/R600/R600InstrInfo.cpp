//===-- R600InstrInfo.cpp - R600 Instruction Information ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief R600 Implementation of TargetInstrInfo.
//
//===----------------------------------------------------------------------===//

#include "R600InstrInfo.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"
#include "R600Defines.h"
#include "R600MachineFunctionInfo.h"
#include "R600RegisterInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

#define GET_INSTRINFO_CTOR
#include "AMDGPUGenDFAPacketizer.inc"

using namespace llvm;

R600InstrInfo::R600InstrInfo(AMDGPUTargetMachine &tm)
  : AMDGPUInstrInfo(tm),
    RI(tm, *this)
  { }

const R600RegisterInfo &R600InstrInfo::getRegisterInfo() const {
  return RI;
}

bool R600InstrInfo::isTrig(const MachineInstr &MI) const {
  return get(MI.getOpcode()).TSFlags & R600_InstFlag::TRIG;
}

bool R600InstrInfo::isVector(const MachineInstr &MI) const {
  return get(MI.getOpcode()).TSFlags & R600_InstFlag::VECTOR;
}

void
R600InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const {
  if (AMDGPU::R600_Reg128RegClass.contains(DestReg)
      && AMDGPU::R600_Reg128RegClass.contains(SrcReg)) {
    for (unsigned I = 0; I < 4; I++) {
      unsigned SubRegIndex = RI.getSubRegFromChannel(I);
      buildDefaultInstruction(MBB, MI, AMDGPU::MOV,
                              RI.getSubReg(DestReg, SubRegIndex),
                              RI.getSubReg(SrcReg, SubRegIndex))
                              .addReg(DestReg,
                                      RegState::Define | RegState::Implicit);
    }
  } else {

    // We can't copy vec4 registers
    assert(!AMDGPU::R600_Reg128RegClass.contains(DestReg)
           && !AMDGPU::R600_Reg128RegClass.contains(SrcReg));

    MachineInstr *NewMI = buildDefaultInstruction(MBB, MI, AMDGPU::MOV,
                                                  DestReg, SrcReg);
    NewMI->getOperand(getOperandIdx(*NewMI, R600Operands::SRC0))
                                    .setIsKill(KillSrc);
  }
}

MachineInstr * R600InstrInfo::getMovImmInstr(MachineFunction *MF,
                                             unsigned DstReg, int64_t Imm) const {
  MachineInstr * MI = MF->CreateMachineInstr(get(AMDGPU::MOV), DebugLoc());
  MachineInstrBuilder MIB(*MF, MI);
  MIB.addReg(DstReg, RegState::Define);
  MIB.addReg(AMDGPU::ALU_LITERAL_X);
  MIB.addImm(Imm);
  MIB.addReg(0); // PREDICATE_BIT

  return MI;
}

unsigned R600InstrInfo::getIEQOpcode() const {
  return AMDGPU::SETE_INT;
}

bool R600InstrInfo::isMov(unsigned Opcode) const {


  switch(Opcode) {
  default: return false;
  case AMDGPU::MOV:
  case AMDGPU::MOV_IMM_F32:
  case AMDGPU::MOV_IMM_I32:
    return true;
  }
}

// Some instructions act as place holders to emulate operations that the GPU
// hardware does automatically. This function can be used to check if
// an opcode falls into this category.
bool R600InstrInfo::isPlaceHolderOpcode(unsigned Opcode) const {
  switch (Opcode) {
  default: return false;
  case AMDGPU::RETURN:
    return true;
  }
}

bool R600InstrInfo::isReductionOp(unsigned Opcode) const {
  switch(Opcode) {
    default: return false;
    case AMDGPU::DOT4_r600_pseudo:
    case AMDGPU::DOT4_eg_pseudo:
      return true;
  }
}

bool R600InstrInfo::isCubeOp(unsigned Opcode) const {
  switch(Opcode) {
    default: return false;
    case AMDGPU::CUBE_r600_pseudo:
    case AMDGPU::CUBE_r600_real:
    case AMDGPU::CUBE_eg_pseudo:
    case AMDGPU::CUBE_eg_real:
      return true;
  }
}

bool R600InstrInfo::isALUInstr(unsigned Opcode) const {
  unsigned TargetFlags = get(Opcode).TSFlags;

  return ((TargetFlags & R600_InstFlag::OP1) |
          (TargetFlags & R600_InstFlag::OP2) |
          (TargetFlags & R600_InstFlag::OP3));
}

DFAPacketizer *R600InstrInfo::CreateTargetScheduleState(const TargetMachine *TM,
    const ScheduleDAG *DAG) const {
  const InstrItineraryData *II = TM->getInstrItineraryData();
  return TM->getSubtarget<AMDGPUSubtarget>().createDFAPacketizer(II);
}

static bool
isPredicateSetter(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::PRED_X:
    return true;
  default:
    return false;
  }
}

static MachineInstr *
findFirstPredicateSetterFrom(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator I) {
  while (I != MBB.begin()) {
    --I;
    MachineInstr *MI = I;
    if (isPredicateSetter(MI->getOpcode()))
      return MI;
  }

  return NULL;
}

bool
R600InstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                             MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond,
                             bool AllowModify) const {
  // Most of the following comes from the ARM implementation of AnalyzeBranch

  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin())
    return false;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return false;
    --I;
  }
  if (static_cast<MachineInstr *>(I)->getOpcode() != AMDGPU::JUMP) {
    return false;
  }

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;

  // If there is only one terminator instruction, process it.
  unsigned LastOpc = LastInst->getOpcode();
  if (I == MBB.begin() ||
      static_cast<MachineInstr *>(--I)->getOpcode() != AMDGPU::JUMP) {
    if (LastOpc == AMDGPU::JUMP) {
      if(!isPredicated(LastInst)) {
        TBB = LastInst->getOperand(0).getMBB();
        return false;
      } else {
        MachineInstr *predSet = I;
        while (!isPredicateSetter(predSet->getOpcode())) {
          predSet = --I;
        }
        TBB = LastInst->getOperand(0).getMBB();
        Cond.push_back(predSet->getOperand(1));
        Cond.push_back(predSet->getOperand(2));
        Cond.push_back(MachineOperand::CreateReg(AMDGPU::PRED_SEL_ONE, false));
        return false;
      }
    }
    return true;  // Can't handle indirect branch.
  }

  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = I;
  unsigned SecondLastOpc = SecondLastInst->getOpcode();

  // If the block ends with a B and a Bcc, handle it.
  if (SecondLastOpc == AMDGPU::JUMP &&
      isPredicated(SecondLastInst) &&
      LastOpc == AMDGPU::JUMP &&
      !isPredicated(LastInst)) {
    MachineInstr *predSet = --I;
    while (!isPredicateSetter(predSet->getOpcode())) {
      predSet = --I;
    }
    TBB = SecondLastInst->getOperand(0).getMBB();
    FBB = LastInst->getOperand(0).getMBB();
    Cond.push_back(predSet->getOperand(1));
    Cond.push_back(predSet->getOperand(2));
    Cond.push_back(MachineOperand::CreateReg(AMDGPU::PRED_SEL_ONE, false));
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

int R600InstrInfo::getBranchInstr(const MachineOperand &op) const {
  const MachineInstr *MI = op.getParent();

  switch (MI->getDesc().OpInfo->RegClass) {
  default: // FIXME: fallthrough??
  case AMDGPU::GPRI32RegClassID: return AMDGPU::BRANCH_COND_i32;
  case AMDGPU::GPRF32RegClassID: return AMDGPU::BRANCH_COND_f32;
  };
}

unsigned
R600InstrInfo::InsertBranch(MachineBasicBlock &MBB,
                            MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB,
                            const SmallVectorImpl<MachineOperand> &Cond,
                            DebugLoc DL) const {
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");

  if (FBB == 0) {
    if (Cond.empty()) {
      BuildMI(&MBB, DL, get(AMDGPU::JUMP)).addMBB(TBB).addReg(0);
      return 1;
    } else {
      MachineInstr *PredSet = findFirstPredicateSetterFrom(MBB, MBB.end());
      assert(PredSet && "No previous predicate !");
      addFlag(PredSet, 0, MO_FLAG_PUSH);
      PredSet->getOperand(2).setImm(Cond[1].getImm());

      BuildMI(&MBB, DL, get(AMDGPU::JUMP))
             .addMBB(TBB)
             .addReg(AMDGPU::PREDICATE_BIT, RegState::Kill);
      return 1;
    }
  } else {
    MachineInstr *PredSet = findFirstPredicateSetterFrom(MBB, MBB.end());
    assert(PredSet && "No previous predicate !");
    addFlag(PredSet, 0, MO_FLAG_PUSH);
    PredSet->getOperand(2).setImm(Cond[1].getImm());
    BuildMI(&MBB, DL, get(AMDGPU::JUMP))
            .addMBB(TBB)
            .addReg(AMDGPU::PREDICATE_BIT, RegState::Kill);
    BuildMI(&MBB, DL, get(AMDGPU::JUMP)).addMBB(FBB).addReg(0);
    return 2;
  }
}

unsigned
R600InstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {

  // Note : we leave PRED* instructions there.
  // They may be needed when predicating instructions.

  MachineBasicBlock::iterator I = MBB.end();

  if (I == MBB.begin()) {
    return 0;
  }
  --I;
  switch (I->getOpcode()) {
  default:
    return 0;
  case AMDGPU::JUMP:
    if (isPredicated(I)) {
      MachineInstr *predSet = findFirstPredicateSetterFrom(MBB, I);
      clearFlag(predSet, 0, MO_FLAG_PUSH);
    }
    I->eraseFromParent();
    break;
  }
  I = MBB.end();

  if (I == MBB.begin()) {
    return 1;
  }
  --I;
  switch (I->getOpcode()) {
    // FIXME: only one case??
  default:
    return 1;
  case AMDGPU::JUMP:
    if (isPredicated(I)) {
      MachineInstr *predSet = findFirstPredicateSetterFrom(MBB, I);
      clearFlag(predSet, 0, MO_FLAG_PUSH);
    }
    I->eraseFromParent();
    break;
  }
  return 2;
}

bool
R600InstrInfo::isPredicated(const MachineInstr *MI) const {
  int idx = MI->findFirstPredOperandIdx();
  if (idx < 0)
    return false;

  unsigned Reg = MI->getOperand(idx).getReg();
  switch (Reg) {
  default: return false;
  case AMDGPU::PRED_SEL_ONE:
  case AMDGPU::PRED_SEL_ZERO:
  case AMDGPU::PREDICATE_BIT:
    return true;
  }
}

bool
R600InstrInfo::isPredicable(MachineInstr *MI) const {
  // XXX: KILL* instructions can be predicated, but they must be the last
  // instruction in a clause, so this means any instructions after them cannot
  // be predicated.  Until we have proper support for instruction clauses in the
  // backend, we will mark KILL* instructions as unpredicable.

  if (MI->getOpcode() == AMDGPU::KILLGT) {
    return false;
  } else if (isVector(*MI)) {
    return false;
  } else {
    return AMDGPUInstrInfo::isPredicable(MI);
  }
}


bool
R600InstrInfo::isProfitableToIfCvt(MachineBasicBlock &MBB,
                                   unsigned NumCyles,
                                   unsigned ExtraPredCycles,
                                   const BranchProbability &Probability) const{
  return true;
}

bool
R600InstrInfo::isProfitableToIfCvt(MachineBasicBlock &TMBB,
                                   unsigned NumTCycles,
                                   unsigned ExtraTCycles,
                                   MachineBasicBlock &FMBB,
                                   unsigned NumFCycles,
                                   unsigned ExtraFCycles,
                                   const BranchProbability &Probability) const {
  return true;
}

bool
R600InstrInfo::isProfitableToDupForIfCvt(MachineBasicBlock &MBB,
                                         unsigned NumCyles,
                                         const BranchProbability &Probability)
                                         const {
  return true;
}

bool
R600InstrInfo::isProfitableToUnpredicate(MachineBasicBlock &TMBB,
                                         MachineBasicBlock &FMBB) const {
  return false;
}


bool
R600InstrInfo::ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  MachineOperand &MO = Cond[1];
  switch (MO.getImm()) {
  case OPCODE_IS_ZERO_INT:
    MO.setImm(OPCODE_IS_NOT_ZERO_INT);
    break;
  case OPCODE_IS_NOT_ZERO_INT:
    MO.setImm(OPCODE_IS_ZERO_INT);
    break;
  case OPCODE_IS_ZERO:
    MO.setImm(OPCODE_IS_NOT_ZERO);
    break;
  case OPCODE_IS_NOT_ZERO:
    MO.setImm(OPCODE_IS_ZERO);
    break;
  default:
    return true;
  }

  MachineOperand &MO2 = Cond[2];
  switch (MO2.getReg()) {
  case AMDGPU::PRED_SEL_ZERO:
    MO2.setReg(AMDGPU::PRED_SEL_ONE);
    break;
  case AMDGPU::PRED_SEL_ONE:
    MO2.setReg(AMDGPU::PRED_SEL_ZERO);
    break;
  default:
    return true;
  }
  return false;
}

bool
R600InstrInfo::DefinesPredicate(MachineInstr *MI,
                                std::vector<MachineOperand> &Pred) const {
  return isPredicateSetter(MI->getOpcode());
}


bool
R600InstrInfo::SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                       const SmallVectorImpl<MachineOperand> &Pred2) const {
  return false;
}


bool
R600InstrInfo::PredicateInstruction(MachineInstr *MI,
                      const SmallVectorImpl<MachineOperand> &Pred) const {
  int PIdx = MI->findFirstPredOperandIdx();

  if (PIdx != -1) {
    MachineOperand &PMO = MI->getOperand(PIdx);
    PMO.setReg(Pred[2].getReg());
    MachineInstrBuilder MIB(*MI->getParent()->getParent(), MI);
    MIB.addReg(AMDGPU::PREDICATE_BIT, RegState::Implicit);
    return true;
  }

  return false;
}

unsigned int R600InstrInfo::getInstrLatency(const InstrItineraryData *ItinData,
                                            const MachineInstr *MI,
                                            unsigned *PredCost) const {
  if (PredCost)
    *PredCost = 2;
  return 2;
}

int R600InstrInfo::getIndirectIndexBegin(const MachineFunction &MF) const {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  int Offset = 0;

  if (MFI->getNumObjects() == 0) {
    return -1;
  }

  if (MRI.livein_empty()) {
    return 0;
  }

  for (MachineRegisterInfo::livein_iterator LI = MRI.livein_begin(),
                                            LE = MRI.livein_end();
                                            LI != LE; ++LI) {
    Offset = std::max(Offset,
                      GET_REG_INDEX(RI.getEncodingValue(LI->first)));
  }

  return Offset + 1;
}

int R600InstrInfo::getIndirectIndexEnd(const MachineFunction &MF) const {
  int Offset = 0;
  const MachineFrameInfo *MFI = MF.getFrameInfo();

  // Variable sized objects are not supported
  assert(!MFI->hasVarSizedObjects());

  if (MFI->getNumObjects() == 0) {
    return -1;
  }

  Offset = TM.getFrameLowering()->getFrameIndexOffset(MF, -1);

  return getIndirectIndexBegin(MF) + Offset;
}

std::vector<unsigned> R600InstrInfo::getIndirectReservedRegs(
                                             const MachineFunction &MF) const {
  const AMDGPUFrameLowering *TFL =
                 static_cast<const AMDGPUFrameLowering*>(TM.getFrameLowering());
  std::vector<unsigned> Regs;

  unsigned StackWidth = TFL->getStackWidth(MF);
  int End = getIndirectIndexEnd(MF);

  if (End == -1) {
    return Regs;
  }

  for (int Index = getIndirectIndexBegin(MF); Index <= End; ++Index) {
    unsigned SuperReg = AMDGPU::R600_Reg128RegClass.getRegister(Index);
    Regs.push_back(SuperReg);
    for (unsigned Chan = 0; Chan < StackWidth; ++Chan) {
      unsigned Reg = AMDGPU::R600_TReg32RegClass.getRegister((4 * Index) + Chan);
      Regs.push_back(Reg);
    }
  }
  return Regs;
}

unsigned R600InstrInfo::calculateIndirectAddress(unsigned RegIndex,
                                                 unsigned Channel) const {
  // XXX: Remove when we support a stack width > 2
  assert(Channel == 0);
  return RegIndex;
}

const TargetRegisterClass * R600InstrInfo::getIndirectAddrStoreRegClass(
                                                     unsigned SourceReg) const {
  return &AMDGPU::R600_TReg32RegClass;
}

const TargetRegisterClass *R600InstrInfo::getIndirectAddrLoadRegClass() const {
  return &AMDGPU::TRegMemRegClass;
}

MachineInstrBuilder R600InstrInfo::buildIndirectWrite(MachineBasicBlock *MBB,
                                       MachineBasicBlock::iterator I,
                                       unsigned ValueReg, unsigned Address,
                                       unsigned OffsetReg) const {
  unsigned AddrReg = AMDGPU::R600_AddrRegClass.getRegister(Address);
  MachineInstr *MOVA = buildDefaultInstruction(*MBB, I, AMDGPU::MOVA_INT_eg,
                                               AMDGPU::AR_X, OffsetReg);
  setImmOperand(MOVA, R600Operands::WRITE, 0);

  MachineInstrBuilder Mov = buildDefaultInstruction(*MBB, I, AMDGPU::MOV,
                                      AddrReg, ValueReg)
                                      .addReg(AMDGPU::AR_X, RegState::Implicit);
  setImmOperand(Mov, R600Operands::DST_REL, 1);
  return Mov;
}

MachineInstrBuilder R600InstrInfo::buildIndirectRead(MachineBasicBlock *MBB,
                                       MachineBasicBlock::iterator I,
                                       unsigned ValueReg, unsigned Address,
                                       unsigned OffsetReg) const {
  unsigned AddrReg = AMDGPU::R600_AddrRegClass.getRegister(Address);
  MachineInstr *MOVA = buildDefaultInstruction(*MBB, I, AMDGPU::MOVA_INT_eg,
                                                       AMDGPU::AR_X,
                                                       OffsetReg);
  setImmOperand(MOVA, R600Operands::WRITE, 0);
  MachineInstrBuilder Mov = buildDefaultInstruction(*MBB, I, AMDGPU::MOV,
                                      ValueReg,
                                      AddrReg)
                                      .addReg(AMDGPU::AR_X, RegState::Implicit);
  setImmOperand(Mov, R600Operands::SRC0_REL, 1);

  return Mov;
}

const TargetRegisterClass *R600InstrInfo::getSuperIndirectRegClass() const {
  return &AMDGPU::IndirectRegRegClass;
}


MachineInstrBuilder R600InstrInfo::buildDefaultInstruction(MachineBasicBlock &MBB,
                                                  MachineBasicBlock::iterator I,
                                                  unsigned Opcode,
                                                  unsigned DstReg,
                                                  unsigned Src0Reg,
                                                  unsigned Src1Reg) const {
  MachineInstrBuilder MIB = BuildMI(MBB, I, MBB.findDebugLoc(I), get(Opcode),
    DstReg);           // $dst

  if (Src1Reg) {
    MIB.addImm(0)     // $update_exec_mask
       .addImm(0);    // $update_predicate
  }
  MIB.addImm(1)        // $write
     .addImm(0)        // $omod
     .addImm(0)        // $dst_rel
     .addImm(0)        // $dst_clamp
     .addReg(Src0Reg)  // $src0
     .addImm(0)        // $src0_neg
     .addImm(0)        // $src0_rel
     .addImm(0)        // $src0_abs
     .addImm(-1);       // $src0_sel

  if (Src1Reg) {
    MIB.addReg(Src1Reg) // $src1
       .addImm(0)       // $src1_neg
       .addImm(0)       // $src1_rel
       .addImm(0)       // $src1_abs
       .addImm(-1);      // $src1_sel
  }

  //XXX: The r600g finalizer expects this to be 1, once we've moved the
  //scheduling to the backend, we can change the default to 0.
  MIB.addImm(1)        // $last
      .addReg(AMDGPU::PRED_SEL_OFF) // $pred_sel
      .addImm(0);        // $literal

  return MIB;
}

MachineInstr *R600InstrInfo::buildMovImm(MachineBasicBlock &BB,
                                         MachineBasicBlock::iterator I,
                                         unsigned DstReg,
                                         uint64_t Imm) const {
  MachineInstr *MovImm = buildDefaultInstruction(BB, I, AMDGPU::MOV, DstReg,
                                                  AMDGPU::ALU_LITERAL_X);
  setImmOperand(MovImm, R600Operands::IMM, Imm);
  return MovImm;
}

int R600InstrInfo::getOperandIdx(const MachineInstr &MI,
                                 R600Operands::Ops Op) const {
  return getOperandIdx(MI.getOpcode(), Op);
}

int R600InstrInfo::getOperandIdx(unsigned Opcode,
                                 R600Operands::Ops Op) const {
  unsigned TargetFlags = get(Opcode).TSFlags;
  unsigned OpTableIdx;

  if (!HAS_NATIVE_OPERANDS(TargetFlags)) {
    switch (Op) {
    case R600Operands::DST: return 0;
    case R600Operands::SRC0: return 1;
    case R600Operands::SRC1: return 2;
    case R600Operands::SRC2: return 3;
    default:
      assert(!"Unknown operand type for instruction");
      return -1;
    }
  }

  if (TargetFlags & R600_InstFlag::OP1) {
    OpTableIdx = 0;
  } else if (TargetFlags & R600_InstFlag::OP2) {
    OpTableIdx = 1;
  } else {
    assert((TargetFlags & R600_InstFlag::OP3) && "OP1, OP2, or OP3 not defined "
                                                 "for this instruction");
    OpTableIdx = 2;
  }

  return R600Operands::ALUOpTable[OpTableIdx][Op];
}

void R600InstrInfo::setImmOperand(MachineInstr *MI, R600Operands::Ops Op,
                                  int64_t Imm) const {
  int Idx = getOperandIdx(*MI, Op);
  assert(Idx != -1 && "Operand not supported for this instruction.");
  assert(MI->getOperand(Idx).isImm());
  MI->getOperand(Idx).setImm(Imm);
}

//===----------------------------------------------------------------------===//
// Instruction flag getters/setters
//===----------------------------------------------------------------------===//

bool R600InstrInfo::hasFlagOperand(const MachineInstr &MI) const {
  return GET_FLAG_OPERAND_IDX(get(MI.getOpcode()).TSFlags) != 0;
}

MachineOperand &R600InstrInfo::getFlagOp(MachineInstr *MI, unsigned SrcIdx,
                                         unsigned Flag) const {
  unsigned TargetFlags = get(MI->getOpcode()).TSFlags;
  int FlagIndex = 0;
  if (Flag != 0) {
    // If we pass something other than the default value of Flag to this
    // function, it means we are want to set a flag on an instruction
    // that uses native encoding.
    assert(HAS_NATIVE_OPERANDS(TargetFlags));
    bool IsOP3 = (TargetFlags & R600_InstFlag::OP3) == R600_InstFlag::OP3;
    switch (Flag) {
    case MO_FLAG_CLAMP:
      FlagIndex = getOperandIdx(*MI, R600Operands::CLAMP);
      break;
    case MO_FLAG_MASK:
      FlagIndex = getOperandIdx(*MI, R600Operands::WRITE);
      break;
    case MO_FLAG_NOT_LAST:
    case MO_FLAG_LAST:
      FlagIndex = getOperandIdx(*MI, R600Operands::LAST);
      break;
    case MO_FLAG_NEG:
      switch (SrcIdx) {
      case 0: FlagIndex = getOperandIdx(*MI, R600Operands::SRC0_NEG); break;
      case 1: FlagIndex = getOperandIdx(*MI, R600Operands::SRC1_NEG); break;
      case 2: FlagIndex = getOperandIdx(*MI, R600Operands::SRC2_NEG); break;
      }
      break;

    case MO_FLAG_ABS:
      assert(!IsOP3 && "Cannot set absolute value modifier for OP3 "
                       "instructions.");
      (void)IsOP3;
      switch (SrcIdx) {
      case 0: FlagIndex = getOperandIdx(*MI, R600Operands::SRC0_ABS); break;
      case 1: FlagIndex = getOperandIdx(*MI, R600Operands::SRC1_ABS); break;
      }
      break;

    default:
      FlagIndex = -1;
      break;
    }
    assert(FlagIndex != -1 && "Flag not supported for this instruction");
  } else {
      FlagIndex = GET_FLAG_OPERAND_IDX(TargetFlags);
      assert(FlagIndex != 0 &&
         "Instruction flags not supported for this instruction");
  }

  MachineOperand &FlagOp = MI->getOperand(FlagIndex);
  assert(FlagOp.isImm());
  return FlagOp;
}

void R600InstrInfo::addFlag(MachineInstr *MI, unsigned Operand,
                            unsigned Flag) const {
  unsigned TargetFlags = get(MI->getOpcode()).TSFlags;
  if (Flag == 0) {
    return;
  }
  if (HAS_NATIVE_OPERANDS(TargetFlags)) {
    MachineOperand &FlagOp = getFlagOp(MI, Operand, Flag);
    if (Flag == MO_FLAG_NOT_LAST) {
      clearFlag(MI, Operand, MO_FLAG_LAST);
    } else if (Flag == MO_FLAG_MASK) {
      clearFlag(MI, Operand, Flag);
    } else {
      FlagOp.setImm(1);
    }
  } else {
      MachineOperand &FlagOp = getFlagOp(MI, Operand);
      FlagOp.setImm(FlagOp.getImm() | (Flag << (NUM_MO_FLAGS * Operand)));
  }
}

void R600InstrInfo::clearFlag(MachineInstr *MI, unsigned Operand,
                              unsigned Flag) const {
  unsigned TargetFlags = get(MI->getOpcode()).TSFlags;
  if (HAS_NATIVE_OPERANDS(TargetFlags)) {
    MachineOperand &FlagOp = getFlagOp(MI, Operand, Flag);
    FlagOp.setImm(0);
  } else {
    MachineOperand &FlagOp = getFlagOp(MI);
    unsigned InstFlags = FlagOp.getImm();
    InstFlags &= ~(Flag << (NUM_MO_FLAGS * Operand));
    FlagOp.setImm(InstFlags);
  }
}
