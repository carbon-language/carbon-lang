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
#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"
#include "R600Defines.h"
#include "R600MachineFunctionInfo.h"
#include "R600RegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

#define GET_INSTRINFO_CTOR
#include "AMDGPUGenDFAPacketizer.inc"

using namespace llvm;

R600InstrInfo::R600InstrInfo(AMDGPUTargetMachine &tm)
  : AMDGPUInstrInfo(tm),
    RI(tm),
    ST(tm.getSubtarget<AMDGPUSubtarget>())
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
    NewMI->getOperand(getOperandIdx(*NewMI, AMDGPU::OpName::src0))
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

  return (TargetFlags & R600_InstFlag::ALU_INST);
}

bool R600InstrInfo::hasInstrModifiers(unsigned Opcode) const {
  unsigned TargetFlags = get(Opcode).TSFlags;

  return ((TargetFlags & R600_InstFlag::OP1) |
          (TargetFlags & R600_InstFlag::OP2) |
          (TargetFlags & R600_InstFlag::OP3));
}

bool R600InstrInfo::isLDSInstr(unsigned Opcode) const {
  unsigned TargetFlags = get(Opcode).TSFlags;

  return ((TargetFlags & R600_InstFlag::LDS_1A) |
          (TargetFlags & R600_InstFlag::LDS_1A1D));
}

bool R600InstrInfo::isTransOnly(unsigned Opcode) const {
  return (get(Opcode).TSFlags & R600_InstFlag::TRANS_ONLY);
}

bool R600InstrInfo::isTransOnly(const MachineInstr *MI) const {
  return isTransOnly(MI->getOpcode());
}

bool R600InstrInfo::usesVertexCache(unsigned Opcode) const {
  return ST.hasVertexCache() && IS_VTX(get(Opcode));
}

bool R600InstrInfo::usesVertexCache(const MachineInstr *MI) const {
  const R600MachineFunctionInfo *MFI = MI->getParent()->getParent()->getInfo<R600MachineFunctionInfo>();
  return MFI->ShaderType != ShaderType::COMPUTE && usesVertexCache(MI->getOpcode());
}

bool R600InstrInfo::usesTextureCache(unsigned Opcode) const {
  return (!ST.hasVertexCache() && IS_VTX(get(Opcode))) || IS_TEX(get(Opcode));
}

bool R600InstrInfo::usesTextureCache(const MachineInstr *MI) const {
  const R600MachineFunctionInfo *MFI = MI->getParent()->getParent()->getInfo<R600MachineFunctionInfo>();
  return (MFI->ShaderType == ShaderType::COMPUTE && usesVertexCache(MI->getOpcode())) ||
         usesTextureCache(MI->getOpcode());
}

bool R600InstrInfo::mustBeLastInClause(unsigned Opcode) const {
  switch (Opcode) {
  case AMDGPU::KILLGT:
  case AMDGPU::GROUP_BARRIER:
    return true;
  default:
    return false;
  }
}

SmallVector<std::pair<MachineOperand *, int64_t>, 3>
R600InstrInfo::getSrcs(MachineInstr *MI) const {
  SmallVector<std::pair<MachineOperand *, int64_t>, 3> Result;

  if (MI->getOpcode() == AMDGPU::DOT_4) {
    static const unsigned OpTable[8][2] = {
      {AMDGPU::OpName::src0_X, AMDGPU::OpName::src0_sel_X},
      {AMDGPU::OpName::src0_Y, AMDGPU::OpName::src0_sel_Y},
      {AMDGPU::OpName::src0_Z, AMDGPU::OpName::src0_sel_Z},
      {AMDGPU::OpName::src0_W, AMDGPU::OpName::src0_sel_W},
      {AMDGPU::OpName::src1_X, AMDGPU::OpName::src1_sel_X},
      {AMDGPU::OpName::src1_Y, AMDGPU::OpName::src1_sel_Y},
      {AMDGPU::OpName::src1_Z, AMDGPU::OpName::src1_sel_Z},
      {AMDGPU::OpName::src1_W, AMDGPU::OpName::src1_sel_W},
    };

    for (unsigned j = 0; j < 8; j++) {
      MachineOperand &MO = MI->getOperand(getOperandIdx(MI->getOpcode(),
                                                        OpTable[j][0]));
      unsigned Reg = MO.getReg();
      if (Reg == AMDGPU::ALU_CONST) {
        unsigned Sel = MI->getOperand(getOperandIdx(MI->getOpcode(),
                                                    OpTable[j][1])).getImm();
        Result.push_back(std::pair<MachineOperand *, int64_t>(&MO, Sel));
        continue;
      }
      
    }
    return Result;
  }

  static const unsigned OpTable[3][2] = {
    {AMDGPU::OpName::src0, AMDGPU::OpName::src0_sel},
    {AMDGPU::OpName::src1, AMDGPU::OpName::src1_sel},
    {AMDGPU::OpName::src2, AMDGPU::OpName::src2_sel},
  };

  for (unsigned j = 0; j < 3; j++) {
    int SrcIdx = getOperandIdx(MI->getOpcode(), OpTable[j][0]);
    if (SrcIdx < 0)
      break;
    MachineOperand &MO = MI->getOperand(SrcIdx);
    unsigned Reg = MI->getOperand(SrcIdx).getReg();
    if (Reg == AMDGPU::ALU_CONST) {
      unsigned Sel = MI->getOperand(
          getOperandIdx(MI->getOpcode(), OpTable[j][1])).getImm();
      Result.push_back(std::pair<MachineOperand *, int64_t>(&MO, Sel));
      continue;
    }
    if (Reg == AMDGPU::ALU_LITERAL_X) {
      unsigned Imm = MI->getOperand(
          getOperandIdx(MI->getOpcode(), AMDGPU::OpName::literal)).getImm();
      Result.push_back(std::pair<MachineOperand *, int64_t>(&MO, Imm));
      continue;
    }
    Result.push_back(std::pair<MachineOperand *, int64_t>(&MO, 0));
  }
  return Result;
}

std::vector<std::pair<int, unsigned> >
R600InstrInfo::ExtractSrcs(MachineInstr *MI,
                           const DenseMap<unsigned, unsigned> &PV)
    const {
  const SmallVector<std::pair<MachineOperand *, int64_t>, 3> Srcs = getSrcs(MI);
  const std::pair<int, unsigned> DummyPair(-1, 0);
  std::vector<std::pair<int, unsigned> > Result;
  unsigned i = 0;
  for (unsigned n = Srcs.size(); i < n; ++i) {
    unsigned Reg = Srcs[i].first->getReg();
    unsigned Index = RI.getEncodingValue(Reg) & 0xff;
    unsigned Chan = RI.getHWRegChan(Reg);
    if (Reg == AMDGPU::OQAP) {
      Result.push_back(std::pair<int, unsigned>(Index, 0));
    }
    if (Index > 127) {
      Result.push_back(DummyPair);
      continue;
    }
    if (PV.find(Reg) != PV.end()) {
      Result.push_back(DummyPair);
      continue;
    }
    Result.push_back(std::pair<int, unsigned>(Index, Chan));
  }
  for (; i < 3; ++i)
    Result.push_back(DummyPair);
  return Result;
}

static std::vector<std::pair<int, unsigned> >
Swizzle(std::vector<std::pair<int, unsigned> > Src,
        R600InstrInfo::BankSwizzle Swz) {
  switch (Swz) {
  case R600InstrInfo::ALU_VEC_012_SCL_210:
    break;
  case R600InstrInfo::ALU_VEC_021_SCL_122:
    std::swap(Src[1], Src[2]);
    break;
  case R600InstrInfo::ALU_VEC_102_SCL_221:
    std::swap(Src[0], Src[1]);
    break;
  case R600InstrInfo::ALU_VEC_120_SCL_212:
    std::swap(Src[0], Src[1]);
    std::swap(Src[0], Src[2]);
    break;
  case R600InstrInfo::ALU_VEC_201:
    std::swap(Src[0], Src[2]);
    std::swap(Src[0], Src[1]);
    break;
  case R600InstrInfo::ALU_VEC_210:
    std::swap(Src[0], Src[2]);
    break;
  }
  return Src;
}

bool
R600InstrInfo::isLegal(
             const std::vector<std::vector<std::pair<int, unsigned> > > &IGSrcs,
             const std::vector<R600InstrInfo::BankSwizzle> &Swz,
             unsigned CheckedSize) const {
  int Vector[4][3];
  memset(Vector, -1, sizeof(Vector));
  for (unsigned i = 0; i < CheckedSize; i++) {
    const std::vector<std::pair<int, unsigned> > &Srcs =
        Swizzle(IGSrcs[i], Swz[i]);
    for (unsigned j = 0; j < 3; j++) {
      const std::pair<int, unsigned> &Src = Srcs[j];
      if (Src.first < 0)
        continue;
      if (Src.first == GET_REG_INDEX(RI.getEncodingValue(AMDGPU::OQAP))) {
        if (Swz[i] != R600InstrInfo::ALU_VEC_012 &&
            Swz[i] != R600InstrInfo::ALU_VEC_021) {
            // The value from output queue A (denoted by register OQAP) can
            // only be fetched during the first cycle.
            return false;
        }
        // OQAP does not count towards the normal read port restrictions
        continue;
      }
      if (Vector[Src.second][j] < 0)
        Vector[Src.second][j] = Src.first;
      if (Vector[Src.second][j] != Src.first)
        return false;
    }
  }
  return true;
}

bool
R600InstrInfo::recursiveFitsFPLimitation(
             const std::vector<std::vector<std::pair<int, unsigned> > > &IGSrcs,
             std::vector<R600InstrInfo::BankSwizzle> &SwzCandidate,
             unsigned Depth) const {
  if (!isLegal(IGSrcs, SwzCandidate, Depth))
    return false;
  if (IGSrcs.size() == Depth)
    return true;
  unsigned i = SwzCandidate[Depth];
  for (; i < 6; i++) {
    SwzCandidate[Depth] = (R600InstrInfo::BankSwizzle) i;
    if (recursiveFitsFPLimitation(IGSrcs, SwzCandidate, Depth + 1))
      return true;
  }
  SwzCandidate[Depth] = R600InstrInfo::ALU_VEC_012;
  return false;
}

bool
R600InstrInfo::fitsReadPortLimitations(const std::vector<MachineInstr *> &IG,
                                      const DenseMap<unsigned, unsigned> &PV,
                                      std::vector<BankSwizzle> &ValidSwizzle)
    const {
  //Todo : support shared src0 - src1 operand

  std::vector<std::vector<std::pair<int, unsigned> > > IGSrcs;
  ValidSwizzle.clear();
  for (unsigned i = 0, e = IG.size(); i < e; ++i) {
    IGSrcs.push_back(ExtractSrcs(IG[i], PV));
    unsigned Op = getOperandIdx(IG[i]->getOpcode(),
        AMDGPU::OpName::bank_swizzle);
    ValidSwizzle.push_back( (R600InstrInfo::BankSwizzle)
        IG[i]->getOperand(Op).getImm());
  }
  bool Result = recursiveFitsFPLimitation(IGSrcs, ValidSwizzle);
  if (!Result)
    return false;
  return true;
}


bool
R600InstrInfo::fitsConstReadLimitations(const std::vector<unsigned> &Consts)
    const {
  assert (Consts.size() <= 12 && "Too many operands in instructions group");
  unsigned Pair1 = 0, Pair2 = 0;
  for (unsigned i = 0, n = Consts.size(); i < n; ++i) {
    unsigned ReadConstHalf = Consts[i] & 2;
    unsigned ReadConstIndex = Consts[i] & (~3);
    unsigned ReadHalfConst = ReadConstIndex | ReadConstHalf;
    if (!Pair1) {
      Pair1 = ReadHalfConst;
      continue;
    }
    if (Pair1 == ReadHalfConst)
      continue;
    if (!Pair2) {
      Pair2 = ReadHalfConst;
      continue;
    }
    if (Pair2 != ReadHalfConst)
      return false;
  }
  return true;
}

bool
R600InstrInfo::canBundle(const std::vector<MachineInstr *> &MIs) const {
  std::vector<unsigned> Consts;
  for (unsigned i = 0, n = MIs.size(); i < n; i++) {
    MachineInstr *MI = MIs[i];
    if (!isALUInstr(MI->getOpcode()))
      continue;

    const SmallVector<std::pair<MachineOperand *, int64_t>, 3> &Srcs =
        getSrcs(MI);

    for (unsigned j = 0, e = Srcs.size(); j < e; j++) {
      std::pair<MachineOperand *, unsigned> Src = Srcs[j];
      if (Src.first->getReg() == AMDGPU::ALU_CONST)
        Consts.push_back(Src.second);
      if (AMDGPU::R600_KC0RegClass.contains(Src.first->getReg()) ||
          AMDGPU::R600_KC1RegClass.contains(Src.first->getReg())) {
        unsigned Index = RI.getEncodingValue(Src.first->getReg()) & 0xff;
        unsigned Chan = RI.getHWRegChan(Src.first->getReg());
        Consts.push_back((Index << 2) | Chan);
      }
    }
  }
  return fitsConstReadLimitations(Consts);
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

static
bool isJump(unsigned Opcode) {
  return Opcode == AMDGPU::JUMP || Opcode == AMDGPU::JUMP_COND;
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
  if (!isJump(static_cast<MachineInstr *>(I)->getOpcode())) {
    return false;
  }

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;

  // If there is only one terminator instruction, process it.
  unsigned LastOpc = LastInst->getOpcode();
  if (I == MBB.begin() ||
          !isJump(static_cast<MachineInstr *>(--I)->getOpcode())) {
    if (LastOpc == AMDGPU::JUMP) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    } else if (LastOpc == AMDGPU::JUMP_COND) {
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
    return true;  // Can't handle indirect branch.
  }

  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = I;
  unsigned SecondLastOpc = SecondLastInst->getOpcode();

  // If the block ends with a B and a Bcc, handle it.
  if (SecondLastOpc == AMDGPU::JUMP_COND && LastOpc == AMDGPU::JUMP) {
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
      BuildMI(&MBB, DL, get(AMDGPU::JUMP)).addMBB(TBB);
      return 1;
    } else {
      MachineInstr *PredSet = findFirstPredicateSetterFrom(MBB, MBB.end());
      assert(PredSet && "No previous predicate !");
      addFlag(PredSet, 0, MO_FLAG_PUSH);
      PredSet->getOperand(2).setImm(Cond[1].getImm());

      BuildMI(&MBB, DL, get(AMDGPU::JUMP_COND))
             .addMBB(TBB)
             .addReg(AMDGPU::PREDICATE_BIT, RegState::Kill);
      return 1;
    }
  } else {
    MachineInstr *PredSet = findFirstPredicateSetterFrom(MBB, MBB.end());
    assert(PredSet && "No previous predicate !");
    addFlag(PredSet, 0, MO_FLAG_PUSH);
    PredSet->getOperand(2).setImm(Cond[1].getImm());
    BuildMI(&MBB, DL, get(AMDGPU::JUMP_COND))
            .addMBB(TBB)
            .addReg(AMDGPU::PREDICATE_BIT, RegState::Kill);
    BuildMI(&MBB, DL, get(AMDGPU::JUMP)).addMBB(FBB);
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
  case AMDGPU::JUMP_COND: {
    MachineInstr *predSet = findFirstPredicateSetterFrom(MBB, I);
    clearFlag(predSet, 0, MO_FLAG_PUSH);
    I->eraseFromParent();
    break;
  }
  case AMDGPU::JUMP:
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
  case AMDGPU::JUMP_COND: {
    MachineInstr *predSet = findFirstPredicateSetterFrom(MBB, I);
    clearFlag(predSet, 0, MO_FLAG_PUSH);
    I->eraseFromParent();
    break;
  }
  case AMDGPU::JUMP:
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
  setImmOperand(MOVA, AMDGPU::OpName::write, 0);

  MachineInstrBuilder Mov = buildDefaultInstruction(*MBB, I, AMDGPU::MOV,
                                      AddrReg, ValueReg)
                                      .addReg(AMDGPU::AR_X,
                                           RegState::Implicit | RegState::Kill);
  setImmOperand(Mov, AMDGPU::OpName::dst_rel, 1);
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
  setImmOperand(MOVA, AMDGPU::OpName::write, 0);
  MachineInstrBuilder Mov = buildDefaultInstruction(*MBB, I, AMDGPU::MOV,
                                      ValueReg,
                                      AddrReg)
                                      .addReg(AMDGPU::AR_X,
                                           RegState::Implicit | RegState::Kill);
  setImmOperand(Mov, AMDGPU::OpName::src0_rel, 1);

  return Mov;
}

const TargetRegisterClass *R600InstrInfo::getSuperIndirectRegClass() const {
  return &AMDGPU::IndirectRegRegClass;
}

unsigned R600InstrInfo::getMaxAlusPerClause() const {
  return 115;
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
      .addImm(0)         // $literal
      .addImm(0);        // $bank_swizzle

  return MIB;
}

#define OPERAND_CASE(Label) \
  case Label: { \
    static const unsigned Ops[] = \
    { \
      Label##_X, \
      Label##_Y, \
      Label##_Z, \
      Label##_W \
    }; \
    return Ops[Slot]; \
  }

static unsigned getSlotedOps(unsigned  Op, unsigned Slot) {
  switch (Op) {
  OPERAND_CASE(AMDGPU::OpName::update_exec_mask)
  OPERAND_CASE(AMDGPU::OpName::update_pred)
  OPERAND_CASE(AMDGPU::OpName::write)
  OPERAND_CASE(AMDGPU::OpName::omod)
  OPERAND_CASE(AMDGPU::OpName::dst_rel)
  OPERAND_CASE(AMDGPU::OpName::clamp)
  OPERAND_CASE(AMDGPU::OpName::src0)
  OPERAND_CASE(AMDGPU::OpName::src0_neg)
  OPERAND_CASE(AMDGPU::OpName::src0_rel)
  OPERAND_CASE(AMDGPU::OpName::src0_abs)
  OPERAND_CASE(AMDGPU::OpName::src0_sel)
  OPERAND_CASE(AMDGPU::OpName::src1)
  OPERAND_CASE(AMDGPU::OpName::src1_neg)
  OPERAND_CASE(AMDGPU::OpName::src1_rel)
  OPERAND_CASE(AMDGPU::OpName::src1_abs)
  OPERAND_CASE(AMDGPU::OpName::src1_sel)
  OPERAND_CASE(AMDGPU::OpName::pred_sel)
  default:
    llvm_unreachable("Wrong Operand");
  }
}

#undef OPERAND_CASE

MachineInstr *R600InstrInfo::buildSlotOfVectorInstruction(
    MachineBasicBlock &MBB, MachineInstr *MI, unsigned Slot, unsigned DstReg)
    const {
  assert (MI->getOpcode() == AMDGPU::DOT_4 && "Not Implemented");
  unsigned Opcode;
  const AMDGPUSubtarget &ST = TM.getSubtarget<AMDGPUSubtarget>();
  if (ST.getGeneration() <= AMDGPUSubtarget::R700)
    Opcode = AMDGPU::DOT4_r600;
  else
    Opcode = AMDGPU::DOT4_eg;
  MachineBasicBlock::iterator I = MI;
  MachineOperand &Src0 = MI->getOperand(
      getOperandIdx(MI->getOpcode(), getSlotedOps(AMDGPU::OpName::src0, Slot)));
  MachineOperand &Src1 = MI->getOperand(
      getOperandIdx(MI->getOpcode(), getSlotedOps(AMDGPU::OpName::src1, Slot)));
  MachineInstr *MIB = buildDefaultInstruction(
      MBB, I, Opcode, DstReg, Src0.getReg(), Src1.getReg());
  static const unsigned  Operands[14] = {
    AMDGPU::OpName::update_exec_mask,
    AMDGPU::OpName::update_pred,
    AMDGPU::OpName::write,
    AMDGPU::OpName::omod,
    AMDGPU::OpName::dst_rel,
    AMDGPU::OpName::clamp,
    AMDGPU::OpName::src0_neg,
    AMDGPU::OpName::src0_rel,
    AMDGPU::OpName::src0_abs,
    AMDGPU::OpName::src0_sel,
    AMDGPU::OpName::src1_neg,
    AMDGPU::OpName::src1_rel,
    AMDGPU::OpName::src1_abs,
    AMDGPU::OpName::src1_sel,
  };

  for (unsigned i = 0; i < 14; i++) {
    MachineOperand &MO = MI->getOperand(
        getOperandIdx(MI->getOpcode(), getSlotedOps(Operands[i], Slot)));
    assert (MO.isImm());
    setImmOperand(MIB, Operands[i], MO.getImm());
  }
  MIB->getOperand(20).setImm(0);
  return MIB;
}

MachineInstr *R600InstrInfo::buildMovImm(MachineBasicBlock &BB,
                                         MachineBasicBlock::iterator I,
                                         unsigned DstReg,
                                         uint64_t Imm) const {
  MachineInstr *MovImm = buildDefaultInstruction(BB, I, AMDGPU::MOV, DstReg,
                                                  AMDGPU::ALU_LITERAL_X);
  setImmOperand(MovImm, AMDGPU::OpName::literal, Imm);
  return MovImm;
}

int R600InstrInfo::getOperandIdx(const MachineInstr &MI, unsigned Op) const {
  return getOperandIdx(MI.getOpcode(), Op);
}

int R600InstrInfo::getOperandIdx(unsigned Opcode, unsigned Op) const {
  return AMDGPU::getNamedOperandIdx(Opcode, Op);
}

void R600InstrInfo::setImmOperand(MachineInstr *MI, unsigned Op,
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
      FlagIndex = getOperandIdx(*MI, AMDGPU::OpName::clamp);
      break;
    case MO_FLAG_MASK:
      FlagIndex = getOperandIdx(*MI, AMDGPU::OpName::write);
      break;
    case MO_FLAG_NOT_LAST:
    case MO_FLAG_LAST:
      FlagIndex = getOperandIdx(*MI, AMDGPU::OpName::last);
      break;
    case MO_FLAG_NEG:
      switch (SrcIdx) {
      case 0: FlagIndex = getOperandIdx(*MI, AMDGPU::OpName::src0_neg); break;
      case 1: FlagIndex = getOperandIdx(*MI, AMDGPU::OpName::src1_neg); break;
      case 2: FlagIndex = getOperandIdx(*MI, AMDGPU::OpName::src2_neg); break;
      }
      break;

    case MO_FLAG_ABS:
      assert(!IsOP3 && "Cannot set absolute value modifier for OP3 "
                       "instructions.");
      (void)IsOP3;
      switch (SrcIdx) {
      case 0: FlagIndex = getOperandIdx(*MI, AMDGPU::OpName::src0_abs); break;
      case 1: FlagIndex = getOperandIdx(*MI, AMDGPU::OpName::src1_abs); break;
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
