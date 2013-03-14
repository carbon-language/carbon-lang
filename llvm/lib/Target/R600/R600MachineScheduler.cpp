//===-- R600MachineScheduler.cpp - R600 Scheduler Interface -*- C++ -*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief R600 Machine Scheduler interface
// TODO: Scheduling is optimised for VLIW4 arch, modify it to support TRANS slot
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "misched"

#include "R600MachineScheduler.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

using namespace llvm;

void R600SchedStrategy::initialize(ScheduleDAGMI *dag) {

  DAG = dag;
  TII = static_cast<const R600InstrInfo*>(DAG->TII);
  TRI = static_cast<const R600RegisterInfo*>(DAG->TRI);
  MRI = &DAG->MRI;
  Available[IDAlu]->clear();
  Available[IDFetch]->clear();
  Available[IDOther]->clear();
  CurInstKind = IDOther;
  CurEmitted = 0;
  OccupedSlotsMask = 15;
  InstKindLimit[IDAlu] = 120; // 120 minus 8 for security


  const AMDGPUSubtarget &ST = DAG->TM.getSubtarget<AMDGPUSubtarget>();
  if (ST.device()->getGeneration() <= AMDGPUDeviceInfo::HD5XXX) {
    InstKindLimit[IDFetch] = 7; // 8 minus 1 for security
  } else {
    InstKindLimit[IDFetch] = 15; // 16 minus 1 for security
  }
}

void R600SchedStrategy::MoveUnits(ReadyQueue *QSrc, ReadyQueue *QDst)
{
  if (QSrc->empty())
    return;
  for (ReadyQueue::iterator I = QSrc->begin(),
      E = QSrc->end(); I != E; ++I) {
    (*I)->NodeQueueId &= ~QSrc->getID();
    QDst->push(*I);
  }
  QSrc->clear();
}

SUnit* R600SchedStrategy::pickNode(bool &IsTopNode) {
  SUnit *SU = 0;
  IsTopNode = true;
  NextInstKind = IDOther;

  // check if we might want to switch current clause type
  bool AllowSwitchToAlu = (CurInstKind == IDOther) ||
      (CurEmitted > InstKindLimit[CurInstKind]) ||
      (Available[CurInstKind]->empty());
  bool AllowSwitchFromAlu = (CurEmitted > InstKindLimit[CurInstKind]) &&
      (!Available[IDFetch]->empty() || !Available[IDOther]->empty());

  if ((AllowSwitchToAlu && CurInstKind != IDAlu) ||
      (!AllowSwitchFromAlu && CurInstKind == IDAlu)) {
    // try to pick ALU
    SU = pickAlu();
    if (SU) {
      if (CurEmitted >  InstKindLimit[IDAlu])
        CurEmitted = 0;
      NextInstKind = IDAlu;
    }
  }

  if (!SU) {
    // try to pick FETCH
    SU = pickOther(IDFetch);
    if (SU)
      NextInstKind = IDFetch;
  }

  // try to pick other
  if (!SU) {
    SU = pickOther(IDOther);
    if (SU)
      NextInstKind = IDOther;
  }

  DEBUG(
      if (SU) {
        dbgs() << "picked node: ";
        SU->dump(DAG);
      } else {
        dbgs() << "NO NODE ";
        for (int i = 0; i < IDLast; ++i) {
          Available[i]->dump();
          Pending[i]->dump();
        }
        for (unsigned i = 0; i < DAG->SUnits.size(); i++) {
          const SUnit &S = DAG->SUnits[i];
          if (!S.isScheduled)
            S.dump(DAG);
        }
      }
  );

  return SU;
}

void R600SchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {

  DEBUG(dbgs() << "scheduled: ");
  DEBUG(SU->dump(DAG));

  if (NextInstKind != CurInstKind) {
    DEBUG(dbgs() << "Instruction Type Switch\n");
    if (NextInstKind != IDAlu)
      OccupedSlotsMask = 15;
    CurEmitted = 0;
    CurInstKind = NextInstKind;
  }

  if (CurInstKind == IDAlu) {
    switch (getAluKind(SU)) {
    case AluT_XYZW:
      CurEmitted += 4;
      break;
    case AluDiscarded:
      break;
    default: {
      ++CurEmitted;
      for (MachineInstr::mop_iterator It = SU->getInstr()->operands_begin(),
          E = SU->getInstr()->operands_end(); It != E; ++It) {
        MachineOperand &MO = *It;
        if (MO.isReg() && MO.getReg() == AMDGPU::ALU_LITERAL_X)
          ++CurEmitted;
      }
    }
    }
  } else {
    ++CurEmitted;
  }


  DEBUG(dbgs() << CurEmitted << " Instructions Emitted in this clause\n");

  if (CurInstKind != IDFetch) {
    MoveUnits(Pending[IDFetch], Available[IDFetch]);
  }
  MoveUnits(Pending[IDOther], Available[IDOther]);
}

void R600SchedStrategy::releaseTopNode(SUnit *SU) {
  int IK = getInstKind(SU);

  DEBUG(dbgs() << IK << " <= ");
  DEBUG(SU->dump(DAG));

  Pending[IK]->push(SU);
}

void R600SchedStrategy::releaseBottomNode(SUnit *SU) {
}

bool R600SchedStrategy::regBelongsToClass(unsigned Reg,
                                          const TargetRegisterClass *RC) const {
  if (!TargetRegisterInfo::isVirtualRegister(Reg)) {
    return RC->contains(Reg);
  } else {
    return MRI->getRegClass(Reg) == RC;
  }
}

R600SchedStrategy::AluKind R600SchedStrategy::getAluKind(SUnit *SU) const {
  MachineInstr *MI = SU->getInstr();

    switch (MI->getOpcode()) {
    case AMDGPU::INTERP_PAIR_XY:
    case AMDGPU::INTERP_PAIR_ZW:
    case AMDGPU::INTERP_VEC_LOAD:
      return AluT_XYZW;
    case AMDGPU::COPY:
      if (TargetRegisterInfo::isPhysicalRegister(MI->getOperand(1).getReg())) {
        // %vregX = COPY Tn_X is likely to be discarded in favor of an
        // assignement of Tn_X to %vregX, don't considers it in scheduling
        return AluDiscarded;
      }
      else if (MI->getOperand(1).isUndef()) {
        // MI will become a KILL, don't considers it in scheduling
        return AluDiscarded;
      }
    default:
      break;
    }

    // Does the instruction take a whole IG ?
    if(TII->isVector(*MI) ||
        TII->isCubeOp(MI->getOpcode()) ||
        TII->isReductionOp(MI->getOpcode()))
      return AluT_XYZW;

    // Is the result already assigned to a channel ?
    unsigned DestSubReg = MI->getOperand(0).getSubReg();
    switch (DestSubReg) {
    case AMDGPU::sub0:
      return AluT_X;
    case AMDGPU::sub1:
      return AluT_Y;
    case AMDGPU::sub2:
      return AluT_Z;
    case AMDGPU::sub3:
      return AluT_W;
    default:
      break;
    }

    // Is the result already member of a X/Y/Z/W class ?
    unsigned DestReg = MI->getOperand(0).getReg();
    if (regBelongsToClass(DestReg, &AMDGPU::R600_TReg32_XRegClass) ||
        regBelongsToClass(DestReg, &AMDGPU::R600_AddrRegClass))
      return AluT_X;
    if (regBelongsToClass(DestReg, &AMDGPU::R600_TReg32_YRegClass))
      return AluT_Y;
    if (regBelongsToClass(DestReg, &AMDGPU::R600_TReg32_ZRegClass))
      return AluT_Z;
    if (regBelongsToClass(DestReg, &AMDGPU::R600_TReg32_WRegClass))
      return AluT_W;
    if (regBelongsToClass(DestReg, &AMDGPU::R600_Reg128RegClass))
      return AluT_XYZW;

    return AluAny;

}

int R600SchedStrategy::getInstKind(SUnit* SU) {
  int Opcode = SU->getInstr()->getOpcode();

  if (TII->isALUInstr(Opcode)) {
    return IDAlu;
  }

  switch (Opcode) {
  case AMDGPU::COPY:
  case AMDGPU::CONST_COPY:
  case AMDGPU::INTERP_PAIR_XY:
  case AMDGPU::INTERP_PAIR_ZW:
  case AMDGPU::INTERP_VEC_LOAD:
  case AMDGPU::DOT4_eg_pseudo:
  case AMDGPU::DOT4_r600_pseudo:
    return IDAlu;
  case AMDGPU::TEX_VTX_CONSTBUF:
  case AMDGPU::TEX_VTX_TEXBUF:
  case AMDGPU::TEX_LD:
  case AMDGPU::TEX_GET_TEXTURE_RESINFO:
  case AMDGPU::TEX_GET_GRADIENTS_H:
  case AMDGPU::TEX_GET_GRADIENTS_V:
  case AMDGPU::TEX_SET_GRADIENTS_H:
  case AMDGPU::TEX_SET_GRADIENTS_V:
  case AMDGPU::TEX_SAMPLE:
  case AMDGPU::TEX_SAMPLE_C:
  case AMDGPU::TEX_SAMPLE_L:
  case AMDGPU::TEX_SAMPLE_C_L:
  case AMDGPU::TEX_SAMPLE_LB:
  case AMDGPU::TEX_SAMPLE_C_LB:
  case AMDGPU::TEX_SAMPLE_G:
  case AMDGPU::TEX_SAMPLE_C_G:
  case AMDGPU::TXD:
  case AMDGPU::TXD_SHADOW:
    return IDFetch;
  default:
    DEBUG(
        dbgs() << "other inst: ";
        SU->dump(DAG);
    );
    return IDOther;
  }
}

SUnit *R600SchedStrategy::PopInst(std::multiset<SUnit *, CompareSUnit> &Q) {
  if (Q.empty())
    return NULL;
  for (std::set<SUnit *, CompareSUnit>::iterator It = Q.begin(), E = Q.end();
      It != E; ++It) {
    SUnit *SU = *It;
    InstructionsGroupCandidate.push_back(SU->getInstr());
    if (TII->canBundle(InstructionsGroupCandidate)) {
      InstructionsGroupCandidate.pop_back();
      Q.erase(It);
      return SU;
    } else {
      InstructionsGroupCandidate.pop_back();
    }
  }
  return NULL;
}

void R600SchedStrategy::LoadAlu() {
  ReadyQueue *QSrc = Pending[IDAlu];
  for (ReadyQueue::iterator I = QSrc->begin(),
        E = QSrc->end(); I != E; ++I) {
      (*I)->NodeQueueId &= ~QSrc->getID();
      AluKind AK = getAluKind(*I);
      AvailableAlus[AK].insert(*I);
    }
    QSrc->clear();
}

void R600SchedStrategy::PrepareNextSlot() {
  DEBUG(dbgs() << "New Slot\n");
  assert (OccupedSlotsMask && "Slot wasn't filled");
  OccupedSlotsMask = 0;
  InstructionsGroupCandidate.clear();
  LoadAlu();
}

void R600SchedStrategy::AssignSlot(MachineInstr* MI, unsigned Slot) {
  unsigned DestReg = MI->getOperand(0).getReg();
  // PressureRegister crashes if an operand is def and used in the same inst
  // and we try to constraint its regclass
  for (MachineInstr::mop_iterator It = MI->operands_begin(),
      E = MI->operands_end(); It != E; ++It) {
    MachineOperand &MO = *It;
    if (MO.isReg() && !MO.isDef() &&
        MO.getReg() == MI->getOperand(0).getReg())
      return;
  }
  // Constrains the regclass of DestReg to assign it to Slot
  switch (Slot) {
  case 0:
    MRI->constrainRegClass(DestReg, &AMDGPU::R600_TReg32_XRegClass);
    break;
  case 1:
    MRI->constrainRegClass(DestReg, &AMDGPU::R600_TReg32_YRegClass);
    break;
  case 2:
    MRI->constrainRegClass(DestReg, &AMDGPU::R600_TReg32_ZRegClass);
    break;
  case 3:
    MRI->constrainRegClass(DestReg, &AMDGPU::R600_TReg32_WRegClass);
    break;
  }
}

SUnit *R600SchedStrategy::AttemptFillSlot(unsigned Slot) {
  static const AluKind IndexToID[] = {AluT_X, AluT_Y, AluT_Z, AluT_W};
  SUnit *SlotedSU = PopInst(AvailableAlus[IndexToID[Slot]]);
  SUnit *UnslotedSU = PopInst(AvailableAlus[AluAny]);
  if (!UnslotedSU) {
    return SlotedSU;
  } else if (!SlotedSU) {
    AssignSlot(UnslotedSU->getInstr(), Slot);
    return UnslotedSU;
  } else {
    //Determine which one to pick (the lesser one)
    if (CompareSUnit()(SlotedSU, UnslotedSU)) {
      AvailableAlus[AluAny].insert(UnslotedSU);
      return SlotedSU;
    } else {
      AvailableAlus[IndexToID[Slot]].insert(SlotedSU);
      AssignSlot(UnslotedSU->getInstr(), Slot);
      return UnslotedSU;
    }
  }
}

bool R600SchedStrategy::isAvailablesAluEmpty() const {
  return Pending[IDAlu]->empty() && AvailableAlus[AluAny].empty() &&
      AvailableAlus[AluT_XYZW].empty() && AvailableAlus[AluT_X].empty() &&
      AvailableAlus[AluT_Y].empty() && AvailableAlus[AluT_Z].empty() &&
      AvailableAlus[AluT_W].empty() && AvailableAlus[AluDiscarded].empty();
}

SUnit* R600SchedStrategy::pickAlu() {
  while (!isAvailablesAluEmpty()) {
    if (!OccupedSlotsMask) {
      // Flush physical reg copies (RA will discard them)
      if (!AvailableAlus[AluDiscarded].empty()) {
        OccupedSlotsMask = 15;
        return PopInst(AvailableAlus[AluDiscarded]);
      }
      // If there is a T_XYZW alu available, use it
      if (!AvailableAlus[AluT_XYZW].empty()) {
        OccupedSlotsMask = 15;
        return PopInst(AvailableAlus[AluT_XYZW]);
      }
    }
    for (unsigned Chan = 0; Chan < 4; ++Chan) {
      bool isOccupied = OccupedSlotsMask & (1 << Chan);
      if (!isOccupied) {
        SUnit *SU = AttemptFillSlot(Chan);
        if (SU) {
          OccupedSlotsMask |= (1 << Chan);
          InstructionsGroupCandidate.push_back(SU->getInstr());
          return SU;
        }
      }
    }
    PrepareNextSlot();
  }
  return NULL;
}

SUnit* R600SchedStrategy::pickOther(int QID) {
  SUnit *SU = 0;
  ReadyQueue *AQ = Available[QID];

  if (AQ->empty()) {
    MoveUnits(Pending[QID], AQ);
  }
  if (!AQ->empty()) {
    SU = *AQ->begin();
    AQ->remove(AQ->begin());
  }
  return SU;
}

