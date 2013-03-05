//===-- R600MachineScheduler.h - R600 Scheduler Interface -*- C++ -*-------===//
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
//
//===----------------------------------------------------------------------===//

#ifndef R600MACHINESCHEDULER_H_
#define R600MACHINESCHEDULER_H_

#include "R600InstrInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/PriorityQueue.h"

using namespace llvm;

namespace llvm {

class CompareSUnit {
public:
  bool operator()(const SUnit *S1, const SUnit *S2) {
    return S1->getDepth() > S2->getDepth();
  }
};

class R600SchedStrategy : public MachineSchedStrategy {

  const ScheduleDAGMI *DAG;
  const R600InstrInfo *TII;
  const R600RegisterInfo *TRI;
  MachineRegisterInfo *MRI;

  enum InstQueue {
    QAlu = 1,
    QFetch = 2,
    QOther = 4
  };

  enum InstKind {
    IDAlu,
    IDFetch,
    IDOther,
    IDLast
  };

  enum AluKind {
    AluAny,
    AluT_X,
    AluT_Y,
    AluT_Z,
    AluT_W,
    AluT_XYZW,
    AluDiscarded, // LLVM Instructions that are going to be eliminated
    AluLast
  };

  ReadyQueue *Available[IDLast], *Pending[IDLast];
  std::multiset<SUnit *, CompareSUnit> AvailableAlus[AluLast];

  InstKind CurInstKind;
  int CurEmitted;
  InstKind NextInstKind;

  int InstKindLimit[IDLast];

  int OccupedSlotsMask;

public:
  R600SchedStrategy() :
    DAG(0), TII(0), TRI(0), MRI(0) {
    Available[IDAlu] = new ReadyQueue(QAlu, "AAlu");
    Available[IDFetch] = new ReadyQueue(QFetch, "AFetch");
    Available[IDOther] = new ReadyQueue(QOther, "AOther");
    Pending[IDAlu] = new ReadyQueue(QAlu<<4, "PAlu");
    Pending[IDFetch] = new ReadyQueue(QFetch<<4, "PFetch");
    Pending[IDOther] = new ReadyQueue(QOther<<4, "POther");
  }

  virtual ~R600SchedStrategy() {
    for (unsigned I = 0; I < IDLast; ++I) {
      delete Available[I];
      delete Pending[I];
    }
  }

  virtual void initialize(ScheduleDAGMI *dag);
  virtual SUnit *pickNode(bool &IsTopNode);
  virtual void schedNode(SUnit *SU, bool IsTopNode);
  virtual void releaseTopNode(SUnit *SU);
  virtual void releaseBottomNode(SUnit *SU);

private:
  SUnit *InstructionsGroupCandidate[4];

  int getInstKind(SUnit *SU);
  bool regBelongsToClass(unsigned Reg, const TargetRegisterClass *RC) const;
  AluKind getAluKind(SUnit *SU) const;
  void LoadAlu();
  bool isAvailablesAluEmpty() const;
  SUnit *AttemptFillSlot (unsigned Slot);
  void PrepareNextSlot();
  SUnit *PopInst(std::multiset<SUnit *, CompareSUnit> &Q);

  void AssignSlot(MachineInstr *MI, unsigned Slot);
  SUnit* pickAlu();
  SUnit* pickOther(int QID);
  bool isBundleable(const MachineInstr& MI);
  void MoveUnits(ReadyQueue *QSrc, ReadyQueue *QDst);
};

} // namespace llvm

#endif /* R600MACHINESCHEDULER_H_ */
