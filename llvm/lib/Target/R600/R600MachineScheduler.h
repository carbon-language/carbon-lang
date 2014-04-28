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
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

namespace llvm {

class R600SchedStrategy : public MachineSchedStrategy {

  const ScheduleDAGMILive *DAG;
  const R600InstrInfo *TII;
  const R600RegisterInfo *TRI;
  MachineRegisterInfo *MRI;

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
    AluPredX,
    AluTrans,
    AluDiscarded, // LLVM Instructions that are going to be eliminated
    AluLast
  };

  std::vector<SUnit *> Available[IDLast], Pending[IDLast];
  std::vector<SUnit *> AvailableAlus[AluLast];
  std::vector<SUnit *> PhysicalRegCopy;

  InstKind CurInstKind;
  int CurEmitted;
  InstKind NextInstKind;

  unsigned AluInstCount;
  unsigned FetchInstCount;

  int InstKindLimit[IDLast];

  int OccupedSlotsMask;

public:
  R600SchedStrategy() :
    DAG(nullptr), TII(nullptr), TRI(nullptr), MRI(nullptr) {
  }

  virtual ~R600SchedStrategy() {
  }

  virtual void initialize(ScheduleDAGMI *dag);
  virtual SUnit *pickNode(bool &IsTopNode);
  virtual void schedNode(SUnit *SU, bool IsTopNode);
  virtual void releaseTopNode(SUnit *SU);
  virtual void releaseBottomNode(SUnit *SU);

private:
  std::vector<MachineInstr *> InstructionsGroupCandidate;
  bool VLIW5;

  int getInstKind(SUnit *SU);
  bool regBelongsToClass(unsigned Reg, const TargetRegisterClass *RC) const;
  AluKind getAluKind(SUnit *SU) const;
  void LoadAlu();
  unsigned AvailablesAluCount() const;
  SUnit *AttemptFillSlot (unsigned Slot, bool AnyAlu);
  void PrepareNextSlot();
  SUnit *PopInst(std::vector<SUnit*> &Q, bool AnyALU);

  void AssignSlot(MachineInstr *MI, unsigned Slot);
  SUnit* pickAlu();
  SUnit* pickOther(int QID);
  void MoveUnits(std::vector<SUnit *> &QSrc, std::vector<SUnit *> &QDst);
};

} // namespace llvm

#endif /* R600MACHINESCHEDULER_H_ */
