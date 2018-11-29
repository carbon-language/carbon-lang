//===- SIInsertWaitcnts.cpp - Insert Wait Instructions --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Insert wait instructions for memory reads and writes.
///
/// Memory reads and writes are issued asynchronously, so we need to insert
/// S_WAITCNT instructions when we want to access any of their results or
/// overwrite any register that's used asynchronously.
///
/// TODO: This pass currently keeps one timeline per hardware counter. A more
/// finely-grained approach that keeps one timeline per event type could
/// sometimes get away with generating weaker s_waitcnt instructions. For
/// example, when both SMEM and LDS are in flight and we need to wait for
/// the i-th-last LDS instruction, then an lgkmcnt(i) is actually sufficient,
/// but the pass will currently generate a conservative lgkmcnt(0) because
/// multiple event types are in flight.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "si-insert-waitcnts"

DEBUG_COUNTER(ForceExpCounter, DEBUG_TYPE"-forceexp",
              "Force emit s_waitcnt expcnt(0) instrs");
DEBUG_COUNTER(ForceLgkmCounter, DEBUG_TYPE"-forcelgkm",
              "Force emit s_waitcnt lgkmcnt(0) instrs");
DEBUG_COUNTER(ForceVMCounter, DEBUG_TYPE"-forcevm",
              "Force emit s_waitcnt vmcnt(0) instrs");

static cl::opt<unsigned> ForceEmitZeroFlag(
  "amdgpu-waitcnt-forcezero",
  cl::desc("Force all waitcnt instrs to be emitted as s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"),
  cl::init(0), cl::Hidden);

namespace {

template <typename EnumT>
class enum_iterator
    : public iterator_facade_base<enum_iterator<EnumT>,
                                  std::forward_iterator_tag, const EnumT> {
  EnumT Value;
public:
  enum_iterator() = default;
  enum_iterator(EnumT Value) : Value(Value) {}

  enum_iterator &operator++() {
    Value = static_cast<EnumT>(Value + 1);
    return *this;
  }

  bool operator==(const enum_iterator &RHS) const { return Value == RHS.Value; }

  EnumT operator*() const { return Value; }
};

// Class of object that encapsulates latest instruction counter score
// associated with the operand.  Used for determining whether
// s_waitcnt instruction needs to be emited.

#define CNT_MASK(t) (1u << (t))

enum InstCounterType { VM_CNT = 0, LGKM_CNT, EXP_CNT, NUM_INST_CNTS };

iterator_range<enum_iterator<InstCounterType>> inst_counter_types() {
  return make_range(enum_iterator<InstCounterType>(VM_CNT),
                    enum_iterator<InstCounterType>(NUM_INST_CNTS));
}

using RegInterval = std::pair<signed, signed>;

struct {
  uint32_t VmcntMax;
  uint32_t ExpcntMax;
  uint32_t LgkmcntMax;
  int32_t NumVGPRsMax;
  int32_t NumSGPRsMax;
} HardwareLimits;

struct {
  unsigned VGPR0;
  unsigned VGPRL;
  unsigned SGPR0;
  unsigned SGPRL;
} RegisterEncoding;

enum WaitEventType {
  VMEM_ACCESS,      // vector-memory read & write
  LDS_ACCESS,       // lds read & write
  GDS_ACCESS,       // gds read & write
  SQ_MESSAGE,       // send message
  SMEM_ACCESS,      // scalar-memory read & write
  EXP_GPR_LOCK,     // export holding on its data src
  GDS_GPR_LOCK,     // GDS holding on its data and addr src
  EXP_POS_ACCESS,   // write to export position
  EXP_PARAM_ACCESS, // write to export parameter
  VMW_GPR_LOCK,     // vector-memory write holding on its data src
  NUM_WAIT_EVENTS,
};

static const uint32_t WaitEventMaskForInst[NUM_INST_CNTS] = {
  (1 << VMEM_ACCESS),
  (1 << SMEM_ACCESS) | (1 << LDS_ACCESS) | (1 << GDS_ACCESS) |
      (1 << SQ_MESSAGE),
  (1 << EXP_GPR_LOCK) | (1 << GDS_GPR_LOCK) | (1 << VMW_GPR_LOCK) |
      (1 << EXP_PARAM_ACCESS) | (1 << EXP_POS_ACCESS),
};

// The mapping is:
//  0                .. SQ_MAX_PGM_VGPRS-1               real VGPRs
//  SQ_MAX_PGM_VGPRS .. NUM_ALL_VGPRS-1                  extra VGPR-like slots
//  NUM_ALL_VGPRS    .. NUM_ALL_VGPRS+SQ_MAX_PGM_SGPRS-1 real SGPRs
// We reserve a fixed number of VGPR slots in the scoring tables for
// special tokens like SCMEM_LDS (needed for buffer load to LDS).
enum RegisterMapping {
  SQ_MAX_PGM_VGPRS = 256, // Maximum programmable VGPRs across all targets.
  SQ_MAX_PGM_SGPRS = 256, // Maximum programmable SGPRs across all targets.
  NUM_EXTRA_VGPRS = 1,    // A reserved slot for DS.
  EXTRA_VGPR_LDS = 0,     // This is a placeholder the Shader algorithm uses.
  NUM_ALL_VGPRS = SQ_MAX_PGM_VGPRS + NUM_EXTRA_VGPRS, // Where SGPR starts.
};

void addWait(AMDGPU::Waitcnt &Wait, InstCounterType T, unsigned Count) {
  switch (T) {
  case VM_CNT:
    Wait.VmCnt = std::min(Wait.VmCnt, Count);
    break;
  case EXP_CNT:
    Wait.ExpCnt = std::min(Wait.ExpCnt, Count);
    break;
  case LGKM_CNT:
    Wait.LgkmCnt = std::min(Wait.LgkmCnt, Count);
    break;
  default:
    llvm_unreachable("bad InstCounterType");
  }
}

// This is a per-basic-block object that maintains current score brackets
// of each wait counter, and a per-register scoreboard for each wait counter.
// We also maintain the latest score for every event type that can change the
// waitcnt in order to know if there are multiple types of events within
// the brackets. When multiple types of event happen in the bracket,
// wait count may get decreased out of order, therefore we need to put in
// "s_waitcnt 0" before use.
class BlockWaitcntBrackets {
public:
  BlockWaitcntBrackets(const GCNSubtarget *SubTarget) : ST(SubTarget) {
    for (auto T : inst_counter_types())
      memset(VgprScores[T], 0, sizeof(VgprScores[T]));
  }

  ~BlockWaitcntBrackets() = default;

  static uint32_t getWaitCountMax(InstCounterType T) {
    switch (T) {
    case VM_CNT:
      return HardwareLimits.VmcntMax;
    case LGKM_CNT:
      return HardwareLimits.LgkmcntMax;
    case EXP_CNT:
      return HardwareLimits.ExpcntMax;
    default:
      break;
    }
    return 0;
  }

  void setScoreLB(InstCounterType T, uint32_t Val) {
    assert(T < NUM_INST_CNTS);
    if (T >= NUM_INST_CNTS)
      return;
    ScoreLBs[T] = Val;
  }

  void setScoreUB(InstCounterType T, uint32_t Val) {
    assert(T < NUM_INST_CNTS);
    if (T >= NUM_INST_CNTS)
      return;
    ScoreUBs[T] = Val;
    if (T == EXP_CNT) {
      uint32_t UB = ScoreUBs[T] - getWaitCountMax(EXP_CNT);
      if (ScoreLBs[T] < UB && UB < ScoreUBs[T])
        ScoreLBs[T] = UB;
    }
  }

  uint32_t getScoreLB(InstCounterType T) const {
    assert(T < NUM_INST_CNTS);
    if (T >= NUM_INST_CNTS)
      return 0;
    return ScoreLBs[T];
  }

  uint32_t getScoreUB(InstCounterType T) const {
    assert(T < NUM_INST_CNTS);
    if (T >= NUM_INST_CNTS)
      return 0;
    return ScoreUBs[T];
  }

  // Mapping from event to counter.
  InstCounterType eventCounter(WaitEventType E) {
    if (E == VMEM_ACCESS)
      return VM_CNT;
    if (WaitEventMaskForInst[LGKM_CNT] & (1 << E))
      return LGKM_CNT;
    assert(WaitEventMaskForInst[EXP_CNT] & (1 << E));
    return EXP_CNT;
  }

  void setRegScore(int GprNo, InstCounterType T, uint32_t Val) {
    if (GprNo < NUM_ALL_VGPRS) {
      if (GprNo > VgprUB) {
        VgprUB = GprNo;
      }
      VgprScores[T][GprNo] = Val;
    } else {
      assert(T == LGKM_CNT);
      if (GprNo - NUM_ALL_VGPRS > SgprUB) {
        SgprUB = GprNo - NUM_ALL_VGPRS;
      }
      SgprScores[GprNo - NUM_ALL_VGPRS] = Val;
    }
  }

  uint32_t getRegScore(int GprNo, InstCounterType T) {
    if (GprNo < NUM_ALL_VGPRS) {
      return VgprScores[T][GprNo];
    }
    assert(T == LGKM_CNT);
    return SgprScores[GprNo - NUM_ALL_VGPRS];
  }

  void clear() {
    memset(ScoreLBs, 0, sizeof(ScoreLBs));
    memset(ScoreUBs, 0, sizeof(ScoreUBs));
    PendingEvents = 0;
    memset(MixedPendingEvents, 0, sizeof(MixedPendingEvents));
    for (auto T : inst_counter_types())
      memset(VgprScores[T], 0, sizeof(VgprScores[T]));
    memset(SgprScores, 0, sizeof(SgprScores));
  }

  RegInterval getRegInterval(const MachineInstr *MI, const SIInstrInfo *TII,
                             const MachineRegisterInfo *MRI,
                             const SIRegisterInfo *TRI, unsigned OpNo,
                             bool Def) const;

  void setExpScore(const MachineInstr *MI, const SIInstrInfo *TII,
                   const SIRegisterInfo *TRI, const MachineRegisterInfo *MRI,
                   unsigned OpNo, uint32_t Val);

  int32_t getMaxVGPR() const { return VgprUB; }
  int32_t getMaxSGPR() const { return SgprUB; }

  bool counterOutOfOrder(InstCounterType T) const;
  bool simplifyWaitcnt(AMDGPU::Waitcnt &Wait) const;
  bool simplifyWaitcnt(InstCounterType T, unsigned &Count) const;
  void determineWait(InstCounterType T, uint32_t ScoreToWait,
                     AMDGPU::Waitcnt &Wait) const;
  void applyWaitcnt(const AMDGPU::Waitcnt &Wait);
  void applyWaitcnt(InstCounterType T, unsigned Count);
  void updateByEvent(const SIInstrInfo *TII, const SIRegisterInfo *TRI,
                     const MachineRegisterInfo *MRI, WaitEventType E,
                     MachineInstr &MI);

  bool hasPendingEvent(WaitEventType E) const {
    return PendingEvents & (1 << E);
  }

  void mergePendingEvents(const BlockWaitcntBrackets &Other);

  bool hasPendingFlat() const {
    return ((LastFlat[LGKM_CNT] > ScoreLBs[LGKM_CNT] &&
             LastFlat[LGKM_CNT] <= ScoreUBs[LGKM_CNT]) ||
            (LastFlat[VM_CNT] > ScoreLBs[VM_CNT] &&
             LastFlat[VM_CNT] <= ScoreUBs[VM_CNT]));
  }

  void setPendingFlat() {
    LastFlat[VM_CNT] = ScoreUBs[VM_CNT];
    LastFlat[LGKM_CNT] = ScoreUBs[LGKM_CNT];
  }

  int pendingFlat(InstCounterType Ct) const { return LastFlat[Ct]; }

  void setLastFlat(InstCounterType Ct, int Val) { LastFlat[Ct] = Val; }

  bool getRevisitLoop() const { return RevisitLoop; }
  void setRevisitLoop(bool RevisitLoopIn) { RevisitLoop = RevisitLoopIn; }

  void setPostOrder(int32_t PostOrderIn) { PostOrder = PostOrderIn; }
  int32_t getPostOrder() const { return PostOrder; }

  void print(raw_ostream &);
  void dump() { print(dbgs()); }

private:
  const GCNSubtarget *ST = nullptr;
  bool RevisitLoop = false;
  int32_t PostOrder = 0;
  uint32_t ScoreLBs[NUM_INST_CNTS] = {0};
  uint32_t ScoreUBs[NUM_INST_CNTS] = {0};
  uint32_t PendingEvents = 0;
  bool MixedPendingEvents[NUM_INST_CNTS] = {false};
  // Remember the last flat memory operation.
  uint32_t LastFlat[NUM_INST_CNTS] = {0};
  // wait_cnt scores for every vgpr.
  // Keep track of the VgprUB and SgprUB to make merge at join efficient.
  int32_t VgprUB = 0;
  int32_t SgprUB = 0;
  uint32_t VgprScores[NUM_INST_CNTS][NUM_ALL_VGPRS];
  // Wait cnt scores for every sgpr, only lgkmcnt is relevant.
  uint32_t SgprScores[SQ_MAX_PGM_SGPRS] = {0};
};

// This is a per-loop-region object that records waitcnt status at the end of
// loop footer from the previous iteration. We also maintain an iteration
// count to track the number of times the loop has been visited. When it
// doesn't converge naturally, we force convergence by inserting s_waitcnt 0
// at the end of the loop footer.
class LoopWaitcntData {
public:
  LoopWaitcntData() = default;
  ~LoopWaitcntData() = default;

  void incIterCnt() { IterCnt++; }
  void resetIterCnt() { IterCnt = 0; }
  unsigned getIterCnt() { return IterCnt; }

  void setWaitcnt(MachineInstr *WaitcntIn) { LfWaitcnt = WaitcntIn; }
  MachineInstr *getWaitcnt() const { return LfWaitcnt; }

  void print() { LLVM_DEBUG(dbgs() << "  iteration " << IterCnt << '\n';); }

private:
  // s_waitcnt added at the end of loop footer to stablize wait scores
  // at the end of the loop footer.
  MachineInstr *LfWaitcnt = nullptr;
  // Number of iterations the loop has been visited, not including the initial
  // walk over.
  int32_t IterCnt = 0;
};

class SIInsertWaitcnts : public MachineFunctionPass {
private:
  const GCNSubtarget *ST = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  const MachineLoopInfo *MLI = nullptr;
  AMDGPU::IsaVersion IV;

  DenseSet<MachineBasicBlock *> BlockVisitedSet;
  DenseSet<MachineInstr *> TrackedWaitcntSet;
  DenseSet<MachineInstr *> VCCZBugHandledSet;

  DenseMap<MachineBasicBlock *, std::unique_ptr<BlockWaitcntBrackets>>
      BlockWaitcntBracketsMap;

  std::vector<MachineBasicBlock *> BlockWaitcntProcessedSet;

  DenseMap<MachineLoop *, std::unique_ptr<LoopWaitcntData>> LoopWaitcntDataMap;

  // ForceEmitZeroWaitcnts: force all waitcnts insts to be s_waitcnt 0
  // because of amdgpu-waitcnt-forcezero flag
  bool ForceEmitZeroWaitcnts;
  bool ForceEmitWaitcnt[NUM_INST_CNTS];

public:
  static char ID;

  SIInsertWaitcnts() : MachineFunctionPass(ID) {
    (void)ForceExpCounter;
    (void)ForceLgkmCounter;
    (void)ForceVMCounter;
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI insert wait instructions";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool isForceEmitWaitcnt() const {
    for (auto T : inst_counter_types())
      if (ForceEmitWaitcnt[T])
        return true;
    return false;
  }

  void setForceEmitWaitcnt() {
// For non-debug builds, ForceEmitWaitcnt has been initialized to false;
// For debug builds, get the debug counter info and adjust if need be
#ifndef NDEBUG
    if (DebugCounter::isCounterSet(ForceExpCounter) &&
        DebugCounter::shouldExecute(ForceExpCounter)) {
      ForceEmitWaitcnt[EXP_CNT] = true;
    } else {
      ForceEmitWaitcnt[EXP_CNT] = false;
    }

    if (DebugCounter::isCounterSet(ForceLgkmCounter) &&
         DebugCounter::shouldExecute(ForceLgkmCounter)) {
      ForceEmitWaitcnt[LGKM_CNT] = true;
    } else {
      ForceEmitWaitcnt[LGKM_CNT] = false;
    }

    if (DebugCounter::isCounterSet(ForceVMCounter) &&
        DebugCounter::shouldExecute(ForceVMCounter)) {
      ForceEmitWaitcnt[VM_CNT] = true;
    } else {
      ForceEmitWaitcnt[VM_CNT] = false;
    }
#endif // NDEBUG
  }

  bool mayAccessLDSThroughFlat(const MachineInstr &MI) const;
  void generateWaitcntInstBefore(MachineInstr &MI,
                                 BlockWaitcntBrackets *ScoreBrackets,
                                 MachineInstr *OldWaitcntInstr);
  void updateEventWaitcntAfter(MachineInstr &Inst,
                               BlockWaitcntBrackets *ScoreBrackets);
  void mergeInputScoreBrackets(MachineBasicBlock &Block);
  bool isLoopBottom(const MachineLoop *Loop, const MachineBasicBlock *Block);
  unsigned countNumBottomBlocks(const MachineLoop *Loop);
  void insertWaitcntInBlock(MachineFunction &MF, MachineBasicBlock &Block);
  void insertWaitcntBeforeCF(MachineBasicBlock &Block, MachineInstr *Inst);
};

} // end anonymous namespace

RegInterval BlockWaitcntBrackets::getRegInterval(const MachineInstr *MI,
                                                 const SIInstrInfo *TII,
                                                 const MachineRegisterInfo *MRI,
                                                 const SIRegisterInfo *TRI,
                                                 unsigned OpNo,
                                                 bool Def) const {
  const MachineOperand &Op = MI->getOperand(OpNo);
  if (!Op.isReg() || !TRI->isInAllocatableClass(Op.getReg()) ||
      (Def && !Op.isDef()))
    return {-1, -1};

  // A use via a PW operand does not need a waitcnt.
  // A partial write is not a WAW.
  assert(!Op.getSubReg() || !Op.isUndef());

  RegInterval Result;
  const MachineRegisterInfo &MRIA = *MRI;

  unsigned Reg = TRI->getEncodingValue(Op.getReg());

  if (TRI->isVGPR(MRIA, Op.getReg())) {
    assert(Reg >= RegisterEncoding.VGPR0 && Reg <= RegisterEncoding.VGPRL);
    Result.first = Reg - RegisterEncoding.VGPR0;
    assert(Result.first >= 0 && Result.first < SQ_MAX_PGM_VGPRS);
  } else if (TRI->isSGPRReg(MRIA, Op.getReg())) {
    assert(Reg >= RegisterEncoding.SGPR0 && Reg < SQ_MAX_PGM_SGPRS);
    Result.first = Reg - RegisterEncoding.SGPR0 + NUM_ALL_VGPRS;
    assert(Result.first >= NUM_ALL_VGPRS &&
           Result.first < SQ_MAX_PGM_SGPRS + NUM_ALL_VGPRS);
  }
  // TODO: Handle TTMP
  // else if (TRI->isTTMP(MRIA, Reg.getReg())) ...
  else
    return {-1, -1};

  const MachineInstr &MIA = *MI;
  const TargetRegisterClass *RC = TII->getOpRegClass(MIA, OpNo);
  unsigned Size = TRI->getRegSizeInBits(*RC);
  Result.second = Result.first + (Size / 32);

  return Result;
}

void BlockWaitcntBrackets::setExpScore(const MachineInstr *MI,
                                       const SIInstrInfo *TII,
                                       const SIRegisterInfo *TRI,
                                       const MachineRegisterInfo *MRI,
                                       unsigned OpNo, uint32_t Val) {
  RegInterval Interval = getRegInterval(MI, TII, MRI, TRI, OpNo, false);
  LLVM_DEBUG({
    const MachineOperand &Opnd = MI->getOperand(OpNo);
    assert(TRI->isVGPR(*MRI, Opnd.getReg()));
  });
  for (signed RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
    setRegScore(RegNo, EXP_CNT, Val);
  }
}

void BlockWaitcntBrackets::updateByEvent(const SIInstrInfo *TII,
                                         const SIRegisterInfo *TRI,
                                         const MachineRegisterInfo *MRI,
                                         WaitEventType E, MachineInstr &Inst) {
  const MachineRegisterInfo &MRIA = *MRI;
  InstCounterType T = eventCounter(E);
  uint32_t CurrScore = getScoreUB(T) + 1;
  if (CurrScore == 0)
    report_fatal_error("InsertWaitcnt score wraparound");
  // PendingEvents and ScoreUB need to be update regardless if this event
  // changes the score of a register or not.
  // Examples including vm_cnt when buffer-store or lgkm_cnt when send-message.
  if (!hasPendingEvent(E)) {
    if (PendingEvents & WaitEventMaskForInst[T])
      MixedPendingEvents[T] = true;
    PendingEvents |= 1 << E;
  }
  setScoreUB(T, CurrScore);

  if (T == EXP_CNT) {
    // Put score on the source vgprs. If this is a store, just use those
    // specific register(s).
    if (TII->isDS(Inst) && (Inst.mayStore() || Inst.mayLoad())) {
      // All GDS operations must protect their address register (same as
      // export.)
      if (Inst.getOpcode() != AMDGPU::DS_APPEND &&
          Inst.getOpcode() != AMDGPU::DS_CONSUME) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::addr),
            CurrScore);
      }
      if (Inst.mayStore()) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data0),
            CurrScore);
        if (AMDGPU::getNamedOperandIdx(Inst.getOpcode(),
                                       AMDGPU::OpName::data1) != -1) {
          setExpScore(&Inst, TII, TRI, MRI,
                      AMDGPU::getNamedOperandIdx(Inst.getOpcode(),
                                                 AMDGPU::OpName::data1),
                      CurrScore);
        }
      } else if (AMDGPU::getAtomicNoRetOp(Inst.getOpcode()) != -1 &&
                 Inst.getOpcode() != AMDGPU::DS_GWS_INIT &&
                 Inst.getOpcode() != AMDGPU::DS_GWS_SEMA_V &&
                 Inst.getOpcode() != AMDGPU::DS_GWS_SEMA_BR &&
                 Inst.getOpcode() != AMDGPU::DS_GWS_SEMA_P &&
                 Inst.getOpcode() != AMDGPU::DS_GWS_BARRIER &&
                 Inst.getOpcode() != AMDGPU::DS_APPEND &&
                 Inst.getOpcode() != AMDGPU::DS_CONSUME &&
                 Inst.getOpcode() != AMDGPU::DS_ORDERED_COUNT) {
        for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
          const MachineOperand &Op = Inst.getOperand(I);
          if (Op.isReg() && !Op.isDef() && TRI->isVGPR(MRIA, Op.getReg())) {
            setExpScore(&Inst, TII, TRI, MRI, I, CurrScore);
          }
        }
      }
    } else if (TII->isFLAT(Inst)) {
      if (Inst.mayStore()) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data),
            CurrScore);
      } else if (AMDGPU::getAtomicNoRetOp(Inst.getOpcode()) != -1) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data),
            CurrScore);
      }
    } else if (TII->isMIMG(Inst)) {
      if (Inst.mayStore()) {
        setExpScore(&Inst, TII, TRI, MRI, 0, CurrScore);
      } else if (AMDGPU::getAtomicNoRetOp(Inst.getOpcode()) != -1) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data),
            CurrScore);
      }
    } else if (TII->isMTBUF(Inst)) {
      if (Inst.mayStore()) {
        setExpScore(&Inst, TII, TRI, MRI, 0, CurrScore);
      }
    } else if (TII->isMUBUF(Inst)) {
      if (Inst.mayStore()) {
        setExpScore(&Inst, TII, TRI, MRI, 0, CurrScore);
      } else if (AMDGPU::getAtomicNoRetOp(Inst.getOpcode()) != -1) {
        setExpScore(
            &Inst, TII, TRI, MRI,
            AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::data),
            CurrScore);
      }
    } else {
      if (TII->isEXP(Inst)) {
        // For export the destination registers are really temps that
        // can be used as the actual source after export patching, so
        // we need to treat them like sources and set the EXP_CNT
        // score.
        for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
          MachineOperand &DefMO = Inst.getOperand(I);
          if (DefMO.isReg() && DefMO.isDef() &&
              TRI->isVGPR(MRIA, DefMO.getReg())) {
            setRegScore(TRI->getEncodingValue(DefMO.getReg()), EXP_CNT,
                        CurrScore);
          }
        }
      }
      for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
        MachineOperand &MO = Inst.getOperand(I);
        if (MO.isReg() && !MO.isDef() && TRI->isVGPR(MRIA, MO.getReg())) {
          setExpScore(&Inst, TII, TRI, MRI, I, CurrScore);
        }
      }
    }
#if 0 // TODO: check if this is handled by MUBUF code above.
  } else if (Inst.getOpcode() == AMDGPU::BUFFER_STORE_DWORD ||
       Inst.getOpcode() == AMDGPU::BUFFER_STORE_DWORDX2 ||
       Inst.getOpcode() == AMDGPU::BUFFER_STORE_DWORDX4) {
    MachineOperand *MO = TII->getNamedOperand(Inst, AMDGPU::OpName::data);
    unsigned OpNo;//TODO: find the OpNo for this operand;
    RegInterval Interval = getRegInterval(&Inst, TII, MRI, TRI, OpNo, false);
    for (signed RegNo = Interval.first; RegNo < Interval.second;
    ++RegNo) {
      setRegScore(RegNo + NUM_ALL_VGPRS, t, CurrScore);
    }
#endif
  } else {
    // Match the score to the destination registers.
    for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
      RegInterval Interval = getRegInterval(&Inst, TII, MRI, TRI, I, true);
      if (T == VM_CNT && Interval.first >= NUM_ALL_VGPRS)
        continue;
      for (signed RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
        setRegScore(RegNo, T, CurrScore);
      }
    }
    if (TII->isDS(Inst) && Inst.mayStore()) {
      setRegScore(SQ_MAX_PGM_VGPRS + EXTRA_VGPR_LDS, T, CurrScore);
    }
  }
}

void BlockWaitcntBrackets::print(raw_ostream &OS) {
  OS << '\n';
  for (auto T : inst_counter_types()) {
    uint32_t LB = getScoreLB(T);
    uint32_t UB = getScoreUB(T);

    switch (T) {
    case VM_CNT:
      OS << "    VM_CNT(" << UB - LB << "): ";
      break;
    case LGKM_CNT:
      OS << "    LGKM_CNT(" << UB - LB << "): ";
      break;
    case EXP_CNT:
      OS << "    EXP_CNT(" << UB - LB << "): ";
      break;
    default:
      OS << "    UNKNOWN(" << UB - LB << "): ";
      break;
    }

    if (LB < UB) {
      // Print vgpr scores.
      for (int J = 0; J <= getMaxVGPR(); J++) {
        uint32_t RegScore = getRegScore(J, T);
        if (RegScore <= LB)
          continue;
        uint32_t RelScore = RegScore - LB - 1;
        if (J < SQ_MAX_PGM_VGPRS + EXTRA_VGPR_LDS) {
          OS << RelScore << ":v" << J << " ";
        } else {
          OS << RelScore << ":ds ";
        }
      }
      // Also need to print sgpr scores for lgkm_cnt.
      if (T == LGKM_CNT) {
        for (int J = 0; J <= getMaxSGPR(); J++) {
          uint32_t RegScore = getRegScore(J + NUM_ALL_VGPRS, LGKM_CNT);
          if (RegScore <= LB)
            continue;
          uint32_t RelScore = RegScore - LB - 1;
          OS << RelScore << ":s" << J << " ";
        }
      }
    }
    OS << '\n';
  }
  OS << '\n';
}

/// Simplify the waitcnt, in the sense of removing redundant counts, and return
/// whether a waitcnt instruction is needed at all.
bool BlockWaitcntBrackets::simplifyWaitcnt(AMDGPU::Waitcnt &Wait) const {
  return simplifyWaitcnt(VM_CNT, Wait.VmCnt) |
         simplifyWaitcnt(EXP_CNT, Wait.ExpCnt) |
         simplifyWaitcnt(LGKM_CNT, Wait.LgkmCnt);
}

bool BlockWaitcntBrackets::simplifyWaitcnt(InstCounterType T,
                                           unsigned &Count) const {
  const uint32_t LB = getScoreLB(T);
  const uint32_t UB = getScoreUB(T);
  if (Count < UB && UB - Count > LB)
    return true;

  Count = ~0u;
  return false;
}

void BlockWaitcntBrackets::determineWait(InstCounterType T,
                                         uint32_t ScoreToWait,
                                         AMDGPU::Waitcnt &Wait) const {
  // If the score of src_operand falls within the bracket, we need an
  // s_waitcnt instruction.
  const uint32_t LB = getScoreLB(T);
  const uint32_t UB = getScoreUB(T);
  if ((UB >= ScoreToWait) && (ScoreToWait > LB)) {
    if ((T == VM_CNT || T == LGKM_CNT) &&
        hasPendingFlat() &&
        !ST->hasFlatLgkmVMemCountInOrder()) {
      // If there is a pending FLAT operation, and this is a VMem or LGKM
      // waitcnt and the target can report early completion, then we need
      // to force a waitcnt 0.
      addWait(Wait, T, 0);
    } else if (counterOutOfOrder(T)) {
      // Counter can get decremented out-of-order when there
      // are multiple types event in the bracket. Also emit an s_wait counter
      // with a conservative value of 0 for the counter.
      addWait(Wait, T, 0);
    } else {
      addWait(Wait, T, UB - ScoreToWait);
    }
  }
}

void BlockWaitcntBrackets::applyWaitcnt(const AMDGPU::Waitcnt &Wait) {
  applyWaitcnt(VM_CNT, Wait.VmCnt);
  applyWaitcnt(EXP_CNT, Wait.ExpCnt);
  applyWaitcnt(LGKM_CNT, Wait.LgkmCnt);
}

void BlockWaitcntBrackets::applyWaitcnt(InstCounterType T, unsigned Count) {
  const uint32_t UB = getScoreUB(T);
  if (Count >= UB)
    return;
  if (Count != 0) {
    if (counterOutOfOrder(T))
      return;
    setScoreLB(T, std::max(getScoreLB(T), UB - Count));
  } else {
    setScoreLB(T, UB);
    MixedPendingEvents[T] = false;
    PendingEvents &= ~WaitEventMaskForInst[T];
  }
}

void BlockWaitcntBrackets::mergePendingEvents(const BlockWaitcntBrackets &Other) {
  for (auto T : inst_counter_types()) {
    uint32_t Old = PendingEvents & WaitEventMaskForInst[T];
    uint32_t New = Other.PendingEvents & WaitEventMaskForInst[T];
    if (Other.MixedPendingEvents[T] || (Old && New && Old != New))
      MixedPendingEvents[T] = true;
    PendingEvents |= New;
  }
}

// Where there are multiple types of event in the bracket of a counter,
// the decrement may go out of order.
bool BlockWaitcntBrackets::counterOutOfOrder(InstCounterType T) const {
  // Scalar memory read always can go out of order.
  if (T == LGKM_CNT && hasPendingEvent(SMEM_ACCESS))
    return true;
  return MixedPendingEvents[T];
}

INITIALIZE_PASS_BEGIN(SIInsertWaitcnts, DEBUG_TYPE, "SI Insert Waitcnts", false,
                      false)
INITIALIZE_PASS_END(SIInsertWaitcnts, DEBUG_TYPE, "SI Insert Waitcnts", false,
                    false)

char SIInsertWaitcnts::ID = 0;

char &llvm::SIInsertWaitcntsID = SIInsertWaitcnts::ID;

FunctionPass *llvm::createSIInsertWaitcntsPass() {
  return new SIInsertWaitcnts();
}

static bool readsVCCZ(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return (Opc == AMDGPU::S_CBRANCH_VCCNZ || Opc == AMDGPU::S_CBRANCH_VCCZ) &&
         !MI.getOperand(1).isUndef();
}

///  Generate s_waitcnt instruction to be placed before cur_Inst.
///  Instructions of a given type are returned in order,
///  but instructions of different types can complete out of order.
///  We rely on this in-order completion
///  and simply assign a score to the memory access instructions.
///  We keep track of the active "score bracket" to determine
///  if an access of a memory read requires an s_waitcnt
///  and if so what the value of each counter is.
///  The "score bracket" is bound by the lower bound and upper bound
///  scores (*_score_LB and *_score_ub respectively).
void SIInsertWaitcnts::generateWaitcntInstBefore(
    MachineInstr &MI, BlockWaitcntBrackets *ScoreBrackets,
    MachineInstr *OldWaitcntInstr) {
  setForceEmitWaitcnt();
  bool IsForceEmitWaitcnt = isForceEmitWaitcnt();

  if (MI.isDebugInstr())
    return;

  AMDGPU::Waitcnt Wait;

  // See if this instruction has a forced S_WAITCNT VM.
  // TODO: Handle other cases of NeedsWaitcntVmBefore()
  if (MI.getOpcode() == AMDGPU::BUFFER_WBINVL1 ||
           MI.getOpcode() == AMDGPU::BUFFER_WBINVL1_SC ||
           MI.getOpcode() == AMDGPU::BUFFER_WBINVL1_VOL) {
    Wait.VmCnt = 0;
  }

  // All waits must be resolved at call return.
  // NOTE: this could be improved with knowledge of all call sites or
  //   with knowledge of the called routines.
  if (MI.getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG ||
      MI.getOpcode() == AMDGPU::S_SETPC_B64_return) {
    Wait = AMDGPU::Waitcnt::allZero();
  }
  // Resolve vm waits before gs-done.
  else if ((MI.getOpcode() == AMDGPU::S_SENDMSG ||
            MI.getOpcode() == AMDGPU::S_SENDMSGHALT) &&
           ((MI.getOperand(0).getImm() & AMDGPU::SendMsg::ID_MASK_) ==
            AMDGPU::SendMsg::ID_GS_DONE)) {
    Wait.VmCnt = 0;
  }
#if 0 // TODO: the following blocks of logic when we have fence.
  else if (MI.getOpcode() == SC_FENCE) {
    const unsigned int group_size =
      context->shader_info->GetMaxThreadGroupSize();
    // group_size == 0 means thread group size is unknown at compile time
    const bool group_is_multi_wave =
      (group_size == 0 || group_size > target_info->GetWaveFrontSize());
    const bool fence_is_global = !((SCInstInternalMisc*)Inst)->IsGroupFence();

    for (unsigned int i = 0; i < Inst->NumSrcOperands(); i++) {
      SCRegType src_type = Inst->GetSrcType(i);
      switch (src_type) {
        case SCMEM_LDS:
          if (group_is_multi_wave ||
            context->OptFlagIsOn(OPT_R1100_LDSMEM_FENCE_CHICKEN_BIT)) {
            EmitWaitcnt |= ScoreBrackets->updateByWait(LGKM_CNT,
                               ScoreBrackets->getScoreUB(LGKM_CNT));
            // LDS may have to wait for VM_CNT after buffer load to LDS
            if (target_info->HasBufferLoadToLDS()) {
              EmitWaitcnt |= ScoreBrackets->updateByWait(VM_CNT,
                                 ScoreBrackets->getScoreUB(VM_CNT));
            }
          }
          break;

        case SCMEM_GDS:
          if (group_is_multi_wave || fence_is_global) {
            EmitWaitcnt |= ScoreBrackets->updateByWait(EXP_CNT,
              ScoreBrackets->getScoreUB(EXP_CNT));
            EmitWaitcnt |= ScoreBrackets->updateByWait(LGKM_CNT,
              ScoreBrackets->getScoreUB(LGKM_CNT));
          }
          break;

        case SCMEM_UAV:
        case SCMEM_TFBUF:
        case SCMEM_RING:
        case SCMEM_SCATTER:
          if (group_is_multi_wave || fence_is_global) {
            EmitWaitcnt |= ScoreBrackets->updateByWait(EXP_CNT,
              ScoreBrackets->getScoreUB(EXP_CNT));
            EmitWaitcnt |= ScoreBrackets->updateByWait(VM_CNT,
              ScoreBrackets->getScoreUB(VM_CNT));
          }
          break;

        case SCMEM_SCRATCH:
        default:
          break;
      }
    }
  }
#endif

  // Export & GDS instructions do not read the EXEC mask until after the export
  // is granted (which can occur well after the instruction is issued).
  // The shader program must flush all EXP operations on the export-count
  // before overwriting the EXEC mask.
  else {
    if (MI.modifiesRegister(AMDGPU::EXEC, TRI)) {
      // Export and GDS are tracked individually, either may trigger a waitcnt
      // for EXEC.
      if (ScoreBrackets->hasPendingEvent(EXP_GPR_LOCK) ||
          ScoreBrackets->hasPendingEvent(EXP_PARAM_ACCESS) ||
          ScoreBrackets->hasPendingEvent(EXP_POS_ACCESS) ||
          ScoreBrackets->hasPendingEvent(GDS_GPR_LOCK)) {
        Wait.ExpCnt = 0;
      }
    }

#if 0 // TODO: the following code to handle CALL.
    // The argument passing for CALLs should suffice for VM_CNT and LGKM_CNT.
    // However, there is a problem with EXP_CNT, because the call cannot
    // easily tell if a register is used in the function, and if it did, then
    // the referring instruction would have to have an S_WAITCNT, which is
    // dependent on all call sites. So Instead, force S_WAITCNT for EXP_CNTs
    // before the call.
    if (MI.getOpcode() == SC_CALL) {
      if (ScoreBrackets->getScoreUB(EXP_CNT) >
        ScoreBrackets->getScoreLB(EXP_CNT)) {
        ScoreBrackets->setScoreLB(EXP_CNT, ScoreBrackets->getScoreUB(EXP_CNT));
        EmitWaitcnt |= CNT_MASK(EXP_CNT);
      }
    }
#endif

    // FIXME: Should not be relying on memoperands.
    // Look at the source operands of every instruction to see if
    // any of them results from a previous memory operation that affects
    // its current usage. If so, an s_waitcnt instruction needs to be
    // emitted.
    // If the source operand was defined by a load, add the s_waitcnt
    // instruction.
    for (const MachineMemOperand *Memop : MI.memoperands()) {
      unsigned AS = Memop->getAddrSpace();
      if (AS != AMDGPUAS::LOCAL_ADDRESS)
        continue;
      unsigned RegNo = SQ_MAX_PGM_VGPRS + EXTRA_VGPR_LDS;
      // VM_CNT is only relevant to vgpr or LDS.
      ScoreBrackets->determineWait(
          VM_CNT, ScoreBrackets->getRegScore(RegNo, VM_CNT), Wait);
    }

    for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
      const MachineOperand &Op = MI.getOperand(I);
      const MachineRegisterInfo &MRIA = *MRI;
      RegInterval Interval =
          ScoreBrackets->getRegInterval(&MI, TII, MRI, TRI, I, false);
      for (signed RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
        if (TRI->isVGPR(MRIA, Op.getReg())) {
          // VM_CNT is only relevant to vgpr or LDS.
          ScoreBrackets->determineWait(
              VM_CNT, ScoreBrackets->getRegScore(RegNo, VM_CNT), Wait);
        }
        ScoreBrackets->determineWait(
            LGKM_CNT, ScoreBrackets->getRegScore(RegNo, LGKM_CNT), Wait);
      }
    }
    // End of for loop that looks at all source operands to decide vm_wait_cnt
    // and lgk_wait_cnt.

    // Two cases are handled for destination operands:
    // 1) If the destination operand was defined by a load, add the s_waitcnt
    // instruction to guarantee the right WAW order.
    // 2) If a destination operand that was used by a recent export/store ins,
    // add s_waitcnt on exp_cnt to guarantee the WAR order.
    if (MI.mayStore()) {
      // FIXME: Should not be relying on memoperands.
      for (const MachineMemOperand *Memop : MI.memoperands()) {
        unsigned AS = Memop->getAddrSpace();
        if (AS != AMDGPUAS::LOCAL_ADDRESS)
          continue;
        unsigned RegNo = SQ_MAX_PGM_VGPRS + EXTRA_VGPR_LDS;
        ScoreBrackets->determineWait(
            VM_CNT, ScoreBrackets->getRegScore(RegNo, VM_CNT), Wait);
        ScoreBrackets->determineWait(
            EXP_CNT, ScoreBrackets->getRegScore(RegNo, EXP_CNT), Wait);
      }
    }
    for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
      MachineOperand &Def = MI.getOperand(I);
      const MachineRegisterInfo &MRIA = *MRI;
      RegInterval Interval =
          ScoreBrackets->getRegInterval(&MI, TII, MRI, TRI, I, true);
      for (signed RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
        if (TRI->isVGPR(MRIA, Def.getReg())) {
          ScoreBrackets->determineWait(
              VM_CNT, ScoreBrackets->getRegScore(RegNo, VM_CNT), Wait);
          ScoreBrackets->determineWait(
              EXP_CNT, ScoreBrackets->getRegScore(RegNo, EXP_CNT), Wait);
        }
        ScoreBrackets->determineWait(
            LGKM_CNT, ScoreBrackets->getRegScore(RegNo, LGKM_CNT), Wait);
      }
    } // End of for loop that looks at all dest operands.
  }

  // Check to see if this is an S_BARRIER, and if an implicit S_WAITCNT 0
  // occurs before the instruction. Doing it here prevents any additional
  // S_WAITCNTs from being emitted if the instruction was marked as
  // requiring a WAITCNT beforehand.
  if (MI.getOpcode() == AMDGPU::S_BARRIER &&
      !ST->hasAutoWaitcntBeforeBarrier()) {
    Wait = AMDGPU::Waitcnt::allZero();
  }

  // TODO: Remove this work-around, enable the assert for Bug 457939
  //       after fixing the scheduler. Also, the Shader Compiler code is
  //       independent of target.
  if (readsVCCZ(MI) && ST->getGeneration() <= AMDGPUSubtarget::SEA_ISLANDS) {
    if (ScoreBrackets->getScoreLB(LGKM_CNT) <
            ScoreBrackets->getScoreUB(LGKM_CNT) &&
        ScoreBrackets->hasPendingEvent(SMEM_ACCESS)) {
      Wait.LgkmCnt = 0;
    }
  }

  // Early-out if no wait is indicated.
  if (!ScoreBrackets->simplifyWaitcnt(Wait) && !IsForceEmitWaitcnt) {
    if (OldWaitcntInstr) {
      if (TrackedWaitcntSet.count(OldWaitcntInstr)) {
        TrackedWaitcntSet.erase(OldWaitcntInstr);
        OldWaitcntInstr->eraseFromParent();
      } else {
        int64_t Imm = OldWaitcntInstr->getOperand(0).getImm();
        ScoreBrackets->applyWaitcnt(AMDGPU::decodeWaitcnt(IV, Imm));
      }
    }
    return;
  }

  if (ForceEmitZeroWaitcnts)
    Wait = AMDGPU::Waitcnt::allZero();

  if (ForceEmitWaitcnt[VM_CNT])
    Wait.VmCnt = 0;
  if (ForceEmitWaitcnt[EXP_CNT])
    Wait.ExpCnt = 0;
  if (ForceEmitWaitcnt[LGKM_CNT])
    Wait.LgkmCnt = 0;

  ScoreBrackets->applyWaitcnt(Wait);

  AMDGPU::Waitcnt OldWait;
  if (OldWaitcntInstr) {
    OldWait =
        AMDGPU::decodeWaitcnt(IV, OldWaitcntInstr->getOperand(0).getImm());
  }
  if (OldWait.dominates(Wait))
    return;

  MachineLoop *ContainingLoop = MLI->getLoopFor(MI.getParent());
  if (ContainingLoop) {
    MachineBasicBlock *TBB = ContainingLoop->getHeader();
    BlockWaitcntBrackets *ScoreBracket = BlockWaitcntBracketsMap[TBB].get();
    if (!ScoreBracket) {
      assert(!BlockVisitedSet.count(TBB));
      BlockWaitcntBracketsMap[TBB] =
          llvm::make_unique<BlockWaitcntBrackets>(ST);
      ScoreBracket = BlockWaitcntBracketsMap[TBB].get();
    }
    ScoreBracket->setRevisitLoop(true);
    LLVM_DEBUG(dbgs() << "set-revisit2: Block"
                      << ContainingLoop->getHeader()->getNumber() << '\n';);
  }

  if (OldWaitcntInstr && !TrackedWaitcntSet.count(OldWaitcntInstr))
    Wait = Wait.combined(OldWait);

  unsigned Enc = AMDGPU::encodeWaitcnt(IV, Wait);
  if (OldWaitcntInstr) {
    OldWaitcntInstr->getOperand(0).setImm(Enc);

    LLVM_DEBUG(dbgs() << "updateWaitcntInBlock\n"
                      << "Old Instr: " << MI << '\n'
                      << "New Instr: " << *OldWaitcntInstr << '\n');
  } else {
    auto SWaitInst = BuildMI(*MI.getParent(), MI.getIterator(),
                             MI.getDebugLoc(), TII->get(AMDGPU::S_WAITCNT))
                         .addImm(Enc);
    TrackedWaitcntSet.insert(SWaitInst);

    LLVM_DEBUG(dbgs() << "insertWaitcntInBlock\n"
                      << "Old Instr: " << MI << '\n'
                      << "New Instr: " << *SWaitInst << '\n');
  }
}

void SIInsertWaitcnts::insertWaitcntBeforeCF(MachineBasicBlock &MBB,
                                             MachineInstr *Waitcnt) {
  if (MBB.empty()) {
    MBB.push_back(Waitcnt);
    return;
  }

  MachineBasicBlock::iterator It = MBB.end();
  MachineInstr *MI = &*(--It);
  if (MI->isBranch()) {
    MBB.insert(It, Waitcnt);
  } else {
    MBB.push_back(Waitcnt);
  }
}

// This is a flat memory operation. Check to see if it has memory
// tokens for both LDS and Memory, and if so mark it as a flat.
bool SIInsertWaitcnts::mayAccessLDSThroughFlat(const MachineInstr &MI) const {
  if (MI.memoperands_empty())
    return true;

  for (const MachineMemOperand *Memop : MI.memoperands()) {
    unsigned AS = Memop->getAddrSpace();
    if (AS == AMDGPUAS::LOCAL_ADDRESS || AS == AMDGPUAS::FLAT_ADDRESS)
      return true;
  }

  return false;
}

void SIInsertWaitcnts::updateEventWaitcntAfter(
    MachineInstr &Inst, BlockWaitcntBrackets *ScoreBrackets) {
  // Now look at the instruction opcode. If it is a memory access
  // instruction, update the upper-bound of the appropriate counter's
  // bracket and the destination operand scores.
  // TODO: Use the (TSFlags & SIInstrFlags::LGKM_CNT) property everywhere.
  if (TII->isDS(Inst) && TII->usesLGKM_CNT(Inst)) {
    if (TII->hasModifiersSet(Inst, AMDGPU::OpName::gds)) {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, GDS_ACCESS, Inst);
      ScoreBrackets->updateByEvent(TII, TRI, MRI, GDS_GPR_LOCK, Inst);
    } else {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, LDS_ACCESS, Inst);
    }
  } else if (TII->isFLAT(Inst)) {
    assert(Inst.mayLoad() || Inst.mayStore());

    if (TII->usesVM_CNT(Inst))
      ScoreBrackets->updateByEvent(TII, TRI, MRI, VMEM_ACCESS, Inst);

    if (TII->usesLGKM_CNT(Inst)) {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, LDS_ACCESS, Inst);

      // This is a flat memory operation, so note it - it will require
      // that both the VM and LGKM be flushed to zero if it is pending when
      // a VM or LGKM dependency occurs.
      if (mayAccessLDSThroughFlat(Inst))
        ScoreBrackets->setPendingFlat();
    }
  } else if (SIInstrInfo::isVMEM(Inst) &&
             // TODO: get a better carve out.
             Inst.getOpcode() != AMDGPU::BUFFER_WBINVL1 &&
             Inst.getOpcode() != AMDGPU::BUFFER_WBINVL1_SC &&
             Inst.getOpcode() != AMDGPU::BUFFER_WBINVL1_VOL) {
    ScoreBrackets->updateByEvent(TII, TRI, MRI, VMEM_ACCESS, Inst);
    if (ST->vmemWriteNeedsExpWaitcnt() &&
        (Inst.mayStore() || AMDGPU::getAtomicNoRetOp(Inst.getOpcode()) != -1)) {
      ScoreBrackets->updateByEvent(TII, TRI, MRI, VMW_GPR_LOCK, Inst);
    }
  } else if (TII->isSMRD(Inst)) {
    ScoreBrackets->updateByEvent(TII, TRI, MRI, SMEM_ACCESS, Inst);
  } else {
    switch (Inst.getOpcode()) {
    case AMDGPU::S_SENDMSG:
    case AMDGPU::S_SENDMSGHALT:
      ScoreBrackets->updateByEvent(TII, TRI, MRI, SQ_MESSAGE, Inst);
      break;
    case AMDGPU::EXP:
    case AMDGPU::EXP_DONE: {
      int Imm = TII->getNamedOperand(Inst, AMDGPU::OpName::tgt)->getImm();
      if (Imm >= 32 && Imm <= 63)
        ScoreBrackets->updateByEvent(TII, TRI, MRI, EXP_PARAM_ACCESS, Inst);
      else if (Imm >= 12 && Imm <= 15)
        ScoreBrackets->updateByEvent(TII, TRI, MRI, EXP_POS_ACCESS, Inst);
      else
        ScoreBrackets->updateByEvent(TII, TRI, MRI, EXP_GPR_LOCK, Inst);
      break;
    }
    case AMDGPU::S_MEMTIME:
    case AMDGPU::S_MEMREALTIME:
      ScoreBrackets->updateByEvent(TII, TRI, MRI, SMEM_ACCESS, Inst);
      break;
    default:
      break;
    }
  }
}

// Merge the score brackets of the Block's predecessors;
// this merged score bracket is used when adding waitcnts to the Block
void SIInsertWaitcnts::mergeInputScoreBrackets(MachineBasicBlock &Block) {
  BlockWaitcntBrackets *ScoreBrackets = BlockWaitcntBracketsMap[&Block].get();
  uint32_t MaxPending[NUM_INST_CNTS] = {0};
  uint32_t MaxFlat[NUM_INST_CNTS] = {0};

  // For single basic block loops, we need to retain the Block's
  // score bracket to have accurate Pred info. So, make a copy of Block's
  // score bracket, clear() it (which retains several important bits of info),
  // populate, and then replace en masse. For non-single basic block loops,
  // just clear Block's current score bracket and repopulate in-place.
  bool IsSelfPred;
  std::unique_ptr<BlockWaitcntBrackets> S;

  IsSelfPred = (std::find(Block.pred_begin(), Block.pred_end(), &Block))
    != Block.pred_end();
  if (IsSelfPred) {
    S = llvm::make_unique<BlockWaitcntBrackets>(*ScoreBrackets);
    ScoreBrackets = S.get();
  }

  ScoreBrackets->clear();

  // See if there are any uninitialized predecessors. If so, emit an
  // s_waitcnt 0 at the beginning of the block.
  for (MachineBasicBlock *Pred : Block.predecessors()) {
    BlockWaitcntBrackets *PredScoreBrackets =
        BlockWaitcntBracketsMap[Pred].get();
    bool Visited = BlockVisitedSet.count(Pred);
    if (!Visited)
      continue;
    for (auto T : inst_counter_types()) {
      uint32_t span =
          PredScoreBrackets->getScoreUB(T) - PredScoreBrackets->getScoreLB(T);
      MaxPending[T] = std::max(MaxPending[T], span);
      span =
          PredScoreBrackets->pendingFlat(T) - PredScoreBrackets->getScoreLB(T);
      MaxFlat[T] = std::max(MaxFlat[T], span);
    }
  }

  // Now set the current Block's brackets to the largest ending bracket.
  for (auto T : inst_counter_types()) {
    ScoreBrackets->setScoreUB(T, MaxPending[T]);
    ScoreBrackets->setScoreLB(T, 0);
    ScoreBrackets->setLastFlat(T, MaxFlat[T]);
  }

  // Set the register scoreboard.
  for (MachineBasicBlock *Pred : Block.predecessors()) {
    if (!BlockVisitedSet.count(Pred)) {
      continue;
    }

    BlockWaitcntBrackets *PredScoreBrackets =
        BlockWaitcntBracketsMap[Pred].get();

    // Now merge the gpr_reg_score information
    for (auto T : inst_counter_types()) {
      uint32_t PredLB = PredScoreBrackets->getScoreLB(T);
      uint32_t PredUB = PredScoreBrackets->getScoreUB(T);
      if (PredLB < PredUB) {
        uint32_t PredScale = MaxPending[T] - PredUB;
        // Merge vgpr scores.
        for (int J = 0; J <= PredScoreBrackets->getMaxVGPR(); J++) {
          uint32_t PredRegScore = PredScoreBrackets->getRegScore(J, T);
          if (PredRegScore <= PredLB)
            continue;
          uint32_t NewRegScore = PredScale + PredRegScore;
          ScoreBrackets->setRegScore(
              J, T, std::max(ScoreBrackets->getRegScore(J, T), NewRegScore));
        }
        // Also need to merge sgpr scores for lgkm_cnt.
        if (T == LGKM_CNT) {
          for (int J = 0; J <= PredScoreBrackets->getMaxSGPR(); J++) {
            uint32_t PredRegScore =
                PredScoreBrackets->getRegScore(J + NUM_ALL_VGPRS, LGKM_CNT);
            if (PredRegScore <= PredLB)
              continue;
            uint32_t NewRegScore = PredScale + PredRegScore;
            ScoreBrackets->setRegScore(
                J + NUM_ALL_VGPRS, LGKM_CNT,
                std::max(
                    ScoreBrackets->getRegScore(J + NUM_ALL_VGPRS, LGKM_CNT),
                    NewRegScore));
          }
        }
      }
    }

    ScoreBrackets->mergePendingEvents(*PredScoreBrackets);
  }

  // if a single block loop, update the score brackets. Not needed for other
  // blocks, as we did this in-place
  if (IsSelfPred) {
    BlockWaitcntBracketsMap[&Block] = llvm::make_unique<BlockWaitcntBrackets>(*ScoreBrackets);
  }
}

/// Return true if the given basic block is a "bottom" block of a loop.
/// This works even if the loop is discontiguous. This also handles
/// multiple back-edges for the same "header" block of a loop.
bool SIInsertWaitcnts::isLoopBottom(const MachineLoop *Loop,
                                    const MachineBasicBlock *Block) {
  for (MachineBasicBlock *MBB : Loop->blocks()) {
    if (MBB == Block && MBB->isSuccessor(Loop->getHeader())) {
      return true;
    }
  }
  return false;
}

/// Count the number of "bottom" basic blocks of a loop.
unsigned SIInsertWaitcnts::countNumBottomBlocks(const MachineLoop *Loop) {
  unsigned Count = 0;
  for (MachineBasicBlock *MBB : Loop->blocks()) {
    if (MBB->isSuccessor(Loop->getHeader())) {
      Count++;
    }
  }
  return Count;
}

// Generate s_waitcnt instructions where needed.
void SIInsertWaitcnts::insertWaitcntInBlock(MachineFunction &MF,
                                            MachineBasicBlock &Block) {
  // Initialize the state information.
  mergeInputScoreBrackets(Block);

  BlockWaitcntBrackets *ScoreBrackets = BlockWaitcntBracketsMap[&Block].get();

  LLVM_DEBUG({
    dbgs() << "*** Block" << Block.getNumber() << " ***";
    ScoreBrackets->dump();
  });

  // Walk over the instructions.
  MachineInstr *OldWaitcntInstr = nullptr;

  for (MachineBasicBlock::iterator Iter = Block.begin(), E = Block.end();
       Iter != E;) {
    MachineInstr &Inst = *Iter;

    // Remove any previously existing waitcnts.
    if (Inst.getOpcode() == AMDGPU::S_WAITCNT) {
      if (OldWaitcntInstr) {
        if (TrackedWaitcntSet.count(OldWaitcntInstr)) {
          TrackedWaitcntSet.erase(OldWaitcntInstr);
          OldWaitcntInstr->eraseFromParent();
          OldWaitcntInstr = nullptr;
        } else if (!TrackedWaitcntSet.count(&Inst)) {
          // Two successive s_waitcnt's, both of which are pre-existing and
          // are therefore preserved.
          int64_t Imm = OldWaitcntInstr->getOperand(0).getImm();
          ScoreBrackets->applyWaitcnt(AMDGPU::decodeWaitcnt(IV, Imm));
        } else {
          ++Iter;
          Inst.eraseFromParent();
          continue;
        }
      }

      OldWaitcntInstr = &Inst;
      ++Iter;
      continue;
    }

    bool VCCZBugWorkAround = false;
    if (readsVCCZ(Inst) &&
        (!VCCZBugHandledSet.count(&Inst))) {
      if (ScoreBrackets->getScoreLB(LGKM_CNT) <
              ScoreBrackets->getScoreUB(LGKM_CNT) &&
          ScoreBrackets->hasPendingEvent(SMEM_ACCESS)) {
        if (ST->getGeneration() <= AMDGPUSubtarget::SEA_ISLANDS)
          VCCZBugWorkAround = true;
      }
    }

    // Generate an s_waitcnt instruction to be placed before
    // cur_Inst, if needed.
    generateWaitcntInstBefore(Inst, ScoreBrackets, OldWaitcntInstr);
    OldWaitcntInstr = nullptr;

    updateEventWaitcntAfter(Inst, ScoreBrackets);

#if 0 // TODO: implement resource type check controlled by options with ub = LB.
    // If this instruction generates a S_SETVSKIP because it is an
    // indexed resource, and we are on Tahiti, then it will also force
    // an S_WAITCNT vmcnt(0)
    if (RequireCheckResourceType(Inst, context)) {
      // Force the score to as if an S_WAITCNT vmcnt(0) is emitted.
      ScoreBrackets->setScoreLB(VM_CNT,
      ScoreBrackets->getScoreUB(VM_CNT));
    }
#endif

    LLVM_DEBUG({
      Inst.print(dbgs());
      ScoreBrackets->dump();
    });

    // Check to see if this is a GWS instruction. If so, and if this is CI or
    // VI, then the generated code sequence will include an S_WAITCNT 0.
    // TODO: Are these the only GWS instructions?
    if (Inst.getOpcode() == AMDGPU::DS_GWS_INIT ||
        Inst.getOpcode() == AMDGPU::DS_GWS_SEMA_V ||
        Inst.getOpcode() == AMDGPU::DS_GWS_SEMA_BR ||
        Inst.getOpcode() == AMDGPU::DS_GWS_SEMA_P ||
        Inst.getOpcode() == AMDGPU::DS_GWS_BARRIER) {
      // TODO: && context->target_info->GwsRequiresMemViolTest() ) {
      ScoreBrackets->applyWaitcnt(AMDGPU::Waitcnt::allZero());
    }

    // TODO: Remove this work-around after fixing the scheduler and enable the
    // assert above.
    if (VCCZBugWorkAround) {
      // Restore the vccz bit.  Any time a value is written to vcc, the vcc
      // bit is updated, so we can restore the bit by reading the value of
      // vcc and then writing it back to the register.
      BuildMI(Block, Inst, Inst.getDebugLoc(), TII->get(AMDGPU::S_MOV_B64),
              AMDGPU::VCC)
          .addReg(AMDGPU::VCC);
      VCCZBugHandledSet.insert(&Inst);
    }

    ++Iter;
  }

  // Check if we need to force convergence at loop footer.
  MachineLoop *ContainingLoop = MLI->getLoopFor(&Block);
  if (ContainingLoop && isLoopBottom(ContainingLoop, &Block)) {
    LoopWaitcntData *WaitcntData = LoopWaitcntDataMap[ContainingLoop].get();
    WaitcntData->print();
    LLVM_DEBUG(dbgs() << '\n';);

    // The iterative waitcnt insertion algorithm aims for optimal waitcnt
    // placement, but doesn't guarantee convergence for a loop. Each
    // loop should take at most (n+1) iterations for it to converge naturally,
    // where n is the number of bottom blocks. If this threshold is reached and
    // the result hasn't converged, then we force convergence by inserting
    // a s_waitcnt at the end of loop footer.
    if (WaitcntData->getIterCnt() > (countNumBottomBlocks(ContainingLoop) + 1)) {
      // To ensure convergence, need to make wait events at loop footer be no
      // more than those from the previous iteration.
      // As a simplification, instead of tracking individual scores and
      // generating the precise wait count, just wait on 0.
      bool HasPending = false;
      MachineInstr *SWaitInst = WaitcntData->getWaitcnt();
      for (auto T : inst_counter_types()) {
        if (ScoreBrackets->getScoreUB(T) > ScoreBrackets->getScoreLB(T)) {
          ScoreBrackets->setScoreLB(T, ScoreBrackets->getScoreUB(T));
          HasPending = true;
          break;
        }
      }

      if (HasPending) {
        if (!SWaitInst) {
          SWaitInst = BuildMI(Block, Block.getFirstNonPHI(),
                              DebugLoc(), TII->get(AMDGPU::S_WAITCNT))
                              .addImm(0);
          TrackedWaitcntSet.insert(SWaitInst);
#if 0 // TODO: Format the debug output
          OutputTransformBanner("insertWaitcntInBlock",0,"Create:",context);
          OutputTransformAdd(SWaitInst, context);
#endif
        }
#if 0 // TODO: ??
        _DEV( REPORTED_STATS->force_waitcnt_converge = 1; )
#endif
      }

      if (SWaitInst) {
        LLVM_DEBUG({
          SWaitInst->print(dbgs());
          dbgs() << "\nAdjusted score board:";
          ScoreBrackets->dump();
        });

        // Add this waitcnt to the block. It is either newly created or
        // created in previous iterations and added back since block traversal
        // always removes waitcnts.
        insertWaitcntBeforeCF(Block, SWaitInst);
        WaitcntData->setWaitcnt(SWaitInst);
      }
    }
  }
}

bool SIInsertWaitcnts::runOnMachineFunction(MachineFunction &MF) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  TII = ST->getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();
  MLI = &getAnalysis<MachineLoopInfo>();
  IV = AMDGPU::getIsaVersion(ST->getCPU());
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  ForceEmitZeroWaitcnts = ForceEmitZeroFlag;
  for (auto T : inst_counter_types())
    ForceEmitWaitcnt[T] = false;

  HardwareLimits.VmcntMax = AMDGPU::getVmcntBitMask(IV);
  HardwareLimits.ExpcntMax = AMDGPU::getExpcntBitMask(IV);
  HardwareLimits.LgkmcntMax = AMDGPU::getLgkmcntBitMask(IV);

  HardwareLimits.NumVGPRsMax = ST->getAddressableNumVGPRs();
  HardwareLimits.NumSGPRsMax = ST->getAddressableNumSGPRs();
  assert(HardwareLimits.NumVGPRsMax <= SQ_MAX_PGM_VGPRS);
  assert(HardwareLimits.NumSGPRsMax <= SQ_MAX_PGM_SGPRS);

  RegisterEncoding.VGPR0 = TRI->getEncodingValue(AMDGPU::VGPR0);
  RegisterEncoding.VGPRL =
      RegisterEncoding.VGPR0 + HardwareLimits.NumVGPRsMax - 1;
  RegisterEncoding.SGPR0 = TRI->getEncodingValue(AMDGPU::SGPR0);
  RegisterEncoding.SGPRL =
      RegisterEncoding.SGPR0 + HardwareLimits.NumSGPRsMax - 1;

  TrackedWaitcntSet.clear();
  BlockVisitedSet.clear();
  VCCZBugHandledSet.clear();
  LoopWaitcntDataMap.clear();
  BlockWaitcntProcessedSet.clear();

  // Walk over the blocks in reverse post order, inserting
  // s_waitcnt where needed.
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  bool Modified = false;
  for (ReversePostOrderTraversal<MachineFunction *>::rpo_iterator
           I = RPOT.begin(),
           E = RPOT.end(), J = RPOT.begin();
       I != E;) {
    MachineBasicBlock &MBB = **I;

    BlockVisitedSet.insert(&MBB);

    BlockWaitcntBrackets *ScoreBrackets = BlockWaitcntBracketsMap[&MBB].get();
    if (!ScoreBrackets) {
      BlockWaitcntBracketsMap[&MBB] = llvm::make_unique<BlockWaitcntBrackets>(ST);
      ScoreBrackets = BlockWaitcntBracketsMap[&MBB].get();
    }
    ScoreBrackets->setPostOrder(MBB.getNumber());
    MachineLoop *ContainingLoop = MLI->getLoopFor(&MBB);
    if (ContainingLoop && LoopWaitcntDataMap[ContainingLoop] == nullptr)
      LoopWaitcntDataMap[ContainingLoop] = llvm::make_unique<LoopWaitcntData>();

    // If we are walking into the block from before the loop, then guarantee
    // at least 1 re-walk over the loop to propagate the information, even if
    // no S_WAITCNT instructions were generated.
    if (ContainingLoop && ContainingLoop->getHeader() == &MBB) {
      unsigned Count = countNumBottomBlocks(ContainingLoop);

      // If the loop has multiple back-edges, and so more than one "bottom"
      // basic block, we have to guarantee a re-walk over every blocks.
      if ((std::count(BlockWaitcntProcessedSet.begin(),
                      BlockWaitcntProcessedSet.end(), &MBB) < (int)Count)) {
        BlockWaitcntBracketsMap[&MBB]->setRevisitLoop(true);
        LLVM_DEBUG(dbgs() << "set-revisit1: Block"
                          << ContainingLoop->getHeader()->getNumber() << '\n';);
      }
    }

    // Walk over the instructions.
    insertWaitcntInBlock(MF, MBB);

    // Record that waitcnts have been processed at least once for this block.
    BlockWaitcntProcessedSet.push_back(&MBB);

    // See if we want to revisit the loop. If a loop has multiple back-edges,
    // we shouldn't revisit the same "bottom" basic block.
    if (ContainingLoop && isLoopBottom(ContainingLoop, &MBB) &&
        std::count(BlockWaitcntProcessedSet.begin(),
                   BlockWaitcntProcessedSet.end(), &MBB) == 1) {
      MachineBasicBlock *EntryBB = ContainingLoop->getHeader();
      BlockWaitcntBrackets *EntrySB = BlockWaitcntBracketsMap[EntryBB].get();
      if (EntrySB && EntrySB->getRevisitLoop()) {
        EntrySB->setRevisitLoop(false);
        J = I;
        int32_t PostOrder = EntrySB->getPostOrder();
        // TODO: Avoid this loop. Find another way to set I.
        for (ReversePostOrderTraversal<MachineFunction *>::rpo_iterator
                 X = RPOT.begin(),
                 Y = RPOT.end();
             X != Y; ++X) {
          MachineBasicBlock &MBBX = **X;
          if (MBBX.getNumber() == PostOrder) {
            I = X;
            break;
          }
        }
        LoopWaitcntData *WaitcntData = LoopWaitcntDataMap[ContainingLoop].get();
        WaitcntData->incIterCnt();
        LLVM_DEBUG(dbgs() << "revisit: Block" << EntryBB->getNumber() << '\n';);
        continue;
      } else {
        LoopWaitcntData *WaitcntData = LoopWaitcntDataMap[ContainingLoop].get();
        // Loop converged, reset iteration count. If this loop gets revisited,
        // it must be from an outer loop, the counter will restart, this will
        // ensure we don't force convergence on such revisits.
        WaitcntData->resetIterCnt();
      }
    }

    J = I;
    ++I;
  }

  SmallVector<MachineBasicBlock *, 4> EndPgmBlocks;

  bool HaveScalarStores = false;

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end(); BI != BE;
       ++BI) {
    MachineBasicBlock &MBB = *BI;

    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;
         ++I) {
      if (!HaveScalarStores && TII->isScalarStore(*I))
        HaveScalarStores = true;

      if (I->getOpcode() == AMDGPU::S_ENDPGM ||
          I->getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG)
        EndPgmBlocks.push_back(&MBB);
    }
  }

  if (HaveScalarStores) {
    // If scalar writes are used, the cache must be flushed or else the next
    // wave to reuse the same scratch memory can be clobbered.
    //
    // Insert s_dcache_wb at wave termination points if there were any scalar
    // stores, and only if the cache hasn't already been flushed. This could be
    // improved by looking across blocks for flushes in postdominating blocks
    // from the stores but an explicitly requested flush is probably very rare.
    for (MachineBasicBlock *MBB : EndPgmBlocks) {
      bool SeenDCacheWB = false;

      for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
           ++I) {
        if (I->getOpcode() == AMDGPU::S_DCACHE_WB)
          SeenDCacheWB = true;
        else if (TII->isScalarStore(*I))
          SeenDCacheWB = false;

        // FIXME: It would be better to insert this before a waitcnt if any.
        if ((I->getOpcode() == AMDGPU::S_ENDPGM ||
             I->getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG) &&
            !SeenDCacheWB) {
          Modified = true;
          BuildMI(*MBB, I, I->getDebugLoc(), TII->get(AMDGPU::S_DCACHE_WB));
        }
      }
    }
  }

  if (!MFI->isEntryFunction()) {
    // Wait for any outstanding memory operations that the input registers may
    // depend on. We can't track them and it's better to the wait after the
    // costly call sequence.

    // TODO: Could insert earlier and schedule more liberally with operations
    // that only use caller preserved registers.
    MachineBasicBlock &EntryBB = MF.front();
    BuildMI(EntryBB, EntryBB.getFirstNonPHI(), DebugLoc(), TII->get(AMDGPU::S_WAITCNT))
      .addImm(0);

    Modified = true;
  }

  return Modified;
}
