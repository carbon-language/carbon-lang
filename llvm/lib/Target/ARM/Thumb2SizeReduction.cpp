//===-- Thumb2SizeReduction.cpp - Thumb2 code size reduction pass -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "t2-reduce-size"
#include "ARM.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMBaseInstrInfo.h"
#include "Thumb2InstrInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumNarrows,  "Number of 32-bit instrs reduced to 16-bit ones");
STATISTIC(Num2Addrs,   "Number of 32-bit instrs reduced to 2addr 16-bit ones");

static cl::opt<int> ReduceLimit("t2-reduce-limit", cl::init(-1), cl::Hidden);

namespace {
  /// ReduceTable - A static table with information on mapping from wide
  /// opcodes to narrow
  struct ReduceEntry {
    unsigned WideOpc;      // Wide opcode
    unsigned NarrowOpc1;   // Narrow opcode to transform to
    unsigned NarrowOpc2;   // Narrow opcode when it's two-address
    uint8_t  Imm1Limit;    // Limit of immediate field (bits)
    uint8_t  Imm2Limit;    // Limit of immediate field when it's two-address
    unsigned LowRegs1 : 1; // Only possible if low-registers are used
    unsigned LowRegs2 : 1; // Only possible if low-registers are used (2addr)
    unsigned PredCC1  : 1; // 0 - If predicated, cc is on and vice versa.
                           // 1 - No cc field.
    unsigned PredCC2  : 1;
    unsigned Special  : 1; // Needs to be dealt with specially
  };

  static const ReduceEntry ReduceTable[] = {
    // Wide,        Narrow1,      Narrow2,     imm1,imm2,  lo1, lo2, P/C, S
    { ARM::t2ADCrr, ARM::tADC,    0,             0,   0,    1,   0,  0,0, 0 },
    // FIXME: t2ADDS variants.
    { ARM::t2ADDri, ARM::tADDi3,  ARM::tADDi8,   3,   8,    1,   1,  0,0, 0 },
    { ARM::t2ADDrr, ARM::tADDrr,  ARM::tADDhirr, 0,   0,    1,   0,  0,1, 0 },
    { ARM::t2ANDrr, 0,            ARM::tAND,     0,   0,    0,   1,  0,0, 0 },
    { ARM::t2ASRri, ARM::tASRri,  0,             5,   0,    1,   0,  0,0, 0 },
    { ARM::t2ASRrr, 0,            ARM::tASRrr,   0,   0,    0,   1,  0,0, 0 },
    { ARM::t2BICrr, 0,            ARM::tBIC,     0,   0,    0,   1,  0,0, 0 },
    { ARM::t2CMNrr, ARM::tCMN,    0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2CMPri, ARM::tCMPi8,  0,             8,   0,    1,   0,  1,0, 0 },
    { ARM::t2CMPrr, ARM::tCMPhir, 0,             0,   0,    0,   0,  1,0, 0 },
    { ARM::t2CMPzri,ARM::tCMPzi8, 0,             8,   0,    1,   0,  1,0, 0 },
    { ARM::t2CMPzrr,ARM::tCMPzhir,0,             0,   0,    0,   0,  1,0, 0 },
    { ARM::t2EORrr, 0,            ARM::tEOR,     0,   0,    0,   1,  0,0, 0 },
    { ARM::t2LSLri, ARM::tLSLri,  0,             5,   0,    1,   0,  0,0, 0 },
    { ARM::t2LSLrr, 0,            ARM::tLSLrr,   0,   0,    0,   1,  0,0, 0 },
    { ARM::t2LSRri, ARM::tLSRri,  0,             5,   0,    1,   0,  0,0, 0 },
    { ARM::t2LSRrr, 0,            ARM::tLSRrr,   0,   0,    0,   1,  0,0, 0 },
    { ARM::t2MOVi,  ARM::tMOVi8,  0,             8,   0,    1,   0,  0,0, 0 },
    // FIXME: Do we need the 16-bit 'S' variant?
    // FIXME: t2MOVcc
    { ARM::t2MOVr,ARM::tMOVgpr2gpr,0,            0,   0,    0,   0,  1,0, 0 },
    { ARM::t2MUL,   0,            ARM::tMUL,     0,   0,    0,   1,  0,0, 0 },
    { ARM::t2MVNr,  ARM::tMVN,    0,             0,   0,    1,   0,  0,0, 0 },
    { ARM::t2ORRrr, 0,            ARM::tORR,     0,   0,    0,   1,  0,0, 0 },
    { ARM::t2REV,   ARM::tREV,    0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2REV16, ARM::tREV16,  0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2REVSH, ARM::tREVSH,  0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2RORrr, 0,            ARM::tROR,     0,   0,    0,   1,  0,0, 0 },
    // FIXME: T2RSBri immediate must be zero. Also need entry for T2RSBS
    //{ ARM::t2RSBri, ARM::tRSB,    0,             0,   0,    1,   0,  0,0, 0 },
    { ARM::t2SUBri, ARM::tSUBi3,  ARM::tSUBi8,   3,   8,    1,   1,  0,0, 0 },
    { ARM::t2SUBrr, ARM::tSUBrr,  0,             0,   0,    1,   0,  0,0, 0 },
    { ARM::t2SXTBr, ARM::tSXTB,   0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2SXTHr, ARM::tSXTH,   0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2TSTrr, ARM::tTST,    0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2UXTBr, ARM::tUXTB,   0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2UXTHr, ARM::tUXTH,   0,             0,   0,    1,   0,  1,0, 0 }
  };

  class VISIBILITY_HIDDEN Thumb2SizeReduce : public MachineFunctionPass {
  public:
    static char ID;
    Thumb2SizeReduce();

    const TargetInstrInfo *TII;

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "Thumb2 instruction size reduction pass";
    }

  private:
    /// ReduceOpcodeMap - Maps wide opcode to index of entry in ReduceTable.
    DenseMap<unsigned, unsigned> ReduceOpcodeMap;

    /// ReduceTo2Addr - Reduce a 32-bit instruction to a 16-bit two-address
    /// instruction.
    bool ReduceTo2Addr(MachineBasicBlock &MBB, MachineInstr *MI,
                       const ReduceEntry &Entry,
                       bool LiveCPSR);

    /// ReduceToNarrow - Reduce a 32-bit instruction to a 16-bit
    /// non-two-address instruction.
    bool ReduceToNarrow(MachineBasicBlock &MBB, MachineInstr *MI,
                        const ReduceEntry &Entry,
                        bool LiveCPSR);

    /// ReduceMBB - Reduce width of instructions in the specified basic block.
    bool ReduceMBB(MachineBasicBlock &MBB);
  };
  char Thumb2SizeReduce::ID = 0;
}

Thumb2SizeReduce::Thumb2SizeReduce() : MachineFunctionPass(&ID) {
  for (unsigned i = 0, e = array_lengthof(ReduceTable); i != e; ++i) {
    unsigned FromOpc = ReduceTable[i].WideOpc;
    if (!ReduceOpcodeMap.insert(std::make_pair(FromOpc, i)).second)
      assert(false && "Duplicated entries?");
  }
}

static bool VerifyPredAndCC(MachineInstr *MI, const ReduceEntry &Entry,
                            bool is2Addr, ARMCC::CondCodes Pred,
                            bool LiveCPSR, bool &HasCC, bool &CCDead) {
  if ((is2Addr  && Entry.PredCC2 == 0) ||
      (!is2Addr && Entry.PredCC1 == 0)) {
    if (Pred == ARMCC::AL) {
      // Not predicated, must set CPSR.
      if (!HasCC) {
        // Original instruction was not setting CPSR, but CPSR is not
        // currently live anyway. It's ok to set it. The CPSR def is
        // dead though.
        if (!LiveCPSR) {
          HasCC = true;
          CCDead = true;
          return true;
        }
        return false;
      }
    } else {
      // Predicated, must not set CPSR.
      if (HasCC)
        return false;
    }
  } else {
    // 16-bit instruction does not set CPSR.
    if (HasCC)
      return false;
  }

  return true;
}

bool
Thumb2SizeReduce::ReduceTo2Addr(MachineBasicBlock &MBB, MachineInstr *MI,
                                const ReduceEntry &Entry,
                                bool LiveCPSR) {
  const TargetInstrDesc &TID = MI->getDesc();
  unsigned Reg0 = MI->getOperand(0).getReg();
  unsigned Reg1 = MI->getOperand(1).getReg();
  if (Reg0 != Reg1)
    return false;
  if (Entry.LowRegs2 && !isARMLowRegister(Reg0))
    return false;
  if (Entry.Imm2Limit) {
    unsigned Imm = MI->getOperand(2).getImm();
    unsigned Limit = (1 << Entry.Imm2Limit) - 1;
    if (Imm > Limit)
      return false;
  } else {
    unsigned Reg2 = MI->getOperand(2).getReg();
    if (Entry.LowRegs2 && !isARMLowRegister(Reg2))
      return false;
  }

  // Check if it's possible / necessary to transfer the predicate.
  const TargetInstrDesc &NewTID = TII->get(Entry.NarrowOpc2);
  unsigned PredReg = 0;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);
  bool SkipPred = false;
  if (Pred != ARMCC::AL) {
    if (!NewTID.isPredicable())
      // Can't transfer predicate, fail.
      return false;
  } else {
    SkipPred = !NewTID.isPredicable();
  }

  bool HasCC = false;
  bool CCDead = false;
  if (TID.hasOptionalDef()) {
    unsigned NumOps = TID.getNumOperands();
    HasCC = (MI->getOperand(NumOps-1).getReg() == ARM::CPSR);
    if (HasCC && MI->getOperand(NumOps-1).isDead())
      CCDead = true;
  }
  if (!VerifyPredAndCC(MI, Entry, true, Pred, LiveCPSR, HasCC, CCDead))
    return false;

  // Add the 16-bit instruction.
  DebugLoc dl = MI->getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(MBB, *MI, dl, TII->get(Entry.NarrowOpc2));
  MIB.addOperand(MI->getOperand(0));
  if (HasCC)
    AddDefaultT1CC(MIB, CCDead);

  // Transfer the rest of operands.
  unsigned NumOps = TID.getNumOperands();
  for (unsigned i = 1, e = MI->getNumOperands(); i != e; ++i) {
    if (i < NumOps && TID.OpInfo[i].isOptionalDef())
      continue;
    if (SkipPred && TID.OpInfo[i].isPredicate())
      continue;
    MIB.addOperand(MI->getOperand(i));
  }

  DOUT << "Converted 32-bit: " << *MI << "       to 16-bit: " << *MIB;

  MBB.erase(MI);
  ++Num2Addrs;
  return true;
}

bool
Thumb2SizeReduce::ReduceToNarrow(MachineBasicBlock &MBB, MachineInstr *MI,
                                 const ReduceEntry &Entry,
                                 bool LiveCPSR) {
  unsigned Limit = ~0U;
  if (Entry.Imm1Limit)
    Limit = (1 << Entry.Imm1Limit) - 1;

  const TargetInstrDesc &TID = MI->getDesc();
  for (unsigned i = 0, e = TID.getNumOperands(); i != e; ++i) {
    if (TID.OpInfo[i].isPredicate())
      continue;
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg()) {
      unsigned Reg = MO.getReg();
      if (!Reg || Reg == ARM::CPSR)
        continue;
      if (Entry.LowRegs1 && !isARMLowRegister(Reg))
        return false;
    } else if (MO.isImm()) {
      if (MO.getImm() > Limit)
        return false;
    }
  }

  // Check if it's possible / necessary to transfer the predicate.
  const TargetInstrDesc &NewTID = TII->get(Entry.NarrowOpc1);
  unsigned PredReg = 0;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);
  bool SkipPred = false;
  if (Pred != ARMCC::AL) {
    if (!NewTID.isPredicable())
      // Can't transfer predicate, fail.
      return false;
  } else {
    SkipPred = !NewTID.isPredicable();
  }

  bool HasCC = false;
  bool CCDead = false;
  if (TID.hasOptionalDef()) {
    unsigned NumOps = TID.getNumOperands();
    HasCC = (MI->getOperand(NumOps-1).getReg() == ARM::CPSR);
    if (HasCC && MI->getOperand(NumOps-1).isDead())
      CCDead = true;
  }
  if (!VerifyPredAndCC(MI, Entry, false, Pred, LiveCPSR, HasCC, CCDead))
    return false;

  // Add the 16-bit instruction.
  DebugLoc dl = MI->getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(MBB, *MI, dl, TII->get(Entry.NarrowOpc1));
  MIB.addOperand(MI->getOperand(0));
  if (HasCC)
    AddDefaultT1CC(MIB, CCDead);

  // Transfer the rest of operands.
  unsigned NumOps = TID.getNumOperands();
  for (unsigned i = 1, e = MI->getNumOperands(); i != e; ++i) {
    if (i < NumOps && TID.OpInfo[i].isOptionalDef())
      continue;
    if (SkipPred && TID.OpInfo[i].isPredicate())
      continue;
    MIB.addOperand(MI->getOperand(i));
  }


  DOUT << "Converted 32-bit: " << *MI << "       to 16-bit: " << *MIB;

  MBB.erase(MI);
  ++NumNarrows;
  return true;
}

static bool UpdateCPSRLiveness(MachineInstr &MI, bool LiveCPSR) {
  bool HasDef = false;
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || MO.isUndef())
      continue;
    if (MO.getReg() != ARM::CPSR)
      continue;
    if (MO.isDef()) {
      if (!MO.isDead())
        HasDef = true;
      continue;
    }

    assert(LiveCPSR && "CPSR liveness tracking is wrong!");
    if (MO.isKill()) {
      LiveCPSR = false;
      break;
    }
  }

  return HasDef || LiveCPSR;
}

bool Thumb2SizeReduce::ReduceMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  bool LiveCPSR = false;
  // Yes, CPSR could be livein.
  for (MachineBasicBlock::const_livein_iterator I = MBB.livein_begin(),
         E = MBB.livein_end(); I != E; ++I) {
    if (*I == ARM::CPSR) {
      LiveCPSR = true;
      break;
    }
  }

  MachineBasicBlock::iterator MII = MBB.begin(), E = MBB.end();
  MachineBasicBlock::iterator NextMII = next(MII);
  for (; MII != E; MII = NextMII) {
    NextMII = next(MII);

    MachineInstr *MI = &*MII;
    unsigned Opcode = MI->getOpcode();
    DenseMap<unsigned, unsigned>::iterator OPI = ReduceOpcodeMap.find(Opcode);
    if (OPI != ReduceOpcodeMap.end()) {
      const ReduceEntry &Entry = ReduceTable[OPI->second];
      // Ignore "special" cases for now.
      if (Entry.Special)
        goto ProcessNext;

      // Try to transform to a 16-bit two-address instruction.
      if (Entry.NarrowOpc2 && ReduceTo2Addr(MBB, MI, Entry, LiveCPSR)) {
        Modified = true;
        MachineBasicBlock::iterator I = prior(NextMII);
        MI = &*I;
        goto ProcessNext;
      }

      // Try to transform ro a 16-bit non-two-address instruction.
      if (Entry.NarrowOpc1 && ReduceToNarrow(MBB, MI, Entry, LiveCPSR))
        Modified = true;
    }

  ProcessNext:
    LiveCPSR = UpdateCPSRLiveness(*MI, LiveCPSR);

    if (ReduceLimit != -1 && ((int)(NumNarrows + Num2Addrs) > ReduceLimit))
      break;
  }

  return Modified;
}

bool Thumb2SizeReduce::runOnMachineFunction(MachineFunction &MF) {
  const TargetMachine &TM = MF.getTarget();
  TII = TM.getInstrInfo();

  bool Modified = false;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    Modified |= ReduceMBB(*I);
  return Modified;
}

/// createThumb2SizeReductionPass - Returns an instance of the Thumb2 size
/// reduction pass.
FunctionPass *llvm::createThumb2SizeReductionPass() {
  return new Thumb2SizeReduce();
}
