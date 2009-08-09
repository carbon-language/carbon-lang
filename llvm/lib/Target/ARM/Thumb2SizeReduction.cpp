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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumNarrows,  "Number of 32-bit instructions reduced to 16-bit ones");
STATISTIC(Num2Addrs,   "Number of 32-bit instructions reduced to 2-address");

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
    // Wide,        Narrow1,      Narrow2,      mm1, imm2, lo1, lo2, P/C, S
    { ARM::t2ADCrr, ARM::tADC,    0,             0,   0,    1,   0,  0,0, 0 },
    { ARM::t2ADDri, ARM::tADDi3,  ARM::tADDi8,   3,   8,    1,   1,  0,0, 0 },
    { ARM::t2ADDrr, ARM::tADDrr,  ARM::tADDhirr, 0,   0,    1,   0,  0,1, 0 },
    { ARM::t2ANDrr, ARM::tAND,    0,             0,   0,    1,   0,  0,0, 0 },
    { ARM::t2ASRri, ARM::tASRri,  0,             5,   0,    1,   1,  0,0, 0 },
    { ARM::t2ASRrr, ARM::tASRrr,  0,             0,   0,    1,   0,  0,0, 0 },
    { ARM::t2BICrr, ARM::tBIC,    0,             0,   0,    1,   0,  0,0, 0 },
    { ARM::t2CMNrr, ARM::tCMN,    0,             0,   0,    1,   0,  1,0, 0 }
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
    bool ReduceTo2Addr(MachineBasicBlock &MBB, MachineInstr &MI,
                       const ReduceEntry &Entry);

    /// ReduceToNarrow - Reduce a 32-bit instruction to a 16-bit
    /// non-two-address instruction.
    bool ReduceToNarrow(MachineBasicBlock &MBB, MachineInstr &MI,
                        const ReduceEntry &Entry);


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

static bool VerifyPredAndCC(MachineInstr &MI, const ReduceEntry &Entry,
                            bool is2Addr, bool &HasCC) {
  const TargetInstrDesc &TID = MI.getDesc();

  // Most thumb1 instructions either can be predicated or set CPSR.
  HasCC = false;
  if (TID.hasOptionalDef()) {
    unsigned NumOps = TID.getNumOperands();
    HasCC = (MI.getOperand(NumOps-1).getReg() == ARM::CPSR);
  }

  unsigned PredReg = 0;
  ARMCC::CondCodes Pred = getInstrPredicate(&MI, PredReg);
  if ((is2Addr  && Entry.PredCC2 == 0) ||
      (!is2Addr && Entry.PredCC1 == 0)) {
    if (Pred == ARMCC::AL) {
      // Not predicated, must set CPSR.
      if (!HasCC) return false;
    } else {
      // Predicated, must not set CPSR.
      if (HasCC) return false;
    }
  } else {
    if (HasCC) return false;
  }

  return true;
}

bool
Thumb2SizeReduce::ReduceTo2Addr(MachineBasicBlock &MBB, MachineInstr &MI,
                                const ReduceEntry &Entry) {
  const TargetInstrDesc &TID = MI.getDesc();
  unsigned Reg0 = MI.getOperand(0).getReg();
  unsigned Reg1 = MI.getOperand(1).getReg();
  if (Reg0 != Reg1)
    return false;
  if (Entry.LowRegs2 && !isARMLowRegister(Reg0))
    return false;
  if (Entry.Imm2Limit) {
    unsigned Imm = MI.getOperand(2).getImm();
    unsigned Limit = (1 << Entry.Imm2Limit) - 1;
    if (Imm > Limit)
      return false;
  } else {
    unsigned Reg2 = MI.getOperand(2).getReg();
    if (Entry.LowRegs2 && !isARMLowRegister(Reg2))
      return false;
  }

  bool HasCC = false;
  if (!VerifyPredAndCC(MI, Entry, true, HasCC))
    return false;

  // Add the 16-bit instruction.
  DebugLoc dl = MI.getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(MBB, MI, dl, TII->get(Entry.NarrowOpc2));
  MIB.addOperand(MI.getOperand(0));
  if (HasCC)
    AddDefaultT1CC(MIB);

  // Transfer the rest of operands.
  unsigned NumOps = TID.getNumOperands();
  for (unsigned i = 1, e = MI.getNumOperands(); i != e; ++i)
    if (!(i < NumOps && TID.OpInfo[i].isOptionalDef()))
      MIB.addOperand(MI.getOperand(i));

  DOUT << "Converted 32-bit: " << MI << "       to 16-bit: " << *MIB;

  MBB.erase(MI);
  ++Num2Addrs;
  ++NumNarrows;
  return true;
}

bool
Thumb2SizeReduce::ReduceToNarrow(MachineBasicBlock &MBB, MachineInstr &MI,
                                 const ReduceEntry &Entry) {
  unsigned Limit = ~0U;
  if (Entry.Imm1Limit)
    Limit = (1 << Entry.Imm1Limit) - 1;

  const TargetInstrDesc &TID = MI.getDesc();
  for (unsigned i = 0, e = TID.getNumOperands(); i != e; ++i) {
    if (TID.OpInfo[i].isPredicate())
      continue;
    const MachineOperand &MO = MI.getOperand(i);
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

  bool HasCC = false;
  if (!VerifyPredAndCC(MI, Entry, false, HasCC))
    return false;

  // Add the 16-bit instruction.
  DebugLoc dl = MI.getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(MBB, MI, dl, TII->get(Entry.NarrowOpc1));
  MIB.addOperand(MI.getOperand(0));
  if (HasCC)
    AddDefaultT1CC(MIB);

  // Transfer the rest of operands.
  unsigned NumOps = TID.getNumOperands();
  for (unsigned i = 1, e = MI.getNumOperands(); i != e; ++i)
    if (!(i < NumOps && TID.OpInfo[i].isOptionalDef()))
      MIB.addOperand(MI.getOperand(i));


  DOUT << "Converted 32-bit: " << MI << "       to 16-bit: " << *MIB;

  MBB.erase(MI);
  ++Num2Addrs;
  ++NumNarrows;
  return true;
}

bool Thumb2SizeReduce::ReduceMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  // FIXME: Track whether CPSR is live. If not, then it's possible to convert
  // one that doesn't set CPSR to one that does.
  MachineBasicBlock::iterator MII = MBB.begin(), E = MBB.end();
  MachineBasicBlock::iterator NextMII = next(MII);
  for (; MII != E; MII = NextMII) {
    NextMII = next(MII);

    MachineInstr &MI = *MII;
    unsigned Opcode = MI.getOpcode();
    DenseMap<unsigned, unsigned>::iterator OPI = ReduceOpcodeMap.find(Opcode);
    if (OPI == ReduceOpcodeMap.end())
      continue;

    const ReduceEntry &Entry = ReduceTable[OPI->second];
    // Ignore "special" cases for now.
    if (Entry.Special)
      continue;

    // Try to transform to a 16-bit two-address instruction.
    if (Entry.NarrowOpc2 && ReduceTo2Addr(MBB, MI, Entry)) {
      Modified = true;
      continue;
    }

    // Try to transform ro a 16-bit non-two-address instruction.
    if (ReduceToNarrow(MBB, MI, Entry)) {
      Modified = true;
      continue;
    }
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
