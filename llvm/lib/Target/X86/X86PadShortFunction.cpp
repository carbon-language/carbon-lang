//===-------- X86PadShortFunction.cpp - pad short functions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass which will pad short functions to prevent
// a stall if a function returns before the return address is ready. This
// is needed for some Intel Atom processors.
//
//===----------------------------------------------------------------------===//

#include <map>
#include <algorithm>

#define DEBUG_TYPE "x86-pad-short-functions"
#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
using namespace llvm;

STATISTIC(NumBBsPadded, "Number of basic blocks padded");

namespace {
  struct PadShortFunc : public MachineFunctionPass {
    static char ID;
    PadShortFunc() : MachineFunctionPass(ID)
                   , Threshold(4)
    {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const
    {
      return "X86 Atom pad short functions";
    }

  private:
    bool addPadding(MachineFunction &MF,
                    MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator &MBBI,
                    unsigned int NOOPsToAdd);

    void findReturn(MachineFunction &MF,
                    MachineBasicBlock &MBB,
                    unsigned int Cycles);

    bool cyclesUntilReturn(MachineFunction &MF,
                        MachineBasicBlock &MBB,
                        unsigned int &Cycles,
                        MachineBasicBlock::iterator *Location = 0);

    const unsigned int Threshold;
    std::map<int, unsigned int> ReturnBBs;
  };

  char PadShortFunc::ID = 0;
}

FunctionPass *llvm::createX86PadShortFunctions() {
  return new PadShortFunc();
}

/// runOnMachineFunction - Loop over all of the basic blocks, inserting
/// NOOP instructions before early exits.
bool PadShortFunc::runOnMachineFunction(MachineFunction &MF) {
  // Process all basic blocks.
  ReturnBBs.clear();

  // Search through basic blocks and mark the ones that have early returns
  findReturn(MF, *MF.begin(), 0);

  int BBNum;
  MachineBasicBlock::iterator ReturnLoc;
  MachineBasicBlock *MBB;

  unsigned int Cycles = 0;
  unsigned int BBCycles;

  // Pad the identified basic blocks with NOOPs
  for (std::map<int, unsigned int>::iterator I = ReturnBBs.begin();
       I != ReturnBBs.end(); ++I) {
    BBNum = I->first;
    Cycles = I->second;

    if (Cycles < Threshold) {
      MBB = MF.getBlockNumbered(BBNum);
      if (!cyclesUntilReturn(MF, *MBB, BBCycles, &ReturnLoc))
        continue;

      addPadding(MF, *MBB, ReturnLoc, Threshold - Cycles);
      NumBBsPadded++;
    }
  }

  return false;
}

/// findReturn - Starting at MBB, follow control flow and add all
/// basic blocks that contain a return to ReturnBBs.
void PadShortFunc::findReturn(MachineFunction &MF,
                              MachineBasicBlock &MBB,
                              unsigned int Cycles)
{
  // If this BB has a return, note how many cycles it takes to get there.
  bool hasReturn = cyclesUntilReturn(MF, MBB, Cycles);
  if (Cycles >= Threshold)
    return;

  if (hasReturn) {
    int BBNum = MBB.getNumber();
    ReturnBBs[BBNum] = std::max(ReturnBBs[BBNum], Cycles);

    return;
  }

  // Follow branches in BB and look for returns
  for (MachineBasicBlock::succ_iterator I = MBB.succ_begin();
       I != MBB.succ_end(); ++I) {
    findReturn(MF, **I, Cycles);
  }
}

/// cyclesUntilReturn - if the MBB has a return instruction, set Location to
/// to the instruction and return true. Return false otherwise.
/// Cycles will be incremented by the number of cycles taken to reach the
/// return or the end of the BB, whichever occurs first.
bool PadShortFunc::cyclesUntilReturn(MachineFunction &MF,
                                  MachineBasicBlock &MBB,
                                  unsigned int &Cycles,
                                  MachineBasicBlock::iterator *Location)
{
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();
  const TargetMachine &Target = MF.getTarget();

  for (MachineBasicBlock::iterator MBBI = MBB.begin(); MBBI != MBB.end();
       ++MBBI) {
    MachineInstr *MI = MBBI;
    // Mark basic blocks with a return instruction. Calls to other functions
    // do not count because the called function will be padded, if necessary
    if (MI->isReturn() && !MI->isCall()) {
      if (Location)
        *Location = MBBI;
      return true;
    }

    Cycles += TII.getInstrLatency(Target.getInstrItineraryData(), MI);
  }

  return false;
}

/// addPadding - Add the given number of NOOP instructions to the function
/// right before the return at MBBI
bool PadShortFunc::addPadding(MachineFunction &MF,
                              MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator &MBBI,
                              unsigned int NOOPsToAdd)
{
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();

  DebugLoc DL = MBBI->getDebugLoc();

  while (NOOPsToAdd-- > 0) {
    // Since Atom has two instruction execution ports,
    // the code emits two noops, which will be executed in parallell
    // during one cycle.
    BuildMI(MBB, MBBI, DL, TII.get(X86::NOOP));
    BuildMI(MBB, MBBI, DL, TII.get(X86::NOOP));
  }

  return true;
}

