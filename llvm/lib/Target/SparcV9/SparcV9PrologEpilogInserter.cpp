//===-- SparcV9PrologEpilogCodeInserter.cpp - Insert Fn Prolog & Epilog ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the SparcV9 target's own PrologEpilogInserter. It creates prolog and
// epilog instructions for functions which have not been compiled using "leaf
// function optimizations". These instructions include the SAVE and RESTORE
// instructions used to rotate the SPARC register windows. Prologs are
// attached to the unique function entry, and epilogs are attached to each
// function exit.
//
//===----------------------------------------------------------------------===//

#include "SparcV9Internals.h"
#include "SparcV9RegClassInfo.h"
#include "SparcV9RegisterInfo.h"
#include "SparcV9FrameInfo.h"
#include "MachineFunctionInfo.h"
#include "MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Intrinsics.h"

namespace llvm {

namespace {
  struct InsertPrologEpilogCode : public MachineFunctionPass {
    const char *getPassName() const { return "SparcV9 Prolog/Epilog Inserter"; }

    bool runOnMachineFunction(MachineFunction &F) {
      if (!F.getInfo<SparcV9FunctionInfo>()->isCompiledAsLeafMethod()) {
        InsertPrologCode(F);
        InsertEpilogCode(F);
      }
      return false;
    }

    void InsertPrologCode(MachineFunction &F);
    void InsertEpilogCode(MachineFunction &F);
  };

}  // End anonymous namespace

static unsigned getStaticStackSize (MachineFunction &MF) {
  const TargetFrameInfo& frameInfo = *MF.getTarget().getFrameInfo();
  unsigned staticStackSize = MF.getInfo<SparcV9FunctionInfo>()->getStaticStackSize();
  if (staticStackSize < (unsigned)SparcV9FrameInfo::MinStackFrameSize)
    staticStackSize = SparcV9FrameInfo::MinStackFrameSize;
  if (unsigned padsz = staticStackSize %
                       SparcV9FrameInfo::StackFrameSizeAlignment)
    staticStackSize += SparcV9FrameInfo::StackFrameSizeAlignment - padsz;
  return staticStackSize;
}

void InsertPrologEpilogCode::InsertPrologCode(MachineFunction &MF)
{
  std::vector<MachineInstr*> mvec;
  const TargetMachine &TM = MF.getTarget();
  const TargetFrameInfo& frameInfo = *TM.getFrameInfo();

  // The second operand is the stack size. If it does not fit in the
  // immediate field, we have to use a free register to hold the size.
  // See the comments below for the choice of this register.
  unsigned staticStackSize = getStaticStackSize (MF);
  int32_t C = - (int) staticStackSize;
  int SP = TM.getRegInfo()->getStackPointer();
  if (TM.getInstrInfo()->constantFitsInImmedField(V9::SAVEi,staticStackSize)) {
    mvec.push_back(BuildMI(V9::SAVEi, 3).addMReg(SP).addSImm(C)
                   .addMReg(SP, MachineOperand::Def));
  } else {
    // We have to put the stack size value into a register before SAVE.
    // Use register %g1 since it is volatile across calls.  Note that the
    // local (%l) and in (%i) registers cannot be used before the SAVE!
    // Do this by creating a code sequence equivalent to:
    //        SETSW -(stackSize), %g1
    int uregNum = TM.getRegInfo()->getUnifiedRegNum(
			 TM.getRegInfo()->getRegClassIDOfType(Type::IntTy),
			 SparcV9IntRegClass::g1);

    MachineInstr* M = BuildMI(V9::SETHI, 2).addSImm(C)
      .addMReg(uregNum, MachineOperand::Def);
    M->getOperand(0).markHi32();
    mvec.push_back(M);

    M = BuildMI(V9::ORi, 3).addMReg(uregNum).addSImm(C)
      .addMReg(uregNum, MachineOperand::Def);
    M->getOperand(1).markLo32();
    mvec.push_back(M);

    M = BuildMI(V9::SRAi5, 3).addMReg(uregNum).addZImm(0)
      .addMReg(uregNum, MachineOperand::Def);
    mvec.push_back(M);

    // Now generate the SAVE using the value in register %g1
    M = BuildMI(V9::SAVEr,3).addMReg(SP).addMReg(uregNum)
          .addMReg(SP,MachineOperand::Def);
    mvec.push_back(M);
  }

  // For varargs function bodies, insert instructions to copy incoming
  // register arguments for the ... list to the stack.
  // The first K=6 arguments are always received via int arg regs
  // (%i0 ... %i5 if K=6) .
  // By copying the varargs arguments to the stack, va_arg() then can
  // simply assume that all vararg arguments are in an array on the stack.
  if (MF.getFunction()->getFunctionType()->isVarArg()) {
    int numFixedArgs    = MF.getFunction()->getFunctionType()->getNumParams();
    int numArgRegs      = TM.getRegInfo()->getNumOfIntArgRegs();
    if (numFixedArgs < numArgRegs) {
      const TargetFrameInfo &FI = *TM.getFrameInfo();
      int firstArgReg   = TM.getRegInfo()->getUnifiedRegNum(
                             TM.getRegInfo()->getRegClassIDOfType(Type::IntTy),
                             SparcV9IntRegClass::i0);
      int fpReg         = SparcV9::i6;
      int argSize       = 8;
      int firstArgOffset= SparcV9FrameInfo::FirstIncomingArgOffsetFromFP;
      int nextArgOffset = firstArgOffset + numFixedArgs * argSize;

      for (int i=numFixedArgs; i < numArgRegs; ++i) {
        mvec.push_back(BuildMI(V9::STXi, 3).addMReg(firstArgReg+i).
                       addMReg(fpReg).addSImm(nextArgOffset));
        nextArgOffset += argSize;
      }
    }
  }

  MF.front().insert(MF.front().begin(), mvec.begin(), mvec.end());
}

void InsertPrologEpilogCode::InsertEpilogCode(MachineFunction &MF)
{
  const TargetMachine &TM = MF.getTarget();
  const TargetInstrInfo &MII = *TM.getInstrInfo();

  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock &MBB = *I;
    const BasicBlock &BB = *I->getBasicBlock();
    const Instruction *TermInst = (Instruction*)BB.getTerminator();
    if (TermInst->getOpcode() == Instruction::Ret)
    {
      int ZR = TM.getRegInfo()->getZeroRegNum();
      MachineInstr *Restore =
        BuildMI(V9::RESTOREi, 3).addMReg(ZR).addSImm(0)
          .addMReg(ZR, MachineOperand::Def);

      MachineCodeForInstruction &termMvec =
        MachineCodeForInstruction::get(TermInst);

      // Remove the NOPs in the delay slots of the return instruction
      unsigned numNOPs = 0;
      while (termMvec.back()->getOpcode() == V9::NOP)
      {
        assert( termMvec.back() == &MBB.back());
        termMvec.pop_back();
        MBB.erase(&MBB.back());
        ++numNOPs;
      }
      assert(termMvec.back() == &MBB.back());

      // Check that we found the right number of NOPs and have the right
      // number of instructions to replace them.
      unsigned ndelays = MII.getNumDelaySlots(termMvec.back()->getOpcode());
      assert(numNOPs == ndelays && "Missing NOPs in delay slots?");
      assert(ndelays == 1 && "Cannot use epilog code for delay slots?");

      // Append the epilog code to the end of the basic block.
      MBB.push_back(Restore);
    }
  }
}

FunctionPass *createPrologEpilogInsertionPass() {
  return new InsertPrologEpilogCode();
}

} // End llvm namespace
