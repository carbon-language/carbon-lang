//===-- PrologEpilogCodeInserter.cpp - Insert Prolog & Epilog code for fn -===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Insert SAVE/RESTORE instructions for the function
//
// Insert prolog code at the unique function entry point.
// Insert epilog code at each function exit point.
// InsertPrologEpilog invokes these only if the function is not compiled
// with the leaf function optimization.
//
//===----------------------------------------------------------------------===//

#include "SparcInternals.h"
#include "SparcRegClassInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Intrinsics.h"

namespace llvm {

namespace {
  struct InsertPrologEpilogCode : public MachineFunctionPass {
    const char *getPassName() const { return "Sparc Prolog/Epilog Inserter"; }
    
    bool runOnMachineFunction(MachineFunction &F) {
      if (!F.getInfo()->isCompiledAsLeafMethod()) {
        InsertPrologCode(F);
        InsertEpilogCode(F);
      }
      return false;
    }
    
    void InsertPrologCode(MachineFunction &F);
    void InsertEpilogCode(MachineFunction &F);
  };

}  // End anonymous namespace

//------------------------------------------------------------------------ 
//   Create prolog and epilog code for procedure entry and exit
//------------------------------------------------------------------------ 

void InsertPrologEpilogCode::InsertPrologCode(MachineFunction &MF)
{
  std::vector<MachineInstr*> mvec;
  const TargetMachine &TM = MF.getTarget();
  const TargetFrameInfo& frameInfo = TM.getFrameInfo();
  
  // The second operand is the stack size. If it does not fit in the
  // immediate field, we have to use a free register to hold the size.
  // See the comments below for the choice of this register.
  // 
  unsigned staticStackSize = MF.getInfo()->getStaticStackSize();
  
  if (staticStackSize < (unsigned) frameInfo.getMinStackFrameSize())
    staticStackSize = (unsigned) frameInfo.getMinStackFrameSize();

  if (unsigned padsz = (staticStackSize %
                        (unsigned) frameInfo.getStackFrameSizeAlignment()))
    staticStackSize += frameInfo.getStackFrameSizeAlignment() - padsz;
  
  int32_t C = - (int) staticStackSize;
  int SP = TM.getRegInfo().getStackPointer();
  if (TM.getInstrInfo().constantFitsInImmedField(V9::SAVEi,staticStackSize)) {
    mvec.push_back(BuildMI(V9::SAVEi, 3).addMReg(SP).addSImm(C)
                   .addMReg(SP, MOTy::Def));
  } else {
    // We have to put the stack size value into a register before SAVE.
    // Use register %g1 since it is volatile across calls.  Note that the
    // local (%l) and in (%i) registers cannot be used before the SAVE!
    // Do this by creating a code sequence equivalent to:
    //        SETSW -(stackSize), %g1
    int uregNum = TM.getRegInfo().getUnifiedRegNum(
			 TM.getRegInfo().getRegClassIDOfType(Type::IntTy),
			 SparcIntRegClass::g1);

    MachineInstr* M = BuildMI(V9::SETHI, 2).addSImm(C)
      .addMReg(uregNum, MOTy::Def);
    M->setOperandHi32(0);
    mvec.push_back(M);
    
    M = BuildMI(V9::ORi, 3).addMReg(uregNum).addSImm(C)
      .addMReg(uregNum, MOTy::Def);
    M->setOperandLo32(1);
    mvec.push_back(M);
    
    M = BuildMI(V9::SRAi5, 3).addMReg(uregNum).addZImm(0)
      .addMReg(uregNum, MOTy::Def);
    mvec.push_back(M);
    
    // Now generate the SAVE using the value in register %g1
    M = BuildMI(V9::SAVEr,3).addMReg(SP).addMReg(uregNum).addMReg(SP,MOTy::Def);
    mvec.push_back(M);
  }

  // For varargs function bodies, insert instructions to copy incoming
  // register arguments for the ... list to the stack.
  // The first K=6 arguments are always received via int arg regs
  // (%i0 ... %i5 if K=6) .
  // By copying the varargs arguments to the stack, va_arg() then can
  // simply assume that all vararg arguments are in an array on the stack. 
  // 
  if (MF.getFunction()->getFunctionType()->isVarArg()) {
    int numFixedArgs    = MF.getFunction()->getFunctionType()->getNumParams();
    int numArgRegs      = TM.getRegInfo().getNumOfIntArgRegs();
    if (numFixedArgs < numArgRegs) {
      bool ignore;
      int firstArgReg   = TM.getRegInfo().getUnifiedRegNum(
                             TM.getRegInfo().getRegClassIDOfType(Type::IntTy),
                             SparcIntRegClass::i0);
      int fpReg         = TM.getFrameInfo().getIncomingArgBaseRegNum();
      int argSize       = TM.getFrameInfo().getSizeOfEachArgOnStack();
      int firstArgOffset=TM.getFrameInfo().getFirstIncomingArgOffset(MF,ignore);
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
  const TargetInstrInfo &MII = TM.getInstrInfo();

  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock &MBB = *I;
    const BasicBlock &BB = *I->getBasicBlock();
    const Instruction *TermInst = (Instruction*)BB.getTerminator();
    if (TermInst->getOpcode() == Instruction::Ret)
    {
      int ZR = TM.getRegInfo().getZeroRegNum();
      MachineInstr *Restore = 
        BuildMI(V9::RESTOREi, 3).addMReg(ZR).addSImm(0).addMReg(ZR, MOTy::Def);
      
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
