//===-- PrologEpilogCodeInserter.cpp - Insert Prolog & Epilog code for fn -===//
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
#include "llvm/CodeGen/InstrSelectionSupport.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"

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
  if (TM.getInstrInfo().constantFitsInImmedField(SAVE, staticStackSize)) {
    mvec.push_back(BuildMI(SAVE, 3).addMReg(SP).addSImm(C).addMReg(SP,
                                                                   MOTy::Def));
  } else {
    // We have to put the stack size value into a register before SAVE.
    // Use register %g1 since it is volatile across calls.  Note that the
    // local (%l) and in (%i) registers cannot be used before the SAVE!
    // Do this by creating a code sequence equivalent to:
    //        SETSW -(stackSize), %g1
    int uregNum = TM.getRegInfo().getUnifiedRegNum(
			 TM.getRegInfo().getRegClassIDOfType(Type::IntTy),
			 SparcIntRegClass::g1);

    MachineInstr* M = BuildMI(SETHI, 2).addSImm(C).addMReg(uregNum, MOTy::Def);
    M->setOperandHi32(0);
    mvec.push_back(M);
    
    M = BuildMI(OR, 3).addMReg(uregNum).addSImm(C).addMReg(uregNum, MOTy::Def);
    M->setOperandLo32(1);
    mvec.push_back(M);
    
    M = BuildMI(SRA, 3).addMReg(uregNum).addZImm(0).addMReg(uregNum, MOTy::Def);
    mvec.push_back(M);
    
    // Now generate the SAVE using the value in register %g1
    M = BuildMI(SAVE, 3).addMReg(SP).addMReg(uregNum).addMReg(SP, MOTy::Def);
    mvec.push_back(M);
  }

  MF.front().insert(MF.front().begin(), mvec.begin(), mvec.end());
}

void InsertPrologEpilogCode::InsertEpilogCode(MachineFunction &MF)
{
  const TargetMachine &TM = MF.getTarget();
  const TargetInstrInfo &MII = TM.getInstrInfo();

  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock &MBB = *I;
    BasicBlock &BB = *I->getBasicBlock();
    Instruction *TermInst = (Instruction*)BB.getTerminator();
    if (TermInst->getOpcode() == Instruction::Ret)
      {
        int ZR = TM.getRegInfo().getZeroRegNum();
        MachineInstr *Restore =
          BuildMI(RESTORE, 3).addMReg(ZR).addSImm(0).addMReg(ZR, MOTy::Def);
        
        MachineCodeForInstruction &termMvec =
          MachineCodeForInstruction::get(TermInst);
        
        // Remove the NOPs in the delay slots of the return instruction
        unsigned numNOPs = 0;
        while (termMvec.back()->getOpCode() == NOP)
          {
            assert( termMvec.back() == MBB.back());
            delete MBB.pop_back();
            termMvec.pop_back();
            ++numNOPs;
          }
        assert(termMvec.back() == MBB.back());
        
        // Check that we found the right number of NOPs and have the right
        // number of instructions to replace them.
        unsigned ndelays = MII.getNumDelaySlots(termMvec.back()->getOpCode());
        assert(numNOPs == ndelays && "Missing NOPs in delay slots?");
        assert(ndelays == 1 && "Cannot use epilog code for delay slots?");
        
        // Append the epilog code to the end of the basic block.
        MBB.push_back(Restore);
      }
  }
}

Pass* UltraSparc::getPrologEpilogInsertionPass() {
  return new InsertPrologEpilogCode();
}
