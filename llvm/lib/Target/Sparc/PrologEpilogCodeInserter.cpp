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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/InstrSelectionSupport.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"

namespace {
  class InsertPrologEpilogCode : public FunctionPass {
    TargetMachine &Target;
  public:
    InsertPrologEpilogCode(TargetMachine &T) : Target(T) {}
    
    const char *getPassName() const { return "Sparc Prolog/Epilog Inserter"; }
    
    bool runOnFunction(Function &F) {
      MachineFunction &mcodeInfo = MachineFunction::get(&F);
      if (!mcodeInfo.isCompiledAsLeafMethod()) {
        InsertPrologCode(F);
        InsertEpilogCode(F);
      }
      return false;
    }
    
    void InsertPrologCode(Function &F);
    void InsertEpilogCode(Function &F);
  };

}  // End anonymous namespace

//------------------------------------------------------------------------ 
// External Function: GetInstructionsForProlog
// External Function: GetInstructionsForEpilog
//
// Purpose:
//   Create prolog and epilog code for procedure entry and exit
//------------------------------------------------------------------------ 

void InsertPrologEpilogCode::InsertPrologCode(Function &F)
{
  std::vector<MachineInstr*> mvec;
  MachineInstr* M;
  const MachineFrameInfo& frameInfo = Target.getFrameInfo();
  
  // The second operand is the stack size. If it does not fit in the
  // immediate field, we have to use a free register to hold the size.
  // See the comments below for the choice of this register.
  // 
  MachineFunction& mcInfo = MachineFunction::get(&F);
  unsigned staticStackSize = mcInfo.getStaticStackSize();
  
  if (staticStackSize < (unsigned) frameInfo.getMinStackFrameSize())
    staticStackSize = (unsigned) frameInfo.getMinStackFrameSize();

  if (unsigned padsz = (staticStackSize %
                        (unsigned) frameInfo.getStackFrameSizeAlignment()))
    staticStackSize += frameInfo.getStackFrameSizeAlignment() - padsz;
  
  int32_t C = - (int) staticStackSize;
  int SP = Target.getRegInfo().getStackPointer();
  if (Target.getInstrInfo().constantFitsInImmedField(SAVE, staticStackSize)) {
    M = BuildMI(SAVE, 3).addMReg(SP).addSImm(C).addMReg(SP);
    mvec.push_back(M);
  } else {
      // We have to put the stack size value into a register before SAVE.
      // Use register %g1 since it is volatile across calls.  Note that the
      // local (%l) and in (%i) registers cannot be used before the SAVE!
      // Do this by creating a code sequence equivalent to:
      //        SETSW -(stackSize), %g1
      int uregNum = Target.getRegInfo().getUnifiedRegNum(
                           Target.getRegInfo().getRegClassIDOfType(Type::IntTy),
                           SparcIntRegClass::g1);
      
      M = BuildMI(SETHI, 2).addSImm(C).addMReg(uregNum);
      M->setOperandHi32(0);
      mvec.push_back(M);
      
      M = BuildMI(OR, 3).addMReg(uregNum).addSImm(C).addMReg(uregNum);
      M->setOperandLo32(1);
      mvec.push_back(M);
      
      M = BuildMI(SRA, 3).addMReg(uregNum).addZImm(0).addMReg(uregNum);
      mvec.push_back(M);
      
      // Now generate the SAVE using the value in register %g1
      M = BuildMI(SAVE, 3).addMReg(SP).addMReg(uregNum).addMReg(SP);
      mvec.push_back(M);
    }

  MachineBasicBlock& bbMvec = mcInfo.front();
  bbMvec.insert(bbMvec.begin(), mvec.begin(), mvec.end());
}

void InsertPrologEpilogCode::InsertEpilogCode(Function &F)
{
  MachineFunction &MF = MachineFunction::get(&F);
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock &MBB = *I;
    BasicBlock &BB = *I->getBasicBlock();
    Instruction *TermInst = (Instruction*)BB.getTerminator();
    if (TermInst->getOpcode() == Instruction::Ret)
      {
        int ZR = Target.getRegInfo().getZeroRegNum();
        MachineInstr *Restore =
          BuildMI(RESTORE, 3).addMReg(ZR).addSImm(0).addMReg(ZR);
        
        MachineCodeForInstruction &termMvec =
          MachineCodeForInstruction::get(TermInst);
        
        // Remove the NOPs in the delay slots of the return instruction
        const MachineInstrInfo &mii = Target.getInstrInfo();
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
        unsigned ndelays = mii.getNumDelaySlots(termMvec.back()->getOpCode());
        assert(numNOPs == ndelays && "Missing NOPs in delay slots?");
        assert(ndelays == 1 && "Cannot use epilog code for delay slots?");
        
        // Append the epilog code to the end of the basic block.
        MBB.push_back(Restore);
      }
  }
}

Pass* UltraSparc::getPrologEpilogInsertionPass() {
  return new InsertPrologEpilogCode(*this);
}
