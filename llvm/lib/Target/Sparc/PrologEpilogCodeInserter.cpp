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
#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/CodeGen/MachineCodeForBasicBlock.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/Instruction.h"

namespace {
  class InsertPrologEpilogCode : public FunctionPass {
    TargetMachine &Target;
  public:
    InsertPrologEpilogCode(TargetMachine &T) : Target(T) {}
    
    const char *getPassName() const { return "Sparc Prolog/Epilog Inserter"; }
    
    bool runOnFunction(Function &F) {
      MachineCodeForMethod &mcodeInfo = MachineCodeForMethod::get(&F);
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
  // We will assume that local register `l0' is unused since the SAVE
  // instruction must be the first instruction in each procedure.
  // 
  MachineCodeForMethod& mcInfo = MachineCodeForMethod::get(&F);
  unsigned int staticStackSize = mcInfo.getStaticStackSize();
  
  if (staticStackSize < (unsigned) frameInfo.getMinStackFrameSize())
    staticStackSize = (unsigned) frameInfo.getMinStackFrameSize();
  
  if (unsigned padsz = (staticStackSize %
                        (unsigned) frameInfo.getStackFrameSizeAlignment()))
    staticStackSize += frameInfo.getStackFrameSizeAlignment() - padsz;
  
  if (Target.getInstrInfo().constantFitsInImmedField(SAVE, staticStackSize))
    {
      M = new MachineInstr(SAVE);
      M->SetMachineOperandReg(0, Target.getRegInfo().getStackPointer());
      M->SetMachineOperandConst(1, MachineOperand::MO_SignExtendedImmed,
                                   - (int) staticStackSize);
      M->SetMachineOperandReg(2, Target.getRegInfo().getStackPointer());
      mvec.push_back(M);
    }
  else
    {
      M = new MachineInstr(SETSW);
      M->SetMachineOperandConst(0, MachineOperand::MO_SignExtendedImmed,
                                - (int) staticStackSize);
      M->SetMachineOperandReg(1, MachineOperand::MO_MachineRegister,
                                 Target.getRegInfo().getUnifiedRegNum(
                           Target.getRegInfo().getRegClassIDOfType(Type::IntTy),
                                  SparcIntRegOrder::l0));
      mvec.push_back(M);
      
      M = new MachineInstr(SAVE);
      M->SetMachineOperandReg(0, Target.getRegInfo().getStackPointer());
      M->SetMachineOperandReg(1, MachineOperand::MO_MachineRegister,
                                 Target.getRegInfo().getUnifiedRegNum(
                           Target.getRegInfo().getRegClassIDOfType(Type::IntTy),
                                  SparcIntRegOrder::l0));
      M->SetMachineOperandReg(2, Target.getRegInfo().getStackPointer());
      mvec.push_back(M);
    }

  MachineCodeForBasicBlock& bbMvec = MachineCodeForBasicBlock::get(&F.getEntryNode());
  bbMvec.insert(bbMvec.begin(), mvec.begin(), mvec.end());
}

void InsertPrologEpilogCode::InsertEpilogCode(Function &F)
{
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    Instruction *TermInst = (Instruction*)I->getTerminator();
    if (TermInst->getOpcode() == Instruction::Ret)
      {
        MachineInstr *Restore = new MachineInstr(RESTORE);
        Restore->SetMachineOperandReg(0, Target.getRegInfo().getZeroRegNum());
        Restore->SetMachineOperandConst(1, MachineOperand::MO_SignExtendedImmed,
                                        (int64_t)0);
        Restore->SetMachineOperandReg(2, Target.getRegInfo().getZeroRegNum());
        
        MachineCodeForBasicBlock& bbMvec = MachineCodeForBasicBlock::get(I);
        MachineCodeForInstruction &termMvec =
          MachineCodeForInstruction::get(TermInst);
        
        // Remove the NOPs in the delay slots of the return instruction
        const MachineInstrInfo &mii = Target.getInstrInfo();
        unsigned numNOPs = 0;
        while (termMvec.back()->getOpCode() == NOP)
          {
            assert( termMvec.back() == bbMvec.back());
            delete bbMvec.pop_back();
            termMvec.pop_back();
            ++numNOPs;
          }
        assert(termMvec.back() == bbMvec.back());
        
        // Check that we found the right number of NOPs and have the right
        // number of instructions to replace them.
        unsigned ndelays = mii.getNumDelaySlots(termMvec.back()->getOpCode());
        assert(numNOPs == ndelays && "Missing NOPs in delay slots?");
        assert(ndelays == 1 && "Cannot use epilog code for delay slots?");
        
        // Append the epilog code to the end of the basic block.
        bbMvec.push_back(Restore);
      }
  }
}

Pass *createPrologEpilogCodeInserter(TargetMachine &TM) {
  return new InsertPrologEpilogCode(TM);
}
