// $Id$
//***************************************************************************
// File:
//	Sparc.cpp
// 
// Purpose:
//	
// History:
//	7/15/01	 -  Vikram Adve  -  Created
//**************************************************************************/


#include "SparcInternals.h"
#include "llvm/Target/Sparc.h"
#include "llvm/CodeGen/InstrScheduling.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/CodeGen/RegisterAllocation.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/PassManager.h"
#include <iostream>
using std::cerr;

// Build the MachineInstruction Description Array...
const MachineInstrDescriptor SparcMachineInstrDesc[] = {
#define I(ENUM, OPCODESTRING, NUMOPERANDS, RESULTPOS, MAXIMM, IMMSE, \
          NUMDELAYSLOTS, LATENCY, SCHEDCLASS, INSTFLAGS)             \
  { OPCODESTRING, NUMOPERANDS, RESULTPOS, MAXIMM, IMMSE,             \
          NUMDELAYSLOTS, LATENCY, SCHEDCLASS, INSTFLAGS },
#include "SparcInstr.def"
};

//----------------------------------------------------------------------------
// allocateSparcTargetMachine - Allocate and return a subclass of TargetMachine
// that implements the Sparc backend. (the llvm/CodeGen/Sparc.h interface)
//----------------------------------------------------------------------------
//

TargetMachine *allocateSparcTargetMachine() { return new UltraSparc(); }


//---------------------------------------------------------------------------
// class InsertPrologEpilogCode
//
// Insert SAVE/RESTORE instructions for the function
//
// Insert prolog code at the unique function entry point.
// Insert epilog code at each function exit point.
// InsertPrologEpilog invokes these only if the function is not compiled
// with the leaf function optimization.
//
//---------------------------------------------------------------------------
static MachineInstr* minstrVec[MAX_INSTR_PER_VMINSTR];

class InsertPrologEpilogCode : public MethodPass {
  TargetMachine &Target;
public:
  inline InsertPrologEpilogCode(TargetMachine &T) : Target(T) {}
  bool runOnMethod(Function *F) {
    MachineCodeForMethod &mcodeInfo = MachineCodeForMethod::get(F);
    if (!mcodeInfo.isCompiledAsLeafMethod()) {
      InsertPrologCode(F);
      InsertEpilogCode(F);
    }
    return false;
  }

  void InsertPrologCode(Function *F);
  void InsertEpilogCode(Function *F);
};

void InsertPrologEpilogCode::InsertPrologCode(Function *F)
{
  BasicBlock *entryBB = F->getEntryNode();
  unsigned N = GetInstructionsForProlog(entryBB, Target, minstrVec);
  assert(N <= MAX_INSTR_PER_VMINSTR);
  MachineCodeForBasicBlock& bbMvec = entryBB->getMachineInstrVec();
  bbMvec.insert(bbMvec.begin(), minstrVec, minstrVec+N);
}


void InsertPrologEpilogCode::InsertEpilogCode(Function *F)
{
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    Instruction *TermInst = (Instruction*)(*I)->getTerminator();
    if (TermInst->getOpcode() == Instruction::Ret)
      {
        BasicBlock* exitBB = *I;
        unsigned N = GetInstructionsForEpilog(exitBB, Target, minstrVec);
        
        MachineCodeForBasicBlock& bbMvec = exitBB->getMachineInstrVec();
        MachineCodeForInstruction &termMvec =
          MachineCodeForInstruction::get(TermInst);
        
        // Remove the NOPs in the delay slots of the return instruction
        const MachineInstrInfo &mii = Target.getInstrInfo();
        unsigned numNOPs = 0;
        while (termMvec.back()->getOpCode() == NOP)
          {
            assert( termMvec.back() == bbMvec.back());
            termMvec.pop_back();
            bbMvec.pop_back();
            ++numNOPs;
          }
        assert(termMvec.back() == bbMvec.back());
        
        // Check that we found the right number of NOPs and have the right
        // number of instructions to replace them.
        unsigned ndelays = mii.getNumDelaySlots(termMvec.back()->getOpCode());
        assert(numNOPs == ndelays && "Missing NOPs in delay slots?");
        assert(N == ndelays && "Cannot use epilog code for delay slots?");
        
        // Append the epilog code to the end of the basic block.
        bbMvec.insert(bbMvec.end(), minstrVec, minstrVec+N);
      }
  }
}


//---------------------------------------------------------------------------
// class UltraSparcFrameInfo 
// 
// Purpose:
//   Interface to stack frame layout info for the UltraSPARC.
//   Starting offsets for each area of the stack frame are aligned at
//   a multiple of getStackFrameSizeAlignment().
//---------------------------------------------------------------------------

int
UltraSparcFrameInfo::getFirstAutomaticVarOffset(MachineCodeForMethod& ,
                                                bool& pos) const
{
  pos = false;                          // static stack area grows downwards
  return StaticAreaOffsetFromFP;
}

int
UltraSparcFrameInfo::getRegSpillAreaOffset(MachineCodeForMethod& mcInfo,
                                           bool& pos) const
{
  pos = false;                          // static stack area grows downwards
  unsigned int autoVarsSize = mcInfo.getAutomaticVarsSize();
  if (int mod = autoVarsSize % getStackFrameSizeAlignment())  
    autoVarsSize += (getStackFrameSizeAlignment() - mod);
  return StaticAreaOffsetFromFP - autoVarsSize; 
}

int
UltraSparcFrameInfo::getTmpAreaOffset(MachineCodeForMethod& mcInfo,
                                      bool& pos) const
{
  pos = false;                          // static stack area grows downwards
  unsigned int autoVarsSize = mcInfo.getAutomaticVarsSize();
  unsigned int spillAreaSize = mcInfo.getRegSpillsSize();
  int offset = autoVarsSize + spillAreaSize;
  if (int mod = offset % getStackFrameSizeAlignment())  
    offset += (getStackFrameSizeAlignment() - mod);
  return StaticAreaOffsetFromFP - offset;
}

int
UltraSparcFrameInfo::getDynamicAreaOffset(MachineCodeForMethod& mcInfo,
                                          bool& pos) const
{
  // Dynamic stack area grows downwards starting at top of opt-args area.
  // The opt-args, required-args, and register-save areas are empty except
  // during calls and traps, so they are shifted downwards on each
  // dynamic-size alloca.
  pos = false;
  unsigned int optArgsSize = mcInfo.getMaxOptionalArgsSize();
  int offset = optArgsSize + FirstOptionalOutgoingArgOffsetFromSP;
  assert((offset - OFFSET) % getStackFrameSizeAlignment() == 0);
  return offset;
}


//---------------------------------------------------------------------------
// class UltraSparcMachine 
// 
// Purpose:
//   Primary interface to machine description for the UltraSPARC.
//   Primarily just initializes machine-dependent parameters in
//   class TargetMachine, and creates machine-dependent subclasses
//   for classes such as MachineInstrInfo. 
// 
//---------------------------------------------------------------------------

UltraSparc::UltraSparc()
  : TargetMachine("UltraSparc-Native"),
    instrInfo(*this),
    schedInfo(*this),
    regInfo(*this),
    frameInfo(*this),
    cacheInfo(*this)
{
  optSizeForSubWordData = 4;
  minMemOpWordSize = 8; 
  maxAtomicMemOpWordSize = 8;
}



//===---------------------------------------------------------------------===//
// GenerateCodeForTarget Pass
// 
// Native code generation for a specified target.
//===---------------------------------------------------------------------===//

class ConstructMachineCodeForFunction : public MethodPass {
  TargetMachine &Target;
public:
  inline ConstructMachineCodeForFunction(TargetMachine &T) : Target(T) {}
  bool runOnMethod(Function *F) {
    MachineCodeForMethod::construct(F, Target);
    return false;
  }
};

class InstructionSelection : public MethodPass {
  TargetMachine &Target;
public:
  inline InstructionSelection(TargetMachine &T) : Target(T) {}
  bool runOnMethod(Function *F) {
    if (SelectInstructionsForMethod(F, Target)) {
      cerr << "Instr selection failed for function " << F->getName() << "\n";
      abort();
    }
    return false;
  }
};

struct FreeMachineCodeForFunction : public MethodPass {
  static void freeMachineCode(Instruction *I) {
    MachineCodeForInstruction::destroy(I);
  }
  
  bool runOnMethod(Function *F) {
    for (Function::iterator FI = F->begin(), FE = F->end(); FI != FE; ++FI)
      for (BasicBlock::iterator I = (*FI)->begin(), E = (*FI)->end();
           I != E; ++I)
        MachineCodeForInstruction::get(*I).dropAllReferences();
    
    for (Method::iterator FI = F->begin(), FE = F->end(); FI != FE; ++FI)
      for (BasicBlock::iterator I = (*FI)->begin(), E = (*FI)->end();
           I != E; ++I)
        freeMachineCode(*I);
    
    return false;
  }
};



// addPassesToEmitAssembly - This method controls the entire code generation
// process for the ultra sparc.
//
void UltraSparc::addPassesToEmitAssembly(PassManager &PM, std::ostream &Out) {
  // Construct and initialize the MachineCodeForMethod object for this fn.
  PM.add(new ConstructMachineCodeForFunction(*this));

  PM.add(new InstructionSelection(*this));

  PM.add(createInstructionSchedulingWithSSAPass(*this));

  PM.add(getRegisterAllocator(*this));
  
  //PM.add(new OptimizeLeafProcedures());
  //PM.add(new DeleteFallThroughBranches());
  //PM.add(new RemoveChainedBranches());    // should be folded with previous
  //PM.add(new RemoveRedundantOps());       // operations with %g0, NOP, etc.
  
  PM.add(new InsertPrologEpilogCode(*this));
  
  // Output assembly language to the .s file.  Assembly emission is split into
  // two parts: Function output and Global value output.  This is because
  // function output is pipelined with all of the rest of code generation stuff,
  // allowing machine code representations for functions to be free'd after the
  // function has been emitted.
  //
  PM.add(getMethodAsmPrinterPass(PM, Out));
  PM.add(new FreeMachineCodeForFunction());  // Free stuff no longer needed

  // Emit Module level assembly after all of the functions have been processed.
  PM.add(getModuleAsmPrinterPass(PM, Out));

  // Emit bytecode to the sparc assembly file into its special section next
  PM.add(getEmitBytecodeToAsmPass(Out));
}
