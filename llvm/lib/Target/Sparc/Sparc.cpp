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
#include "llvm/CodeGen/PhyRegAlloc.h"
#include "llvm/Method.h"
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


//----------------------------------------------------------------------------
// Entry point for register allocation for a module
//----------------------------------------------------------------------------

class RegisterAllocation : public MethodPass {
  TargetMachine &Target;
public:
  inline RegisterAllocation(TargetMachine &T) : Target(T) {}
  bool runOnMethod(Method *M) {
    if (DEBUG_RA)
      cerr << "\n******************** Method "<< M->getName()
           << " ********************\n";
    
    MethodLiveVarInfo LVI(M );   // Analyze live varaibles
    LVI.analyze();
    
    PhyRegAlloc PRA(M, Target, &LVI); // allocate registers
    PRA.allocateRegisters();

    if (DEBUG_RA) cerr << "\nRegister allocation complete!\n";
    return false;
  }
};

static MachineInstr* minstrVec[MAX_INSTR_PER_VMINSTR];

//---------------------------------------------------------------------------
// class InsertPrologEpilogCode
//
// Insert SAVE/RESTORE instructions for the method
//
// Insert prolog code at the unique method entry point.
// Insert epilog code at each method exit point.
// InsertPrologEpilog invokes these only if the method is not compiled
// with the leaf method optimization.
//
//---------------------------------------------------------------------------

class InsertPrologEpilogCode : public MethodPass {
  TargetMachine &Target;
public:
  inline InsertPrologEpilogCode(TargetMachine &T) : Target(T) {}
  bool runOnMethod(Method *M) {
    MachineCodeForMethod &mcodeInfo = MachineCodeForMethod::get(M);
    if (!mcodeInfo.isCompiledAsLeafMethod()) {
      InsertPrologCode(M);
      InsertEpilogCode(M);
    }
    return false;
  }

  void InsertPrologCode(Method *M);
  void InsertEpilogCode(Method *M);
};

void InsertPrologEpilogCode::InsertPrologCode(Method* method)
{
  BasicBlock* entryBB = method->getEntryNode();
  unsigned N = GetInstructionsForProlog(entryBB, Target, minstrVec);
  assert(N <= MAX_INSTR_PER_VMINSTR);
  MachineCodeForBasicBlock& bbMvec = entryBB->getMachineInstrVec();
  bbMvec.insert(bbMvec.begin(), minstrVec, minstrVec+N);
}


void InsertPrologEpilogCode::InsertEpilogCode(Method* method)
{
  for (Method::iterator I=method->begin(), E=method->end(); I != E; ++I)
    if ((*I)->getTerminator()->getOpcode() == Instruction::Ret)
      {
        BasicBlock* exitBB = *I;
        unsigned N = GetInstructionsForEpilog(exitBB, Target, minstrVec);
        
        MachineCodeForBasicBlock& bbMvec = exitBB->getMachineInstrVec();
        MachineCodeForInstruction &termMvec =
          MachineCodeForInstruction::get(exitBB->getTerminator());
        
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
        bbMvec.push_back(minstrVec[0]);
      }
}





//---------------------------------------------------------------------------
// class UltraSparcSchedInfo 
// 
// Purpose:
//   Scheduling information for the UltraSPARC.
//   Primarily just initializes machine-dependent parameters in
//   class MachineSchedInfo.
//---------------------------------------------------------------------------

/*ctor*/
UltraSparcSchedInfo::UltraSparcSchedInfo(const TargetMachine& tgt)
  : MachineSchedInfo(tgt,
                     (unsigned int) SPARC_NUM_SCHED_CLASSES,
		     SparcRUsageDesc,
		     SparcInstrUsageDeltas,
		     SparcInstrIssueDeltas,
		     sizeof(SparcInstrUsageDeltas)/sizeof(InstrRUsageDelta),
		     sizeof(SparcInstrIssueDeltas)/sizeof(InstrIssueDelta))
{
  maxNumIssueTotal = 4;
  longestIssueConflict = 0;		// computed from issuesGaps[]
  
  branchMispredictPenalty = 4;		// 4 for SPARC IIi
  branchTargetUnknownPenalty = 2;	// 2 for SPARC IIi
  l1DCacheMissPenalty = 8;		// 7 or 9 for SPARC IIi
  l1ICacheMissPenalty = 8;		// ? for SPARC IIi
  
  inOrderLoads = true;			// true for SPARC IIi
  inOrderIssue = true;			// true for SPARC IIi
  inOrderExec  = false;			// false for most architectures
  inOrderRetire= true;			// true for most architectures
  
  // must be called after above parameters are initialized.
  this->initializeResources();
}

void
UltraSparcSchedInfo::initializeResources()
{
  // Compute MachineSchedInfo::instrRUsages and MachineSchedInfo::issueGaps
  MachineSchedInfo::initializeResources();
  
  // Machine-dependent fixups go here.  None for now.
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
  // dynamic stack area grows downwards starting at top of opt-args area
  unsigned int optArgsSize = mcInfo.getMaxOptionalArgsSize();
  int offset = optArgsSize + FirstOptionalOutgoingArgOffsetFromSP;
  assert(offset % getStackFrameSizeAlignment() == 0);
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

class ConstructMachineCodeForMethod : public MethodPass {
  TargetMachine &Target;
public:
  inline ConstructMachineCodeForMethod(TargetMachine &T) : Target(T) {}
  bool runOnMethod(Method *M) {
    MachineCodeForMethod::construct(M, Target);
    return false;
  }
};

class InstructionSelection : public MethodPass {
  TargetMachine &Target;
public:
  inline InstructionSelection(TargetMachine &T) : Target(T) {}
  bool runOnMethod(Method *M) {
    if (SelectInstructionsForMethod(M, Target))
      cerr << "Instr selection failed for method " << M->getName() << "\n";
    return false;
  }
};

class InstructionScheduling : public MethodPass {
  TargetMachine &Target;
public:
  inline InstructionScheduling(TargetMachine &T) : Target(T) {}
  bool runOnMethod(Method *M) {
    if (ScheduleInstructionsWithSSA(M, Target))
      cerr << "Instr scheduling failed for method " << M->getName() << "\n\n";
    return false;
  }
};

struct FreeMachineCodeForMethod : public MethodPass {
  static void freeMachineCode(Instruction *I) {
    MachineCodeForInstruction::destroy(I);
  }

  bool runOnMethod(Method *M) {
    for_each(M->inst_begin(), M->inst_end(), freeMachineCode);
    // Don't destruct MachineCodeForMethod - The global printer needs it
    //MachineCodeForMethod::destruct(M);
    return false;
  }
};


void UltraSparc::addPassesToEmitAssembly(PassManager &PM, std::ostream &Out) {
  // Construct and initialize the MachineCodeForMethod object for this method.
  PM.add(new ConstructMachineCodeForMethod(*this));

  PM.add(new InstructionSelection(*this));

  //PM.add(new InstructionScheduling(*this));

  PM.add(new RegisterAllocation(*this));
  
  //PM.add(new OptimizeLeafProcedures());
  //PM.add(new DeleteFallThroughBranches());
  //PM.add(new RemoveChainedBranches());    // should be folded with previous
  //PM.add(new RemoveRedundantOps());       // operations with %g0, NOP, etc.
  
  PM.add(new InsertPrologEpilogCode(*this));
  
  // Output assembly language to the .s file.  Assembly emission is split into
  // two parts: Method output and Global value output.  This is because method
  // output is pipelined with all of the rest of code generation stuff,
  // allowing machine code representations for methods to be free'd after the
  // method has been emitted.
  //
  PM.add(getMethodAsmPrinterPass(PM, Out));
  PM.add(new FreeMachineCodeForMethod());  // Free stuff no longer needed

  // Emit Module level assembly after all of the methods have been processed.
  PM.add(getModuleAsmPrinterPass(PM, Out));
}
