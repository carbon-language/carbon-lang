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
#include "llvm/CodeGen/PhyRegAlloc.h"
#include "llvm/Analysis/LiveVar/MethodLiveVarInfo.h"
#include "llvm/Method.h"


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

void AllocateRegisters(Method *M, TargetMachine &target)
{
 
  if ( (M)->isExternal() )     // don't process prototypes
    return;
    
  if( DEBUG_RA ) {
    cerr << endl << "******************** Method "<< (M)->getName();
    cerr <<        " ********************" <<endl;
  }
    
  MethodLiveVarInfo LVI(M );   // Analyze live varaibles
  LVI.analyze();
  
    
  PhyRegAlloc PRA(M, target, &LVI); // allocate registers
  PRA.allocateRegisters();
    

  if( DEBUG_RA )  cerr << endl << "Register allocation complete!" << endl;

}


//---------------------------------------------------------------------------
// Function InsertPrologCode
// Function InsertEpilogCode
// Function InsertPrologEpilog
// 
// Insert prolog code at the unique method entry point.
// Insert epilog code at each method exit point.
// InsertPrologEpilog invokes these only if the method is not compiled
// with the leaf method optimization.
//---------------------------------------------------------------------------

static MachineInstr* minstrVec[MAX_INSTR_PER_VMINSTR];

static void
InsertPrologCode(Method* method, TargetMachine& target)
{
  BasicBlock* entryBB = method->getEntryNode();
  unsigned N = GetInstructionsForProlog(entryBB, target, minstrVec);
  assert(N <= MAX_INSTR_PER_VMINSTR);
  if (N > 0)
    {
      MachineCodeForBasicBlock& bbMvec = entryBB->getMachineInstrVec();
      bbMvec.insert(bbMvec.begin(), minstrVec, minstrVec+N);
    }
}


static void
InsertEpilogCode(Method* method, TargetMachine& target)
{
  for (Method::iterator I=method->begin(), E=method->end(); I != E; ++I)
    if ((*I)->getTerminator()->getOpcode() == Instruction::Ret)
      {
        BasicBlock* exitBB = *I;
        unsigned N = GetInstructionsForEpilog(exitBB, target, minstrVec);
        
        MachineCodeForBasicBlock& bbMvec = exitBB->getMachineInstrVec();
        MachineCodeForVMInstr& termMvec =
          exitBB->getTerminator()->getMachineInstrVec();
        
        // Remove the NOPs in the delay slots of the return instruction
        const MachineInstrInfo& mii = target.getInstrInfo();
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


// Insert SAVE/RESTORE instructions for the method
static void
InsertPrologEpilog(Method *method, TargetMachine &target)
{
  MachineCodeForMethod& mcodeInfo = MachineCodeForMethod::get(method);
  if (mcodeInfo.isCompiledAsLeafMethod())
    return;                             // nothing to do
  
  InsertPrologCode(method, target);
  InsertEpilogCode(method, target);
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
//   Note that there is no machine-independent interface to this information
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
  return  StaticAreaOffsetFromFP - autoVarsSize;
}

int
UltraSparcFrameInfo::getTmpAreaOffset(MachineCodeForMethod& mcInfo,
                                      bool& pos) const
{
  pos = false;                          // static stack area grows downwards
  unsigned int autoVarsSize = mcInfo.getAutomaticVarsSize();
  unsigned int spillAreaSize = mcInfo.getRegSpillsSize();
  return StaticAreaOffsetFromFP - (autoVarsSize + spillAreaSize);
}

int
UltraSparcFrameInfo::getDynamicAreaOffset(MachineCodeForMethod& mcInfo,
                                          bool& pos) const
{
  // dynamic stack area grows downwards starting at top of opt-args area
  unsigned int optArgsSize = mcInfo.getMaxOptionalArgsSize();
  return optArgsSize + FirstOptionalOutgoingArgOffsetFromSP;
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


void
ApplyPeepholeOptimizations(Method *method, TargetMachine &target)
{
  return;
  
  // OptimizeLeafProcedures();
  // DeleteFallThroughBranches();
  // RemoveChainedBranches();    // should be folded with previous
  // RemoveRedundantOps();       // operations with %g0, NOP, etc.
}



bool
UltraSparc::compileMethod(Method *method)
{
  // Construct and initialize the MachineCodeForMethod object for this method.
  (void) MachineCodeForMethod::construct(method, *this);
  
  if (SelectInstructionsForMethod(method, *this))
    {
      cerr << "Instruction selection failed for method " << method->getName()
	   << "\n\n";
      return true;
    }
  
  if (ScheduleInstructionsWithSSA(method, *this))
    {
      cerr << "Instruction scheduling before allocation failed for method "
	   << method->getName() << "\n\n";
      return true;
    }
  
  AllocateRegisters(method, *this);          // allocate registers
  
  ApplyPeepholeOptimizations(method, *this); // machine-dependent peephole opts
  
  InsertPrologEpilog(method, *this);
  
  return false;
}
