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
#include "llvm/Method.h"
#include "llvm/CodeGen/InstrScheduling.h"
#include "llvm/CodeGen/InstrSelection.h"



//---------------------------------------------------------------------------
// class UltraSparcInstrInfo 
// 
// Purpose:
//   Information about individual instructions.
//   Most information is stored in the SparcMachineInstrDesc array above.
//   Other information is computed on demand, and most such functions
//   default to member functions in base class MachineInstrInfo. 
//---------------------------------------------------------------------------

/*ctor*/
UltraSparcInstrInfo::UltraSparcInstrInfo()
  : MachineInstrInfo(SparcMachineInstrDesc,
		     /*descSize = */ NUM_TOTAL_OPCODES,
		     /*numRealOpCodes = */ NUM_REAL_OPCODES)
{
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
UltraSparcSchedInfo::UltraSparcSchedInfo(const MachineInstrInfo* mii)
  : MachineSchedInfo((unsigned int) SPARC_NUM_SCHED_CLASSES,
		     mii,
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
// class UltraSparcMachine 
// 
// Purpose:
//   Primary interface to machine description for the UltraSPARC.
//   Primarily just initializes machine-dependent parameters in
//   class TargetMachine, and creates machine-dependent subclasses
//   for classes such as MachineInstrInfo. 
// 
//---------------------------------------------------------------------------

UltraSparc::UltraSparc() : TargetMachine("UltraSparc-Native") {
  machineInstrInfo = new UltraSparcInstrInfo();
  machineSchedInfo = new UltraSparcSchedInfo(machineInstrInfo); 
  
  optSizeForSubWordData = 4;
  minMemOpWordSize = 8; 
  maxAtomicMemOpWordSize = 8;
  zeroRegNum = 0;			// %g0 always gives 0 on Sparc
}

UltraSparc::~UltraSparc() {
  delete (UltraSparcInstrInfo*) machineInstrInfo;
  delete (UltraSparcSchedInfo*) machineSchedInfo;
}


bool UltraSparc::compileMethod(Method *M) {
  if (SelectInstructionsForMethod(M, *this)) {
    cerr << "Instruction selection failed for method " << M->getName()
       << "\n\n";
    return true;
  }
  
  if (ScheduleInstructionsWithSSA(M, *this)) {
    cerr << "Instruction scheduling before allocation failed for method "
       << M->getName() << "\n\n";
    return true;
  }
  return false;
}
