//===-- TargetMachine.cpp - General Target Information ---------------------==//
//
// This file describes the general parts of a Target machine.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/Machine.h"
#include "llvm/DerivedTypes.h"

// External object describing the machine instructions
// Initialized only when the TargetMachine class is created
// and reset when that class is destroyed.
// 
const MachineInstrDescriptor* TargetInstrDescriptors = NULL;

resourceId_t MachineResource::nextId = 0;

static cycles_t	ComputeMinGap		(const InstrRUsage& fromRU,
					 const InstrRUsage& toRU);

static bool	RUConflict		(const vector<resourceId_t>& fromRVec,
					 const vector<resourceId_t>& fromRVec);

//---------------------------------------------------------------------------
// class TargetMachine
// 
// Purpose:
//   Machine description.
// 
//---------------------------------------------------------------------------


// function TargetMachine::findOptimalStorageSize 
// 
// Purpose:
//   This default implementation assumes that all sub-word data items use
//   space equal to optSizeForSubWordData, and all other primitive data
//   items use space according to the type.
//   
unsigned int TargetMachine::findOptimalStorageSize(const Type* ty) const {
  switch(ty->getPrimitiveID()) {
  case Type::BoolTyID:
  case Type::UByteTyID:
  case Type::SByteTyID:     
  case Type::UShortTyID:
  case Type::ShortTyID:     
    return optSizeForSubWordData;
    
  default:
    return DataLayout.getTypeSize(ty);
  }
}


//---------------------------------------------------------------------------
// class MachineInstructionInfo
//	Interface to description of machine instructions
//---------------------------------------------------------------------------


/*ctor*/
MachineInstrInfo::MachineInstrInfo(const MachineInstrDescriptor* _desc,
				   unsigned int _descSize,
				   unsigned int _numRealOpCodes)
  : desc(_desc), descSize(_descSize), numRealOpCodes(_numRealOpCodes)
{
  assert(TargetInstrDescriptors == NULL && desc != NULL);
  TargetInstrDescriptors = desc;	// initialize global variable
}  


/*dtor*/
MachineInstrInfo::~MachineInstrInfo()
{
  TargetInstrDescriptors = NULL;	// reset global variable
}


bool
MachineInstrInfo::constantFitsInImmedField(MachineOpCode opCode,
					   int64_t intValue) const
{
  // First, check if opCode has an immed field.
  bool isSignExtended;
  uint64_t maxImmedValue = this->maxImmedConstant(opCode, isSignExtended);
  if (maxImmedValue != 0)
    {
      // Now check if the constant fits
      if (intValue <= (int64_t) maxImmedValue &&
	  intValue >= -((int64_t) maxImmedValue+1))
	return true;
    }
  
  return false;
}


//---------------------------------------------------------------------------
// class MachineSchedInfo
//	Interface to machine description for instruction scheduling
//---------------------------------------------------------------------------

/*ctor*/
MachineSchedInfo::MachineSchedInfo(int                     _numSchedClasses,
				   const MachineInstrInfo* _mii,
				   const InstrClassRUsage* _classRUsages,
				   const InstrRUsageDelta* _usageDeltas,
				   const InstrIssueDelta*  _issueDeltas,
				   unsigned int		   _numUsageDeltas,
				   unsigned int		   _numIssueDeltas)
  : numSchedClasses(_numSchedClasses),
    mii(_mii),
    classRUsages(_classRUsages),
    usageDeltas(_usageDeltas),
    issueDeltas(_issueDeltas),
    numUsageDeltas(_numUsageDeltas),
    numIssueDeltas(_numIssueDeltas)
{
}

void
MachineSchedInfo::initializeResources()
{
  assert(MAX_NUM_SLOTS >= (int) getMaxNumIssueTotal()
	 && "Insufficient slots for static data! Increase MAX_NUM_SLOTS");
  
  // First, compute common resource usage info for each class because
  // most instructions will probably behave the same as their class.
  // Cannot allocate a vector of InstrRUsage so new each one.
  // 
  vector<InstrRUsage> instrRUForClasses;
  instrRUForClasses.resize(numSchedClasses);
  for (InstrSchedClass sc=0; sc < numSchedClasses; sc++)
    {
      // instrRUForClasses.push_back(new InstrRUsage);
      instrRUForClasses[sc].setMaxSlots(getMaxNumIssueTotal());
      instrRUForClasses[sc] = classRUsages[sc];
    }
  
  computeInstrResources(instrRUForClasses);
  
  computeIssueGaps(instrRUForClasses);
}


void
MachineSchedInfo::computeInstrResources(const vector<InstrRUsage>& instrRUForClasses)
{
  int numOpCodes =  mii->getNumRealOpCodes();
  instrRUsages.resize(numOpCodes);
  
  // First get the resource usage information from the class resource usages.
  for (MachineOpCode op=0; op < numOpCodes; op++)
    {
      InstrSchedClass sc = getSchedClass(op);
      assert(sc >= 0 && sc < numSchedClasses);
      instrRUsages[op] = instrRUForClasses[sc];
    }
  
  // Now, modify the resource usages as specified in the deltas.
  for (unsigned i=0; i < numUsageDeltas; i++)
    {
      MachineOpCode op = usageDeltas[i].opCode;
      assert(op < numOpCodes);
      instrRUsages[op].addUsageDelta(usageDeltas[i]);
    }
  
  // Then modify the issue restrictions as specified in the deltas.
  for (unsigned i=0; i < numIssueDeltas; i++)
    {
      MachineOpCode op = issueDeltas[i].opCode;
      assert(op < numOpCodes);
      instrRUsages[issueDeltas[i].opCode].addIssueDelta(issueDeltas[i]);
    }
}


void
MachineSchedInfo::computeIssueGaps(const vector<InstrRUsage>& instrRUForClasses)
{
  int numOpCodes =  mii->getNumRealOpCodes();
  instrRUsages.resize(numOpCodes);
  
  assert(numOpCodes < (1 << MAX_OPCODE_SIZE) - 1
	 && "numOpCodes invalid for implementation of class OpCodePair!");
  
  // First, compute issue gaps between pairs of classes based on common
  // resources usages for each class, because most instruction pairs will
  // usually behave the same as their class.
  // 
  int classPairGaps[numSchedClasses][numSchedClasses];
  for (InstrSchedClass fromSC=0; fromSC < numSchedClasses; fromSC++)
    for (InstrSchedClass toSC=0; toSC < numSchedClasses; toSC++)
      {
	int classPairGap = ComputeMinGap(instrRUForClasses[fromSC],
				      instrRUForClasses[toSC]);
	classPairGaps[fromSC][toSC] = classPairGap; 
      }
  
  // Now, for each pair of instructions, use the class pair gap if both
  // instructions have identical resource usage as their respective classes.
  // If not, recompute the gap for the pair from scratch.

  longestIssueConflict = 0;
  
  for (MachineOpCode fromOp=0; fromOp < numOpCodes; fromOp++)
    for (MachineOpCode toOp=0; toOp < numOpCodes; toOp++)
    {
      int instrPairGap = 
	(instrRUsages[fromOp].sameAsClass && instrRUsages[toOp].sameAsClass)
	? classPairGaps[getSchedClass(fromOp)][getSchedClass(toOp)]
	: ComputeMinGap(instrRUsages[fromOp], instrRUsages[toOp]);
      
      if (instrPairGap > 0)
	{
	  issueGaps[OpCodePair(fromOp,toOp)] = instrPairGap;
	  conflictLists[fromOp].push_back(toOp);
	  longestIssueConflict = max(longestIssueConflict, instrPairGap);
	}
    }
}


// Check if fromRVec and toRVec have *any* common entries.
// Assume the vectors are sorted in increasing order.
// Algorithm copied from function set_intersection() for sorted ranges (stl_algo.h).
inline static bool 
RUConflict(const vector<resourceId_t>& fromRVec,
	   const vector<resourceId_t>& toRVec)
{
  bool commonElementFound = false;
  
  unsigned fN = fromRVec.size(), tN = toRVec.size(); 
  unsigned fi = 0, ti = 0;
  while (fi < fN && ti < tN)
    if (fromRVec[fi] < toRVec[ti])
      ++fi;
    else if (toRVec[ti] < fromRVec[fi])
      ++ti;
    else
      {
	commonElementFound = true;
	break;
      }
  
  return commonElementFound; 
}


static cycles_t
ComputeMinGap(const InstrRUsage& fromRU, const InstrRUsage& toRU)
{
  cycles_t minGap = 0;
  
  if (fromRU.numBubbles > 0)
    minGap = fromRU.numBubbles;
  
  if (minGap < fromRU.numCycles)
    {
      // only need to check from cycle `minGap' onwards
      for (cycles_t gap=minGap; gap <= fromRU.numCycles-1; gap++)
	{
	  // check if instr. #2 can start executing `gap' cycles after #1
	  // by checking for resource conflicts in each overlapping cycle
	  cycles_t numOverlap = min(fromRU.numCycles - gap, toRU.numCycles);
	  for (cycles_t c = 0; c <= numOverlap-1; c++)
	    if (RUConflict(fromRU.resourcesByCycle[gap + c],
			   toRU.resourcesByCycle[c]))
	      {// conflict found so minGap must be more than `gap'
		minGap = gap+1;
		break;
	      }
	}
    }
  
  return minGap;
}

//---------------------------------------------------------------------------
