//===-- SchedInfo.cpp - Generic code to support target schedulers ----------==//
//
// This file implements the generic part of a Scheduler description for a
// target.  This functionality is defined in the llvm/Target/SchedInfo.h file.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetSchedInfo.h"
#include "llvm/Target/TargetMachine.h"

resourceId_t MachineResource::nextId = 0;

// Check if fromRVec and toRVec have *any* common entries.
// Assume the vectors are sorted in increasing order.
// Algorithm copied from function set_intersection() for sorted ranges
// (stl_algo.h).
//
inline static bool
RUConflict(const std::vector<resourceId_t>& fromRVec,
	   const std::vector<resourceId_t>& toRVec)
{
  
  unsigned fN = fromRVec.size(), tN = toRVec.size(); 
  unsigned fi = 0, ti = 0;

  while (fi < fN && ti < tN) {
    if (fromRVec[fi] < toRVec[ti])
      ++fi;
    else if (toRVec[ti] < fromRVec[fi])
      ++ti;
    else
      return true;
  }
  return false;
}


static cycles_t
ComputeMinGap(const InstrRUsage &fromRU, 
	      const InstrRUsage &toRU)
{
  cycles_t minGap = 0;
  
  if (fromRU.numBubbles > 0)
    minGap = fromRU.numBubbles;
  
  if (minGap < fromRU.numCycles) {
    // only need to check from cycle `minGap' onwards
    for (cycles_t gap=minGap; gap <= fromRU.numCycles-1; gap++) {
      // check if instr. #2 can start executing `gap' cycles after #1
      // by checking for resource conflicts in each overlapping cycle
      cycles_t numOverlap =std::min(fromRU.numCycles - gap, toRU.numCycles);
      for (cycles_t c = 0; c <= numOverlap-1; c++)
        if (RUConflict(fromRU.resourcesByCycle[gap + c],
                       toRU.resourcesByCycle[c])) {
          // conflict found so minGap must be more than `gap'
          minGap = gap+1;
          break;
        }
    }
  }
  
  return minGap;
}


//---------------------------------------------------------------------------
// class TargetSchedInfo
//	Interface to machine description for instruction scheduling
//---------------------------------------------------------------------------

TargetSchedInfo::TargetSchedInfo(const TargetMachine&    tgt,
                                 int                     NumSchedClasses,
                                 const InstrClassRUsage* ClassRUsages,
                                 const InstrRUsageDelta* UsageDeltas,
                                 const InstrIssueDelta*  IssueDeltas,
                                 unsigned NumUsageDeltas,
                                 unsigned NumIssueDeltas)
  : target(tgt),
    numSchedClasses(NumSchedClasses), mii(& tgt.getInstrInfo()),
    classRUsages(ClassRUsages), usageDeltas(UsageDeltas),
    issueDeltas(IssueDeltas), numUsageDeltas(NumUsageDeltas),
    numIssueDeltas(NumIssueDeltas)
{}

void
TargetSchedInfo::initializeResources()
{
  assert(MAX_NUM_SLOTS >= (int)getMaxNumIssueTotal()
	 && "Insufficient slots for static data! Increase MAX_NUM_SLOTS");
  
  // First, compute common resource usage info for each class because
  // most instructions will probably behave the same as their class.
  // Cannot allocate a vector of InstrRUsage so new each one.
  // 
  std::vector<InstrRUsage> instrRUForClasses;
  instrRUForClasses.resize(numSchedClasses);
  for (InstrSchedClass sc = 0; sc < numSchedClasses; sc++) {
    // instrRUForClasses.push_back(new InstrRUsage);
    instrRUForClasses[sc].setMaxSlots(getMaxNumIssueTotal());
    instrRUForClasses[sc].setTo(classRUsages[sc]);
  }
  
  computeInstrResources(instrRUForClasses);
  computeIssueGaps(instrRUForClasses);
}


void
TargetSchedInfo::computeInstrResources(const std::vector<InstrRUsage>&
					instrRUForClasses)
{
  int numOpCodes =  mii->getNumRealOpCodes();
  instrRUsages.resize(numOpCodes);
  
  // First get the resource usage information from the class resource usages.
  for (MachineOpCode op = 0; op < numOpCodes; ++op) {
    InstrSchedClass sc = getSchedClass(op);
    assert(sc < numSchedClasses);
    instrRUsages[op] = instrRUForClasses[sc];
  }
  
  // Now, modify the resource usages as specified in the deltas.
  for (unsigned i = 0; i < numUsageDeltas; ++i) {
    MachineOpCode op = usageDeltas[i].opCode;
    assert(op < numOpCodes);
    instrRUsages[op].addUsageDelta(usageDeltas[i]);
  }
  
  // Then modify the issue restrictions as specified in the deltas.
  for (unsigned i = 0; i < numIssueDeltas; ++i) {
    MachineOpCode op = issueDeltas[i].opCode;
    assert(op < numOpCodes);
    instrRUsages[issueDeltas[i].opCode].addIssueDelta(issueDeltas[i]);
  }
}


void
TargetSchedInfo::computeIssueGaps(const std::vector<InstrRUsage>&
				   instrRUForClasses)
{
  int numOpCodes =  mii->getNumRealOpCodes();
  issueGaps.resize(numOpCodes);
  conflictLists.resize(numOpCodes);

  assert(numOpCodes < (1 << MAX_OPCODE_SIZE) - 1
         && "numOpCodes invalid for implementation of class OpCodePair!");

  // First, compute issue gaps between pairs of classes based on common
  // resources usages for each class, because most instruction pairs will
  // usually behave the same as their class.
  // 
  int classPairGaps[numSchedClasses][numSchedClasses];
  for (InstrSchedClass fromSC=0; fromSC < numSchedClasses; fromSC++)
    for (InstrSchedClass toSC=0; toSC < numSchedClasses; toSC++) {
      int classPairGap = ComputeMinGap(instrRUForClasses[fromSC],
                                       instrRUForClasses[toSC]);
      classPairGaps[fromSC][toSC] = classPairGap; 
    }

  // Now, for each pair of instructions, use the class pair gap if both
  // instructions have identical resource usage as their respective classes.
  // If not, recompute the gap for the pair from scratch.

  longestIssueConflict = 0;

  for (MachineOpCode fromOp=0; fromOp < numOpCodes; fromOp++)
    for (MachineOpCode toOp=0; toOp < numOpCodes; toOp++) {
      int instrPairGap = 
        (instrRUsages[fromOp].sameAsClass && instrRUsages[toOp].sameAsClass)
        ? classPairGaps[getSchedClass(fromOp)][getSchedClass(toOp)]
        : ComputeMinGap(instrRUsages[fromOp], instrRUsages[toOp]);

      if (instrPairGap > 0) {
        this->setGap(instrPairGap, fromOp, toOp);
        conflictLists[fromOp].push_back(toOp);
        longestIssueConflict=std::max(longestIssueConflict, instrPairGap);
      }
    }
}


void InstrRUsage::setTo(const InstrClassRUsage& classRU) {
  sameAsClass	= true;
  isSingleIssue = classRU.isSingleIssue;
  breaksGroup   = classRU.breaksGroup; 
  numBubbles    = classRU.numBubbles;
  
  for (unsigned i=0; i < classRU.numSlots; i++) {
    unsigned slot = classRU.feasibleSlots[i];
    assert(slot < feasibleSlots.size() && "Invalid slot specified!");
    this->feasibleSlots[slot] = true;
  }
  
  numCycles   = classRU.totCycles;
  resourcesByCycle.resize(this->numCycles);
  
  for (unsigned i=0; i < classRU.numRUEntries; i++)
    for (unsigned c=classRU.V[i].startCycle, NC = c + classRU.V[i].numCycles;
	 c < NC; c++)
      this->resourcesByCycle[c].push_back(classRU.V[i].resourceId);
  
  // Sort each resource usage vector by resourceId_t to speed up conflict
  // checking
  for (unsigned i=0; i < this->resourcesByCycle.size(); i++)
    sort(resourcesByCycle[i].begin(), resourcesByCycle[i].end());
}

// Add the extra resource usage requirements specified in the delta.
// Note that a negative value of `numCycles' means one entry for that
// resource should be deleted for each cycle.
// 
void InstrRUsage::addUsageDelta(const InstrRUsageDelta &delta) {
  int NC = delta.numCycles;
  sameAsClass = false;
  
  // resize the resources vector if more cycles are specified
  unsigned maxCycles = this->numCycles;
  maxCycles = std::max(maxCycles, delta.startCycle + abs(NC) - 1);
  if (maxCycles > this->numCycles) {
    this->resourcesByCycle.resize(maxCycles);
    this->numCycles = maxCycles;
  }
    
  if (NC >= 0)
    for (unsigned c=delta.startCycle, last=c+NC-1; c <= last; c++)
      this->resourcesByCycle[c].push_back(delta.resourceId);
  else
    // Remove the resource from all NC cycles.
    for (unsigned c=delta.startCycle, last=(c-NC)-1; c <= last; c++) {
      // Look for the resource backwards so we remove the last entry
      // for that resource in each cycle.
      std::vector<resourceId_t>& rvec = this->resourcesByCycle[c];
      int r;
      for (r = rvec.size() - 1; r >= 0; r--)
        if (rvec[r] == delta.resourceId) {
          // found last entry for the resource
          rvec.erase(rvec.begin() + r);
          break;
        }
      assert(r >= 0 && "Resource to remove was unused in cycle c!");
    }
}
