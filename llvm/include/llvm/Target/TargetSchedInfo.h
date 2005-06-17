//===- Target/TargetSchedInfo.h - Target Instruction Sched Info -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the target machine to the instruction scheduler.
//
// NOTE: This file is currently sparc V9 specific.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETSCHEDINFO_H
#define LLVM_TARGET_TARGETSCHEDINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/hash_map"
#include <string>

namespace llvm {

typedef long long CycleCount_t;
static const CycleCount_t HUGE_LATENCY = ~((long long) 1 << (sizeof(CycleCount_t)-2));
static const CycleCount_t INVALID_LATENCY = -HUGE_LATENCY;

//---------------------------------------------------------------------------
// class MachineResource
// class CPUResource
//
// Purpose:
//   Representation of a single machine resource used in specifying
//   resource usages of machine instructions for scheduling.
//---------------------------------------------------------------------------


typedef unsigned resourceId_t;

struct CPUResource {
  const std::string rname;
  resourceId_t rid;
  int maxNumUsers;   // MAXINT if no restriction

  CPUResource(const std::string& resourceName, int maxUsers);
  static CPUResource* getCPUResource(resourceId_t id);
private:
  static resourceId_t nextId;
};


//---------------------------------------------------------------------------
// struct InstrClassRUsage
// struct InstrRUsageDelta
// struct InstrIssueDelta
// struct InstrRUsage
//
// Purpose:
//   The first three are structures used to specify machine resource
//   usages for each instruction in a machine description file:
//    InstrClassRUsage : resource usages common to all instrs. in a class
//    InstrRUsageDelta : add/delete resource usage for individual instrs.
//    InstrIssueDelta  : add/delete instr. issue info for individual instrs
//
//   The last one (InstrRUsage) is the internal representation of
//   instruction resource usage constructed from the above three.
//---------------------------------------------------------------------------

const int MAX_NUM_SLOTS  = 32;
const int MAX_NUM_CYCLES = 32;

struct InstrClassRUsage {
  InstrSchedClass schedClass;
  int             totCycles;

  // Issue restrictions common to instructions in this class
  unsigned      maxNumIssue;
  bool          isSingleIssue;
  bool          breaksGroup;
  CycleCount_t  numBubbles;

  // Feasible slots to use for instructions in this class.
  // The size of vector S[] is `numSlots'.
  unsigned      numSlots;
  unsigned      feasibleSlots[MAX_NUM_SLOTS];

  // Resource usages common to instructions in this class.
  // The size of vector V[] is `numRUEntries'.
  unsigned      numRUEntries;
  struct {
    resourceId_t resourceId;
    unsigned    startCycle;
    int         numCycles;
  } V[MAX_NUM_CYCLES];
};

struct InstrRUsageDelta {
  MachineOpCode opCode;
  resourceId_t  resourceId;
  unsigned      startCycle;
  int  numCycles;
};

// Specify instruction issue restrictions for individual instructions
// that differ from the common rules for the class.
//
struct InstrIssueDelta {
  MachineOpCode opCode;
  bool isSingleIssue;
  bool breaksGroup;
  CycleCount_t numBubbles;
};


struct InstrRUsage {
  bool  sameAsClass;

  // Issue restrictions for this instruction
  bool  isSingleIssue;
  bool  breaksGroup;
  CycleCount_t numBubbles;

  // Feasible slots to use for this instruction.
  std::vector<bool> feasibleSlots;

  // Resource usages for this instruction, with one resource vector per cycle.
  CycleCount_t numCycles;
  std::vector<std::vector<resourceId_t> > resourcesByCycle;

private:
  // Conveniences for initializing this structure
  void setTo(const InstrClassRUsage& classRU);

  void addIssueDelta(const InstrIssueDelta& delta) {
    sameAsClass = false;
    isSingleIssue = delta.isSingleIssue;
    breaksGroup = delta.breaksGroup;
    numBubbles = delta.numBubbles;
  }

  void addUsageDelta(const InstrRUsageDelta& delta);
  void setMaxSlots(int maxNumSlots) {
    feasibleSlots.resize(maxNumSlots);
  }

  friend class TargetSchedInfo; // give access to these functions
};


//---------------------------------------------------------------------------
/// TargetSchedInfo - Common interface to machine information for
/// instruction scheduling
///
class TargetSchedInfo {
public:
  const TargetMachine& target;

  unsigned maxNumIssueTotal;
  int longestIssueConflict;

protected:
  inline const InstrRUsage& getInstrRUsage(MachineOpCode opCode) const {
    assert(opCode >= 0 && opCode < (int) instrRUsages.size());
    return instrRUsages[opCode];
  }
  const InstrClassRUsage& getClassRUsage(const InstrSchedClass& sc) const {
    assert(sc < numSchedClasses);
    return classRUsages[sc];
  }

private:
  TargetSchedInfo(const TargetSchedInfo &);  // DO NOT IMPLEMENT
  void operator=(const TargetSchedInfo &);  // DO NOT IMPLEMENT
public:
  TargetSchedInfo(const TargetMachine& tgt,
                  int _numSchedClasses,
                  const InstrClassRUsage* _classRUsages,
                  const InstrRUsageDelta* _usageDeltas,
                  const InstrIssueDelta*  _issueDeltas,
                  unsigned _numUsageDeltas,
                  unsigned _numIssueDeltas);
  virtual ~TargetSchedInfo() {}

  inline const TargetInstrInfo& getInstrInfo() const {
    return *mii;
  }

  inline int getNumSchedClasses()  const {
    return numSchedClasses;
  }

  inline  unsigned getMaxNumIssueTotal() const {
    return maxNumIssueTotal;
  }

  inline  unsigned getMaxIssueForClass(const InstrSchedClass& sc) const {
    assert(sc < numSchedClasses);
    return classRUsages[sc].maxNumIssue;
  }

  inline InstrSchedClass getSchedClass(MachineOpCode opCode) const {
    return getInstrInfo().getSchedClass(opCode);
  }

  inline  bool instrCanUseSlot(MachineOpCode opCode,
                               unsigned s) const {
    assert(s < getInstrRUsage(opCode).feasibleSlots.size() && "Invalid slot!");
    return getInstrRUsage(opCode).feasibleSlots[s];
  }

  inline int getLongestIssueConflict() const {
    return longestIssueConflict;
  }

  inline  int getMinIssueGap(MachineOpCode fromOp,
                             MachineOpCode toOp)   const {
    assert(fromOp < (int) issueGaps.size());
    const std::vector<int>& toGaps = issueGaps[fromOp];
    return (toOp < (int) toGaps.size())? toGaps[toOp] : 0;
  }

  inline const std::vector<MachineOpCode>&
    getConflictList(MachineOpCode opCode) const {
    assert(opCode < (int) conflictLists.size());
    return conflictLists[opCode];
  }

  inline  bool isSingleIssue(MachineOpCode opCode) const {
    return getInstrRUsage(opCode).isSingleIssue;
  }

  inline  bool breaksIssueGroup(MachineOpCode opCode) const {
    return getInstrRUsage(opCode).breaksGroup;
  }

  inline  unsigned numBubblesAfter(MachineOpCode opCode) const {
    return getInstrRUsage(opCode).numBubbles;
  }

  inline unsigned getCPUResourceNum(int rd)const{
    for(unsigned i=0;i<resourceNumVector.size();i++){
      if(resourceNumVector[i].first == rd) return resourceNumVector[i].second;
    }
    assert( 0&&"resource not found");
    return 0;
  }


protected:
  virtual void initializeResources();

private:
  void computeInstrResources(const std::vector<InstrRUsage>& instrRUForClasses);
  void computeIssueGaps(const std::vector<InstrRUsage>& instrRUForClasses);

  void setGap(int gap, MachineOpCode fromOp, MachineOpCode toOp) {
    std::vector<int>& toGaps = issueGaps[fromOp];
    if (toOp >= (int) toGaps.size())
      toGaps.resize(toOp+1);
    toGaps[toOp] = gap;
  }

public:
  std::vector<std::pair<int,int> > resourceNumVector;

protected:
  unsigned           numSchedClasses;
  const TargetInstrInfo*   mii;
  const InstrClassRUsage*  classRUsages;        // raw array by sclass
  const InstrRUsageDelta*  usageDeltas;         // raw array [1:numUsageDeltas]
  const InstrIssueDelta*   issueDeltas;         // raw array [1:numIssueDeltas]
  unsigned      numUsageDeltas;
  unsigned      numIssueDeltas;

  std::vector<InstrRUsage> instrRUsages;    // indexed by opcode
  std::vector<std::vector<int> > issueGaps; // indexed by [opcode1][opcode2]
  std::vector<std::vector<MachineOpCode> >
      conflictLists;   // indexed by [opcode]


  friend class ModuloSchedulingPass;
  friend class ModuloSchedulingSBPass;
  friend class MSSchedule;
  friend class MSScheduleSB;
  friend class MSchedGraphSB;

};

} // End llvm namespace

#endif
