//===- Target/TargetSchedInfo.h - Target Instruction Sched Info --*- C++ -*-==//
//
// This file describes the target machine to the instruction scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETSCHEDINFO_H
#define LLVM_TARGET_TARGETSCHEDINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "Support/hash_map"
#include <string>

typedef long long cycles_t; 
static const cycles_t HUGE_LATENCY = ~((long long) 1 << (sizeof(cycles_t)-2));
static const cycles_t INVALID_LATENCY = -HUGE_LATENCY; 
static const unsigned MAX_OPCODE_SIZE = 16;

class OpCodePair {
public:
  long val;			// make long by concatenating two opcodes
  OpCodePair(MachineOpCode op1, MachineOpCode op2)
    : val((op1 < 0 || op2 < 0)?
	-1 : (long)((((unsigned) op1) << MAX_OPCODE_SIZE) | (unsigned) op2)) {}
  bool operator==(const OpCodePair& op) const {
    return val == op.val;
  }
private:
  OpCodePair();			// disable for now
};

namespace HASH_NAMESPACE {
  template <> struct hash<OpCodePair> {
    size_t operator()(const OpCodePair& pair) const {
      return hash<long>()(pair.val);
    }
  };
}

//---------------------------------------------------------------------------
// class MachineResource 
// class CPUResource
// 
// Purpose:
//   Representation of a single machine resource used in specifying
//   resource usages of machine instructions for scheduling.
//---------------------------------------------------------------------------


typedef unsigned resourceId_t;

struct MachineResource {
  const std::string rname;
  resourceId_t rid;
  
  MachineResource(const std::string &resourceName)
    : rname(resourceName), rid(nextId++) {}
  
private:
  static resourceId_t nextId;
  MachineResource();			// disable
};


struct CPUResource : public MachineResource {
  int maxNumUsers;   // MAXINT if no restriction
  
  CPUResource(const std::string& rname, int maxUsers)
    : MachineResource(rname), maxNumUsers(maxUsers) {}
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
  int		totCycles;
  
  // Issue restrictions common to instructions in this class
  unsigned      maxNumIssue;
  bool	        isSingleIssue;
  bool	        breaksGroup;
  cycles_t      numBubbles;
  
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
    int	        numCycles;
  } V[MAX_NUM_CYCLES];
};

struct InstrRUsageDelta {
  MachineOpCode opCode;
  resourceId_t	resourceId;
  unsigned      startCycle;
  int		numCycles;
};

// Specify instruction issue restrictions for individual instructions
// that differ from the common rules for the class.
// 
struct InstrIssueDelta {
  MachineOpCode	opCode;
  bool		isSingleIssue;
  bool		breaksGroup;
  cycles_t	numBubbles;
};


struct InstrRUsage {
  bool		sameAsClass;
  
  // Issue restrictions for this instruction
  bool		isSingleIssue;
  bool		breaksGroup;
  cycles_t	numBubbles;
  
  // Feasible slots to use for this instruction.
  std::vector<bool> feasibleSlots;
  
  // Resource usages for this instruction, with one resource vector per cycle.
  cycles_t	numCycles;
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

  void addUsageDelta	(const InstrRUsageDelta& delta);
  void setMaxSlots	(int maxNumSlots) {
    feasibleSlots.resize(maxNumSlots);
  }
  
  friend class TargetSchedInfo;	// give access to these functions
};


//---------------------------------------------------------------------------
/// TargetSchedInfo - Common interface to machine information for 
/// instruction scheduling
///
struct TargetSchedInfo {
  const TargetMachine& target;
  
  unsigned maxNumIssueTotal;
  int	longestIssueConflict;
  
  int	branchMispredictPenalty;	// 4 for SPARC IIi
  int	branchTargetUnknownPenalty;	// 2 for SPARC IIi
  int   l1DCacheMissPenalty;		// 7 or 9 for SPARC IIi
  int   l1ICacheMissPenalty;		// ? for SPARC IIi
  
  bool	inOrderLoads;			// true for SPARC IIi
  bool	inOrderIssue;			// true for SPARC IIi
  bool	inOrderExec;			// false for most architectures
  bool	inOrderRetire;			// true for most architectures
  
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
  /*ctor*/	   TargetSchedInfo	(const TargetMachine& tgt,
                                         int                  _numSchedClasses,
					 const InstrClassRUsage* _classRUsages,
					 const InstrRUsageDelta* _usageDeltas,
					 const InstrIssueDelta*  _issueDeltas,
					 unsigned _numUsageDeltas,
					 unsigned _numIssueDeltas);
  /*dtor*/ virtual ~TargetSchedInfo() {}
  
  inline const TargetInstrInfo& getInstrInfo() const {
    return *mii;
  }
  
  inline int		getNumSchedClasses()  const {
    return numSchedClasses;
  }  
  
  inline  unsigned getMaxNumIssueTotal() const {
    return maxNumIssueTotal;
  }
  
  inline  unsigned getMaxIssueForClass(const InstrSchedClass& sc) const {
    assert(sc < numSchedClasses);
    return classRUsages[sc].maxNumIssue;
  }

  inline InstrSchedClass getSchedClass	(MachineOpCode opCode) const {
    return getInstrInfo().getSchedClass(opCode);
  } 
  
  inline  bool	instrCanUseSlot		(MachineOpCode opCode,
					 unsigned s) const {
    assert(s < getInstrRUsage(opCode).feasibleSlots.size() && "Invalid slot!");
    return getInstrRUsage(opCode).feasibleSlots[s];
  }
  
  inline int	getLongestIssueConflict	() const {
    return longestIssueConflict;
  }

  inline  int 	getMinIssueGap		(MachineOpCode fromOp,
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

  inline  bool	isSingleIssue		(MachineOpCode opCode) const {
    return getInstrRUsage(opCode).isSingleIssue;
  }
  
  inline  bool	breaksIssueGroup	(MachineOpCode opCode) const {
    return getInstrRUsage(opCode).breaksGroup;
  }
  
  inline  unsigned numBubblesAfter	(MachineOpCode opCode) const {
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
  virtual void	initializeResources	();
  
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
  std::vector<pair<int,int> > resourceNumVector;
  
protected:
  unsigned	           numSchedClasses;
  const TargetInstrInfo*   mii;
  const	InstrClassRUsage*  classRUsages;        // raw array by sclass
  const	InstrRUsageDelta*  usageDeltas;	        // raw array [1:numUsageDeltas]
  const InstrIssueDelta*   issueDeltas;	        // raw array [1:numIssueDeltas]
  unsigned 		   numUsageDeltas;
  unsigned 		   numIssueDeltas;
  
  std::vector<InstrRUsage> instrRUsages;    // indexed by opcode
  std::vector<std::vector<int> > issueGaps; // indexed by [opcode1][opcode2]
  std::vector<std::vector<MachineOpCode> >
			   conflictLists;   // indexed by [opcode]



  friend class ModuloSchedGraph;
  friend class ModuloScheduling;
  
};

#endif
