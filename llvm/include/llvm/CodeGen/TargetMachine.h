// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	TargetMachine.h
// 
// Purpose:
//	
// History:
//	7/12/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#ifndef LLVM_CODEGEN_TARGETMACHINE_H
#define LLVM_CODEGEN_TARGETMACHINE_H

//*********************** System Include Files *****************************/

#include <string>
#include <vector>
#include <hash_map>
#include <hash_set>
#include <algorithm>

//************************ User Include Files *****************************/

#include "llvm/CodeGen/TargetData.h"
#include "llvm/Support/NonCopyable.h"
#include "llvm/Support/DataTypes.h"

//************************ Opaque Declarations*****************************/

class Type;
class StructType;
struct MachineInstrDescriptor;
class TargetMachine;

//************************ Exported Data Types *****************************/

//---------------------------------------------------------------------------
// Data types used to define information about a single machine instruction
//---------------------------------------------------------------------------

typedef int MachineOpCode;
typedef int OpCodeMask;
typedef int InstrSchedClass;

static const unsigned MAX_OPCODE_SIZE = 16;

typedef long long cycles_t; 
const cycles_t HUGE_LATENCY = ~((unsigned long long) 1 << sizeof(cycles_t)-1);
const cycles_t INVALID_LATENCY = -HUGE_LATENCY; 


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


template <> struct hash<OpCodePair> {
  size_t operator()(const OpCodePair& pair) const {
    return hash<long>()(pair.val);
  }
};


// Global variable holding an array of descriptors for machine instructions.
// The actual object needs to be created separately for each target machine.
// This variable is initialized and reset by class MachineInstrInfo.
// 
extern const MachineInstrDescriptor* TargetInstrDescriptors;


//---------------------------------------------------------------------------
// struct MachineInstrDescriptor:
//	Predefined information about each machine instruction.
//	Designed to initialized statically.
// 
// class MachineInstructionInfo
//	Interface to description of machine instructions
// 
//---------------------------------------------------------------------------


const unsigned int	M_NOP_FLAG		= 1;
const unsigned int	M_BRANCH_FLAG		= 1 << 1;
const unsigned int	M_CALL_FLAG		= 1 << 2;
const unsigned int	M_RET_FLAG		= 1 << 3;
const unsigned int	M_ARITH_FLAG		= 1 << 4;
const unsigned int	M_CC_FLAG		= 1 << 6;
const unsigned int	M_LOGICAL_FLAG		= 1 << 6;
const unsigned int	M_INT_FLAG		= 1 << 7;
const unsigned int	M_FLOAT_FLAG		= 1 << 8;
const unsigned int	M_CONDL_FLAG		= 1 << 9;
const unsigned int	M_LOAD_FLAG		= 1 << 10;
const unsigned int	M_PREFETCH_FLAG		= 1 << 11;
const unsigned int	M_STORE_FLAG		= 1 << 12;
const unsigned int	M_DUMMY_PHI_FLAG	= 1 << 13;


struct MachineInstrDescriptor {
  string	opCodeString;	// Assembly language mnemonic for the opcode.
  int		numOperands;	// Number of args; -1 if variable #args
  int		resultPos;	// Position of the result; -1 if no result
  unsigned int	maxImmedConst;	// Largest +ve constant in IMMMED field or 0.
  bool		immedIsSignExtended;	// Is IMMED field sign-extended? If so,
				//   smallest -ve value is -(maxImmedConst+1).
  unsigned int  numDelaySlots;	// Number of delay slots after instruction
  unsigned int  latency;	// Latency in machine cycles
  InstrSchedClass schedClass;	// enum  identifying instr sched class
  unsigned int	  iclass;	// flags identifying machine instr class
};


class MachineInstrInfo : public NonCopyableV {
protected:
  const MachineInstrDescriptor* desc;	// raw array to allow static init'n
  unsigned int descSize;		// number of entries in the desc array
  unsigned int numRealOpCodes;		// number of non-dummy op codes
  
public:
  /*ctor*/		MachineInstrInfo(const MachineInstrDescriptor* _desc,
					 unsigned int _descSize,
					 unsigned int _numRealOpCodes);
  /*dtor*/ virtual	~MachineInstrInfo();
  
  unsigned int		getNumRealOpCodes() const {
    return numRealOpCodes;
  }
  
  unsigned int		getNumTotalOpCodes() const {
    return descSize;
  }
  
  const MachineInstrDescriptor& getDescriptor(MachineOpCode opCode) const {
    assert(opCode >= 0 && opCode < (int) descSize);
    return desc[opCode];
  }
  
  int			getNumOperands	(MachineOpCode opCode) const {
    return getDescriptor(opCode).numOperands;
  }
  
  int			getResultPos	(MachineOpCode opCode) const {
    return getDescriptor(opCode).resultPos;
  }
  
  unsigned int		getNumDelaySlots(MachineOpCode opCode) const {
    return getDescriptor(opCode).numDelaySlots;
  }
  
  InstrSchedClass	getSchedClass	(MachineOpCode opCode) const {
    return getDescriptor(opCode).schedClass;
  }
  
  //
  // Query instruction class flags according to the machine-independent
  // flags listed above.
  // 
  unsigned int	getIClass		(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass;
  }
  bool		isNop			(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_NOP_FLAG;
  }
  bool		isBranch		(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_BRANCH_FLAG;
  }
  bool		isCall			(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_CALL_FLAG;
  }
  bool		isReturn		(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_RET_FLAG;
  }
  bool		isControlFlow		(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_BRANCH_FLAG
        || getDescriptor(opCode).iclass & M_CALL_FLAG
        || getDescriptor(opCode).iclass & M_RET_FLAG;
  }
  bool		isArith			(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_RET_FLAG;
  }
  bool		isCCInstr		(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_CC_FLAG;
  }
  bool		isLogical		(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_LOGICAL_FLAG;
  }
  bool		isIntInstr		(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_INT_FLAG;
  }
  bool		isFloatInstr		(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_FLOAT_FLAG;
  }
  bool		isConditional		(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_CONDL_FLAG;
  }
  bool		isLoad			(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_LOAD_FLAG;
  }
  bool		isPrefetch		(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_PREFETCH_FLAG;
  }
  bool		isLoadOrPrefetch	(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_LOAD_FLAG
        || getDescriptor(opCode).iclass & M_PREFETCH_FLAG;
  }
  bool		isStore			(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_STORE_FLAG;
  }
  bool		isMemoryAccess		(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_LOAD_FLAG
        || getDescriptor(opCode).iclass & M_PREFETCH_FLAG
        || getDescriptor(opCode).iclass & M_STORE_FLAG;
  }
  bool		isDummyPhiInstr		(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_DUMMY_PHI_FLAG;
  }
  
  // 
  // Check if an instruction can be issued before its operands are ready,
  // or if a subsequent instruction that uses its result can be issued
  // before the results are ready.
  // Default to true since most instructions on many architectures allow this.
  // 
  virtual bool		hasOperandInterlock(MachineOpCode opCode) const {
    return true;
  }
  
  virtual bool		hasResultInterlock(MachineOpCode opCode) const {
    return true;
  }
  
  // 
  // Latencies for individual instructions and instruction pairs
  // 
  virtual int		minLatency	(MachineOpCode opCode) const {
    return getDescriptor(opCode).latency;
  }
  
  virtual int		maxLatency	(MachineOpCode opCode) const {
    return getDescriptor(opCode).latency;
  }
  
  // Check if the specified constant fits in the immediate field
  // of this machine instruction
  // 
  virtual bool		constantFitsInImmedField(MachineOpCode opCode,
						 int64_t intValue) const;
  
  // Return the largest +ve constant that can be held in the IMMMED field
  // of this machine instruction.
  // isSignExtended is set to true if the value is sign-extended before use
  // (this is true for all immediate fields in SPARC instructions).
  // Return 0 if the instruction has no IMMED field.
  // 
  virtual uint64_t	maxImmedConstant(MachineOpCode opCode,
					 bool& isSignExtended) const {
    isSignExtended = getDescriptor(opCode).immedIsSignExtended;
    return getDescriptor(opCode).maxImmedConst;
  }
};


//---------------------------------------------------------------------------
// class MachineResource 
// class CPUResource
// 
// Purpose:
//   Representation of a single machine resource used in specifying
//   resource usages of machine instructions for scheduling.
//---------------------------------------------------------------------------


typedef unsigned int resourceId_t;

class MachineResource {
public:
  const string	rname;
  resourceId_t	rid;
  
  /*ctor*/	MachineResource(const string& resourceName)
			: rname(resourceName), rid(nextId++) {}
  
private:
  static resourceId_t nextId;
  MachineResource();			// disable
};


class CPUResource : public MachineResource {
public:
  int		maxNumUsers;		// MAXINT if no restriction
  
  /*ctor*/	CPUResource(const string& rname, int maxUsers)
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
  unsigned int	maxNumIssue;
  bool		isSingleIssue;
  bool		breaksGroup;
  cycles_t	numBubbles;
  
  // Feasible slots to use for instructions in this class.
  // The size of vector S[] is `numSlots'.
  unsigned int	numSlots;
  unsigned int	feasibleSlots[MAX_NUM_SLOTS];
  
  // Resource usages common to instructions in this class.
  // The size of vector V[] is `numRUEntries'.
  unsigned int	numRUEntries;
  struct {
    resourceId_t resourceId;
    unsigned int startCycle;
    int		 numCycles;
  }		V[MAX_NUM_CYCLES];
};

struct InstrRUsageDelta {
  MachineOpCode opCode;
  resourceId_t	resourceId;
  unsigned int	startCycle;
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
  /*ctor*/	InstrRUsage	() {}
  /*ctor*/	InstrRUsage	(const InstrRUsage& instrRU);
  InstrRUsage&	operator=	(const InstrRUsage& instrRU);
  
  bool		sameAsClass;
  
  // Issue restrictions for this instruction
  bool		isSingleIssue;
  bool		breaksGroup;
  cycles_t	numBubbles;
  
  // Feasible slots to use for this instruction.
  vector<bool>	feasibleSlots;
  
  // Resource usages for this instruction, with one resource vector per cycle.
  cycles_t	numCycles;
  vector<vector<resourceId_t> > resourcesByCycle;
  
private:
  // Conveniences for initializing this structure
  InstrRUsage&	operator=	(const InstrClassRUsage& classRU);
  void		addIssueDelta	(const InstrIssueDelta& delta);
  void		addUsageDelta	(const InstrRUsageDelta& delta);
  void		setMaxSlots	(int maxNumSlots);
  
  friend class MachineSchedInfo;	// give access to these functions
};


inline void
InstrRUsage::setMaxSlots(int maxNumSlots)
{
  feasibleSlots.resize(maxNumSlots);
}

inline InstrRUsage&
InstrRUsage::operator=(const InstrRUsage& instrRU)
{
  sameAsClass	   = instrRU.sameAsClass;
  isSingleIssue    = instrRU.isSingleIssue;
  breaksGroup      = instrRU.breaksGroup; 
  numBubbles       = instrRU.numBubbles;
  feasibleSlots    = instrRU.feasibleSlots;
  numCycles	   = instrRU.numCycles;
  resourcesByCycle = instrRU.resourcesByCycle;
  return *this;
}

inline /*ctor*/
InstrRUsage::InstrRUsage(const InstrRUsage& instrRU)
{
  *this = instrRU;
}

inline InstrRUsage&
InstrRUsage::operator=(const InstrClassRUsage& classRU)
{
  sameAsClass	= true;
  isSingleIssue = classRU.isSingleIssue;
  breaksGroup   = classRU.breaksGroup; 
  numBubbles    = classRU.numBubbles;
  
  for (unsigned i=0; i < classRU.numSlots; i++)
    {
      unsigned slot = classRU.feasibleSlots[i];
      assert(slot < feasibleSlots.size() && "Invalid slot specified!");
      this->feasibleSlots[slot] = true;
    }
  
  this->numCycles   = classRU.totCycles;
  this->resourcesByCycle.resize(this->numCycles);
  
  for (unsigned i=0; i < classRU.numRUEntries; i++)
    for (unsigned c=classRU.V[i].startCycle, NC = c + classRU.V[i].numCycles;
	 c < NC; c++)
      this->resourcesByCycle[c].push_back(classRU.V[i].resourceId);
  
  // Sort each resource usage vector by resourceId_t to speed up conflict checking
  for (unsigned i=0; i < this->resourcesByCycle.size(); i++)
    sort(resourcesByCycle[i].begin(), resourcesByCycle[i].end());
  
  return *this;
}


inline void
InstrRUsage::addIssueDelta(const InstrIssueDelta&  delta)
{
  sameAsClass = false;
  isSingleIssue = delta.isSingleIssue;
  breaksGroup = delta.breaksGroup;
  numBubbles = delta.numBubbles;
}


// Add the extra resource usage requirements specified in the delta.
// Note that a negative value of `numCycles' means one entry for that
// resource should be deleted for each cycle.
// 
inline void
InstrRUsage::addUsageDelta(const InstrRUsageDelta& delta)
{
  int NC = delta.numCycles;
    
  this->sameAsClass = false;
  
  // resize the resources vector if more cycles are specified
  unsigned maxCycles = this->numCycles;
  maxCycles = max(maxCycles, delta.startCycle + abs(NC) - 1);
  if (maxCycles > this->numCycles)
    {
      this->resourcesByCycle.resize(maxCycles);
      this->numCycles = maxCycles;
    }
    
  if (NC >= 0)
    for (unsigned c=delta.startCycle, last=c+NC-1; c <= last; c++)
      this->resourcesByCycle[c].push_back(delta.resourceId);
  else
    // Remove the resource from all NC cycles.
    for (unsigned c=delta.startCycle, last=(c-NC)-1; c <= last; c++)
      {
	// Look for the resource backwards so we remove the last entry
	// for that resource in each cycle.
	vector<resourceId_t>& rvec = this->resourcesByCycle[c];
	int r;
	for (r = (int) rvec.size(); r >= 0; r--)
	  if (rvec[r] == delta.resourceId)
	    {// found last entry for the resource
	      rvec.erase(rvec.begin() + r);
	      break;
	    }
	assert(r >= 0 && "Resource to remove was unused in cycle c!");
      }
}


//---------------------------------------------------------------------------
// class MachineSchedInfo
//
// Purpose:
//   Common interface to machine information for instruction scheduling
//---------------------------------------------------------------------------

class MachineSchedInfo : public NonCopyableV {
public:
  unsigned int	maxNumIssueTotal;
  int	longestIssueConflict;
  
  int	branchMispredictPenalty;	// 4 for SPARC IIi
  int	branchTargetUnknownPenalty;	// 2 for SPARC IIi
  int   l1DCacheMissPenalty;		// 7 or 9 for SPARC IIi
  int   l1ICacheMissPenalty;		// ? for SPARC IIi
  
  bool	inOrderLoads ;			// true for SPARC IIi
  bool	inOrderIssue;			// true for SPARC IIi
  bool	inOrderExec;			// false for most architectures
  bool	inOrderRetire;			// true for most architectures
  
protected:
  inline const InstrRUsage& getInstrRUsage(MachineOpCode opCode) const {
    assert(opCode >= 0 && opCode < (int) instrRUsages.size());
    return instrRUsages[opCode];
  }
  inline const InstrClassRUsage&
			getClassRUsage(const InstrSchedClass& sc) const {
    assert(sc >= 0 && sc < numSchedClasses);
    return classRUsages[sc];
  }
  
public:
  /*ctor*/	   MachineSchedInfo	(int _numSchedClasses,
					 const MachineInstrInfo* _mii,
					 const InstrClassRUsage* _classRUsages,
					 const InstrRUsageDelta* _usageDeltas,
					 const InstrIssueDelta*  _issueDeltas,
					 unsigned int _numUsageDeltas,
					 unsigned int _numIssueDeltas);
  /*dtor*/ virtual ~MachineSchedInfo	() {}
  
  inline const MachineInstrInfo& getInstrInfo() const {
    return *mii;
  }
  
  inline int		getNumSchedClasses()  const {
    return numSchedClasses;
  }  
  
  inline  unsigned int	getMaxNumIssueTotal() const {
    return maxNumIssueTotal;
  }
  
  inline  unsigned int	getMaxIssueForClass(const InstrSchedClass& sc) const {
    assert(sc >= 0 && sc < numSchedClasses);
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
    hash_map<OpCodePair,int>::const_iterator
      I = issueGaps.find(OpCodePair(fromOp, toOp));
    return (I == issueGaps.end())? 0 : (*I).second;
  }
  
  inline const vector<MachineOpCode>*
		getConflictList(MachineOpCode opCode) const {
    hash_map<MachineOpCode,vector<MachineOpCode> >::const_iterator
      I = conflictLists.find(opCode);
    return (I == conflictLists.end())? NULL : & (*I).second;
  }
  
  inline  bool	isSingleIssue		(MachineOpCode opCode) const {
    return getInstrRUsage(opCode).isSingleIssue;
  }
  
  inline  bool	breaksIssueGroup	(MachineOpCode opCode) const {
    return getInstrRUsage(opCode).breaksGroup;
  }
  
  inline  unsigned int 	numBubblesAfter	(MachineOpCode opCode) const {
    return getInstrRUsage(opCode).numBubbles;
  }
  
protected:
  virtual void	initializeResources	();
  
private:
  void computeInstrResources(const vector<InstrRUsage>& instrRUForClasses);
  void computeIssueGaps(const vector<InstrRUsage>& instrRUForClasses);
  
protected:
  int		           numSchedClasses;
  const MachineInstrInfo*  mii;
  const	InstrClassRUsage*  classRUsages;	// raw array by sclass
  const	InstrRUsageDelta*  usageDeltas;		// raw array [1:numUsageDeltas]
  const InstrIssueDelta*   issueDeltas;		// raw array [1:numIssueDeltas]
  unsigned int		   numUsageDeltas;
  unsigned int		   numIssueDeltas;
  
  vector<InstrRUsage>      instrRUsages;	// indexed by opcode
  hash_map<OpCodePair,int> issueGaps;		// indexed by opcode pair
  hash_map<MachineOpCode,vector<MachineOpCode> >
			   conflictLists;	// indexed by opcode
};


//---------------------------------------------------------------------------
// class TargetMachine
// 
// Purpose:
//   Primary interface to machine description for the target machine.
// 
//---------------------------------------------------------------------------

class TargetMachine : public NonCopyableV {
public:
  const string     TargetName;
  const TargetData DataLayout;		// Calculates type size & alignment
  int              optSizeForSubWordData;
  int	           minMemOpWordSize;
  int	           maxAtomicMemOpWordSize;
  
  // Register information.  This needs to be reorganized into a single class.
  int		zeroRegNum;	// register that gives 0 if any (-1 if none)
  
public:
  /*ctor*/ TargetMachine(const string &targetname,
			 unsigned char PtrSize = 8, unsigned char PtrAl = 8,
			 unsigned char DoubleAl = 8, unsigned char FloatAl = 4,
			 unsigned char LongAl = 8, unsigned char IntAl = 4,
			 unsigned char ShortAl = 2, unsigned char ByteAl = 1)
    : TargetName(targetname), DataLayout(targetname, PtrSize, PtrAl,
					 DoubleAl, FloatAl, LongAl, IntAl,
					 ShortAl, ByteAl)
				    {}
  
  /*dtor*/ virtual ~TargetMachine() {}
  
  const MachineInstrInfo& getInstrInfo	() const { return *machineInstrInfo; }
  
  const MachineSchedInfo& getSchedInfo() const { return *machineSchedInfo; }
  
  virtual unsigned int	findOptimalStorageSize	(const Type* ty) const;
  
  // This really should be in the register info class
  virtual bool		regsMayBeAliased	(unsigned int regNum1,
						 unsigned int regNum2) const {
    return (regNum1 == regNum2);
  }
  
protected:
  // Description of machine instructions
  // Protect so that subclass can control alloc/dealloc
  MachineInstrInfo* machineInstrInfo;
  MachineSchedInfo* machineSchedInfo;
};

//**************************************************************************/

#endif
