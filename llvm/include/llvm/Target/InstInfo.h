//===-- llvm/Target/InstInfo.h - Target Instruction Information --*- C++ -*-==//
//
// This file describes the target machine instructions to the code generator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_INSTINFO_H
#define LLVM_TARGET_INSTINFO_H

#include "llvm/Target/Machine.h"

typedef int InstrSchedClass;


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


  // delete this later *******
  bool isPhi(MachineOpCode opCode) { return isDummyPhiInstr(opCode); }  
  


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

#endif
