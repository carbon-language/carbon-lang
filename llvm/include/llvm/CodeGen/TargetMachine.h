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

#include "llvm/Support/NonCopyable.h"
#include "llvm/Support/DataTypes.h"
#include <string>

class Type;
class StructType;
struct MachineInstrDescriptor;


//---------------------------------------------------------------------------
// Data types used to define information about a single machine instruction
//---------------------------------------------------------------------------

typedef int MachineOpCode;
typedef int OpCodeMask;


// Global variable holding an array of descriptors for machine instructions.
// The actual object needs to be created separately for each target machine.
// This variable is initialized and reset by class MachineInstrInfo.
// 
extern const MachineInstrDescriptor* TargetInstrDescriptors;


//---------------------------------------------------------------------------
// struct MachineInstrInfo:
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


struct MachineInstrDescriptor {
  string	opCodeString;	// Assembly language mnemonic for the opcode.
  int		numOperands;	// Number of args; -1 if variable #args
  int		resultPos;	// Position of the result; -1 if no result
  unsigned int	maxImmedConst;	// Largest +ve constant in IMMMED field or 0.
  bool		immedIsSignExtended;	// Is IMMED field sign-extended? If so,
				//   smallest -ve value is -(maxImmedConst+1).
  unsigned int  numDelaySlots;	// Number of delay slots after instruction
  unsigned int  latency;	// Latency in machine cycles
  unsigned int	iclass;		// Flags identifying instruction class
};


class MachineInstrInfo : public NonCopyableV {
protected:
  const MachineInstrDescriptor* desc;	// raw array to allow static init'n
  unsigned int descSize;		// number of entries in the array
  
public:
  /*ctor*/		MachineInstrInfo(const MachineInstrDescriptor* _desc,
					 unsigned int _descSize);
  /*dtor*/ virtual	~MachineInstrInfo();
  
  const MachineInstrDescriptor& getDescriptor (MachineOpCode opCode) const {
				   assert(opCode < (int) descSize);
				   return desc[opCode]; }
  
  virtual bool		isBranch	(MachineOpCode opCode) const {
				   return desc[opCode].iclass & M_BRANCH_FLAG;}
  
  virtual bool		isLoad		(MachineOpCode opCode) const {
				   return desc[opCode].iclass & M_LOAD_FLAG
				     || desc[opCode].iclass & M_PREFETCH_FLAG;}
  
  virtual bool		isStore		(MachineOpCode opCode) const {
				   return desc[opCode].iclass & M_STORE_FLAG;}
  
  // Check if an instruction can be issued before its operands are ready,
  // or if a subsequent instruction that uses its result can be issued
  // before the results are ready.
  // Default to true since most instructions on many architectures allow this.
  // 
  virtual bool		hasOperandInterlock(MachineOpCode opCode) const {
						return true; }
  virtual bool		hasResultInterlock(MachineOpCode opCode) const {
						return true; }
  
  // 
  // Latencies for individual instructions and instruction pairs
  // 
  virtual int		minLatency	(MachineOpCode opCode) const {
				   return desc[opCode].latency; }
  
  virtual int		maxLatency	(MachineOpCode opCode) const {
				   return desc[opCode].latency; }
  
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
			    isSignExtended = desc[opCode].immedIsSignExtended;
			    return desc[opCode].maxImmedConst; }
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
  int		optSizeForSubWordData;
  int		intSize;
  int		longSize;
  int		floatSize;
  int		doubleSize;
  int		longDoubleSize;
  int		pointerSize;
  int		minMemOpWordSize;
  int		maxAtomicMemOpWordSize;
  
  // Register information.  This needs to be reorganized into a single class.
  int		zeroRegNum;	// register that gives 0 if any (-1 if none)
  
public:
  /*ctor*/		TargetMachine	(MachineInstrInfo* mii)
						: machineInstrInfo(mii) {}
  /*dtor*/ virtual	~TargetMachine	() {}
  
  const MachineInstrInfo& getInstrInfo	() const { return *machineInstrInfo; }
  
  // const MachineSchedInfo& getSchedInfo() const { return *machineSchedInfo; }
  
  virtual unsigned int	findOptimalStorageSize	(const Type* ty) const;
  
  virtual unsigned int*	findOptimalMemberOffsets(const StructType* stype)const;
  
  
protected:
  // Description of machine instructions
  // Protect so that subclass can control alloc/dealloc
  MachineInstrInfo* machineInstrInfo;
  // MachineSchedInfo* machineSchedInfo;
  
private:
  /*ctor*/		TargetMachine	();	// disable
};


//**************************************************************************/

#endif
