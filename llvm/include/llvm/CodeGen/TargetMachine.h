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

#include "llvm/Support/Unique.h"
#include "llvm/Support/DataTypes.h"
#include <string>

class Type;
class StructType;


//---------------------------------------------------------------------------
// Data types used to define information about a single machine instruction
//---------------------------------------------------------------------------

typedef int MachineOpCode;
typedef int OpCodeMask;


// struct MachineInstrInfo:
//	Predefined information about each machine instruction.
// 
struct MachineInstrInfo {
  string	opCodeString;	// Assembly language mnemonic for the opcode.
  unsigned int	numOperands;	// Number of arguments for the instruction.
  int		resultPos;	// Position of the result; -1 if no result
  unsigned int	maxImmedConst;	// Largest +ve constant in IMMMED field or 0.
  bool    immedIsSignExtended;	// Is the IMMED field sign-extended? If so,
				//   smallest -ve value is -(maxImmedConst+1).
  
  
  // Check if the specified constant fits in the immediate field
  // of this machine instruction
  // 
  bool	constantFitsInImmedField	(int64_t intValue) const;
  
  // Return the largest +ve constant that can be held in the IMMMED field
  // of this machine instruction.
  // isSignExtended is set to true if the value is sign-extended before use
  // (this is true for all immediate fields in SPARC instructions).
  // Return 0 if the instruction has no IMMED field.
  // 
  inline uint64_t	maxImmedConstant(bool& isSignExtended) const {
				isSignExtended = immedIsSignExtended;
				return maxImmedConst; }
};

// Global variable holding an array of the above structures.
// This needs to be defined separately for each target machine.
// 
extern const MachineInstrInfo* TargetMachineInstrInfo;


//---------------------------------------------------------------------------
// class TargetMachine
// 
// Purpose:
//   Machine description.
// 
//---------------------------------------------------------------------------

class TargetMachine: public Unique {
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
  
  // Description of machine instructions (array indexed by machine opcode)
  const MachineInstrInfo* machineInstrInfo;
  
  // Register information.  This needs to be reorganized into a single class.
  int		zeroRegNum;	// register that gives 0 if any (-1 if none)
  
public:
  /*ctor*/		TargetMachine		() {}
  /*dtor*/ virtual	~TargetMachine		() {}
  
  virtual unsigned int	findOptimalStorageSize	(const Type* ty) const;
  
  virtual unsigned int*	findOptimalMemberOffsets(const StructType* stype)const;
};

//**************************************************************************/

#endif
