//===-- llvm/Target/InstrInfo.h - Target Instruction Information --*-C++-*-==//
//
// This file describes the target machine instructions to the code generator.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MACHINEINSTRINFO_H
#define LLVM_TARGET_MACHINEINSTRINFO_H

#include "Support/NonCopyable.h"
#include "Support/DataTypes.h"
#include <string>
#include <vector>

class MachineInstrDescriptor;
class TmpInstruction;
class MachineInstr;
class TargetMachine;
class Value;
class Instruction;
class Function;
class MachineCodeForInstruction;

//---------------------------------------------------------------------------
// Data types used to define information about a single machine instruction
//---------------------------------------------------------------------------

typedef int MachineOpCode;
typedef int OpCodeMask;
typedef int InstrSchedClass;

const MachineOpCode INVALID_MACHINE_OPCODE = -1;


// Global variable holding an array of descriptors for machine instructions.
// The actual object needs to be created separately for each target machine.
// This variable is initialized and reset by class MachineInstrInfo.
// 
// FIXME: This should be a property of the target so that more than one target
// at a time can be active...
//
extern const MachineInstrDescriptor *TargetInstrDescriptors;


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
const unsigned int      M_PSEUDO_FLAG           = 1 << 14;


struct MachineInstrDescriptor {
  std::string     opCodeString;  // Assembly language mnemonic for the opcode.
  int	          numOperands;   // Number of args; -1 if variable #args
  int	          resultPos;     // Position of the result; -1 if no result
  unsigned int	  maxImmedConst; // Largest +ve constant in IMMMED field or 0.
  bool	          immedIsSignExtended; // Is IMMED field sign-extended? If so,
				 //   smallest -ve value is -(maxImmedConst+1).
  unsigned int    numDelaySlots; // Number of delay slots after instruction
  unsigned int    latency;	 // Latency in machine cycles
  InstrSchedClass schedClass;	 // enum  identifying instr sched class
  unsigned int	  iclass;	 // flags identifying machine instr class
};


class MachineInstrInfo : public NonCopyableV {
public:
  const TargetMachine& target;

protected:
  const MachineInstrDescriptor* desc;	// raw array to allow static init'n
  unsigned int descSize;		// number of entries in the desc array
  unsigned int numRealOpCodes;		// number of non-dummy op codes
  
public:
  MachineInstrInfo(const TargetMachine& tgt,
                   const MachineInstrDescriptor *desc, unsigned descSize,
		   unsigned numRealOpCodes);
  virtual ~MachineInstrInfo();
  
  unsigned getNumRealOpCodes()  const { return numRealOpCodes; }
  unsigned getNumTotalOpCodes() const { return descSize; }
  
  const MachineInstrDescriptor& getDescriptor(MachineOpCode opCode) const {
    assert(opCode >= 0 && opCode < (int)descSize);
    return desc[opCode];
  }
  
  int getNumOperands(MachineOpCode opCode) const {
    return getDescriptor(opCode).numOperands;
  }
  
  int getResultPos(MachineOpCode opCode) const {
    return getDescriptor(opCode).resultPos;
  }
  
  unsigned getNumDelaySlots(MachineOpCode opCode) const {
    return getDescriptor(opCode).numDelaySlots;
  }
  
  InstrSchedClass getSchedClass(MachineOpCode opCode) const {
    return getDescriptor(opCode).schedClass;
  }
  
  //
  // Query instruction class flags according to the machine-independent
  // flags listed above.
  // 
  unsigned int getIClass(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass;
  }
  bool isNop(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_NOP_FLAG;
  }
  bool isBranch(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_BRANCH_FLAG;
  }
  bool isCall(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_CALL_FLAG;
  }
  bool isReturn(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_RET_FLAG;
  }
  bool isControlFlow(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_BRANCH_FLAG
        || getDescriptor(opCode).iclass & M_CALL_FLAG
        || getDescriptor(opCode).iclass & M_RET_FLAG;
  }
  bool isArith(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_RET_FLAG;
  }
  bool isCCInstr(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_CC_FLAG;
  }
  bool isLogical(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_LOGICAL_FLAG;
  }
  bool isIntInstr(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_INT_FLAG;
  }
  bool isFloatInstr(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_FLOAT_FLAG;
  }
  bool isConditional(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_CONDL_FLAG;
  }
  bool isLoad(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_LOAD_FLAG;
  }
  bool isPrefetch(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_PREFETCH_FLAG;
  }
  bool isLoadOrPrefetch(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_LOAD_FLAG
        || getDescriptor(opCode).iclass & M_PREFETCH_FLAG;
  }
  bool isStore(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_STORE_FLAG;
  }
  bool isMemoryAccess(MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_LOAD_FLAG
        || getDescriptor(opCode).iclass & M_PREFETCH_FLAG
        || getDescriptor(opCode).iclass & M_STORE_FLAG;
  }
  bool isDummyPhiInstr(const MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_DUMMY_PHI_FLAG;
  }
  bool isPseudoInstr(const MachineOpCode opCode) const {
    return getDescriptor(opCode).iclass & M_PSEUDO_FLAG;
  }
  
  // Check if an instruction can be issued before its operands are ready,
  // or if a subsequent instruction that uses its result can be issued
  // before the results are ready.
  // Default to true since most instructions on many architectures allow this.
  // 
  virtual bool hasOperandInterlock(MachineOpCode opCode) const {
    return true;
  }
  
  virtual bool hasResultInterlock(MachineOpCode opCode) const {
    return true;
  }
  
  // 
  // Latencies for individual instructions and instruction pairs
  // 
  virtual int minLatency(MachineOpCode opCode) const {
    return getDescriptor(opCode).latency;
  }
  
  virtual int maxLatency(MachineOpCode opCode) const {
    return getDescriptor(opCode).latency;
  }

  //
  // Which operand holds an immediate constant?  Returns -1 if none
  // 
  virtual int getImmedConstantPos(MachineOpCode opCode) const {
    return -1; // immediate position is machine specific, so say -1 == "none"
  }
  
  // Check if the specified constant fits in the immediate field
  // of this machine instruction
  // 
  virtual bool constantFitsInImmedField(MachineOpCode opCode,
					int64_t intValue) const;
  
  // Return the largest +ve constant that can be held in the IMMMED field
  // of this machine instruction.
  // isSignExtended is set to true if the value is sign-extended before use
  // (this is true for all immediate fields in SPARC instructions).
  // Return 0 if the instruction has no IMMED field.
  // 
  virtual uint64_t maxImmedConstant(MachineOpCode opCode,
				    bool &isSignExtended) const {
    isSignExtended = getDescriptor(opCode).immedIsSignExtended;
    return getDescriptor(opCode).maxImmedConst;
  }

  //-------------------------------------------------------------------------
  // Code generation support for creating individual machine instructions
  //-------------------------------------------------------------------------
  
  // Create an instruction sequence to put the constant `val' into
  // the virtual register `dest'.  `val' may be a Constant or a
  // GlobalValue, viz., the constant address of a global variable or function.
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Symbolic constants or constants that must be accessed from memory
  // are added to the constant pool via MachineCodeForMethod::get(F).
  // 
  virtual void  CreateCodeToLoadConst(const TargetMachine& target,
                                      Function* F,
                                      Value* val,
                                      Instruction* dest,
                                      std::vector<MachineInstr*>& mvec,
                                      MachineCodeForInstruction& mcfi) const=0;
  
  // Create an instruction sequence to copy an integer value `val'
  // to a floating point value `dest' by copying to memory and back.
  // val must be an integral type.  dest must be a Float or Double.
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void  CreateCodeToCopyIntToFloat(const TargetMachine& target,
                                       Function* F,
                                       Value* val,
                                       Instruction* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi)const=0;

  // Similarly, create an instruction sequence to copy an FP value
  // `val' to an integer value `dest' by copying to memory and back.
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void  CreateCodeToCopyFloatToInt(const TargetMachine& target,
                                       Function* F,
                                       Value* val,
                                       Instruction* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi)const=0;
  
  // Create instruction(s) to copy src to dest, for arbitrary types
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void CreateCopyInstructionsByType(const TargetMachine& target,
                                       Function* F,
                                       Value* src,
                                       Instruction* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi)const=0;

  // Create instruction sequence to produce a sign-extended register value
  // from an arbitrary sized value (sized in bits, not bytes).
  // The generated instructions are appended to `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void CreateSignExtensionInstructions(const TargetMachine& target,
                                       Function* F,
                                       Value* srcVal,
                                       unsigned int srcSizeInBits,
                                       Value* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const=0;

  // Create instruction sequence to produce a zero-extended register value
  // from an arbitrary sized value (sized in bits, not bytes).
  // The generated instructions are appended to `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void CreateZeroExtensionInstructions(const TargetMachine& target,
                                       Function* F,
                                       Value* srcVal,
                                       unsigned int srcSizeInBits,
                                       Value* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const=0;
};

#endif
