//===-- llvm/Target/TargetInstrInfo.h - Instruction Info --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the target machine instructions to the code generator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETINSTRINFO_H
#define LLVM_TARGET_TARGETINSTRINFO_H

#include "Support/DataTypes.h"
#include <vector>
#include <cassert>

namespace llvm {

class MachineInstr;
class TargetMachine;
class Value;
class Type;
class Instruction;
class Constant;
class Function;
class MachineCodeForInstruction;

//---------------------------------------------------------------------------
// Data types used to define information about a single machine instruction
//---------------------------------------------------------------------------

typedef short MachineOpCode;
typedef unsigned InstrSchedClass;

//---------------------------------------------------------------------------
// struct TargetInstrDescriptor:
//	Predefined information about each machine instruction.
//	Designed to initialized statically.
//

const unsigned M_NOP_FLAG		= 1 << 0;
const unsigned M_BRANCH_FLAG		= 1 << 1;
const unsigned M_CALL_FLAG		= 1 << 2;
const unsigned M_RET_FLAG		= 1 << 3;
const unsigned M_CC_FLAG		= 1 << 6;
const unsigned M_LOAD_FLAG		= 1 << 10;
const unsigned M_STORE_FLAG		= 1 << 12;
const unsigned M_DUMMY_PHI_FLAG	= 1 << 13;
const unsigned M_PSEUDO_FLAG           = 1 << 14;       // Pseudo instruction
// 3-addr instructions which really work like 2-addr ones, eg. X86 add/sub
const unsigned M_2_ADDR_FLAG           = 1 << 15;

// M_TERMINATOR_FLAG - Is this instruction part of the terminator for a basic
// block?  Typically this is things like return and branch instructions.
// Various passes use this to insert code into the bottom of a basic block, but
// before control flow occurs.
const unsigned M_TERMINATOR_FLAG       = 1 << 16;

struct TargetInstrDescriptor {
  const char *    Name;          // Assembly language mnemonic for the opcode.
  int             numOperands;   // Number of args; -1 if variable #args
  int             resultPos;     // Position of the result; -1 if no result
  unsigned        maxImmedConst; // Largest +ve constant in IMMED field or 0.
  bool	          immedIsSignExtended; // Is IMMED field sign-extended? If so,
                                 //   smallest -ve value is -(maxImmedConst+1).
  unsigned        numDelaySlots; // Number of delay slots after instruction
  unsigned        latency;       // Latency in machine cycles
  InstrSchedClass schedClass;    // enum  identifying instr sched class
  unsigned        Flags;         // flags identifying machine instr class
  unsigned        TSFlags;       // Target Specific Flag values
  const unsigned *ImplicitUses;  // Registers implicitly read by this instr
  const unsigned *ImplicitDefs;  // Registers implicitly defined by this instr
};


//---------------------------------------------------------------------------
/// 
/// TargetInstrInfo - Interface to description of machine instructions
/// 
class TargetInstrInfo {
  const TargetInstrDescriptor* desc;    // raw array to allow static init'n
  unsigned NumOpcodes;                  // number of entries in the desc array
  unsigned numRealOpCodes;              // number of non-dummy op codes
  
  TargetInstrInfo(const TargetInstrInfo &);  // DO NOT IMPLEMENT
  void operator=(const TargetInstrInfo &);   // DO NOT IMPLEMENT
public:
  TargetInstrInfo(const TargetInstrDescriptor *desc, unsigned NumOpcodes);
  virtual ~TargetInstrInfo();

  // Invariant: All instruction sets use opcode #0 as the PHI instruction
  enum { PHI = 0 };
  
  unsigned getNumOpcodes() const { return NumOpcodes; }
  
  /// get - Return the machine instruction descriptor that corresponds to the
  /// specified instruction opcode.
  ///
  const TargetInstrDescriptor& get(MachineOpCode opCode) const {
    assert((unsigned)opCode < NumOpcodes);
    return desc[opCode];
  }

  const char *getName(MachineOpCode opCode) const {
    return get(opCode).Name;
  }
  
  int getNumOperands(MachineOpCode opCode) const {
    return get(opCode).numOperands;
  }


  InstrSchedClass getSchedClass(MachineOpCode opCode) const {
    return get(opCode).schedClass;
  }

  const unsigned *getImplicitUses(MachineOpCode opCode) const {
    return get(opCode).ImplicitUses;
  }

  const unsigned *getImplicitDefs(MachineOpCode opCode) const {
    return get(opCode).ImplicitDefs;
  }


  //
  // Query instruction class flags according to the machine-independent
  // flags listed above.
  // 
  bool isReturn(MachineOpCode opCode) const {
    return get(opCode).Flags & M_RET_FLAG;
  }

  bool isPseudoInstr(MachineOpCode opCode) const {
    return get(opCode).Flags & M_PSEUDO_FLAG;
  }
  bool isTwoAddrInstr(MachineOpCode opCode) const {
    return get(opCode).Flags & M_2_ADDR_FLAG;
  }
  bool isTerminatorInstr(unsigned Opcode) const {
    return get(Opcode).Flags & M_TERMINATOR_FLAG;
  }

  //
  // Return true if the instruction is a register to register move and
  // leave the source and dest operands in the passed parameters.
  //
  virtual bool isMoveInstr(const MachineInstr& MI,
                           unsigned& sourceReg,
                           unsigned& destReg) const {
    return false;
  }




  //-------------------------------------------------------------------------
  // Code generation support for creating individual machine instructions
  //
  // WARNING: These methods are Sparc specific
  //
  // DO NOT USE ANY OF THESE METHODS THEY ARE DEPRECATED!
  //
  //-------------------------------------------------------------------------

  int getResultPos(MachineOpCode opCode) const {
    return get(opCode).resultPos;
  }
  unsigned getNumDelaySlots(MachineOpCode opCode) const {
    return get(opCode).numDelaySlots;
  }
  bool isCCInstr(MachineOpCode opCode) const {
    return get(opCode).Flags & M_CC_FLAG;
  }
  bool isNop(MachineOpCode opCode) const {
    return get(opCode).Flags & M_NOP_FLAG;
  }
  bool isBranch(MachineOpCode opCode) const {
    return get(opCode).Flags & M_BRANCH_FLAG;
  }
  bool isCall(MachineOpCode opCode) const {
    return get(opCode).Flags & M_CALL_FLAG;
  }
  bool isLoad(MachineOpCode opCode) const {
    return get(opCode).Flags & M_LOAD_FLAG;
  }
  bool isStore(MachineOpCode opCode) const {
    return get(opCode).Flags & M_STORE_FLAG;
  }
  bool isDummyPhiInstr(MachineOpCode opCode) const {
    return get(opCode).Flags & M_DUMMY_PHI_FLAG;
  }

  virtual bool hasResultInterlock(MachineOpCode opCode) const {
    return true;
  }

  // 
  // Latencies for individual instructions and instruction pairs
  // 
  virtual int minLatency(MachineOpCode opCode) const {
    return get(opCode).latency;
  }
  
  virtual int maxLatency(MachineOpCode opCode) const {
    return get(opCode).latency;
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
  
  // Return the largest positive constant that can be held in the IMMED field
  // of this machine instruction.
  // isSignExtended is set to true if the value is sign-extended before use
  // (this is true for all immediate fields in SPARC instructions).
  // Return 0 if the instruction has no IMMED field.
  // 
  virtual uint64_t maxImmedConstant(MachineOpCode opCode,
				    bool &isSignExtended) const {
    isSignExtended = get(opCode).immedIsSignExtended;
    return get(opCode).maxImmedConst;
  }
};

} // End llvm namespace

#endif
