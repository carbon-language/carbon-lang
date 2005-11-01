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

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/DataTypes.h"
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
class TargetRegisterClass;

//---------------------------------------------------------------------------
// Data types used to define information about a single machine instruction
//---------------------------------------------------------------------------

typedef short MachineOpCode;
typedef unsigned InstrSchedClass;

//---------------------------------------------------------------------------
// struct TargetInstrDescriptor:
//  Predefined information about each machine instruction.
//  Designed to initialized statically.
//

const unsigned M_NOP_FLAG              = 1 << 0;
const unsigned M_BRANCH_FLAG           = 1 << 1;
const unsigned M_CALL_FLAG             = 1 << 2;
const unsigned M_RET_FLAG              = 1 << 3;
const unsigned M_BARRIER_FLAG          = 1 << 4;
const unsigned M_DELAY_SLOT_FLAG       = 1 << 5;
const unsigned M_CC_FLAG               = 1 << 6;
const unsigned M_LOAD_FLAG             = 1 << 7;
const unsigned M_STORE_FLAG            = 1 << 8;

// M_2_ADDR_FLAG - 3-addr instructions which really work like 2-addr ones.
const unsigned M_2_ADDR_FLAG           = 1 << 9;

// M_CONVERTIBLE_TO_3_ADDR - This is a M_2_ADDR_FLAG instruction which can be
// changed into a 3-address instruction if the first two operands cannot be
// assigned to the same register.  The target must implement the
// TargetInstrInfo::convertToThreeAddress method for this instruction.
const unsigned M_CONVERTIBLE_TO_3_ADDR = 1 << 10;

// This M_COMMUTABLE - is a 2- or 3-address instruction (of the form X = op Y,
// Z), which produces the same result if Y and Z are exchanged.
const unsigned M_COMMUTABLE            = 1 << 11;

// M_TERMINATOR_FLAG - Is this instruction part of the terminator for a basic
// block?  Typically this is things like return and branch instructions.
// Various passes use this to insert code into the bottom of a basic block, but
// before control flow occurs.
const unsigned M_TERMINATOR_FLAG       = 1 << 12;

// M_USES_CUSTOM_DAG_SCHED_INSERTION - Set if this instruction requires custom
// insertion support when the DAG scheduler is inserting it into a machine basic
// block.
const unsigned M_USES_CUSTOM_DAG_SCHED_INSERTION = 1 << 13;

/// TargetOperandInfo - This holds information about one operand of a machine
/// instruction, indicating the register class for register operands, etc.
///
class TargetOperandInfo {
public:
  /// RegClass - This specifies the register class of the operand if the
  /// operand is a register.  If not, this contains null.
  const TargetRegisterClass *RegClass;
  
  /// Currently no other information.
};


class TargetInstrDescriptor {
public:
  const char *    Name;          // Assembly language mnemonic for the opcode.
  int             numOperands;   // Number of args; -1 if variable #args
  int             resultPos;     // Position of the result; -1 if no result
  unsigned        maxImmedConst; // Largest +ve constant in IMMED field or 0.
  bool            immedIsSignExtended; // Is IMMED field sign-extended? If so,
                                 //   smallest -ve value is -(maxImmedConst+1).
  unsigned        numDelaySlots; // Number of delay slots after instruction
  unsigned        latency;       // Latency in machine cycles
  InstrSchedClass schedClass;    // enum  identifying instr sched class
  unsigned        Flags;         // flags identifying machine instr class
  unsigned        TSFlags;       // Target Specific Flag values
  const unsigned *ImplicitUses;  // Registers implicitly read by this instr
  const unsigned *ImplicitDefs;  // Registers implicitly defined by this instr
  const TargetOperandInfo *OpInfo; // 'numOperands' entries about operands.
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
  const TargetInstrDescriptor& get(MachineOpCode Opcode) const {
    assert((unsigned)Opcode < NumOpcodes);
    return desc[Opcode];
  }

  const char *getName(MachineOpCode Opcode) const {
    return get(Opcode).Name;
  }

  int getNumOperands(MachineOpCode Opcode) const {
    return get(Opcode).numOperands;
  }

  InstrSchedClass getSchedClass(MachineOpCode Opcode) const {
    return get(Opcode).schedClass;
  }

  const unsigned *getImplicitUses(MachineOpCode Opcode) const {
    return get(Opcode).ImplicitUses;
  }

  const unsigned *getImplicitDefs(MachineOpCode Opcode) const {
    return get(Opcode).ImplicitDefs;
  }


  //
  // Query instruction class flags according to the machine-independent
  // flags listed above.
  //
  bool isReturn(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_RET_FLAG;
  }

  bool isTwoAddrInstr(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_2_ADDR_FLAG;
  }
  bool isTerminatorInstr(unsigned Opcode) const {
    return get(Opcode).Flags & M_TERMINATOR_FLAG;
  }
  
  bool isBranch(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_BRANCH_FLAG;
  }
  
  /// isBarrier - Returns true if the specified instruction stops control flow
  /// from executing the instruction immediately following it.  Examples include
  /// unconditional branches and return instructions.
  bool isBarrier(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_BARRIER_FLAG;
  }
  
  bool isCall(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_CALL_FLAG;
  }
  bool isLoad(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_LOAD_FLAG;
  }
  bool isStore(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_STORE_FLAG;
  }
  
  /// usesCustomDAGSchedInsertionHook - Return true if this instruction requires
  /// custom insertion support when the DAG scheduler is inserting it into a
  /// machine basic block.
  bool usesCustomDAGSchedInsertionHook(unsigned Opcode) const {
    return get(Opcode).Flags & M_USES_CUSTOM_DAG_SCHED_INSERTION;
  }

  /// Return true if the instruction is a register to register move
  /// and leave the source and dest operands in the passed parameters.
  virtual bool isMoveInstr(const MachineInstr& MI,
                           unsigned& sourceReg,
                           unsigned& destReg) const {
    return false;
  }

  /// convertToThreeAddress - This method must be implemented by targets that
  /// set the M_CONVERTIBLE_TO_3_ADDR flag.  When this flag is set, the target
  /// may be able to convert a two-address instruction into a true
  /// three-address instruction on demand.  This allows the X86 target (for
  /// example) to convert ADD and SHL instructions into LEA instructions if they
  /// would require register copies due to two-addressness.
  ///
  /// This method returns a null pointer if the transformation cannot be
  /// performed, otherwise it returns the new instruction.
  ///
  virtual MachineInstr *convertToThreeAddress(MachineInstr *TA) const {
    return 0;
  }

  /// commuteInstruction - If a target has any instructions that are commutable,
  /// but require converting to a different instruction or making non-trivial
  /// changes to commute them, this method can overloaded to do this.  The
  /// default implementation of this method simply swaps the first two operands
  /// of MI and returns it.
  ///
  /// If a target wants to make more aggressive changes, they can construct and
  /// return a new machine instruction.  If an instruction cannot commute, it
  /// can also return null.
  ///
  virtual MachineInstr *commuteInstruction(MachineInstr *MI) const;

  /// Insert a goto (unconditional branch) sequence to TMBB, at the
  /// end of MBB
  virtual void insertGoto(MachineBasicBlock& MBB,
                          MachineBasicBlock& TMBB) const {
    assert(0 && "Target didn't implement insertGoto!");
  }

  /// Reverses the branch condition of the MachineInstr pointed by
  /// MI. The instruction is replaced and the new MI is returned.
  virtual MachineBasicBlock::iterator
  reverseBranchCondition(MachineBasicBlock::iterator MI) const {
    assert(0 && "Target didn't implement reverseBranchCondition!");
    abort();
    return MI;
  }
  

  //-------------------------------------------------------------------------
  // Code generation support for creating individual machine instructions
  //
  // WARNING: These methods are Sparc specific
  //
  // DO NOT USE ANY OF THESE METHODS THEY ARE DEPRECATED!
  //
  //-------------------------------------------------------------------------

  unsigned getNumDelaySlots(MachineOpCode Opcode) const {
    return get(Opcode).numDelaySlots;
  }
  bool isCCInstr(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_CC_FLAG;
  }
  bool isNop(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_NOP_FLAG;
  }
  
  /// hasDelaySlot - Returns true if the specified instruction has a delay slot
  /// which must be filled by the code generator.
  bool hasDelaySlot(unsigned Opcode) const {
    return get(Opcode).Flags & M_DELAY_SLOT_FLAG;
  }

  virtual bool hasResultInterlock(MachineOpCode Opcode) const {
    return true;
  }

  //
  // Latencies for individual instructions and instruction pairs
  //
  virtual int minLatency(MachineOpCode Opcode) const {
    return get(Opcode).latency;
  }

  virtual int maxLatency(MachineOpCode Opcode) const {
    return get(Opcode).latency;
  }

  //
  // Which operand holds an immediate constant?  Returns -1 if none
  //
  virtual int getImmedConstantPos(MachineOpCode Opcode) const {
    return -1; // immediate position is machine specific, so say -1 == "none"
  }

  // Check if the specified constant fits in the immediate field
  // of this machine instruction
  //
  virtual bool constantFitsInImmedField(MachineOpCode Opcode,
                                        int64_t intValue) const;

  // Return the largest positive constant that can be held in the IMMED field
  // of this machine instruction.
  // isSignExtended is set to true if the value is sign-extended before use
  // (this is true for all immediate fields in SPARC instructions).
  // Return 0 if the instruction has no IMMED field.
  //
  virtual uint64_t maxImmedConstant(MachineOpCode Opcode,
                                    bool &isSignExtended) const {
    isSignExtended = get(Opcode).immedIsSignExtended;
    return get(Opcode).maxImmedConst;
  }
};

} // End llvm namespace

#endif
