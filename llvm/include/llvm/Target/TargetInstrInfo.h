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

const unsigned M_BRANCH_FLAG           = 1 << 0;
const unsigned M_CALL_FLAG             = 1 << 1;
const unsigned M_RET_FLAG              = 1 << 2;
const unsigned M_BARRIER_FLAG          = 1 << 3;
const unsigned M_DELAY_SLOT_FLAG       = 1 << 4;
const unsigned M_LOAD_FLAG             = 1 << 5;
const unsigned M_STORE_FLAG            = 1 << 6;

// M_2_ADDR_FLAG - 3-addr instructions which really work like 2-addr ones.
const unsigned M_2_ADDR_FLAG           = 1 << 7;

// M_CONVERTIBLE_TO_3_ADDR - This is a M_2_ADDR_FLAG instruction which can be
// changed into a 3-address instruction if the first two operands cannot be
// assigned to the same register.  The target must implement the
// TargetInstrInfo::convertToThreeAddress method for this instruction.
const unsigned M_CONVERTIBLE_TO_3_ADDR = 1 << 8;

// This M_COMMUTABLE - is a 2- or 3-address instruction (of the form X = op Y,
// Z), which produces the same result if Y and Z are exchanged.
const unsigned M_COMMUTABLE            = 1 << 9;

// M_TERMINATOR_FLAG - Is this instruction part of the terminator for a basic
// block?  Typically this is things like return and branch instructions.
// Various passes use this to insert code into the bottom of a basic block, but
// before control flow occurs.
const unsigned M_TERMINATOR_FLAG       = 1 << 10;

// M_USES_CUSTOM_DAG_SCHED_INSERTION - Set if this instruction requires custom
// insertion support when the DAG scheduler is inserting it into a machine basic
// block.
const unsigned M_USES_CUSTOM_DAG_SCHED_INSERTION = 1 << 11;

// M_VARIABLE_OPS - Set if this instruction can have a variable number of extra
// operands in addition to the minimum number operands specified.
const unsigned M_VARIABLE_OPS = 1 << 12;

// Machine operand flags
// M_LOOK_UP_PTR_REG_CLASS - Set if this operand is a pointer value and it
// requires a callback to look up its register class.
const unsigned M_LOOK_UP_PTR_REG_CLASS = 1 << 0;

/// TargetOperandInfo - This holds information about one operand of a machine
/// instruction, indicating the register class for register operands, etc.
///
class TargetOperandInfo {
public:
  /// RegClass - This specifies the register class enumeration of the operand 
  /// if the operand is a register.  If not, this contains 0.
  unsigned short RegClass;
  unsigned short Flags;
  /// Currently no other information.
};


class TargetInstrDescriptor {
public:
  const char *    Name;          // Assembly language mnemonic for the opcode.
  unsigned        numOperands;   // Num of args (may be more if variable_ops).
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

  // Invariant opcodes: All instruction sets have these as their low opcodes.
  enum { 
    PHI = 0,
    INLINEASM = 1
  };

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
  bool isCommutableInstr(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_COMMUTABLE;
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

  bool hasVariableOperands(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_VARIABLE_OPS;
  }

  /// Return true if the instruction is a register to register move
  /// and leave the source and dest operands in the passed parameters.
  virtual bool isMoveInstr(const MachineInstr& MI,
                           unsigned& sourceReg,
                           unsigned& destReg) const {
    return false;
  }
  
  /// isLoadFromStackSlot - If the specified machine instruction is a direct
  /// load from a stack slot, return the virtual or physical register number of
  /// the destination along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than loading from the stack slot.
  virtual unsigned isLoadFromStackSlot(MachineInstr *MI, int &FrameIndex) const{
    return 0;
  }
  
  /// isStoreToStackSlot - If the specified machine instruction is a direct
  /// store to a stack slot, return the virtual or physical register number of
  /// the source reg along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than storing to the stack slot.
  virtual unsigned isStoreToStackSlot(MachineInstr *MI, int &FrameIndex) const {
    return 0;
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
  
  /// insertNoop - Insert a noop into the instruction stream at the specified
  /// point.
  virtual void insertNoop(MachineBasicBlock &MBB, 
                          MachineBasicBlock::iterator MI) const {
    assert(0 && "Target didn't implement insertNoop!");
    abort();
  }

  /// getPointerRegClass - Returns a TargetRegisterClass used for pointer
  /// values.
  virtual const TargetRegisterClass *getPointerRegClass() const {
    assert(0 && "Target didn't implement getPointerRegClass!");
    abort();
  }
  
  /// hasDelaySlot - Returns true if the specified instruction has a delay slot
  /// which must be filled by the code generator.
  bool hasDelaySlot(unsigned Opcode) const {
    return get(Opcode).Flags & M_DELAY_SLOT_FLAG;
  }
};

} // End llvm namespace

#endif
