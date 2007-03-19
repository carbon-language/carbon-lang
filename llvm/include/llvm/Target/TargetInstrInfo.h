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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/DataTypes.h"
#include <vector>
#include <cassert>

namespace llvm {

class MachineInstr;
class TargetMachine;
class MachineCodeForInstruction;
class TargetRegisterClass;
class LiveVariables;

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

// M_CONVERTIBLE_TO_3_ADDR - This is a 2-address instruction which can be
// changed into a 3-address instruction if the first two operands cannot be
// assigned to the same register.  The target must implement the
// TargetInstrInfo::convertToThreeAddress method for this instruction.
const unsigned M_CONVERTIBLE_TO_3_ADDR = 1 << 7;

// This M_COMMUTABLE - is a 2- or 3-address instruction (of the form X = op Y,
// Z), which produces the same result if Y and Z are exchanged.
const unsigned M_COMMUTABLE            = 1 << 8;

// M_TERMINATOR_FLAG - Is this instruction part of the terminator for a basic
// block?  Typically this is things like return and branch instructions.
// Various passes use this to insert code into the bottom of a basic block, but
// before control flow occurs.
const unsigned M_TERMINATOR_FLAG       = 1 << 9;

// M_USES_CUSTOM_DAG_SCHED_INSERTION - Set if this instruction requires custom
// insertion support when the DAG scheduler is inserting it into a machine basic
// block.
const unsigned M_USES_CUSTOM_DAG_SCHED_INSERTION = 1 << 10;

// M_VARIABLE_OPS - Set if this instruction can have a variable number of extra
// operands in addition to the minimum number operands specified.
const unsigned M_VARIABLE_OPS = 1 << 11;

// M_PREDICATED - Set if this instruction has a predicate that controls its
// execution.
const unsigned M_PREDICATED = 1 << 12;

// M_REMATERIALIZIBLE - Set if this instruction can be trivally re-materialized
// at any time, e.g. constant generation, load from constant pool.
const unsigned M_REMATERIALIZIBLE = 1 << 13;


// Machine operand flags
// M_LOOK_UP_PTR_REG_CLASS - Set if this operand is a pointer value and it
// requires a callback to look up its register class.
const unsigned M_LOOK_UP_PTR_REG_CLASS = 1 << 0;

/// M_PREDICATE_OPERAND - Set if this is the first operand of a predicate
/// operand that controls an M_PREDICATED instruction.
const unsigned M_PREDICATE_OPERAND = 1 << 1;

namespace TOI {
  // Operand constraints: only "tied_to" for now.
  enum OperandConstraint {
    TIED_TO = 0  // Must be allocated the same register as.
  };
}

/// TargetOperandInfo - This holds information about one operand of a machine
/// instruction, indicating the register class for register operands, etc.
///
class TargetOperandInfo {
public:
  /// RegClass - This specifies the register class enumeration of the operand 
  /// if the operand is a register.  If not, this contains 0.
  unsigned short RegClass;
  unsigned short Flags;
  /// Lower 16 bits are used to specify which constraints are set. The higher 16
  /// bits are used to specify the value of constraints (4 bits each).
  unsigned int Constraints;
  /// Currently no other information.
};


class TargetInstrDescriptor {
public:
  MachineOpCode   Opcode;        // The opcode.
  unsigned short  numOperands;   // Num of args (may be more if variable_ops).
  const char *    Name;          // Assembly language mnemonic for the opcode.
  InstrSchedClass schedClass;    // enum  identifying instr sched class
  unsigned        Flags;         // flags identifying machine instr class
  unsigned        TSFlags;       // Target Specific Flag values
  const unsigned *ImplicitUses;  // Registers implicitly read by this instr
  const unsigned *ImplicitDefs;  // Registers implicitly defined by this instr
  const TargetOperandInfo *OpInfo; // 'numOperands' entries about operands.

  /// getOperandConstraint - Returns the value of the specific constraint if
  /// it is set. Returns -1 if it is not set.
  int getOperandConstraint(unsigned OpNum,
                           TOI::OperandConstraint Constraint) const {
    assert((OpNum < numOperands || (Flags & M_VARIABLE_OPS)) &&
           "Invalid operand # of TargetInstrInfo");
    if (OpNum < numOperands &&
        (OpInfo[OpNum].Constraints & (1 << Constraint))) {
      unsigned Pos = 16 + Constraint * 4;
      return (int)(OpInfo[OpNum].Constraints >> Pos) & 0xf;
    }
    return -1;
  }

  /// findTiedToSrcOperand - Returns the operand that is tied to the specified
  /// dest operand. Returns -1 if there isn't one.
  int findTiedToSrcOperand(unsigned OpNum) const;
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
    INLINEASM = 1,
    LABEL = 2
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

  bool isPredicated(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_PREDICATED;
  }
  bool isReMaterializable(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_REMATERIALIZIBLE;
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
  
  /// hasDelaySlot - Returns true if the specified instruction has a delay slot
  /// which must be filled by the code generator.
  bool hasDelaySlot(unsigned Opcode) const {
    return get(Opcode).Flags & M_DELAY_SLOT_FLAG;
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

  /// getOperandConstraint - Returns the value of the specific constraint if
  /// it is set. Returns -1 if it is not set.
  int getOperandConstraint(MachineOpCode Opcode, unsigned OpNum,
                           TOI::OperandConstraint Constraint) const {
    return get(Opcode).getOperandConstraint(OpNum, Constraint);
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
  /// may be able to convert a two-address instruction into one or moretrue
  /// three-address instructions on demand.  This allows the X86 target (for
  /// example) to convert ADD and SHL instructions into LEA instructions if they
  /// would require register copies due to two-addressness.
  ///
  /// This method returns a null pointer if the transformation cannot be
  /// performed, otherwise it returns the last new instruction.
  ///
  virtual MachineInstr *
  convertToThreeAddress(MachineFunction::iterator &MFI,
                   MachineBasicBlock::iterator &MBBI, LiveVariables &LV) const {
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

  /// AnalyzeBranch - Analyze the branching code at the end of MBB, returning
  /// true if it cannot be understood (e.g. it's a switch dispatch or isn't
  /// implemented for a target).  Upon success, this returns false and returns
  /// with the following information in various cases:
  ///
  /// 1. If this block ends with no branches (it just falls through to its succ)
  ///    just return false, leaving TBB/FBB null.
  /// 2. If this block ends with only an unconditional branch, it sets TBB to be
  ///    the destination block.
  /// 3. If this block ends with an conditional branch, it returns the 'true'
  ///    destination in TBB, the 'false' destination in FBB, and a list of
  ///    operands that evaluate the condition.  These operands can be passed to
  ///    other TargetInstrInfo methods to create new branches.
  ///
  /// Note that RemoveBranch and InsertBranch must be implemented to support
  /// cases where this method returns success.
  ///
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             std::vector<MachineOperand> &Cond) const {
    return true;
  }
  
  /// RemoveBranch - Remove the branching code at the end of the specific MBB.
  /// this is only invoked in cases where AnalyzeBranch returns success.
  virtual void RemoveBranch(MachineBasicBlock &MBB) const {
    assert(0 && "Target didn't implement TargetInstrInfo::RemoveBranch!"); 
  }
  
  /// InsertBranch - Insert a branch into the end of the specified
  /// MachineBasicBlock.  This operands to this method are the same as those
  /// returned by AnalyzeBranch.  This is invoked in cases where AnalyzeBranch
  /// returns success and when an unconditional branch (TBB is non-null, FBB is
  /// null, Cond is empty) needs to be inserted.
  virtual void InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB,
                            const std::vector<MachineOperand> &Cond) const {
    assert(0 && "Target didn't implement TargetInstrInfo::InsertBranch!"); 
  }
  
  /// BlockHasNoFallThrough - Return true if the specified block does not
  /// fall-through into its successor block.  This is primarily used when a
  /// branch is unanalyzable.  It is useful for things like unconditional
  /// indirect branches (jump tables).
  virtual bool BlockHasNoFallThrough(MachineBasicBlock &MBB) const {
    return false;
  }
  
  /// ReverseBranchCondition - Reverses the branch condition of the specified
  /// condition list, returning false on success and true if it cannot be
  /// reversed.
  virtual bool ReverseBranchCondition(std::vector<MachineOperand> &Cond) const {
    return true;
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
    return 0; // Must return a value in order to compile with VS 2005
  }
};

} // End llvm namespace

#endif
