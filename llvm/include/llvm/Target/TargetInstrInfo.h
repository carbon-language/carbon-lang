//===-- llvm/Target/TargetInstrInfo.h - Instruction Info --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
class TargetRegisterClass;
class LiveVariables;
class CalleeSavedInfo;
class SDNode;
class SelectionDAG;

template<class T> class SmallVectorImpl;

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
  
/// M_SIMPLE_LOAD_FLAG - This flag is set for instructions that are simple loads
/// from memory.  This should only be set on instructions that load a value from
/// memory and return it in their only virtual register definition.
const unsigned M_SIMPLE_LOAD_FLAG      = 1 << 5;
  
/// M_MAY_STORE_FLAG - This flag is set to any instruction that could possibly
/// modify memory.  Instructions with this flag set are not necessarily simple
/// store instructions, they may store a modified value based on their operands,
/// or may not actually modify anything, for example.
const unsigned M_MAY_STORE_FLAG        = 1 << 6;
  
const unsigned M_INDIRECT_FLAG         = 1 << 7;
const unsigned M_IMPLICIT_DEF_FLAG     = 1 << 8;

// M_CONVERTIBLE_TO_3_ADDR - This is a 2-address instruction which can be
// changed into a 3-address instruction if the first two operands cannot be
// assigned to the same register.  The target must implement the
// TargetInstrInfo::convertToThreeAddress method for this instruction.
const unsigned M_CONVERTIBLE_TO_3_ADDR = 1 << 9;

// This M_COMMUTABLE - is a 2- or 3-address instruction (of the form X = op Y,
// Z), which produces the same result if Y and Z are exchanged.
const unsigned M_COMMUTABLE            = 1 << 10;

// M_TERMINATOR_FLAG - Is this instruction part of the terminator for a basic
// block?  Typically this is things like return and branch instructions.
// Various passes use this to insert code into the bottom of a basic block, but
// before control flow occurs.
const unsigned M_TERMINATOR_FLAG       = 1 << 11;

// M_USES_CUSTOM_DAG_SCHED_INSERTION - Set if this instruction requires custom
// insertion support when the DAG scheduler is inserting it into a machine basic
// block.
const unsigned M_USES_CUSTOM_DAG_SCHED_INSERTION = 1 << 12;

// M_VARIABLE_OPS - Set if this instruction can have a variable number of extra
// operands in addition to the minimum number operands specified.
const unsigned M_VARIABLE_OPS          = 1 << 13;

// M_PREDICABLE - Set if this instruction has a predicate operand that
// controls execution. It may be set to 'always'.
const unsigned M_PREDICABLE            = 1 << 14;

// M_REMATERIALIZIBLE - Set if this instruction can be trivally re-materialized
// at any time, e.g. constant generation, load from constant pool.
const unsigned M_REMATERIALIZIBLE      = 1 << 15;

// M_NOT_DUPLICABLE - Set if this instruction cannot be safely duplicated.
// (e.g. instructions with unique labels attached).
const unsigned M_NOT_DUPLICABLE        = 1 << 16;

// M_HAS_OPTIONAL_DEF - Set if this instruction has an optional definition, e.g.
// ARM instructions which can set condition code if 's' bit is set.
const unsigned M_HAS_OPTIONAL_DEF      = 1 << 17;

// M_NEVER_HAS_SIDE_EFFECTS - Set if this instruction has no side effects that
// are not captured by any operands of the instruction or other flags, and when
// *all* instances of the instruction of that opcode have no side effects.
//
// Note: This and M_MAY_HAVE_SIDE_EFFECTS are mutually exclusive. You can't set
// both! If neither flag is set, then the instruction *always* has side effects.
const unsigned M_NEVER_HAS_SIDE_EFFECTS = 1 << 18;

// M_MAY_HAVE_SIDE_EFFECTS - Set if some instances of this instruction can have
// side effects. The virtual method "isReallySideEffectFree" is called to
// determine this. Load instructions are an example of where this is useful. In
// general, loads always have side effects. However, loads from constant pools
// don't. We let the specific back end make this determination.
//
// Note: This and M_NEVER_HAS_SIDE_EFFECTS are mutually exclusive. You can't set
// both! If neither flag is set, then the instruction *always* has side effects.
const unsigned M_MAY_HAVE_SIDE_EFFECTS = 1 << 19;

// Machine operand flags
// M_LOOK_UP_PTR_REG_CLASS - Set if this operand is a pointer value and it
// requires a callback to look up its register class.
const unsigned M_LOOK_UP_PTR_REG_CLASS = 1 << 0;

/// M_PREDICATE_OPERAND - Set if this is one of the operands that made up of the
/// predicate operand that controls an M_PREDICATED instruction.
const unsigned M_PREDICATE_OPERAND = 1 << 1;

/// M_OPTIONAL_DEF_OPERAND - Set if this operand is a optional def.
///
const unsigned M_OPTIONAL_DEF_OPERAND = 1 << 2;

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
  unsigned short  numDefs;       // Num of args that are definitions.
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
  
  
  /// isSimpleLoad - Return true for instructions that are simple loads from
  /// memory.  This should only be set on instructions that load a value from
  /// memory and return it in their only virtual register definition.
  /// Instructions that return a value loaded from memory and then modified in
  /// some way should not return true for this.
  bool isSimpleLoad() const {
    return Flags & M_SIMPLE_LOAD_FLAG;
  }
  
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
    LABEL = 2,
    EXTRACT_SUBREG = 3,
    INSERT_SUBREG = 4
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

  int getNumDefs(MachineOpCode Opcode) const {
    return get(Opcode).numDefs;
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

  bool isCommutableInstr(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_COMMUTABLE;
  }
  bool isTerminatorInstr(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_TERMINATOR_FLAG;
  }
  
  bool isBranch(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_BRANCH_FLAG;
  }
  
  bool isIndirectBranch(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_INDIRECT_FLAG;
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

  /// mayStore - Return true if this instruction could possibly modify memory.
  /// Instructions with this flag set are not necessarily simple store
  /// instructions, they may store a modified value based on their operands, or
  /// may not actually modify anything, for example.
  bool mayStore(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_MAY_STORE_FLAG;
  }
  
  /// hasDelaySlot - Returns true if the specified instruction has a delay slot
  /// which must be filled by the code generator.
  bool hasDelaySlot(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_DELAY_SLOT_FLAG;
  }
  
  /// usesCustomDAGSchedInsertionHook - Return true if this instruction requires
  /// custom insertion support when the DAG scheduler is inserting it into a
  /// machine basic block.
  bool usesCustomDAGSchedInsertionHook(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_USES_CUSTOM_DAG_SCHED_INSERTION;
  }

  bool hasVariableOperands(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_VARIABLE_OPS;
  }

  bool isPredicable(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_PREDICABLE;
  }

  bool isNotDuplicable(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_NOT_DUPLICABLE;
  }

  bool hasOptionalDef(MachineOpCode Opcode) const {
    return get(Opcode).Flags & M_HAS_OPTIONAL_DEF;
  }

  /// isTriviallyReMaterializable - Return true if the instruction is trivially
  /// rematerializable, meaning it has no side effects and requires no operands
  /// that aren't always available.
  bool isTriviallyReMaterializable(MachineInstr *MI) const {
    return (MI->getInstrDescriptor()->Flags & M_REMATERIALIZIBLE) &&
           isReallyTriviallyReMaterializable(MI);
  }

  /// hasUnmodelledSideEffects - Returns true if the instruction has side
  /// effects that are not captured by any operands of the instruction or other
  /// flags.
  bool hasUnmodelledSideEffects(MachineInstr *MI) const {
    const TargetInstrDescriptor *TID = MI->getInstrDescriptor();
    if (TID->Flags & M_NEVER_HAS_SIDE_EFFECTS) return false;
    if (!(TID->Flags & M_MAY_HAVE_SIDE_EFFECTS)) return true;
    return !isReallySideEffectFree(MI); // May have side effects
  }
protected:
  /// isReallyTriviallyReMaterializable - For instructions with opcodes for
  /// which the M_REMATERIALIZABLE flag is set, this function tests whether the
  /// instruction itself is actually trivially rematerializable, considering
  /// its operands.  This is used for targets that have instructions that are
  /// only trivially rematerializable for specific uses.  This predicate must
  /// return false if the instruction has any side effects other than
  /// producing a value, or if it requres any address registers that are not
  /// always available.
  virtual bool isReallyTriviallyReMaterializable(MachineInstr *MI) const {
    return true;
  }

  /// isReallySideEffectFree - If the M_MAY_HAVE_SIDE_EFFECTS flag is set, this
  /// method is called to determine if the specific instance of this
  /// instruction has side effects. This is useful in cases of instructions,
  /// like loads, which generally always have side effects. A load from a
  /// constant pool doesn't have side effects, though. So we need to
  /// differentiate it from the general case.
  virtual bool isReallySideEffectFree(MachineInstr *MI) const {
    return false;
  }
public:
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
  /// may be able to convert a two-address instruction into one or more true
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
  virtual MachineInstr *commuteInstruction(MachineInstr *MI) const = 0;

  /// AnalyzeBranch - Analyze the branching code at the end of MBB, returning
  /// true if it cannot be understood (e.g. it's a switch dispatch or isn't
  /// implemented for a target).  Upon success, this returns false and returns
  /// with the following information in various cases:
  ///
  /// 1. If this block ends with no branches (it just falls through to its succ)
  ///    just return false, leaving TBB/FBB null.
  /// 2. If this block ends with only an unconditional branch, it sets TBB to be
  ///    the destination block.
  /// 3. If this block ends with an conditional branch and it falls through to
  ///    an successor block, it sets TBB to be the branch destination block and a
  ///    list of operands that evaluate the condition. These
  ///    operands can be passed to other TargetInstrInfo methods to create new
  ///    branches.
  /// 4. If this block ends with an conditional branch and an unconditional
  ///    block, it returns the 'true' destination in TBB, the 'false' destination
  ///    in FBB, and a list of operands that evaluate the condition. These
  ///    operands can be passed to other TargetInstrInfo methods to create new
  ///    branches.
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
  /// this is only invoked in cases where AnalyzeBranch returns success. It
  /// returns the number of instructions that were removed.
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const {
    assert(0 && "Target didn't implement TargetInstrInfo::RemoveBranch!"); 
    return 0;
  }
  
  /// InsertBranch - Insert a branch into the end of the specified
  /// MachineBasicBlock.  This operands to this method are the same as those
  /// returned by AnalyzeBranch.  This is invoked in cases where AnalyzeBranch
  /// returns success and when an unconditional branch (TBB is non-null, FBB is
  /// null, Cond is empty) needs to be inserted. It returns the number of
  /// instructions inserted.
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB,
                            const std::vector<MachineOperand> &Cond) const {
    assert(0 && "Target didn't implement TargetInstrInfo::InsertBranch!"); 
    return 0;
  }
  
  /// copyRegToReg - Add a copy between a pair of registers
  virtual void copyRegToReg(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            unsigned DestReg, unsigned SrcReg,
                            const TargetRegisterClass *DestRC,
                            const TargetRegisterClass *SrcRC) const {
    assert(0 && "Target didn't implement TargetInstrInfo::copyRegToReg!");
  }
  
  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass *RC) const {
    assert(0 && "Target didn't implement TargetInstrInfo::storeRegToStackSlot!");
  }

  virtual void storeRegToAddr(MachineFunction &MF, unsigned SrcReg, bool isKill,
                              SmallVectorImpl<MachineOperand> &Addr,
                              const TargetRegisterClass *RC,
                              SmallVectorImpl<MachineInstr*> &NewMIs) const {
    assert(0 && "Target didn't implement TargetInstrInfo::storeRegToAddr!");
  }

  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    unsigned DestReg, int FrameIndex,
                                    const TargetRegisterClass *RC) const {
    assert(0 && "Target didn't implement TargetInstrInfo::loadRegFromStackSlot!");
  }

  virtual void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                               SmallVectorImpl<MachineOperand> &Addr,
                               const TargetRegisterClass *RC,
                               SmallVectorImpl<MachineInstr*> &NewMIs) const {
    assert(0 && "Target didn't implement TargetInstrInfo::loadRegFromAddr!");
  }
  
  /// spillCalleeSavedRegisters - Issues instruction(s) to spill all callee
  /// saved registers and returns true if it isn't possible / profitable to do
  /// so by issuing a series of store instructions via
  /// storeRegToStackSlot(). Returns false otherwise.
  virtual bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
    return false;
  }

  /// restoreCalleeSavedRegisters - Issues instruction(s) to restore all callee
  /// saved registers and returns true if it isn't possible / profitable to do
  /// so by issuing a series of load instructions via loadRegToStackSlot().
  /// Returns false otherwise.
  virtual bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
    return false;
  }
  
  /// foldMemoryOperand - Attempt to fold a load or store of the specified stack
  /// slot into the specified machine instruction for the specified operand(s).
  /// If this is possible, a new instruction is returned with the specified
  /// operand folded, otherwise NULL is returned. The client is responsible for
  /// removing the old instruction and adding the new one in the instruction
  /// stream.
  virtual MachineInstr* foldMemoryOperand(MachineInstr* MI,
                                          SmallVectorImpl<unsigned> &Ops,
                                          int FrameIndex) const {
    return 0;
  }

  /// foldMemoryOperand - Same as the previous version except it allows folding
  /// of any load and store from / to any address, not just from a specific
  /// stack slot.
  virtual MachineInstr* foldMemoryOperand(MachineInstr* MI,
                                          SmallVectorImpl<unsigned> &Ops,
                                          MachineInstr* LoadMI) const {
    return 0;
  }

  /// canFoldMemoryOperand - Returns true if the specified load / store is
  /// folding is possible.
  virtual
  bool canFoldMemoryOperand(MachineInstr *MI,
                            SmallVectorImpl<unsigned> &Ops) const{
    return false;
  }

  /// unfoldMemoryOperand - Separate a single instruction which folded a load or
  /// a store or a load and a store into two or more instruction. If this is
  /// possible, returns true as well as the new instructions by reference.
  virtual bool unfoldMemoryOperand(MachineFunction &MF, MachineInstr *MI,
                                unsigned Reg, bool UnfoldLoad, bool UnfoldStore,
                                  SmallVectorImpl<MachineInstr*> &NewMIs) const{
    return false;
  }

  virtual bool unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                                   SmallVectorImpl<SDNode*> &NewNodes) const {
    return false;
  }

  /// getOpcodeAfterMemoryUnfold - Returns the opcode of the would be new
  /// instruction after load / store are unfolded from an instruction of the
  /// specified opcode. It returns zero if the specified unfolding is not
  /// possible.
  virtual unsigned getOpcodeAfterMemoryUnfold(unsigned Opc,
                                      bool UnfoldLoad, bool UnfoldStore) const {
    return 0;
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

  /// isPredicated - Returns true if the instruction is already predicated.
  ///
  virtual bool isPredicated(const MachineInstr *MI) const {
    return false;
  }

  /// isUnpredicatedTerminator - Returns true if the instruction is a
  /// terminator instruction that has not been predicated.
  virtual bool isUnpredicatedTerminator(const MachineInstr *MI) const;

  /// PredicateInstruction - Convert the instruction into a predicated
  /// instruction. It returns true if the operation was successful.
  virtual
  bool PredicateInstruction(MachineInstr *MI,
                            const std::vector<MachineOperand> &Pred) const = 0;

  /// SubsumesPredicate - Returns true if the first specified predicate
  /// subsumes the second, e.g. GE subsumes GT.
  virtual
  bool SubsumesPredicate(const std::vector<MachineOperand> &Pred1,
                         const std::vector<MachineOperand> &Pred2) const {
    return false;
  }

  /// DefinesPredicate - If the specified instruction defines any predicate
  /// or condition code register(s) used for predication, returns true as well
  /// as the definition predicate(s) by reference.
  virtual bool DefinesPredicate(MachineInstr *MI,
                                std::vector<MachineOperand> &Pred) const {
    return false;
  }

  /// getPointerRegClass - Returns a TargetRegisterClass used for pointer
  /// values.
  virtual const TargetRegisterClass *getPointerRegClass() const {
    assert(0 && "Target didn't implement getPointerRegClass!");
    abort();
    return 0; // Must return a value in order to compile with VS 2005
  }
};

/// TargetInstrInfoImpl - This is the default implementation of
/// TargetInstrInfo, which just provides a couple of default implementations
/// for various methods.  This separated out because it is implemented in
/// libcodegen, not in libtarget.
class TargetInstrInfoImpl : public TargetInstrInfo {
protected:
  TargetInstrInfoImpl(const TargetInstrDescriptor *desc, unsigned NumOpcodes)
  : TargetInstrInfo(desc, NumOpcodes) {}
public:
  virtual MachineInstr *commuteInstruction(MachineInstr *MI) const;
  virtual bool PredicateInstruction(MachineInstr *MI,
                              const std::vector<MachineOperand> &Pred) const;
  
};

} // End llvm namespace

#endif
