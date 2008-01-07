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

class TargetRegisterClass;
class LiveVariables;
class CalleeSavedInfo;
class SDNode;
class SelectionDAG;

template<class T> class SmallVectorImpl;

//===----------------------------------------------------------------------===//
// Machine Operand Flags and Description
//===----------------------------------------------------------------------===//
  
namespace TOI {
  // Operand constraints: only "tied_to" for now.
  enum OperandConstraint {
    TIED_TO = 0  // Must be allocated the same register as.
  };
  
  /// OperandFlags - These are flags set on operands, but should be considered
  /// private, all access should go through the TargetOperandInfo accessors.
  /// See the accessors for a description of what these are.
  enum OperandFlags {
    LookupPtrRegClass = 0,
    Predicate,
    OptionalDef
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
  
  /// isLookupPtrRegClass - Set if this operand is a pointer value and it
  /// requires a callback to look up its register class.
  bool isLookupPtrRegClass() const { return Flags&(1 <<TOI::LookupPtrRegClass);}
  
  /// isPredicate - Set if this is one of the operands that made up of
  /// the predicate operand that controls an isPredicable() instruction.
  bool isPredicate() const { return Flags & (1 << TOI::Predicate); }
  
  /// isOptionalDef - Set if this operand is a optional def.
  ///
  bool isOptionalDef() const { return Flags & (1 << TOI::OptionalDef); }
};

  
//===----------------------------------------------------------------------===//
// Machine Instruction Flags and Description
//===----------------------------------------------------------------------===//

/// TargetInstrDesc flags - These should be considered private to the
/// implementation of the TargetInstrDesc class.  Clients should use the
/// predicate methods on TargetInstrDesc, not use these directly.  These
/// all correspond to bitfields in the TargetInstrDesc::Flags field.
namespace TID {
  enum {
    Variadic = 0,
    HasOptionalDef,
    Return,
    Call,
    ImplicitDef,
    Barrier,
    Terminator,
    Branch,
    IndirectBranch,
    Predicable,
    NotDuplicable,
    DelaySlot,
    SimpleLoad,
    MayStore,
    NeverHasSideEffects,
    MayHaveSideEffects,
    Commutable,
    ConvertibleTo3Addr,
    UsesCustomDAGSchedInserter,
    Rematerializable
  };
}

/// TargetInstrDesc - Describe properties that are true of each
/// instruction in the target description file.  This captures information about
/// side effects, register use and many other things.  There is one instance of
/// this struct for each target instruction class, and the MachineInstr class
/// points to this struct directly to describe itself.
class TargetInstrDesc {
public:
  unsigned short  Opcode;        // The opcode number.
  unsigned short  NumOperands;   // Num of args (may be more if variable_ops)
  unsigned short  NumDefs;       // Num of args that are definitions.
  unsigned short  SchedClass;    // enum identifying instr sched class
  const char *    Name;          // Name of the instruction record in td file.
  unsigned        Flags;         // flags identifying machine instr class
  unsigned        TSFlags;       // Target Specific Flag values
  const unsigned *ImplicitUses;  // Registers implicitly read by this instr
  const unsigned *ImplicitDefs;  // Registers implicitly defined by this instr
  const TargetOperandInfo *OpInfo; // 'NumOperands' entries about operands.

  /// getOperandConstraint - Returns the value of the specific constraint if
  /// it is set. Returns -1 if it is not set.
  int getOperandConstraint(unsigned OpNum,
                           TOI::OperandConstraint Constraint) const {
    assert((OpNum < NumOperands || isVariadic()) &&
           "Invalid operand # of TargetInstrInfo");
    if (OpNum < NumOperands &&
        (OpInfo[OpNum].Constraints & (1 << Constraint))) {
      unsigned Pos = 16 + Constraint * 4;
      return (int)(OpInfo[OpNum].Constraints >> Pos) & 0xf;
    }
    return -1;
  }

  /// findTiedToSrcOperand - Returns the operand that is tied to the specified
  /// dest operand. Returns -1 if there isn't one.
  int findTiedToSrcOperand(unsigned OpNum) const;
  
  /// getOpcode - Return the opcode number for this descriptor.
  unsigned getOpcode() const {
    return Opcode;
  }
  
  /// getName - Return the name of the record in the .td file for this
  /// instruction, for example "ADD8ri".
  const char *getName() const {
    return Name;
  }
  
  /// getNumOperands - Return the number of declared MachineOperands for this
  /// MachineInstruction.  Note that variadic (isVariadic() returns true)
  /// instructions may have additional operands at the end of the list, and note
  /// that the machine instruction may include implicit register def/uses as
  /// well.
  unsigned getNumOperands() const {
    return NumOperands;
  }
  
  /// getNumDefs - Return the number of MachineOperands that are register
  /// definitions.  Register definitions always occur at the start of the 
  /// machine operand list.  This is the number of "outs" in the .td file.
  unsigned getNumDefs() const {
    return NumDefs;
  }
  
  /// isVariadic - Return true if this instruction can have a variable number of
  /// operands.  In this case, the variable operands will be after the normal
  /// operands but before the implicit definitions and uses (if any are
  /// present).
  bool isVariadic() const {
    return Flags & (1 << TID::Variadic);
  }
  
  /// hasOptionalDef - Set if this instruction has an optional definition, e.g.
  /// ARM instructions which can set condition code if 's' bit is set.
  bool hasOptionalDef() const {
    return Flags & (1 << TID::HasOptionalDef);
  }
  
  /// getImplicitUses - Return a list of machine operands that are potentially
  /// read by any instance of this machine instruction.  For example, on X86,
  /// the "adc" instruction adds two register operands and adds the carry bit in
  /// from the flags register.  In this case, the instruction is marked as
  /// implicitly reading the flags.  Likewise, the variable shift instruction on
  /// X86 is marked as implicitly reading the 'CL' register, which it always
  /// does.
  ///
  /// This method returns null if the instruction has no implicit uses.
  const unsigned *getImplicitUses() const {
    return ImplicitUses;
  }
  
  /// getImplicitDefs - Return a list of machine operands that are potentially
  /// written by any instance of this machine instruction.  For example, on X86,
  /// many instructions implicitly set the flags register.  In this case, they
  /// are marked as setting the FLAGS.  Likewise, many instructions always
  /// deposit their result in a physical register.  For example, the X86 divide
  /// instruction always deposits the quotient and remainder in the EAX/EDX
  /// registers.  For that instruction, this will return a list containing the
  /// EAX/EDX/EFLAGS registers.
  ///
  /// This method returns null if the instruction has no implicit uses.
  const unsigned *getImplicitDefs() const {
    return ImplicitDefs;
  }

  /// getSchedClass - Return the scheduling class for this instruction.  The
  /// scheduling class is an index into the InstrItineraryData table.  This
  /// returns zero if there is no known scheduling information for the
  /// instruction.
  ///
  unsigned getSchedClass() const {
    return SchedClass;
  }
  
  bool isReturn() const {
    return Flags & (1 << TID::Return);
  }
  
  bool isCall() const {
    return Flags & (1 << TID::Call);
  }
  
  /// isImplicitDef - Return true if this is an "IMPLICIT_DEF" instruction,
  /// which defines a register to an unspecified value.  These basically
  /// correspond to x = undef.
  bool isImplicitDef() const {
    return Flags & (1 << TID::ImplicitDef);
  }
  
  /// isBarrier - Returns true if the specified instruction stops control flow
  /// from executing the instruction immediately following it.  Examples include
  /// unconditional branches and return instructions.
  bool isBarrier() const {
    return Flags & (1 << TID::Barrier);
  }
  
  /// isTerminator - Returns true if this instruction part of the terminator for
  /// a basic block.  Typically this is things like return and branch
  /// instructions.
  ///
  /// Various passes use this to insert code into the bottom of a basic block,
  /// but before control flow occurs.
  bool isTerminator() const {
    return Flags & (1 << TID::Terminator);
  }
  
  /// isBranch - Returns true if this is a conditional, unconditional, or
  /// indirect branch.  Predicates below can be used to discriminate between
  /// these cases, and the TargetInstrInfo::AnalyzeBranch method can be used to
  /// get more information.
  bool isBranch() const {
    return Flags & (1 << TID::Branch);
  }

  /// isIndirectBranch - Return true if this is an indirect branch, such as a
  /// branch through a register.
  bool isIndirectBranch() const {
    return Flags & (1 << TID::IndirectBranch);
  }
  
  /// isConditionalBranch - Return true if this is a branch which may fall
  /// through to the next instruction or may transfer control flow to some other
  /// block.  The TargetInstrInfo::AnalyzeBranch method can be used to get more
  /// information about this branch.
  bool isConditionalBranch() const {
    return isBranch() & !isBarrier() & !isIndirectBranch();
  }
  
  /// isUnconditionalBranch - Return true if this is a branch which always
  /// transfers control flow to some other block.  The
  /// TargetInstrInfo::AnalyzeBranch method can be used to get more information
  /// about this branch.
  bool isUnconditionalBranch() const {
    return isBranch() & isBarrier() & !isIndirectBranch();
  }
  
  // isPredicable - Return true if this instruction has a predicate operand that
  // controls execution.  It may be set to 'always', or may be set to other
  /// values.   There are various methods in TargetInstrInfo that can be used to
  /// control and modify the predicate in this instruction.
  bool isPredicable() const {
    return Flags & (1 << TID::Predicable);
  }
  
  /// isNotDuplicable - Return true if this instruction cannot be safely
  /// duplicated.  For example, if the instruction has a unique labels attached
  /// to it, duplicating it would cause multiple definition errors.
  bool isNotDuplicable() const {
    return Flags & (1 << TID::NotDuplicable);
  }
  
  /// hasDelaySlot - Returns true if the specified instruction has a delay slot
  /// which must be filled by the code generator.
  bool hasDelaySlot() const {
    return Flags & (1 << TID::DelaySlot);
  }
  
  /// isSimpleLoad - Return true for instructions that are simple loads from
  /// memory.  This should only be set on instructions that load a value from
  /// memory and return it in their only virtual register definition.
  /// Instructions that return a value loaded from memory and then modified in
  /// some way should not return true for this.
  bool isSimpleLoad() const {
    return Flags & (1 << TID::SimpleLoad);
  }
  
  //===--------------------------------------------------------------------===//
  // Side Effect Analysis
  //===--------------------------------------------------------------------===//
  
  /// mayStore - Return true if this instruction could possibly modify memory.
  /// Instructions with this flag set are not necessarily simple store
  /// instructions, they may store a modified value based on their operands, or
  /// may not actually modify anything, for example.
  bool mayStore() const {
    return Flags & (1 << TID::MayStore);
  }
  
  // TODO: mayLoad.
  
  /// hasNoSideEffects - Return true if all instances of this instruction are
  /// guaranteed to have no side effects other than:
  ///   1. The register operands that are def/used by the MachineInstr.
  ///   2. Registers that are implicitly def/used by the MachineInstr.
  ///   3. Memory Accesses captured by mayLoad() or mayStore().
  ///
  /// Examples of other side effects would be calling a function, modifying
  /// 'invisible' machine state like a control register, etc.
  ///
  /// If some instances of this instruction are side-effect free but others are
  /// not, the hasConditionalSideEffects() property should return true, not this
  /// one.
  ///
  /// Note that you should not call this method directly, instead, call the
  /// TargetInstrInfo::hasUnmodelledSideEffects method, which handles analysis
  /// of the machine instruction.
  bool hasNoSideEffects() const {
    return Flags & (1 << TID::NeverHasSideEffects);
  }
  
  /// hasConditionalSideEffects - Return true if some instances of this
  /// instruction are guaranteed to have no side effects other than those listed
  /// for hasNoSideEffects().  To determine whether a specific machineinstr has
  /// side effects, the TargetInstrInfo::isReallySideEffectFree virtual method
  /// is invoked to decide.
  ///
  /// Note that you should not call this method directly, instead, call the
  /// TargetInstrInfo::hasUnmodelledSideEffects method, which handles analysis
  /// of the machine instruction.
  bool hasConditionalSideEffects() const {
    return Flags & (1 << TID::MayHaveSideEffects);
  }
  
  //===--------------------------------------------------------------------===//
  // Flags that indicate whether an instruction can be modified by a method.
  //===--------------------------------------------------------------------===//
  
  /// isCommutable - Return true if this may be a 2- or 3-address
  /// instruction (of the form "X = op Y, Z, ..."), which produces the same
  /// result if Y and Z are exchanged.  If this flag is set, then the 
  /// TargetInstrInfo::commuteInstruction method may be used to hack on the
  /// instruction.
  ///
  /// Note that this flag may be set on instructions that are only commutable
  /// sometimes.  In these cases, the call to commuteInstruction will fail.
  /// Also note that some instructions require non-trivial modification to
  /// commute them.
  bool isCommutable() const {
    return Flags & (1 << TID::Commutable);
  }
  
  /// isConvertibleTo3Addr - Return true if this is a 2-address instruction
  /// which can be changed into a 3-address instruction if needed.  Doing this
  /// transformation can be profitable in the register allocator, because it
  /// means that the instruction can use a 2-address form if possible, but
  /// degrade into a less efficient form if the source and dest register cannot
  /// be assigned to the same register.  For example, this allows the x86
  /// backend to turn a "shl reg, 3" instruction into an LEA instruction, which
  /// is the same speed as the shift but has bigger code size.
  ///
  /// If this returns true, then the target must implement the
  /// TargetInstrInfo::convertToThreeAddress method for this instruction, which
  /// is allowed to fail if the transformation isn't valid for this specific
  /// instruction (e.g. shl reg, 4 on x86).
  ///
  bool isConvertibleTo3Addr() const {
    return Flags & (1 << TID::ConvertibleTo3Addr);
  }
  
  /// usesCustomDAGSchedInsertionHook - Return true if this instruction requires
  /// custom insertion support when the DAG scheduler is inserting it into a
  /// machine basic block.  If this is true for the instruction, it basically
  /// means that it is a pseudo instruction used at SelectionDAG time that is 
  /// expanded out into magic code by the target when MachineInstrs are formed.
  ///
  /// If this is true, the TargetLoweringInfo::InsertAtEndOfBasicBlock method
  /// is used to insert this into the MachineBasicBlock.
  bool usesCustomDAGSchedInsertionHook() const {
    return Flags & (1 << TID::UsesCustomDAGSchedInserter);
  }
  
  /// isRematerializable - Returns true if this instruction is a candidate for
  /// remat.  This flag is deprecated, please don't use it anymore.  If this
  /// flag is set, the isReallyTriviallyReMaterializable() method is called to
  /// verify the instruction is really rematable.
  bool isRematerializable() const {
    return Flags & (1 << TID::Rematerializable);
  }
};


//---------------------------------------------------------------------------
///
/// TargetInstrInfo - Interface to description of machine instructions
///
class TargetInstrInfo {
  const TargetInstrDesc *Descriptors; // Raw array to allow static init'n
  unsigned NumOpcodes;                // Number of entries in the desc array

  TargetInstrInfo(const TargetInstrInfo &);  // DO NOT IMPLEMENT
  void operator=(const TargetInstrInfo &);   // DO NOT IMPLEMENT
public:
  TargetInstrInfo(const TargetInstrDesc *desc, unsigned NumOpcodes);
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
  const TargetInstrDesc &get(unsigned Opcode) const {
    assert(Opcode < NumOpcodes && "Invalid opcode!");
    return Descriptors[Opcode];
  }

  /// isTriviallyReMaterializable - Return true if the instruction is trivially
  /// rematerializable, meaning it has no side effects and requires no operands
  /// that aren't always available.
  bool isTriviallyReMaterializable(MachineInstr *MI) const {
    return MI->getDesc().isRematerializable() &&
           isReallyTriviallyReMaterializable(MI);
  }

  /// hasUnmodelledSideEffects - Returns true if the instruction has side
  /// effects that are not captured by any operands of the instruction or other
  /// flags.
  bool hasUnmodelledSideEffects(MachineInstr *MI) const {
    const TargetInstrDesc &TID = MI->getDesc();
    if (TID.hasNoSideEffects()) return false;
    if (!TID.hasConditionalSideEffects()) return true;
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
  TargetInstrInfoImpl(const TargetInstrDesc *desc, unsigned NumOpcodes)
  : TargetInstrInfo(desc, NumOpcodes) {}
public:
  virtual MachineInstr *commuteInstruction(MachineInstr *MI) const;
  virtual bool PredicateInstruction(MachineInstr *MI,
                              const std::vector<MachineOperand> &Pred) const;
  
};

} // End llvm namespace

#endif
