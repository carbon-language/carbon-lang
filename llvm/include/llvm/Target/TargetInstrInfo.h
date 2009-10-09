//===-- llvm/Target/TargetInstrInfo.h - Instruction Info --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the target machine instruction set to the code generator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETINSTRINFO_H
#define LLVM_TARGET_TARGETINSTRINFO_H

#include "llvm/Target/TargetInstrDesc.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class MCAsmInfo;
class TargetRegisterClass;
class TargetRegisterInfo;
class LiveVariables;
class CalleeSavedInfo;
class SDNode;
class SelectionDAG;

template<class T> class SmallVectorImpl;


//---------------------------------------------------------------------------
///
/// TargetInstrInfo - Interface to description of machine instruction set
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
    DBG_LABEL = 2,
    EH_LABEL = 3,
    GC_LABEL = 4,

    /// KILL - This instruction is a noop that is used only to adjust the liveness
    /// of registers. This can be useful when dealing with sub-registers.
    KILL = 5,

    /// EXTRACT_SUBREG - This instruction takes two operands: a register
    /// that has subregisters, and a subregister index. It returns the
    /// extracted subregister value. This is commonly used to implement
    /// truncation operations on target architectures which support it.
    EXTRACT_SUBREG = 6,

    /// INSERT_SUBREG - This instruction takes three operands: a register
    /// that has subregisters, a register providing an insert value, and a
    /// subregister index. It returns the value of the first register with
    /// the value of the second register inserted. The first register is
    /// often defined by an IMPLICIT_DEF, as is commonly used to implement
    /// anyext operations on target architectures which support it.
    INSERT_SUBREG = 7,

    /// IMPLICIT_DEF - This is the MachineInstr-level equivalent of undef.
    IMPLICIT_DEF = 8,

    /// SUBREG_TO_REG - This instruction is similar to INSERT_SUBREG except
    /// that the first operand is an immediate integer constant. This constant
    /// is often zero, as is commonly used to implement zext operations on
    /// target architectures which support it, such as with x86-64 (with
    /// zext from i32 to i64 via implicit zero-extension).
    SUBREG_TO_REG = 9,

    /// COPY_TO_REGCLASS - This instruction is a placeholder for a plain
    /// register-to-register copy into a specific register class. This is only
    /// used between instruction selection and MachineInstr creation, before
    /// virtual registers have been created for all the instructions, and it's
    /// only needed in cases where the register classes implied by the
    /// instructions are insufficient. The actual MachineInstrs to perform
    /// the copy are emitted with the TargetInstrInfo::copyRegToReg hook.
    COPY_TO_REGCLASS = 10
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
  bool isTriviallyReMaterializable(const MachineInstr *MI,
                                   AliasAnalysis *AA = 0) const {
    return MI->getOpcode() == IMPLICIT_DEF ||
           (MI->getDesc().isRematerializable() &&
            (isReallyTriviallyReMaterializable(MI) ||
             isReallyTriviallyReMaterializableGeneric(MI, AA)));
  }

protected:
  /// isReallyTriviallyReMaterializable - For instructions with opcodes for
  /// which the M_REMATERIALIZABLE flag is set, this hook lets the target
  /// specify whether the instruction is actually trivially rematerializable,
  /// taking into consideration its operands. This predicate must return false
  /// if the instruction has any side effects other than producing a value, or
  /// if it requres any address registers that are not always available.
  virtual bool isReallyTriviallyReMaterializable(const MachineInstr *MI) const {
    return false;
  }

private:
  /// isReallyTriviallyReMaterializableGeneric - For instructions with opcodes
  /// for which the M_REMATERIALIZABLE flag is set and the target hook
  /// isReallyTriviallyReMaterializable returns false, this function does
  /// target-independent tests to determine if the instruction is really
  /// trivially rematerializable.
  bool isReallyTriviallyReMaterializableGeneric(const MachineInstr *MI,
                                                AliasAnalysis *AA) const;

public:
  /// Return true if the instruction is a register to register move and return
  /// the source and dest operands and their sub-register indices by reference.
  virtual bool isMoveInstr(const MachineInstr& MI,
                           unsigned& SrcReg, unsigned& DstReg,
                           unsigned& SrcSubIdx, unsigned& DstSubIdx) const {
    return false;
  }
  
  /// isLoadFromStackSlot - If the specified machine instruction is a direct
  /// load from a stack slot, return the virtual or physical register number of
  /// the destination along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than loading from the stack slot.
  virtual unsigned isLoadFromStackSlot(const MachineInstr *MI,
                                       int &FrameIndex) const {
    return 0;
  }
  
  /// isStoreToStackSlot - If the specified machine instruction is a direct
  /// store to a stack slot, return the virtual or physical register number of
  /// the source reg along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than storing to the stack slot.
  virtual unsigned isStoreToStackSlot(const MachineInstr *MI,
                                      int &FrameIndex) const {
    return 0;
  }

  /// reMaterialize - Re-issue the specified 'original' instruction at the
  /// specific location targeting a new destination register.
  virtual void reMaterialize(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MI,
                             unsigned DestReg, unsigned SubIdx,
                             const MachineInstr *Orig) const = 0;

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
                   MachineBasicBlock::iterator &MBBI, LiveVariables *LV) const {
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
  /// If NewMI is true, then a new machine instruction must be created.
  ///
  virtual MachineInstr *commuteInstruction(MachineInstr *MI,
                                           bool NewMI = false) const = 0;

  /// findCommutedOpIndices - If specified MI is commutable, return the two
  /// operand indices that would swap value. Return true if the instruction
  /// is not in a form which this routine understands.
  virtual bool findCommutedOpIndices(MachineInstr *MI, unsigned &SrcOpIdx1,
                                     unsigned &SrcOpIdx2) const = 0;

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
  ///    a successor block, it sets TBB to be the branch destination block and
  ///    a list of operands that evaluate the condition. These
  ///    operands can be passed to other TargetInstrInfo methods to create new
  ///    branches.
  /// 4. If this block ends with a conditional branch followed by an
  ///    unconditional branch, it returns the 'true' destination in TBB, the
  ///    'false' destination in FBB, and a list of operands that evaluate the
  ///    condition.  These operands can be passed to other TargetInstrInfo
  ///    methods to create new branches.
  ///
  /// Note that RemoveBranch and InsertBranch must be implemented to support
  /// cases where this method returns success.
  ///
  /// If AllowModify is true, then this routine is allowed to modify the basic
  /// block (e.g. delete instructions after the unconditional branch).
  ///
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond,
                             bool AllowModify = false) const {
    return true;
  }

  /// RemoveBranch - Remove the branching code at the end of the specific MBB.
  /// This is only invoked in cases where AnalyzeBranch returns success. It
  /// returns the number of instructions that were removed.
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const {
    assert(0 && "Target didn't implement TargetInstrInfo::RemoveBranch!"); 
    return 0;
  }

  /// InsertBranch - Insert branch code into the end of the specified
  /// MachineBasicBlock.  The operands to this method are the same as those
  /// returned by AnalyzeBranch.  This is only invoked in cases where
  /// AnalyzeBranch returns success. It returns the number of instructions
  /// inserted.
  ///
  /// It is also invoked by tail merging to add unconditional branches in
  /// cases where AnalyzeBranch doesn't apply because there was no original
  /// branch to analyze.  At least this much must be implemented, else tail
  /// merging needs to be disabled.
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB,
                            const SmallVectorImpl<MachineOperand> &Cond) const {
    assert(0 && "Target didn't implement TargetInstrInfo::InsertBranch!"); 
    return 0;
  }
  
  /// copyRegToReg - Emit instructions to copy between a pair of registers. It
  /// returns false if the target does not how to copy between the specified
  /// registers.
  virtual bool copyRegToReg(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            unsigned DestReg, unsigned SrcReg,
                            const TargetRegisterClass *DestRC,
                            const TargetRegisterClass *SrcRC) const {
    assert(0 && "Target didn't implement TargetInstrInfo::copyRegToReg!");
    return false;
  }
  
  /// storeRegToStackSlot - Store the specified register of the given register
  /// class to the specified stack frame index. The store instruction is to be
  /// added to the given machine basic block before the specified machine
  /// instruction. If isKill is true, the register operand is the last use and
  /// must be marked kill.
  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass *RC) const {
    assert(0 && "Target didn't implement TargetInstrInfo::storeRegToStackSlot!");
  }

  /// loadRegFromStackSlot - Load the specified register of the given register
  /// class from the specified stack frame index. The load instruction is to be
  /// added to the given machine basic block before the specified machine
  /// instruction.
  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    unsigned DestReg, int FrameIndex,
                                    const TargetRegisterClass *RC) const {
    assert(0 && "Target didn't implement TargetInstrInfo::loadRegFromStackSlot!");
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
  MachineInstr* foldMemoryOperand(MachineFunction &MF,
                                  MachineInstr* MI,
                                  const SmallVectorImpl<unsigned> &Ops,
                                  int FrameIndex) const;

  /// foldMemoryOperand - Same as the previous version except it allows folding
  /// of any load and store from / to any address, not just from a specific
  /// stack slot.
  MachineInstr* foldMemoryOperand(MachineFunction &MF,
                                  MachineInstr* MI,
                                  const SmallVectorImpl<unsigned> &Ops,
                                  MachineInstr* LoadMI) const;

protected:
  /// foldMemoryOperandImpl - Target-dependent implementation for
  /// foldMemoryOperand. Target-independent code in foldMemoryOperand will
  /// take care of adding a MachineMemOperand to the newly created instruction.
  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                          MachineInstr* MI,
                                          const SmallVectorImpl<unsigned> &Ops,
                                          int FrameIndex) const {
    return 0;
  }

  /// foldMemoryOperandImpl - Target-dependent implementation for
  /// foldMemoryOperand. Target-independent code in foldMemoryOperand will
  /// take care of adding a MachineMemOperand to the newly created instruction.
  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                              MachineInstr* MI,
                                              const SmallVectorImpl<unsigned> &Ops,
                                              MachineInstr* LoadMI) const {
    return 0;
  }

public:
  /// canFoldMemoryOperand - Returns true for the specified load / store if
  /// folding is possible.
  virtual
  bool canFoldMemoryOperand(const MachineInstr *MI,
                            const SmallVectorImpl<unsigned> &Ops) const {
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
  virtual bool BlockHasNoFallThrough(const MachineBasicBlock &MBB) const {
    return false;
  }
  
  /// ReverseBranchCondition - Reverses the branch condition of the specified
  /// condition list, returning false on success and true if it cannot be
  /// reversed.
  virtual
  bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
    return true;
  }
  
  /// insertNoop - Insert a noop into the instruction stream at the specified
  /// point.
  virtual void insertNoop(MachineBasicBlock &MBB, 
                          MachineBasicBlock::iterator MI) const;
  
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
                        const SmallVectorImpl<MachineOperand> &Pred) const = 0;

  /// SubsumesPredicate - Returns true if the first specified predicate
  /// subsumes the second, e.g. GE subsumes GT.
  virtual
  bool SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                         const SmallVectorImpl<MachineOperand> &Pred2) const {
    return false;
  }

  /// DefinesPredicate - If the specified instruction defines any predicate
  /// or condition code register(s) used for predication, returns true as well
  /// as the definition predicate(s) by reference.
  virtual bool DefinesPredicate(MachineInstr *MI,
                                std::vector<MachineOperand> &Pred) const {
    return false;
  }

  /// isSafeToMoveRegClassDefs - Return true if it's safe to move a machine
  /// instruction that defines the specified register class.
  virtual bool isSafeToMoveRegClassDefs(const TargetRegisterClass *RC) const {
    return true;
  }

  /// isDeadInstruction - Return true if the instruction is considered dead.
  /// This allows some late codegen passes to delete them.
  virtual bool isDeadInstruction(const MachineInstr *MI) const = 0;

  /// GetInstSize - Returns the size of the specified Instruction.
  /// 
  virtual unsigned GetInstSizeInBytes(const MachineInstr *MI) const {
    assert(0 && "Target didn't implement TargetInstrInfo::GetInstSize!");
    return 0;
  }

  /// GetFunctionSizeInBytes - Returns the size of the specified
  /// MachineFunction.
  /// 
  virtual unsigned GetFunctionSizeInBytes(const MachineFunction &MF) const = 0;
  
  /// Measure the specified inline asm to determine an approximation of its
  /// length.
  virtual unsigned getInlineAsmLength(const char *Str,
                                      const MCAsmInfo &MAI) const;
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
  virtual MachineInstr *commuteInstruction(MachineInstr *MI,
                                           bool NewMI = false) const;
  virtual bool findCommutedOpIndices(MachineInstr *MI, unsigned &SrcOpIdx1,
                                     unsigned &SrcOpIdx2) const;
  virtual bool PredicateInstruction(MachineInstr *MI,
                            const SmallVectorImpl<MachineOperand> &Pred) const;
  virtual void reMaterialize(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MI,
                             unsigned DestReg, unsigned SubReg,
                             const MachineInstr *Orig) const;
  virtual bool isDeadInstruction(const MachineInstr *MI) const;

  virtual unsigned GetFunctionSizeInBytes(const MachineFunction &MF) const;
};

} // End llvm namespace

#endif
