//===-- llvm/CodeGen/MachineInstr.h - MachineInstr class --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MachineInstr class, which is the
// basic representation for all target dependent machine instructions used by
// the back end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEINSTR_H
#define LLVM_CODEGEN_MACHINEINSTR_H

#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/Target/TargetInstrDesc.h"
#include "llvm/Target/TargetOpcodes.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/DebugLoc.h"
#include <vector>

namespace llvm {

class AliasAnalysis;
class TargetInstrDesc;
class TargetInstrInfo;
class TargetRegisterInfo;
class MachineFunction;
class MachineMemOperand;

//===----------------------------------------------------------------------===//
/// MachineInstr - Representation of each machine instruction.
///
class MachineInstr : public ilist_node<MachineInstr> {
public:
  typedef MachineMemOperand **mmo_iterator;

  /// Flags to specify different kinds of comments to output in
  /// assembly code.  These flags carry semantic information not
  /// otherwise easily derivable from the IR text.
  ///
  enum CommentFlag {
    ReloadReuse = 0x1
  };
  
private:
  const TargetInstrDesc *TID;           // Instruction descriptor.
  unsigned short NumImplicitOps;        // Number of implicit operands (which
                                        // are determined at construction time).

  unsigned short AsmPrinterFlags;       // Various bits of information used by
                                        // the AsmPrinter to emit helpful
                                        // comments.  This is *not* semantic
                                        // information.  Do not use this for
                                        // anything other than to convey comment
                                        // information to AsmPrinter.

  std::vector<MachineOperand> Operands; // the operands
  mmo_iterator MemRefs;                 // information on memory references
  mmo_iterator MemRefsEnd;
  MachineBasicBlock *Parent;            // Pointer to the owning basic block.
  DebugLoc debugLoc;                    // Source line information.

  // OperandComplete - Return true if it's illegal to add a new operand
  bool OperandsComplete() const;

  MachineInstr(const MachineInstr&);   // DO NOT IMPLEMENT
  void operator=(const MachineInstr&); // DO NOT IMPLEMENT

  // Intrusive list support
  friend struct ilist_traits<MachineInstr>;
  friend struct ilist_traits<MachineBasicBlock>;
  void setParent(MachineBasicBlock *P) { Parent = P; }

  /// MachineInstr ctor - This constructor creates a copy of the given
  /// MachineInstr in the given MachineFunction.
  MachineInstr(MachineFunction &, const MachineInstr &);

  /// MachineInstr ctor - This constructor creates a dummy MachineInstr with
  /// TID NULL and no operands.
  MachineInstr();

  // The next two constructors have DebugLoc and non-DebugLoc versions;
  // over time, the non-DebugLoc versions should be phased out and eventually
  // removed.

  /// MachineInstr ctor - This constructor create a MachineInstr and add the
  /// implicit operands.  It reserves space for number of operands specified by
  /// TargetInstrDesc.  The version with a DebugLoc should be preferred.
  explicit MachineInstr(const TargetInstrDesc &TID, bool NoImp = false);

  /// MachineInstr ctor - Work exactly the same as the ctor above, except that
  /// the MachineInstr is created and added to the end of the specified basic
  /// block.  The version with a DebugLoc should be preferred.
  ///
  MachineInstr(MachineBasicBlock *MBB, const TargetInstrDesc &TID);

  /// MachineInstr ctor - This constructor create a MachineInstr and add the
  /// implicit operands.  It reserves space for number of operands specified by
  /// TargetInstrDesc.  An explicit DebugLoc is supplied.
  explicit MachineInstr(const TargetInstrDesc &TID, const DebugLoc dl, 
                        bool NoImp = false);

  /// MachineInstr ctor - Work exactly the same as the ctor above, except that
  /// the MachineInstr is created and added to the end of the specified basic
  /// block.
  ///
  MachineInstr(MachineBasicBlock *MBB, const DebugLoc dl, 
               const TargetInstrDesc &TID);

  ~MachineInstr();

  // MachineInstrs are pool-allocated and owned by MachineFunction.
  friend class MachineFunction;

public:
  const MachineBasicBlock* getParent() const { return Parent; }
  MachineBasicBlock* getParent() { return Parent; }

  /// getAsmPrinterFlags - Return the asm printer flags bitvector.
  ///
  unsigned short getAsmPrinterFlags() const { return AsmPrinterFlags; }

  /// getAsmPrinterFlag - Return whether an AsmPrinter flag is set.
  ///
  bool getAsmPrinterFlag(CommentFlag Flag) const {
    return AsmPrinterFlags & Flag;
  }

  /// setAsmPrinterFlag - Set a flag for the AsmPrinter.
  ///
  void setAsmPrinterFlag(CommentFlag Flag) {
    AsmPrinterFlags |= (unsigned short)Flag;
  }

  /// getDebugLoc - Returns the debug location id of this MachineInstr.
  ///
  DebugLoc getDebugLoc() const { return debugLoc; }
  
  /// getDesc - Returns the target instruction descriptor of this
  /// MachineInstr.
  const TargetInstrDesc &getDesc() const { return *TID; }

  /// getOpcode - Returns the opcode of this MachineInstr.
  ///
  int getOpcode() const { return TID->Opcode; }

  /// Access to explicit operands of the instruction.
  ///
  unsigned getNumOperands() const { return (unsigned)Operands.size(); }

  const MachineOperand& getOperand(unsigned i) const {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return Operands[i];
  }
  MachineOperand& getOperand(unsigned i) {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return Operands[i];
  }

  /// getNumExplicitOperands - Returns the number of non-implicit operands.
  ///
  unsigned getNumExplicitOperands() const;
  
  /// Access to memory operands of the instruction
  mmo_iterator memoperands_begin() const { return MemRefs; }
  mmo_iterator memoperands_end() const { return MemRefsEnd; }
  bool memoperands_empty() const { return MemRefsEnd == MemRefs; }

  /// hasOneMemOperand - Return true if this instruction has exactly one
  /// MachineMemOperand.
  bool hasOneMemOperand() const {
    return MemRefsEnd - MemRefs == 1;
  }

  enum MICheckType {
    CheckDefs,      // Check all operands for equality
    IgnoreDefs,     // Ignore all definitions
    IgnoreVRegDefs  // Ignore virtual register definitions
  };

  /// isIdenticalTo - Return true if this instruction is identical to (same
  /// opcode and same operands as) the specified instruction.
  bool isIdenticalTo(const MachineInstr *Other,
                     MICheckType Check = CheckDefs) const;

  /// removeFromParent - This method unlinks 'this' from the containing basic
  /// block, and returns it, but does not delete it.
  MachineInstr *removeFromParent();
  
  /// eraseFromParent - This method unlinks 'this' from the containing basic
  /// block and deletes it.
  void eraseFromParent();

  /// isLabel - Returns true if the MachineInstr represents a label.
  ///
  bool isLabel() const {
    return getOpcode() == TargetOpcode::DBG_LABEL ||
           getOpcode() == TargetOpcode::EH_LABEL ||
           getOpcode() == TargetOpcode::GC_LABEL;
  }
  
  bool isDebugLabel() const { return getOpcode() == TargetOpcode::DBG_LABEL; }
  bool isEHLabel() const { return getOpcode() == TargetOpcode::EH_LABEL; }
  bool isGCLabel() const { return getOpcode() == TargetOpcode::GC_LABEL; }
  bool isDebugValue() const { return getOpcode() == TargetOpcode::DBG_VALUE; }
  
  bool isPHI() const { return getOpcode() == TargetOpcode::PHI; }
  bool isKill() const { return getOpcode() == TargetOpcode::KILL; }
  bool isImplicitDef() const { return getOpcode()==TargetOpcode::IMPLICIT_DEF; }
  bool isInlineAsm() const { return getOpcode() == TargetOpcode::INLINEASM; }
  bool isExtractSubreg() const {
    return getOpcode() == TargetOpcode::EXTRACT_SUBREG;
  }
  bool isInsertSubreg() const {
    return getOpcode() == TargetOpcode::INSERT_SUBREG;
  }
  bool isSubregToReg() const {
    return getOpcode() == TargetOpcode::SUBREG_TO_REG;
  }
  
  /// readsRegister - Return true if the MachineInstr reads the specified
  /// register. If TargetRegisterInfo is passed, then it also checks if there
  /// is a read of a super-register.
  bool readsRegister(unsigned Reg, const TargetRegisterInfo *TRI = NULL) const {
    return findRegisterUseOperandIdx(Reg, false, TRI) != -1;
  }

  /// killsRegister - Return true if the MachineInstr kills the specified
  /// register. If TargetRegisterInfo is passed, then it also checks if there is
  /// a kill of a super-register.
  bool killsRegister(unsigned Reg, const TargetRegisterInfo *TRI = NULL) const {
    return findRegisterUseOperandIdx(Reg, true, TRI) != -1;
  }

  /// modifiesRegister - Return true if the MachineInstr modifies the
  /// specified register. If TargetRegisterInfo is passed, then it also checks
  /// if there is a def of a super-register.
  bool modifiesRegister(unsigned Reg,
                        const TargetRegisterInfo *TRI = NULL) const {
    return findRegisterDefOperandIdx(Reg, false, TRI) != -1;
  }

  /// registerDefIsDead - Returns true if the register is dead in this machine
  /// instruction. If TargetRegisterInfo is passed, then it also checks
  /// if there is a dead def of a super-register.
  bool registerDefIsDead(unsigned Reg,
                         const TargetRegisterInfo *TRI = NULL) const {
    return findRegisterDefOperandIdx(Reg, true, TRI) != -1;
  }

  /// findRegisterUseOperandIdx() - Returns the operand index that is a use of
  /// the specific register or -1 if it is not found. It further tightens
  /// the search criteria to a use that kills the register if isKill is true.
  int findRegisterUseOperandIdx(unsigned Reg, bool isKill = false,
                                const TargetRegisterInfo *TRI = NULL) const;

  /// findRegisterUseOperand - Wrapper for findRegisterUseOperandIdx, it returns
  /// a pointer to the MachineOperand rather than an index.
  MachineOperand *findRegisterUseOperand(unsigned Reg, bool isKill = false,
                                         const TargetRegisterInfo *TRI = NULL) {
    int Idx = findRegisterUseOperandIdx(Reg, isKill, TRI);
    return (Idx == -1) ? NULL : &getOperand(Idx);
  }
  
  /// findRegisterDefOperandIdx() - Returns the operand index that is a def of
  /// the specified register or -1 if it is not found. If isDead is true, defs
  /// that are not dead are skipped. If TargetRegisterInfo is non-null, then it
  /// also checks if there is a def of a super-register.
  int findRegisterDefOperandIdx(unsigned Reg, bool isDead = false,
                                const TargetRegisterInfo *TRI = NULL) const;

  /// findRegisterDefOperand - Wrapper for findRegisterDefOperandIdx, it returns
  /// a pointer to the MachineOperand rather than an index.
  MachineOperand *findRegisterDefOperand(unsigned Reg, bool isDead = false,
                                         const TargetRegisterInfo *TRI = NULL) {
    int Idx = findRegisterDefOperandIdx(Reg, isDead, TRI);
    return (Idx == -1) ? NULL : &getOperand(Idx);
  }

  /// findFirstPredOperandIdx() - Find the index of the first operand in the
  /// operand list that is used to represent the predicate. It returns -1 if
  /// none is found.
  int findFirstPredOperandIdx() const;
  
  /// isRegTiedToUseOperand - Given the index of a register def operand,
  /// check if the register def is tied to a source operand, due to either
  /// two-address elimination or inline assembly constraints. Returns the
  /// first tied use operand index by reference is UseOpIdx is not null.
  bool isRegTiedToUseOperand(unsigned DefOpIdx, unsigned *UseOpIdx = 0) const;

  /// isRegTiedToDefOperand - Return true if the use operand of the specified
  /// index is tied to an def operand. It also returns the def operand index by
  /// reference if DefOpIdx is not null.
  bool isRegTiedToDefOperand(unsigned UseOpIdx, unsigned *DefOpIdx = 0) const;

  /// copyKillDeadInfo - Copies kill / dead operand properties from MI.
  ///
  void copyKillDeadInfo(const MachineInstr *MI);

  /// copyPredicates - Copies predicate operand(s) from MI.
  void copyPredicates(const MachineInstr *MI);

  /// addRegisterKilled - We have determined MI kills a register. Look for the
  /// operand that uses it and mark it as IsKill. If AddIfNotFound is true,
  /// add a implicit operand if it's not found. Returns true if the operand
  /// exists / is added.
  bool addRegisterKilled(unsigned IncomingReg,
                         const TargetRegisterInfo *RegInfo,
                         bool AddIfNotFound = false);

  /// addRegisterDead - We have determined MI defined a register without a use.
  /// Look for the operand that defines it and mark it as IsDead. If
  /// AddIfNotFound is true, add a implicit operand if it's not found. Returns
  /// true if the operand exists / is added.
  bool addRegisterDead(unsigned IncomingReg, const TargetRegisterInfo *RegInfo,
                       bool AddIfNotFound = false);

  /// addRegisterDefined - We have determined MI defines a register. Make sure
  /// there is an operand defining Reg.
  void addRegisterDefined(unsigned IncomingReg,
                          const TargetRegisterInfo *RegInfo);

  /// isSafeToMove - Return true if it is safe to move this instruction. If
  /// SawStore is set to true, it means that there is a store (or call) between
  /// the instruction's location and its intended destination.
  bool isSafeToMove(const TargetInstrInfo *TII, AliasAnalysis *AA,
                    bool &SawStore) const;

  /// isSafeToReMat - Return true if it's safe to rematerialize the specified
  /// instruction which defined the specified register instead of copying it.
  bool isSafeToReMat(const TargetInstrInfo *TII, AliasAnalysis *AA,
                     unsigned DstReg) const;

  /// hasVolatileMemoryRef - Return true if this instruction may have a
  /// volatile memory reference, or if the information describing the
  /// memory reference is not available. Return false if it is known to
  /// have no volatile memory references.
  bool hasVolatileMemoryRef() const;

  /// isInvariantLoad - Return true if this instruction is loading from a
  /// location whose value is invariant across the function.  For example,
  /// loading a value from the constant pool or from the argument area of
  /// a function if it does not change.  This should only return true of *all*
  /// loads the instruction does are invariant (if it does multiple loads).
  bool isInvariantLoad(AliasAnalysis *AA) const;

  /// isConstantValuePHI - If the specified instruction is a PHI that always
  /// merges together the same virtual register, return the register, otherwise
  /// return 0.
  unsigned isConstantValuePHI() const;

  //
  // Debugging support
  //
  void print(raw_ostream &OS, const TargetMachine *TM = 0) const;
  void dump() const;

  //===--------------------------------------------------------------------===//
  // Accessors used to build up machine instructions.

  /// addOperand - Add the specified operand to the instruction.  If it is an
  /// implicit operand, it is added to the end of the operand list.  If it is
  /// an explicit operand it is added at the end of the explicit operand list
  /// (before the first implicit operand). 
  void addOperand(const MachineOperand &Op);
  
  /// setDesc - Replace the instruction descriptor (thus opcode) of
  /// the current instruction with a new one.
  ///
  void setDesc(const TargetInstrDesc &tid) { TID = &tid; }

  /// setDebugLoc - Replace current source information with new such.
  /// Avoid using this, the constructor argument is preferable.
  ///
  void setDebugLoc(const DebugLoc dl) { debugLoc = dl; }

  /// RemoveOperand - Erase an operand  from an instruction, leaving it with one
  /// fewer operand than it started with.
  ///
  void RemoveOperand(unsigned i);

  /// addMemOperand - Add a MachineMemOperand to the machine instruction.
  /// This function should be used only occasionally. The setMemRefs function
  /// is the primary method for setting up a MachineInstr's MemRefs list.
  void addMemOperand(MachineFunction &MF, MachineMemOperand *MO);

  /// setMemRefs - Assign this MachineInstr's memory reference descriptor
  /// list. This does not transfer ownership.
  void setMemRefs(mmo_iterator NewMemRefs, mmo_iterator NewMemRefsEnd) {
    MemRefs = NewMemRefs;
    MemRefsEnd = NewMemRefsEnd;
  }

private:
  /// getRegInfo - If this instruction is embedded into a MachineFunction,
  /// return the MachineRegisterInfo object for the current function, otherwise
  /// return null.
  MachineRegisterInfo *getRegInfo();

  /// addImplicitDefUseOperands - Add all implicit def and use operands to
  /// this instruction.
  void addImplicitDefUseOperands();
  
  /// RemoveRegOperandsFromUseLists - Unlink all of the register operands in
  /// this instruction from their respective use lists.  This requires that the
  /// operands already be on their use lists.
  void RemoveRegOperandsFromUseLists();
  
  /// AddRegOperandsToUseLists - Add all of the register operands in
  /// this instruction from their respective use lists.  This requires that the
  /// operands not be on their use lists yet.
  void AddRegOperandsToUseLists(MachineRegisterInfo &RegInfo);
};

/// MachineInstrExpressionTrait - Special DenseMapInfo traits to compare
/// MachineInstr* by *value* of the instruction rather than by pointer value.
/// The hashing and equality testing functions ignore definitions so this is
/// useful for CSE, etc.
struct MachineInstrExpressionTrait : DenseMapInfo<MachineInstr*> {
  static inline MachineInstr *getEmptyKey() {
    return 0;
  }

  static inline MachineInstr *getTombstoneKey() {
    return reinterpret_cast<MachineInstr*>(-1);
  }

  static unsigned getHashValue(const MachineInstr* const &MI);

  static bool isEqual(const MachineInstr* const &LHS,
                      const MachineInstr* const &RHS) {
    if (RHS == getEmptyKey() || RHS == getTombstoneKey() ||
        LHS == getEmptyKey() || LHS == getTombstoneKey())
      return LHS == RHS;
    return LHS->isIdenticalTo(RHS, MachineInstr::IgnoreVRegDefs);
  }
};

//===----------------------------------------------------------------------===//
// Debugging Support

inline raw_ostream& operator<<(raw_ostream &OS, const MachineInstr &MI) {
  MI.print(OS);
  return OS;
}

} // End llvm namespace

#endif
