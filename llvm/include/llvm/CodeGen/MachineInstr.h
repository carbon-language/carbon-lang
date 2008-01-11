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

namespace llvm {

class TargetInstrDesc;

template <typename T> struct ilist_traits;
template <typename T> struct ilist;

//===----------------------------------------------------------------------===//
/// MachineInstr - Representation of each machine instruction.
///
class MachineInstr {
  const TargetInstrDesc *TID;           // Instruction descriptor.
  unsigned short NumImplicitOps;        // Number of implicit operands (which
                                        // are determined at construction time).

  std::vector<MachineOperand> Operands; // the operands
  MachineInstr *Prev, *Next;            // Links for MBB's intrusive list.
  MachineBasicBlock *Parent;            // Pointer to the owning basic block.

  // OperandComplete - Return true if it's illegal to add a new operand
  bool OperandsComplete() const;

  MachineInstr(const MachineInstr&);
  void operator=(const MachineInstr&); // DO NOT IMPLEMENT

  // Intrusive list support
  friend struct ilist_traits<MachineInstr>;
  friend struct ilist_traits<MachineBasicBlock>;
  void setParent(MachineBasicBlock *P) { Parent = P; }
public:
  /// MachineInstr ctor - This constructor creates a dummy MachineInstr with
  /// TID NULL and no operands.
  MachineInstr();

  /// MachineInstr ctor - This constructor create a MachineInstr and add the
  /// implicit operands.  It reserves space for number of operands specified by
  /// TargetInstrDesc.
  explicit MachineInstr(const TargetInstrDesc &TID, bool NoImp = false);

  /// MachineInstr ctor - Work exactly the same as the ctor above, except that
  /// the MachineInstr is created and added to the end of the specified basic
  /// block.
  ///
  MachineInstr(MachineBasicBlock *MBB, const TargetInstrDesc &TID);

  ~MachineInstr();

  const MachineBasicBlock* getParent() const { return Parent; }
  MachineBasicBlock* getParent() { return Parent; }
  
  /// getDesc - Returns the target instruction descriptor of this
  /// MachineInstr.
  const TargetInstrDesc &getDesc() const { return *TID; }

  /// getOpcode - Returns the opcode of this MachineInstr.
  ///
  int getOpcode() const;

  /// Access to explicit operands of the instruction.
  ///
  unsigned getNumOperands() const { return Operands.size(); }

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
  
  /// isIdenticalTo - Return true if this instruction is identical to (same
  /// opcode and same operands as) the specified instruction.
  bool isIdenticalTo(const MachineInstr *Other) const {
    if (Other->getOpcode() != getOpcode() ||
        Other->getNumOperands() != getNumOperands())
      return false;
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
      if (!getOperand(i).isIdenticalTo(Other->getOperand(i)))
        return false;
    return true;
  }

  /// clone - Create a copy of 'this' instruction that is identical in
  /// all ways except the the instruction has no parent, prev, or next.
  MachineInstr* clone() const { return new MachineInstr(*this); }
  
  /// removeFromParent - This method unlinks 'this' from the containing basic
  /// block, and returns it, but does not delete it.
  MachineInstr *removeFromParent();
  
  /// eraseFromParent - This method unlinks 'this' from the containing basic
  /// block and deletes it.
  void eraseFromParent() {
    delete removeFromParent();
  }

  /// findRegisterUseOperandIdx() - Returns the operand index that is a use of
  /// the specific register or -1 if it is not found. It further tightening
  /// the search criteria to a use that kills the register if isKill is true.
  int findRegisterUseOperandIdx(unsigned Reg, bool isKill = false) const;
  
  /// findRegisterDefOperand() - Returns the MachineOperand that is a def of
  /// the specific register or NULL if it is not found.
  MachineOperand *findRegisterDefOperand(unsigned Reg);

  /// findFirstPredOperandIdx() - Find the index of the first operand in the
  /// operand list that is used to represent the predicate. It returns -1 if
  /// none is found.
  int findFirstPredOperandIdx() const;
  
  /// isRegReDefinedByTwoAddr - Returns true if the Reg re-definition is due
  /// to two addr elimination.
  bool isRegReDefinedByTwoAddr(unsigned Reg) const;

  /// copyKillDeadInfo - Copies kill / dead operand properties from MI.
  ///
  void copyKillDeadInfo(const MachineInstr *MI);

  /// copyPredicates - Copies predicate operand(s) from MI.
  void copyPredicates(const MachineInstr *MI);

  //
  // Debugging support
  //
  void print(std::ostream *OS, const TargetMachine *TM) const {
    if (OS) print(*OS, TM);
  }
  void print(std::ostream &OS, const TargetMachine *TM = 0) const;
  void print(std::ostream *OS) const { if (OS) print(*OS); }
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

  /// RemoveOperand - Erase an operand  from an instruction, leaving it with one
  /// fewer operand than it started with.
  ///
  void RemoveOperand(unsigned i);

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

//===----------------------------------------------------------------------===//
// Debugging Support

inline std::ostream& operator<<(std::ostream &OS, const MachineInstr &MI) {
  MI.print(OS);
  return OS;
}

} // End llvm namespace

#endif
