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

class TargetInstrDescriptor;

template <typename T> struct ilist_traits;
template <typename T> struct ilist;

//===----------------------------------------------------------------------===//
/// MachineInstr - Representation of each machine instruction.
///
class MachineInstr {
  const TargetInstrDescriptor *TID;     // Instruction descriptor.
  unsigned short NumImplicitOps;        // Number of implicit operands (which
                                        // are determined at construction time).

  std::vector<MachineOperand> Operands; // the operands
  MachineInstr* prev, *next;            // links for our intrusive list
  MachineBasicBlock* parent;            // pointer to the owning basic block

  // OperandComplete - Return true if it's illegal to add a new operand
  bool OperandsComplete() const;

  MachineInstr(const MachineInstr&);
  void operator=(const MachineInstr&); // DO NOT IMPLEMENT

  // Intrusive list support
  friend struct ilist_traits<MachineInstr>;

public:
  /// MachineInstr ctor - This constructor creates a dummy MachineInstr with
  /// TID NULL and no operands.
  MachineInstr();

  /// MachineInstr ctor - This constructor create a MachineInstr and add the
  /// implicit operands.  It reserves space for number of operands specified by
  /// TargetInstrDescriptor.
  explicit MachineInstr(const TargetInstrDescriptor &TID, bool NoImp = false);

  /// MachineInstr ctor - Work exactly the same as the ctor above, except that
  /// the MachineInstr is created and added to the end of the specified basic
  /// block.
  ///
  MachineInstr(MachineBasicBlock *MBB, const TargetInstrDescriptor &TID);

  ~MachineInstr();

  const MachineBasicBlock* getParent() const { return parent; }
  MachineBasicBlock* getParent() { return parent; }
  
  /// getInstrDescriptor - Returns the target instruction descriptor of this
  /// MachineInstr.
  const TargetInstrDescriptor *getInstrDescriptor() const { return TID; }

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
  // Accessors to add operands when building up machine instructions.
  //
  void addOperand(const MachineOperand &Op) {
    bool isImpReg = Op.isRegister() && Op.isImplicit();
    assert((isImpReg || !OperandsComplete()) &&
           "Trying to add an operand to a machine instr that is already done!");
    if (isImpReg || NumImplicitOps == 0) {// This is true most of the time.
      Operands.push_back(Op);
      Operands.back().ParentMI = this;
    } else {
      // Insert a real operand before any implicit ones.
      unsigned OpNo = Operands.size()-NumImplicitOps;
      Operands.insert(Operands.begin()+OpNo, Op);
      Operands[OpNo].ParentMI = this;
    }
  }
  
  //===--------------------------------------------------------------------===//
  // Accessors used to modify instructions in place.
  //

  /// setInstrDescriptor - Replace the instruction descriptor (thus opcode) of
  /// the current instruction with a new one.
  ///
  void setInstrDescriptor(const TargetInstrDescriptor &tid) { TID = &tid; }

  /// RemoveOperand - Erase an operand  from an instruction, leaving it with one
  /// fewer operand than it started with.
  ///
  void RemoveOperand(unsigned i) {
    Operands.erase(Operands.begin()+i);
  }
private:

  /// addImplicitDefUseOperands - Add all implicit def and use operands to
  /// this instruction.
  void addImplicitDefUseOperands();
};

//===----------------------------------------------------------------------===//
// Debugging Support

inline std::ostream& operator<<(std::ostream &OS, const MachineInstr &MI) {
  MI.print(OS);
  return OS;
}

} // End llvm namespace

#endif
