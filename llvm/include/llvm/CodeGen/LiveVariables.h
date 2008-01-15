//===-- llvm/CodeGen/LiveVariables.h - Live Variable Analysis ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveVariable analysis pass.  For each machine
// instruction in the function, this pass calculates the set of registers that
// are immediately dead after the instruction (i.e., the instruction calculates
// the value, but it is never used) and the set of registers that are used by
// the instruction, but are never used after the instruction (i.e., they are
// killed).
//
// This class computes live variables using are sparse implementation based on
// the machine code SSA form.  This class computes live variable information for
// each virtual and _register allocatable_ physical register in a function.  It
// uses the dominance properties of SSA form to efficiently compute live
// variables for virtual registers, and assumes that physical registers are only
// live within a single basic block (allowing it to do a single local analysis
// to resolve physical register lifetimes in each basic block).  If a physical
// register is not register allocatable, it is not tracked.  This is useful for
// things like the stack pointer and condition codes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEVARIABLES_H
#define LLVM_CODEGEN_LIVEVARIABLES_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include <map>

namespace llvm {

class MRegisterInfo;

class LiveVariables : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  LiveVariables() : MachineFunctionPass((intptr_t)&ID) {}

  /// VarInfo - This represents the regions where a virtual register is live in
  /// the program.  We represent this with three different pieces of
  /// information: the instruction that uniquely defines the value, the set of
  /// blocks the instruction is live into and live out of, and the set of 
  /// non-phi instructions that are the last users of the value.
  ///
  /// In the common case where a value is defined and killed in the same block,
  /// DefInst is the defining inst, there is one killing instruction, and 
  /// AliveBlocks is empty.
  ///
  /// Otherwise, the value is live out of the block.  If the value is live
  /// across any blocks, these blocks are listed in AliveBlocks.  Blocks where
  /// the liveness range ends are not included in AliveBlocks, instead being
  /// captured by the Kills set.  In these blocks, the value is live into the
  /// block (unless the value is defined and killed in the same block) and lives
  /// until the specified instruction.  Note that there cannot ever be a value
  /// whose Kills set contains two instructions from the same basic block.
  ///
  /// PHI nodes complicate things a bit.  If a PHI node is the last user of a
  /// value in one of its predecessor blocks, it is not listed in the kills set,
  /// but does include the predecessor block in the AliveBlocks set (unless that
  /// block also defines the value).  This leads to the (perfectly sensical)
  /// situation where a value is defined in a block, and the last use is a phi
  /// node in the successor.  In this case, DefInst will be the defining
  /// instruction, AliveBlocks is empty (the value is not live across any 
  /// blocks) and Kills is empty (phi nodes are not included).  This is sensical
  /// because the value must be live to the end of the block, but is not live in
  /// any successor blocks.
  struct VarInfo {
    /// DefInst - The machine instruction that defines this register.
    ///
    MachineInstr *DefInst;

    /// AliveBlocks - Set of blocks of which this value is alive completely
    /// through.  This is a bit set which uses the basic block number as an
    /// index.
    ///
    BitVector AliveBlocks;

    /// UsedBlocks - Set of blocks of which this value is actually used. This
    /// is a bit set which uses the basic block number as an index.
    BitVector UsedBlocks;

    /// NumUses - Number of uses of this register across the entire function.
    ///
    unsigned NumUses;

    /// Kills - List of MachineInstruction's which are the last use of this
    /// virtual register (kill it) in their basic block.
    ///
    std::vector<MachineInstr*> Kills;

    VarInfo() : DefInst(0), NumUses(0) {}

    /// removeKill - Delete a kill corresponding to the specified
    /// machine instruction. Returns true if there was a kill
    /// corresponding to this instruction, false otherwise.
    bool removeKill(MachineInstr *MI) {
      for (std::vector<MachineInstr*>::iterator i = Kills.begin(),
             e = Kills.end(); i != e; ++i)
        if (*i == MI) {
          Kills.erase(i);
          return true;
        }
      return false;
    }
    
    void dump() const;
  };

private:
  /// VirtRegInfo - This list is a mapping from virtual register number to
  /// variable information.  FirstVirtualRegister is subtracted from the virtual
  /// register number before indexing into this list.
  ///
  std::vector<VarInfo> VirtRegInfo;

  /// ReservedRegisters - This vector keeps track of which registers
  /// are reserved register which are not allocatable by the target machine.
  /// We can not track liveness for values that are in this set.
  ///
  BitVector ReservedRegisters;

private:   // Intermediate data structures
  MachineFunction *MF;

  const MRegisterInfo *RegInfo;

  // PhysRegInfo - Keep track of which instruction was the last def/use of a
  // physical register. This is a purely local property, because all physical
  // register references as presumed dead across basic blocks.
  MachineInstr **PhysRegInfo;

  // PhysRegUsed - Keep track whether the physical register has been used after
  // its last definition. This is local property.
  bool          *PhysRegUsed;

  // PhysRegPartUse - Keep track of which instruction was the last partial use
  // of a physical register (e.g. on X86 a def of EAX followed by a use of AX).
  // This is a purely local property.
  MachineInstr **PhysRegPartUse;

  // PhysRegPartDef - Keep track of a list of instructions which "partially"
  // defined the physical register (e.g. on X86 AX partially defines EAX).
  // These are turned into use/mod/write if there is a use of the register
  // later in the same block. This is local property.
  SmallVector<MachineInstr*, 4> *PhysRegPartDef;

  SmallVector<unsigned, 4> *PHIVarInfo;

  void addRegisterKills(unsigned Reg, MachineInstr *MI,
                        SmallSet<unsigned, 4> &SubKills);

  /// HandlePhysRegKill - Add kills of Reg and its sub-registers to the
  /// uses. Pay special attention to the sub-register uses which may come below
  /// the last use of the whole register.
  bool HandlePhysRegKill(unsigned Reg, MachineInstr *MI,
                         SmallSet<unsigned, 4> &SubKills);
  bool HandlePhysRegKill(unsigned Reg, MachineInstr *MI);
  void HandlePhysRegUse(unsigned Reg, MachineInstr *MI);
  void HandlePhysRegDef(unsigned Reg, MachineInstr *MI);

  /// analyzePHINodes - Gather information about the PHI nodes in here. In
  /// particular, we want to map the variable information of a virtual
  /// register which is used in a PHI node. We map that to the BB the vreg
  /// is coming from.
  void analyzePHINodes(const MachineFunction& Fn);
public:

  virtual bool runOnMachineFunction(MachineFunction &MF);

  /// KillsRegister - Return true if the specified instruction kills the
  /// specified register.
  bool KillsRegister(MachineInstr *MI, unsigned Reg) const;
  
  /// RegisterDefIsDead - Return true if the specified instruction defines the
  /// specified register, but that definition is dead.
  bool RegisterDefIsDead(MachineInstr *MI, unsigned Reg) const;

  /// ModifiesRegister - Return true if the specified instruction modifies the
  /// specified register.
  bool ModifiesRegister(MachineInstr *MI, unsigned Reg) const;
  
  //===--------------------------------------------------------------------===//
  //  API to update live variable information

  /// instructionChanged - When the address of an instruction changes, this
  /// method should be called so that live variables can update its internal
  /// data structures.  This removes the records for OldMI, transfering them to
  /// the records for NewMI.
  void instructionChanged(MachineInstr *OldMI, MachineInstr *NewMI);

  /// transferKillDeadInfo - Similar to instructionChanged except it does not
  /// update live variables internal data structures.
  static void transferKillDeadInfo(MachineInstr *OldMI, MachineInstr *NewMI,
                                   const MRegisterInfo *RegInfo);

  /// addRegisterKilled - We have determined MI kills a register. Look for the
  /// operand that uses it and mark it as IsKill. If AddIfNotFound is true,
  /// add a implicit operand if it's not found. Returns true if the operand
  /// exists / is added.
  static bool addRegisterKilled(unsigned IncomingReg, MachineInstr *MI,
                                const MRegisterInfo *RegInfo,
                                bool AddIfNotFound = false);

  /// addVirtualRegisterKilled - Add information about the fact that the
  /// specified register is killed after being used by the specified
  /// instruction. If AddIfNotFound is true, add a implicit operand if it's
  /// not found.
  void addVirtualRegisterKilled(unsigned IncomingReg, MachineInstr *MI,
                                bool AddIfNotFound = false) {
    if (addRegisterKilled(IncomingReg, MI, RegInfo, AddIfNotFound))
      getVarInfo(IncomingReg).Kills.push_back(MI); 
  }

  /// removeVirtualRegisterKilled - Remove the specified virtual
  /// register from the live variable information. Returns true if the
  /// variable was marked as killed by the specified instruction,
  /// false otherwise.
  bool removeVirtualRegisterKilled(unsigned reg,
                                   MachineBasicBlock *MBB,
                                   MachineInstr *MI) {
    if (!getVarInfo(reg).removeKill(MI))
      return false;

    bool Removed = false;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isRegister() && MO.isKill() && MO.getReg() == reg) {
        MO.setIsKill(false);
        Removed = true;
        break;
      }
    }

    assert(Removed && "Register is not used by this instruction!");
    return true;
  }

  /// removeVirtualRegistersKilled - Remove all killed info for the specified
  /// instruction.
  void removeVirtualRegistersKilled(MachineInstr *MI);
  
  /// addRegisterDead - We have determined MI defined a register without a use.
  /// Look for the operand that defines it and mark it as IsDead. If
  /// AddIfNotFound is true, add a implicit operand if it's not found. Returns
  /// true if the operand exists / is added.
  static bool addRegisterDead(unsigned IncomingReg, MachineInstr *MI,
                              const MRegisterInfo *RegInfo,
                              bool AddIfNotFound = false);

  /// addVirtualRegisterDead - Add information about the fact that the specified
  /// register is dead after being used by the specified instruction. If
  /// AddIfNotFound is true, add a implicit operand if it's not found.
  void addVirtualRegisterDead(unsigned IncomingReg, MachineInstr *MI,
                              bool AddIfNotFound = false) {
    if (addRegisterDead(IncomingReg, MI, RegInfo, AddIfNotFound))
        getVarInfo(IncomingReg).Kills.push_back(MI);
  }

  /// removeVirtualRegisterDead - Remove the specified virtual
  /// register from the live variable information. Returns true if the
  /// variable was marked dead at the specified instruction, false
  /// otherwise.
  bool removeVirtualRegisterDead(unsigned reg,
                                 MachineBasicBlock *MBB,
                                 MachineInstr *MI) {
    if (!getVarInfo(reg).removeKill(MI))
      return false;

    bool Removed = false;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isRegister() && MO.isDef() && MO.getReg() == reg) {
        MO.setIsDead(false);
        Removed = true;
        break;
      }
    }
    assert(Removed && "Register is not defined by this instruction!");
    return true;
  }

  /// removeVirtualRegistersDead - Remove all of the dead registers for the
  /// specified instruction from the live variable information.
  void removeVirtualRegistersDead(MachineInstr *MI);
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  virtual void releaseMemory() {
    VirtRegInfo.clear();
  }

  /// getVarInfo - Return the VarInfo structure for the specified VIRTUAL
  /// register.
  VarInfo &getVarInfo(unsigned RegIdx);

  void MarkVirtRegAliveInBlock(VarInfo& VRInfo, MachineBasicBlock* DefBlock,
                               MachineBasicBlock *BB);
  void MarkVirtRegAliveInBlock(VarInfo& VRInfo, MachineBasicBlock* DefBlock,
                               MachineBasicBlock *BB,
                               std::vector<MachineBasicBlock*> &WorkList);
  void HandleVirtRegUse(unsigned reg, MachineBasicBlock *MBB,
                        MachineInstr *MI);
};

} // End llvm namespace

#endif
