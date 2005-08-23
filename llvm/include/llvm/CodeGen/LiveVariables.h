//===-- llvm/CodeGen/LiveVariables.h - Live Variable Analysis ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include <map>

namespace llvm {

class MRegisterInfo;

class LiveVariables : public MachineFunctionPass {
public:
  struct VarInfo {
    /// DefInst - The machine instruction that defines this register.
    MachineInstr      *DefInst;

    /// AliveBlocks - Set of blocks of which this value is alive completely
    /// through.  This is a bit set which uses the basic block number as an
    /// index.
    ///
    std::vector<bool> AliveBlocks;

    /// Kills - List of MachineInstruction's which are the last use of this
    /// virtual register (kill it) in their basic block.
    ///
    std::vector<MachineInstr*> Kills;

    VarInfo() : DefInst(0) {}

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
  };

private:
  /// VirtRegInfo - This list is a mapping from virtual register number to
  /// variable information.  FirstVirtualRegister is subtracted from the virtual
  /// register number before indexing into this list.
  ///
  std::vector<VarInfo> VirtRegInfo;

  /// RegistersKilled - This multimap keeps track of all of the registers that
  /// are dead immediately after an instruction reads its operands.  If an
  /// instruction does not have an entry in this map, it kills no registers.
  ///
  std::multimap<MachineInstr*, unsigned> RegistersKilled;

  /// RegistersDead - This multimap keeps track of all of the registers that are
  /// dead immediately after an instruction executes, which are not dead after
  /// the operands are evaluated.  In practice, this only contains registers
  /// which are defined by an instruction, but never used.
  ///
  std::multimap<MachineInstr*, unsigned> RegistersDead;

  /// AllocatablePhysicalRegisters - This vector keeps track of which registers
  /// are actually register allocatable by the target machine.  We can not track
  /// liveness for values that are not in this set.
  ///
  std::vector<bool> AllocatablePhysicalRegisters;

private:   // Intermediate data structures
  const MRegisterInfo *RegInfo;

  MachineInstr **PhysRegInfo;
  bool          *PhysRegUsed;

  void HandlePhysRegUse(unsigned Reg, MachineInstr *MI);
  void HandlePhysRegDef(unsigned Reg, MachineInstr *MI);

public:

  virtual bool runOnMachineFunction(MachineFunction &MF);

  /// killed_iterator - Iterate over registers killed by a machine instruction
  ///
  typedef std::multimap<MachineInstr*, unsigned>::iterator killed_iterator;

  /// killed_begin/end - Get access to the range of registers killed by a
  /// machine instruction.
  killed_iterator killed_begin(MachineInstr *MI) {
    return RegistersKilled.lower_bound(MI);
  }
  killed_iterator killed_end(MachineInstr *MI) {
    return RegistersKilled.upper_bound(MI);
  }
  std::pair<killed_iterator, killed_iterator>
  killed_range(MachineInstr *MI) {
    return RegistersKilled.equal_range(MI);
  }

  /// KillsRegister - Return true if the specified instruction kills the
  /// specified register.
  bool KillsRegister(MachineInstr *MI, unsigned Reg) const {
    typedef std::multimap<MachineInstr*, unsigned>::const_iterator cki;
    std::pair<cki, cki> KIP = RegistersKilled.equal_range(MI);
    for (; KIP.first != KIP.second; ++KIP.first)
      if (KIP.first->second == Reg)
        return true;
    return false;
  }

  killed_iterator dead_begin(MachineInstr *MI) {
    return RegistersDead.lower_bound(MI);
  }
  killed_iterator dead_end(MachineInstr *MI) {
    return RegistersDead.upper_bound(MI);
  }
  std::pair<killed_iterator, killed_iterator>
  dead_range(MachineInstr *MI) {
    return RegistersDead.equal_range(MI);
  }
  
  /// RegisterDefIsDead - Return true if the specified instruction defines the
  /// specified register, but that definition is dead.
  bool RegisterDefIsDead(MachineInstr *MI, unsigned Reg) const {
    typedef std::multimap<MachineInstr*, unsigned>::const_iterator cki;
    std::pair<cki, cki> KIP = RegistersDead.equal_range(MI);
    for (; KIP.first != KIP.second; ++KIP.first)
      if (KIP.first->second == Reg)
        return true;
    return false;
  }

  //===--------------------------------------------------------------------===//
  //  API to update live variable information

  /// instructionChanged - When the address of an instruction changes, this
  /// method should be called so that live variables can update its internal
  /// data structures.  This removes the records for OldMI, transfering them to
  /// the records for NewMI.
  void instructionChanged(MachineInstr *OldMI, MachineInstr *NewMI);

  /// addVirtualRegisterKilled - Add information about the fact that the
  /// specified register is killed after being used by the specified
  /// instruction.
  ///
  void addVirtualRegisterKilled(unsigned IncomingReg, MachineInstr *MI) {
    RegistersKilled.insert(std::make_pair(MI, IncomingReg));
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
    for (killed_iterator i = killed_begin(MI), e = killed_end(MI); i != e; ) {
      if (i->second == reg)
        RegistersKilled.erase(i++);
      else
        ++i;
    }
    return true;
  }

  /// removeVirtualRegistersKilled - Remove all of the specified killed
  /// registers from the live variable information.
  void removeVirtualRegistersKilled(killed_iterator B, killed_iterator E) {
    for (killed_iterator I = B; I != E; ++I) { // Remove VarInfo entries...
      bool removed = getVarInfo(I->second).removeKill(I->first);
      assert(removed && "kill not in register's VarInfo?");
    }
    RegistersKilled.erase(B, E);
  }

  /// addVirtualRegisterDead - Add information about the fact that the specified
  /// register is dead after being used by the specified instruction.
  ///
  void addVirtualRegisterDead(unsigned IncomingReg, MachineInstr *MI) {
    RegistersDead.insert(std::make_pair(MI, IncomingReg));
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

    for (killed_iterator i = killed_begin(MI), e = killed_end(MI); i != e; ) {
      if (i->second == reg)
        RegistersKilled.erase(i++);
      else
        ++i;
    }
    return true;
  }

  /// removeVirtualRegistersDead - Remove all of the specified dead
  /// registers from the live variable information.
  void removeVirtualRegistersDead(killed_iterator B, killed_iterator E) {
    for (killed_iterator I = B; I != E; ++I)  // Remove VarInfo entries...
      getVarInfo(I->second).removeKill(I->first);
    RegistersDead.erase(B, E);
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  virtual void releaseMemory() {
    VirtRegInfo.clear();
    RegistersKilled.clear();
    RegistersDead.clear();
  }

  /// getVarInfo - Return the VarInfo structure for the specified VIRTUAL
  /// register.
  VarInfo &getVarInfo(unsigned RegIdx);

  void MarkVirtRegAliveInBlock(VarInfo &VRInfo, MachineBasicBlock *BB);
  void HandleVirtRegUse(VarInfo &VRInfo, MachineBasicBlock *MBB,
                        MachineInstr *MI);
};

} // End llvm namespace

#endif
