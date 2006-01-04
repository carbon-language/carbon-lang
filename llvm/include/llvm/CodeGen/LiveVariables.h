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
  /// VarInfo - This represents the regions where a virtual register is live in
  /// the program.  We represent this with three difference pieces of
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
    
    void dump() const;
  };

private:
  /// VirtRegInfo - This list is a mapping from virtual register number to
  /// variable information.  FirstVirtualRegister is subtracted from the virtual
  /// register number before indexing into this list.
  ///
  std::vector<VarInfo> VirtRegInfo;

  /// RegistersKilled - This map keeps track of all of the registers that
  /// are dead immediately after an instruction reads its operands.  If an
  /// instruction does not have an entry in this map, it kills no registers.
  ///
  std::map<MachineInstr*, std::vector<unsigned> > RegistersKilled;

  /// RegistersDead - This map keeps track of all of the registers that are
  /// dead immediately after an instruction executes, which are not dead after
  /// the operands are evaluated.  In practice, this only contains registers
  /// which are defined by an instruction, but never used.
  ///
  std::map<MachineInstr*, std::vector<unsigned> > RegistersDead;
  
  /// Dummy - An always empty vector used for instructions without dead or
  /// killed operands.
  std::vector<unsigned> Dummy;

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
  typedef std::vector<unsigned>::iterator killed_iterator;

  std::vector<unsigned> &getKillsVector(MachineInstr *MI) {
    std::map<MachineInstr*, std::vector<unsigned> >::iterator I = 
      RegistersKilled.find(MI);
    return I != RegistersKilled.end() ? I->second : Dummy;
  }
  std::vector<unsigned> &getDeadDefsVector(MachineInstr *MI) {
    std::map<MachineInstr*, std::vector<unsigned> >::iterator I = 
      RegistersDead.find(MI);
    return I != RegistersDead.end() ? I->second : Dummy;
  }
  
    
  /// killed_begin/end - Get access to the range of registers killed by a
  /// machine instruction.
  killed_iterator killed_begin(MachineInstr *MI) {
    return getKillsVector(MI).begin();
  }
  killed_iterator killed_end(MachineInstr *MI) {
    return getKillsVector(MI).end();
  }
  std::pair<killed_iterator, killed_iterator>
  killed_range(MachineInstr *MI) {
    std::vector<unsigned> &V = getKillsVector(MI);
    return std::make_pair(V.begin(), V.end());
  }

  /// KillsRegister - Return true if the specified instruction kills the
  /// specified register.
  bool KillsRegister(MachineInstr *MI, unsigned Reg) const;
  
  killed_iterator dead_begin(MachineInstr *MI) {
    return getDeadDefsVector(MI).begin();
  }
  killed_iterator dead_end(MachineInstr *MI) {
    return getDeadDefsVector(MI).end();
  }
  std::pair<killed_iterator, killed_iterator>
  dead_range(MachineInstr *MI) {
    std::vector<unsigned> &V = getDeadDefsVector(MI);
    return std::make_pair(V.begin(), V.end());
  }
  
  /// RegisterDefIsDead - Return true if the specified instruction defines the
  /// specified register, but that definition is dead.
  bool RegisterDefIsDead(MachineInstr *MI, unsigned Reg) const;
  
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
    std::vector<unsigned> &V = RegistersKilled[MI];
    // Insert in a sorted order.
    if (V.empty() || IncomingReg > V.back()) {
      V.push_back(IncomingReg);
    } else {
      std::vector<unsigned>::iterator I = V.begin();
      for (; *I < IncomingReg; ++I)
        /*empty*/;
      if (*I != IncomingReg)   // Don't insert duplicates.
        V.insert(I, IncomingReg);
    }
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

    std::vector<unsigned> &V = getKillsVector(MI);
    for (unsigned i = 0, e = V.size(); i != e; ++i)
      if (V[i] == reg) {
        V.erase(V.begin()+i);
        return true;
      }
    return true;
  }

  /// removeVirtualRegistersKilled - Remove all killed info for the specified
  /// instruction.
  void removeVirtualRegistersKilled(MachineInstr *MI) {
    std::map<MachineInstr*, std::vector<unsigned> >::iterator I = 
      RegistersKilled.find(MI);
    if (I != RegistersKilled.end()) {
      std::vector<unsigned> &Regs = I->second;
      for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
        bool removed = getVarInfo(Regs[i]).removeKill(MI);
        assert(removed && "kill not in register's VarInfo?");
      }
      RegistersKilled.erase(I);
    }
  }

  /// addVirtualRegisterDead - Add information about the fact that the specified
  /// register is dead after being used by the specified instruction.
  ///
  void addVirtualRegisterDead(unsigned IncomingReg, MachineInstr *MI) {
    std::vector<unsigned> &V = RegistersDead[MI];
    // Insert in a sorted order.
    if (V.empty() || IncomingReg > V.back()) {
      V.push_back(IncomingReg);
    } else {
      std::vector<unsigned>::iterator I = V.begin();
      for (; *I < IncomingReg; ++I)
        /*empty*/;
      if (*I != IncomingReg)   // Don't insert duplicates.
        V.insert(I, IncomingReg);
    }
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

    std::vector<unsigned> &V = getDeadDefsVector(MI);
    for (unsigned i = 0, e = V.size(); i != e; ++i)
      if (V[i] == reg) {
        V.erase(V.begin()+i);
        return true;
      }
    return true;
  }

  /// removeVirtualRegistersDead - Remove all of the specified dead
  /// registers from the live variable information.
  void removeVirtualRegistersDead(MachineInstr *MI) {
    std::map<MachineInstr*, std::vector<unsigned> >::iterator I = 
      RegistersDead.find(MI);
    if (I != RegistersDead.end()) {
      std::vector<unsigned> &Regs = I->second;
      for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
        bool removed = getVarInfo(Regs[i]).removeKill(MI);
        assert(removed && "kill not in register's VarInfo?");
      }
      RegistersDead.erase(I);
    }
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
