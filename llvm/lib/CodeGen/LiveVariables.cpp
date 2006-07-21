//===-- LiveVariables.cpp - Live Variable Analysis for Machine Code -------===//
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

#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/alloca.h"
#include <algorithm>
#include <iostream>
using namespace llvm;

static RegisterAnalysis<LiveVariables> X("livevars", "Live Variable Analysis");

void LiveVariables::VarInfo::dump() const {
  std::cerr << "Register Defined by: ";
  if (DefInst) 
    std::cerr << *DefInst;
  else
    std::cerr << "<null>\n";
  std::cerr << "  Alive in blocks: ";
  for (unsigned i = 0, e = AliveBlocks.size(); i != e; ++i)
    if (AliveBlocks[i]) std::cerr << i << ", ";
  std::cerr << "\n  Killed by:";
  if (Kills.empty())
    std::cerr << " No instructions.\n";
  else {
    for (unsigned i = 0, e = Kills.size(); i != e; ++i)
      std::cerr << "\n    #" << i << ": " << *Kills[i];
    std::cerr << "\n";
  }
}

LiveVariables::VarInfo &LiveVariables::getVarInfo(unsigned RegIdx) {
  assert(MRegisterInfo::isVirtualRegister(RegIdx) &&
         "getVarInfo: not a virtual register!");
  RegIdx -= MRegisterInfo::FirstVirtualRegister;
  if (RegIdx >= VirtRegInfo.size()) {
    if (RegIdx >= 2*VirtRegInfo.size())
      VirtRegInfo.resize(RegIdx*2);
    else
      VirtRegInfo.resize(2*VirtRegInfo.size());
  }
  return VirtRegInfo[RegIdx];
}

bool LiveVariables::KillsRegister(MachineInstr *MI, unsigned Reg) const {
  std::map<MachineInstr*, std::vector<unsigned> >::const_iterator I = 
  RegistersKilled.find(MI);
  if (I == RegistersKilled.end()) return false;
  
  // Do a binary search, as these lists can grow pretty big, particularly for
  // call instructions on targets with lots of call-clobbered registers.
  return std::binary_search(I->second.begin(), I->second.end(), Reg);
}

bool LiveVariables::RegisterDefIsDead(MachineInstr *MI, unsigned Reg) const {
  std::map<MachineInstr*, std::vector<unsigned> >::const_iterator I = 
  RegistersDead.find(MI);
  if (I == RegistersDead.end()) return false;
  
  // Do a binary search, as these lists can grow pretty big, particularly for
  // call instructions on targets with lots of call-clobbered registers.
  return std::binary_search(I->second.begin(), I->second.end(), Reg);
}


void LiveVariables::MarkVirtRegAliveInBlock(VarInfo &VRInfo,
                                            MachineBasicBlock *MBB) {
  unsigned BBNum = MBB->getNumber();

  // Check to see if this basic block is one of the killing blocks.  If so,
  // remove it...
  for (unsigned i = 0, e = VRInfo.Kills.size(); i != e; ++i)
    if (VRInfo.Kills[i]->getParent() == MBB) {
      VRInfo.Kills.erase(VRInfo.Kills.begin()+i);  // Erase entry
      break;
    }

  if (MBB == VRInfo.DefInst->getParent()) return;  // Terminate recursion

  if (VRInfo.AliveBlocks.size() <= BBNum)
    VRInfo.AliveBlocks.resize(BBNum+1);  // Make space...

  if (VRInfo.AliveBlocks[BBNum])
    return;  // We already know the block is live

  // Mark the variable known alive in this bb
  VRInfo.AliveBlocks[BBNum] = true;

  for (MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(),
         E = MBB->pred_end(); PI != E; ++PI)
    MarkVirtRegAliveInBlock(VRInfo, *PI);
}

void LiveVariables::HandleVirtRegUse(VarInfo &VRInfo, MachineBasicBlock *MBB,
                                     MachineInstr *MI) {
  assert(VRInfo.DefInst && "Register use before def!");

  // Check to see if this basic block is already a kill block...
  if (!VRInfo.Kills.empty() && VRInfo.Kills.back()->getParent() == MBB) {
    // Yes, this register is killed in this basic block already.  Increase the
    // live range by updating the kill instruction.
    VRInfo.Kills.back() = MI;
    return;
  }

#ifndef NDEBUG
  for (unsigned i = 0, e = VRInfo.Kills.size(); i != e; ++i)
    assert(VRInfo.Kills[i]->getParent() != MBB && "entry should be at end!");
#endif

  assert(MBB != VRInfo.DefInst->getParent() &&
         "Should have kill for defblock!");

  // Add a new kill entry for this basic block.
  VRInfo.Kills.push_back(MI);

  // Update all dominating blocks to mark them known live.
  for (MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(),
         E = MBB->pred_end(); PI != E; ++PI)
    MarkVirtRegAliveInBlock(VRInfo, *PI);
}

void LiveVariables::HandlePhysRegUse(unsigned Reg, MachineInstr *MI) {
  PhysRegInfo[Reg] = MI;
  PhysRegUsed[Reg] = true;

  for (const unsigned *AliasSet = RegInfo->getAliasSet(Reg);
       unsigned Alias = *AliasSet; ++AliasSet) {
    PhysRegInfo[Alias] = MI;
    PhysRegUsed[Alias] = true;
  }
}

void LiveVariables::HandlePhysRegDef(unsigned Reg, MachineInstr *MI) {
  // Does this kill a previous version of this register?
  if (MachineInstr *LastUse = PhysRegInfo[Reg]) {
    if (PhysRegUsed[Reg])
      RegistersKilled[LastUse].push_back(Reg);
    else
      RegistersDead[LastUse].push_back(Reg);
  }
  PhysRegInfo[Reg] = MI;
  PhysRegUsed[Reg] = false;

  for (const unsigned *AliasSet = RegInfo->getAliasSet(Reg);
       unsigned Alias = *AliasSet; ++AliasSet) {
    if (MachineInstr *LastUse = PhysRegInfo[Alias]) {
      if (PhysRegUsed[Alias])
        RegistersKilled[LastUse].push_back(Alias);
      else
        RegistersDead[LastUse].push_back(Alias);
    }
    PhysRegInfo[Alias] = MI;
    PhysRegUsed[Alias] = false;
  }
}

bool LiveVariables::runOnMachineFunction(MachineFunction &MF) {
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();
  RegInfo = MF.getTarget().getRegisterInfo();
  assert(RegInfo && "Target doesn't have register information?");

  AllocatablePhysicalRegisters = RegInfo->getAllocatableSet(MF);

  // PhysRegInfo - Keep track of which instruction was the last use of a
  // physical register.  This is a purely local property, because all physical
  // register references as presumed dead across basic blocks.
  //
  PhysRegInfo = (MachineInstr**)alloca(sizeof(MachineInstr*) *
                                       RegInfo->getNumRegs());
  PhysRegUsed = (bool*)alloca(sizeof(bool)*RegInfo->getNumRegs());
  std::fill(PhysRegInfo, PhysRegInfo+RegInfo->getNumRegs(), (MachineInstr*)0);

  /// Get some space for a respectable number of registers...
  VirtRegInfo.resize(64);

  // Mark live-in registers as live-in.
  for (MachineFunction::livein_iterator I = MF.livein_begin(),
         E = MF.livein_end(); I != E; ++I) {
    assert(MRegisterInfo::isPhysicalRegister(I->first) &&
           "Cannot have a live-in virtual register!");
    HandlePhysRegDef(I->first, 0);
  }

  // Calculate live variable information in depth first order on the CFG of the
  // function.  This guarantees that we will see the definition of a virtual
  // register before its uses due to dominance properties of SSA (except for PHI
  // nodes, which are treated as a special case).
  //
  MachineBasicBlock *Entry = MF.begin();
  std::set<MachineBasicBlock*> Visited;
  for (df_ext_iterator<MachineBasicBlock*> DFI = df_ext_begin(Entry, Visited),
         E = df_ext_end(Entry, Visited); DFI != E; ++DFI) {
    MachineBasicBlock *MBB = *DFI;
    unsigned BBNum = MBB->getNumber();

    // Loop over all of the instructions, processing them.
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      MachineInstr *MI = I;
      const TargetInstrDescriptor &MID = TII.get(MI->getOpcode());

      // Process all of the operands of the instruction...
      unsigned NumOperandsToProcess = MI->getNumOperands();

      // Unless it is a PHI node.  In this case, ONLY process the DEF, not any
      // of the uses.  They will be handled in other basic blocks.
      if (MI->getOpcode() == TargetInstrInfo::PHI)
        NumOperandsToProcess = 1;

      // Loop over implicit uses, using them.
      if (MID.ImplicitUses) {
        for (const unsigned *ImplicitUses = MID.ImplicitUses;
             *ImplicitUses; ++ImplicitUses)
          HandlePhysRegUse(*ImplicitUses, MI);
      }

      // Process all explicit uses...
      for (unsigned i = 0; i != NumOperandsToProcess; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (MO.isUse() && MO.isRegister() && MO.getReg()) {
          if (MRegisterInfo::isVirtualRegister(MO.getReg())){
            HandleVirtRegUse(getVarInfo(MO.getReg()), MBB, MI);
          } else if (MRegisterInfo::isPhysicalRegister(MO.getReg()) &&
                     AllocatablePhysicalRegisters[MO.getReg()]) {
            HandlePhysRegUse(MO.getReg(), MI);
          }
        }
      }

      // Loop over implicit defs, defining them.
      if (MID.ImplicitDefs) {
        for (const unsigned *ImplicitDefs = MID.ImplicitDefs;
             *ImplicitDefs; ++ImplicitDefs)
          HandlePhysRegDef(*ImplicitDefs, MI);
      }

      // Process all explicit defs...
      for (unsigned i = 0; i != NumOperandsToProcess; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (MO.isDef() && MO.isRegister() && MO.getReg()) {
          if (MRegisterInfo::isVirtualRegister(MO.getReg())) {
            VarInfo &VRInfo = getVarInfo(MO.getReg());

            assert(VRInfo.DefInst == 0 && "Variable multiply defined!");
            VRInfo.DefInst = MI;
            // Defaults to dead
            VRInfo.Kills.push_back(MI);
          } else if (MRegisterInfo::isPhysicalRegister(MO.getReg()) &&
                     AllocatablePhysicalRegisters[MO.getReg()]) {
            HandlePhysRegDef(MO.getReg(), MI);
          }
        }
      }
    }

    // Handle any virtual assignments from PHI nodes which might be at the
    // bottom of this basic block.  We check all of our successor blocks to see
    // if they have PHI nodes, and if so, we simulate an assignment at the end
    // of the current block.
    for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
           E = MBB->succ_end(); SI != E; ++SI) {
      MachineBasicBlock *Succ = *SI;

      // PHI nodes are guaranteed to be at the top of the block...
      for (MachineBasicBlock::iterator MI = Succ->begin(), ME = Succ->end();
           MI != ME && MI->getOpcode() == TargetInstrInfo::PHI; ++MI) {
        for (unsigned i = 1; ; i += 2) {
          assert(MI->getNumOperands() > i+1 &&
                 "Didn't find an entry for our predecessor??");
          if (MI->getOperand(i+1).getMachineBasicBlock() == MBB) {
            MachineOperand &MO = MI->getOperand(i);
            VarInfo &VRInfo = getVarInfo(MO.getReg());
            assert(VRInfo.DefInst && "Register use before def (or no def)!");

            // Only mark it alive only in the block we are representing.
            MarkVirtRegAliveInBlock(VRInfo, MBB);
            break;   // Found the PHI entry for this block.
          }
        }
      }
    }

    // Finally, if the last block in the function is a return, make sure to mark
    // it as using all of the live-out values in the function.
    if (!MBB->empty() && TII.isReturn(MBB->back().getOpcode())) {
      MachineInstr *Ret = &MBB->back();
      for (MachineFunction::liveout_iterator I = MF.liveout_begin(),
             E = MF.liveout_end(); I != E; ++I) {
        assert(MRegisterInfo::isPhysicalRegister(*I) &&
               "Cannot have a live-in virtual register!");
        HandlePhysRegUse(*I, Ret);
      }
    }

    // Loop over PhysRegInfo, killing any registers that are available at the
    // end of the basic block.  This also resets the PhysRegInfo map.
    for (unsigned i = 0, e = RegInfo->getNumRegs(); i != e; ++i)
      if (PhysRegInfo[i])
        HandlePhysRegDef(i, 0);
  }

  // Convert the information we have gathered into VirtRegInfo and transform it
  // into a form usable by RegistersKilled.
  //
  for (unsigned i = 0, e = VirtRegInfo.size(); i != e; ++i)
    for (unsigned j = 0, e = VirtRegInfo[i].Kills.size(); j != e; ++j) {
      if (VirtRegInfo[i].Kills[j] == VirtRegInfo[i].DefInst)
        RegistersDead[VirtRegInfo[i].Kills[j]].push_back(
                                    i + MRegisterInfo::FirstVirtualRegister);

      else
        RegistersKilled[VirtRegInfo[i].Kills[j]].push_back(
                                    i + MRegisterInfo::FirstVirtualRegister);
    }

  // Walk through the RegistersKilled/Dead sets, and sort the registers killed
  // or dead.  This allows us to use efficient binary search for membership
  // testing.
  for (std::map<MachineInstr*, std::vector<unsigned> >::iterator
       I = RegistersKilled.begin(), E = RegistersKilled.end(); I != E; ++I)
    std::sort(I->second.begin(), I->second.end());
  for (std::map<MachineInstr*, std::vector<unsigned> >::iterator
       I = RegistersDead.begin(), E = RegistersDead.end(); I != E; ++I)
    std::sort(I->second.begin(), I->second.end());
  
  // Check to make sure there are no unreachable blocks in the MC CFG for the
  // function.  If so, it is due to a bug in the instruction selector or some
  // other part of the code generator if this happens.
#ifndef NDEBUG
  for(MachineFunction::iterator i = MF.begin(), e = MF.end(); i != e; ++i)
    assert(Visited.count(&*i) != 0 && "unreachable basic block found");
#endif

  return false;
}

/// instructionChanged - When the address of an instruction changes, this
/// method should be called so that live variables can update its internal
/// data structures.  This removes the records for OldMI, transfering them to
/// the records for NewMI.
void LiveVariables::instructionChanged(MachineInstr *OldMI,
                                       MachineInstr *NewMI) {
  // If the instruction defines any virtual registers, update the VarInfo for
  // the instruction.
  for (unsigned i = 0, e = OldMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = OldMI->getOperand(i);
    if (MO.isRegister() && MO.getReg() &&
        MRegisterInfo::isVirtualRegister(MO.getReg())) {
      unsigned Reg = MO.getReg();
      VarInfo &VI = getVarInfo(Reg);
      if (MO.isDef()) {
        // Update the defining instruction.
        if (VI.DefInst == OldMI)
          VI.DefInst = NewMI;
      }
      if (MO.isUse()) {
        // If this is a kill of the value, update the VI kills list.
        if (VI.removeKill(OldMI))
          VI.Kills.push_back(NewMI);   // Yes, there was a kill of it
      }
    }
  }

  // Move the killed information over...
  killed_iterator I, E;
  tie(I, E) = killed_range(OldMI);
  if (I != E) {
    std::vector<unsigned> &V = RegistersKilled[NewMI];
    bool WasEmpty = V.empty();
    V.insert(V.end(), I, E);
    if (!WasEmpty)
      std::sort(V.begin(), V.end());   // Keep the reg list sorted.
    RegistersKilled.erase(OldMI);
  }

  // Move the dead information over...
  tie(I, E) = dead_range(OldMI);
  if (I != E) {
    std::vector<unsigned> &V = RegistersDead[NewMI];
    bool WasEmpty = V.empty();
    V.insert(V.end(), I, E);
    if (!WasEmpty)
      std::sort(V.begin(), V.end());   // Keep the reg list sorted.
    RegistersDead.erase(OldMI);
  }
}
