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
#include "Support/DepthFirstIterator.h"
#include "Support/STLExtras.h"
using namespace llvm;

static RegisterAnalysis<LiveVariables> X("livevars", "Live Variable Analysis");

/// getIndexMachineBasicBlock() - Given a block index, return the
/// MachineBasicBlock corresponding to it.
MachineBasicBlock *LiveVariables::getIndexMachineBasicBlock(unsigned Idx) {
  if (BBIdxMap.empty()) {
    BBIdxMap.resize(BBMap.size());
    for (std::map<MachineBasicBlock*, unsigned>::iterator I = BBMap.begin(),
           E = BBMap.end(); I != E; ++I) {
      assert(BBIdxMap.size() > I->second && "Indices are not sequential");
      assert(BBIdxMap[I->second] == 0 && "Multiple idx collision!");
      BBIdxMap[I->second] = I->first;
    }
  }
  assert(Idx < BBIdxMap.size() && "BB Index out of range!");
  return BBIdxMap[Idx];
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



void LiveVariables::MarkVirtRegAliveInBlock(VarInfo &VRInfo,
					    MachineBasicBlock *MBB) {
  unsigned BBNum = getMachineBasicBlockIndex(MBB);

  // Check to see if this basic block is one of the killing blocks.  If so,
  // remove it...
  for (unsigned i = 0, e = VRInfo.Kills.size(); i != e; ++i)
    if (VRInfo.Kills[i].first == MBB) {
      VRInfo.Kills.erase(VRInfo.Kills.begin()+i);  // Erase entry
      break;
    }

  if (MBB == VRInfo.DefBlock) return;  // Terminate recursion

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
  // Check to see if this basic block is already a kill block...
  if (!VRInfo.Kills.empty() && VRInfo.Kills.back().first == MBB) {
    // Yes, this register is killed in this basic block already.  Increase the
    // live range by updating the kill instruction.
    VRInfo.Kills.back().second = MI;
    return;
  }

#ifndef NDEBUG
  for (unsigned i = 0, e = VRInfo.Kills.size(); i != e; ++i)
    assert(VRInfo.Kills[i].first != MBB && "entry should be at end!");
#endif

  assert(MBB != VRInfo.DefBlock && "Should have kill for defblock!");

  // Add a new kill entry for this basic block.
  VRInfo.Kills.push_back(std::make_pair(MBB, MI));

  // Update all dominating blocks to mark them known live.
  const BasicBlock *BB = MBB->getBasicBlock();
  for (MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(),
         E = MBB->pred_end(); PI != E; ++PI)
    MarkVirtRegAliveInBlock(VRInfo, *PI);
}

void LiveVariables::HandlePhysRegUse(unsigned Reg, MachineInstr *MI) {
  PhysRegInfo[Reg] = MI;
  PhysRegUsed[Reg] = true;
}

void LiveVariables::HandlePhysRegDef(unsigned Reg, MachineInstr *MI) {
  // Does this kill a previous version of this register?
  if (MachineInstr *LastUse = PhysRegInfo[Reg]) {
    if (PhysRegUsed[Reg])
      RegistersKilled.insert(std::make_pair(LastUse, Reg));
    else
      RegistersDead.insert(std::make_pair(LastUse, Reg));
  }
  PhysRegInfo[Reg] = MI;
  PhysRegUsed[Reg] = false;

  for (const unsigned *AliasSet = RegInfo->getAliasSet(Reg);
       *AliasSet; ++AliasSet) {
    unsigned Alias = *AliasSet;
    if (MachineInstr *LastUse = PhysRegInfo[Alias]) {
      if (PhysRegUsed[Alias])
	RegistersKilled.insert(std::make_pair(LastUse, Alias));
      else
	RegistersDead.insert(std::make_pair(LastUse, Alias));
    }
    PhysRegInfo[Alias] = MI;
    PhysRegUsed[Alias] = false;
  }
}

bool LiveVariables::runOnMachineFunction(MachineFunction &MF) {
  const TargetInstrInfo &TII = MF.getTarget().getInstrInfo();
  RegInfo = MF.getTarget().getRegisterInfo();
  assert(RegInfo && "Target doesn't have register information?");

  // First time though, initialize AllocatablePhysicalRegisters for the target
  if (AllocatablePhysicalRegisters.empty()) {
    // Make space, initializing to false...
    AllocatablePhysicalRegisters.resize(RegInfo->getNumRegs());

    // Loop over all of the register classes...
    for (MRegisterInfo::regclass_iterator RCI = RegInfo->regclass_begin(),
           E = RegInfo->regclass_end(); RCI != E; ++RCI)
      // Loop over all of the allocatable registers in the function...
      for (TargetRegisterClass::iterator I = (*RCI)->allocation_order_begin(MF),
             E = (*RCI)->allocation_order_end(MF); I != E; ++I)
        AllocatablePhysicalRegisters[*I] = true;  // The reg is allocatable!
  }

  // Build BBMap... 
  unsigned BBNum = 0;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    BBMap[I] = BBNum++;

  // PhysRegInfo - Keep track of which instruction was the last use of a
  // physical register.  This is a purely local property, because all physical
  // register references as presumed dead across basic blocks.
  //
  MachineInstr *PhysRegInfoA[RegInfo->getNumRegs()];
  bool          PhysRegUsedA[RegInfo->getNumRegs()];
  std::fill(PhysRegInfoA, PhysRegInfoA+RegInfo->getNumRegs(), (MachineInstr*)0);
  PhysRegInfo = PhysRegInfoA;
  PhysRegUsed = PhysRegUsedA;

  /// Get some space for a respectable number of registers...
  VirtRegInfo.resize(64);
  
  // Calculate live variable information in depth first order on the CFG of the
  // function.  This guarantees that we will see the definition of a virtual
  // register before its uses due to dominance properties of SSA (except for PHI
  // nodes, which are treated as a special case).
  //
  MachineBasicBlock *Entry = MF.begin();
  for (df_iterator<MachineBasicBlock*> DFI = df_begin(Entry), E = df_end(Entry);
       DFI != E; ++DFI) {
    MachineBasicBlock *MBB = *DFI;
    unsigned BBNum = getMachineBasicBlockIndex(MBB);

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
      for (const unsigned *ImplicitUses = MID.ImplicitUses;
           *ImplicitUses; ++ImplicitUses)
	HandlePhysRegUse(*ImplicitUses, MI);

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
      for (const unsigned *ImplicitDefs = MID.ImplicitDefs;
           *ImplicitDefs; ++ImplicitDefs)
        HandlePhysRegDef(*ImplicitDefs, MI);

      // Process all explicit defs...
      for (unsigned i = 0; i != NumOperandsToProcess; ++i) {
	MachineOperand &MO = MI->getOperand(i);
	if (MO.isDef() && MO.isRegister() && MO.getReg()) {
	  if (MRegisterInfo::isVirtualRegister(MO.getReg())) {
	    VarInfo &VRInfo = getVarInfo(MO.getReg());

	    assert(VRInfo.DefBlock == 0 && "Variable multiply defined!");
	    VRInfo.DefBlock = MBB;                           // Created here...
	    VRInfo.DefInst = MI;
	    VRInfo.Kills.push_back(std::make_pair(MBB, MI)); // Defaults to dead
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
	    if (!MO.getVRegValueOrNull()) {
	      VarInfo &VRInfo = getVarInfo(MO.getReg());

	      // Only mark it alive only in the block we are representing...
	      MarkVirtRegAliveInBlock(VRInfo, MBB);
	      break;   // Found the PHI entry for this block...
	    }
	  }
        }
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
      if (VirtRegInfo[i].Kills[j].second == VirtRegInfo[i].DefInst)
	RegistersDead.insert(std::make_pair(VirtRegInfo[i].Kills[j].second,
		    i + MRegisterInfo::FirstVirtualRegister));

      else
	RegistersKilled.insert(std::make_pair(VirtRegInfo[i].Kills[j].second,
		    i + MRegisterInfo::FirstVirtualRegister));
    }
  
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
    if (MO.isRegister() && MO.isDef() && MO.getReg() &&
        MRegisterInfo::isVirtualRegister(MO.getReg())) {
      unsigned Reg = MO.getReg();
      VarInfo &VI = getVarInfo(Reg);
      if (VI.DefInst == OldMI)
        VI.DefInst = NewMI;
    }
  }

  // Move the killed information over...
  killed_iterator I, E;
  tie(I, E) = killed_range(OldMI);
  std::vector<unsigned> Regs;
  for (killed_iterator A = I; A != E; ++A)
    Regs.push_back(A->second);
  RegistersKilled.erase(I, E);

  for (unsigned i = 0, e = Regs.size(); i != e; ++i)
    RegistersKilled.insert(std::make_pair(NewMI, Regs[i]));
  Regs.clear();


  // Move the dead information over...
  tie(I, E) = dead_range(OldMI);
  for (killed_iterator A = I; A != E; ++A)
    Regs.push_back(A->second);
  RegistersDead.erase(I, E);

  for (unsigned i = 0, e = Regs.size(); i != e; ++i)
    RegistersDead.insert(std::make_pair(NewMI, Regs[i]));
}
