//===-- LiveVariables.cpp - Live Variable Analysis for Machine Code -------===//
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
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CFG.h"
#include "Support/DepthFirstIterator.h"

static RegisterAnalysis<LiveVariables> X("livevars", "Live Variable Analysis");

const std::pair<MachineBasicBlock*, unsigned> &
LiveVariables::getMachineBasicBlockInfo(MachineBasicBlock *MBB) const{
  return BBMap.find(MBB->getBasicBlock())->second;
}

LiveVariables::VarInfo &LiveVariables::getVarInfo(unsigned RegIdx) {
  assert(RegIdx >= MRegisterInfo::FirstVirtualRegister &&
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
					    const BasicBlock *BB) {
  const std::pair<MachineBasicBlock*,unsigned> &Info = BBMap.find(BB)->second;
  MachineBasicBlock *MBB = Info.first;
  unsigned BBNum = Info.second;

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

  for (pred_const_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
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
  for (pred_const_iterator PI = pred_begin(BB), E = pred_end(BB);
       PI != E; ++PI)
    MarkVirtRegAliveInBlock(VRInfo, *PI);
}

void LiveVariables::HandlePhysRegUse(unsigned Reg, MachineInstr *MI) {
  if (PhysRegInfo[Reg]) {
    PhysRegInfo[Reg] = MI;
    PhysRegUsed[Reg] = true;
  } else {
    for (const unsigned *AliasSet = RegInfo->getAliasSet(Reg);
         *AliasSet; ++AliasSet) {
      if (MachineInstr *LastUse = PhysRegInfo[*AliasSet]) {
	PhysRegInfo[*AliasSet] = MI;
	PhysRegUsed[*AliasSet] = true;
      }
    }
  }
}

void LiveVariables::HandlePhysRegDef(unsigned Reg, MachineInstr *MI) {
  // Does this kill a previous version of this register?
  if (MachineInstr *LastUse = PhysRegInfo[Reg]) {
    if (PhysRegUsed[Reg])
      RegistersKilled.insert(std::make_pair(LastUse, Reg));
    else
      RegistersDead.insert(std::make_pair(LastUse, Reg));
  } else {
    for (const unsigned *AliasSet = RegInfo->getAliasSet(Reg);
         *AliasSet; ++AliasSet) {
      if (MachineInstr *LastUse = PhysRegInfo[*AliasSet]) {
	if (PhysRegUsed[*AliasSet])
	  RegistersKilled.insert(std::make_pair(LastUse, *AliasSet));
	else
	  RegistersDead.insert(std::make_pair(LastUse, *AliasSet));
	PhysRegInfo[*AliasSet] = 0;  // Kill the aliased register
      }
    }
  }
  PhysRegInfo[Reg] = MI;
  PhysRegUsed[Reg] = false;
}

bool LiveVariables::runOnMachineFunction(MachineFunction &MF) {
  // First time though, initialize AllocatablePhysicalRegisters for the target
  if (AllocatablePhysicalRegisters.empty()) {
    const MRegisterInfo &MRI = *MF.getTarget().getRegisterInfo();
    assert(&MRI && "Target doesn't have register information?");

    // Make space, initializing to false...
    AllocatablePhysicalRegisters.resize(MRegisterInfo::FirstVirtualRegister);

    // Loop over all of the register classes...
    for (MRegisterInfo::regclass_iterator RCI = MRI.regclass_begin(),
           E = MRI.regclass_end(); RCI != E; ++RCI)
      // Loop over all of the allocatable registers in the function...
      for (TargetRegisterClass::iterator I = (*RCI)->allocation_order_begin(MF),
             E = (*RCI)->allocation_order_end(MF); I != E; ++I)
        AllocatablePhysicalRegisters[*I] = true;  // The reg is allocatable!
  }

  // Build BBMap... 
  unsigned BBNum = 0;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    BBMap[I->getBasicBlock()] = std::make_pair(I, BBNum++);

  // PhysRegInfo - Keep track of which instruction was the last use of a
  // physical register.  This is a purely local property, because all physical
  // register references as presumed dead across basic blocks.
  //
  MachineInstr *PhysRegInfoA[MRegisterInfo::FirstVirtualRegister];
  bool          PhysRegUsedA[MRegisterInfo::FirstVirtualRegister];
  std::fill(PhysRegInfoA, PhysRegInfoA+MRegisterInfo::FirstVirtualRegister,
	    (MachineInstr*)0);
  PhysRegInfo = PhysRegInfoA;
  PhysRegUsed = PhysRegUsedA;

  const TargetInstrInfo &TII = MF.getTarget().getInstrInfo();
  RegInfo = MF.getTarget().getRegisterInfo();

  /// Get some space for a respectable number of registers...
  VirtRegInfo.resize(64);
  
  // Calculate live variable information in depth first order on the CFG of the
  // function.  This guarantees that we will see the definition of a virtual
  // register before its uses due to dominance properties of SSA (except for PHI
  // nodes, which are treated as a special case).
  //
  const BasicBlock *Entry = MF.getFunction()->begin();
  for (df_iterator<const BasicBlock*> DFI = df_begin(Entry), E = df_end(Entry);
       DFI != E; ++DFI) {
    const BasicBlock *BB = *DFI;
    std::pair<MachineBasicBlock*, unsigned> &BBRec = BBMap.find(BB)->second;
    MachineBasicBlock *MBB = BBRec.first;
    unsigned BBNum = BBRec.second;

    // Loop over all of the instructions, processing them.
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
	 I != E; ++I) {
      MachineInstr *MI = *I;
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
	if (MO.opIsUse() || MO.opIsDefAndUse()) {
	  if (MO.isVirtualRegister() && !MO.getVRegValueOrNull()) {
	    HandleVirtRegUse(getVarInfo(MO.getReg()), MBB, MI);
	  } else if (MO.isPhysicalRegister() && 
                     AllocatablePhysicalRegisters[MO.getReg()]) {
	    HandlePhysRegUse(MO.getReg(), MI);
	  }
	}
      }

      // Loop over implicit defs, defining them.
      if (const unsigned *ImplicitDefs = MID.ImplicitDefs)
	for (unsigned i = 0; ImplicitDefs[i]; ++i)
	  HandlePhysRegDef(ImplicitDefs[i], MI);

      // Process all explicit defs...
      for (unsigned i = 0; i != NumOperandsToProcess; ++i) {
	MachineOperand &MO = MI->getOperand(i);
	if (MO.opIsDefOnly() || MO.opIsDefAndUse()) {
	  if (MO.isVirtualRegister()) {
	    VarInfo &VRInfo = getVarInfo(MO.getReg());

	    assert(VRInfo.DefBlock == 0 && "Variable multiply defined!");
	    VRInfo.DefBlock = MBB;                           // Created here...
	    VRInfo.DefInst = MI;
	    VRInfo.Kills.push_back(std::make_pair(MBB, MI)); // Defaults to dead
	  } else if (MO.isPhysicalRegister() &&
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
    for (succ_const_iterator SI = succ_begin(BB), E = succ_end(BB);
         SI != E; ++SI) {
      MachineBasicBlock *Succ = BBMap.find(*SI)->second.first;
      
      // PHI nodes are guaranteed to be at the top of the block...
      for (MachineBasicBlock::iterator I = Succ->begin(), E = Succ->end();
	   I != E && (*I)->getOpcode() == TargetInstrInfo::PHI; ++I) {
        MachineInstr *MI = *I;
	for (unsigned i = 1; ; i += 2)
	  if (MI->getOperand(i+1).getMachineBasicBlock() == MBB) {
	    MachineOperand &MO = MI->getOperand(i);
	    if (!MO.getVRegValueOrNull()) {
	      VarInfo &VRInfo = getVarInfo(MO.getReg());

	      // Only mark it alive only in the block we are representing...
	      MarkVirtRegAliveInBlock(VRInfo, BB);
	      break;   // Found the PHI entry for this block...
	    }
	  }
      }
    }
    
    // Loop over PhysRegInfo, killing any registers that are available at the
    // end of the basic block.  This also resets the PhysRegInfo map.
    for (unsigned i = 0, e = MRegisterInfo::FirstVirtualRegister; i != e; ++i)
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
