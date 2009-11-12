//===----- AggressiveAntiDepBreaker.cpp - Anti-dep breaker -------- ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AggressiveAntiDepBreaker class, which
// implements register anti-dependence breaking during post-RA
// scheduling. It attempts to break all anti-dependencies within a
// block.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "post-RA-sched"
#include "AggressiveAntiDepBreaker.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static cl::opt<int>
AntiDepTrials("agg-antidep-trials",
              cl::desc("Maximum number of anti-dependency breaking passes"),
              cl::init(1), cl::Hidden);

AggressiveAntiDepState::AggressiveAntiDepState(MachineBasicBlock *BB) :
  GroupNodes(TargetRegisterInfo::FirstVirtualRegister, 0) {
  // Initialize all registers to be in their own group. Initially we
  // assign the register to the same-indexed GroupNode.
  for (unsigned i = 0; i < TargetRegisterInfo::FirstVirtualRegister; ++i)
    GroupNodeIndices[i] = i;

  // Initialize the indices to indicate that no registers are live.
  std::fill(KillIndices, array_endof(KillIndices), ~0u);
  std::fill(DefIndices, array_endof(DefIndices), BB->size());
}

unsigned AggressiveAntiDepState::GetGroup(unsigned Reg)
{
  unsigned Node = GroupNodeIndices[Reg];
  while (GroupNodes[Node] != Node)
    Node = GroupNodes[Node];

  return Node;
}

void AggressiveAntiDepState::GetGroupRegs(unsigned Group, std::vector<unsigned> &Regs)
{
  for (unsigned Reg = 0; Reg != TargetRegisterInfo::FirstVirtualRegister; ++Reg) {
    if (GetGroup(Reg) == Group)
      Regs.push_back(Reg);
  }
}

unsigned AggressiveAntiDepState::UnionGroups(unsigned Reg1, unsigned Reg2)
{
  assert(GroupNodes[0] == 0 && "GroupNode 0 not parent!");
  assert(GroupNodeIndices[0] == 0 && "Reg 0 not in Group 0!");
  
  // find group for each register
  unsigned Group1 = GetGroup(Reg1);
  unsigned Group2 = GetGroup(Reg2);
  
  // if either group is 0, then that must become the parent
  unsigned Parent = (Group1 == 0) ? Group1 : Group2;
  unsigned Other = (Parent == Group1) ? Group2 : Group1;
  GroupNodes.at(Other) = Parent;
  return Parent;
}
  
unsigned AggressiveAntiDepState::LeaveGroup(unsigned Reg)
{
  // Create a new GroupNode for Reg. Reg's existing GroupNode must
  // stay as is because there could be other GroupNodes referring to
  // it.
  unsigned idx = GroupNodes.size();
  GroupNodes.push_back(idx);
  GroupNodeIndices[Reg] = idx;
  return idx;
}

bool AggressiveAntiDepState::IsLive(unsigned Reg)
{
  // KillIndex must be defined and DefIndex not defined for a register
  // to be live.
  return((KillIndices[Reg] != ~0u) && (DefIndices[Reg] == ~0u));
}



AggressiveAntiDepBreaker::
AggressiveAntiDepBreaker(MachineFunction& MFi,
                         TargetSubtarget::ExcludedRCVector& ExcludedRCs) : 
  AntiDepBreaker(), MF(MFi),
  MRI(MF.getRegInfo()),
  TRI(MF.getTarget().getRegisterInfo()),
  AllocatableSet(TRI->getAllocatableSet(MF)),
  State(NULL), SavedState(NULL) {
  /* Remove all registers from excluded RCs from the allocatable
     register set. */
  for (unsigned i = 0, e = ExcludedRCs.size(); i < e; ++i) {
    BitVector NotRenameable = TRI->getAllocatableSet(MF, ExcludedRCs[i]).flip();
    AllocatableSet &= NotRenameable;
  }

  DEBUG(errs() << "AntiDep Renameable Registers:");
  DEBUG(for (int r = AllocatableSet.find_first(); r != -1; 
             r = AllocatableSet.find_next(r))
          errs() << " " << TRI->getName(r));
}

AggressiveAntiDepBreaker::~AggressiveAntiDepBreaker() {
  delete State;
  delete SavedState;
}

unsigned AggressiveAntiDepBreaker::GetMaxTrials() {
  if (AntiDepTrials <= 0)
    return 1;
  return AntiDepTrials;
}

void AggressiveAntiDepBreaker::StartBlock(MachineBasicBlock *BB) {
  assert(State == NULL);
  State = new AggressiveAntiDepState(BB);

  bool IsReturnBlock = (!BB->empty() && BB->back().getDesc().isReturn());
  unsigned *KillIndices = State->GetKillIndices();
  unsigned *DefIndices = State->GetDefIndices();

  // Determine the live-out physregs for this block.
  if (IsReturnBlock) {
    // In a return block, examine the function live-out regs.
    for (MachineRegisterInfo::liveout_iterator I = MRI.liveout_begin(),
         E = MRI.liveout_end(); I != E; ++I) {
      unsigned Reg = *I;
      State->UnionGroups(Reg, 0);
      KillIndices[Reg] = BB->size();
      DefIndices[Reg] = ~0u;
      // Repeat, for all aliases.
      for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
        unsigned AliasReg = *Alias;
        State->UnionGroups(AliasReg, 0);
        KillIndices[AliasReg] = BB->size();
        DefIndices[AliasReg] = ~0u;
      }
    }
  } else {
    // In a non-return block, examine the live-in regs of all successors.
    for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
         SE = BB->succ_end(); SI != SE; ++SI)
      for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
           E = (*SI)->livein_end(); I != E; ++I) {
        unsigned Reg = *I;
        State->UnionGroups(Reg, 0);
        KillIndices[Reg] = BB->size();
        DefIndices[Reg] = ~0u;
        // Repeat, for all aliases.
        for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
          unsigned AliasReg = *Alias;
          State->UnionGroups(AliasReg, 0);
          KillIndices[AliasReg] = BB->size();
          DefIndices[AliasReg] = ~0u;
        }
      }
  }

  // Mark live-out callee-saved registers. In a return block this is
  // all callee-saved registers. In non-return this is any
  // callee-saved register that is not saved in the prolog.
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  BitVector Pristine = MFI->getPristineRegs(BB);
  for (const unsigned *I = TRI->getCalleeSavedRegs(); *I; ++I) {
    unsigned Reg = *I;
    if (!IsReturnBlock && !Pristine.test(Reg)) continue;
    State->UnionGroups(Reg, 0);
    KillIndices[Reg] = BB->size();
    DefIndices[Reg] = ~0u;
    // Repeat, for all aliases.
    for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
      unsigned AliasReg = *Alias;
      State->UnionGroups(AliasReg, 0);
      KillIndices[AliasReg] = BB->size();
      DefIndices[AliasReg] = ~0u;
    }
  }
}

void AggressiveAntiDepBreaker::FinishBlock() {
  delete State;
  State = NULL;
  delete SavedState;
  SavedState = NULL;
}

void AggressiveAntiDepBreaker::Observe(MachineInstr *MI, unsigned Count,
                                     unsigned InsertPosIndex) {
  assert(Count < InsertPosIndex && "Instruction index out of expected range!");

  std::set<unsigned> PassthruRegs;
  GetPassthruRegs(MI, PassthruRegs);
  PrescanInstruction(MI, Count, PassthruRegs);
  ScanInstruction(MI, Count);

  DEBUG(errs() << "Observe: ");
  DEBUG(MI->dump());
  DEBUG(errs() << "\tRegs:");

  unsigned *DefIndices = State->GetDefIndices();
  for (unsigned Reg = 0; Reg != TargetRegisterInfo::FirstVirtualRegister; ++Reg) {
    // If Reg is current live, then mark that it can't be renamed as
    // we don't know the extent of its live-range anymore (now that it
    // has been scheduled). If it is not live but was defined in the
    // previous schedule region, then set its def index to the most
    // conservative location (i.e. the beginning of the previous
    // schedule region).
    if (State->IsLive(Reg)) {
      DEBUG(if (State->GetGroup(Reg) != 0)
              errs() << " " << TRI->getName(Reg) << "=g" << 
                State->GetGroup(Reg) << "->g0(region live-out)");
      State->UnionGroups(Reg, 0);
    } else if ((DefIndices[Reg] < InsertPosIndex) && (DefIndices[Reg] >= Count)) {
      DefIndices[Reg] = Count;
    }
  }
  DEBUG(errs() << '\n');

  // We're starting a new schedule region so forget any saved state.
  delete SavedState;
  SavedState = NULL;
}

bool AggressiveAntiDepBreaker::IsImplicitDefUse(MachineInstr *MI,
                                            MachineOperand& MO)
{
  if (!MO.isReg() || !MO.isImplicit())
    return false;

  unsigned Reg = MO.getReg();
  if (Reg == 0)
    return false;

  MachineOperand *Op = NULL;
  if (MO.isDef())
    Op = MI->findRegisterUseOperand(Reg, true);
  else
    Op = MI->findRegisterDefOperand(Reg);

  return((Op != NULL) && Op->isImplicit());
}

void AggressiveAntiDepBreaker::GetPassthruRegs(MachineInstr *MI,
                                           std::set<unsigned>& PassthruRegs) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;
    if ((MO.isDef() && MI->isRegTiedToUseOperand(i)) || 
        IsImplicitDefUse(MI, MO)) {
      const unsigned Reg = MO.getReg();
      PassthruRegs.insert(Reg);
      for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
           *Subreg; ++Subreg) {
        PassthruRegs.insert(*Subreg);
      }
    }
  }
}

/// AntiDepPathStep - Return SUnit that SU has an anti-dependence on.
static void AntiDepPathStep(SUnit *SU, AntiDepBreaker::AntiDepRegVector& Regs,
                            std::vector<SDep*>& Edges) {
  AntiDepBreaker::AntiDepRegSet RegSet;
  for (unsigned i = 0, e = Regs.size(); i < e; ++i)
    RegSet.insert(Regs[i]);

  for (SUnit::pred_iterator P = SU->Preds.begin(), PE = SU->Preds.end();
       P != PE; ++P) {
    if ((P->getKind() == SDep::Anti) || (P->getKind() == SDep::Output)) {
      unsigned Reg = P->getReg();
      if (RegSet.count(Reg) != 0) {
        Edges.push_back(&*P);
        RegSet.erase(Reg);
      }
    }
  }

  assert(RegSet.empty() && "Expected all antidep registers to be found");
}

void AggressiveAntiDepBreaker::HandleLastUse(unsigned Reg, unsigned KillIdx,
                                             const char *tag) {
  unsigned *KillIndices = State->GetKillIndices();
  unsigned *DefIndices = State->GetDefIndices();
  std::multimap<unsigned, AggressiveAntiDepState::RegisterReference>& 
    RegRefs = State->GetRegRefs();

  if (!State->IsLive(Reg)) {
    KillIndices[Reg] = KillIdx;
    DefIndices[Reg] = ~0u;
    RegRefs.erase(Reg);
    State->LeaveGroup(Reg);
    DEBUG(errs() << "->g" << State->GetGroup(Reg) << tag);
  }
  // Repeat for subregisters.
  for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
       *Subreg; ++Subreg) {
    unsigned SubregReg = *Subreg;
    if (!State->IsLive(SubregReg)) {
      KillIndices[SubregReg] = KillIdx;
      DefIndices[SubregReg] = ~0u;
      RegRefs.erase(SubregReg);
      State->LeaveGroup(SubregReg);
      DEBUG(errs() << " " << TRI->getName(SubregReg) << "->g" <<
            State->GetGroup(SubregReg) << tag);
    }
  }
}

void AggressiveAntiDepBreaker::PrescanInstruction(MachineInstr *MI, unsigned Count,
                                              std::set<unsigned>& PassthruRegs) {
  unsigned *DefIndices = State->GetDefIndices();
  std::multimap<unsigned, AggressiveAntiDepState::RegisterReference>& 
    RegRefs = State->GetRegRefs();

  // Handle dead defs by simulating a last-use of the register just
  // after the def. A dead def can occur because the def is truely
  // dead, or because only a subregister is live at the def. If we
  // don't do this the dead def will be incorrectly merged into the
  // previous def.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isDef()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;
    
    DEBUG(errs() << "\tDead Def: " << TRI->getName(Reg));
    HandleLastUse(Reg, Count + 1, "");
    DEBUG(errs() << '\n');
  }

  DEBUG(errs() << "\tDef Groups:");
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isDef()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;

    DEBUG(errs() << " " << TRI->getName(Reg) << "=g" << State->GetGroup(Reg)); 

    // If MI's defs have a special allocation requirement, don't allow
    // any def registers to be changed. Also assume all registers
    // defined in a call must not be changed (ABI).
    if (MI->getDesc().isCall() || MI->getDesc().hasExtraDefRegAllocReq()) {
      DEBUG(if (State->GetGroup(Reg) != 0) errs() << "->g0(alloc-req)");
      State->UnionGroups(Reg, 0);
    }

    // Any aliased that are live at this point are completely or
    // partially defined here, so group those aliases with Reg.
    for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
      unsigned AliasReg = *Alias;
      if (State->IsLive(AliasReg)) {
        State->UnionGroups(Reg, AliasReg);
        DEBUG(errs() << "->g" << State->GetGroup(Reg) << "(via " << 
              TRI->getName(AliasReg) << ")");
      }
    }
    
    // Note register reference...
    const TargetRegisterClass *RC = NULL;
    if (i < MI->getDesc().getNumOperands())
      RC = MI->getDesc().OpInfo[i].getRegClass(TRI);
    AggressiveAntiDepState::RegisterReference RR = { &MO, RC };
    RegRefs.insert(std::make_pair(Reg, RR));
  }

  DEBUG(errs() << '\n');

  // Scan the register defs for this instruction and update
  // live-ranges.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isDef()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;
    // Ignore passthru registers for liveness...
    if (PassthruRegs.count(Reg) != 0) continue;

    // Update def for Reg and subregs.
    DefIndices[Reg] = Count;
    for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
         *Subreg; ++Subreg) {
      unsigned SubregReg = *Subreg;
      DefIndices[SubregReg] = Count;
    }
  }
}

void AggressiveAntiDepBreaker::ScanInstruction(MachineInstr *MI,
                                           unsigned Count) {
  DEBUG(errs() << "\tUse Groups:");
  std::multimap<unsigned, AggressiveAntiDepState::RegisterReference>& 
    RegRefs = State->GetRegRefs();

  // Scan the register uses for this instruction and update
  // live-ranges, groups and RegRefs.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isUse()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;
    
    DEBUG(errs() << " " << TRI->getName(Reg) << "=g" << 
          State->GetGroup(Reg)); 

    // It wasn't previously live but now it is, this is a kill. Forget
    // the previous live-range information and start a new live-range
    // for the register.
    HandleLastUse(Reg, Count, "(last-use)");

    // If MI's uses have special allocation requirement, don't allow
    // any use registers to be changed. Also assume all registers
    // used in a call must not be changed (ABI).
    if (MI->getDesc().isCall() || MI->getDesc().hasExtraSrcRegAllocReq()) {
      DEBUG(if (State->GetGroup(Reg) != 0) errs() << "->g0(alloc-req)");
      State->UnionGroups(Reg, 0);
    }

    // Note register reference...
    const TargetRegisterClass *RC = NULL;
    if (i < MI->getDesc().getNumOperands())
      RC = MI->getDesc().OpInfo[i].getRegClass(TRI);
    AggressiveAntiDepState::RegisterReference RR = { &MO, RC };
    RegRefs.insert(std::make_pair(Reg, RR));
  }
  
  DEBUG(errs() << '\n');

  // Form a group of all defs and uses of a KILL instruction to ensure
  // that all registers are renamed as a group.
  if (MI->getOpcode() == TargetInstrInfo::KILL) {
    DEBUG(errs() << "\tKill Group:");

    unsigned FirstReg = 0;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;
      
      if (FirstReg != 0) {
        DEBUG(errs() << "=" << TRI->getName(Reg));
        State->UnionGroups(FirstReg, Reg);
      } else {
        DEBUG(errs() << " " << TRI->getName(Reg));
        FirstReg = Reg;
      }
    }
  
    DEBUG(errs() << "->g" << State->GetGroup(FirstReg) << '\n');
  }
}

BitVector AggressiveAntiDepBreaker::GetRenameRegisters(unsigned Reg) {
  BitVector BV(TRI->getNumRegs(), false);
  bool first = true;

  // Check all references that need rewriting for Reg. For each, use
  // the corresponding register class to narrow the set of registers
  // that are appropriate for renaming.
  std::pair<std::multimap<unsigned, 
                     AggressiveAntiDepState::RegisterReference>::iterator,
            std::multimap<unsigned,
                     AggressiveAntiDepState::RegisterReference>::iterator>
    Range = State->GetRegRefs().equal_range(Reg);
  for (std::multimap<unsigned, AggressiveAntiDepState::RegisterReference>::iterator
         Q = Range.first, QE = Range.second; Q != QE; ++Q) {
    const TargetRegisterClass *RC = Q->second.RC;
    if (RC == NULL) continue;

    BitVector RCBV = TRI->getAllocatableSet(MF, RC);
    if (first) {
      BV |= RCBV;
      first = false;
    } else {
      BV &= RCBV;
    }

    DEBUG(errs() << " " << RC->getName());
  }
  
  return BV;
}  

bool AggressiveAntiDepBreaker::FindSuitableFreeRegisters(
                                unsigned AntiDepGroupIndex,
                                RenameOrderType& RenameOrder,
                                std::map<unsigned, unsigned> &RenameMap) {
  unsigned *KillIndices = State->GetKillIndices();
  unsigned *DefIndices = State->GetDefIndices();
  std::multimap<unsigned, AggressiveAntiDepState::RegisterReference>& 
    RegRefs = State->GetRegRefs();

  // Collect all registers in the same group as AntiDepReg. These all
  // need to be renamed together if we are to break the
  // anti-dependence.
  std::vector<unsigned> Regs;
  State->GetGroupRegs(AntiDepGroupIndex, Regs);
  assert(Regs.size() > 0 && "Empty register group!");
  if (Regs.size() == 0)
    return false;

  // Find the "superest" register in the group. At the same time,
  // collect the BitVector of registers that can be used to rename
  // each register.
  DEBUG(errs() << "\tRename Candidates for Group g" << AntiDepGroupIndex << ":\n");
  std::map<unsigned, BitVector> RenameRegisterMap;
  unsigned SuperReg = 0;
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    unsigned Reg = Regs[i];
    if ((SuperReg == 0) || TRI->isSuperRegister(SuperReg, Reg))
      SuperReg = Reg;

    // If Reg has any references, then collect possible rename regs
    if (RegRefs.count(Reg) > 0) {
      DEBUG(errs() << "\t\t" << TRI->getName(Reg) << ":");
    
      BitVector BV = GetRenameRegisters(Reg);
      RenameRegisterMap.insert(std::pair<unsigned, BitVector>(Reg, BV));

      DEBUG(errs() << " ::");
      DEBUG(for (int r = BV.find_first(); r != -1; r = BV.find_next(r))
              errs() << " " << TRI->getName(r));
      DEBUG(errs() << "\n");
    }
  }

  // All group registers should be a subreg of SuperReg.
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    unsigned Reg = Regs[i];
    if (Reg == SuperReg) continue;
    bool IsSub = TRI->isSubRegister(SuperReg, Reg);
    assert(IsSub && "Expecting group subregister");
    if (!IsSub)
      return false;
  }

  // FIXME: for now just handle single register in group case...
  // FIXME: check only regs that have references...
  if (Regs.size() > 1)
    return false;

  // Check each possible rename register for SuperReg in round-robin
  // order. If that register is available, and the corresponding
  // registers are available for the other group subregisters, then we
  // can use those registers to rename.
  BitVector SuperBV = RenameRegisterMap[SuperReg];
  const TargetRegisterClass *SuperRC = 
    TRI->getPhysicalRegisterRegClass(SuperReg, MVT::Other);
  
  const TargetRegisterClass::iterator RB = SuperRC->allocation_order_begin(MF);
  const TargetRegisterClass::iterator RE = SuperRC->allocation_order_end(MF);
  if (RB == RE) {
    DEBUG(errs() << "\tEmpty Regclass!!\n");
    return false;
  }

  if (RenameOrder.count(SuperRC) == 0)
    RenameOrder.insert(RenameOrderType::value_type(SuperRC, RE));

  DEBUG(errs() << "\tFind Register:");

  const TargetRegisterClass::iterator OrigR = RenameOrder[SuperRC];
  const TargetRegisterClass::iterator EndR = ((OrigR == RE) ? RB : OrigR);
  TargetRegisterClass::iterator R = OrigR;
  do {
    if (R == RB) R = RE;
    --R;
    const unsigned Reg = *R;
    // Don't replace a register with itself.
    if (Reg == SuperReg) continue;
    
    DEBUG(errs() << " " << TRI->getName(Reg));
    
    // If Reg is dead and Reg's most recent def is not before
    // SuperRegs's kill, it's safe to replace SuperReg with Reg. We
    // must also check all subregisters of Reg.
    if (State->IsLive(Reg) || (KillIndices[SuperReg] > DefIndices[Reg])) {
      DEBUG(errs() << "(live)");
      continue;
    } else {
      bool found = false;
      for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
           *Subreg; ++Subreg) {
        unsigned SubregReg = *Subreg;
        if (State->IsLive(SubregReg) || (KillIndices[SuperReg] > DefIndices[SubregReg])) {
          DEBUG(errs() << "(subreg " << TRI->getName(SubregReg) << " live)");
          found = true;
          break;
        }
      }
      if (found)
        continue;
    }
    
    if (Reg != 0) { 
      DEBUG(errs() << '\n');
      RenameOrder.erase(SuperRC);
      RenameOrder.insert(RenameOrderType::value_type(SuperRC, R));
      RenameMap.insert(std::pair<unsigned, unsigned>(SuperReg, Reg));
      return true;
    }
  } while (R != EndR);

  DEBUG(errs() << '\n');

  // No registers are free and available!
  return false;
}

/// BreakAntiDependencies - Identifiy anti-dependencies within the
/// ScheduleDAG and break them by renaming registers.
///
unsigned AggressiveAntiDepBreaker::BreakAntiDependencies(
                              std::vector<SUnit>& SUnits,
                              CandidateMap& Candidates,
                              MachineBasicBlock::iterator& Begin,
                              MachineBasicBlock::iterator& End,
                              unsigned InsertPosIndex) {
  unsigned *KillIndices = State->GetKillIndices();
  unsigned *DefIndices = State->GetDefIndices();
  std::multimap<unsigned, AggressiveAntiDepState::RegisterReference>& 
    RegRefs = State->GetRegRefs();

  // The code below assumes that there is at least one instruction,
  // so just duck out immediately if the block is empty.
  if (SUnits.empty()) return 0;
  
  // Manage saved state to enable multiple passes...
  if (AntiDepTrials > 1) {
    if (SavedState == NULL) {
      SavedState = new AggressiveAntiDepState(*State);
    } else {
      delete State;
      State = new AggressiveAntiDepState(*SavedState);
    }
  }
  
  // For each regclass the next register to use for renaming.
  RenameOrderType RenameOrder;

  // ...need a map from MI to SUnit.
  std::map<MachineInstr *, SUnit *> MISUnitMap;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    SUnit *SU = &SUnits[i];
    MISUnitMap.insert(std::pair<MachineInstr *, SUnit *>(SU->getInstr(), SU));
  }

  // Even if there are no anti-dependencies we still need to go
  // through the instructions to update Def, Kills, etc.
#ifndef NDEBUG 
  if (Candidates.empty()) {
    DEBUG(errs() << "\n===== No anti-dependency candidates\n");
  } else {
    DEBUG(errs() << "\n===== Attempting to break " << Candidates.size() << 
          " anti-dependencies\n");
    DEBUG(errs() << "Available regs:");
    for (unsigned Reg = 0; Reg < TRI->getNumRegs(); ++Reg) {
      if (!State->IsLive(Reg))
        DEBUG(errs() << " " << TRI->getName(Reg));
    }
    DEBUG(errs() << '\n');
  }
#endif

  // Attempt to break anti-dependence edges. Walk the instructions
  // from the bottom up, tracking information about liveness as we go
  // to help determine which registers are available.
  unsigned Broken = 0;
  unsigned Count = InsertPosIndex - 1;
  for (MachineBasicBlock::iterator I = End, E = Begin;
       I != E; --Count) {
    MachineInstr *MI = --I;

    DEBUG(errs() << "Anti: ");
    DEBUG(MI->dump());

    std::set<unsigned> PassthruRegs;
    GetPassthruRegs(MI, PassthruRegs);

    // Process the defs in MI...
    PrescanInstruction(MI, Count, PassthruRegs);

    std::vector<SDep*> Edges;
    SUnit *PathSU = MISUnitMap[MI];
    AntiDepBreaker::CandidateMap::iterator 
      citer = Candidates.find(PathSU);
    if (citer != Candidates.end())
      AntiDepPathStep(PathSU, citer->second, Edges);
      
    // Ignore KILL instructions (they form a group in ScanInstruction
    // but don't cause any anti-dependence breaking themselves)
    if (MI->getOpcode() != TargetInstrInfo::KILL) {
      // Attempt to break each anti-dependency...
      for (unsigned i = 0, e = Edges.size(); i != e; ++i) {
        SDep *Edge = Edges[i];
        SUnit *NextSU = Edge->getSUnit();
        
        if ((Edge->getKind() != SDep::Anti) &&
            (Edge->getKind() != SDep::Output)) continue;
        
        unsigned AntiDepReg = Edge->getReg();
        DEBUG(errs() << "\tAntidep reg: " << TRI->getName(AntiDepReg));
        assert(AntiDepReg != 0 && "Anti-dependence on reg0?");
        
        if (!AllocatableSet.test(AntiDepReg)) {
          // Don't break anti-dependencies on non-allocatable registers.
          DEBUG(errs() << " (non-allocatable)\n");
          continue;
        } else if (PassthruRegs.count(AntiDepReg) != 0) {
          // If the anti-dep register liveness "passes-thru", then
          // don't try to change it. It will be changed along with
          // the use if required to break an earlier antidep.
          DEBUG(errs() << " (passthru)\n");
          continue;
        } else {
          // No anti-dep breaking for implicit deps
          MachineOperand *AntiDepOp = MI->findRegisterDefOperand(AntiDepReg);
          assert(AntiDepOp != NULL && "Can't find index for defined register operand");
          if ((AntiDepOp == NULL) || AntiDepOp->isImplicit()) {
            DEBUG(errs() << " (implicit)\n");
            continue;
          }
          
          // If the SUnit has other dependencies on the SUnit that
          // it anti-depends on, don't bother breaking the
          // anti-dependency since those edges would prevent such
          // units from being scheduled past each other
          // regardless.
          for (SUnit::pred_iterator P = PathSU->Preds.begin(),
                 PE = PathSU->Preds.end(); P != PE; ++P) {
            if ((P->getSUnit() == NextSU) && (P->getKind() != SDep::Anti)) {
              DEBUG(errs() << " (real dependency)\n");
              AntiDepReg = 0;
              break;
            }
          }
          
          if (AntiDepReg == 0) continue;
        }
        
        assert(AntiDepReg != 0);
        if (AntiDepReg == 0) continue;
        
        // Determine AntiDepReg's register group.
        const unsigned GroupIndex = State->GetGroup(AntiDepReg);
        if (GroupIndex == 0) {
          DEBUG(errs() << " (zero group)\n");
          continue;
        }
        
        DEBUG(errs() << '\n');
        
        // Look for a suitable register to use to break the anti-dependence.
        std::map<unsigned, unsigned> RenameMap;
        if (FindSuitableFreeRegisters(GroupIndex, RenameOrder, RenameMap)) {
          DEBUG(errs() << "\tBreaking anti-dependence edge on "
                << TRI->getName(AntiDepReg) << ":");
          
          // Handle each group register...
          for (std::map<unsigned, unsigned>::iterator
                 S = RenameMap.begin(), E = RenameMap.end(); S != E; ++S) {
            unsigned CurrReg = S->first;
            unsigned NewReg = S->second;
            
            DEBUG(errs() << " " << TRI->getName(CurrReg) << "->" << 
                  TRI->getName(NewReg) << "(" <<  
                  RegRefs.count(CurrReg) << " refs)");
            
            // Update the references to the old register CurrReg to
            // refer to the new register NewReg.
            std::pair<std::multimap<unsigned, 
                              AggressiveAntiDepState::RegisterReference>::iterator,
                      std::multimap<unsigned,
                              AggressiveAntiDepState::RegisterReference>::iterator>
              Range = RegRefs.equal_range(CurrReg);
            for (std::multimap<unsigned, AggressiveAntiDepState::RegisterReference>::iterator
                   Q = Range.first, QE = Range.second; Q != QE; ++Q) {
              Q->second.Operand->setReg(NewReg);
            }
            
            // We just went back in time and modified history; the
            // liveness information for CurrReg is now inconsistent. Set
            // the state as if it were dead.
            State->UnionGroups(NewReg, 0);
            RegRefs.erase(NewReg);
            DefIndices[NewReg] = DefIndices[CurrReg];
            KillIndices[NewReg] = KillIndices[CurrReg];
            
            State->UnionGroups(CurrReg, 0);
            RegRefs.erase(CurrReg);
            DefIndices[CurrReg] = KillIndices[CurrReg];
            KillIndices[CurrReg] = ~0u;
            assert(((KillIndices[CurrReg] == ~0u) !=
                    (DefIndices[CurrReg] == ~0u)) &&
                   "Kill and Def maps aren't consistent for AntiDepReg!");
          }
          
          ++Broken;
          DEBUG(errs() << '\n');
        }
      }
    }

    ScanInstruction(MI, Count);
  }
  
  return Broken;
}
