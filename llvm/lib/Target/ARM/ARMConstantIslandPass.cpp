//===-- ARMConstantIslandPass.cpp - ARM constant islands --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that splits the constant pool up into 'islands'
// which are scattered through-out the function.  This is required due to the
// limited pc-relative displacements that ARM has.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-cp-islands"
#include "ARM.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMInstrInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include <iostream>
using namespace llvm;

STATISTIC(NumSplit,    "Number of uncond branches inserted");
STATISTIC(NumCBrFixed, "Number of cond branches fixed");
STATISTIC(NumUBrFixed, "Number of uncond branches fixed");

namespace {
  /// ARMConstantIslands - Due to limited pc-relative displacements, ARM
  /// requires constant pool entries to be scattered among the instructions
  /// inside a function.  To do this, it completely ignores the normal LLVM
  /// constant pool, instead, it places constants where-ever it feels like with
  /// special instructions.
  ///
  /// The terminology used in this pass includes:
  ///   Islands - Clumps of constants placed in the function.
  ///   Water   - Potential places where an island could be formed.
  ///   CPE     - A constant pool entry that has been placed somewhere, which
  ///             tracks a list of users.
  class VISIBILITY_HIDDEN ARMConstantIslands : public MachineFunctionPass {
    /// NextUID - Assign unique ID's to CPE's.
    unsigned NextUID;
    
    /// BBSizes - The size of each MachineBasicBlock in bytes of code, indexed
    /// by MBB Number.
    std::vector<unsigned> BBSizes;
    
    /// WaterList - A sorted list of basic blocks where islands could be placed
    /// (i.e. blocks that don't fall through to the following block, due
    /// to a return, unreachable, or unconditional branch).
    std::vector<MachineBasicBlock*> WaterList;
    
    /// CPUser - One user of a constant pool, keeping the machine instruction
    /// pointer, the constant pool being referenced, and the max displacement
    /// allowed from the instruction to the CP.
    struct CPUser {
      MachineInstr *MI;
      MachineInstr *CPEMI;
      unsigned MaxDisp;
      CPUser(MachineInstr *mi, MachineInstr *cpemi, unsigned maxdisp)
        : MI(mi), CPEMI(cpemi), MaxDisp(maxdisp) {}
    };
    
    /// CPUsers - Keep track of all of the machine instructions that use various
    /// constant pools and their max displacement.
    std::vector<CPUser> CPUsers;
    
    /// ImmBranch - One per immediate branch, keeping the machine instruction
    /// pointer, conditional or unconditional, the max displacement,
    /// and (if isCond is true) the corresponding unconditional branch
    /// opcode.
    struct ImmBranch {
      MachineInstr *MI;
      unsigned MaxDisp : 31;
      bool isCond : 1;
      int UncondBr;
      ImmBranch(MachineInstr *mi, unsigned maxdisp, bool cond, int ubr)
        : MI(mi), MaxDisp(maxdisp), isCond(cond), UncondBr(ubr) {}
    };

    /// Branches - Keep track of all the immediate branch instructions.
    ///
    std::vector<ImmBranch> ImmBranches;

    /// PushPopMIs - Keep track of all the Thumb push / pop instructions.
    ///
    std::vector<MachineInstr*> PushPopMIs;

    /// HasFarJump - True if any far jump instruction has been emitted during
    /// the branch fix up pass.
    bool HasFarJump;

    const TargetInstrInfo *TII;
    const ARMFunctionInfo *AFI;
  public:
    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "ARM constant island placement and branch shortening pass";
    }
    
  private:
    void DoInitialPlacement(MachineFunction &Fn,
                            std::vector<MachineInstr*> &CPEMIs);
    void InitialFunctionScan(MachineFunction &Fn,
                             const std::vector<MachineInstr*> &CPEMIs);
    MachineBasicBlock *SplitBlockBeforeInstr(MachineInstr *MI);
    void UpdateForInsertedWaterBlock(MachineBasicBlock *NewBB);
    bool HandleConstantPoolUser(MachineFunction &Fn, CPUser &U);
    bool CPEIsInRange(MachineInstr *MI, MachineInstr *CPEMI, unsigned Disp);
    bool BBIsInRange(MachineInstr *MI, MachineBasicBlock *BB, unsigned Disp);
    bool FixUpImmediateBr(MachineFunction &Fn, ImmBranch &Br);
    bool FixUpConditionalBr(MachineFunction &Fn, ImmBranch &Br);
    bool FixUpUnconditionalBr(MachineFunction &Fn, ImmBranch &Br);
    bool UndoLRSpillRestore();

    unsigned GetOffsetOf(MachineInstr *MI) const;
    unsigned GetOffsetOf(MachineBasicBlock *MBB) const;
  };
}

/// createARMConstantIslandPass - returns an instance of the constpool
/// island pass.
FunctionPass *llvm::createARMConstantIslandPass() {
  return new ARMConstantIslands();
}

bool ARMConstantIslands::runOnMachineFunction(MachineFunction &Fn) {
  MachineConstantPool &MCP = *Fn.getConstantPool();
  
  TII = Fn.getTarget().getInstrInfo();
  AFI = Fn.getInfo<ARMFunctionInfo>();

  HasFarJump = false;

  // Renumber all of the machine basic blocks in the function, guaranteeing that
  // the numbers agree with the position of the block in the function.
  Fn.RenumberBlocks();

  // Perform the initial placement of the constant pool entries.  To start with,
  // we put them all at the end of the function.
  std::vector<MachineInstr*> CPEMIs;
  if (!MCP.isEmpty())
    DoInitialPlacement(Fn, CPEMIs);
  
  /// The next UID to take is the first unused one.
  NextUID = CPEMIs.size();
  
  // Do the initial scan of the function, building up information about the
  // sizes of each block, the location of all the water, and finding all of the
  // constant pool users.
  InitialFunctionScan(Fn, CPEMIs);
  CPEMIs.clear();
  
  // Iteratively place constant pool entries and fix up branches until there
  // is no change.
  bool MadeChange = false;
  while (true) {
    bool Change = false;
    for (unsigned i = 0, e = CPUsers.size(); i != e; ++i)
      Change |= HandleConstantPoolUser(Fn, CPUsers[i]);
    for (unsigned i = 0, e = ImmBranches.size(); i != e; ++i)
      Change |= FixUpImmediateBr(Fn, ImmBranches[i]);
    if (!Change)
      break;
    MadeChange = true;
  }
  
  // If LR has been forced spilled and no far jumps (i.e. BL) has been issued.
  // Undo the spill / restore of LR if possible.
  if (!HasFarJump && AFI->isLRForceSpilled() && AFI->isThumbFunction())
    MadeChange |= UndoLRSpillRestore();

  BBSizes.clear();
  WaterList.clear();
  CPUsers.clear();
  ImmBranches.clear();

  return MadeChange;
}

/// DoInitialPlacement - Perform the initial placement of the constant pool
/// entries.  To start with, we put them all at the end of the function.
void ARMConstantIslands::DoInitialPlacement(MachineFunction &Fn,
                                            std::vector<MachineInstr*> &CPEMIs){
  // Create the basic block to hold the CPE's.
  MachineBasicBlock *BB = new MachineBasicBlock();
  Fn.getBasicBlockList().push_back(BB);
  
  // Add all of the constants from the constant pool to the end block, use an
  // identity mapping of CPI's to CPE's.
  const std::vector<MachineConstantPoolEntry> &CPs =
    Fn.getConstantPool()->getConstants();
  
  const TargetData &TD = *Fn.getTarget().getTargetData();
  for (unsigned i = 0, e = CPs.size(); i != e; ++i) {
    unsigned Size = TD.getTypeSize(CPs[i].getType());
    // Verify that all constant pool entries are a multiple of 4 bytes.  If not,
    // we would have to pad them out or something so that instructions stay
    // aligned.
    assert((Size & 3) == 0 && "CP Entry not multiple of 4 bytes!");
    MachineInstr *CPEMI =
      BuildMI(BB, TII->get(ARM::CONSTPOOL_ENTRY))
                           .addImm(i).addConstantPoolIndex(i).addImm(Size);
    CPEMIs.push_back(CPEMI);
    DEBUG(std::cerr << "Moved CPI#" << i << " to end of function as #"
                    << i << "\n");
  }
}

/// BBHasFallthrough - Return true of the specified basic block can fallthrough
/// into the block immediately after it.
static bool BBHasFallthrough(MachineBasicBlock *MBB) {
  // Get the next machine basic block in the function.
  MachineFunction::iterator MBBI = MBB;
  if (next(MBBI) == MBB->getParent()->end())  // Can't fall off end of function.
    return false;
  
  MachineBasicBlock *NextBB = next(MBBI);
  for (MachineBasicBlock::succ_iterator I = MBB->succ_begin(),
       E = MBB->succ_end(); I != E; ++I)
    if (*I == NextBB)
      return true;
  
  return false;
}

/// InitialFunctionScan - Do the initial scan of the function, building up
/// information about the sizes of each block, the location of all the water,
/// and finding all of the constant pool users.
void ARMConstantIslands::InitialFunctionScan(MachineFunction &Fn,
                                     const std::vector<MachineInstr*> &CPEMIs) {
  for (MachineFunction::iterator MBBI = Fn.begin(), E = Fn.end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock &MBB = *MBBI;
    
    // If this block doesn't fall through into the next MBB, then this is
    // 'water' that a constant pool island could be placed.
    if (!BBHasFallthrough(&MBB))
      WaterList.push_back(&MBB);
    
    unsigned MBBSize = 0;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
         I != E; ++I) {
      // Add instruction size to MBBSize.
      MBBSize += ARM::GetInstSize(I);

      int Opc = I->getOpcode();
      if (TII->isBranch(Opc)) {
        bool isCond = false;
        unsigned Bits = 0;
        unsigned Scale = 1;
        int UOpc = Opc;
        switch (Opc) {
        default:
          continue;  // Ignore JT branches
        case ARM::Bcc:
          isCond = true;
          UOpc = ARM::B;
          // Fallthrough
        case ARM::B:
          Bits = 24;
          Scale = 4;
          break;
        case ARM::tBcc:
          isCond = true;
          UOpc = ARM::tB;
          Bits = 8;
          Scale = 2;
          break;
        case ARM::tB:
          Bits = 11;
          Scale = 2;
          break;
        }
        unsigned MaxDisp = (1 << (Bits-1)) * Scale;
        ImmBranches.push_back(ImmBranch(I, MaxDisp, isCond, UOpc));
      }

      if (Opc == ARM::tPUSH || Opc == ARM::tPOP_RET)
        PushPopMIs.push_back(I);

      // Scan the instructions for constant pool operands.
      for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op)
        if (I->getOperand(op).isConstantPoolIndex()) {
          // We found one.  The addressing mode tells us the max displacement
          // from the PC that this instruction permits.
          unsigned MaxOffs = 0;
          
          // Basic size info comes from the TSFlags field.
          unsigned TSFlags = I->getInstrDescriptor()->TSFlags;
          switch (TSFlags & ARMII::AddrModeMask) {
          default: 
            // Constant pool entries can reach anything.
            if (I->getOpcode() == ARM::CONSTPOOL_ENTRY)
              continue;
            assert(0 && "Unknown addressing mode for CP reference!");
          case ARMII::AddrMode1: // AM1: 8 bits << 2
            MaxOffs = 1 << (8+2);   // Taking the address of a CP entry.
            break;
          case ARMII::AddrMode2:
            MaxOffs = 1 << 12;   // +-offset_12
            break;
          case ARMII::AddrMode3:
            MaxOffs = 1 << 8;   // +-offset_8
            break;
            // addrmode4 has no immediate offset.
          case ARMII::AddrMode5:
            MaxOffs = 1 << (8+2);   // +-(offset_8*4)
            break;
          case ARMII::AddrModeT1:
            MaxOffs = 1 << 5;
            break;
          case ARMII::AddrModeT2:
            MaxOffs = 1 << (5+1);
            break;
          case ARMII::AddrModeT4:
            MaxOffs = 1 << (5+2);
            break;
          case ARMII::AddrModeTs:
            MaxOffs = 1 << (8+2);
            break;
          }
          
          // Remember that this is a user of a CP entry.
          MachineInstr *CPEMI =CPEMIs[I->getOperand(op).getConstantPoolIndex()];
          CPUsers.push_back(CPUser(I, CPEMI, MaxOffs));
          
          // Instructions can only use one CP entry, don't bother scanning the
          // rest of the operands.
          break;
        }
    }
    BBSizes.push_back(MBBSize);
  }
}

/// GetOffsetOf - Return the current offset of the specified machine instruction
/// from the start of the function.  This offset changes as stuff is moved
/// around inside the function.
unsigned ARMConstantIslands::GetOffsetOf(MachineInstr *MI) const {
  MachineBasicBlock *MBB = MI->getParent();
  
  // The offset is composed of two things: the sum of the sizes of all MBB's
  // before this instruction's block, and the offset from the start of the block
  // it is in.
  unsigned Offset = 0;
  
  // Sum block sizes before MBB.
  for (unsigned BB = 0, e = MBB->getNumber(); BB != e; ++BB)
    Offset += BBSizes[BB];

  // Sum instructions before MI in MBB.
  for (MachineBasicBlock::iterator I = MBB->begin(); ; ++I) {
    assert(I != MBB->end() && "Didn't find MI in its own basic block?");
    if (&*I == MI) return Offset;
    Offset += ARM::GetInstSize(I);
  }
}

/// GetOffsetOf - Return the current offset of the specified machine BB
/// from the start of the function.  This offset changes as stuff is moved
/// around inside the function.
unsigned ARMConstantIslands::GetOffsetOf(MachineBasicBlock *MBB) const {
  // Sum block sizes before MBB.
  unsigned Offset = 0;  
  for (unsigned BB = 0, e = MBB->getNumber(); BB != e; ++BB)
    Offset += BBSizes[BB];

  return Offset;
}

/// CompareMBBNumbers - Little predicate function to sort the WaterList by MBB
/// ID.
static bool CompareMBBNumbers(const MachineBasicBlock *LHS,
                              const MachineBasicBlock *RHS) {
  return LHS->getNumber() < RHS->getNumber();
}

/// UpdateForInsertedWaterBlock - When a block is newly inserted into the
/// machine function, it upsets all of the block numbers.  Renumber the blocks
/// and update the arrays that parallel this numbering.
void ARMConstantIslands::UpdateForInsertedWaterBlock(MachineBasicBlock *NewBB) {
  // Renumber the MBB's to keep them consequtive.
  NewBB->getParent()->RenumberBlocks(NewBB);
  
  // Insert a size into BBSizes to align it properly with the (newly
  // renumbered) block numbers.
  BBSizes.insert(BBSizes.begin()+NewBB->getNumber(), 0);
  
  // Next, update WaterList.  Specifically, we need to add NewMBB as having 
  // available water after it.
  std::vector<MachineBasicBlock*>::iterator IP =
    std::lower_bound(WaterList.begin(), WaterList.end(), NewBB,
                     CompareMBBNumbers);
  WaterList.insert(IP, NewBB);
}


/// Split the basic block containing MI into two blocks, which are joined by
/// an unconditional branch.  Update datastructures and renumber blocks to
/// account for this change and returns the newly created block.
MachineBasicBlock *ARMConstantIslands::SplitBlockBeforeInstr(MachineInstr *MI) {
  MachineBasicBlock *OrigBB = MI->getParent();
  bool isThumb = AFI->isThumbFunction();

  // Create a new MBB for the code after the OrigBB.
  MachineBasicBlock *NewBB = new MachineBasicBlock(OrigBB->getBasicBlock());
  MachineFunction::iterator MBBI = OrigBB; ++MBBI;
  OrigBB->getParent()->getBasicBlockList().insert(MBBI, NewBB);
  
  // Splice the instructions starting with MI over to NewBB.
  NewBB->splice(NewBB->end(), OrigBB, MI, OrigBB->end());
  
  // Add an unconditional branch from OrigBB to NewBB.
  // Note the new unconditional branch is not being recorded.
  BuildMI(OrigBB, TII->get(isThumb ? ARM::tB : ARM::B)).addMBB(NewBB);
  NumSplit++;
  
  // Update the CFG.  All succs of OrigBB are now succs of NewBB.
  while (!OrigBB->succ_empty()) {
    MachineBasicBlock *Succ = *OrigBB->succ_begin();
    OrigBB->removeSuccessor(Succ);
    NewBB->addSuccessor(Succ);
    
    // This pass should be run after register allocation, so there should be no
    // PHI nodes to update.
    assert((Succ->empty() || Succ->begin()->getOpcode() != TargetInstrInfo::PHI)
           && "PHI nodes should be eliminated by now!");
  }
  
  // OrigBB branches to NewBB.
  OrigBB->addSuccessor(NewBB);
  
  // Update internal data structures to account for the newly inserted MBB.
  UpdateForInsertedWaterBlock(NewBB);
  
  // Figure out how large the first NewMBB is.
  unsigned NewBBSize = 0;
  for (MachineBasicBlock::iterator I = NewBB->begin(), E = NewBB->end();
       I != E; ++I)
    NewBBSize += ARM::GetInstSize(I);
  
  // Set the size of NewBB in BBSizes.
  BBSizes[NewBB->getNumber()] = NewBBSize;
  
  // We removed instructions from UserMBB, subtract that off from its size.
  // Add 2 or 4 to the block to count the unconditional branch we added to it.
  BBSizes[OrigBB->getNumber()] -= NewBBSize - (isThumb ? 2 : 4);

  return NewBB;
}

/// CPEIsInRange - Returns true is the distance between specific MI and
/// specific ConstPool entry instruction can fit in MI's displacement field.
bool ARMConstantIslands::CPEIsInRange(MachineInstr *MI, MachineInstr *CPEMI,
                                      unsigned MaxDisp) {
  unsigned PCAdj      = AFI->isThumbFunction() ? 4 : 8;
  unsigned UserOffset = GetOffsetOf(MI) + PCAdj;
  unsigned CPEOffset  = GetOffsetOf(CPEMI);
  
  DEBUG(std::cerr << "User of CPE#" << CPEMI->getOperand(0).getImm()
                  << " max delta=" << MaxDisp
                  << " at offset " << int(UserOffset-CPEOffset) << "\t"
                  << *MI);

  if (UserOffset <= CPEOffset) {
    // User before the CPE.
    if (CPEOffset-UserOffset <= MaxDisp)
      return true;
  } else if (!AFI->isThumbFunction()) {
    // Thumb LDR cannot encode negative offset.
    if (UserOffset-CPEOffset <= MaxDisp)
      return true;
  }
  return false;
}

/// HandleConstantPoolUser - Analyze the specified user, checking to see if it
/// is out-of-range.  If so, pick it up the constant pool value and move it some
/// place in-range.
bool ARMConstantIslands::HandleConstantPoolUser(MachineFunction &Fn, CPUser &U){
  MachineInstr *UserMI = U.MI;
  MachineInstr *CPEMI  = U.CPEMI;

  // Check to see if the CPE is already in-range.
  if (CPEIsInRange(UserMI, CPEMI, U.MaxDisp))
    return false;

  // Solution guaranteed to work: split the user's MBB right after the user and
  // insert a clone the CPE into the newly created water.

  MachineBasicBlock *UserMBB = UserMI->getParent();
  MachineBasicBlock *NewMBB;

  // TODO: Search for the best place to split the code.  In practice, using
  // loop nesting information to insert these guys outside of loops would be
  // sufficient.    
  if (&UserMBB->back() == UserMI) {
    assert(BBHasFallthrough(UserMBB) && "Expected a fallthrough BB!");
    NewMBB = next(MachineFunction::iterator(UserMBB));
    // Add an unconditional branch from UserMBB to fallthrough block.
    // Note the new unconditional branch is not being recorded.
    bool isThumb = AFI->isThumbFunction();
    BuildMI(UserMBB, TII->get(isThumb ? ARM::tB : ARM::B)).addMBB(NewMBB);
    BBSizes[UserMBB->getNumber()] += isThumb ? 2 : 4;
  } else {
    MachineInstr *NextMI = next(MachineBasicBlock::iterator(UserMI));
    NewMBB = SplitBlockBeforeInstr(NextMI);
  }

  // Okay, we know we can put an island before UserMBB now, do it!
  MachineBasicBlock *NewIsland = new MachineBasicBlock();
  Fn.getBasicBlockList().insert(NewMBB, NewIsland);

  // Update internal data structures to account for the newly inserted MBB.
  UpdateForInsertedWaterBlock(NewIsland);

  // Now that we have an island to add the CPE to, clone the original CPE and
  // add it to the island.
  unsigned ID  = NextUID++;
  unsigned CPI = CPEMI->getOperand(1).getConstantPoolIndex();
  unsigned Size = CPEMI->getOperand(2).getImm();
  
  // Build a new CPE for this user.
  U.CPEMI = BuildMI(NewIsland, TII->get(ARM::CONSTPOOL_ENTRY))
                .addImm(ID).addConstantPoolIndex(CPI).addImm(Size);
  
  // Increase the size of the island block to account for the new entry.
  BBSizes[NewIsland->getNumber()] += Size;
  
  // Finally, change the CPI in the instruction operand to be ID.
  for (unsigned i = 0, e = UserMI->getNumOperands(); i != e; ++i)
    if (UserMI->getOperand(i).isConstantPoolIndex()) {
      UserMI->getOperand(i).setConstantPoolIndex(ID);
      break;
    }
      
  DEBUG(std::cerr << "  Moved CPE to #" << ID << " CPI=" << CPI << "\t"
                  << *UserMI);
      
  return true;
}

/// BBIsInRange - Returns true is the distance between specific MI and
/// specific BB can fit in MI's displacement field.
bool ARMConstantIslands::BBIsInRange(MachineInstr *MI,MachineBasicBlock *DestBB,
                                     unsigned MaxDisp) {
  unsigned PCAdj      = AFI->isThumbFunction() ? 4 : 8;
  unsigned BrOffset   = GetOffsetOf(MI) + PCAdj;
  unsigned DestOffset = GetOffsetOf(DestBB);

  DEBUG(std::cerr << "Branch of destination BB#" << DestBB->getNumber()
                  << " max delta=" << MaxDisp
                  << " at offset " << int(BrOffset-DestOffset) << "\t"
                  << *MI);

  if (BrOffset <= DestOffset) {
    if (DestOffset - BrOffset < MaxDisp)
      return true;
  } else {
    if (BrOffset - DestOffset <= MaxDisp)
      return true;
  }
  return false;
}

/// FixUpImmediateBr - Fix up an immediate branch whose destination is too far
/// away to fit in its displacement field.
bool ARMConstantIslands::FixUpImmediateBr(MachineFunction &Fn, ImmBranch &Br) {
  MachineInstr *MI = Br.MI;
  MachineBasicBlock *DestBB = MI->getOperand(0).getMachineBasicBlock();

  // Check to see if the DestBB is already in-range.
  if (BBIsInRange(MI, DestBB, Br.MaxDisp))
    return false;

  if (!Br.isCond)
    return FixUpUnconditionalBr(Fn, Br);
  return FixUpConditionalBr(Fn, Br);
}

/// FixUpUnconditionalBr - Fix up an unconditional branches whose destination is
/// too far away to fit in its displacement field. If LR register has been
/// spilled in the epilogue, then we can use BL to implement a far jump.
/// Otherwise, add a intermediate branch instruction to to a branch.
bool
ARMConstantIslands::FixUpUnconditionalBr(MachineFunction &Fn, ImmBranch &Br) {
  MachineInstr *MI = Br.MI;
  MachineBasicBlock *MBB = MI->getParent();
  assert(AFI->isThumbFunction() && "Expected a Thumb function!");

  // Use BL to implement far jump.
  Br.MaxDisp = (1 << 21) * 2;
  MI->setInstrDescriptor(TII->get(ARM::tBfar));
  BBSizes[MBB->getNumber()] += 2;
  HasFarJump = true;
  NumUBrFixed++;
  return true;
}

/// getUnconditionalBrDisp - Returns the maximum displacement that can fit in the
/// specific unconditional branch instruction.
static inline unsigned getUnconditionalBrDisp(int Opc) {
  return (Opc == ARM::tB) ? (1<<10)*2 : (1<<23)*4;
}

/// FixUpConditionalBr - Fix up a conditional branches whose destination is too
/// far away to fit in its displacement field. It is converted to an inverse
/// conditional branch + an unconditional branch to the destination.
bool
ARMConstantIslands::FixUpConditionalBr(MachineFunction &Fn, ImmBranch &Br) {
  MachineInstr *MI = Br.MI;
  MachineBasicBlock *DestBB = MI->getOperand(0).getMachineBasicBlock();

  // Add a unconditional branch to the destination and invert the branch
  // condition to jump over it:
  // blt L1
  // =>
  // bge L2
  // b   L1
  // L2:
  ARMCC::CondCodes CC = (ARMCC::CondCodes)MI->getOperand(1).getImmedValue();
  CC = ARMCC::getOppositeCondition(CC);

  // If the branch is at the end of its MBB and that has a fall-through block,
  // direct the updated conditional branch to the fall-through block. Otherwise,
  // split the MBB before the next instruction.
  MachineBasicBlock *MBB = MI->getParent();
  MachineInstr *BackMI = &MBB->back();
  bool NeedSplit = (BackMI != MI) || !BBHasFallthrough(MBB);

  NumCBrFixed++;
  if (BackMI != MI) {
    if (next(MachineBasicBlock::iterator(MI)) == MBB->back() &&
        BackMI->getOpcode() == Br.UncondBr) {
      // Last MI in the BB is a unconditional branch. Can we simply invert the
      // condition and swap destinations:
      // beq L1
      // b   L2
      // =>
      // bne L2
      // b   L1
      MachineBasicBlock *NewDest = BackMI->getOperand(0).getMachineBasicBlock();
      if (BBIsInRange(MI, NewDest, Br.MaxDisp)) {
        BackMI->getOperand(0).setMachineBasicBlock(DestBB);
        MI->getOperand(0).setMachineBasicBlock(NewDest);
        MI->getOperand(1).setImm(CC);
        return true;
      }
    }
  }

  if (NeedSplit) {
    SplitBlockBeforeInstr(MI);
    // No need for the branch to the next block. We're adding a unconditional
    // branch to the destination.
    MBB->back().eraseFromParent();
  }
  MachineBasicBlock *NextBB = next(MachineFunction::iterator(MBB));

  // Insert a unconditional branch and replace the conditional branch.
  // Also update the ImmBranch as well as adding a new entry for the new branch.
  BuildMI(MBB, TII->get(MI->getOpcode())).addMBB(NextBB).addImm(CC);
  Br.MI = &MBB->back();
  BuildMI(MBB, TII->get(Br.UncondBr)).addMBB(DestBB);
  unsigned MaxDisp = getUnconditionalBrDisp(Br.UncondBr);
  ImmBranches.push_back(ImmBranch(&MBB->back(), MaxDisp, false, Br.UncondBr));
  MI->eraseFromParent();

  // Increase the size of MBB to account for the new unconditional branch.
  BBSizes[MBB->getNumber()] += ARM::GetInstSize(&MBB->back());
  return true;
}


/// UndoLRSpillRestore - Remove Thumb push / pop instructions that only spills
/// LR / restores LR to pc.
bool ARMConstantIslands::UndoLRSpillRestore() {
  bool MadeChange = false;
  for (unsigned i = 0, e = PushPopMIs.size(); i != e; ++i) {
    MachineInstr *MI = PushPopMIs[i];
    if (MI->getNumOperands() == 1) {
        if (MI->getOpcode() == ARM::tPOP_RET &&
            MI->getOperand(0).getReg() == ARM::PC)
          BuildMI(MI->getParent(), TII->get(ARM::tBX_RET));
        MI->eraseFromParent();
        MadeChange = true;
    }
  }
  return MadeChange;
}
