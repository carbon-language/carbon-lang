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
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include <iostream>
using namespace llvm;

STATISTIC(NumSplit, "Number of uncond branches inserted");

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

    const TargetInstrInfo *TII;
    const TargetAsmInfo   *TAI;
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
    void SplitBlockBeforeInstr(MachineInstr *MI);
    void UpdateForInsertedWaterBlock(MachineBasicBlock *NewBB);
    bool HandleConstantPoolUser(MachineFunction &Fn, CPUser &U);
    bool FixUpImmediateBranch(MachineFunction &Fn, ImmBranch &Br);

    unsigned GetInstSize(MachineInstr *MI) const;
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
  TAI = Fn.getTarget().getTargetAsmInfo();
  
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
  
  // Iteratively place constant pool entries until there is no change.
  bool MadeChange;
  do {
    MadeChange = false;
    for (unsigned i = 0, e = CPUsers.size(); i != e; ++i)
      MadeChange |= HandleConstantPoolUser(Fn, CPUsers[i]);
    for (unsigned i = 0, e = ImmBranches.size(); i != e; ++i)
      MadeChange |= FixUpImmediateBranch(Fn, ImmBranches[i]);
  } while (MadeChange);
  
  BBSizes.clear();
  WaterList.clear();
  CPUsers.clear();
  ImmBranches.clear();
    
  return true;
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
      MBBSize += GetInstSize(I);

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

/// FIXME: Works around a gcc miscompilation with -fstrict-aliasing
static unsigned getNumJTEntries(const std::vector<MachineJumpTableEntry> &JT,
                                unsigned JTI) DISABLE_INLINE;
static unsigned getNumJTEntries(const std::vector<MachineJumpTableEntry> &JT,
                                unsigned JTI) {
  return JT[JTI].MBBs.size();
}

/// GetInstSize - Return the size of the specified MachineInstr.
///
unsigned ARMConstantIslands::GetInstSize(MachineInstr *MI) const {
  // Basic size info comes from the TSFlags field.
  unsigned TSFlags = MI->getInstrDescriptor()->TSFlags;
  
  switch ((TSFlags & ARMII::SizeMask) >> ARMII::SizeShift) {
  default:
    // If this machine instr is an inline asm, measure it.
    if (MI->getOpcode() == ARM::INLINEASM)
      return TAI->getInlineAsmLength(MI->getOperand(0).getSymbolName());
    if (MI->getOpcode() == ARM::LABEL)
      return 0;
    assert(0 && "Unknown or unset size field for instr!");
    break;
  case ARMII::Size8Bytes: return 8;          // Arm instruction x 2.
  case ARMII::Size4Bytes: return 4;          // Arm instruction.
  case ARMII::Size2Bytes: return 2;          // Thumb instruction.
  case ARMII::SizeSpecial: {
    switch (MI->getOpcode()) {
    case ARM::CONSTPOOL_ENTRY:
      // If this machine instr is a constant pool entry, its size is recorded as
      // operand #2.
      return MI->getOperand(2).getImm();
    case ARM::BR_JTr:
    case ARM::BR_JTm:
    case ARM::BR_JTadd: {
      // These are jumptable branches, i.e. a branch followed by an inlined
      // jumptable. The size is 4 + 4 * number of entries.
      unsigned JTI = MI->getOperand(MI->getNumOperands()-2).getJumpTableIndex();
      const MachineFunction *MF = MI->getParent()->getParent();
      MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
      const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
      assert(JTI < JT.size());
      return getNumJTEntries(JT, JTI) * 4 + 4;
    }
    default:
      // Otherwise, pseudo-instruction sizes are zero.
      return 0;
    }
  }
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
    Offset += GetInstSize(I);
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
/// account for this change.
void ARMConstantIslands::SplitBlockBeforeInstr(MachineInstr *MI) {
  MachineBasicBlock *OrigBB = MI->getParent();
  const ARMFunctionInfo *AFI = OrigBB->getParent()->getInfo<ARMFunctionInfo>();
  bool isThumb = AFI->isThumbFunction();

  // Create a new MBB for the code after the OrigBB.
  MachineBasicBlock *NewBB = new MachineBasicBlock(OrigBB->getBasicBlock());
  MachineFunction::iterator MBBI = OrigBB; ++MBBI;
  OrigBB->getParent()->getBasicBlockList().insert(MBBI, NewBB);
  
  // Splice the instructions starting with MI over to NewBB.
  NewBB->splice(NewBB->end(), OrigBB, MI, OrigBB->end());
  
  // Add an unconditional branch from OrigBB to NewBB.
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
    NewBBSize += GetInstSize(I);
  
  // Set the size of NewBB in BBSizes.
  BBSizes[NewBB->getNumber()] = NewBBSize;
  
  // We removed instructions from UserMBB, subtract that off from its size.
  // Add 2 or 4 to the block to count the unconditional branch we added to it.
  BBSizes[OrigBB->getNumber()] -= NewBBSize - (isThumb ? 2 : 4);
}

/// HandleConstantPoolUser - Analyze the specified user, checking to see if it
/// is out-of-range.  If so, pick it up the constant pool value and move it some
/// place in-range.
bool ARMConstantIslands::HandleConstantPoolUser(MachineFunction &Fn, CPUser &U){
  MachineInstr *UserMI = U.MI;
  MachineInstr *CPEMI  = U.CPEMI;

  unsigned UserOffset = GetOffsetOf(UserMI);
  unsigned CPEOffset  = GetOffsetOf(CPEMI);
  
  DEBUG(std::cerr << "User of CPE#" << CPEMI->getOperand(0).getImm()
                  << " max delta=" << U.MaxDisp
                  << " at offset " << int(UserOffset-CPEOffset) << "\t"
                  << *UserMI);

  // Check to see if the CPE is already in-range.
  if (UserOffset < CPEOffset) {
    // User before the CPE.
    if (CPEOffset-UserOffset <= U.MaxDisp)
      return false;
  } else {
    if (UserOffset-CPEOffset <= U.MaxDisp)
      return false;
  }
  
 
  // Solution guaranteed to work: split the user's MBB right before the user and
  // insert a clone the CPE into the newly created water.
  
  // If the user isn't at the start of its MBB, or if there is a fall-through
  // into the user's MBB, split the MBB before the User.
  MachineBasicBlock *UserMBB = UserMI->getParent();
  if (&UserMBB->front() != UserMI ||
      UserMBB == &Fn.front() || // entry MBB of function.
      BBHasFallthrough(prior(MachineFunction::iterator(UserMBB)))) {
    // TODO: Search for the best place to split the code.  In practice, using
    // loop nesting information to insert these guys outside of loops would be
    // sufficient.    
    SplitBlockBeforeInstr(UserMI);
    
    // UserMI's BB may have changed.
    UserMBB = UserMI->getParent();
  }
  
  // Okay, we know we can put an island before UserMBB now, do it!
  MachineBasicBlock *NewIsland = new MachineBasicBlock();
  Fn.getBasicBlockList().insert(UserMBB, NewIsland);

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

/// FixUpImmediateBranch - Fix up immediate branches whose destination is too
/// far away to fit in its displacement field. If it is a conditional branch,
/// then it is converted to an inverse conditional branch + an unconditional
/// branch to the destination. If it is an unconditional branch, then it is
/// converted to a branch to a branch.
bool
ARMConstantIslands::FixUpImmediateBranch(MachineFunction &Fn, ImmBranch &Br) {
  MachineInstr *MI = Br.MI;
  MachineBasicBlock *DestBB = MI->getOperand(0).getMachineBasicBlock();

  unsigned BrOffset   = GetOffsetOf(MI);
  unsigned DestOffset = GetOffsetOf(DestBB);

  // Check to see if the destination BB is in range.
  if (BrOffset < DestOffset) {
    if (DestOffset - BrOffset < Br.MaxDisp)
      return false;
  } else {
    if (BrOffset - DestOffset <= Br.MaxDisp)
      return false;
  }

  if (!Br.isCond) {
    // Unconditional branch. We have to insert a branch somewhere to perform
    // a two level branch (branch to branch). FIXME: not yet implemented.
    assert(false && "Can't handle unconditional branch yet!");
    return false;
  }

  // Otherwise, add a unconditional branch to the destination and 
  // invert the branch condition to jump over it:
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
  if (&MBB->back() != MI || !BBHasFallthrough(MBB)) {
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
  unsigned MaxDisp = (Br.UncondBr == ARM::tB) ? (1<<10)*2 : (1<<23)*4;
  ImmBranches.push_back(ImmBranch(&MBB->back(), MaxDisp, false, Br.UncondBr));
  MI->eraseFromParent();

  // Increase the size of MBB to account for the new unconditional branch.
  BBSizes[MBB->getNumber()] += GetInstSize(&MBB->back());
  return true;
}
