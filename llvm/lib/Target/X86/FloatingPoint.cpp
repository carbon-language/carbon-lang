//===-- FloatingPoint.cpp - Floating point Reg -> Stack converter ---------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the pass which converts floating point instructions from
// virtual registers into register stack instructions.  This pass uses live
// variable information to indicate where the FPn registers are used and their
// lifetimes.
//
// This pass is hampered by the lack of decent CFG manipulation routines for
// machine code.  In particular, this wants to be able to split critical edges
// as necessary, traverse the machine basic block CFG in depth-first order, and
// allow there to be multiple machine basic blocks for each LLVM basicblock
// (needed for critical edge splitting).
//
// In particular, this pass currently barfs on critical edges.  Because of this,
// it requires the instruction selector to insert FP_REG_KILL instructions on
// the exits of any basic block that has critical edges going from it, or which
// branch to a critical basic block.
//
// FIXME: this is not implemented yet.  The stackifier pass only works on local
// basic blocks.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "fp"
#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Function.h"     // FIXME: remove when using MBB CFG!
#include "llvm/Support/CFG.h"  // FIXME: remove when using MBB CFG!
#include "Support/Debug.h"
#include "Support/DepthFirstIterator.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include <algorithm>
#include <set>
using namespace llvm;

namespace {
  Statistic<> NumFXCH("x86-codegen", "Number of fxch instructions inserted");
  Statistic<> NumFP  ("x86-codegen", "Number of floating point instructions");

  struct FPS : public MachineFunctionPass {
    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const { return "X86 FP Stackifier"; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LiveVariables>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  private:
    LiveVariables     *LV;    // Live variable info for current function...
    MachineBasicBlock *MBB;   // Current basic block
    unsigned Stack[8];        // FP<n> Registers in each stack slot...
    unsigned RegMap[8];       // Track which stack slot contains each register
    unsigned StackTop;        // The current top of the FP stack.

    void dumpStack() const {
      std::cerr << "Stack contents:";
      for (unsigned i = 0; i != StackTop; ++i) {
	std::cerr << " FP" << Stack[i];
	assert(RegMap[Stack[i]] == i && "Stack[] doesn't match RegMap[]!"); 
      }
      std::cerr << "\n";
    }
  private:
    // getSlot - Return the stack slot number a particular register number is
    // in...
    unsigned getSlot(unsigned RegNo) const {
      assert(RegNo < 8 && "Regno out of range!");
      return RegMap[RegNo];
    }

    // getStackEntry - Return the X86::FP<n> register in register ST(i)
    unsigned getStackEntry(unsigned STi) const {
      assert(STi < StackTop && "Access past stack top!");
      return Stack[StackTop-1-STi];
    }

    // getSTReg - Return the X86::ST(i) register which contains the specified
    // FP<RegNo> register
    unsigned getSTReg(unsigned RegNo) const {
      return StackTop - 1 - getSlot(RegNo) + llvm::X86::ST0;
    }

    // pushReg - Push the specified FP<n> register onto the stack
    void pushReg(unsigned Reg) {
      assert(Reg < 8 && "Register number out of range!");
      assert(StackTop < 8 && "Stack overflow!");
      Stack[StackTop] = Reg;
      RegMap[Reg] = StackTop++;
    }

    bool isAtTop(unsigned RegNo) const { return getSlot(RegNo) == StackTop-1; }
    void moveToTop(unsigned RegNo, MachineBasicBlock::iterator &I) {
      if (!isAtTop(RegNo)) {
	unsigned Slot = getSlot(RegNo);
	unsigned STReg = getSTReg(RegNo);
	unsigned RegOnTop = getStackEntry(0);

	// Swap the slots the regs are in
	std::swap(RegMap[RegNo], RegMap[RegOnTop]);

	// Swap stack slot contents
	assert(RegMap[RegOnTop] < StackTop);
	std::swap(Stack[RegMap[RegOnTop]], Stack[StackTop-1]);

	// Emit an fxch to update the runtime processors version of the state
	MachineInstr *MI = BuildMI(X86::FXCH, 1).addReg(STReg);
	MBB->insert(I, MI);
	NumFXCH++;
      }
    }

    void duplicateToTop(unsigned RegNo, unsigned AsReg,
			MachineBasicBlock::iterator &I) {
      unsigned STReg = getSTReg(RegNo);
      pushReg(AsReg);   // New register on top of stack

      MachineInstr *MI = BuildMI(X86::FLDrr, 1).addReg(STReg);
      MBB->insert(I, MI);
    }

    // popStackAfter - Pop the current value off of the top of the FP stack
    // after the specified instruction.
    void popStackAfter(MachineBasicBlock::iterator &I);

    bool processBasicBlock(MachineFunction &MF, MachineBasicBlock &MBB);

    void handleZeroArgFP(MachineBasicBlock::iterator &I);
    void handleOneArgFP(MachineBasicBlock::iterator &I);
    void handleOneArgFPRW(MachineBasicBlock::iterator &I);
    void handleTwoArgFP(MachineBasicBlock::iterator &I);
    void handleSpecialFP(MachineBasicBlock::iterator &I);
  };
}

FunctionPass *llvm::createX86FloatingPointStackifierPass() { return new FPS(); }

/// runOnMachineFunction - Loop over all of the basic blocks, transforming FP
/// register references into FP stack references.
///
bool FPS::runOnMachineFunction(MachineFunction &MF) {
  LV = &getAnalysis<LiveVariables>();
  StackTop = 0;

  // Figure out the mapping of MBB's to BB's.
  //
  // FIXME: Eventually we should be able to traverse the MBB CFG directly, and
  // we will need to extend this when one llvm basic block can codegen to
  // multiple MBBs.
  //
  // FIXME again: Just use the mapping established by LiveVariables!
  //
  std::map<const BasicBlock*, MachineBasicBlock *> MBBMap;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    MBBMap[I->getBasicBlock()] = I;

  // Process the function in depth first order so that we process at least one
  // of the predecessors for every reachable block in the function.
  std::set<const BasicBlock*> Processed;
  const BasicBlock *Entry = MF.getFunction()->begin();

  bool Changed = false;
  for (df_ext_iterator<const BasicBlock*, std::set<const BasicBlock*> >
         I = df_ext_begin(Entry, Processed), E = df_ext_end(Entry, Processed);
       I != E; ++I)
    Changed |= processBasicBlock(MF, *MBBMap[*I]);

  assert(MBBMap.size() == Processed.size() &&
         "Doesn't handle unreachable code yet!");

  return Changed;
}

/// processBasicBlock - Loop over all of the instructions in the basic block,
/// transforming FP instructions into their stack form.
///
bool FPS::processBasicBlock(MachineFunction &MF, MachineBasicBlock &BB) {
  const TargetInstrInfo &TII = MF.getTarget().getInstrInfo();
  bool Changed = false;
  MBB = &BB;
  
  for (MachineBasicBlock::iterator I = BB.begin(); I != BB.end(); ++I) {
    MachineInstr *MI = I;
    unsigned Flags = TII.get(MI->getOpcode()).TSFlags;
    if ((Flags & X86II::FPTypeMask) == X86II::NotFP)
      continue;  // Efficiently ignore non-fp insts!

    MachineInstr *PrevMI = 0;
    if (I != BB.begin())
        PrevMI = prior(I);

    ++NumFP;  // Keep track of # of pseudo instrs
    DEBUG(std::cerr << "\nFPInst:\t";
	  MI->print(std::cerr, MF.getTarget()));

    // Get dead variables list now because the MI pointer may be deleted as part
    // of processing!
    LiveVariables::killed_iterator IB = LV->dead_begin(MI);
    LiveVariables::killed_iterator IE = LV->dead_end(MI);

    DEBUG(const MRegisterInfo *MRI = MF.getTarget().getRegisterInfo();
	  LiveVariables::killed_iterator I = LV->killed_begin(MI);
	  LiveVariables::killed_iterator E = LV->killed_end(MI);
	  if (I != E) {
	    std::cerr << "Killed Operands:";
	    for (; I != E; ++I)
	      std::cerr << " %" << MRI->getName(I->second);
	    std::cerr << "\n";
	  });

    switch (Flags & X86II::FPTypeMask) {
    case X86II::ZeroArgFP:  handleZeroArgFP(I); break;
    case X86II::OneArgFP:   handleOneArgFP(I);  break;   // fstp ST(0)
    case X86II::OneArgFPRW: handleOneArgFPRW(I); break; // ST(0) = fsqrt(ST(0))
    case X86II::TwoArgFP:   handleTwoArgFP(I);  break;
    case X86II::SpecialFP:  handleSpecialFP(I); break;
    default: assert(0 && "Unknown FP Type!");
    }

    // Check to see if any of the values defined by this instruction are dead
    // after definition.  If so, pop them.
    for (; IB != IE; ++IB) {
      unsigned Reg = IB->second;
      if (Reg >= X86::FP0 && Reg <= X86::FP6) {
	DEBUG(std::cerr << "Register FP#" << Reg-X86::FP0 << " is dead!\n");
	++I;                         // Insert fxch AFTER the instruction
	moveToTop(Reg-X86::FP0, I);  // Insert fxch if necessary
	--I;                         // Move to fxch or old instruction
	popStackAfter(I);            // Pop the top of the stack, killing value
      }
    }
    
    // Print out all of the instructions expanded to if -debug
    DEBUG(
      MachineBasicBlock::iterator PrevI(PrevMI);
      if (I == PrevI) {
        std::cerr<< "Just deleted pseudo instruction\n";
      } else {
        MachineBasicBlock::iterator Start = I;
        // Rewind to first instruction newly inserted.
        while (Start != BB.begin() && prior(Start) != PrevI) --Start;
        std::cerr << "Inserted instructions:\n\t";
        Start->print(std::cerr, MF.getTarget());
        while (++Start != next(I));
      }
      dumpStack();
    );

    Changed = true;
  }

  assert(StackTop == 0 && "Stack not empty at end of basic block?");
  return Changed;
}

//===----------------------------------------------------------------------===//
// Efficient Lookup Table Support
//===----------------------------------------------------------------------===//

namespace {
  struct TableEntry {
    unsigned from;
    unsigned to;
    bool operator<(const TableEntry &TE) const { return from < TE.from; }
    bool operator<(unsigned V) const { return from < V; }
  };
}

static bool TableIsSorted(const TableEntry *Table, unsigned NumEntries) {
  for (unsigned i = 0; i != NumEntries-1; ++i)
    if (!(Table[i] < Table[i+1])) return false;
  return true;
}

static int Lookup(const TableEntry *Table, unsigned N, unsigned Opcode) {
  const TableEntry *I = std::lower_bound(Table, Table+N, Opcode);
  if (I != Table+N && I->from == Opcode)
    return I->to;
  return -1;
}

#define ARRAY_SIZE(TABLE)  \
   (sizeof(TABLE)/sizeof(TABLE[0]))

#ifdef NDEBUG
#define ASSERT_SORTED(TABLE)
#else
#define ASSERT_SORTED(TABLE)                                              \
  { static bool TABLE##Checked = false;                                   \
    if (!TABLE##Checked)                                                  \
       assert(TableIsSorted(TABLE, ARRAY_SIZE(TABLE)) &&                  \
              "All lookup tables must be sorted for efficient access!");  \
  }
#endif


//===----------------------------------------------------------------------===//
// Helper Methods
//===----------------------------------------------------------------------===//

// PopTable - Sorted map of instructions to their popping version.  The first
// element is an instruction, the second is the version which pops.
//
static const TableEntry PopTable[] = {
  { X86::FADDrST0 , X86::FADDPrST0  },

  { X86::FDIVRrST0, X86::FDIVRPrST0 },
  { X86::FDIVrST0 , X86::FDIVPrST0  },

  { X86::FISTm16  , X86::FISTPm16   },
  { X86::FISTm32  , X86::FISTPm32   },

  { X86::FMULrST0 , X86::FMULPrST0  },

  { X86::FSTm32   , X86::FSTPm32    },
  { X86::FSTm64   , X86::FSTPm64    },
  { X86::FSTrr    , X86::FSTPrr     },

  { X86::FSUBRrST0, X86::FSUBRPrST0 },
  { X86::FSUBrST0 , X86::FSUBPrST0  },

  { X86::FUCOMPr  , X86::FUCOMPPr   },
  { X86::FUCOMr   , X86::FUCOMPr    },
};

/// popStackAfter - Pop the current value off of the top of the FP stack after
/// the specified instruction.  This attempts to be sneaky and combine the pop
/// into the instruction itself if possible.  The iterator is left pointing to
/// the last instruction, be it a new pop instruction inserted, or the old
/// instruction if it was modified in place.
///
void FPS::popStackAfter(MachineBasicBlock::iterator &I) {
  ASSERT_SORTED(PopTable);
  assert(StackTop > 0 && "Cannot pop empty stack!");
  RegMap[Stack[--StackTop]] = ~0;     // Update state

  // Check to see if there is a popping version of this instruction...
  int Opcode = Lookup(PopTable, ARRAY_SIZE(PopTable), I->getOpcode());
  if (Opcode != -1) {
    I->setOpcode(Opcode);
    if (Opcode == X86::FUCOMPPr)
      I->RemoveOperand(0);

  } else {    // Insert an explicit pop
    MachineInstr *MI = BuildMI(X86::FSTPrr, 1).addReg(X86::ST0);
    I = MBB->insert(++I, MI);
  }
}

static unsigned getFPReg(const MachineOperand &MO) {
  assert(MO.isRegister() && "Expected an FP register!");
  unsigned Reg = MO.getReg();
  assert(Reg >= X86::FP0 && Reg <= X86::FP6 && "Expected FP register!");
  return Reg - X86::FP0;
}


//===----------------------------------------------------------------------===//
// Instruction transformation implementation
//===----------------------------------------------------------------------===//

/// handleZeroArgFP - ST(0) = fld0    ST(0) = flds <mem>
///
void FPS::handleZeroArgFP(MachineBasicBlock::iterator &I) {
  MachineInstr *MI = I;
  unsigned DestReg = getFPReg(MI->getOperand(0));
  MI->RemoveOperand(0);   // Remove the explicit ST(0) operand

  // Result gets pushed on the stack...
  pushReg(DestReg);
}

/// handleOneArgFP - fst <mem>, ST(0)
///
void FPS::handleOneArgFP(MachineBasicBlock::iterator &I) {
  MachineInstr *MI = I;
  assert((MI->getNumOperands() == 5 || MI->getNumOperands() == 1) &&
         "Can only handle fst* & ftst instructions!");

  // Is this the last use of the source register?
  unsigned Reg = getFPReg(MI->getOperand(MI->getNumOperands()-1));
  bool KillsSrc = false;
  for (LiveVariables::killed_iterator KI = LV->killed_begin(MI),
	 E = LV->killed_end(MI); KI != E; ++KI)
    KillsSrc |= KI->second == X86::FP0+Reg;

  // FSTPr80 and FISTPr64 are strange because there are no non-popping versions.
  // If we have one _and_ we don't want to pop the operand, duplicate the value
  // on the stack instead of moving it.  This ensure that popping the value is
  // always ok.
  //
  if ((MI->getOpcode() == X86::FSTPm80 ||
       MI->getOpcode() == X86::FISTPm64) && !KillsSrc) {
    duplicateToTop(Reg, 7 /*temp register*/, I);
  } else {
    moveToTop(Reg, I);            // Move to the top of the stack...
  }
  MI->RemoveOperand(MI->getNumOperands()-1);    // Remove explicit ST(0) operand
  
  if (MI->getOpcode() == X86::FSTPm80 || MI->getOpcode() == X86::FISTPm64) {
    assert(StackTop > 0 && "Stack empty??");
    --StackTop;
  } else if (KillsSrc) { // Last use of operand?
    popStackAfter(I);
  }
}


/// handleOneArgFPRW - fchs - ST(0) = -ST(0)
///
void FPS::handleOneArgFPRW(MachineBasicBlock::iterator &I) {
  MachineInstr *MI = I;
  assert(MI->getNumOperands() == 2 && "Can only handle fst* instructions!");

  // Is this the last use of the source register?
  unsigned Reg = getFPReg(MI->getOperand(1));
  bool KillsSrc = false;
  for (LiveVariables::killed_iterator KI = LV->killed_begin(MI),
	 E = LV->killed_end(MI); KI != E; ++KI)
    KillsSrc |= KI->second == X86::FP0+Reg;

  if (KillsSrc) {
    // If this is the last use of the source register, just make sure it's on
    // the top of the stack.
    moveToTop(Reg, I);
    assert(StackTop > 0 && "Stack cannot be empty!");
    --StackTop;
    pushReg(getFPReg(MI->getOperand(0)));
  } else {
    // If this is not the last use of the source register, _copy_ it to the top
    // of the stack.
    duplicateToTop(Reg, getFPReg(MI->getOperand(0)), I);
  }

  MI->RemoveOperand(1);   // Drop the source operand.
  MI->RemoveOperand(0);   // Drop the destination operand.
}


//===----------------------------------------------------------------------===//
// Define tables of various ways to map pseudo instructions
//

// ForwardST0Table - Map: A = B op C  into: ST(0) = ST(0) op ST(i)
static const TableEntry ForwardST0Table[] = {
  { X86::FpADD,  X86::FADDST0r  },
  { X86::FpDIV,  X86::FDIVST0r  },
  { X86::FpMUL,  X86::FMULST0r  },
  { X86::FpSUB,  X86::FSUBST0r  },
  { X86::FpUCOM, X86::FUCOMr    },
};

// ReverseST0Table - Map: A = B op C  into: ST(0) = ST(i) op ST(0)
static const TableEntry ReverseST0Table[] = {
  { X86::FpADD,  X86::FADDST0r  },   // commutative
  { X86::FpDIV,  X86::FDIVRST0r },
  { X86::FpMUL,  X86::FMULST0r  },   // commutative
  { X86::FpSUB,  X86::FSUBRST0r },
  { X86::FpUCOM, ~0             },
};

// ForwardSTiTable - Map: A = B op C  into: ST(i) = ST(0) op ST(i)
static const TableEntry ForwardSTiTable[] = {
  { X86::FpADD,  X86::FADDrST0  },   // commutative
  { X86::FpDIV,  X86::FDIVRrST0 },
  { X86::FpMUL,  X86::FMULrST0  },   // commutative
  { X86::FpSUB,  X86::FSUBRrST0 },
  { X86::FpUCOM, X86::FUCOMr    },
};

// ReverseSTiTable - Map: A = B op C  into: ST(i) = ST(i) op ST(0)
static const TableEntry ReverseSTiTable[] = {
  { X86::FpADD,  X86::FADDrST0 },
  { X86::FpDIV,  X86::FDIVrST0 },
  { X86::FpMUL,  X86::FMULrST0 },
  { X86::FpSUB,  X86::FSUBrST0 },
  { X86::FpUCOM, ~0            },
};


/// handleTwoArgFP - Handle instructions like FADD and friends which are virtual
/// instructions which need to be simplified and possibly transformed.
///
/// Result: ST(0) = fsub  ST(0), ST(i)
///         ST(i) = fsub  ST(0), ST(i)
///         ST(0) = fsubr ST(0), ST(i)
///         ST(i) = fsubr ST(0), ST(i)
///
/// In addition to three address instructions, this also handles the FpUCOM
/// instruction which only has two operands, but no destination.  This
/// instruction is also annoying because there is no "reverse" form of it
/// available.
/// 
void FPS::handleTwoArgFP(MachineBasicBlock::iterator &I) {
  ASSERT_SORTED(ForwardST0Table); ASSERT_SORTED(ReverseST0Table);
  ASSERT_SORTED(ForwardSTiTable); ASSERT_SORTED(ReverseSTiTable);
  MachineInstr *MI = I;

  unsigned NumOperands = MI->getNumOperands();
  assert(NumOperands == 3 ||
	 (NumOperands == 2 && MI->getOpcode() == X86::FpUCOM) &&
	 "Illegal TwoArgFP instruction!");
  unsigned Dest = getFPReg(MI->getOperand(0));
  unsigned Op0 = getFPReg(MI->getOperand(NumOperands-2));
  unsigned Op1 = getFPReg(MI->getOperand(NumOperands-1));
  bool KillsOp0 = false, KillsOp1 = false;

  for (LiveVariables::killed_iterator KI = LV->killed_begin(MI),
	 E = LV->killed_end(MI); KI != E; ++KI) {
    KillsOp0 |= (KI->second == X86::FP0+Op0);
    KillsOp1 |= (KI->second == X86::FP0+Op1);
  }

  // If this is an FpUCOM instruction, we must make sure the first operand is on
  // the top of stack, the other one can be anywhere...
  if (MI->getOpcode() == X86::FpUCOM)
    moveToTop(Op0, I);

  unsigned TOS = getStackEntry(0);

  // One of our operands must be on the top of the stack.  If neither is yet, we
  // need to move one.
  if (Op0 != TOS && Op1 != TOS) {   // No operand at TOS?
    // We can choose to move either operand to the top of the stack.  If one of
    // the operands is killed by this instruction, we want that one so that we
    // can update right on top of the old version.
    if (KillsOp0) {
      moveToTop(Op0, I);         // Move dead operand to TOS.
      TOS = Op0;
    } else if (KillsOp1) {
      moveToTop(Op1, I);
      TOS = Op1;
    } else {
      // All of the operands are live after this instruction executes, so we
      // cannot update on top of any operand.  Because of this, we must
      // duplicate one of the stack elements to the top.  It doesn't matter
      // which one we pick.
      //
      duplicateToTop(Op0, Dest, I);
      Op0 = TOS = Dest;
      KillsOp0 = true;
    }
  } else if (!KillsOp0 && !KillsOp1 && MI->getOpcode() != X86::FpUCOM)  {
    // If we DO have one of our operands at the top of the stack, but we don't
    // have a dead operand, we must duplicate one of the operands to a new slot
    // on the stack.
    duplicateToTop(Op0, Dest, I);
    Op0 = TOS = Dest;
    KillsOp0 = true;
  }

  // Now we know that one of our operands is on the top of the stack, and at
  // least one of our operands is killed by this instruction.
  assert((TOS == Op0 || TOS == Op1) &&
	 (KillsOp0 || KillsOp1 || MI->getOpcode() == X86::FpUCOM) &&
	 "Stack conditions not set up right!");

  // We decide which form to use based on what is on the top of the stack, and
  // which operand is killed by this instruction.
  const TableEntry *InstTable;
  bool isForward = TOS == Op0;
  bool updateST0 = (TOS == Op0 && !KillsOp1) || (TOS == Op1 && !KillsOp0);
  if (updateST0) {
    if (isForward)
      InstTable = ForwardST0Table;
    else
      InstTable = ReverseST0Table;
  } else {
    if (isForward)
      InstTable = ForwardSTiTable;
    else
      InstTable = ReverseSTiTable;
  }
  
  int Opcode = Lookup(InstTable, ARRAY_SIZE(ForwardST0Table), MI->getOpcode());
  assert(Opcode != -1 && "Unknown TwoArgFP pseudo instruction!");

  // NotTOS - The register which is not on the top of stack...
  unsigned NotTOS = (TOS == Op0) ? Op1 : Op0;

  // Replace the old instruction with a new instruction
  MBB->remove(I);
  I = MBB->insert(I, BuildMI(Opcode, 1).addReg(getSTReg(NotTOS)));

  // If both operands are killed, pop one off of the stack in addition to
  // overwriting the other one.
  if (KillsOp0 && KillsOp1 && Op0 != Op1) {
    assert(!updateST0 && "Should have updated other operand!");
    popStackAfter(I);   // Pop the top of stack
  }

  // Insert an explicit pop of the "updated" operand for FUCOM 
  if (MI->getOpcode() == X86::FpUCOM) {
    if (KillsOp0 && !KillsOp1)
      popStackAfter(I);   // If we kill the first operand, pop it!
    else if (KillsOp1 && Op0 != Op1) {
      if (getStackEntry(0) == Op1) {
	popStackAfter(I);     // If it's right at the top of stack, just pop it
      } else {
	// Otherwise, move the top of stack into the dead slot, killing the
	// operand without having to add in an explicit xchg then pop.
	//
	unsigned STReg    = getSTReg(Op1);
	unsigned OldSlot  = getSlot(Op1);
	unsigned TopReg   = Stack[StackTop-1];
	Stack[OldSlot]    = TopReg;
	RegMap[TopReg]    = OldSlot;
	RegMap[Op1]       = ~0;
	Stack[--StackTop] = ~0;
	
	MachineInstr *MI = BuildMI(X86::FSTPrr, 1).addReg(STReg);
	I = MBB->insert(++I, MI);
      }
    }
  }
      
  // Update stack information so that we know the destination register is now on
  // the stack.
  if (MI->getOpcode() != X86::FpUCOM) {  
    unsigned UpdatedSlot = getSlot(updateST0 ? TOS : NotTOS);
    assert(UpdatedSlot < StackTop && Dest < 7);
    Stack[UpdatedSlot]   = Dest;
    RegMap[Dest]         = UpdatedSlot;
  }
  delete MI;   // Remove the old instruction
}


/// handleSpecialFP - Handle special instructions which behave unlike other
/// floating point instructions.  This is primarily intended for use by pseudo
/// instructions.
///
void FPS::handleSpecialFP(MachineBasicBlock::iterator &I) {
  MachineInstr *MI = I;
  switch (MI->getOpcode()) {
  default: assert(0 && "Unknown SpecialFP instruction!");
  case X86::FpGETRESULT:  // Appears immediately after a call returning FP type!
    assert(StackTop == 0 && "Stack should be empty after a call!");
    pushReg(getFPReg(MI->getOperand(0)));
    break;
  case X86::FpSETRESULT:
    assert(StackTop == 1 && "Stack should have one element on it to return!");
    --StackTop;   // "Forget" we have something on the top of stack!
    break;
  case X86::FpMOV: {
    unsigned SrcReg = getFPReg(MI->getOperand(1));
    unsigned DestReg = getFPReg(MI->getOperand(0));
    bool KillsSrc = false;
    for (LiveVariables::killed_iterator KI = LV->killed_begin(MI),
	   E = LV->killed_end(MI); KI != E; ++KI)
      KillsSrc |= KI->second == X86::FP0+SrcReg;

    if (KillsSrc) {
      // If the input operand is killed, we can just change the owner of the
      // incoming stack slot into the result.
      unsigned Slot = getSlot(SrcReg);
      assert(Slot < 7 && DestReg < 7 && "FpMOV operands invalid!");
      Stack[Slot] = DestReg;
      RegMap[DestReg] = Slot;

    } else {
      // For FMOV we just duplicate the specified value to a new stack slot.
      // This could be made better, but would require substantial changes.
      duplicateToTop(SrcReg, DestReg, I);
    }
    break;
  }
  }

  I = MBB->erase(I);  // Remove the pseudo instruction
  --I;
}
