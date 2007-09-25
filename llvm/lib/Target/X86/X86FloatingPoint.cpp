//===-- X86FloatingPoint.cpp - Floating point Reg -> Stack converter ------===//
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

#define DEBUG_TYPE "x86-codegen"
#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <set>
using namespace llvm;

STATISTIC(NumFXCH, "Number of fxch instructions inserted");
STATISTIC(NumFP  , "Number of floating point instructions");

namespace {
  struct VISIBILITY_HIDDEN FPS : public MachineFunctionPass {
    static char ID;
    FPS() : MachineFunctionPass((intptr_t)&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const { return "X86 FP Stackifier"; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LiveVariables>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  private:
    const TargetInstrInfo *TII; // Machine instruction info.
    LiveVariables     *LV;      // Live variable info for current function...
    MachineBasicBlock *MBB;     // Current basic block
    unsigned Stack[8];          // FP<n> Registers in each stack slot...
    unsigned RegMap[8];         // Track which stack slot contains each register
    unsigned StackTop;          // The current top of the FP stack.

    void dumpStack() const {
      cerr << "Stack contents:";
      for (unsigned i = 0; i != StackTop; ++i) {
        cerr << " FP" << Stack[i];
        assert(RegMap[Stack[i]] == i && "Stack[] doesn't match RegMap[]!");
      }
      cerr << "\n";
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
        unsigned STReg = getSTReg(RegNo);
        unsigned RegOnTop = getStackEntry(0);

        // Swap the slots the regs are in
        std::swap(RegMap[RegNo], RegMap[RegOnTop]);

        // Swap stack slot contents
        assert(RegMap[RegOnTop] < StackTop);
        std::swap(Stack[RegMap[RegOnTop]], Stack[StackTop-1]);

        // Emit an fxch to update the runtime processors version of the state
        BuildMI(*MBB, I, TII->get(X86::XCH_F)).addReg(STReg);
        NumFXCH++;
      }
    }

    void duplicateToTop(unsigned RegNo, unsigned AsReg, MachineInstr *I) {
      unsigned STReg = getSTReg(RegNo);
      pushReg(AsReg);   // New register on top of stack

      BuildMI(*MBB, I, TII->get(X86::LD_Frr)).addReg(STReg);
    }

    // popStackAfter - Pop the current value off of the top of the FP stack
    // after the specified instruction.
    void popStackAfter(MachineBasicBlock::iterator &I);

    // freeStackSlotAfter - Free the specified register from the register stack,
    // so that it is no longer in a register.  If the register is currently at
    // the top of the stack, we just pop the current instruction, otherwise we
    // store the current top-of-stack into the specified slot, then pop the top
    // of stack.
    void freeStackSlotAfter(MachineBasicBlock::iterator &I, unsigned Reg);

    bool processBasicBlock(MachineFunction &MF, MachineBasicBlock &MBB);

    void handleZeroArgFP(MachineBasicBlock::iterator &I);
    void handleOneArgFP(MachineBasicBlock::iterator &I);
    void handleOneArgFPRW(MachineBasicBlock::iterator &I);
    void handleTwoArgFP(MachineBasicBlock::iterator &I);
    void handleCompareFP(MachineBasicBlock::iterator &I);
    void handleCondMovFP(MachineBasicBlock::iterator &I);
    void handleSpecialFP(MachineBasicBlock::iterator &I);
  };
  char FPS::ID = 0;
}

FunctionPass *llvm::createX86FloatingPointStackifierPass() { return new FPS(); }

/// runOnMachineFunction - Loop over all of the basic blocks, transforming FP
/// register references into FP stack references.
///
bool FPS::runOnMachineFunction(MachineFunction &MF) {
  // We only need to run this pass if there are any FP registers used in this
  // function.  If it is all integer, there is nothing for us to do!
  bool FPIsUsed = false;

  assert(X86::FP6 == X86::FP0+6 && "Register enums aren't sorted right!");
  for (unsigned i = 0; i <= 6; ++i)
    if (MF.isPhysRegUsed(X86::FP0+i)) {
      FPIsUsed = true;
      break;
    }

  // Early exit.
  if (!FPIsUsed) return false;

  TII = MF.getTarget().getInstrInfo();
  LV = &getAnalysis<LiveVariables>();
  StackTop = 0;

  // Process the function in depth first order so that we process at least one
  // of the predecessors for every reachable block in the function.
  std::set<MachineBasicBlock*> Processed;
  MachineBasicBlock *Entry = MF.begin();

  bool Changed = false;
  for (df_ext_iterator<MachineBasicBlock*, std::set<MachineBasicBlock*> >
         I = df_ext_begin(Entry, Processed), E = df_ext_end(Entry, Processed);
       I != E; ++I)
    Changed |= processBasicBlock(MF, **I);

  return Changed;
}

/// processBasicBlock - Loop over all of the instructions in the basic block,
/// transforming FP instructions into their stack form.
///
bool FPS::processBasicBlock(MachineFunction &MF, MachineBasicBlock &BB) {
  bool Changed = false;
  MBB = &BB;

  for (MachineBasicBlock::iterator I = BB.begin(); I != BB.end(); ++I) {
    MachineInstr *MI = I;
    unsigned Flags = MI->getInstrDescriptor()->TSFlags;
    if ((Flags & X86II::FPTypeMask) == X86II::NotFP)
      continue;  // Efficiently ignore non-fp insts!

    MachineInstr *PrevMI = 0;
    if (I != BB.begin())
        PrevMI = prior(I);

    ++NumFP;  // Keep track of # of pseudo instrs
    DOUT << "\nFPInst:\t" << *MI;

    // Get dead variables list now because the MI pointer may be deleted as part
    // of processing!
    SmallVector<unsigned, 8> DeadRegs;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      if (MO.isRegister() && MO.isDead())
        DeadRegs.push_back(MO.getReg());
    }

    switch (Flags & X86II::FPTypeMask) {
    case X86II::ZeroArgFP:  handleZeroArgFP(I); break;
    case X86II::OneArgFP:   handleOneArgFP(I);  break;  // fstp ST(0)
    case X86II::OneArgFPRW: handleOneArgFPRW(I); break; // ST(0) = fsqrt(ST(0))
    case X86II::TwoArgFP:   handleTwoArgFP(I);  break;
    case X86II::CompareFP:  handleCompareFP(I); break;
    case X86II::CondMovFP:  handleCondMovFP(I); break;
    case X86II::SpecialFP:  handleSpecialFP(I); break;
    default: assert(0 && "Unknown FP Type!");
    }

    // Check to see if any of the values defined by this instruction are dead
    // after definition.  If so, pop them.
    for (unsigned i = 0, e = DeadRegs.size(); i != e; ++i) {
      unsigned Reg = DeadRegs[i];
      if (Reg >= X86::FP0 && Reg <= X86::FP6) {
        DOUT << "Register FP#" << Reg-X86::FP0 << " is dead!\n";
        freeStackSlotAfter(I, Reg-X86::FP0);
      }
    }

    // Print out all of the instructions expanded to if -debug
    DEBUG(
      MachineBasicBlock::iterator PrevI(PrevMI);
      if (I == PrevI) {
        cerr << "Just deleted pseudo instruction\n";
      } else {
        MachineBasicBlock::iterator Start = I;
        // Rewind to first instruction newly inserted.
        while (Start != BB.begin() && prior(Start) != PrevI) --Start;
        cerr << "Inserted instructions:\n\t";
        Start->print(*cerr.stream(), &MF.getTarget());
        while (++Start != next(I)) {}
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
    friend bool operator<(const TableEntry &TE, unsigned V) {
      return TE.from < V;
    }
    friend bool operator<(unsigned V, const TableEntry &TE) {
      return V < TE.from;
    }
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

#ifdef NDEBUG
#define ASSERT_SORTED(TABLE)
#else
#define ASSERT_SORTED(TABLE)                                              \
  { static bool TABLE##Checked = false;                                   \
    if (!TABLE##Checked) {                                                \
       assert(TableIsSorted(TABLE, array_lengthof(TABLE)) &&              \
              "All lookup tables must be sorted for efficient access!");  \
       TABLE##Checked = true;                                             \
    }                                                                     \
  }
#endif

//===----------------------------------------------------------------------===//
// Register File -> Register Stack Mapping Methods
//===----------------------------------------------------------------------===//

// OpcodeTable - Sorted map of register instructions to their stack version.
// The first element is an register file pseudo instruction, the second is the
// concrete X86 instruction which uses the register stack.
//
static const TableEntry OpcodeTable[] = {
  { X86::ABS_Fp32     , X86::ABS_F     },
  { X86::ABS_Fp64     , X86::ABS_F     },
  { X86::ABS_Fp80     , X86::ABS_F     },
  { X86::ADD_Fp32m    , X86::ADD_F32m  },
  { X86::ADD_Fp64m    , X86::ADD_F64m  },
  { X86::ADD_Fp64m32  , X86::ADD_F32m  },
  { X86::ADD_Fp80m32  , X86::ADD_F32m  },
  { X86::ADD_Fp80m64  , X86::ADD_F64m  },
  { X86::ADD_FpI16m32 , X86::ADD_FI16m },
  { X86::ADD_FpI16m64 , X86::ADD_FI16m },
  { X86::ADD_FpI16m80 , X86::ADD_FI16m },
  { X86::ADD_FpI32m32 , X86::ADD_FI32m },
  { X86::ADD_FpI32m64 , X86::ADD_FI32m },
  { X86::ADD_FpI32m80 , X86::ADD_FI32m },
  { X86::CHS_Fp32     , X86::CHS_F     },
  { X86::CHS_Fp64     , X86::CHS_F     },
  { X86::CHS_Fp80     , X86::CHS_F     },
  { X86::CMOVBE_Fp32  , X86::CMOVBE_F  },
  { X86::CMOVBE_Fp64  , X86::CMOVBE_F  },
  { X86::CMOVBE_Fp80  , X86::CMOVBE_F  },
  { X86::CMOVB_Fp32   , X86::CMOVB_F   },
  { X86::CMOVB_Fp64   , X86::CMOVB_F  },
  { X86::CMOVB_Fp80   , X86::CMOVB_F  },
  { X86::CMOVE_Fp32   , X86::CMOVE_F  },
  { X86::CMOVE_Fp64   , X86::CMOVE_F   },
  { X86::CMOVE_Fp80   , X86::CMOVE_F   },
  { X86::CMOVNBE_Fp32 , X86::CMOVNBE_F },
  { X86::CMOVNBE_Fp64 , X86::CMOVNBE_F },
  { X86::CMOVNBE_Fp80 , X86::CMOVNBE_F },
  { X86::CMOVNB_Fp32  , X86::CMOVNB_F  },
  { X86::CMOVNB_Fp64  , X86::CMOVNB_F  },
  { X86::CMOVNB_Fp80  , X86::CMOVNB_F  },
  { X86::CMOVNE_Fp32  , X86::CMOVNE_F  },
  { X86::CMOVNE_Fp64  , X86::CMOVNE_F  },
  { X86::CMOVNE_Fp80  , X86::CMOVNE_F  },
  { X86::CMOVNP_Fp32  , X86::CMOVNP_F  },
  { X86::CMOVNP_Fp64  , X86::CMOVNP_F  },
  { X86::CMOVNP_Fp80  , X86::CMOVNP_F  },
  { X86::CMOVP_Fp32   , X86::CMOVP_F   },
  { X86::CMOVP_Fp64   , X86::CMOVP_F   },
  { X86::CMOVP_Fp80   , X86::CMOVP_F   },
  { X86::COS_Fp32     , X86::COS_F     },
  { X86::COS_Fp64     , X86::COS_F     },
  { X86::COS_Fp80     , X86::COS_F     },
  { X86::DIVR_Fp32m   , X86::DIVR_F32m },
  { X86::DIVR_Fp64m   , X86::DIVR_F64m },
  { X86::DIVR_Fp64m32 , X86::DIVR_F32m },
  { X86::DIVR_Fp80m32 , X86::DIVR_F32m },
  { X86::DIVR_Fp80m64 , X86::DIVR_F64m },
  { X86::DIVR_FpI16m32, X86::DIVR_FI16m},
  { X86::DIVR_FpI16m64, X86::DIVR_FI16m},
  { X86::DIVR_FpI16m80, X86::DIVR_FI16m},
  { X86::DIVR_FpI32m32, X86::DIVR_FI32m},
  { X86::DIVR_FpI32m64, X86::DIVR_FI32m},
  { X86::DIVR_FpI32m80, X86::DIVR_FI32m},
  { X86::DIV_Fp32m    , X86::DIV_F32m  },
  { X86::DIV_Fp64m    , X86::DIV_F64m  },
  { X86::DIV_Fp64m32  , X86::DIV_F32m  },
  { X86::DIV_Fp80m32  , X86::DIV_F32m  },
  { X86::DIV_Fp80m64  , X86::DIV_F64m  },
  { X86::DIV_FpI16m32 , X86::DIV_FI16m },
  { X86::DIV_FpI16m64 , X86::DIV_FI16m },
  { X86::DIV_FpI16m80 , X86::DIV_FI16m },
  { X86::DIV_FpI32m32 , X86::DIV_FI32m },
  { X86::DIV_FpI32m64 , X86::DIV_FI32m },
  { X86::DIV_FpI32m80 , X86::DIV_FI32m },
  { X86::ILD_Fp16m32  , X86::ILD_F16m  },
  { X86::ILD_Fp16m64  , X86::ILD_F16m  },
  { X86::ILD_Fp16m80  , X86::ILD_F16m  },
  { X86::ILD_Fp32m32  , X86::ILD_F32m  },
  { X86::ILD_Fp32m64  , X86::ILD_F32m  },
  { X86::ILD_Fp32m80  , X86::ILD_F32m  },
  { X86::ILD_Fp64m32  , X86::ILD_F64m  },
  { X86::ILD_Fp64m64  , X86::ILD_F64m  },
  { X86::ILD_Fp64m80  , X86::ILD_F64m  },
  { X86::ISTT_Fp16m32 , X86::ISTT_FP16m},
  { X86::ISTT_Fp16m64 , X86::ISTT_FP16m},
  { X86::ISTT_Fp16m80 , X86::ISTT_FP16m},
  { X86::ISTT_Fp32m32 , X86::ISTT_FP32m},
  { X86::ISTT_Fp32m64 , X86::ISTT_FP32m},
  { X86::ISTT_Fp32m80 , X86::ISTT_FP32m},
  { X86::ISTT_Fp64m32 , X86::ISTT_FP64m},
  { X86::ISTT_Fp64m64 , X86::ISTT_FP64m},
  { X86::ISTT_Fp64m80 , X86::ISTT_FP64m},
  { X86::IST_Fp16m32  , X86::IST_F16m  },
  { X86::IST_Fp16m64  , X86::IST_F16m  },
  { X86::IST_Fp16m80  , X86::IST_F16m  },
  { X86::IST_Fp32m32  , X86::IST_F32m  },
  { X86::IST_Fp32m64  , X86::IST_F32m  },
  { X86::IST_Fp32m80  , X86::IST_F32m  },
  { X86::IST_Fp64m32  , X86::IST_FP64m },
  { X86::IST_Fp64m64  , X86::IST_FP64m },
  { X86::IST_Fp64m80  , X86::IST_FP64m },
  { X86::LD_Fp032     , X86::LD_F0     },
  { X86::LD_Fp064     , X86::LD_F0     },
  { X86::LD_Fp080     , X86::LD_F0     },
  { X86::LD_Fp132     , X86::LD_F1     },
  { X86::LD_Fp164     , X86::LD_F1     },
  { X86::LD_Fp180     , X86::LD_F1     },
  { X86::LD_Fp32m     , X86::LD_F32m   },
  { X86::LD_Fp32m64   , X86::LD_F32m   },
  { X86::LD_Fp32m80   , X86::LD_F32m   },
  { X86::LD_Fp64m     , X86::LD_F64m   },
  { X86::LD_Fp64m80   , X86::LD_F64m   },
  { X86::LD_Fp80m     , X86::LD_F80m   },
  { X86::MUL_Fp32m    , X86::MUL_F32m  },
  { X86::MUL_Fp64m    , X86::MUL_F64m  },
  { X86::MUL_Fp64m32  , X86::MUL_F32m  },
  { X86::MUL_Fp80m32  , X86::MUL_F32m  },
  { X86::MUL_Fp80m64  , X86::MUL_F64m  },
  { X86::MUL_FpI16m32 , X86::MUL_FI16m },
  { X86::MUL_FpI16m64 , X86::MUL_FI16m },
  { X86::MUL_FpI16m80 , X86::MUL_FI16m },
  { X86::MUL_FpI32m32 , X86::MUL_FI32m },
  { X86::MUL_FpI32m64 , X86::MUL_FI32m },
  { X86::MUL_FpI32m80 , X86::MUL_FI32m },

  // TEMPORARY
  { X86::NEW_CMOVBE_Fp32  , X86::CMOVBE_F  },
  { X86::NEW_CMOVBE_Fp64  , X86::CMOVBE_F  },
  { X86::NEW_CMOVBE_Fp80  , X86::CMOVBE_F  },
  { X86::NEW_CMOVB_Fp32   , X86::CMOVB_F   },
  { X86::NEW_CMOVB_Fp64   , X86::CMOVB_F  },
  { X86::NEW_CMOVB_Fp80   , X86::CMOVB_F  },
  { X86::NEW_CMOVE_Fp32   , X86::CMOVE_F  },
  { X86::NEW_CMOVE_Fp64   , X86::CMOVE_F   },
  { X86::NEW_CMOVE_Fp80   , X86::CMOVE_F   },
  { X86::NEW_CMOVNBE_Fp32 , X86::CMOVNBE_F },
  { X86::NEW_CMOVNBE_Fp64 , X86::CMOVNBE_F },
  { X86::NEW_CMOVNBE_Fp80 , X86::CMOVNBE_F },
  { X86::NEW_CMOVNB_Fp32  , X86::CMOVNB_F  },
  { X86::NEW_CMOVNB_Fp64  , X86::CMOVNB_F  },
  { X86::NEW_CMOVNB_Fp80  , X86::CMOVNB_F  },
  { X86::NEW_CMOVNE_Fp32  , X86::CMOVNE_F  },
  { X86::NEW_CMOVNE_Fp64  , X86::CMOVNE_F  },
  { X86::NEW_CMOVNE_Fp80  , X86::CMOVNE_F  },
  { X86::NEW_CMOVNP_Fp32  , X86::CMOVNP_F  },
  { X86::NEW_CMOVNP_Fp64  , X86::CMOVNP_F  },
  { X86::NEW_CMOVNP_Fp80  , X86::CMOVNP_F  },
  { X86::NEW_CMOVP_Fp32   , X86::CMOVP_F   },
  { X86::NEW_CMOVP_Fp64   , X86::CMOVP_F   },
  { X86::NEW_CMOVP_Fp80   , X86::CMOVP_F   },
  { X86::NEW_UCOM_FpIr32  , X86::UCOM_FIr  },
  { X86::NEW_UCOM_FpIr64  , X86::UCOM_FIr  },
  { X86::NEW_UCOM_FpIr80  , X86::UCOM_FIr  },
  { X86::NEW_UCOM_Fpr32   , X86::UCOM_Fr   },
  { X86::NEW_UCOM_Fpr64   , X86::UCOM_Fr   },
  { X86::NEW_UCOM_Fpr80   , X86::UCOM_Fr   },

  { X86::SIN_Fp32     , X86::SIN_F     },
  { X86::SIN_Fp64     , X86::SIN_F     },
  { X86::SIN_Fp80     , X86::SIN_F     },
  { X86::SQRT_Fp32    , X86::SQRT_F    },
  { X86::SQRT_Fp64    , X86::SQRT_F    },
  { X86::SQRT_Fp80    , X86::SQRT_F    },
  { X86::ST_Fp32m     , X86::ST_F32m   },
  { X86::ST_Fp64m     , X86::ST_F64m   },
  { X86::ST_Fp64m32   , X86::ST_F32m   },
  { X86::ST_Fp80m32   , X86::ST_F32m   },
  { X86::ST_Fp80m64   , X86::ST_F64m   },
  { X86::ST_FpP80m    , X86::ST_FP80m  },
  { X86::SUBR_Fp32m   , X86::SUBR_F32m },
  { X86::SUBR_Fp64m   , X86::SUBR_F64m },
  { X86::SUBR_Fp64m32 , X86::SUBR_F32m },
  { X86::SUBR_Fp80m32 , X86::SUBR_F32m },
  { X86::SUBR_Fp80m64 , X86::SUBR_F64m },
  { X86::SUBR_FpI16m32, X86::SUBR_FI16m},
  { X86::SUBR_FpI16m64, X86::SUBR_FI16m},
  { X86::SUBR_FpI16m80, X86::SUBR_FI16m},
  { X86::SUBR_FpI32m32, X86::SUBR_FI32m},
  { X86::SUBR_FpI32m64, X86::SUBR_FI32m},
  { X86::SUBR_FpI32m80, X86::SUBR_FI32m},
  { X86::SUB_Fp32m    , X86::SUB_F32m  },
  { X86::SUB_Fp64m    , X86::SUB_F64m  },
  { X86::SUB_Fp64m32  , X86::SUB_F32m  },
  { X86::SUB_Fp80m32  , X86::SUB_F32m  },
  { X86::SUB_Fp80m64  , X86::SUB_F64m  },
  { X86::SUB_FpI16m32 , X86::SUB_FI16m },
  { X86::SUB_FpI16m64 , X86::SUB_FI16m },
  { X86::SUB_FpI16m80 , X86::SUB_FI16m },
  { X86::SUB_FpI32m32 , X86::SUB_FI32m },
  { X86::SUB_FpI32m64 , X86::SUB_FI32m },
  { X86::SUB_FpI32m80 , X86::SUB_FI32m },
  { X86::TST_Fp32     , X86::TST_F     },
  { X86::TST_Fp64     , X86::TST_F     },
  { X86::TST_Fp80     , X86::TST_F     },
  { X86::UCOM_FpIr32  , X86::UCOM_FIr  },
  { X86::UCOM_FpIr64  , X86::UCOM_FIr  },
  { X86::UCOM_FpIr80  , X86::UCOM_FIr  },
  { X86::UCOM_Fpr32   , X86::UCOM_Fr   },
  { X86::UCOM_Fpr64   , X86::UCOM_Fr   },
  { X86::UCOM_Fpr80   , X86::UCOM_Fr   },
};

static unsigned getConcreteOpcode(unsigned Opcode) {
  ASSERT_SORTED(OpcodeTable);
  int Opc = Lookup(OpcodeTable, array_lengthof(OpcodeTable), Opcode);
  assert(Opc != -1 && "FP Stack instruction not in OpcodeTable!");
  return Opc;
}

//===----------------------------------------------------------------------===//
// Helper Methods
//===----------------------------------------------------------------------===//

// PopTable - Sorted map of instructions to their popping version.  The first
// element is an instruction, the second is the version which pops.
//
static const TableEntry PopTable[] = {
  { X86::ADD_FrST0 , X86::ADD_FPrST0  },

  { X86::DIVR_FrST0, X86::DIVR_FPrST0 },
  { X86::DIV_FrST0 , X86::DIV_FPrST0  },

  { X86::IST_F16m  , X86::IST_FP16m   },
  { X86::IST_F32m  , X86::IST_FP32m   },

  { X86::MUL_FrST0 , X86::MUL_FPrST0  },

  { X86::ST_F32m   , X86::ST_FP32m    },
  { X86::ST_F64m   , X86::ST_FP64m    },
  { X86::ST_Frr    , X86::ST_FPrr     },

  { X86::SUBR_FrST0, X86::SUBR_FPrST0 },
  { X86::SUB_FrST0 , X86::SUB_FPrST0  },

  { X86::UCOM_FIr  , X86::UCOM_FIPr   },

  { X86::UCOM_FPr  , X86::UCOM_FPPr   },
  { X86::UCOM_Fr   , X86::UCOM_FPr    },
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
  int Opcode = Lookup(PopTable, array_lengthof(PopTable), I->getOpcode());
  if (Opcode != -1) {
    I->setInstrDescriptor(TII->get(Opcode));
    if (Opcode == X86::UCOM_FPPr)
      I->RemoveOperand(0);
  } else {    // Insert an explicit pop
    I = BuildMI(*MBB, ++I, TII->get(X86::ST_FPrr)).addReg(X86::ST0);
  }
}

/// freeStackSlotAfter - Free the specified register from the register stack, so
/// that it is no longer in a register.  If the register is currently at the top
/// of the stack, we just pop the current instruction, otherwise we store the
/// current top-of-stack into the specified slot, then pop the top of stack.
void FPS::freeStackSlotAfter(MachineBasicBlock::iterator &I, unsigned FPRegNo) {
  if (getStackEntry(0) == FPRegNo) {  // already at the top of stack? easy.
    popStackAfter(I);
    return;
  }

  // Otherwise, store the top of stack into the dead slot, killing the operand
  // without having to add in an explicit xchg then pop.
  //
  unsigned STReg    = getSTReg(FPRegNo);
  unsigned OldSlot  = getSlot(FPRegNo);
  unsigned TopReg   = Stack[StackTop-1];
  Stack[OldSlot]    = TopReg;
  RegMap[TopReg]    = OldSlot;
  RegMap[FPRegNo]   = ~0;
  Stack[--StackTop] = ~0;
  I = BuildMI(*MBB, ++I, TII->get(X86::ST_FPrr)).addReg(STReg);
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

  // Change from the pseudo instruction to the concrete instruction.
  MI->RemoveOperand(0);   // Remove the explicit ST(0) operand
  MI->setInstrDescriptor(TII->get(getConcreteOpcode(MI->getOpcode())));
  
  // Result gets pushed on the stack.
  pushReg(DestReg);
}

/// handleOneArgFP - fst <mem>, ST(0)
///
void FPS::handleOneArgFP(MachineBasicBlock::iterator &I) {
  MachineInstr *MI = I;
  unsigned NumOps = MI->getInstrDescriptor()->numOperands;
  assert((NumOps == 5 || NumOps == 1) &&
         "Can only handle fst* & ftst instructions!");

  // Is this the last use of the source register?
  unsigned Reg = getFPReg(MI->getOperand(NumOps-1));
  bool KillsSrc = LV->KillsRegister(MI, X86::FP0+Reg);

  // FISTP64m is strange because there isn't a non-popping versions.
  // If we have one _and_ we don't want to pop the operand, duplicate the value
  // on the stack instead of moving it.  This ensure that popping the value is
  // always ok.
  // Ditto FISTTP16m, FISTTP32m, FISTTP64m, ST_FpP80m.
  //
  if (!KillsSrc &&
      (MI->getOpcode() == X86::IST_Fp64m32 ||
       MI->getOpcode() == X86::ISTT_Fp16m32 ||
       MI->getOpcode() == X86::ISTT_Fp32m32 ||
       MI->getOpcode() == X86::ISTT_Fp64m32 ||
       MI->getOpcode() == X86::IST_Fp64m64 ||
       MI->getOpcode() == X86::ISTT_Fp16m64 ||
       MI->getOpcode() == X86::ISTT_Fp32m64 ||
       MI->getOpcode() == X86::ISTT_Fp64m64 ||
       MI->getOpcode() == X86::IST_Fp64m80 ||
       MI->getOpcode() == X86::ISTT_Fp16m80 ||
       MI->getOpcode() == X86::ISTT_Fp32m80 ||
       MI->getOpcode() == X86::ISTT_Fp64m80 ||
       MI->getOpcode() == X86::ST_FpP80m)) {
    duplicateToTop(Reg, 7 /*temp register*/, I);
  } else {
    moveToTop(Reg, I);            // Move to the top of the stack...
  }
  
  // Convert from the pseudo instruction to the concrete instruction.
  MI->RemoveOperand(NumOps-1);    // Remove explicit ST(0) operand
  MI->setInstrDescriptor(TII->get(getConcreteOpcode(MI->getOpcode())));

  if (MI->getOpcode() == X86::IST_FP64m ||
      MI->getOpcode() == X86::ISTT_FP16m ||
      MI->getOpcode() == X86::ISTT_FP32m ||
      MI->getOpcode() == X86::ISTT_FP64m ||
      MI->getOpcode() == X86::ST_FP80m) {
    assert(StackTop > 0 && "Stack empty??");
    --StackTop;
  } else if (KillsSrc) { // Last use of operand?
    popStackAfter(I);
  }
}


/// handleOneArgFPRW: Handle instructions that read from the top of stack and
/// replace the value with a newly computed value.  These instructions may have
/// non-fp operands after their FP operands.
///
///  Examples:
///     R1 = fchs R2
///     R1 = fadd R2, [mem]
///
void FPS::handleOneArgFPRW(MachineBasicBlock::iterator &I) {
  MachineInstr *MI = I;
  unsigned NumOps = MI->getInstrDescriptor()->numOperands;
  assert(NumOps >= 2 && "FPRW instructions must have 2 ops!!");

  // Is this the last use of the source register?
  unsigned Reg = getFPReg(MI->getOperand(1));
  bool KillsSrc = LV->KillsRegister(MI, X86::FP0+Reg);

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

  // Change from the pseudo instruction to the concrete instruction.
  MI->RemoveOperand(1);   // Drop the source operand.
  MI->RemoveOperand(0);   // Drop the destination operand.
  MI->setInstrDescriptor(TII->get(getConcreteOpcode(MI->getOpcode())));
}


//===----------------------------------------------------------------------===//
// Define tables of various ways to map pseudo instructions
//

// ForwardST0Table - Map: A = B op C  into: ST(0) = ST(0) op ST(i)
static const TableEntry ForwardST0Table[] = {
  { X86::ADD_Fp32  , X86::ADD_FST0r },
  { X86::ADD_Fp64  , X86::ADD_FST0r },
  { X86::ADD_Fp80  , X86::ADD_FST0r },
  { X86::DIV_Fp32  , X86::DIV_FST0r },
  { X86::DIV_Fp64  , X86::DIV_FST0r },
  { X86::DIV_Fp80  , X86::DIV_FST0r },
  { X86::MUL_Fp32  , X86::MUL_FST0r },
  { X86::MUL_Fp64  , X86::MUL_FST0r },
  { X86::MUL_Fp80  , X86::MUL_FST0r },
  { X86::SUB_Fp32  , X86::SUB_FST0r },
  { X86::SUB_Fp64  , X86::SUB_FST0r },
  { X86::SUB_Fp80  , X86::SUB_FST0r },
};

// ReverseST0Table - Map: A = B op C  into: ST(0) = ST(i) op ST(0)
static const TableEntry ReverseST0Table[] = {
  { X86::ADD_Fp32  , X86::ADD_FST0r  },   // commutative
  { X86::ADD_Fp64  , X86::ADD_FST0r  },   // commutative
  { X86::ADD_Fp80  , X86::ADD_FST0r  },   // commutative
  { X86::DIV_Fp32  , X86::DIVR_FST0r },
  { X86::DIV_Fp64  , X86::DIVR_FST0r },
  { X86::DIV_Fp80  , X86::DIVR_FST0r },
  { X86::MUL_Fp32  , X86::MUL_FST0r  },   // commutative
  { X86::MUL_Fp64  , X86::MUL_FST0r  },   // commutative
  { X86::MUL_Fp80  , X86::MUL_FST0r  },   // commutative
  { X86::SUB_Fp32  , X86::SUBR_FST0r },
  { X86::SUB_Fp64  , X86::SUBR_FST0r },
  { X86::SUB_Fp80  , X86::SUBR_FST0r },
};

// ForwardSTiTable - Map: A = B op C  into: ST(i) = ST(0) op ST(i)
static const TableEntry ForwardSTiTable[] = {
  { X86::ADD_Fp32  , X86::ADD_FrST0  },   // commutative
  { X86::ADD_Fp64  , X86::ADD_FrST0  },   // commutative
  { X86::ADD_Fp80  , X86::ADD_FrST0  },   // commutative
  { X86::DIV_Fp32  , X86::DIVR_FrST0 },
  { X86::DIV_Fp64  , X86::DIVR_FrST0 },
  { X86::DIV_Fp80  , X86::DIVR_FrST0 },
  { X86::MUL_Fp32  , X86::MUL_FrST0  },   // commutative
  { X86::MUL_Fp64  , X86::MUL_FrST0  },   // commutative
  { X86::MUL_Fp80  , X86::MUL_FrST0  },   // commutative
  { X86::SUB_Fp32  , X86::SUBR_FrST0 },
  { X86::SUB_Fp64  , X86::SUBR_FrST0 },
  { X86::SUB_Fp80  , X86::SUBR_FrST0 },
};

// ReverseSTiTable - Map: A = B op C  into: ST(i) = ST(i) op ST(0)
static const TableEntry ReverseSTiTable[] = {
  { X86::ADD_Fp32  , X86::ADD_FrST0 },
  { X86::ADD_Fp64  , X86::ADD_FrST0 },
  { X86::ADD_Fp80  , X86::ADD_FrST0 },
  { X86::DIV_Fp32  , X86::DIV_FrST0 },
  { X86::DIV_Fp64  , X86::DIV_FrST0 },
  { X86::DIV_Fp80  , X86::DIV_FrST0 },
  { X86::MUL_Fp32  , X86::MUL_FrST0 },
  { X86::MUL_Fp64  , X86::MUL_FrST0 },
  { X86::MUL_Fp80  , X86::MUL_FrST0 },
  { X86::SUB_Fp32  , X86::SUB_FrST0 },
  { X86::SUB_Fp64  , X86::SUB_FrST0 },
  { X86::SUB_Fp80  , X86::SUB_FrST0 },
};


/// handleTwoArgFP - Handle instructions like FADD and friends which are virtual
/// instructions which need to be simplified and possibly transformed.
///
/// Result: ST(0) = fsub  ST(0), ST(i)
///         ST(i) = fsub  ST(0), ST(i)
///         ST(0) = fsubr ST(0), ST(i)
///         ST(i) = fsubr ST(0), ST(i)
///
void FPS::handleTwoArgFP(MachineBasicBlock::iterator &I) {
  ASSERT_SORTED(ForwardST0Table); ASSERT_SORTED(ReverseST0Table);
  ASSERT_SORTED(ForwardSTiTable); ASSERT_SORTED(ReverseSTiTable);
  MachineInstr *MI = I;

  unsigned NumOperands = MI->getInstrDescriptor()->numOperands;
  assert(NumOperands == 3 && "Illegal TwoArgFP instruction!");
  unsigned Dest = getFPReg(MI->getOperand(0));
  unsigned Op0 = getFPReg(MI->getOperand(NumOperands-2));
  unsigned Op1 = getFPReg(MI->getOperand(NumOperands-1));
  bool KillsOp0 = LV->KillsRegister(MI, X86::FP0+Op0);
  bool KillsOp1 = LV->KillsRegister(MI, X86::FP0+Op1);

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
  } else if (!KillsOp0 && !KillsOp1) {
    // If we DO have one of our operands at the top of the stack, but we don't
    // have a dead operand, we must duplicate one of the operands to a new slot
    // on the stack.
    duplicateToTop(Op0, Dest, I);
    Op0 = TOS = Dest;
    KillsOp0 = true;
  }

  // Now we know that one of our operands is on the top of the stack, and at
  // least one of our operands is killed by this instruction.
  assert((TOS == Op0 || TOS == Op1) && (KillsOp0 || KillsOp1) &&
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

  int Opcode = Lookup(InstTable, array_lengthof(ForwardST0Table),
                      MI->getOpcode());
  assert(Opcode != -1 && "Unknown TwoArgFP pseudo instruction!");

  // NotTOS - The register which is not on the top of stack...
  unsigned NotTOS = (TOS == Op0) ? Op1 : Op0;

  // Replace the old instruction with a new instruction
  MBB->remove(I++);
  I = BuildMI(*MBB, I, TII->get(Opcode)).addReg(getSTReg(NotTOS));

  // If both operands are killed, pop one off of the stack in addition to
  // overwriting the other one.
  if (KillsOp0 && KillsOp1 && Op0 != Op1) {
    assert(!updateST0 && "Should have updated other operand!");
    popStackAfter(I);   // Pop the top of stack
  }

  // Update stack information so that we know the destination register is now on
  // the stack.
  unsigned UpdatedSlot = getSlot(updateST0 ? TOS : NotTOS);
  assert(UpdatedSlot < StackTop && Dest < 7);
  Stack[UpdatedSlot]   = Dest;
  RegMap[Dest]         = UpdatedSlot;
  delete MI;   // Remove the old instruction
}

/// handleCompareFP - Handle FUCOM and FUCOMI instructions, which have two FP
/// register arguments and no explicit destinations.
///
void FPS::handleCompareFP(MachineBasicBlock::iterator &I) {
  ASSERT_SORTED(ForwardST0Table); ASSERT_SORTED(ReverseST0Table);
  ASSERT_SORTED(ForwardSTiTable); ASSERT_SORTED(ReverseSTiTable);
  MachineInstr *MI = I;

  unsigned NumOperands = MI->getInstrDescriptor()->numOperands;
  assert(NumOperands == 2 && "Illegal FUCOM* instruction!");
  unsigned Op0 = getFPReg(MI->getOperand(NumOperands-2));
  unsigned Op1 = getFPReg(MI->getOperand(NumOperands-1));
  bool KillsOp0 = LV->KillsRegister(MI, X86::FP0+Op0);
  bool KillsOp1 = LV->KillsRegister(MI, X86::FP0+Op1);

  // Make sure the first operand is on the top of stack, the other one can be
  // anywhere.
  moveToTop(Op0, I);

  // Change from the pseudo instruction to the concrete instruction.
  MI->getOperand(0).setReg(getSTReg(Op1));
  MI->RemoveOperand(1);
  MI->setInstrDescriptor(TII->get(getConcreteOpcode(MI->getOpcode())));

  // If any of the operands are killed by this instruction, free them.
  if (KillsOp0) freeStackSlotAfter(I, Op0);
  if (KillsOp1 && Op0 != Op1) freeStackSlotAfter(I, Op1);
}

/// handleCondMovFP - Handle two address conditional move instructions.  These
/// instructions move a st(i) register to st(0) iff a condition is true.  These
/// instructions require that the first operand is at the top of the stack, but
/// otherwise don't modify the stack at all.
void FPS::handleCondMovFP(MachineBasicBlock::iterator &I) {
  MachineInstr *MI = I;

  unsigned Op0 = getFPReg(MI->getOperand(0));
  unsigned Op1 = getFPReg(MI->getOperand(2));
  bool KillsOp1 = LV->KillsRegister(MI, X86::FP0+Op1);

  // The first operand *must* be on the top of the stack.
  moveToTop(Op0, I);

  // Change the second operand to the stack register that the operand is in.
  // Change from the pseudo instruction to the concrete instruction.
  MI->RemoveOperand(0);
  MI->RemoveOperand(1);
  MI->getOperand(0).setReg(getSTReg(Op1));
  MI->setInstrDescriptor(TII->get(getConcreteOpcode(MI->getOpcode())));
  
  // If we kill the second operand, make sure to pop it from the stack.
  if (Op0 != Op1 && KillsOp1) {
    // Get this value off of the register stack.
    freeStackSlotAfter(I, Op1);
  }
}


/// handleSpecialFP - Handle special instructions which behave unlike other
/// floating point instructions.  This is primarily intended for use by pseudo
/// instructions.
///
void FPS::handleSpecialFP(MachineBasicBlock::iterator &I) {
  MachineInstr *MI = I;
  switch (MI->getOpcode()) {
  default: assert(0 && "Unknown SpecialFP instruction!");
  case X86::FpGETRESULT32:  // Appears immediately after a call returning FP type!
  case X86::FpGETRESULT64:  // Appears immediately after a call returning FP type!
  case X86::FpGETRESULT80:
    assert(StackTop == 0 && "Stack should be empty after a call!");
    pushReg(getFPReg(MI->getOperand(0)));
    break;
  case X86::FpSETRESULT32:
  case X86::FpSETRESULT64:
  case X86::FpSETRESULT80:
    assert(StackTop == 1 && "Stack should have one element on it to return!");
    --StackTop;   // "Forget" we have something on the top of stack!
    break;
  case X86::MOV_Fp3232:
  case X86::MOV_Fp3264:
  case X86::MOV_Fp6432:
  case X86::MOV_Fp6464: 
  case X86::MOV_Fp3280:
  case X86::MOV_Fp6480:
  case X86::MOV_Fp8032:
  case X86::MOV_Fp8064: 
  case X86::MOV_Fp8080: {
    unsigned SrcReg = getFPReg(MI->getOperand(1));
    unsigned DestReg = getFPReg(MI->getOperand(0));

    if (LV->KillsRegister(MI, X86::FP0+SrcReg)) {
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
