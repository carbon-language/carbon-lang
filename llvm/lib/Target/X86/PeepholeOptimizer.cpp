//===-- PeepholeOptimizer.cpp - X86 Peephole Optimizer --------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains a peephole optimizer for the X86.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/MRegisterInfo.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"

using namespace llvm;

namespace {
  Statistic<> NumPHOpts("x86-peephole",
                        "Number of peephole optimization performed");
  struct PH : public MachineFunctionPass {
    virtual bool runOnMachineFunction(MachineFunction &MF);

    bool PeepholeOptimize(MachineBasicBlock &MBB,
			  MachineBasicBlock::iterator &I);

    virtual const char *getPassName() const { return "X86 Peephole Optimizer"; }
  };
}

FunctionPass *llvm::createX86PeepholeOptimizerPass() { return new PH(); }

bool PH::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  for (MachineFunction::iterator BI = MF.begin(), E = MF.end(); BI != E; ++BI)
    for (MachineBasicBlock::iterator I = BI->begin(); I != BI->end(); )
      if (PeepholeOptimize(*BI, I)) {
	Changed = true;
        ++NumPHOpts;
      } else
	++I;

  return Changed;
}


bool PH::PeepholeOptimize(MachineBasicBlock &MBB,
			  MachineBasicBlock::iterator &I) {
  assert(I != MBB.end());
  MachineBasicBlock::iterator NextI = next(I);

  MachineInstr *MI = I;
  MachineInstr *Next = (NextI != MBB.end()) ? &*NextI : (MachineInstr*)0;
  unsigned Size = 0;
  switch (MI->getOpcode()) {
  case X86::MOVrr8:
  case X86::MOVrr16:
  case X86::MOVrr32:   // Destroy X = X copies...
    if (MI->getOperand(0).getReg() == MI->getOperand(1).getReg()) {
      I = MBB.erase(I);
      return true;
    }
    return false;

    // A large number of X86 instructions have forms which take an 8-bit
    // immediate despite the fact that the operands are 16 or 32 bits.  Because
    // this can save three bytes of code size (and icache space), we want to
    // shrink them if possible.
  case X86::IMULrri16: case X86::IMULrri32:
    assert(MI->getNumOperands() == 3 && "These should all have 3 operands!");
    if (MI->getOperand(2).isImmediate()) {
      int Val = MI->getOperand(2).getImmedValue();
      // If the value is the same when signed extended from 8 bits...
      if (Val == (signed int)(signed char)Val) {
        unsigned Opcode;
        switch (MI->getOpcode()) {
        default: assert(0 && "Unknown opcode value!");
        case X86::IMULrri16: Opcode = X86::IMULrri16b; break;
        case X86::IMULrri32: Opcode = X86::IMULrri32b; break;
        }
        unsigned R0 = MI->getOperand(0).getReg();
        unsigned R1 = MI->getOperand(1).getReg();
        I = MBB.insert(MBB.erase(I),
                       BuildMI(Opcode, 2, R0).addReg(R1).addZImm((char)Val));
        return true;
      }
    }
    return false;

  case X86::IMULrmi16: case X86::IMULrmi32:
    assert(MI->getNumOperands() == 6 && "These should all have 6 operands!");
    if (MI->getOperand(5).isImmediate()) {
      int Val = MI->getOperand(5).getImmedValue();
      // If the value is the same when signed extended from 8 bits...
      if (Val == (signed int)(signed char)Val) {
        unsigned Opcode;
        switch (MI->getOpcode()) {
        default: assert(0 && "Unknown opcode value!");
        case X86::IMULrmi16: Opcode = X86::IMULrmi16b; break;
        case X86::IMULrmi32: Opcode = X86::IMULrmi32b; break;
        }
        unsigned R0 = MI->getOperand(0).getReg();
        unsigned R1 = MI->getOperand(1).getReg();
        unsigned Scale = MI->getOperand(2).getImmedValue();
        unsigned R2 = MI->getOperand(3).getReg();
        unsigned Offset = MI->getOperand(3).getImmedValue();
        I = MBB.insert(MBB.erase(I),
                       BuildMI(Opcode, 2, R0).addReg(R1).addZImm(Scale).
                             addReg(R2).addSImm(Offset).addZImm((char)Val));
        return true;
      }
    }
    return false;

  case X86::ADDri16:  case X86::ADDri32:
  case X86::ADDmi16:  case X86::ADDmi32:
  case X86::SUBri16:  case X86::SUBri32:
  case X86::ANDri16:  case X86::ANDri32:
  case X86::ORri16:   case X86::ORri32:
  case X86::XORri16:  case X86::XORri32:
    assert(MI->getNumOperands() == 2 && "These should all have 2 operands!");
    if (MI->getOperand(1).isImmediate()) {
      int Val = MI->getOperand(1).getImmedValue();
      // If the value is the same when signed extended from 8 bits...
      if (Val == (signed int)(signed char)Val) {
        unsigned Opcode;
        switch (MI->getOpcode()) {
        default: assert(0 && "Unknown opcode value!");
        case X86::ADDri16:  Opcode = X86::ADDri16b; break;
        case X86::ADDri32:  Opcode = X86::ADDri32b; break;
        case X86::ADDmi16:  Opcode = X86::ADDmi16b; break;
        case X86::ADDmi32:  Opcode = X86::ADDmi32b; break;
        case X86::SUBri16:  Opcode = X86::SUBri16b; break;
        case X86::SUBri32:  Opcode = X86::SUBri32b; break;
        case X86::ANDri16:  Opcode = X86::ANDri16b; break;
        case X86::ANDri32:  Opcode = X86::ANDri32b; break;
        case X86::ORri16:   Opcode = X86::ORri16b; break;
        case X86::ORri32:   Opcode = X86::ORri32b; break;
        case X86::XORri16:  Opcode = X86::XORri16b; break;
        case X86::XORri32:  Opcode = X86::XORri32b; break;
        }
        unsigned R0 = MI->getOperand(0).getReg();
        I = MBB.insert(MBB.erase(I),
                    BuildMI(Opcode, 1, R0, MOTy::UseAndDef).addZImm((char)Val));
        return true;
      }
    }
    return false;

#if 0
  case X86::MOVir32: Size++;
  case X86::MOVir16: Size++;
  case X86::MOVir8:
    // FIXME: We can only do this transformation if we know that flags are not
    // used here, because XOR clobbers the flags!
    if (MI->getOperand(1).isImmediate()) {         // avoid mov EAX, <value>
      int Val = MI->getOperand(1).getImmedValue();
      if (Val == 0) {                              // mov EAX, 0 -> xor EAX, EAX
	static const unsigned Opcode[] ={X86::XORrr8,X86::XORrr16,X86::XORrr32};
	unsigned Reg = MI->getOperand(0).getReg();
	I = MBB.insert(MBB.erase(I),
                       BuildMI(Opcode[Size], 2, Reg).addReg(Reg).addReg(Reg));
	return true;
      } else if (Val == -1) {                     // mov EAX, -1 -> or EAX, -1
	// TODO: 'or Reg, -1' has a smaller encoding than 'mov Reg, -1'
      }
    }
    return false;
#endif
  case X86::BSWAPr32:        // Change bswap EAX, bswap EAX into nothing
    if (Next->getOpcode() == X86::BSWAPr32 &&
	MI->getOperand(0).getReg() == Next->getOperand(0).getReg()) {
      I = MBB.erase(MBB.erase(I));
      return true;
    }
    return false;
  default:
    return false;
  }
}

namespace {
  class UseDefChains : public MachineFunctionPass {
    std::vector<MachineInstr*> DefiningInst;
  public:
    // getDefinition - Return the machine instruction that defines the specified
    // SSA virtual register.
    MachineInstr *getDefinition(unsigned Reg) {
      assert(MRegisterInfo::isVirtualRegister(Reg) &&
             "use-def chains only exist for SSA registers!");
      assert(Reg - MRegisterInfo::FirstVirtualRegister < DefiningInst.size() &&
             "Unknown register number!");
      assert(DefiningInst[Reg-MRegisterInfo::FirstVirtualRegister] &&
             "Unknown register number!");
      return DefiningInst[Reg-MRegisterInfo::FirstVirtualRegister];
    }

    // setDefinition - Update the use-def chains to indicate that MI defines
    // register Reg.
    void setDefinition(unsigned Reg, MachineInstr *MI) {
      if (Reg-MRegisterInfo::FirstVirtualRegister >= DefiningInst.size())
        DefiningInst.resize(Reg-MRegisterInfo::FirstVirtualRegister+1);
      DefiningInst[Reg-MRegisterInfo::FirstVirtualRegister] = MI;
    }

    // removeDefinition - Update the use-def chains to forget about Reg
    // entirely.
    void removeDefinition(unsigned Reg) {
      assert(getDefinition(Reg));      // Check validity
      DefiningInst[Reg-MRegisterInfo::FirstVirtualRegister] = 0;
    }

    virtual bool runOnMachineFunction(MachineFunction &MF) {
      for (MachineFunction::iterator BI = MF.begin(), E = MF.end(); BI!=E; ++BI)
        for (MachineBasicBlock::iterator I = BI->begin(); I != BI->end(); ++I) {
          for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
            MachineOperand &MO = I->getOperand(i);
            if (MO.isRegister() && MO.isDef() && !MO.isUse() &&
                MRegisterInfo::isVirtualRegister(MO.getReg()))
              setDefinition(MO.getReg(), I);
          }
        }
      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual void releaseMemory() {
      std::vector<MachineInstr*>().swap(DefiningInst);
    }
  };

  RegisterAnalysis<UseDefChains> X("use-def-chains",
                                "use-def chain construction for machine code");
}


namespace {
  Statistic<> NumSSAPHOpts("x86-ssa-peephole",
                           "Number of SSA peephole optimization performed");

  /// SSAPH - This pass is an X86-specific, SSA-based, peephole optimizer.  This
  /// pass is really a bad idea: a better instruction selector should completely
  /// supersume it.  However, that will take some time to develop, and the
  /// simple things this can do are important now.
  class SSAPH : public MachineFunctionPass {
    UseDefChains *UDC;
  public:
    virtual bool runOnMachineFunction(MachineFunction &MF);

    bool PeepholeOptimize(MachineBasicBlock &MBB,
			  MachineBasicBlock::iterator &I);

    virtual const char *getPassName() const {
      return "X86 SSA-based Peephole Optimizer";
    }

    /// Propagate - Set MI[DestOpNo] = Src[SrcOpNo], optionally change the
    /// opcode of the instruction, then return true.
    bool Propagate(MachineInstr *MI, unsigned DestOpNo,
                   MachineInstr *Src, unsigned SrcOpNo, unsigned NewOpcode = 0){
      MI->getOperand(DestOpNo) = Src->getOperand(SrcOpNo);
      if (NewOpcode) MI->setOpcode(NewOpcode);
      return true;
    }

    /// OptimizeAddress - If we can fold the addressing arithmetic for this
    /// memory instruction into the instruction itself, do so and return true.
    bool OptimizeAddress(MachineInstr *MI, unsigned OpNo);

    /// getDefininingInst - If the specified operand is a read of an SSA
    /// register, return the machine instruction defining it, otherwise, return
    /// null.
    MachineInstr *getDefiningInst(MachineOperand &MO) {
      if (MO.isDef() || !MO.isRegister() ||
          !MRegisterInfo::isVirtualRegister(MO.getReg())) return 0;
      return UDC->getDefinition(MO.getReg());
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<UseDefChains>();
      AU.addPreserved<UseDefChains>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
}

FunctionPass *llvm::createX86SSAPeepholeOptimizerPass() { return new SSAPH(); }

bool SSAPH::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  bool LocalChanged;

  UDC = &getAnalysis<UseDefChains>();

  do {
    LocalChanged = false;

    for (MachineFunction::iterator BI = MF.begin(), E = MF.end(); BI != E; ++BI)
      for (MachineBasicBlock::iterator I = BI->begin(); I != BI->end(); )
        if (PeepholeOptimize(*BI, I)) {
          LocalChanged = true;
          ++NumSSAPHOpts;
        } else
          ++I;
    Changed |= LocalChanged;
  } while (LocalChanged);

  return Changed;
}

static bool isValidScaleAmount(unsigned Scale) {
  return Scale == 1 || Scale == 2 || Scale == 4 || Scale == 8;
}

/// OptimizeAddress - If we can fold the addressing arithmetic for this
/// memory instruction into the instruction itself, do so and return true.
bool SSAPH::OptimizeAddress(MachineInstr *MI, unsigned OpNo) {
  MachineOperand &BaseRegOp      = MI->getOperand(OpNo+0);
  MachineOperand &ScaleOp        = MI->getOperand(OpNo+1);
  MachineOperand &IndexRegOp     = MI->getOperand(OpNo+2);
  MachineOperand &DisplacementOp = MI->getOperand(OpNo+3);

  unsigned BaseReg  = BaseRegOp.hasAllocatedReg() ? BaseRegOp.getReg() : 0;
  unsigned Scale    = ScaleOp.getImmedValue();
  unsigned IndexReg = IndexRegOp.hasAllocatedReg() ? IndexRegOp.getReg() : 0;

  bool Changed = false;

  // If the base register is unset, and the index register is set with a scale
  // of 1, move it to be the base register.
  if (BaseRegOp.hasAllocatedReg() && BaseReg == 0 &&
      Scale == 1 && IndexReg != 0) {
    BaseRegOp.setReg(IndexReg);
    IndexRegOp.setReg(0);
    return true;
  }

  // Attempt to fold instructions used by the base register into the instruction
  if (MachineInstr *DefInst = getDefiningInst(BaseRegOp)) {
    switch (DefInst->getOpcode()) {
    case X86::MOVir32:
      // If there is no displacement set for this instruction set one now.
      // FIXME: If we can fold two immediates together, we should do so!
      if (DisplacementOp.isImmediate() && !DisplacementOp.getImmedValue()) {
        if (DefInst->getOperand(1).isImmediate()) {
          BaseRegOp.setReg(0);
          return Propagate(MI, OpNo+3, DefInst, 1);
        }
      }
      break;

    case X86::ADDrr32:
      // If the source is a register-register add, and we do not yet have an
      // index register, fold the add into the memory address.
      if (IndexReg == 0) {
        BaseRegOp = DefInst->getOperand(1);
        IndexRegOp = DefInst->getOperand(2);
        ScaleOp.setImmedValue(1);
        return true;
      }
      break;

    case X86::SHLir32:
      // If this shift could be folded into the index portion of the address if
      // it were the index register, move it to the index register operand now,
      // so it will be folded in below.
      if ((Scale == 1 || (IndexReg == 0 && IndexRegOp.hasAllocatedReg())) &&
          DefInst->getOperand(2).getImmedValue() < 4) {
        std::swap(BaseRegOp, IndexRegOp);
        ScaleOp.setImmedValue(1); Scale = 1;
        std::swap(IndexReg, BaseReg);
        Changed = true;
        break;
      }
    }
  }

  // Attempt to fold instructions used by the index into the instruction
  if (MachineInstr *DefInst = getDefiningInst(IndexRegOp)) {
    switch (DefInst->getOpcode()) {
    case X86::SHLir32: {
      // Figure out what the resulting scale would be if we folded this shift.
      unsigned ResScale = Scale * (1 << DefInst->getOperand(2).getImmedValue());
      if (isValidScaleAmount(ResScale)) {
        IndexRegOp = DefInst->getOperand(1);
        ScaleOp.setImmedValue(ResScale);
        return true;
      }
      break;
    }
    }
  }

  return Changed;
}

bool SSAPH::PeepholeOptimize(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator &I) {
    MachineBasicBlock::iterator NextI = next(I);

  MachineInstr *MI = I;
  MachineInstr *Next = (NextI != MBB.end()) ? &*NextI : (MachineInstr*)0;

  bool Changed = false;

  // Scan the operands of this instruction.  If any operands are
  // register-register copies, replace the operand with the source.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
    // Is this an SSA register use?
    if (MachineInstr *DefInst = getDefiningInst(MI->getOperand(i)))
      // If the operand is a vreg-vreg copy, it is always safe to replace the
      // source value with the input operand.
      if (DefInst->getOpcode() == X86::MOVrr8  ||
          DefInst->getOpcode() == X86::MOVrr16 ||
          DefInst->getOpcode() == X86::MOVrr32) {
        // Don't propagate physical registers into PHI nodes...
        if (MI->getOpcode() != X86::PHI ||
            (DefInst->getOperand(1).isRegister() &&
             MRegisterInfo::isVirtualRegister(DefInst->getOperand(1).getReg())))
        Changed = Propagate(MI, i, DefInst, 1);
      }
  
  
  // Perform instruction specific optimizations.
  switch (MI->getOpcode()) {

    // Register to memory stores.  Format: <base,scale,indexreg,immdisp>, srcreg
  case X86::MOVrm32: case X86::MOVrm16: case X86::MOVrm8:
  case X86::MOVim32: case X86::MOVim16: case X86::MOVim8:
    // Check to see if we can fold the source instruction into this one...
    if (MachineInstr *SrcInst = getDefiningInst(MI->getOperand(4))) {
      switch (SrcInst->getOpcode()) {
        // Fold the immediate value into the store, if possible.
      case X86::MOVir8:  return Propagate(MI, 4, SrcInst, 1, X86::MOVim8);
      case X86::MOVir16: return Propagate(MI, 4, SrcInst, 1, X86::MOVim16);
      case X86::MOVir32: return Propagate(MI, 4, SrcInst, 1, X86::MOVim32);
      default: break;
      }
    }

    // If we can optimize the addressing expression, do so now.
    if (OptimizeAddress(MI, 0))
      return true;
    break;

  case X86::MOVmr32:
  case X86::MOVmr16:
  case X86::MOVmr8:
    // If we can optimize the addressing expression, do so now.
    if (OptimizeAddress(MI, 1))
      return true;
    break;

  default: break;
  }

  return Changed;
}
