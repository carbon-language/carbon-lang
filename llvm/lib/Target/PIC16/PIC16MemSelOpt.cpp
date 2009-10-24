//===-- PIC16MemSelOpt.cpp - PIC16 banksel optimizer  --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass which optimizes the emitting of banksel 
// instructions before accessing data memory. This currently works within
// a basic block only and keep tracks of the last accessed memory bank.
// If memory access continues to be in the same bank it just makes banksel
// immediate, which is a part of the insn accessing the data memory, from 1
// to zero. The asm printer emits a banksel only if that immediate is 1. 
//
// FIXME: this is not implemented yet.  The banksel pass only works on local
// basic blocks.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pic16-codegen"
#include "PIC16.h"
#include "PIC16ABINames.h"
#include "PIC16InstrInfo.h"
#include "PIC16MCAsmInfo.h"
#include "PIC16TargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/GlobalValue.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

namespace {
  struct VISIBILITY_HIDDEN MemSelOpt : public MachineFunctionPass {
    static char ID;
    MemSelOpt() : MachineFunctionPass(&ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreservedID(MachineLoopInfoID);
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const { 
      return "PIC16 Memsel Optimizer"; 
    }

   bool processBasicBlock(MachineFunction &MF, MachineBasicBlock &MBB);
   bool processInstruction(MachineInstr *MI);

  private:
    const TargetInstrInfo *TII; // Machine instruction info.
    MachineBasicBlock *MBB;     // Current basic block
    std::string CurBank;

  };
  char MemSelOpt::ID = 0;
}

FunctionPass *llvm::createPIC16MemSelOptimizerPass() { 
  return new MemSelOpt(); 
}


/// runOnMachineFunction - Loop over all of the basic blocks, transforming FP
/// register references into FP stack references.
///
bool MemSelOpt::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  bool Changed = false;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    Changed |= processBasicBlock(MF, *I);
  }

  return Changed;
}

/// processBasicBlock - Loop over all of the instructions in the basic block,
/// transforming FP instructions into their stack form.
///
bool MemSelOpt::processBasicBlock(MachineFunction &MF, MachineBasicBlock &BB) {
  bool Changed = false;
  MBB = &BB;

  // Let us assume that when entering a basic block now bank is selected.
  // Ideally we should look at the predecessors for this information.
  CurBank=""; 

  for (MachineBasicBlock::iterator I = BB.begin(); I != BB.end(); ++I) {
    Changed |= processInstruction(I);
  }
  return Changed;
}

bool MemSelOpt::processInstruction(MachineInstr *MI) {
  bool Changed = false;

  unsigned NumOperands = MI->getNumOperands();
  if (NumOperands == 0) return false;


  // If this insn is not going to access any memory, return.
  const TargetInstrDesc &TID = TII->get(MI->getOpcode());
  if (!(TID.isBranch() || TID.isCall() || TID.mayLoad() || TID.mayStore()))
    return false;

  // Scan for the memory address operand.
  // FIXME: Should we use standard interfaces like memoperands_iterator,
  // hasMemOperand() etc ?
  int MemOpPos = -1;
  for (unsigned i = 0; i < NumOperands; i++) {
    MachineOperand Op = MI->getOperand(i);
    if (Op.getType() ==  MachineOperand::MO_GlobalAddress ||
        Op.getType() ==  MachineOperand::MO_ExternalSymbol || 
        Op.getType() ==  MachineOperand::MO_MachineBasicBlock) {
      // We found one mem operand. Next one may be BS.
      MemOpPos = i;
      break;
    }
  }

  // If we did not find an insn accessing memory. Continue.
  if (MemOpPos == -1) return Changed;
 
  // Get the MemOp.
  MachineOperand &Op = MI->getOperand(MemOpPos);

  // If this is a pagesel material, handle it first.
  if (MI->getOpcode() == PIC16::CALL ||
      MI->getOpcode() == PIC16::br_uncond) {
    DebugLoc dl = MI->getDebugLoc();
    BuildMI(*MBB, MI, dl, TII->get(PIC16::pagesel)).
      addOperand(Op);
    return true;
  }

  // Get the section name(NewBank) for MemOp.
  // This assumes that the section names for globals are laready set by
  // AsmPrinter->doInitialization.
  std::string NewBank = CurBank;
  if (Op.getType() ==  MachineOperand::MO_GlobalAddress &&
      Op.getGlobal()->getType()->getAddressSpace() == PIC16ISD::RAM_SPACE) {
    NewBank = Op.getGlobal()->getSection();
  } else if (Op.getType() ==  MachineOperand::MO_ExternalSymbol) {
    // External Symbol is generated for temp data and arguments. They are
    // in fpdata.<functionname>.# section.
    std::string Sym = Op.getSymbolName();
    NewBank = PAN::getSectionNameForSym(Sym);
  }
 
  // If the previous and new section names are same, we don't need to
  // emit banksel. 
  if (NewBank.compare(CurBank) != 0 ) {
    DebugLoc dl = MI->getDebugLoc();
    BuildMI(*MBB, MI, dl, TII->get(PIC16::banksel)).
      addOperand(Op);
    Changed = true;
    CurBank = NewBank;
  }

  return Changed;
}

