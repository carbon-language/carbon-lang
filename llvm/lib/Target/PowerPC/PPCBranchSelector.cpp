//===-- PPCBranchSelector.cpp - Emit long conditional branches-----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Baegeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that scans a machine function to determine which
// conditional branches need more than 16 bits of displacement to reach their
// target basic block.  It does this in two passes; a calculation of basic block
// positions pass, and a branch psuedo op to machine branch opcode pass.  This
// pass should be run last, just before the assembly printer.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCInstrBuilder.h"
#include "PPCInstrInfo.h"
#include "PPCPredicates.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

static Statistic<> NumExpanded("ppc-branch-select",
                               "Num branches expanded to long format");

namespace {
  struct VISIBILITY_HIDDEN PPCBSel : public MachineFunctionPass {
    /// OffsetMap - Mapping between BB # and byte offset from start of function.
    std::vector<unsigned> OffsetMap;

    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "PowerPC Branch Selection";
    }
  };
}

/// createPPCBranchSelectionPass - returns an instance of the Branch Selection
/// Pass
///
FunctionPass *llvm::createPPCBranchSelectionPass() {
  return new PPCBSel();
}

/// getNumBytesForInstruction - Return the number of bytes of code the specified
/// instruction may be.  This returns the maximum number of bytes.
///
static unsigned getNumBytesForInstruction(MachineInstr *MI) {
  switch (MI->getOpcode()) {
  case PPC::BCC:
    // while this will be 4 most of the time, if we emit 8 it is just a
    // minor pessimization that saves us from having to worry about
    // keeping the offsets up to date later when we emit long branch glue.
    return 8;
  case PPC::IMPLICIT_DEF_GPRC: // no asm emitted
  case PPC::IMPLICIT_DEF_G8RC: // no asm emitted
  case PPC::IMPLICIT_DEF_F4:   // no asm emitted
  case PPC::IMPLICIT_DEF_F8:   // no asm emitted
  case PPC::IMPLICIT_DEF_VRRC: // no asm emitted
    return 0;
  case PPC::INLINEASM: {       // Inline Asm: Variable size.
    MachineFunction *MF = MI->getParent()->getParent();
    const char *AsmStr = MI->getOperand(0).getSymbolName();
    return MF->getTarget().getTargetAsmInfo()->getInlineAsmLength(AsmStr);
  }
  default:
    return 4; // PowerPC instructions are all 4 bytes
  }
}


bool PPCBSel::runOnMachineFunction(MachineFunction &Fn) {
  // Running total of instructions encountered since beginning of function
  unsigned ByteCount = 0;
  
  OffsetMap.resize(Fn.getNumBlockIDs());
  
  // For each MBB, add its offset to the offset map, and count up its
  // instructions
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock *MBB = MFI;
    OffsetMap[MBB->getNumber()] = ByteCount;
    
    for (MachineBasicBlock::iterator MBBI = MBB->begin(), EE = MBB->end();
         MBBI != EE; ++MBBI)
      ByteCount += getNumBytesForInstruction(MBBI);
  }
  
  // We're about to run over the MBB's again, so reset the ByteCount
  ByteCount = 0;
  
  // For each MBB, find the conditional branch pseudo instructions, and
  // calculate the difference between the target MBB and the current ICount
  // to decide whether or not to emit a short or long branch.
  //
  // short branch:
  // bCC .L_TARGET_MBB
  //
  // long branch:
  // bInverseCC $PC+8
  // b .L_TARGET_MBB
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock *MBB = MFI;
    
    for (MachineBasicBlock::iterator MBBI = MBB->begin(), EE = MBB->end();
         MBBI != EE; ++MBBI) {
      // We may end up deleting the MachineInstr that MBBI points to, so
      // remember its opcode now so we can refer to it after calling erase()
      unsigned ByteSize = getNumBytesForInstruction(MBBI);
      if (MBBI->getOpcode() != PPC::BCC) {
        ByteCount += ByteSize;
        continue;
      }
      
      // condbranch operands:
      // 0. CR register
      // 1. PPC branch opcode
      // 2. Target MBB
      MachineBasicBlock *DestMBB = MBBI->getOperand(2).getMachineBasicBlock();
      PPC::Predicate Pred = (PPC::Predicate)MBBI->getOperand(0).getImm();
      unsigned CRReg = MBBI->getOperand(1).getReg();
      int Displacement = OffsetMap[DestMBB->getNumber()] - ByteCount;

      bool ShortBranchOk = Displacement >= -32768 && Displacement <= 32767;
      
      // Branch on opposite condition if a short branch isn't ok.
      if (!ShortBranchOk)
        Pred = PPC::InvertPredicate(Pred);
        
      unsigned Opcode;
      switch (Pred) {
      default: assert(0 && "Unknown cond branch predicate!");
      case PPC::PRED_LT: Opcode = PPC::BLT; break;
      case PPC::PRED_LE: Opcode = PPC::BLE; break;
      case PPC::PRED_EQ: Opcode = PPC::BEQ; break;
      case PPC::PRED_GE: Opcode = PPC::BGE; break;
      case PPC::PRED_GT: Opcode = PPC::BGT; break;
      case PPC::PRED_NE: Opcode = PPC::BNE; break;
      case PPC::PRED_UN: Opcode = PPC::BUN; break;
      case PPC::PRED_NU: Opcode = PPC::BNU; break;
      }
      
      MachineBasicBlock::iterator MBBJ;
      if (ShortBranchOk) {
        MBBJ = BuildMI(*MBB, MBBI, Opcode, 2).addReg(CRReg).addMBB(DestMBB);
      } else {
        // Long branch, skip next branch instruction (i.e. $PC+8).
        ++NumExpanded;
        BuildMI(*MBB, MBBI, Opcode, 2).addReg(CRReg).addImm(2);
        MBBJ = BuildMI(*MBB, MBBI, PPC::B, 1).addMBB(DestMBB);
      }
      
      // Erase the psuedo BCC instruction, and then back up the
      // iterator so that when the for loop increments it, we end up in
      // the correct place rather than iterating off the end.
      MBB->erase(MBBI);
      MBBI = MBBJ;
      ByteCount += ByteSize;
    }
  }
  
  OffsetMap.clear();
  return true;
}

