//===-- PowerPCBranchSelector.cpp - Emit long conditional branches-*- C++ -*-=//
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

#define DEBUG_TYPE "bsel"
#include "PowerPC.h"
#include "PowerPCInstrBuilder.h"
#include "PowerPCInstrInfo.h"
#include "PPC32InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/Debug.h"
#include <map>
using namespace llvm;

namespace {
  struct BSel : public MachineFunctionPass {
    // OffsetMap - Mapping between BB and byte offset from start of function
    std::map<MachineBasicBlock*, unsigned> OffsetMap;

    /// bytesForOpcode - A convenience function for totalling up the number of
    /// bytes in a basic block.
    ///
    static unsigned bytesForOpcode(unsigned opcode) {
      switch (opcode) {
      case PPC::COND_BRANCH:
        // while this will be 4 most of the time, if we emit 12 it is just a
        // minor pessimization that saves us from having to worry about
        // keeping the offsets up to date later when we emit long branch glue.
        return 12;
      case PPC::IMPLICIT_DEF_GPR: // no asm emitted
      case PPC::IMPLICIT_DEF_FP: // no asm emitted
        return 0;
      default:
        break;
      }
      return 4; // PowerPC instructions are all 4 bytes
    }

    virtual bool runOnMachineFunction(MachineFunction &Fn) {
      // Running total of instructions encountered since beginning of function
      unsigned ByteCount = 0;

      // For each MBB, add its offset to the offset map, and count up its
      // instructions
      for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
           ++MFI) {
        MachineBasicBlock *MBB = MFI;
        OffsetMap[MBB] = ByteCount;

        for (MachineBasicBlock::iterator MBBI = MBB->begin(), EE = MBB->end();
             MBBI != EE; ++MBBI)
          ByteCount += bytesForOpcode(MBBI->getOpcode());
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
      // b .L_FALLTHROUGH_MBB

      for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
           ++MFI) {
        MachineBasicBlock *MBB = MFI;

        for (MachineBasicBlock::iterator MBBI = MBB->begin(), EE = MBB->end();
             MBBI != EE; ++MBBI) {
          // We may end up deleting the MachineInstr that MBBI points to, so
          // remember its opcode now so we can refer to it after calling erase()
          unsigned OpcodeToReplace = MBBI->getOpcode();

          if (OpcodeToReplace == PPC::COND_BRANCH) {
            MachineBasicBlock::iterator MBBJ = MBBI;
            ++MBBJ;

            // condbranch operands:
            // 0. CR0 register
            // 1. bc opcode
            // 2. target MBB
            // 3. fallthrough MBB
            MachineBasicBlock *trueMBB =
              MBBI->getOperand(2).getMachineBasicBlock();
            MachineBasicBlock *falseMBB =
              MBBI->getOperand(3).getMachineBasicBlock();

            int Displacement = OffsetMap[trueMBB] - ByteCount;
            unsigned Opcode = MBBI->getOperand(1).getImmedValue();
            unsigned Inverted = PPC32InstrInfo::invertPPCBranchOpcode(Opcode);

            if (Displacement >= -32768 && Displacement <= 32767) {
              BuildMI(*MBB, MBBJ, Opcode, 2).addReg(PPC::CR0).addMBB(trueMBB);
            } else {
              BuildMI(*MBB, MBBJ, Inverted, 2).addReg(PPC::CR0).addSImm(8);
              BuildMI(*MBB, MBBJ, PPC::B, 1).addMBB(trueMBB);
              BuildMI(*MBB, MBBJ, PPC::B, 1).addMBB(falseMBB);
            }

            // Erase the psuedo COND_BRANCH instruction, and then back up the
            // iterator so that when the for loop increments it, we end up in
            // the correct place rather than iterating off the end.
            MBB->erase(MBBI);
            MBBI = --MBBJ;
          }
          ByteCount += bytesForOpcode(OpcodeToReplace);
        }
      }

      OffsetMap.clear();
      return true;
    }

    virtual const char *getPassName() const {
      return "PowerPC Branch Selection";
    }
  };
}

/// createPPCBranchSelectionPass - returns an instance of the Branch Selection
/// Pass
///
FunctionPass *llvm::createPPCBranchSelectionPass() {
  return new BSel();
}
