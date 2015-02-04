//===---------- PPCTLSDynamicCall.cpp - TLS Dynamic Call Fixup ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass fixes up GETtls[ld]ADDR[32] machine instructions so that
// they read and write GPR3.  These are really call instructions, so
// must use the calling convention registers.  This is done in a late
// pass so that TLS variable accesses can be fully commoned.
//
//===----------------------------------------------------------------------===//

#include "PPCInstrInfo.h"
#include "PPC.h"
#include "PPCInstrBuilder.h"
#include "PPCTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "ppc-tls-dynamic-call"

namespace llvm {
  void initializePPCTLSDynamicCallPass(PassRegistry&);
}

namespace {
  // PPCTLSDynamicCall pass - Add copies to and from GPR3 around
  // GETtls[ld]ADDR[32] machine instructions.  These instructions
  // are actually call instructions, so the register choice is
  // constrained.  We delay introducing these copies as late as
  // possible so that TLS variable accesses can be fully commoned.
  struct PPCTLSDynamicCall : public MachineFunctionPass {
    static char ID;
    PPCTLSDynamicCall() : MachineFunctionPass(ID) {
      initializePPCTLSDynamicCallPass(*PassRegistry::getPassRegistry());
    }

    const PPCTargetMachine *TM;
    const PPCInstrInfo *TII;

protected:
    bool processBlock(MachineBasicBlock &MBB) {
      bool Changed = false;
      bool Is64Bit = TM->getSubtargetImpl()->isPPC64();

      for (MachineBasicBlock::iterator I = MBB.begin(), IE = MBB.end();
           I != IE; ++I) {
        MachineInstr *MI = I;

        if (MI->getOpcode() != PPC::GETtlsADDR &&
            MI->getOpcode() != PPC::GETtlsldADDR &&
            MI->getOpcode() != PPC::GETtlsADDR32 &&
            MI->getOpcode() != PPC::GETtlsldADDR32)
          continue;

        DEBUG(dbgs() << "TLS Dynamic Call Fixup:\n    " << *MI;);

        unsigned OutReg = MI->getOperand(0).getReg();
        unsigned InReg  = MI->getOperand(1).getReg();
        DebugLoc DL = MI->getDebugLoc();
        unsigned GPR3 = Is64Bit ? PPC::X3 : PPC::R3;

        BuildMI(MBB, I, DL, TII->get(TargetOpcode::COPY), GPR3)
          .addReg(InReg);
        MI->getOperand(0).setReg(GPR3);
        MI->getOperand(1).setReg(GPR3);
        BuildMI(MBB, ++I, DL, TII->get(TargetOpcode::COPY), OutReg)
          .addReg(GPR3);

        Changed = true;
      }

      return Changed;
    }

public:
    bool runOnMachineFunction(MachineFunction &MF) override {
      TM = static_cast<const PPCTargetMachine *>(&MF.getTarget());
      TII = TM->getSubtargetImpl()->getInstrInfo();

      bool Changed = false;

      for (MachineFunction::iterator I = MF.begin(); I != MF.end();) {
        MachineBasicBlock &B = *I++;
        if (processBlock(B))
          Changed = true;
      }

      return Changed;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
}

INITIALIZE_PASS_BEGIN(PPCTLSDynamicCall, DEBUG_TYPE,
                      "PowerPC TLS Dynamic Call Fixup", false, false)
INITIALIZE_PASS_END(PPCTLSDynamicCall, DEBUG_TYPE,
                    "PowerPC TLS Dynamic Call Fixup", false, false)

char PPCTLSDynamicCall::ID = 0;
FunctionPass*
llvm::createPPCTLSDynamicCallPass() { return new PPCTLSDynamicCall(); }
