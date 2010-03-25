//===- SSEDomainFix.cpp - Use proper int/float domain for SSE ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SSEDomainFix pass.
//
// Some SSE instructions like mov, and, or, xor are available in different
// variants for different operand types. These variant instructions are
// equivalent, but on Nehalem and newer cpus there is extra latency
// transferring data between integer and floating point domains.
//
// This pass changes the variant instructions to minimize domain crossings.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sse-domain-fix"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class SSEDomainFixPass : public MachineFunctionPass {
  static char ID;
  const X86InstrInfo *TII;

  MachineFunction *MF;
  MachineBasicBlock *MBB;
public:
  SSEDomainFixPass() : MachineFunctionPass(&ID) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  virtual const char *getPassName() const {
    return "SSE execution domain fixup";
  }

private:
  void enterBasicBlock(MachineBasicBlock *MBB);
};
}

char SSEDomainFixPass::ID = 0;

void SSEDomainFixPass::enterBasicBlock(MachineBasicBlock *mbb) {
  MBB = mbb;
  DEBUG(dbgs() << "Entering MBB " << MBB->getName() << "\n");
}

bool SSEDomainFixPass::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  TII = static_cast<const X86InstrInfo*>(MF->getTarget().getInstrInfo());

  // If no XMM registers are used in the function, we can skip it completely.
  bool XMMIsUsed = false;
  for (TargetRegisterClass::const_iterator I = X86::VR128RegClass.begin(),
         E = X86::VR128RegClass.end(); I != E; ++I)
    if (MF->getRegInfo().isPhysRegUsed(*I)) {
      XMMIsUsed = true;
      break;
    }
  if (!XMMIsUsed) return false;

  MachineBasicBlock *Entry = MF->begin();
  SmallPtrSet<MachineBasicBlock*, 16> Visited;
  for (df_ext_iterator<MachineBasicBlock*,
         SmallPtrSet<MachineBasicBlock*, 16> >
         DFI = df_ext_begin(Entry, Visited), DFE = df_ext_end(Entry, Visited);
       DFI != DFE; ++DFI) {
    enterBasicBlock(*DFI);
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
        ++I) {
      MachineInstr *MI = I;
      const unsigned *equiv = 0;
      X86InstrInfo::SSEDomain domain = TII->GetSSEDomain(MI, equiv);
      (void) domain;
      DEBUG(dbgs() << "-isd"[domain] << (equiv ? "* " : "  ") << *MI);
    }
  }
  return false;
}

FunctionPass *llvm::createSSEDomainFixPass() {
  return new SSEDomainFixPass();
}
