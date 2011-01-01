//===-- PTXMFInfoExtract.cpp - Extract PTX machine function info ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an information extractor for PTX machine functions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ptx-mf-info-extract"

#include "PTX.h"
#include "PTXTargetMachine.h"
#include "PTXMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

// NOTE: PTXMFInfoExtract must after register allocation!

namespace llvm {
  /// PTXMFInfoExtract - PTX specific code to extract of PTX machine
  /// function information for PTXAsmPrinter
  ///
  class PTXMFInfoExtract : public MachineFunctionPass {
    private:
      static char ID;

    public:
      PTXMFInfoExtract(PTXTargetMachine &TM, CodeGenOpt::Level OptLevel)
        : MachineFunctionPass(ID) {}

      virtual bool runOnMachineFunction(MachineFunction &MF);

      virtual const char *getPassName() const {
        return "PTX Machine Function Info Extractor";
      }
  }; // class PTXMFInfoExtract
} // namespace llvm

using namespace llvm;

char PTXMFInfoExtract::ID = 0;

bool PTXMFInfoExtract::runOnMachineFunction(MachineFunction &MF) {
  PTXMachineFunctionInfo *MFI = MF.getInfo<PTXMachineFunctionInfo>();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  DEBUG(dbgs() << "******** PTX FUNCTION LOCAL VAR REG DEF ********\n");

  unsigned retreg = MFI->retReg();

  DEBUG(dbgs()
        << "PTX::NoRegister == " << PTX::NoRegister << "\n"
        << "PTX::NUM_TARGET_REGS == " << PTX::NUM_TARGET_REGS << "\n");

  DEBUG(for (unsigned reg = PTX::NoRegister + 1;
             reg < PTX::NUM_TARGET_REGS; ++reg)
          if (MRI.isPhysRegUsed(reg))
            dbgs() << "Used Reg: " << reg << "\n";);

  // FIXME: This is a slow linear scanning
  for (unsigned reg = PTX::NoRegister + 1; reg < PTX::NUM_TARGET_REGS; ++reg)
    if (MRI.isPhysRegUsed(reg) && reg != retreg && !MFI->isArgReg(reg))
      MFI->addLocalVarReg(reg);

  // Notify MachineFunctionInfo that I've done adding local var reg
  MFI->doneAddLocalVar();

  DEBUG(dbgs() << "Return Reg: " << retreg << "\n");

  DEBUG(for (PTXMachineFunctionInfo::reg_iterator
             i = MFI->argRegBegin(), e = MFI->argRegEnd();
	     i != e; ++i)
        dbgs() << "Arg Reg: " << *i << "\n";);

  DEBUG(for (PTXMachineFunctionInfo::reg_iterator
             i = MFI->localVarRegBegin(), e = MFI->localVarRegEnd();
	     i != e; ++i)
        dbgs() << "Local Var Reg: " << *i << "\n";);

  return false;
}

FunctionPass *llvm::createPTXMFInfoExtract(PTXTargetMachine &TM,
                                           CodeGenOpt::Level OptLevel) {
  return new PTXMFInfoExtract(TM, OptLevel);
}
