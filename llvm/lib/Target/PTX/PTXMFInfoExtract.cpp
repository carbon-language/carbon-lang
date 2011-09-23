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

  // Generate list of all virtual registers used in this function
  for (unsigned i = 0; i < MRI.getNumVirtRegs(); ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    const TargetRegisterClass *TRC = MRI.getRegClass(Reg);
    MFI->addVirtualRegister(TRC, Reg);
  }

  return false;
}

FunctionPass *llvm::createPTXMFInfoExtract(PTXTargetMachine &TM,
                                           CodeGenOpt::Level OptLevel) {
  return new PTXMFInfoExtract(TM, OptLevel);
}
