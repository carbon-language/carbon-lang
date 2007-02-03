//===-- PPCMachOWriter.cpp - Emit a Mach-O file for the PowerPC backend ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a Mach-O writer for the PowerPC backend.  The public
// interface to this file is the createPPCMachOObjectWriterPass function.
//
//===----------------------------------------------------------------------===//

#include "PPCTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachOWriter.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

namespace {
  struct VISIBILITY_HIDDEN PPCMachOWriter : public MachOWriter {
    PPCMachOWriter(std::ostream &O, PPCTargetMachine &TM)
      : MachOWriter(O, TM) {}
    };
}

/// addPPCMachOObjectWriterPass - Returns a pass that outputs the generated code
/// as a Mach-O object file.
///
void llvm::addPPCMachOObjectWriterPass(FunctionPassManager &FPM,
                                       std::ostream &O, PPCTargetMachine &TM) {
  PPCMachOWriter *MOW = new PPCMachOWriter(O, TM);
  FPM.add(MOW);
  FPM.add(createPPCCodeEmitterPass(TM, MOW->getMachineCodeEmitter()));
}
