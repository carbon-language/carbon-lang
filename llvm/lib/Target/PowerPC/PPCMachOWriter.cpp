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
#include "llvm/Support/Visibility.h"
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN PPCMachOWriter : public MachOWriter {
  public:
    PPCMachOWriter(std::ostream &O, PPCTargetMachine &TM) : MachOWriter(O, TM) {
      // FIMXE: choose ppc64 when appropriate
      Header.cputype = MachOHeader::CPU_TYPE_POWERPC;
      Header.cpusubtype = MachOHeader::CPU_SUBTYPE_POWERPC_ALL;
    }

  };
}

/// addPPCMachOObjectWriterPass - Returns a pass that outputs the generated code
/// as a Mach-O object file.
///
void llvm::addPPCMachOObjectWriterPass(PassManager &FPM,
                                       std::ostream &O, PPCTargetMachine &TM) {
  PPCMachOWriter *EW = new PPCMachOWriter(O, TM);
  FPM.add(EW);
  FPM.add(createPPCCodeEmitterPass(TM, EW->getMachineCodeEmitter()));
}
