//===-- llvm/Target/TargetMachineImpls.h - Target Descriptions --*- C++ -*-===//
//
// This file defines the entry point to getting access to the various target
// machine implementations available to LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETMACHINEIMPLS_H
#define LLVM_TARGET_TARGETMACHINEIMPLS_H

namespace TM {
  enum {
    PtrSizeMask  = 1,
    PtrSize32    = 0,
    PtrSize64    = 1,

    EndianMask   = 2,
    LittleEndian = 0,
    BigEndian    = 2,
  };
}

class TargetMachine;

// allocateSparcTargetMachine - Allocate and return a subclass of TargetMachine
// that implements the Sparc backend.
//
TargetMachine *allocateSparcTargetMachine(unsigned Configuration =
                                          TM::PtrSize64|TM::BigEndian);

// allocateX86TargetMachine - Allocate and return a subclass of TargetMachine
// that implements the X86 backend.  The X86 target machine can run in
// "emulation" mode, where it is capable of emulating machines of larger pointer
// size and different endianness if desired.
//
TargetMachine *allocateX86TargetMachine(unsigned Configuration =
                                        TM::PtrSize32|TM::LittleEndian);

#endif
