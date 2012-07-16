//===-- AMDGPUUtil.h - AMDGPU Utility function declarations -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Declarations for utility functions common to all hw codegen targets.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPU_UTIL_H
#define AMDGPU_UTIL_H

namespace llvm {

class MachineFunction;
class MachineRegisterInfo;
class TargetInstrInfo;

namespace AMDGPU {

bool isPlaceHolderOpcode(unsigned opcode);

bool isTransOp(unsigned opcode);
bool isTexOp(unsigned opcode);
bool isReductionOp(unsigned opcode);
bool isCubeOp(unsigned opcode);
bool isFCOp(unsigned opcode);

// XXX: Move these to AMDGPUInstrInfo.h
#define MO_FLAG_CLAMP (1 << 0)
#define MO_FLAG_NEG   (1 << 1)
#define MO_FLAG_ABS   (1 << 2)
#define MO_FLAG_MASK  (1 << 3)

void utilAddLiveIn(MachineFunction * MF, MachineRegisterInfo & MRI,
    const TargetInstrInfo * TII, unsigned physReg, unsigned virtReg);

} // End namespace AMDGPU

} // End namespace llvm

#endif // AMDGPU_UTIL_H
