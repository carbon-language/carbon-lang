//===-- AMDGPU.h - MachineFunction passes hw codegen --------------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef AMDGPU_H
#define AMDGPU_H

#include "AMDGPUTargetMachine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class FunctionPass;
class AMDGPUTargetMachine;

// R600 Passes
FunctionPass* createR600KernelParametersPass(const DataLayout *TD);
FunctionPass *createR600ExpandSpecialInstrsPass(TargetMachine &tm);
FunctionPass *createR600EmitClauseMarkers(TargetMachine &tm);

// SI Passes
FunctionPass *createSIAnnotateControlFlowPass();
FunctionPass *createSILowerControlFlowPass(TargetMachine &tm);
FunctionPass *createSICodeEmitterPass(formatted_raw_ostream &OS);
FunctionPass *createSIInsertWaits(TargetMachine &tm);

// Passes common to R600 and SI
Pass *createAMDGPUStructurizeCFGPass();
FunctionPass *createAMDGPUConvertToISAPass(TargetMachine &tm);
FunctionPass* createAMDGPUIndirectAddressingPass(TargetMachine &tm);

} // End namespace llvm

namespace ShaderType {
  enum Type {
    PIXEL = 0,
    VERTEX = 1,
    GEOMETRY = 2,
    COMPUTE = 3
  };
}

#endif // AMDGPU_H
