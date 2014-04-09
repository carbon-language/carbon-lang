//===-- ARM64.h - Top-level interface for ARM64 representation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// ARM64 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_ARM64_H
#define TARGET_ARM64_H

#include "Utils/ARM64BaseInfo.h"
#include "MCTargetDesc/ARM64MCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class ARM64TargetMachine;
class FunctionPass;
class MachineFunctionPass;

FunctionPass *createARM64DeadRegisterDefinitions();
FunctionPass *createARM64ConditionalCompares();
FunctionPass *createARM64AdvSIMDScalar();
FunctionPass *createARM64BranchRelaxation();
FunctionPass *createARM64ISelDag(ARM64TargetMachine &TM,
                                 CodeGenOpt::Level OptLevel);
FunctionPass *createARM64StorePairSuppressPass();
FunctionPass *createARM64ExpandPseudoPass();
FunctionPass *createARM64LoadStoreOptimizationPass();
ModulePass *createARM64PromoteConstantPass();
FunctionPass *createARM64AddressTypePromotionPass();
/// \brief Creates an ARM-specific Target Transformation Info pass.
ImmutablePass *createARM64TargetTransformInfoPass(const ARM64TargetMachine *TM);

FunctionPass *createARM64CleanupLocalDynamicTLSPass();

FunctionPass *createARM64CollectLOHPass();
} // end namespace llvm

#endif
