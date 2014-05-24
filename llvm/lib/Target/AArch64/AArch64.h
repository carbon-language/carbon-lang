//==-- AArch64.h - Top-level interface for AArch64  --------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// AArch64 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_AArch64_H
#define TARGET_AArch64_H

#include "Utils/AArch64BaseInfo.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class AArch64TargetMachine;
class FunctionPass;
class MachineFunctionPass;

FunctionPass *createAArch64DeadRegisterDefinitions();
FunctionPass *createAArch64ConditionalCompares();
FunctionPass *createAArch64AdvSIMDScalar();
FunctionPass *createAArch64BranchRelaxation();
FunctionPass *createAArch64ISelDag(AArch64TargetMachine &TM,
                                 CodeGenOpt::Level OptLevel);
FunctionPass *createAArch64StorePairSuppressPass();
FunctionPass *createAArch64ExpandPseudoPass();
FunctionPass *createAArch64LoadStoreOptimizationPass();
ModulePass *createAArch64PromoteConstantPass();
FunctionPass *createAArch64AddressTypePromotionPass();
/// \brief Creates an ARM-specific Target Transformation Info pass.
ImmutablePass *
createAArch64TargetTransformInfoPass(const AArch64TargetMachine *TM);

FunctionPass *createAArch64CleanupLocalDynamicTLSPass();

FunctionPass *createAArch64CollectLOHPass();
} // end namespace llvm

#endif
