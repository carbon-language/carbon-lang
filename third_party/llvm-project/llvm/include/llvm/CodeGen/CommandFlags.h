//===-- CommandFlags.h - Command Line Flags Interface -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains codegen-specific flags that are shared between different
// command line tools. The tools "llc" and "opt" both use this file to prevent
// flag duplication.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_COMMANDFLAGS_H
#define LLVM_CODEGEN_COMMANDFLAGS_H

#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetOptions.h"
#include <string>
#include <vector>

namespace llvm {

class Module;
class AttrBuilder;
class Function;
class Triple;

namespace codegen {

std::string getMArch();

std::string getMCPU();

std::vector<std::string> getMAttrs();

Reloc::Model getRelocModel();
Optional<Reloc::Model> getExplicitRelocModel();

ThreadModel::Model getThreadModel();

CodeModel::Model getCodeModel();
Optional<CodeModel::Model> getExplicitCodeModel();

llvm::ExceptionHandling getExceptionModel();

Optional<CodeGenFileType> getExplicitFileType();

CodeGenFileType getFileType();

FramePointerKind getFramePointerUsage();

bool getEnableUnsafeFPMath();

bool getEnableNoInfsFPMath();

bool getEnableNoNaNsFPMath();

bool getEnableNoSignedZerosFPMath();

bool getEnableApproxFuncFPMath();

bool getEnableNoTrappingFPMath();

DenormalMode::DenormalModeKind getDenormalFPMath();
DenormalMode::DenormalModeKind getDenormalFP32Math();

bool getEnableHonorSignDependentRoundingFPMath();

llvm::FloatABI::ABIType getFloatABIForCalls();

llvm::FPOpFusion::FPOpFusionMode getFuseFPOps();

SwiftAsyncFramePointerMode getSwiftAsyncFramePointer();

bool getDontPlaceZerosInBSS();

bool getEnableGuaranteedTailCallOpt();

bool getEnableAIXExtendedAltivecABI();

bool getDisableTailCalls();

bool getStackSymbolOrdering();

unsigned getOverrideStackAlignment();

bool getStackRealign();

std::string getTrapFuncName();

bool getUseCtors();

bool getLowerGlobalDtorsViaCxaAtExit();

bool getRelaxELFRelocations();

bool getDataSections();
Optional<bool> getExplicitDataSections();

bool getFunctionSections();
Optional<bool> getExplicitFunctionSections();

bool getIgnoreXCOFFVisibility();

bool getXCOFFTracebackTable();

std::string getBBSections();

unsigned getTLSSize();

bool getEmulatedTLS();

bool getUniqueSectionNames();

bool getUniqueBasicBlockSectionNames();

llvm::EABI getEABIVersion();

llvm::DebuggerKind getDebuggerTuningOpt();

bool getEnableStackSizeSection();

bool getEnableAddrsig();

bool getEmitCallSiteInfo();

bool getEnableMachineFunctionSplitter();

bool getEnableDebugEntryValues();

bool getValueTrackingVariableLocations();
Optional<bool> getExplicitValueTrackingVariableLocations();

bool getForceDwarfFrameSection();

bool getXRayOmitFunctionIndex();

bool getDebugStrictDwarf();

unsigned getAlignLoops();

bool getJMCInstrument();

/// Create this object with static storage to register codegen-related command
/// line options.
struct RegisterCodeGenFlags {
  RegisterCodeGenFlags();
};

llvm::BasicBlockSection getBBSectionsMode(llvm::TargetOptions &Options);

/// Common utility function tightly tied to the options listed here. Initializes
/// a TargetOptions object with CodeGen flags and returns it.
/// \p TheTriple is used to determine the default value for options if
///    options are not explicitly specified. If those triple dependant options
///    value do not have effect for your component, a default Triple() could be
///    passed in.
TargetOptions InitTargetOptionsFromCodeGenFlags(const llvm::Triple &TheTriple);

std::string getCPUStr();

std::string getFeaturesStr();

std::vector<std::string> getFeatureList();

void renderBoolStringAttr(AttrBuilder &B, StringRef Name, bool Val);

/// Set function attributes of function \p F based on CPU, Features, and command
/// line flags.
void setFunctionAttributes(StringRef CPU, StringRef Features, Function &F);

/// Set function attributes of functions in Module M based on CPU,
/// Features, and command line flags.
void setFunctionAttributes(StringRef CPU, StringRef Features, Module &M);

/// Should value-tracking variable locations / instruction referencing be
/// enabled by default for this triple?
bool getDefaultValueTrackingVariableLocations(const llvm::Triple &T);
} // namespace codegen
} // namespace llvm

#endif // LLVM_CODEGEN_COMMANDFLAGS_H
