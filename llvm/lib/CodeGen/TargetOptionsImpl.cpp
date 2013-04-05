//===-- TargetOptionsImpl.cpp - Options that apply to all targets ----------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the methods in the TargetOptions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

/// DisableFramePointerElim - This returns true if frame pointer elimination
/// optimization should be disabled for the given machine function.
bool TargetOptions::DisableFramePointerElim(const MachineFunction &MF) const {
  // Check to see if we should eliminate non-leaf frame pointers and then
  // check to see if we should eliminate all frame pointers.
  if (NoFramePointerElimNonLeaf && !NoFramePointerElim) {
    const MachineFrameInfo *MFI = MF.getFrameInfo();
    return MFI->hasCalls();
  }

  return NoFramePointerElim;
}

/// LessPreciseFPMAD - This flag return true when -enable-fp-mad option
/// is specified on the command line.  When this flag is off(default), the
/// code generator is not allowed to generate mad (multiply add) if the
/// result is "less precise" than doing those operations individually.
bool TargetOptions::LessPreciseFPMAD() const {
  return UnsafeFPMath || LessPreciseFPMADOption;
}

/// HonorSignDependentRoundingFPMath - Return true if the codegen must assume
/// that the rounding mode of the FPU can change from its default.
bool TargetOptions::HonorSignDependentRoundingFPMath() const {
  return !UnsafeFPMath && HonorSignDependentRoundingFPMathOption;
}

/// getTrapFunctionName - If this returns a non-empty string, this means isel
/// should lower Intrinsic::trap to a call to the specified function name
/// instead of an ISD::TRAP node.
StringRef TargetOptions::getTrapFunctionName() const {
  return TrapFuncName;
}

bool TargetOptions::operator==(const TargetOptions &TO) {
#define ARE_EQUAL(X) X == TO.X
  return
    ARE_EQUAL(UnsafeFPMath) &&
    ARE_EQUAL(NoInfsFPMath) &&
    ARE_EQUAL(NoNaNsFPMath) &&
    ARE_EQUAL(HonorSignDependentRoundingFPMathOption) &&
    ARE_EQUAL(UseSoftFloat) &&
    ARE_EQUAL(NoZerosInBSS) &&
    ARE_EQUAL(JITExceptionHandling) &&
    ARE_EQUAL(JITEmitDebugInfo) &&
    ARE_EQUAL(JITEmitDebugInfoToDisk) &&
    ARE_EQUAL(GuaranteedTailCallOpt) &&
    ARE_EQUAL(DisableTailCalls) &&
    ARE_EQUAL(StackAlignmentOverride) &&
    ARE_EQUAL(RealignStack) &&
    ARE_EQUAL(SSPBufferSize) &&
    ARE_EQUAL(EnableFastISel) &&
    ARE_EQUAL(PositionIndependentExecutable) &&
    ARE_EQUAL(EnableSegmentedStacks) &&
    ARE_EQUAL(UseInitArray) &&
    ARE_EQUAL(TrapFuncName) &&
    ARE_EQUAL(FloatABIType) &&
    ARE_EQUAL(AllowFPOpFusion);
#undef ARE_EQUAL
}
