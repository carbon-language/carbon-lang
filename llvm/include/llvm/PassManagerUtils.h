//===-- llvm/Support/PassManagerUtils.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides interface to pass manager utilities.
//
//===----------------------------------------------------------------------===//

namespace llvm {

class FunctionPassManager;
class PassManager;

/// AddOptimizationPasses - This routine adds optimization passes 
/// based on selected optimization level, OptLevel. This routine is
/// used by llvm-gcc and other tools.
///
/// OptLevel - Optimization Level
/// EnableIPO - Enables IPO passes. llvm-gcc enables this when
///             flag_unit_at_a_time is set.
/// InlinerSelection - 1 : Add function inliner.
///                  - 2 : Add AlwaysInliner.
/// OptLibCalls - Simplify lib calls, if set.
/// PruneEH - Add PruneEHPass, if set.
/// UnrollLoop - Unroll loops, if set.
void AddOptimizationPasses(FunctionPassManager &FPM, PassManager &MPM,
                           unsigned OptLevel, bool EnableIPO,
                           unsigned InlinerSelection, bool OptLibCalls,
                           bool PruneEH, bool UnrollLoop);

}
