//===--------- Definition of the HWAddressSanitizer class -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Hardware AddressSanitizer class which is a port of the
// legacy HWAddressSanitizer pass to use the new PassManager infrastructure.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_HWADDRESSSANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_HWADDRESSSANITIZER_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

struct HWAddressSanitizerOptions {
  HWAddressSanitizerOptions()
      : HWAddressSanitizerOptions(false, false, false){};
  HWAddressSanitizerOptions(bool CompileKernel, bool Recover,
                            bool DisableOptimization)
      : CompileKernel(CompileKernel), Recover(Recover),
        DisableOptimization(DisableOptimization){};
  bool CompileKernel;
  bool Recover;
  bool DisableOptimization;
};

/// This is a public interface to the hardware address sanitizer pass for
/// instrumenting code to check for various memory errors at runtime, similar to
/// AddressSanitizer but based on partial hardware assistance.
class HWAddressSanitizerPass : public PassInfoMixin<HWAddressSanitizerPass> {
public:
  explicit HWAddressSanitizerPass(HWAddressSanitizerOptions Options)
      : Options(Options){};
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static bool isRequired() { return true; }

private:
  HWAddressSanitizerOptions Options;
};

FunctionPass *
createHWAddressSanitizerLegacyPassPass(bool CompileKernel = false,
                                       bool Recover = false,
                                       bool DisableOptimization = false);

namespace HWASanAccessInfo {

// Bit field positions for the accessinfo parameter to
// llvm.hwasan.check.memaccess. Shared between the pass and the backend. Bits
// 0-15 are also used by the runtime.
enum {
  AccessSizeShift = 0, // 4 bits
  IsWriteShift = 4,
  RecoverShift = 5,
  MatchAllShift = 16, // 8 bits
  HasMatchAllShift = 24,
  CompileKernelShift = 25,
};

enum { RuntimeMask = 0xffff };

} // namespace HWASanAccessInfo

} // namespace llvm

#endif
