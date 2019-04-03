//===--- Passes/RetpolineInsertion.h ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_RETPOLINE_INSERTION_H
#define LLVM_TOOLS_LLVM_BOLT_RETPOLINE_INSERTION_H

#include "BinaryPasses.h"
#include "BinarySection.h"
#include <string>
#include <unordered_map>

namespace llvm {
namespace bolt {

struct IndirectBranchInfo {
private:
  bool IsMem = false;
  bool IsCall = false;
  bool IsTailCall = false;

public:
  IndirectBranchInfo(MCInst &Inst, MCPlusBuilder &MIB);
  bool isMem() const { return IsMem; }
  bool isReg() const { return !IsMem; }
  bool isCall() const { return IsCall; }
  bool isJump() const { return !IsCall; }
  bool isTailCall() const { return IsTailCall; }

  struct MemOpInfo {
    unsigned BaseRegNum;
    int64_t ScaleValue;
    unsigned IndexRegNum;
    int64_t DispValue;
    unsigned SegRegNum;
    const MCExpr *DispExpr{nullptr};
  };

  union {
    // Register branch information
    MCPhysReg BranchReg;

    // Memory branch information
    MemOpInfo Memory;
  };
};

class RetpolineInsertion : public BinaryFunctionPass {
private:
  std::unordered_map<std::string, BinaryFunction *> CreatedRetpolines;

  BinaryFunction *getOrCreateRetpoline(BinaryContext &BC,
                                       const IndirectBranchInfo &BrInfo,
                                       bool R11Available);

public:
  /// Register r11 availability options
  enum AvailabilityOptions : char {
    ALWAYS = 0, ///  r11 available before calls and jumps
    ABI = 1,    ///  r11 available before calls
    NEVER = 2   ///  r11 not available
  };

  explicit RetpolineInsertion(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "retpoline-insertion"; }

  void runOnFunctions(BinaryContext &BC) override;
};
} // namespace bolt
} // namespace llvm

#endif
