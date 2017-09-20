//===--- Passes/LongJmp.h -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_LONGJMP_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_LONGJMP_H

#include "BinaryPasses.h"

namespace llvm {
namespace bolt {

/// LongJmp is veneer-insertion pass originally written for AArch64 that
/// compensates for its short-range branches, typically done during linking. We
/// pull this pass inside BOLT because here we can do a better job at stub
/// inserting by manipulating the CFG, something linkers can't do.
///
/// LongJmp is a two-step process. In the first step, when function sizes are
/// still unknown because we can insert an arbitrary amount of code to reach
/// far-away code, this pass expands all PC-relative instructions that refer to
/// a symbol at an unknown location likely to violate the branch range.
/// This expansion inserts the equivalent of "linker stubs", small
/// blocks of code that load a 64-bit address into a pre-allocated register and
//  then executes an unconditional indirect branch on this register. By using a
/// 64-bit range, we guarantee it can reach any code location.
///
/// In the second step, we iteratively repeat the following until no
/// modification is done: we do a tentative layout with the current function
/// sizes; then we remove stubs for branches that we know are close enough to be
/// encoded in a direct branch or a smaller stub (32-bit).
///
/// Notice that this iteration is possible since step 2 strictly reduces sizes
/// and distances between branches and their destinations.
///
class LongJmpPass : public BinaryFunctionPass {
  using StubMapTy = DenseMap<const BinaryFunction *,
    DenseMap<const MCSymbol *, BinaryBasicBlock *>>;
  /// Used to quickly fetch stubs based on the target they jump to
  StubMapTy HotStubs;
  StubMapTy ColdStubs;

  /// Used to quickly identify whether a BB is a stub, sharded by function
  DenseMap<const BinaryFunction *, std::set<const BinaryBasicBlock *>> Stubs;

  using FuncAddressesMapTy = DenseMap<const BinaryFunction *, uint64_t>;
  /// Hold tentative addresses during step 2
  FuncAddressesMapTy HotAddresses;
  FuncAddressesMapTy ColdAddresses;
  DenseMap<const BinaryBasicBlock *, uint64_t> BBAddresses;

  /// Used to remove unused stubs
  DenseMap<const BinaryBasicBlock *, int> StubRefCount;
  /// Used to identify the stub size
  DenseMap<const BinaryBasicBlock *, int> StubBits;

  /// Replace the target of call or conditional branch in \p Inst with a
  /// a stub that in turn will branch to the target (perform stub insertion).
  /// If a new stub was created, return it.
  std::unique_ptr<BinaryBasicBlock>
  replaceTargetWithStub(const BinaryContext &BC, BinaryFunction &BF,
                        BinaryBasicBlock &BB, MCInst &Inst);

  ///                     -- Step 1 methods --
  /// Process all functions and insert maximum-size stubs so every branch in the
  /// program is encodable without violating relocation ranges (relax all
  /// branches).
  void insertStubs(const BinaryContext &BC, BinaryFunction &BF);

  ///                     -- Step 2 methods --
  /// Try to do layout before running the emitter, by looking at BinaryFunctions
  /// and MCInsts -- this is an estimation. To be correct for longjmp inserter
  /// purposes, we need to do a size worst-case estimation. Real layout is done
  /// by RewriteInstance::mapFileSections()
  void tentativeLayout(const BinaryContext &BC,
                       std::vector<BinaryFunction *> &SortedFunctions);
  uint64_t
  tentativeLayoutRelocMode(const BinaryContext &BC,
                           std::vector<BinaryFunction *> &SortedFunctions,
                           uint64_t DotAddress);
  void tentativeBBLayout(const BinaryContext &BC, const BinaryFunction &Func);

   /// Helper to identify whether \p Inst is branching to a stub
  bool usesStub(const BinaryContext &BC, const BinaryFunction &Func,
                const MCInst &Inst) const;

  /// Helper to resolve a symbol address according to our tentative layout
  uint64_t getSymbolAddress(const BinaryContext &BC, const MCSymbol *Target,
                            const BinaryBasicBlock *TgtBB) const;
  /// Change \p Inst to not use a stub anymore, back to its original form
  void removeStubRef(const BinaryContext &BC,
                     BinaryBasicBlock *BB, MCInst &Inst,
                     BinaryBasicBlock *StubBB,
                     const MCSymbol *Target, BinaryBasicBlock *TgtBB);

  /// Step 2 main entry point: Iterate through functions reducing stubs size
  /// or completely removing them.
  bool removeOrShrinkStubs(const BinaryContext &BC, BinaryFunction &BF);

public:
  /// BinaryPass public interface

  explicit LongJmpPass(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "long-jmp"; }

  void runOnFunctions(BinaryContext &BC,
                      std::map<uint64_t, BinaryFunction> &BFs,
                      std::set<uint64_t> &LargeFunctions) override;
};
}
}

#endif
