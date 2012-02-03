//===-- Passes.h - Target independent code generation passes ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code generation
// passes provided by the LLVM backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PASSES_H
#define LLVM_CODEGEN_PASSES_H

#include "llvm/Target/TargetMachine.h"
#include <string>

namespace llvm {

  class FunctionPass;
  class MachineFunctionPass;
  class PassInfo;
  class TargetLowering;
  class TargetRegisterClass;
  class raw_ostream;
}

namespace llvm {

/// Target-Independent Code Generator Pass Configuration Options.
///
/// FIXME: Why are we passing the DisableVerify flags around instead of setting
/// an options in the target machine, like all the other driver options?
class TargetPassConfig {
protected:
  TargetMachine *TM;
  PassManagerBase &PM;
  bool DisableVerify;

public:
  TargetPassConfig(TargetMachine *tm, PassManagerBase &pm,
                   bool DisableVerifyFlag)
    : TM(tm), PM(pm), DisableVerify(DisableVerifyFlag) {}

  virtual ~TargetPassConfig() {}

  /// Get the right type of TargetMachine for this target.
  template<typename TMC> TMC &getTM() const {
    return *static_cast<TMC*>(TM);
  }

  CodeGenOpt::Level getOptLevel() const { return TM->getOptLevel(); }

  const TargetLowering *getTargetLowering() const { return TM->getTargetLowering(); }

  /// Add the complete, standard set of LLVM CodeGen passes.
  /// Fully developed targets will not generally override this.
  virtual bool addCodeGenPasses(MCContext *&OutContext);

protected:
  /// Convenient points in the common codegen pass pipeline for inserting
  /// passes, and major CodeGen stages that some targets may override.
  ///

  /// addPreISelPasses - This method should add any "last minute" LLVM->LLVM
  /// passes (which are run just before instruction selector).
  virtual bool addPreISel() {
    return true;
  }

  /// addInstSelector - This method should install an instruction selector pass,
  /// which converts from LLVM code to machine instructions.
  virtual bool addInstSelector() {
    return true;
  }

  /// addPreRegAlloc - This method may be implemented by targets that want to
  /// run passes immediately before register allocation. This should return
  /// true if -print-machineinstrs should print after these passes.
  virtual bool addPreRegAlloc() {
    return false;
  }

  /// addPostRegAlloc - This method may be implemented by targets that want
  /// to run passes after register allocation but before prolog-epilog
  /// insertion.  This should return true if -print-machineinstrs should print
  /// after these passes.
  virtual bool addPostRegAlloc() {
    return false;
  }

  /// getEnableTailMergeDefault - the default setting for -enable-tail-merge
  /// on this target.  User flag overrides.
  virtual bool getEnableTailMergeDefault() const { return true; }

  /// addPreSched2 - This method may be implemented by targets that want to
  /// run passes after prolog-epilog insertion and before the second instruction
  /// scheduling pass.  This should return true if -print-machineinstrs should
  /// print after these passes.
  virtual bool addPreSched2() {
    return false;
  }

  /// addPreEmitPass - This pass may be implemented by targets that want to run
  /// passes immediately before machine code is emitted.  This should return
  /// true if -print-machineinstrs should print out the code after the passes.
  virtual bool addPreEmitPass() {
    return false;
  }

  /// Utilities for targets to add passes to the pass manager.
  ///

  /// Add a target-independent CodeGen pass at this point in the pipeline.
  void addCommonPass(char &ID);

  /// printNoVerify - Add a pass to dump the machine function, if debugging is
  /// enabled.
  ///
  void printNoVerify(const char *Banner) const;

  /// printAndVerify - Add a pass to dump then verify the machine function, if
  /// those steps are enabled.
  ///
  void printAndVerify(const char *Banner) const;
};
} // namespace llvm

/// List of target independent CodeGen pass IDs.
namespace llvm {
  /// createUnreachableBlockEliminationPass - The LLVM code generator does not
  /// work well with unreachable basic blocks (what live ranges make sense for a
  /// block that cannot be reached?).  As such, a code generator should either
  /// not instruction select unreachable blocks, or run this pass as its
  /// last LLVM modifying pass to clean up blocks that are not reachable from
  /// the entry block.
  FunctionPass *createUnreachableBlockEliminationPass();

  /// MachineFunctionPrinter pass - This pass prints out the machine function to
  /// the given stream as a debugging tool.
  MachineFunctionPass *
  createMachineFunctionPrinterPass(raw_ostream &OS,
                                   const std::string &Banner ="");

  /// MachineLoopInfo pass - This pass is a loop analysis pass.
  ///
  extern char &MachineLoopInfoID;

  /// MachineLoopRanges pass - This pass is an on-demand loop coverage
  /// analysis pass.
  ///
  extern char &MachineLoopRangesID;

  /// MachineDominators pass - This pass is a machine dominators analysis pass.
  ///
  extern char &MachineDominatorsID;

  /// EdgeBundles analysis - Bundle machine CFG edges.
  ///
  extern char &EdgeBundlesID;

  /// PHIElimination pass - This pass eliminates machine instruction PHI nodes
  /// by inserting copy instructions.  This destroys SSA information, but is the
  /// desired input for some register allocators.  This pass is "required" by
  /// these register allocator like this: AU.addRequiredID(PHIEliminationID);
  ///
  extern char &PHIEliminationID;

  /// StrongPHIElimination pass - This pass eliminates machine instruction PHI
  /// nodes by inserting copy instructions.  This destroys SSA information, but
  /// is the desired input for some register allocators.  This pass is
  /// "required" by these register allocator like this:
  ///    AU.addRequiredID(PHIEliminationID);
  ///  This pass is still in development
  extern char &StrongPHIEliminationID;

  /// LiveStacks pass. An analysis keeping track of the liveness of stack slots.
  extern char &LiveStacksID;

  /// TwoAddressInstruction pass - This pass reduces two-address instructions to
  /// use two operands. This destroys SSA information but it is desired by
  /// register allocators.
  extern char &TwoAddressInstructionPassID;

  /// RegisteCoalescer pass - This pass merges live ranges to eliminate copies.
  extern char &RegisterCoalescerPassID;

  /// MachineScheduler pass - This pass schedules machine instructions.
  extern char &MachineSchedulerID;

  /// SpillPlacement analysis. Suggest optimal placement of spill code between
  /// basic blocks.
  ///
  extern char &SpillPlacementID;

  /// UnreachableMachineBlockElimination pass - This pass removes unreachable
  /// machine basic blocks.
  extern char &UnreachableMachineBlockElimID;

  /// DeadMachineInstructionElim pass - This pass removes dead machine
  /// instructions.
  ///
  FunctionPass *createDeadMachineInstructionElimPass();

  /// Creates a register allocator as the user specified on the command line, or
  /// picks one that matches OptLevel.
  ///
  FunctionPass *createRegisterAllocator(CodeGenOpt::Level OptLevel);

  /// FastRegisterAllocation Pass - This pass register allocates as fast as
  /// possible. It is best suited for debug code where live ranges are short.
  ///
  FunctionPass *createFastRegisterAllocator();

  /// BasicRegisterAllocation Pass - This pass implements a degenerate global
  /// register allocator using the basic regalloc framework.
  ///
  FunctionPass *createBasicRegisterAllocator();

  /// Greedy register allocation pass - This pass implements a global register
  /// allocator for optimized builds.
  ///
  FunctionPass *createGreedyRegisterAllocator();

  /// PBQPRegisterAllocation Pass - This pass implements the Partitioned Boolean
  /// Quadratic Prograaming (PBQP) based register allocator.
  ///
  FunctionPass *createDefaultPBQPRegisterAllocator();

  /// PrologEpilogCodeInserter Pass - This pass inserts prolog and epilog code,
  /// and eliminates abstract frame references.
  ///
  FunctionPass *createPrologEpilogCodeInserter();

  /// ExpandPostRAPseudos Pass - This pass expands pseudo instructions after
  /// register allocation.
  ///
  FunctionPass *createExpandPostRAPseudosPass();

  /// createPostRAScheduler - This pass performs post register allocation
  /// scheduling.
  FunctionPass *createPostRAScheduler(CodeGenOpt::Level OptLevel);

  /// BranchFolding Pass - This pass performs machine code CFG based
  /// optimizations to delete branches to branches, eliminate branches to
  /// successor blocks (creating fall throughs), and eliminating branches over
  /// branches.
  FunctionPass *createBranchFoldingPass(bool DefaultEnableTailMerge);

  /// TailDuplicate Pass - Duplicate blocks with unconditional branches
  /// into tails of their predecessors.
  FunctionPass *createTailDuplicatePass(bool PreRegAlloc = false);

  /// IfConverter Pass - This pass performs machine code if conversion.
  FunctionPass *createIfConverterPass();

  /// MachineBlockPlacement Pass - This pass places basic blocks based on branch
  /// probabilities.
  FunctionPass *createMachineBlockPlacementPass();

  /// MachineBlockPlacementStats Pass - This pass collects statistics about the
  /// basic block placement using branch probabilities and block frequency
  /// information.
  FunctionPass *createMachineBlockPlacementStatsPass();

  /// Code Placement Pass - This pass optimize code placement and aligns loop
  /// headers to target specific alignment boundary.
  FunctionPass *createCodePlacementOptPass();

  /// IntrinsicLowering Pass - Performs target-independent LLVM IR
  /// transformations for highly portable strategies.
  FunctionPass *createGCLoweringPass();

  /// MachineCodeAnalysis Pass - Target-independent pass to mark safe points in
  /// machine code. Must be added very late during code generation, just prior
  /// to output, and importantly after all CFG transformations (such as branch
  /// folding).
  FunctionPass *createGCMachineCodeAnalysisPass();

  /// Deleter Pass - Releases GC metadata.
  ///
  FunctionPass *createGCInfoDeleter();

  /// Creates a pass to print GC metadata.
  ///
  FunctionPass *createGCInfoPrinter(raw_ostream &OS);

  /// createMachineCSEPass - This pass performs global CSE on machine
  /// instructions.
  FunctionPass *createMachineCSEPass();

  /// createMachineLICMPass - This pass performs LICM on machine instructions.
  ///
  FunctionPass *createMachineLICMPass(bool PreRegAlloc = true);

  /// createMachineSinkingPass - This pass performs sinking on machine
  /// instructions.
  FunctionPass *createMachineSinkingPass();

  /// createMachineCopyPropagationPass - This pass performs copy propagation on
  /// machine instructions.
  FunctionPass *createMachineCopyPropagationPass();

  /// createPeepholeOptimizerPass - This pass performs peephole optimizations -
  /// like extension and comparison eliminations.
  FunctionPass *createPeepholeOptimizerPass();

  /// createOptimizePHIsPass - This pass optimizes machine instruction PHIs
  /// to take advantage of opportunities created during DAG legalization.
  FunctionPass *createOptimizePHIsPass();

  /// createStackSlotColoringPass - This pass performs stack slot coloring.
  FunctionPass *createStackSlotColoringPass(bool);

  /// createStackProtectorPass - This pass adds stack protectors to functions.
  FunctionPass *createStackProtectorPass(const TargetLowering *tli);

  /// createMachineVerifierPass - This pass verifies cenerated machine code
  /// instructions for correctness.
  FunctionPass *createMachineVerifierPass(const char *Banner = 0);

  /// createDwarfEHPass - This pass mulches exception handling code into a form
  /// adapted to code generation.  Required if using dwarf exception handling.
  FunctionPass *createDwarfEHPass(const TargetMachine *tm);

  /// createSjLjEHPass - This pass adapts exception handling code to use
  /// the GCC-style builtin setjmp/longjmp (sjlj) to handling EH control flow.
  FunctionPass *createSjLjEHPass(const TargetLowering *tli);

  /// createLocalStackSlotAllocationPass - This pass assigns local frame
  /// indices to stack slots relative to one another and allocates
  /// base registers to access them when it is estimated by the target to
  /// be out of range of normal frame pointer or stack pointer index
  /// addressing.
  FunctionPass *createLocalStackSlotAllocationPass();

  /// createExpandISelPseudosPass - This pass expands pseudo-instructions.
  ///
  FunctionPass *createExpandISelPseudosPass();

  /// createExecutionDependencyFixPass - This pass fixes execution time
  /// problems with dependent instructions, such as switching execution
  /// domains to match.
  ///
  /// The pass will examine instructions using and defining registers in RC.
  ///
  FunctionPass *createExecutionDependencyFixPass(const TargetRegisterClass *RC);

  /// createUnpackMachineBundles - This pass unpack machine instruction bundles.
  ///
  FunctionPass *createUnpackMachineBundlesPass();

  /// createFinalizeMachineBundles - This pass finalize machine instruction
  /// bundles (created earlier, e.g. during pre-RA scheduling).
  ///
  FunctionPass *createFinalizeMachineBundlesPass();

} // End llvm namespace

#endif
