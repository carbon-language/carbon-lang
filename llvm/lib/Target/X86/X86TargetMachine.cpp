//===-- X86TargetMachine.cpp - Define TargetMachine for the X86 -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file defines the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "X86TargetMachine.h"
#include "X86.h"
#include "llvm/IntrinsicLowering.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Transforms/Scalar.h"
#include "Support/CommandLine.h"
#include "Support/Statistic.h"
using namespace llvm;

namespace {
  cl::opt<bool> NoPatternISel("disable-pattern-isel", cl::init(true),
                        cl::desc("Use the 'simple' X86 instruction selector"));
  cl::opt<bool> NoSSAPeephole("disable-ssa-peephole", cl::init(true),
                        cl::desc("Disable the ssa-based peephole optimizer "
                                 "(defaults to disabled)"));
  cl::opt<bool> DisableOutput("disable-x86-llc-output", cl::Hidden,
                              cl::desc("Disable the X86 asm printer, for use "
                                       "when profiling the code generator."));
}

// allocateX86TargetMachine - Allocate and return a subclass of TargetMachine
// that implements the X86 backend.
//
TargetMachine *llvm::allocateX86TargetMachine(const Module &M,
                                              IntrinsicLowering *IL) {
  return new X86TargetMachine(M, IL);
}


/// X86TargetMachine ctor - Create an ILP32 architecture model
///
X86TargetMachine::X86TargetMachine(const Module &M, IntrinsicLowering *IL)
  : TargetMachine("X86", IL, true, 4, 4, 4, 4, 4),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8/*16 for SSE*/, 4),
    JITInfo(*this) {
}


// addPassesToEmitAssembly - We currently use all of the same passes as the JIT
// does to emit statically compiled machine code.
bool X86TargetMachine::addPassesToEmitAssembly(PassManager &PM,
					       std::ostream &Out) {
  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: The code generator does not properly handle functions with
  // unreachable basic blocks.
  PM.add(createCFGSimplificationPass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  // FIXME: Add support for the select instruction natively.
  PM.add(createLowerSelectPass());

  if (NoPatternISel)
    PM.add(createX86SimpleInstructionSelector(*this));
  else
    PM.add(createX86PatternInstructionSelector(*this));

  // Run optional SSA-based machine code optimizations next...
  if (!NoSSAPeephole)
    PM.add(createX86SSAPeepholeOptimizerPass());

  // Print the instruction selected machine code...
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  // Perform register allocation to convert to a concrete x86 representation
  PM.add(createRegisterAllocator());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createX86FloatingPointStackifierPass());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  PM.add(createPrologEpilogCodeInserter());

  PM.add(createX86PeepholeOptimizerPass());

  if (PrintMachineCode)  // Print the register-allocated code
    PM.add(createX86CodePrinterPass(std::cerr, *this));

  if (!DisableOutput)
    PM.add(createX86CodePrinterPass(Out, *this));

  // Delete machine code for this function
  PM.add(createMachineCodeDeleter());

  return false; // success!
}

/// addPassesToJITCompile - Add passes to the specified pass manager to
/// implement a fast dynamic compiler for this target.  Return true if this is
/// not supported for this target.
///
void X86JITInfo::addPassesToJITCompile(FunctionPassManager &PM) {

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: The code generator does not properly handle functions with
  // unreachable basic blocks.
  PM.add(createCFGSimplificationPass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  // FIXME: Add support for the select instruction natively.
  PM.add(createLowerSelectPass());

  if (NoPatternISel)
    PM.add(createX86SimpleInstructionSelector(TM));
  else
    PM.add(createX86PatternInstructionSelector(TM));

  // Run optional SSA-based machine code optimizations next...
  if (!NoSSAPeephole)
    PM.add(createX86SSAPeepholeOptimizerPass());

  // FIXME: Add SSA based peephole optimizer here.

  // Print the instruction selected machine code...
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  // Perform register allocation to convert to a concrete x86 representation
  PM.add(createRegisterAllocator());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createX86FloatingPointStackifierPass());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  PM.add(createPrologEpilogCodeInserter());

  PM.add(createX86PeepholeOptimizerPass());

  if (PrintMachineCode)  // Print the register-allocated code
    PM.add(createX86CodePrinterPass(std::cerr, TM));
}

