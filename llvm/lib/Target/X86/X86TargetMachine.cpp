//===-- X86TargetMachine.cpp - Define TargetMachine for the X86 -----------===//
// 
// This file defines the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "X86TargetMachine.h"
#include "X86.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Transforms/Scalar.h"
#include "Support/CommandLine.h"
#include "Support/Statistic.h"

namespace {
  cl::opt<bool> NoLocalRA("disable-local-ra",
                          cl::desc("Use Simple RA instead of Local RegAlloc"));
  cl::opt<bool> PrintCode("print-machineinstrs",
			  cl::desc("Print generated machine code"));
  cl::opt<bool> NoPatternISel("disable-pattern-isel", cl::init(true),
                        cl::desc("Use the 'simple' X86 instruction selector"));
}

// allocateX86TargetMachine - Allocate and return a subclass of TargetMachine
// that implements the X86 backend.
//
TargetMachine *allocateX86TargetMachine(const Module &M) {
  return new X86TargetMachine(M);
}


/// X86TargetMachine ctor - Create an ILP32 architecture model
///
X86TargetMachine::X86TargetMachine(const Module &M)
  : TargetMachine("X86", 
		  M.getEndianness() != Module::BigEndian,
                  M.getPointerSize() != Module::Pointer64 ? 4 : 8,
                  M.getPointerSize() != Module::Pointer64 ? 4 : 8,
                  M.getPointerSize() != Module::Pointer64 ? 4 : 8,
                  4, M.getPointerSize() != Module::Pointer64 ? 4 : 8),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8/*16 for SSE*/, 4) {
}


// addPassesToEmitAssembly - We currently use all of the same passes as the JIT
// does to emit statically compiled machine code.
bool X86TargetMachine::addPassesToEmitAssembly(PassManager &PM,
					       std::ostream &Out) {
  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  if (NoPatternISel)
    PM.add(createX86SimpleInstructionSelector(*this));
  else
    PM.add(createX86PatternInstructionSelector(*this));

  // TODO: optional optimizations go here

  // FIXME: Add SSA based peephole optimizer here.

  // Print the instruction selected machine code...
  if (PrintCode)
    PM.add(createMachineFunctionPrinterPass());

  // Perform register allocation to convert to a concrete x86 representation
  if (NoLocalRA)
    PM.add(createSimpleRegisterAllocator());
  else
    PM.add(createLocalRegisterAllocator());

  if (PrintCode)
    PM.add(createMachineFunctionPrinterPass());

  PM.add(createX86FloatingPointStackifierPass());

  if (PrintCode)
    PM.add(createMachineFunctionPrinterPass());

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  PM.add(createPrologEpilogCodeInserter());

  PM.add(createX86PeepholeOptimizerPass());

  if (PrintCode)  // Print the register-allocated code
    PM.add(createX86CodePrinterPass(std::cerr, *this));

  PM.add(createX86CodePrinterPass(Out, *this));
  return false; // success!
}

/// addPassesToJITCompile - Add passes to the specified pass manager to
/// implement a fast dynamic compiler for this target.  Return true if this is
/// not supported for this target.
///
bool X86TargetMachine::addPassesToJITCompile(FunctionPassManager &PM) {
  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  if (NoPatternISel)
    PM.add(createX86SimpleInstructionSelector(*this));
  else
    PM.add(createX86PatternInstructionSelector(*this));

  // TODO: optional optimizations go here

  // FIXME: Add SSA based peephole optimizer here.

  // Print the instruction selected machine code...
  if (PrintCode)
    PM.add(createMachineFunctionPrinterPass());

  // Perform register allocation to convert to a concrete x86 representation
  if (NoLocalRA)
    PM.add(createSimpleRegisterAllocator());
  else
    PM.add(createLocalRegisterAllocator());

  if (PrintCode)
    PM.add(createMachineFunctionPrinterPass());

  PM.add(createX86FloatingPointStackifierPass());

  if (PrintCode)
    PM.add(createMachineFunctionPrinterPass());

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  PM.add(createPrologEpilogCodeInserter());

  PM.add(createX86PeepholeOptimizerPass());

  if (PrintCode)  // Print the register-allocated code
    PM.add(createX86CodePrinterPass(std::cerr, *this));
  return false; // success!
}

