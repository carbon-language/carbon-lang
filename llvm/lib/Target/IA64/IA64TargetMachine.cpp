//===-- IA64TargetMachine.cpp - Define TargetMachine for IA64 -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duraid Madina and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the IA64 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "IA64TargetMachine.h"
#include "IA64.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/Statistic.h"
#include <iostream>
using namespace llvm;

/// IA64TargetMachineModule - Note that this is used on hosts that cannot link
/// in a library unless there are references into the library.  In particular,
/// it seems that it is not possible to get things to work on Win32 without
/// this.  Though it is unused, do not remove it.
extern "C" int IA64TargetMachineModule;
int IA64TargetMachineModule = 0;

namespace {
  cl::opt<bool> DisableOutput("disable-ia64-llc-output", cl::Hidden,
                              cl::desc("Disable the IA64 asm printer, for use "
                                       "when profiling the code generator."));

  cl::opt<bool> EnableDAGIsel("enable-ia64-dag-isel", cl::Hidden,
		              cl::desc("Enable the IA64 DAG->DAG isel"));

  // Register the target.
  RegisterTarget<IA64TargetMachine> X("ia64", "  IA-64 (Itanium)");
}

unsigned IA64TargetMachine::getModuleMatchQuality(const Module &M) {
  // we match [iI][aA]*64
  bool seenIA64=false;
  std::string TT = M.getTargetTriple();

  if (TT.size() >= 4) {
    if( (TT[0]=='i' || TT[0]=='I') &&
        (TT[1]=='a' || TT[1]=='A') ) {
      for(unsigned int i=2; i<(TT.size()-1); i++)
        if(TT[i]=='6' && TT[i+1]=='4')
          seenIA64=true;
    }

    if (seenIA64)
      return 20; // strong match
  }

#if defined(__ia64__) || defined(__IA64__)
  return 5;
#else
  return 0;
#endif
}

/// IA64TargetMachine ctor - Create an LP64 architecture model
///
IA64TargetMachine::IA64TargetMachine(const Module &M, const std::string &FS)
  : TargetMachine("IA64"), DataLayout("e"),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 16, 0),
    TLInfo(*this) { // FIXME? check this stuff
}

// addPassesToEmitFile - We currently use all of the same passes as the JIT
// does to emit statically compiled machine code.
bool IA64TargetMachine::addPassesToEmitFile(PassManager &PM,
                                            std::ostream &Out,
                                            CodeGenFileType FileType,
                                            bool Fast) {
  if (FileType != TargetMachine::AssemblyFile) return true;

  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass(704, 16)); // on ia64 linux, jmpbufs are 704
                                          // bytes and must be 16byte aligned

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  // Add an instruction selector
// FIXME: reap this option one day:  if(EnableDAGIsel)
  PM.add(createIA64DAGToDAGInstructionSelector(*this));
  
/* XXX not yet. ;)
  // Run optional SSA-based machine code optimizations next...
  if (!NoSSAPeephole)
    PM.add(createIA64SSAPeepholeOptimizerPass());
*/

  // Print the instruction selected machine code...
  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  // Perform register allocation to convert to a concrete IA64 representation
  PM.add(createRegisterAllocator());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  PM.add(createPrologEpilogCodeInserter());

/* XXX no, not just yet */
//  PM.add(createIA64PeepholeOptimizerPass());

  // Make sure everything is bundled happily
  PM.add(createIA64BundlingPass(*this));

  if (PrintMachineCode)  // Print the register-allocated code
    PM.add(createIA64CodePrinterPass(std::cerr, *this));

  if (!DisableOutput)
    PM.add(createIA64CodePrinterPass(Out, *this));

  // Delete machine code for this function
  PM.add(createMachineCodeDeleter());

  return false; // success!
}

