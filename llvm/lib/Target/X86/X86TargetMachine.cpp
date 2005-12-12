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
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

X86VectorEnum llvm::X86Vector = NoSSE;
bool llvm::X86ScalarSSE = false;

/// X86TargetMachineModule - Note that this is used on hosts that cannot link
/// in a library unless there are references into the library.  In particular,
/// it seems that it is not possible to get things to work on Win32 without
/// this.  Though it is unused, do not remove it.
extern "C" int X86TargetMachineModule;
int X86TargetMachineModule = 0;

namespace {
  cl::opt<bool> DisableOutput("disable-x86-llc-output", cl::Hidden,
                              cl::desc("Disable the X86 asm printer, for use "
                                       "when profiling the code generator."));
  cl::opt<bool, true> EnableSSEFP("enable-sse-scalar-fp",
                cl::desc("Perform FP math in SSE regs instead of the FP stack"),
                cl::location(X86ScalarSSE),
                cl::init(false));

  cl::opt<bool> EnableX86DAGDAG("enable-x86-dag-isel", cl::Hidden,
                      cl::desc("Enable DAG-to-DAG isel for X86"));
  
  // FIXME: This should eventually be handled with target triples and
  // subtarget support!
  cl::opt<X86VectorEnum, true>
  SSEArg(
    cl::desc("Enable SSE support in the X86 target:"),
    cl::values(
       clEnumValN(SSE,  "sse", "  Enable SSE support"),
       clEnumValN(SSE2, "sse2", "  Enable SSE and SSE2 support"),
       clEnumValN(SSE3, "sse3", "  Enable SSE, SSE2, and SSE3 support"),
       clEnumValEnd),
    cl::location(X86Vector), cl::init(NoSSE));

  // Register the target.
  RegisterTarget<X86TargetMachine> X("x86", "  IA-32 (Pentium and above)");
}

unsigned X86TargetMachine::getJITMatchQuality() {
#if defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
  return 10;
#else
  return 0;
#endif
}

unsigned X86TargetMachine::getModuleMatchQuality(const Module &M) {
  // We strongly match "i[3-9]86-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 5 && TT[0] == 'i' && TT[2] == '8' && TT[3] == '6' &&
      TT[4] == '-' && TT[1] - '3' < 6)
    return 20;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer32)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return getJITMatchQuality()/2;
}

/// X86TargetMachine ctor - Create an ILP32 architecture model
///
X86TargetMachine::X86TargetMachine(const Module &M,
                                  IntrinsicLowering *IL,
                                  const std::string &FS)
  : TargetMachine("X86", IL, true, 4, 4, 4, 4, 4),
    Subtarget(M, FS),
    FrameInfo(TargetFrameInfo::StackGrowsDown,
              Subtarget.getStackAlignment(), -4),
    JITInfo(*this) {
  // Scalar SSE FP requires at least SSE2
  X86ScalarSSE &= X86Vector >= SSE2;
}


// addPassesToEmitFile - We currently use all of the same passes as the JIT
// does to emit statically compiled machine code.
bool X86TargetMachine::addPassesToEmitFile(PassManager &PM, std::ostream &Out,
                                           CodeGenFileType FileType,
                                           bool Fast) {
  if (FileType != TargetMachine::AssemblyFile &&
      FileType != TargetMachine::ObjectFile) return true;

  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  // Install an instruction selector.
  if (EnableX86DAGDAG)
    PM.add(createX86ISelDag(*this));
  else
    PM.add(createX86ISelPattern(*this));

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
    switch (FileType) {
    default:
      assert(0 && "Unexpected filetype here!");
    case TargetMachine::AssemblyFile:
      PM.add(createX86CodePrinterPass(Out, *this));
      break;
    case TargetMachine::ObjectFile:
      // FIXME: We only support emission of ELF files for now, this should check
      // the target triple and decide on the format to write (e.g. COFF on
      // win32).
      addX86ELFObjectWriterPass(PM, Out, *this);
      break;
    }

  // Delete machine code for this function
  PM.add(createMachineCodeDeleter());

  return false; // success!
}

/// addPassesToJITCompile - Add passes to the specified pass manager to
/// implement a fast dynamic compiler for this target.  Return true if this is
/// not supported for this target.
///
void X86JITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  // Install an instruction selector.
  if (EnableX86DAGDAG)
    PM.add(createX86ISelDag(TM));
  else
    PM.add(createX86ISelPattern(TM));

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

bool X86TargetMachine::addPassesToEmitMachineCode(FunctionPassManager &PM,
                                                  MachineCodeEmitter &MCE) {
  PM.add(createX86CodeEmitterPass(MCE));
  // Delete machine code for this function
  PM.add(createMachineCodeDeleter());
  return false;
}
