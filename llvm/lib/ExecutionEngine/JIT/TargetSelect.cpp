//===-- TargetSelect.cpp - Target Chooser Code ----------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the hideously gross code that is currently used to select
// a particular TargetMachine for the JIT to use.  This should obviously be
// improved in the future, probably by having the TargetMachines register
// themselves with the runtime, and then have them choose themselves if they
// match the current machine.
//
//===----------------------------------------------------------------------===//

#include "JIT.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "Support/CommandLine.h"
using namespace llvm;

#if !defined(ENABLE_X86_JIT) && !defined(ENABLE_SPARC_JIT)
#define NO_JITS_ENABLED
#endif

namespace {
  enum ArchName { x86, SparcV9 };

#ifndef NO_JITS_ENABLED
  cl::opt<ArchName>
  Arch("march", cl::desc("Architecture to JIT to:"), cl::Prefix,
       cl::values(
#ifdef ENABLE_X86_JIT
                  clEnumVal(x86, "  IA-32 (Pentium and above)"),
#endif
#ifdef ENABLE_SPARC_JIT
                  clEnumValN(SparcV9, "sparcv9", "  Sparc-V9"),
#endif
                  0),
#if defined(ENABLE_X86_JIT)
  cl::init(x86)
#elif defined(ENABLE_SPARC_JIT)
  cl::init(SparcV9)
#endif
       );
#endif /* NO_JITS_ENABLED */
}

/// create - Create an return a new JIT compiler if there is one available
/// for the current target.  Otherwise, return null.
///
ExecutionEngine *JIT::create(ModuleProvider *MP, IntrinsicLowering *IL) {
  TargetMachine* (*TargetMachineAllocator)(const Module &,
                                           IntrinsicLowering *IL) = 0;

  // Allow a command-line switch to override what *should* be the default target
  // machine for this platform. This allows for debugging a Sparc JIT on X86 --
  // our X86 machines are much faster at recompiling LLVM and linking LLI.
#ifndef NO_JITS_ENABLED

  switch (Arch) {
#ifdef ENABLE_X86_JIT
  case x86:
    TargetMachineAllocator = allocateX86TargetMachine;
    break;
#endif
#ifdef ENABLE_SPARC_JIT
  case SparcV9:
    TargetMachineAllocator = allocateSparcV9TargetMachine;
    break;
#endif
  default:
    assert(0 && "-march flag not supported on this host!");
  }
#else
  return 0;
#endif

  // Allocate a target...
  TargetMachine *Target = TargetMachineAllocator(*MP->getModule(), IL);
  assert(Target && "Could not allocate target machine!");

  // If the target supports JIT code generation, return a new JIT now.
  if (TargetJITInfo *TJ = Target->getJITInfo())
    return new JIT(MP, *Target, *TJ);
  return 0;
}


