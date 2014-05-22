//===-- X86.h - Top-level interface for X86 representation ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the x86
// target library, as used by the LLVM JIT.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_X86_H
#define TARGET_X86_H

#include "llvm/Support/CodeGen.h"

namespace llvm {

class FunctionPass;
class ImmutablePass;
class JITCodeEmitter;
class X86TargetMachine;

/// createX86ISelDag - This pass converts a legalized DAG into a
/// X86-specific DAG, ready for instruction scheduling.
///
FunctionPass *createX86ISelDag(X86TargetMachine &TM,
                               CodeGenOpt::Level OptLevel);

/// createX86GlobalBaseRegPass - This pass initializes a global base
/// register for PIC on x86-32.
FunctionPass* createX86GlobalBaseRegPass();

/// createCleanupLocalDynamicTLSPass() - This pass combines multiple accesses
/// to local-dynamic TLS variables so that the TLS base address for the module
/// is only fetched once per execution path through the function.
FunctionPass *createCleanupLocalDynamicTLSPass();

/// createX86FloatingPointStackifierPass - This function returns a pass which
/// converts floating point register references and pseudo instructions into
/// floating point stack references and physical instructions.
///
FunctionPass *createX86FloatingPointStackifierPass();

/// createX86IssueVZeroUpperPass - This pass inserts AVX vzeroupper instructions
/// before each call to avoid transition penalty between functions encoded with
/// AVX and SSE.
FunctionPass *createX86IssueVZeroUpperPass();

/// createX86CodeEmitterPass - Return a pass that emits the collected X86 code
/// to the specified MCE object.
FunctionPass *createX86JITCodeEmitterPass(X86TargetMachine &TM,
                                          JITCodeEmitter &JCE);

/// createX86EmitCodeToMemory - Returns a pass that converts a register
/// allocated function into raw machine code in a dynamically
/// allocated chunk of memory.
///
FunctionPass *createEmitX86CodeToMemory();

/// \brief Creates an X86-specific Target Transformation Info pass.
ImmutablePass *createX86TargetTransformInfoPass(const X86TargetMachine *TM);

/// createX86PadShortFunctions - Return a pass that pads short functions
/// with NOOPs. This will prevent a stall when returning on the Atom.
FunctionPass *createX86PadShortFunctions();
/// createX86FixupLEAs - Return a a pass that selectively replaces
/// certain instructions (like add, sub, inc, dec, some shifts,
/// and some multiplies) by equivalent LEA instructions, in order
/// to eliminate execution delays in some Atom processors.
FunctionPass *createX86FixupLEAs();

} // End llvm namespace

#endif
