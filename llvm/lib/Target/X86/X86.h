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

#include "llvm/Target/TargetMachine.h"

namespace llvm {

class FunctionPass;
class JITCodeEmitter;
class MCCodeEmitter;
class MCContext;
class MCObjectWriter;
class MachineCodeEmitter;
class Target;
class TargetAsmBackend;
class X86TargetMachine;
class formatted_raw_ostream;
class raw_ostream;

/// createX86ISelDag - This pass converts a legalized DAG into a 
/// X86-specific DAG, ready for instruction scheduling.
///
FunctionPass *createX86ISelDag(X86TargetMachine &TM,
                               CodeGenOpt::Level OptLevel);

/// createGlobalBaseRegPass - This pass initializes a global base
/// register for PIC on x86-32.
FunctionPass* createGlobalBaseRegPass();

/// createX86FloatingPointStackifierPass - This function returns a pass which
/// converts floating point register references and pseudo instructions into
/// floating point stack references and physical instructions.
///
FunctionPass *createX86FloatingPointStackifierPass();

/// createSSEDomainFixPass - This pass twiddles SSE opcodes to prevent domain
/// crossings.
FunctionPass *createSSEDomainFixPass();

/// createX86CodeEmitterPass - Return a pass that emits the collected X86 code
/// to the specified MCE object.
FunctionPass *createX86JITCodeEmitterPass(X86TargetMachine &TM,
                                          JITCodeEmitter &JCE);

MCCodeEmitter *createX86_32MCCodeEmitter(const Target &, TargetMachine &TM,
                                         MCContext &Ctx);
MCCodeEmitter *createX86_64MCCodeEmitter(const Target &, TargetMachine &TM,
                                         MCContext &Ctx);

TargetAsmBackend *createX86_32AsmBackend(const Target &, const std::string &);
TargetAsmBackend *createX86_64AsmBackend(const Target &, const std::string &);

/// createX86EmitCodeToMemory - Returns a pass that converts a register
/// allocated function into raw machine code in a dynamically
/// allocated chunk of memory.
///
FunctionPass *createEmitX86CodeToMemory();

/// createX86MaxStackAlignmentHeuristicPass - This function returns a pass
/// which determines whether the frame pointer register should be
/// reserved in case dynamic stack alignment is later required.
///
FunctionPass *createX86MaxStackAlignmentHeuristicPass();


/// createX86MachObjectWriter - Construct an X86 Mach-O object writer.
MCObjectWriter *createX86MachObjectWriter(raw_ostream &OS,
                                          bool Is64Bit,
                                          uint32_t CPUType,
                                          uint32_t CPUSubtype);

extern Target TheX86_32Target, TheX86_64Target;

} // End llvm namespace

// Defines symbolic names for X86 registers.  This defines a mapping from
// register name to register number.
//
#include "X86GenRegisterNames.inc"

// Defines symbolic names for the X86 instructions.
//
#include "X86GenInstrNames.inc"

#endif
