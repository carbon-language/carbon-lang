//===-- Passes.h - Target independent code generation passes ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code generation
// passes provided by the LLVM backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PASSES_H
#define LLVM_CODEGEN_PASSES_H

namespace llvm {

class FunctionPass;
class PassInfo;
class TargetMachine;

// PHIElimination pass - This pass eliminates machine instruction PHI nodes by
// inserting copy instructions.  This destroys SSA information, but is the
// desired input for some register allocators.  This pass is "required" by these
// register allocator like this:  AU.addRequiredID(PHIEliminationID);
//
extern const PassInfo *PHIEliminationID;

// TwoAddressInstruction pass - This pass reduces two-address
// instructions to use two operands. This destroys SSA information but
// it is desired by register allocators.
extern const PassInfo *TwoAddressInstructionPassID;

/// Creates a register allocator as the user specified on the command
/// line.
FunctionPass *createRegisterAllocator();

/// SimpleRegisterAllocation Pass - This pass converts the input machine code
/// from SSA form to use explicit registers by spilling every register.  Wow,
/// great policy huh?
///
FunctionPass *createSimpleRegisterAllocator();

/// LocalRegisterAllocation Pass - This pass register allocates the input code a
/// basic block at a time, yielding code better than the simple register
/// allocator, but not as good as a global allocator.
/// 
FunctionPass *createLocalRegisterAllocator();

/// LinearScanRegisterAllocation Pass - This pass implements the
/// linear scan register allocation algorithm, a global register
/// allocator.
///
FunctionPass *createLinearScanRegisterAllocator();

/// PrologEpilogCodeInserter Pass - This pass inserts prolog and epilog code,
/// and eliminates abstract frame references.
///
FunctionPass *createPrologEpilogCodeInserter();

/// getRegisterAllocator - This creates an instance of the register allocator
/// for the Sparc.
FunctionPass *getRegisterAllocator(TargetMachine &T);

} // End llvm namespace

#endif
