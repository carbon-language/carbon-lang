//===-- Passes.h - Target independent code generation passes ----*- C++ -*-===//
//
// This file defines interfaces to access the target independent code generation
// passes provided by the LLVM backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PASSES_H
#define LLVM_CODEGEN_PASSES_H

class FunctionPass;
class PassInfo;

// PHIElimination pass - This pass eliminates machine instruction PHI nodes by
// inserting copy instructions.  This destroys SSA information, but is the
// desired input for some register allocators.  This pass is "required" by these
// register allocator like this:  AU.addRequiredID(PHIEliminationID);
//
extern const PassInfo *PHIEliminationID;

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

/// PrologEpilogCodeInserter Pass - This pass inserts prolog and epilog code,
/// and eliminates abstract frame references.
///
FunctionPass *createPrologEpilogCodeInserter();

#endif
