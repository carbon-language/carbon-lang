//===-- AsmAnnotationWriter.h - Itf for annotation .ll files - --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Clients of the assembly writer can use this interface to add their own
// special-purpose annotations to LLVM assembly language printouts.  Note that
// the assembly parser won't be able to parse these, in general, so
// implementations are advised to print stuff as LLVM comments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_ASMANNOTATIONWRITER_H
#define LLVM_ASSEMBLY_ASMANNOTATIONWRITER_H

#include <iosfwd>
class Function;
class BasicBlock;
class Instruction;

struct AssemblyAnnotationWriter {
  
  // emitFunctionAnnot - This may be implemented to emit a string right before
  // the start of a function.
  virtual void emitFunctionAnnot(const Function *F, std::ostream &OS) {}

  // emitBasicBlockAnnot - This may be implemented to emit a string right after
  // the basic block label, but before the first instruction in the block.
  virtual void emitBasicBlockAnnot(const BasicBlock *BB, std::ostream &OS) {}

  // emitInstructionAnnot - This may be implemented to emit a string right
  // before an instruction is emitted.
  virtual void emitInstructionAnnot(const Instruction *I, std::ostream &OS) {}
};

#endif
