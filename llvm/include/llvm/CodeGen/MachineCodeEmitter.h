//===-- llvm/CodeGen/MachineCodeEmitter.h - Code emission -------*- C++ -*-===//
//
// This file defines an abstract interface that is used by the machine code
// emission framework to output the code.  This allows machine code emission to
// be seperated from concerns such as resolution of call targets, and where the
// machine code will be written (memory or disk, f.e.).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINE_CODE_EMITTER_H
#define LLVM_CODEGEN_MACHINE_CODE_EMITTER_H

class MachineFunction;
class MachineBasicBlock;

struct MachineCodeEmitter {

  /// startFunction - This callback is invoked when the specified function is
  /// about to be code generated.
  ///
  virtual void startFunction(MachineFunction &F) {}
  
  /// finishFunction - This callback is invoked when the specified function has
  /// finished code generation.
  ///
  virtual void finishFunction(MachineFunction &F) {}

  /// startBasicBlock - This callback is invoked when a new basic block is about
  /// to be emitted.
  ///
  virtual void startBasicBlock(MachineBasicBlock &BB) {}

  /// emitByte - This callback is invoked when a byte needs to be written to the
  /// output stream.
  virtual void emitByte(unsigned char B) {}
};

#endif
