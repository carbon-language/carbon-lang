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

#include <string>
class MachineFunction;
class MachineBasicBlock;
class MachineConstantPool;
class Value;
class GlobalValue;
class Function;

struct MachineCodeEmitter {
  virtual ~MachineCodeEmitter() {}

  /// startFunction - This callback is invoked when the specified function is
  /// about to be code generated.
  ///
  virtual void startFunction(MachineFunction &F) {}
  
  /// finishFunction - This callback is invoked when the specified function has
  /// finished code generation.
  ///
  virtual void finishFunction(MachineFunction &F) {}

  /// emitConstantPool - This callback is invoked to output the constant pool
  /// for the function.
  virtual void emitConstantPool(MachineConstantPool *MCP) {}

  /// startBasicBlock - This callback is invoked when a new basic block is about
  /// to be emitted.
  ///
  virtual void startBasicBlock(MachineBasicBlock &BB) {}

  /// startFunctionStub - This callback is invoked when the JIT needs the
  /// address of a function that has not been code generated yet.  The StubSize
  /// specifies the total size required by the stub.  Stubs are not allowed to
  /// have constant pools, the can only use the other emit* methods.
  ///
  virtual void startFunctionStub(const Function &F, unsigned StubSize) {}

  /// finishFunctionStub - This callback is invoked to terminate a function
  /// stub.
  ///
  virtual void *finishFunctionStub(const Function &F) { return 0; }

  /// emitByte - This callback is invoked when a byte needs to be written to the
  /// output stream.
  ///
  virtual void emitByte(unsigned char B) {}

  /// emitPCRelativeDisp - This callback is invoked when we need to write out a
  /// PC relative displacement for the specified Value*.  This is used for call
  /// and jump instructions typically.
  ///
  virtual void emitPCRelativeDisp(Value *V) {}

  /// emitGlobalAddress - This callback is invoked when we need to write out the
  /// address of a global value to machine code.  This is important for indirect
  /// calls as well as accessing global variables.
  ///
  virtual void emitGlobalAddress(GlobalValue *V, bool isPCRelative) {}
  virtual void emitGlobalAddress(const std::string &Name, bool isPCRelative) {}

  /// emitFunctionConstantValueAddress - This callback is invoked when the
  /// address of a constant, which was spilled to memory, needs to be addressed.
  /// This is used for constants which cannot be directly specified as operands
  /// to instructions, such as large integer values on the sparc, or floating
  /// point constants on the X86.
  ///
  virtual void emitFunctionConstantValueAddress(unsigned ConstantNum,
						int Offset) {}

  /// createDebugMachineCodeEmitter - Return a dynamically allocated machine
  /// code emitter, which just prints the opcodes and fields out the cout.  This
  /// can be used for debugging users of the MachineCodeEmitter interface.
  ///
  static MachineCodeEmitter *createDebugMachineCodeEmitter();

  /// createFilePrinterMachineCodeEmitter - Return a dynamically allocated
  /// machine code emitter, which prints binary code to a file.  This
  /// can be used for debugging users of the MachineCodeEmitter interface.
  ///
  static MachineCodeEmitter*
  createFilePrinterMachineCodeEmitter(MachineCodeEmitter&);
};

#endif
