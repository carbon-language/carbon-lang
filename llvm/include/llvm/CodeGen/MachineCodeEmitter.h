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
#include "Support/DataTypes.h"
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

  /// emitWord - This callback is invoked when a word needs to be written to the
  /// output stream.
  ///
  virtual void emitWord(unsigned W) = 0;

  /// getGlobalValueAddress - This method is used to get the address of the
  /// specified global value.  In some cases, however, the address may not yet
  /// be known at the point that the method is called (for example, getting the
  /// address of a function which has not yet been code generated).  If this is
  /// the case, the function returns zero, and the callee has to be able to
  /// handle the situation.
  ///
  virtual uint64_t getGlobalValueAddress(GlobalValue *V) = 0;
  virtual uint64_t getGlobalValueAddress(const std::string &Name) = 0;

  // getConstantPoolEntryAddress - Return the address of the 'Index' entry in
  // the constant pool that was last emitted with the 'emitConstantPool' method.
  //
  virtual uint64_t getConstantPoolEntryAddress(unsigned Index) = 0;


  // getCurrentPCValue - This returns the address that the next emitted byte
  // will be output to.
  //
  virtual uint64_t getCurrentPCValue() = 0;

  // forceCompilationOf - Force the compilation of the specified function, and
  // return its address, because we REALLY need the address now.
  //
  // FIXME: This is JIT specific!
  //
  virtual uint64_t forceCompilationOf(Function *F) = 0;
  

  /// createDebugEmitter - Return a dynamically allocated machine
  /// code emitter, which just prints the opcodes and fields out the cout.  This
  /// can be used for debugging users of the MachineCodeEmitter interface.
  ///
  static MachineCodeEmitter *createDebugEmitter();

  /// createFilePrinterEmitter - Return a dynamically allocated
  /// machine code emitter, which prints binary code to a file.  This
  /// can be used for debugging users of the MachineCodeEmitter interface.
  ///
  static MachineCodeEmitter *createFilePrinterEmitter(MachineCodeEmitter&);
};

#endif
