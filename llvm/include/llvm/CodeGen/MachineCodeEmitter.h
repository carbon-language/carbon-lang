//===-- llvm/CodeGen/MachineCodeEmitter.h - Code emission -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an abstract interface that is used by the machine code
// emission framework to output the code.  This allows machine code emission to
// be separated from concerns such as resolution of call targets, and where the
// machine code will be written (memory or disk, f.e.).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECODEEMITTER_H
#define LLVM_CODEGEN_MACHINECODEEMITTER_H

#include "llvm/Support/DataTypes.h"
#include <map>

namespace llvm {

class MachineBasicBlock;
class MachineConstantPool;
class MachineJumpTableInfo;
class MachineFunction;
class MachineRelocation;
class Value;
class GlobalValue;
class Function;

/// MachineCodeEmitter - This class defines two sorts of methods: those for
/// emitting the actual bytes of machine code, and those for emitting auxillary
/// structures, such as jump tables, relocations, etc.
///
/// Emission of machine code is complicated by the fact that we don't (in
/// general) know the size of the machine code that we're about to emit before
/// we emit it.  As such, we preallocate a certain amount of memory, and set the
/// BufferBegin/BufferEnd pointers to the start and end of the buffer.  As we
/// emit machine instructions, we advance the CurBufferPtr to indicate the
/// location of the next byte to emit.  In the case of a buffer overflow (we
/// need to emit more machine code than we have allocated space for), the
/// CurBufferPtr will saturate to BufferEnd and ignore stores.  Once the entire
/// function has been emitted, the overflow condition is checked, and if it has
/// occurred, more memory is allocated, and we reemit the code into it.
/// 
class MachineCodeEmitter {
protected:
  /// BufferBegin/BufferEnd - Pointers to the start and end of the memory
  /// allocated for this code buffer.
  unsigned char *BufferBegin, *BufferEnd;
  
  /// CurBufferPtr - Pointer to the next byte of memory to fill when emitting 
  /// code.  This is guranteed to be in the range [BufferBegin,BufferEnd].  If
  /// this pointer is at BufferEnd, it will never move due to code emission, and
  /// all code emission requests will be ignored (this is the buffer overflow
  /// condition).
  unsigned char *CurBufferPtr;
public:
  virtual ~MachineCodeEmitter() {}

  /// startFunction - This callback is invoked when the specified function is
  /// about to be code generated.  This initializes the BufferBegin/End/Ptr
  /// fields.
  ///
  virtual void startFunction(MachineFunction &F) {}

  /// finishFunction - This callback is invoked when the specified function has
  /// finished code generation.  If a buffer overflow has occurred, this method
  /// returns true (the callee is required to try again), otherwise it returns
  /// false.
  ///
  virtual bool finishFunction(MachineFunction &F) {
    return CurBufferPtr == BufferEnd;
  }

  /// emitConstantPool - This callback is invoked to output the constant pool
  /// for the function.
  virtual void emitConstantPool(MachineConstantPool *MCP) {}

  /// initJumpTableInfo - This callback is invoked by the JIT to allocate the
  /// necessary memory to hold the jump tables.
  virtual void initJumpTableInfo(MachineJumpTableInfo *MJTI) {}
  
  /// emitJumpTableInfo - This callback is invoked to output the jump tables
  /// for the function.  In addition to a pointer to the MachineJumpTableInfo,
  /// this function also takes a map of MBBs to addresses, so that the final
  /// addresses of the MBBs can be written to the jump tables.
  virtual void emitJumpTableInfo(MachineJumpTableInfo *MJTI,
                                 std::map<MachineBasicBlock*,uint64_t> &MBBM) {}
  
  /// startFunctionStub - This callback is invoked when the JIT needs the
  /// address of a function that has not been code generated yet.  The StubSize
  /// specifies the total size required by the stub.  Stubs are not allowed to
  /// have constant pools, the can only use the other emitByte*/emitWord*
  /// methods.
  ///
  virtual void startFunctionStub(unsigned StubSize) {}

  /// finishFunctionStub - This callback is invoked to terminate a function
  /// stub.
  ///
  virtual void *finishFunctionStub(const Function *F) { return 0; }

  /// emitByte - This callback is invoked when a byte needs to be written to the
  /// output stream.
  ///
  void emitByte(unsigned char B) {
    if (CurBufferPtr != BufferEnd)
      *CurBufferPtr++ = B;
  }

  /// emitWord - This callback is invoked when a word needs to be written to the
  /// output stream.
  ///
  void emitWord(unsigned W) {
    // FIXME: handle endian mismatches for .o file emission.
    if (CurBufferPtr+4 <= BufferEnd) {
      *(unsigned*)CurBufferPtr = W;
      CurBufferPtr += 4;
    } else {
      CurBufferPtr = BufferEnd;
    }
  }

  /// getCurrentPCValue - This returns the address that the next emitted byte
  /// will be output to.
  ///
  virtual intptr_t getCurrentPCValue() const {
    return (intptr_t)CurBufferPtr;
  }

  /// getCurrentPCOffset - Return the offset from the start of the emitted
  /// buffer that we are currently writing to.
  intptr_t getCurrentPCOffset() const {
    return CurBufferPtr-BufferBegin;
  }

  /// addRelocation - Whenever a relocatable address is needed, it should be
  /// noted with this interface.
  virtual void addRelocation(const MachineRelocation &MR) = 0;

  /// getConstantPoolEntryAddress - Return the address of the 'Index' entry in
  /// the constant pool that was last emitted with the emitConstantPool method.
  ///
  virtual uint64_t getConstantPoolEntryAddress(unsigned Index) = 0;

  /// getJumpTableEntryAddress - Return the address of the jump table with index
  /// 'Index' in the function that last called initJumpTableInfo.
  ///
  virtual uint64_t getJumpTableEntryAddress(unsigned Index) = 0;
  
  // allocateGlobal - Allocate some space for a global variable.
  virtual unsigned char* allocateGlobal(unsigned size, unsigned alignment) = 0;
};

} // End llvm namespace

#endif
