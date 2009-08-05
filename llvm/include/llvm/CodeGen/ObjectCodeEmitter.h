//===-- llvm/CodeGen/ObjectCodeEmitter.h - Object Code Emitter -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Generalized Object Code Emitter, works with ObjectModule and BinaryObject.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_OBJECTCODEEMITTER_H
#define LLVM_CODEGEN_OBJECTCODEEMITTER_H

#include "llvm/CodeGen/MachineCodeEmitter.h"

namespace llvm {

class BinaryObject;
class MachineBasicBlock;
class MachineCodeEmitter;
class MachineFunction;
class MachineConstantPool;
class MachineJumpTableInfo;
class MachineModuleInfo;

class ObjectCodeEmitter : public MachineCodeEmitter {
protected:

  /// Binary Object (Section or Segment) we are emitting to.
  BinaryObject *BO;

  /// MBBLocations - This vector is a mapping from MBB ID's to their address.
  /// It is filled in by the StartMachineBasicBlock callback and queried by
  /// the getMachineBasicBlockAddress callback.
  std::vector<uintptr_t> MBBLocations;

  /// LabelLocations - This vector is a mapping from Label ID's to their 
  /// address.
  std::vector<uintptr_t> LabelLocations;

  /// CPLocations - This is a map of constant pool indices to offsets from the
  /// start of the section for that constant pool index.
  std::vector<uintptr_t> CPLocations;

  /// CPSections - This is a map of constant pool indices to the Section
  /// containing the constant pool entry for that index.
  std::vector<uintptr_t> CPSections;

  /// JTLocations - This is a map of jump table indices to offsets from the
  /// start of the section for that jump table index.
  std::vector<uintptr_t> JTLocations;

public:
  ObjectCodeEmitter();
  ObjectCodeEmitter(BinaryObject *bo);
  virtual ~ObjectCodeEmitter();

  /// setBinaryObject - set the BinaryObject we are writting to
  void setBinaryObject(BinaryObject *bo);

  /// emitByte - This callback is invoked when a byte needs to be 
  /// written to the data stream, without buffer overflow testing.
  void emitByte(uint8_t B);

  /// emitWordLE - This callback is invoked when a 32-bit word needs to be
  /// written to the data stream in little-endian format.
  void emitWordLE(uint32_t W);

  /// emitWordBE - This callback is invoked when a 32-bit word needs to be
  /// written to the data stream in big-endian format.
  void emitWordBE(uint32_t W);

  /// emitDWordLE - This callback is invoked when a 64-bit word needs to be
  /// written to the data stream in little-endian format.
  void emitDWordLE(uint64_t W);

  /// emitDWordBE - This callback is invoked when a 64-bit word needs to be
  /// written to the data stream in big-endian format.
  void emitDWordBE(uint64_t W);

  /// emitAlignment - Move the CurBufferPtr pointer up the the specified
  /// alignment (saturated to BufferEnd of course).
  void emitAlignment(unsigned Alignment = 0, uint8_t fill = 0);

  /// emitULEB128Bytes - This callback is invoked when a ULEB128 needs to be
  /// written to the data stream.
  void emitULEB128Bytes(uint64_t Value);

  /// emitSLEB128Bytes - This callback is invoked when a SLEB128 needs to be
  /// written to the data stream.
  void emitSLEB128Bytes(uint64_t Value);

  /// emitString - This callback is invoked when a String needs to be
  /// written to the data stream.
  void emitString(const std::string &String);

  /// getCurrentPCValue - This returns the address that the next emitted byte
  /// will be output to.
  uintptr_t getCurrentPCValue() const;

  /// getCurrentPCOffset - Return the offset from the start of the emitted
  /// buffer that we are currently writing to.
  uintptr_t getCurrentPCOffset() const;

  /// addRelocation - Whenever a relocatable address is needed, it should be
  /// noted with this interface.
  void addRelocation(const MachineRelocation& relocation);

  /// earlyResolveAddresses - True if the code emitter can use symbol addresses 
  /// during code emission time. The JIT is capable of doing this because it
  /// creates jump tables or constant pools in memory on the fly while the
  /// object code emitters rely on a linker to have real addresses and should
  /// use relocations instead.
  bool earlyResolveAddresses() const { return false; }

  /// startFunction - This callback is invoked when the specified function is
  /// about to be code generated.  This initializes the BufferBegin/End/Ptr
  /// fields.
  virtual void startFunction(MachineFunction &F) = 0;

  /// finishFunction - This callback is invoked when the specified function has
  /// finished code generation.  If a buffer overflow has occurred, this method
  /// returns true (the callee is required to try again), otherwise it returns
  /// false.
  virtual bool finishFunction(MachineFunction &F) = 0;

  /// StartMachineBasicBlock - This should be called by the target when a new
  /// basic block is about to be emitted.  This way the MCE knows where the
  /// start of the block is, and can implement getMachineBasicBlockAddress.
  virtual void StartMachineBasicBlock(MachineBasicBlock *MBB);

  /// getMachineBasicBlockAddress - Return the address of the specified
  /// MachineBasicBlock, only usable after the label for the MBB has been
  /// emitted.
  virtual uintptr_t getMachineBasicBlockAddress(MachineBasicBlock *MBB) const;

  /// emitLabel - Emits a label
  virtual void emitLabel(uint64_t LabelID) = 0;

  /// getLabelAddress - Return the address of the specified LabelID, only usable
  /// after the LabelID has been emitted.
  virtual uintptr_t getLabelAddress(uint64_t LabelID) const = 0;

  /// emitJumpTables - Emit all the jump tables for a given jump table info
  /// record to the appropriate section.
  virtual void emitJumpTables(MachineJumpTableInfo *MJTI) = 0;

  /// getJumpTableEntryAddress - Return the address of the jump table with index
  /// 'Index' in the function that last called initJumpTableInfo.
  virtual uintptr_t getJumpTableEntryAddress(unsigned Index) const;

  /// emitConstantPool - For each constant pool entry, figure out which section
  /// the constant should live in, allocate space for it, and emit it to the 
  /// Section data buffer.
  virtual void emitConstantPool(MachineConstantPool *MCP) = 0;

  /// getConstantPoolEntryAddress - Return the address of the 'Index' entry in
  /// the constant pool that was last emitted with the emitConstantPool method.
  virtual uintptr_t getConstantPoolEntryAddress(unsigned Index) const;

  /// getConstantPoolEntrySection - Return the section of the 'Index' entry in
  /// the constant pool that was last emitted with the emitConstantPool method.
  virtual uintptr_t getConstantPoolEntrySection(unsigned Index) const;

  /// Specifies the MachineModuleInfo object. This is used for exception handling
  /// purposes.
  virtual void setModuleInfo(MachineModuleInfo* Info) = 0;
  // to be implemented or depreciated with MachineModuleInfo

}; // end class ObjectCodeEmitter

} // end namespace llvm

#endif

