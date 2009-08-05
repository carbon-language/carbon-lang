//===-- llvm/CodeGen/ObjectCodeEmitter.cpp -------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/BinaryObject.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineRelocation.h"
#include "llvm/CodeGen/ObjectCodeEmitter.h"

//===----------------------------------------------------------------------===//
//                       ObjectCodeEmitter Implementation
//===----------------------------------------------------------------------===//

namespace llvm {

ObjectCodeEmitter::ObjectCodeEmitter() : BO(0) {}
ObjectCodeEmitter::ObjectCodeEmitter(BinaryObject *bo) : BO(bo) {}
ObjectCodeEmitter::~ObjectCodeEmitter() {}

/// setBinaryObject - set the BinaryObject we are writting to
void ObjectCodeEmitter::setBinaryObject(BinaryObject *bo) { BO = bo; }

/// emitByte - This callback is invoked when a byte needs to be
/// written to the data stream, without buffer overflow testing.
void ObjectCodeEmitter::emitByte(uint8_t B) {
  BO->emitByte(B);
}

/// emitWordLE - This callback is invoked when a 32-bit word needs to be
/// written to the data stream in little-endian format.
void ObjectCodeEmitter::emitWordLE(uint32_t W) {
  BO->emitWordLE(W);
}

/// emitWordBE - This callback is invoked when a 32-bit word needs to be
/// written to the data stream in big-endian format.
void ObjectCodeEmitter::emitWordBE(uint32_t W) {
  BO->emitWordBE(W);
}

/// emitDWordLE - This callback is invoked when a 64-bit word needs to be
/// written to the data stream in little-endian format.
void ObjectCodeEmitter::emitDWordLE(uint64_t W) {
  BO->emitDWordLE(W);
}

/// emitDWordBE - This callback is invoked when a 64-bit word needs to be
/// written to the data stream in big-endian format.
void ObjectCodeEmitter::emitDWordBE(uint64_t W) {
  BO->emitDWordBE(W);
}

/// emitAlignment - Align 'BO' to the necessary alignment boundary.
void ObjectCodeEmitter::emitAlignment(unsigned Alignment /* 0 */,
                                      uint8_t fill /* 0 */) {
  BO->emitAlignment(Alignment, fill);
}

/// emitULEB128Bytes - This callback is invoked when a ULEB128 needs to be
/// written to the data stream.
void ObjectCodeEmitter::emitULEB128Bytes(uint64_t Value) {
  BO->emitULEB128Bytes(Value);
}

/// emitSLEB128Bytes - This callback is invoked when a SLEB128 needs to be
/// written to the data stream.
void ObjectCodeEmitter::emitSLEB128Bytes(uint64_t Value) {
  BO->emitSLEB128Bytes(Value);
}

/// emitString - This callback is invoked when a String needs to be
/// written to the data stream.
void ObjectCodeEmitter::emitString(const std::string &String) {
  BO->emitString(String);
}

/// getCurrentPCValue - This returns the address that the next emitted byte
/// will be output to.
uintptr_t ObjectCodeEmitter::getCurrentPCValue() const {
  return BO->getCurrentPCOffset();
}

/// getCurrentPCOffset - Return the offset from the start of the emitted
/// buffer that we are currently writing to.
uintptr_t ObjectCodeEmitter::getCurrentPCOffset() const {
  return BO->getCurrentPCOffset();
}

/// addRelocation - Whenever a relocatable address is needed, it should be
/// noted with this interface.
void ObjectCodeEmitter::addRelocation(const MachineRelocation& relocation) {
  BO->addRelocation(relocation);
}

/// StartMachineBasicBlock - This should be called by the target when a new
/// basic block is about to be emitted.  This way the MCE knows where the
/// start of the block is, and can implement getMachineBasicBlockAddress.
void ObjectCodeEmitter::StartMachineBasicBlock(MachineBasicBlock *MBB) {
  if (MBBLocations.size() <= (unsigned)MBB->getNumber())
    MBBLocations.resize((MBB->getNumber()+1)*2);
  MBBLocations[MBB->getNumber()] = getCurrentPCOffset();
}

/// getMachineBasicBlockAddress - Return the address of the specified
/// MachineBasicBlock, only usable after the label for the MBB has been
/// emitted.
uintptr_t
ObjectCodeEmitter::getMachineBasicBlockAddress(MachineBasicBlock *MBB) const {
  assert(MBBLocations.size() > (unsigned)MBB->getNumber() &&
         MBBLocations[MBB->getNumber()] && "MBB not emitted!");
  return MBBLocations[MBB->getNumber()];
}

/// getJumpTableEntryAddress - Return the address of the jump table with index
/// 'Index' in the function that last called initJumpTableInfo.
uintptr_t ObjectCodeEmitter::getJumpTableEntryAddress(unsigned Index) const {
  assert(JTLocations.size() > Index && "JT not emitted!");
  return JTLocations[Index];
}

/// getConstantPoolEntryAddress - Return the address of the 'Index' entry in
/// the constant pool that was last emitted with the emitConstantPool method.
uintptr_t ObjectCodeEmitter::getConstantPoolEntryAddress(unsigned Index) const {
  assert(CPLocations.size() > Index && "CP not emitted!");
  return CPLocations[Index];
}

/// getConstantPoolEntrySection - Return the section of the 'Index' entry in
/// the constant pool that was last emitted with the emitConstantPool method.
uintptr_t ObjectCodeEmitter::getConstantPoolEntrySection(unsigned Index) const {
  assert(CPSections.size() > Index && "CP not emitted!");
  return CPSections[Index];
}

} // end namespace llvm

