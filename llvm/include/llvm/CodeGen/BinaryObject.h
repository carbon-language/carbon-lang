//===-- llvm/CodeGen/BinaryObject.h - Binary Object. -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a Binary Object Aka. "blob" for holding data from code
// generators, ready for data to the object module code writters.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BINARYOBJECT_H
#define LLVM_CODEGEN_BINARYOBJECT_H

#include "llvm/CodeGen/MachineRelocation.h"
#include "llvm/Support/DataTypes.h"

#include <string>
#include <vector>

namespace llvm {

typedef std::vector<uint8_t> BinaryData;

class BinaryObject {
protected:
  std::string Name;
  bool IsLittleEndian;
  bool Is64Bit;
  BinaryData Data;
  std::vector<MachineRelocation> Relocations;

public:
  /// Constructors and destructor
  BinaryObject() {}

  BinaryObject(bool isLittleEndian, bool is64Bit)
    : IsLittleEndian(isLittleEndian), Is64Bit(is64Bit) {}

  BinaryObject(const std::string &name, bool isLittleEndian, bool is64Bit)
    : Name(name), IsLittleEndian(isLittleEndian), Is64Bit(is64Bit) {}

  ~BinaryObject() {}

  /// getName - get name of BinaryObject
  inline std::string getName() const { return Name; }

  /// get size of binary data
  size_t size() const {
    return Data.size();
  }

  /// get binary data
  BinaryData& getData() {
    return Data;
  }

  /// get machine relocations
  const std::vector<MachineRelocation>& getRelocations() const {
    return Relocations;
  }

  /// hasRelocations - Return true if 'Relocations' is not empty
  bool hasRelocations() const {
    return !Relocations.empty();
  }

  /// emitZeros - This callback is invoked to emit a arbitrary number 
  /// of zero bytes to the data stream.
  inline void emitZeros(unsigned Size) {
    for (unsigned i=0; i < Size; ++i)
      emitByte(0);
  }

  /// emitByte - This callback is invoked when a byte needs to be
  /// written to the data stream.
  inline void emitByte(uint8_t B) {
    Data.push_back(B);
  }

  /// emitWord16 - This callback is invoked when a 16-bit word needs to be
  /// written to the data stream in correct endian format and correct size.
  inline void emitWord16(uint16_t W) {
    if (IsLittleEndian)
      emitWord16LE(W);
    else
      emitWord16BE(W);
  }

  /// emitWord16LE - This callback is invoked when a 16-bit word needs to be
  /// written to the data stream in correct endian format and correct size.
  inline void emitWord16LE(uint16_t W) {
    Data.push_back((uint8_t)(W >> 0));
    Data.push_back((uint8_t)(W >> 8));
  }

  /// emitWord16BE - This callback is invoked when a 16-bit word needs to be
  /// written to the data stream in correct endian format and correct size.
  inline void emitWord16BE(uint16_t W) {
    Data.push_back((uint8_t)(W >> 8));
    Data.push_back((uint8_t)(W >> 0));
  }

  /// emitWord - This callback is invoked when a word needs to be
  /// written to the data stream in correct endian format and correct size.
  inline void emitWord(uint64_t W) {
    if (!Is64Bit)
      emitWord32(W);
    else
      emitWord64(W);
  }

  /// emitWord32 - This callback is invoked when a 32-bit word needs to be
  /// written to the data stream in correct endian format.
  inline void emitWord32(uint32_t W) {
    if (IsLittleEndian)
      emitWordLE(W);
    else
      emitWordBE(W);
  }

  /// emitWord64 - This callback is invoked when a 32-bit word needs to be
  /// written to the data stream in correct endian format.
  inline void emitWord64(uint64_t W) {
    if (IsLittleEndian)
      emitDWordLE(W);
    else
      emitDWordBE(W);
  }

  /// emitWord64 - This callback is invoked when a x86_fp80 needs to be
  /// written to the data stream in correct endian format.
  inline void emitWordFP80(const uint64_t *W, unsigned PadSize) {
    if (IsLittleEndian) {
      emitWord64(W[0]);
      emitWord16(W[1]);  
    } else {
      emitWord16(W[1]);  
      emitWord64(W[0]);
    }
    emitZeros(PadSize);
  }

  /// emitWordLE - This callback is invoked when a 32-bit word needs to be
  /// written to the data stream in little-endian format.
  inline void emitWordLE(uint32_t W) {
    Data.push_back((uint8_t)(W >>  0));
    Data.push_back((uint8_t)(W >>  8));
    Data.push_back((uint8_t)(W >> 16));
    Data.push_back((uint8_t)(W >> 24));
  }

  /// emitWordBE - This callback is invoked when a 32-bit word needs to be
  /// written to the data stream in big-endian format.
  ///
  inline void emitWordBE(uint32_t W) {
    Data.push_back((uint8_t)(W >> 24));
    Data.push_back((uint8_t)(W >> 16));
    Data.push_back((uint8_t)(W >>  8));
    Data.push_back((uint8_t)(W >>  0));
  }

  /// emitDWordLE - This callback is invoked when a 64-bit word needs to be
  /// written to the data stream in little-endian format.
  inline void emitDWordLE(uint64_t W) {
    Data.push_back((uint8_t)(W >>  0));
    Data.push_back((uint8_t)(W >>  8));
    Data.push_back((uint8_t)(W >> 16));
    Data.push_back((uint8_t)(W >> 24));
    Data.push_back((uint8_t)(W >> 32));
    Data.push_back((uint8_t)(W >> 40));
    Data.push_back((uint8_t)(W >> 48));
    Data.push_back((uint8_t)(W >> 56));
  }

  /// emitDWordBE - This callback is invoked when a 64-bit word needs to be
  /// written to the data stream in big-endian format.
  inline void emitDWordBE(uint64_t W) {
    Data.push_back((uint8_t)(W >> 56));
    Data.push_back((uint8_t)(W >> 48));
    Data.push_back((uint8_t)(W >> 40));
    Data.push_back((uint8_t)(W >> 32));
    Data.push_back((uint8_t)(W >> 24));
    Data.push_back((uint8_t)(W >> 16));
    Data.push_back((uint8_t)(W >>  8));
    Data.push_back((uint8_t)(W >>  0));
  }

  /// fixByte - This callback is invoked when a byte needs to be
  /// fixup the buffer.
  inline void fixByte(uint8_t B, uint32_t offset) {
    Data[offset] = B;
  }

  /// fixWord16 - This callback is invoked when a 16-bit word needs to
  /// fixup the data stream in correct endian format.
  inline void fixWord16(uint16_t W, uint32_t offset) {
    if (IsLittleEndian)
      fixWord16LE(W, offset);
    else
      fixWord16BE(W, offset);
  }

  /// emitWord16LE - This callback is invoked when a 16-bit word needs to
  /// fixup the data stream in little endian format.
  inline void fixWord16LE(uint16_t W, uint32_t offset) {
    Data[offset]   = (uint8_t)(W >> 0);
    Data[++offset] = (uint8_t)(W >> 8);
  }

  /// fixWord16BE - This callback is invoked when a 16-bit word needs to
  /// fixup data stream in big endian format.
  inline void fixWord16BE(uint16_t W, uint32_t offset) {
    Data[offset]   = (uint8_t)(W >> 8);
    Data[++offset] = (uint8_t)(W >> 0);
  }

  /// emitWord - This callback is invoked when a word needs to
  /// fixup the data in correct endian format and correct size.
  inline void fixWord(uint64_t W, uint32_t offset) {
    if (!Is64Bit)
      fixWord32(W, offset);
    else
      fixWord64(W, offset);
  }

  /// fixWord32 - This callback is invoked when a 32-bit word needs to
  /// fixup the data in correct endian format.
  inline void fixWord32(uint32_t W, uint32_t offset) {
    if (IsLittleEndian)
      fixWord32LE(W, offset);
    else
      fixWord32BE(W, offset);
  }

  /// fixWord32LE - This callback is invoked when a 32-bit word needs to
  /// fixup the data in little endian format.
  inline void fixWord32LE(uint32_t W, uint32_t offset) {
    Data[offset]   = (uint8_t)(W >>  0);
    Data[++offset] = (uint8_t)(W >>  8);
    Data[++offset] = (uint8_t)(W >> 16);
    Data[++offset] = (uint8_t)(W >> 24);
  }

  /// fixWord32BE - This callback is invoked when a 32-bit word needs to
  /// fixup the data in big endian format.
  inline void fixWord32BE(uint32_t W, uint32_t offset) {
    Data[offset]   = (uint8_t)(W >> 24);
    Data[++offset] = (uint8_t)(W >> 16);
    Data[++offset] = (uint8_t)(W >>  8);
    Data[++offset] = (uint8_t)(W >>  0);
  }

  /// fixWord64 - This callback is invoked when a 64-bit word needs to
  /// fixup the data in correct endian format.
  inline void fixWord64(uint64_t W, uint32_t offset) {
    if (IsLittleEndian)
      fixWord64LE(W, offset);
    else
      fixWord64BE(W, offset);
  }

  /// fixWord64BE - This callback is invoked when a 64-bit word needs to
  /// fixup the data in little endian format.
  inline void fixWord64LE(uint64_t W, uint32_t offset) {
    Data[offset]   = (uint8_t)(W >>  0);
    Data[++offset] = (uint8_t)(W >>  8);
    Data[++offset] = (uint8_t)(W >> 16);
    Data[++offset] = (uint8_t)(W >> 24);
    Data[++offset] = (uint8_t)(W >> 32);
    Data[++offset] = (uint8_t)(W >> 40);
    Data[++offset] = (uint8_t)(W >> 48);
    Data[++offset] = (uint8_t)(W >> 56);
  }

  /// fixWord64BE - This callback is invoked when a 64-bit word needs to
  /// fixup the data in big endian format.
  inline void fixWord64BE(uint64_t W, uint32_t offset) {
    Data[offset]   = (uint8_t)(W >> 56);
    Data[++offset] = (uint8_t)(W >> 48);
    Data[++offset] = (uint8_t)(W >> 40);
    Data[++offset] = (uint8_t)(W >> 32);
    Data[++offset] = (uint8_t)(W >> 24);
    Data[++offset] = (uint8_t)(W >> 16);
    Data[++offset] = (uint8_t)(W >>  8);
    Data[++offset] = (uint8_t)(W >>  0);
  }

  /// emitAlignment - Pad the data to the specified alignment.
  void emitAlignment(unsigned Alignment, uint8_t fill = 0) {
    if (Alignment <= 1) return;
    unsigned PadSize = -Data.size() & (Alignment-1);
    for (unsigned i = 0; i<PadSize; ++i)
      Data.push_back(fill);
  }

  /// emitULEB128Bytes - This callback is invoked when a ULEB128 needs to be
  /// written to the data stream.
  void emitULEB128Bytes(uint64_t Value) {
    do {
      uint8_t Byte = (uint8_t)(Value & 0x7f);
      Value >>= 7;
      if (Value) Byte |= 0x80;
      emitByte(Byte);
    } while (Value);
  }

  /// emitSLEB128Bytes - This callback is invoked when a SLEB128 needs to be
  /// written to the data stream.
  void emitSLEB128Bytes(int64_t Value) {
    int Sign = Value >> (8 * sizeof(Value) - 1);
    bool IsMore;

    do {
      uint8_t Byte = (uint8_t)(Value & 0x7f);
      Value >>= 7;
      IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
      if (IsMore) Byte |= 0x80;
      emitByte(Byte);
    } while (IsMore);
  }

  /// emitString - This callback is invoked when a String needs to be
  /// written to the data stream.
  void emitString(const std::string &String) {
    for (unsigned i = 0, N = static_cast<unsigned>(String.size()); i<N; ++i) {
      unsigned char C = String[i];
      emitByte(C);
    }
    emitByte(0);
  }

  /// getCurrentPCOffset - Return the offset from the start of the emitted
  /// buffer that we are currently writing to.
  uintptr_t getCurrentPCOffset() const {
    return Data.size();
  }

  /// addRelocation - Whenever a relocatable address is needed, it should be
  /// noted with this interface.
  void addRelocation(const MachineRelocation& relocation) {
    Relocations.push_back(relocation);
  }

};

} // end namespace llvm

#endif

