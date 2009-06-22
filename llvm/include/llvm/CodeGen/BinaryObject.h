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

#include <string>
#include <vector>

namespace llvm {

class MachineRelocation;
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
    Data.push_back((W >> 0) & 255);
    Data.push_back((W >> 8) & 255);
  }

  /// emitWord16BE - This callback is invoked when a 16-bit word needs to be
  /// written to the data stream in correct endian format and correct size.
  inline void emitWord16BE(uint16_t W) {
    Data.push_back((W >> 8) & 255);
    Data.push_back((W >> 0) & 255);
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

  /// emitWordLE - This callback is invoked when a 32-bit word needs to be
  /// written to the data stream in little-endian format.
  inline void emitWordLE(uint32_t W) {
    Data.push_back((W >>  0) & 255);
    Data.push_back((W >>  8) & 255);
    Data.push_back((W >> 16) & 255);
    Data.push_back((W >> 24) & 255);
  }

  /// emitWordBE - This callback is invoked when a 32-bit word needs to be
  /// written to the data stream in big-endian format.
  ///
  inline void emitWordBE(uint32_t W) {
    Data.push_back((W >> 24) & 255);
    Data.push_back((W >> 16) & 255);
    Data.push_back((W >>  8) & 255);
    Data.push_back((W >>  0) & 255);
  }

  /// emitDWordLE - This callback is invoked when a 64-bit word needs to be
  /// written to the data stream in little-endian format.
  inline void emitDWordLE(uint64_t W) {
    Data.push_back(unsigned(W >>  0) & 255);
    Data.push_back(unsigned(W >>  8) & 255);
    Data.push_back(unsigned(W >> 16) & 255);
    Data.push_back(unsigned(W >> 24) & 255);
    Data.push_back(unsigned(W >> 32) & 255);
    Data.push_back(unsigned(W >> 40) & 255);
    Data.push_back(unsigned(W >> 48) & 255);
    Data.push_back(unsigned(W >> 56) & 255);
  }

  /// emitDWordBE - This callback is invoked when a 64-bit word needs to be
  /// written to the data stream in big-endian format.
  inline void emitDWordBE(uint64_t W) {
    Data.push_back(unsigned(W >> 56) & 255);
    Data.push_back(unsigned(W >> 48) & 255);
    Data.push_back(unsigned(W >> 40) & 255);
    Data.push_back(unsigned(W >> 32) & 255);
    Data.push_back(unsigned(W >> 24) & 255);
    Data.push_back(unsigned(W >> 16) & 255);
    Data.push_back(unsigned(W >>  8) & 255);
    Data.push_back(unsigned(W >>  0) & 255);
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
    Data[offset++] = W & 255;
    Data[offset] = (W >> 8) & 255;
  }

  /// fixWord16BE - This callback is invoked when a 16-bit word needs to
  /// fixup data stream in big endian format.
  inline void fixWord16BE(uint16_t W, uint32_t offset) {
    Data[offset++] = (W >> 8) & 255;
    Data[offset] = W & 255;
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
    Data[offset++] = W & 255;
    Data[offset++] = (W >> 8) & 255;
    Data[offset++] = (W >> 16) & 255;
    Data[offset] = (W >> 24) & 255;
  }

  /// fixWord32BE - This callback is invoked when a 32-bit word needs to
  /// fixup the data in big endian format.
  inline void fixWord32BE(uint32_t W, uint32_t offset) {
    Data[offset++] = (W >> 24) & 255;
    Data[offset++] = (W >> 16) & 255;
    Data[offset++] = (W >> 8) & 255;
    Data[offset] = W & 255;
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
    Data[offset++] = W & 255;
    Data[offset++] = (W >> 8) & 255;
    Data[offset++] = (W >> 16) & 255;
    Data[offset++] = (W >> 24) & 255;
    Data[offset++] = (W >> 32) & 255;
    Data[offset++] = (W >> 40) & 255;
    Data[offset++] = (W >> 48) & 255;
    Data[offset] = (W >> 56) & 255;
  }

  /// fixWord64BE - This callback is invoked when a 64-bit word needs to
  /// fixup the data in big endian format.
  inline void fixWord64BE(uint64_t W, uint32_t offset) {
    Data[offset++] = (W >> 56) & 255;
    Data[offset++] = (W >> 48) & 255;
    Data[offset++] = (W >> 40) & 255;
    Data[offset++] = (W >> 32) & 255;
    Data[offset++] = (W >> 24) & 255;
    Data[offset++] = (W >> 16) & 255;
    Data[offset++] = (W >> 8) & 255;
    Data[offset] = W & 255;
  }

  /// emitAlignment - Pad the data to the specified alignment.
  void emitAlignment(unsigned Alignment) {
    if (Alignment <= 1) return;
    unsigned PadSize = -Data.size() & (Alignment-1);
    for (unsigned i = 0; i<PadSize; ++i)
      Data.push_back(0);
  }

  /// emitULEB128Bytes - This callback is invoked when a ULEB128 needs to be
  /// written to the data stream.
  void emitULEB128Bytes(uint64_t Value) {
    do {
      unsigned char Byte = Value & 0x7f;
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
      unsigned char Byte = Value & 0x7f;
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

