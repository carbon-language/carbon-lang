//===-- ARMUnwindOpAsm.h - ARM Unwind Opcodes Assembler ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the unwind opcode assmebler for ARM exception handling
// table.
//
//===----------------------------------------------------------------------===//

#ifndef ARM_UNWIND_OP_ASM_H
#define ARM_UNWIND_OP_ASM_H

#include "ARMUnwindOp.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class MCSymbol;

class UnwindOpcodeAssembler {
private:
  llvm::SmallVector<uint8_t, 8> Ops;

  unsigned Offset;
  unsigned PersonalityIndex;
  bool HasPersonality;

  enum {
    // The number of bytes to be preserved for the size and personality index
    // prefix of unwind opcodes.
    NUM_PRESERVED_PREFIX_BUF = 2
  };

public:
  UnwindOpcodeAssembler()
      : Ops(NUM_PRESERVED_PREFIX_BUF), Offset(NUM_PRESERVED_PREFIX_BUF),
        PersonalityIndex(NUM_PERSONALITY_INDEX), HasPersonality(0) {
  }

  /// Reset the unwind opcode assembler.
  void Reset() {
    Ops.resize(NUM_PRESERVED_PREFIX_BUF);
    Offset = NUM_PRESERVED_PREFIX_BUF;
    PersonalityIndex = NUM_PERSONALITY_INDEX;
    HasPersonality = 0;
  }

  /// Get the size of the payload (including the size byte)
  size_t size() const {
    return Ops.size() - Offset;
  }

  /// Get the beginning of the payload
  const uint8_t *begin() const {
    return Ops.begin() + Offset;
  }

  /// Get the payload
  StringRef data() const {
    return StringRef(reinterpret_cast<const char *>(begin()), size());
  }

  /// Set the personality index
  void setPersonality(const MCSymbol *Per) {
    HasPersonality = 1;
  }

  /// Get the personality index
  unsigned getPersonalityIndex() const {
    return PersonalityIndex;
  }

  /// Emit unwind opcodes for .save directives
  void EmitRegSave(uint32_t RegSave);

  /// Emit unwind opcodes for .vsave directives
  void EmitVFPRegSave(uint32_t VFPRegSave);

  /// Emit unwind opcodes for .setfp directives
  void EmitSetFP(uint16_t FPReg);

  /// Emit unwind opcodes to update stack pointer
  void EmitSPOffset(int64_t Offset);

  /// Finalize the unwind opcode sequence for EmitBytes()
  void Finalize();

private:
  /// Get the size of the opcodes in bytes.
  size_t getOpcodeSize() const {
    return Ops.size() - NUM_PRESERVED_PREFIX_BUF;
  }

  /// Add the length prefix to the payload
  void AddOpcodeSizePrefix(size_t Pos);

  /// Add personality index prefix in some compact format
  void AddPersonalityIndexPrefix(size_t Pos, unsigned PersonalityIndex);

  /// Fill the words with finish opcode if it is not aligned
  void EmitFinishOpcodes();
};

} // namespace llvm

#endif // ARM_UNWIND_OP_ASM_H
