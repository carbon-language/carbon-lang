//===- HexagonRelocationFunction.h ----------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_HEXAGON_HEXAGON_RELOCATION_FUNCTIONS_H
#define LLD_READER_WRITER_ELF_HEXAGON_HEXAGON_RELOCATION_FUNCTIONS_H

namespace lld {
namespace elf {

/// \brief HexagonInstruction which is used to store various values
typedef struct {
  uint32_t insnMask;
  uint32_t insnCmpMask;
  uint32_t insnBitMask;
  bool isDuplex;
} Instruction;

#include "HexagonEncodings.h"

#define FINDV4BITMASK(INSN)                                                    \
  findBitMask((uint32_t) * ((llvm::support::ulittle32_t *) INSN),              \
              insn_encodings,                                                  \
              sizeof(insn_encodings) / sizeof(Instruction))

/// \brief finds the scatter Bits that need to be used to apply relocations
inline uint32_t
findBitMask(uint32_t insn, Instruction *encodings, int32_t numInsns) {
  for (int32_t i = 0; i < numInsns ; i++) {
    if (((insn & 0xc000) == 0) && !(encodings[i].isDuplex))
      continue;

    if (((insn & 0xc000) != 0) && (encodings[i].isDuplex))
      continue;

    if (((encodings[i].insnMask) & insn) == encodings[i].insnCmpMask)
      return encodings[i].insnBitMask;
  }
  llvm_unreachable("found unknown instruction");
}

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_HEXAGON_HEXAGON_RELOCATION_FUNCTIONS_H
