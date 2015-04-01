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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Endian.h"

namespace lld {
namespace elf {

/// \brief Applying fixup on Hexagon requires the relocator to fetch the fixup
/// mask from the instruction. To fetch the fixup encoding, the linker uses a
/// static array that contains the instruction mask, the compare mask and the
/// relocation mask.
typedef struct {
  uint32_t insnMask;    // Instruction mask.
  uint32_t insnCmpMask; // Compare mask.
  uint32_t insnBitMask; // Relocation mask.
  bool isDuplex;        // Indicates if the instruction is a duplex instruction.
} Instruction;

#include "HexagonEncodings.h"

/// \brief finds the scatter Bits that need to be used to apply relocations
inline uint32_t findv4bitmask(uint8_t *location) {
  uint32_t insn = llvm::support::endian::read32le(location);
  for (int32_t i = 0, e = llvm::array_lengthof(insn_encodings); i < e; i++) {
    if (((insn & 0xc000) == 0) && !(insn_encodings[i].isDuplex))
      continue;
    if (((insn & 0xc000) != 0) && (insn_encodings[i].isDuplex))
      continue;
    if (((insn_encodings[i].insnMask) & insn) == insn_encodings[i].insnCmpMask)
      return insn_encodings[i].insnBitMask;
  }
  llvm_unreachable("found unknown Hexagon instruction");
}

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_HEXAGON_HEXAGON_RELOCATION_FUNCTIONS_H
