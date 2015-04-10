//===- lld/ReaderWriter/RelocationHelperFunctions.h------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_RELOCATION_HELPER_FUNCTIONS_H
#define LLD_READER_WRITER_RELOCATION_HELPER_FUNCTIONS_H

namespace lld {

/// Scatter val's bits as specified by the mask. Example:
///
///  Val:    0bABCDEFG
///  Mask:   0b10111100001011
///  Output: 0b00ABCD0000E0FG
template <typename T> T scatterBits(T val, T mask) {
  T result = 0;
  size_t off = 0;

  for (size_t bit = 0; bit < sizeof(T) * 8; ++bit) {
    bool maskBit = (mask >> bit) & 1;
    if (maskBit) {
      bool valBit = (val >> off) & 1;
      result |= static_cast<T>(valBit) << bit;
      ++off;
    }
  }
  return result;
}

} // namespace lld

#endif // LLD_READER_WRITER_RELOCATION_HELPER_FUNCTIONS_H
