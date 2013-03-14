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

/// \brief Return the bits that are described by the mask
template < typename T >
T gatherBits(T val, T mask)
{
  T result = 0;
  size_t off = 0;

  for (size_t bit = 0; bit != sizeof (T) * 8; ++bit) {
    const bool valBit = (val >> bit) & 1;
    const bool maskBit = (mask >> bit) & 1;
    if (maskBit) {
      result |= static_cast <T> (valBit) << off;
      ++off;
    }
  }
  return result;
}

/// \brief Set the bits as described by the mask
template <typename T>
T scatterBits(T val, T mask)
{
  T result = 0;
  size_t off = 0;

  for (size_t bit = 0; bit != sizeof (T) * 8; ++bit) {
    const bool valBit = (val >> off) & 1;
    const bool maskBit = (mask >> bit) & 1;
    if (maskBit) {
      result |= static_cast<T>(valBit) << bit;
      ++off;
    }
  }
  return result;
}

} // namespace lld

#endif // LLD_READER_WRITER_RELOCATION_HELPER_FUNCTIONS_H
