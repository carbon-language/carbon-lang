//===- SwapByteOrder.h - Generic and optimized byte swaps -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares generic and optimized functions to swap the byte order of
// an integral type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_SWAP_BYTE_ORDER_H
#define LLVM_SYSTEM_SWAP_BYTE_ORDER_H

#include "llvm/Support/type_traits.h"
#include "llvm/System/DataTypes.h"
#include <cstddef>
#include <limits>

namespace llvm {
namespace sys {

template<typename value_type>
inline
typename enable_if_c<sizeof(value_type) == 1
                     && std::numeric_limits<value_type>::is_integer,
                     value_type>::type
SwapByteOrder(value_type Value) {
  // No swapping needed.
  return Value;
}

template<typename value_type>
inline
typename enable_if_c<sizeof(value_type) == 2
                     && std::numeric_limits<value_type>::is_integer,
                     value_type>::type
SwapByteOrder(value_type Value) {
  // Cast signed types to unsigned before swapping.
  uint16_t value = static_cast<uint16_t>(Value);
#if defined(_MSC_VER) && !defined(_DEBUG)
  // The DLL version of the runtime lacks these functions (bug!?), but in a
  // release build they're replaced with BSWAP instructions anyway.
  return _byteswap_ushort(value);
#else
  uint16_t Hi = value << 8;
  uint16_t Lo = value >> 8;
  return value_type(Hi | Lo);
#endif
}

template<typename value_type>
inline
typename enable_if_c<sizeof(value_type) == 4
                     && std::numeric_limits<value_type>::is_integer,
                     value_type>::type
SwapByteOrder(value_type Value) {
  // Cast signed types to unsigned before swapping.
  uint32_t value = static_cast<uint32_t>(Value);
#if defined(__llvm__) || \
    (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3)) && !defined(__ICC)
  return __builtin_bswap32(value);
#elif defined(_MSC_VER) && !defined(_DEBUG)
  return _byteswap_ulong(value);
#else
  uint32_t Byte0 = value & 0x000000FF;
  uint32_t Byte1 = value & 0x0000FF00;
  uint32_t Byte2 = value & 0x00FF0000;
  uint32_t Byte3 = value & 0xFF000000;
  return value_type(
    (Byte0 << 24) | (Byte1 << 8) | (Byte2 >> 8) | (Byte3 >> 24));
#endif
}

template<typename value_type>
inline
typename enable_if_c<sizeof(value_type) == 8
                     && std::numeric_limits<value_type>::is_integer,
                     value_type>::type
SwapByteOrder(value_type Value) {
  // Cast signed types to unsigned before swapping.
  uint64_t value = static_cast<uint64_t>(Value);
#if defined(__llvm__) || \
    (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3)) && !defined(__ICC)
  return __builtin_bswap64(value);
#elif defined(_MSC_VER) && !defined(_DEBUG)
  return _byteswap_uint64(value);
#else
  uint64_t Hi = SwapByteOrder<uint32_t>(uint32_t(value));
  uint32_t Lo = SwapByteOrder<uint32_t>(uint32_t(value >> 32));
  return value_type((Hi << 32) | Lo);
#endif
}

} // end namespace sys
} // end namespace llvm

#endif
