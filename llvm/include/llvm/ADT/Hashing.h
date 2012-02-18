//===-- llvm/ADT/Hashing.h - Utilities for hashing --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for computing hash values for various data types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_HASHING_H
#define LLVM_ADT_HASHING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

/// Class to compute a hash value from multiple data fields of arbitrary
/// types. Note that if you are hashing a single data type, such as a
/// string, it may be cheaper to use a hash algorithm that is tailored
/// for that specific data type.
/// Typical Usage:
///    GeneralHash Hash;
///    Hash.add(someValue);
///    Hash.add(someOtherValue);
///    return Hash.finish();
/// Adapted from MurmurHash2 by Austin Appleby
class GeneralHash {
private:
  enum {
    M = 0x5bd1e995
  };
  unsigned Hash;
  unsigned Count;
public:
  GeneralHash(unsigned Seed = 0) : Hash(Seed), Count(0) {}

  /// Add a pointer value.
  /// Note: this adds pointers to the hash using sizes and endianness that
  /// depend on the host.  It doesn't matter however, because hashing on
  /// pointer values is inherently unstable.
  template<typename T>
  GeneralHash& add(const T *PtrVal) {
    addBits(&PtrVal, &PtrVal + 1);
    return *this;
  }

  /// Add an ArrayRef of arbitrary data.
  template<typename T>
  GeneralHash& add(ArrayRef<T> ArrayVal) {
    addBits(ArrayVal.begin(), ArrayVal.end());
    return *this;
  }

  /// Add a string
  GeneralHash& add(StringRef StrVal) {
    addBits(StrVal.begin(), StrVal.end());
    return *this;
  }

  /// Add an signed 32-bit integer.
  GeneralHash& add(int32_t Data) {
    addInt(uint32_t(Data));
    return *this;
  }

  /// Add an unsigned 32-bit integer.
  GeneralHash& add(uint32_t Data) {
    addInt(Data);
    return *this;
  }

  /// Add an signed 64-bit integer.
  GeneralHash& add(int64_t Data) {
    addInt(uint64_t(Data));
    return *this;
  }

  /// Add an unsigned 64-bit integer.
  GeneralHash& add(uint64_t Data) {
    addInt(Data);
    return *this;
  }

  /// Add a float
  GeneralHash& add(float Data) {
    union {
      float D; uint32_t I;
    };
    D = Data;
    addInt(I);
    return *this;
  }

  /// Add a double
  GeneralHash& add(double Data) {
    union {
      double D; uint64_t I;
    };
    D = Data;
    addInt(I);
    return *this;
  }

  // Do a few final mixes of the hash to ensure the last few
  // bytes are well-incorporated.
  unsigned finish() {
    mix(Count);
    Hash ^= Hash >> 13;
    Hash *= M;
    Hash ^= Hash >> 15;
    return Hash;
  }

private:
  void mix(uint32_t Data) {
    ++Count;
    Data *= M;
    Data ^= Data >> 24;
    Data *= M;
    Hash *= M;
    Hash ^= Data;
  }

  // Add a single uint32 value
  void addInt(uint32_t Val) {
    mix(Val);
  }

  // Add a uint64 value
  void addInt(uint64_t Val) {
    mix(uint32_t(Val >> 32));
    mix(uint32_t(Val));
  }

  template<typename T, bool isAligned>
  struct addBitsImpl {
    static void add(GeneralHash &Hash, const T *I, const T *E) {
      Hash.addUnaligned(
        reinterpret_cast<const uint8_t *>(I),
        reinterpret_cast<const uint8_t *>(E));
    }
  };

  template<typename T>
  struct addBitsImpl<T, true> {
    static void add(GeneralHash &Hash, const T *I, const T *E) {
      Hash.addAligned(
        reinterpret_cast<const uint32_t *>(I),
        reinterpret_cast<const uint32_t *>(E));
    }
  };

  // Add a range of bits from I to E.
  template<typename T>
  void addBits(const T *I, const T *E) {
    addBitsImpl<T, AlignOf<T>::Alignment_GreaterEqual_4Bytes>::add(*this, I, E);
  }

  // Add a range of uint32s
  void addAligned(const uint32_t *I, const uint32_t *E) {
    while (I < E) {
      mix(*I++);
    }
  }

  // Add a possibly unaligned sequence of bytes.
  void addUnaligned(const uint8_t *I, const uint8_t *E);
};

} // end namespace llvm

#endif
