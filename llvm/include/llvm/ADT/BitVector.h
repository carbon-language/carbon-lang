//===- llvm/ADT/BitVector.h - Bit vectors -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the BitVector class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_BITVECTOR_H
#define LLVM_ADT_BITVECTOR_H

#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdlib>
#include <cassert>

namespace llvm {

class BitVector {
  typedef unsigned long BitWord;

  enum { BITWORD_SIZE = sizeof(BitWord) * 8 };

  BitWord  *Bits;        // Actual bits. 
  unsigned Size;         // Size of bitvector in bits.
  unsigned Capacity;     // Size of allocated memory in BitWord.

public:
  // Encapsulation of a single bit.
  class reference {
    friend class BitVector;

    BitWord *WordRef;
    unsigned BitPos;

    reference();  // Undefined

  public:
    reference(BitVector &b, unsigned Idx) {
      WordRef = &b.Bits[Idx / BITWORD_SIZE];
      BitPos = Idx % BITWORD_SIZE;
    }

    ~reference() {}

    reference& operator=(bool t) {
      if (t)
        *WordRef |= 1L << BitPos;
      else
        *WordRef &= ~(1L << BitPos);
      return *this;
    }

    operator bool() const {
      return (*WordRef) & (1L << BitPos);
    }
  };


  /// BitVector default ctor - Creates an empty bitvector.
  BitVector() : Size(0), Capacity(0) {
    Bits = NULL;
  }

  /// BitVector ctor - Creates a bitvector of specified number of bits. All
  /// bits are initialized to the specified value.
  explicit BitVector(unsigned s, bool t = false) : Size(s) {
    Capacity = NumBitWords(s);
    Bits = new BitWord[Capacity];
    init_words(Bits, Capacity, t);
    if (t)
      clear_unused_bits();
  }

  /// BitVector copy ctor.
  BitVector(const BitVector &RHS) : Size(RHS.size()) {
    if (Size == 0) {
      Bits = NULL;
      Capacity = 0;
      return;
    }

    Capacity = NumBitWords(RHS.size());
    Bits = new BitWord[Capacity];
    std::copy(RHS.Bits, &RHS.Bits[Capacity], Bits);
  }
  
  ~BitVector() {
    delete[] Bits;
  }

  /// size - Returns the number of bits in this bitvector.
  unsigned size() const { return Size; }

  /// count - Returns the number of bits which are set.
  unsigned count() const {
    unsigned NumBits = 0;
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      if (sizeof(BitWord) == 4)
        NumBits += CountPopulation_32(Bits[i]);
      else if (sizeof(BitWord) == 8)
        NumBits += CountPopulation_64(Bits[i]);
      else
        assert(0 && "Unsupported!");
    return NumBits;
  }

  /// any - Returns true if any bit is set.
  bool any() const {
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      if (Bits[i] != 0)
        return true;
    return false;
  }

  /// none - Returns true if none of the bits are set.
  bool none() const {
    return !any();
  }

  /// find_first - Returns the index of the first set bit, -1 if none
  /// of the bits are set.
  int find_first() const {
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      if (Bits[i] != 0) {
        if (sizeof(BitWord) == 4)
          return i * BITWORD_SIZE + CountTrailingZeros_32(Bits[i]);
        else if (sizeof(BitWord) == 8)
          return i * BITWORD_SIZE + CountTrailingZeros_64(Bits[i]);
        else
          assert(0 && "Unsupported!");
      }
    return -1;
  }

  /// find_next - Returns the index of the next set bit following the
  /// "Prev" bit. Returns -1 if the next set bit is not found.
  int find_next(unsigned Prev) const {
    ++Prev;
    if (Prev >= Size)
      return -1;

    unsigned WordPos = Prev / BITWORD_SIZE;
    unsigned BitPos = Prev % BITWORD_SIZE;
    BitWord Copy = Bits[WordPos];
    // Mask off previous bits.
    Copy &= ~0L << BitPos;

    if (Copy != 0) {
      if (sizeof(BitWord) == 4)
        return WordPos * BITWORD_SIZE + CountTrailingZeros_32(Copy);
      else if (sizeof(BitWord) == 8)
        return WordPos * BITWORD_SIZE + CountTrailingZeros_64(Copy);
      else
        assert(0 && "Unsupported!");
    }

    // Check subsequent words.
    for (unsigned i = WordPos+1; i < NumBitWords(size()); ++i)
      if (Bits[i] != 0) {
        if (sizeof(BitWord) == 4)
          return i * BITWORD_SIZE + CountTrailingZeros_32(Bits[i]);
        else if (sizeof(BitWord) == 8)
          return i * BITWORD_SIZE + CountTrailingZeros_64(Bits[i]);
        else
          assert(0 && "Unsupported!");
      }
    return -1;
  }

  /// clear - Clear all bits.
  void clear() {
    Size = 0;
  }

  /// resize - Grow or shrink the bitvector.
  void resize(unsigned N, bool t = false) {
    if (N > Capacity * BITWORD_SIZE) {
      unsigned OldCapacity = Capacity;
      grow(N);
      init_words(&Bits[OldCapacity], (Capacity-OldCapacity), t);
    }
    Size = N;
    clear_unused_bits();
  }

  void reserve(unsigned N) {
    if (N > Capacity * BITWORD_SIZE)
      grow(N);
  }

  // Set, reset, flip
  BitVector &set() {
    init_words(Bits, Capacity, true);
    clear_unused_bits();
    return *this;
  }

  BitVector &set(unsigned Idx) {
    Bits[Idx / BITWORD_SIZE] |= 1L << (Idx % BITWORD_SIZE);
    return *this;
  }

  BitVector &reset() {
    init_words(Bits, Capacity, false);
    return *this;
  }

  BitVector &reset(unsigned Idx) {
    Bits[Idx / BITWORD_SIZE] &= ~(1L << (Idx % BITWORD_SIZE));
    return *this;
  }

  BitVector &flip() {
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      Bits[i] = ~Bits[i];
    clear_unused_bits();
    return *this;
  }

  BitVector &flip(unsigned Idx) {
    Bits[Idx / BITWORD_SIZE] ^= 1L << (Idx % BITWORD_SIZE);
    return *this;
  }

  // No argument flip.
  BitVector operator~() const {
    return BitVector(*this).flip();
  }

  // Indexing.
  reference operator[](unsigned Idx) {
    return reference(*this, Idx);
  }

  bool operator[](unsigned Idx) const {
    BitWord Mask = 1L << (Idx % BITWORD_SIZE);
    return (Bits[Idx / BITWORD_SIZE] & Mask) != 0;
  }

  bool test(unsigned Idx) const {
    return (*this)[Idx];
  }

  // Comparison operators.
  bool operator==(const BitVector &RHS) const {
    if (Size != RHS.Size)
      return false;

    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      if (Bits[i] != RHS.Bits[i])
        return false;
    return true;
  }

  bool operator!=(const BitVector &RHS) const {
    return !(*this == RHS);
  }

  // Intersection, union, disjoint union.
  BitVector operator&=(const BitVector &RHS) {
    assert(Size == RHS.Size && "Illegal operation!");
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      Bits[i] &= RHS.Bits[i];
    return *this;
  }

  BitVector operator|=(const BitVector &RHS) {
    assert(Size == RHS.Size && "Illegal operation!");
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      Bits[i] |= RHS.Bits[i];
    return *this;
  }

  BitVector operator^=(const BitVector &RHS) {
    assert(Size == RHS.Size && "Illegal operation!");
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      Bits[i] ^= RHS.Bits[i];
    return *this;
  }
  
  // Assignment operator.
  const BitVector &operator=(const BitVector &RHS) {
    if (this == &RHS) return *this;

    Size = RHS.size();
    unsigned RHSWords = NumBitWords(Size);
    if (Size <= Capacity * BITWORD_SIZE) {
      std::copy(RHS.Bits, &RHS.Bits[RHSWords], Bits);
      clear_unused_bits();
      return *this;
    }
  
    // Grow the bitvector to have enough elements.
    Capacity = NumBitWords(Size);
    BitWord *NewBits = new BitWord[Capacity];
    std::copy(RHS.Bits, &RHS.Bits[RHSWords], NewBits);

    // Destroy the old bits.
    delete[] Bits;
    Bits = NewBits;

    return *this;
  }

private:
  unsigned NumBitWords(unsigned S) const {
    return (S + BITWORD_SIZE-1) / BITWORD_SIZE;
  }

  // Clear the unused top bits in the high word.
  void clear_unused_bits() {
    unsigned ExtraBits = Size % BITWORD_SIZE;
    if (ExtraBits) {
      unsigned index = Size / BITWORD_SIZE;
      Bits[index] &= ~(~0L << ExtraBits);
    }
  }

  void grow(unsigned NewSize) {
    unsigned OldCapacity = Capacity;
    Capacity = NumBitWords(NewSize);
    BitWord *NewBits = new BitWord[Capacity];

    // Copy the old bits over.
    if (OldCapacity != 0)
      std::copy(Bits, &Bits[OldCapacity], NewBits);

    // Destroy the old bits.
    delete[] Bits;
    Bits = NewBits;
  }

  void init_words(BitWord *B, unsigned NumWords, bool t) {
    memset(B, 0 - (int)t, NumWords*sizeof(BitWord));
  } 
};

inline BitVector operator&(const BitVector &LHS, const BitVector &RHS) {
  BitVector Result(LHS);
  Result &= RHS;
  return Result;
}

inline BitVector operator|(const BitVector &LHS, const BitVector &RHS) {
  BitVector Result(LHS);
  Result |= RHS;
  return Result;
}

inline BitVector operator^(const BitVector &LHS, const BitVector &RHS) {
  BitVector Result(LHS);
  Result ^= RHS;
  return Result;
}
 
} // End llvm namespace
#endif
