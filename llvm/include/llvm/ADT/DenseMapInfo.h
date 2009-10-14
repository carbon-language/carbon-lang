//===- llvm/ADT/DenseMapInfo.h - Type traits for DenseMap -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines DenseMapInfo traits for DenseMap.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_DENSEMAPINFO_H
#define LLVM_ADT_DENSEMAPINFO_H

#include "llvm/Support/PointerLikeTypeTraits.h"
#include <utility>

namespace llvm {

template<typename T>
struct DenseMapInfo {
  //static inline T getEmptyKey();
  //static inline T getTombstoneKey();
  //static unsigned getHashValue(const T &Val);
  //static bool isEqual(const T &LHS, const T &RHS);
  //static bool isPod()
};

// Provide DenseMapInfo for all pointers.
template<typename T>
struct DenseMapInfo<T*> {
  static inline T* getEmptyKey() {
    intptr_t Val = -1;
    Val <<= PointerLikeTypeTraits<T*>::NumLowBitsAvailable;
    return reinterpret_cast<T*>(Val);
  }
  static inline T* getTombstoneKey() {
    intptr_t Val = -2;
    Val <<= PointerLikeTypeTraits<T*>::NumLowBitsAvailable;
    return reinterpret_cast<T*>(Val);
  }
  static unsigned getHashValue(const T *PtrVal) {
    return (unsigned((uintptr_t)PtrVal) >> 4) ^
           (unsigned((uintptr_t)PtrVal) >> 9);
  }
  static bool isEqual(const T *LHS, const T *RHS) { return LHS == RHS; }
  static bool isPod() { return true; }
};

// Provide DenseMapInfo for chars.
template<> struct DenseMapInfo<char> {
  static inline char getEmptyKey() { return ~0; }
  static inline char getTombstoneKey() { return ~0 - 1; }
  static unsigned getHashValue(const char& Val) { return Val * 37; }
  static bool isPod() { return true; }
  static bool isEqual(const char &LHS, const char &RHS) {
    return LHS == RHS;
  }
};
  
// Provide DenseMapInfo for unsigned ints.
template<> struct DenseMapInfo<unsigned> {
  static inline unsigned getEmptyKey() { return ~0; }
  static inline unsigned getTombstoneKey() { return ~0U - 1; }
  static unsigned getHashValue(const unsigned& Val) { return Val * 37; }
  static bool isPod() { return true; }
  static bool isEqual(const unsigned& LHS, const unsigned& RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for unsigned longs.
template<> struct DenseMapInfo<unsigned long> {
  static inline unsigned long getEmptyKey() { return ~0UL; }
  static inline unsigned long getTombstoneKey() { return ~0UL - 1L; }
  static unsigned getHashValue(const unsigned long& Val) {
    return Val * 37UL;
  }
  static bool isPod() { return true; }
  static bool isEqual(const unsigned long& LHS, const unsigned long& RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for unsigned long longs.
template<> struct DenseMapInfo<unsigned long long> {
  static inline unsigned long long getEmptyKey() { return ~0ULL; }
  static inline unsigned long long getTombstoneKey() { return ~0ULL - 1ULL; }
  static unsigned getHashValue(const unsigned long long& Val) {
    return (unsigned)Val * 37ULL;
  }
  static bool isPod() { return true; }
  static bool isEqual(const unsigned long long& LHS,
                      const unsigned long long& RHS) {
    return LHS == RHS;
  }
};

// Provide DenseMapInfo for all pairs whose members have info.
template<typename T, typename U>
struct DenseMapInfo<std::pair<T, U> > {
  typedef std::pair<T, U> Pair;
  typedef DenseMapInfo<T> FirstInfo;
  typedef DenseMapInfo<U> SecondInfo;

  static inline Pair getEmptyKey() {
    return std::make_pair(FirstInfo::getEmptyKey(),
                          SecondInfo::getEmptyKey());
  }
  static inline Pair getTombstoneKey() {
    return std::make_pair(FirstInfo::getTombstoneKey(),
                            SecondInfo::getEmptyKey());
  }
  static unsigned getHashValue(const Pair& PairVal) {
    uint64_t key = (uint64_t)FirstInfo::getHashValue(PairVal.first) << 32
          | (uint64_t)SecondInfo::getHashValue(PairVal.second);
    key += ~(key << 32);
    key ^= (key >> 22);
    key += ~(key << 13);
    key ^= (key >> 8);
    key += (key << 3);
    key ^= (key >> 15);
    key += ~(key << 27);
    key ^= (key >> 31);
    return (unsigned)key;
  }
  static bool isEqual(const Pair& LHS, const Pair& RHS) { return LHS == RHS; }
  static bool isPod() { return FirstInfo::isPod() && SecondInfo::isPod(); }
};

} // end namespace llvm

#endif
