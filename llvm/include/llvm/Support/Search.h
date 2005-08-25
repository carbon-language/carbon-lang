//===- llvm/Support/Search.h - Support for searching algorithms -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These templates impliment various generic search algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SEARCH_H
#define LLVM_SUPPORT_SEARCH_H

namespace llvm {
  // SearchString - generalized string compare method container.  Used for
  // search templates.
  struct SearchString {
    static inline int Compare(const char *A, const char *B) {
      return strcmp(A, B);
    }
  };
      
  // LinearSearch - search an array of items for a match using linear search
  // algorithm.  Return find index or -1 if not found.
  //   ItemType - type of elements in array.
  //   CompareClass - container for compare method in form of
  //                  static int Compare(ItemType A, ItemType B)
  //                         returns < 0 for A < B
  //                                 > 0 for A > B
  //                                 == 0 for A == B
  //   Match - item to match in array.
  //   Array - an array of items to be searched.
  //   Size - size of array in bytes.  
  //
  // Eg.  LinearSearch<const char *, SearchString>("key", List, sizeof(List));
  //
  template<typename ItemType, class CompareClass>
  inline int LinearSearch(ItemType Match, ItemType Array[], size_t Size) {
    unsigned N = Size / sizeof(ItemType);
    for (unsigned Index = 0; Index < N; Index++) {
      if (!CompareClass::Compare(Match, Array[Index])) return Index;
    }
    return -1;
  }
      
  // BinarySearch - search an array of items for a match using binary search
  // algorithm.  Return find index or -1 if not found.
  //   ItemType - type of elements in array.
  //   CompareClass - container for compare method in form of
  //                  static int Compare(ItemType A, ItemType B)
  //                         returns < 0 for A < B
  //                                 > 0 for A > B
  //                                 == 0 for A == B
  //   Match - item to match in array.
  //   Array - a sorted array of items to be searched.
  //   Size - size of array in bytes.  
  //
  // Eg.  BinarySearch<const char *, SearchString>("key", List, sizeof(List));
  //
  template<typename ItemType, class CompareClass>
  inline int BinarySearch(ItemType Match, ItemType Array[], size_t Size) {
    int Lo = 0, Hi = Size / sizeof(ItemType);
    while (Lo <= Hi) {
      unsigned Mid = (Lo + Hi) >> 1;
      int Result = CompareClass::Compare(Match, Array[Mid]);
      if      (Result < 0) Hi = Mid - 1;
      else if (Result > 0) Lo = Mid + 1;
      else                 return Mid;
    }
    return -1;
  }
}

#endif
  
