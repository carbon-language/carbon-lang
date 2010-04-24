//===-- UsuallyTinyPtrVector.h - Pointer vector class -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the UsuallyTinyPtrVector class, which is a vector that
//  optimizes the case where there is only one element.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_USUALLY_TINY_PTR_VECTOR_H
#define LLVM_CLANG_AST_USUALLY_TINY_PTR_VECTOR_H

#include <vector>

namespace clang {

/// \brief A vector class template that is optimized for storing a single 
/// pointer element.
template<typename T>
class UsuallyTinyPtrVector {
  /// \brief Storage for the vector.
  ///
  /// When the low bit is zero, this is a T *. When the
  /// low bit is one, this is a std::vector<T *> *.
  mutable uintptr_t Storage;

  typedef std::vector<T*> vector_type;

public:
  UsuallyTinyPtrVector() : Storage(0) { }
  explicit UsuallyTinyPtrVector(T *Element) 
    : Storage(reinterpret_cast<uintptr_t>(Element)) { }
  
  bool empty() const { return !Storage; }

  typedef const T **iterator;
  iterator begin() const;
  iterator end() const;

  void push_back(T *Method);
  void Destroy();
};

template<typename T>
typename UsuallyTinyPtrVector<T>::iterator 
UsuallyTinyPtrVector<T>::begin() const {
  if ((Storage & 0x01) == 0)
    return reinterpret_cast<iterator>(&Storage);

  vector_type *Vec = reinterpret_cast<vector_type *>(Storage & ~0x01);
  return &Vec->front();
}


template<typename T>
typename UsuallyTinyPtrVector<T>::iterator 
UsuallyTinyPtrVector<T>::end() const {
  if ((Storage & 0x01) == 0) {
    if (Storage == 0)
      return reinterpret_cast<iterator>(&Storage);

    return reinterpret_cast<iterator>(&Storage) + 1;
  }

  vector_type *Vec = reinterpret_cast<vector_type *>(Storage & ~0x01);
  return &Vec->front() + Vec->size();
}

template<typename T>
void UsuallyTinyPtrVector<T>::push_back(T *Element) {
  if (Storage == 0) {
    // 0 -> 1 element.
    Storage = reinterpret_cast<uintptr_t>(Element);
    return;
  }

  vector_type *Vec;
  if ((Storage & 0x01) == 0) {
    // 1 -> 2 elements. Allocate a new vector and push the element into that
    // vector.
    Vec = new vector_type;
    Vec->push_back(reinterpret_cast<T *>(Storage));
    Storage = reinterpret_cast<uintptr_t>(Vec) | 0x01;
  } else
    Vec = reinterpret_cast<vector_type *>(Storage & ~0x01);

  // Add the new element to the vector.
  Vec->push_back(Element);
}

template<typename T>
void UsuallyTinyPtrVector<T>::Destroy() {
  if (Storage & 0x01)
    delete reinterpret_cast<vector_type *>(Storage & ~0x01);
  
  Storage = 0;
}

}
#endif 
