//===-- llvm/ADT/SortedVector.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SORTEDVECTOR_H
#define LLVM_ADT_SORTEDVECTOR_H

#include <vector>
#include <cassert>
#include <functional>
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// \brief Lazily maintains a sorted and unique vector of elements of type T.
template<typename T, typename CMP = std::less<T>>
class SortedVector {
public:
  typedef typename std::vector<T> VectorType;
  typedef typename VectorType::iterator iterator;
  typedef typename VectorType::const_iterator const_iterator;

private:
  VectorType Vector;
  bool IsSorted = true;

  void doCheck() const {
    assert(IsSorted && "Unsorted SortedVector access; call sortUnique prior.");
  }

public:
  /// \brief Appends Entry to the sorted unique vector; sets the IsSorted flag
  /// to false if appending Entry puts Vector into an unsorted state.
  void insert(const T &Entry) {
    if (!Vector.size())
      Vector.push_back(Entry);

    // Vector is sorted and Entry is a duplicate of the previous so skip.
    if (IsSorted && Entry == Vector.back())
      return;

    IsSorted &= (CMP()(Vector.back(), Entry));
    Vector.push_back(Entry);
  }

  // \brief Sorts and uniques Vector.
  void sortUnique() {
    if (IsSorted)
      return;

    std::sort(Vector.begin(), Vector.end());
    Vector.erase(std::unique(Vector.begin(), Vector.end()), Vector.end());
    IsSorted = true;
  }

  /// \brief Tells if Entry is in Vector without relying on sorted-uniqueness.
  bool has(T Entry) const {
    if (IsSorted)
      return std::binary_search(Vector.begin(), Vector.end(), Entry);

    return std::find(Vector.begin(), Vector.end(), Entry) != Vector.end();
  }

  /// \brief Returns a reference to the entry with the specified index.
  const T &operator[](unsigned index) const {
    assert(index < size() && "SortedVector index is out of range!");
    doCheck();
    return Vector[index];
  }

  /// \brief Return an iterator to the start of the vector.
  iterator begin() {
    doCheck();
    return Vector.begin();
  }

  /// \brief Returns const iterator to the start of the vector.
  const_iterator begin() const {
    doCheck();
    return Vector.begin();
  }

  /// \brief Returns iterator to the end of Vector.
  iterator end() {
    doCheck();
    return Vector.end();
  }

  /// \brief Returns const iterator to the end of Vector. Assert if unsorted.
  const_iterator end() const {
    doCheck();
    return Vector.end();
  }

  /// \brief Erases Vector at position. Asserts if Vector is unsorted.
  iterator erase(iterator position) {
    doCheck();
    return Vector.erase(position);
  }

  /// \brief Erases Vector entirely.
  iterator erase() {
    IsSorted = true;
    return Vector.erase();
  }

  /// \brief Returns number of entries in Vector; asserts if it is unsorted.
  size_t size() const {
    doCheck();
    return Vector.size();
  }

  /// \brief Returns true if Vector is empty.
  bool empty() const {
    return Vector.empty();
  }

  /// \brief Clears all the entries.
  void reset() {
    IsSorted = true;
    Vector.resize(0, 0);
  }
};

} // End of namespace llvm

#endif // LLVM_ADT_SORTEDVECTOR_H

