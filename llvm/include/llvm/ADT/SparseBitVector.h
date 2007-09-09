//===- llvm/ADT/SparseBitVector.h - Efficient Sparse BitVector -*- C++ -*- ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Daniel Berlin and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SparseBitVector class.  See the doxygen comment for
// SparseBitVector for more details on the algorithm used.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SPARSEBITVECTOR_H
#define LLVM_ADT_SPARSEBITVECTOR_H

#include <cassert>
#include <cstring>
#include <list>
#include <algorithm>
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace llvm {

/// SparseBitVector is an implementation of a bitvector that is sparse by only
/// storing the elements that have non-zero bits set.  In order to make this
/// fast for the most common cases, SparseBitVector is implemented as a linked
/// list of SparseBitVectorElements.  We maintain a pointer to the last
/// SparseBitVectorElement accessed (in the form of a list iterator), in order
/// to make multiple in-order test/set constant time after the first one is
/// executed.  Note that using vectors to store SparseBitVectorElement's does
/// not work out very well because it causes insertion in the middle to take
/// enormous amounts of time with a large amount of bits.  Other structures that
/// have better worst cases for insertion in the middle (various balanced trees,
/// etc) do not perform as well in practice as a linked list with this iterator
/// kept up to date.  They are also significantly more memory intensive.


template <unsigned ElementSize = 128>
struct SparseBitVectorElement {
public:
  typedef unsigned long BitWord;
  enum {
    BITWORD_SIZE = sizeof(BitWord) * 8,
    BITWORDS_PER_ELEMENT = (ElementSize + BITWORD_SIZE - 1) / BITWORD_SIZE,
    BITS_PER_ELEMENT = ElementSize
  };
private:
  // Index of Element in terms of where first bit starts.
  unsigned ElementIndex;
  BitWord Bits[BITWORDS_PER_ELEMENT];
  SparseBitVectorElement();
public:
  explicit SparseBitVectorElement(unsigned Idx) {
    ElementIndex = Idx;
    memset(&Bits[0], 0, sizeof (BitWord) * BITWORDS_PER_ELEMENT);
  }

  ~SparseBitVectorElement() {
  }

  // Copy ctor.
  SparseBitVectorElement(const SparseBitVectorElement &RHS) {
    ElementIndex = RHS.ElementIndex;
    std::copy(&RHS.Bits[0], &RHS.Bits[BITWORDS_PER_ELEMENT], Bits);
  }

  // Comparison.
  bool operator==(const SparseBitVectorElement &RHS) const {
    if (ElementIndex != RHS.ElementIndex)
      return false;
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i)
      if (Bits[i] != RHS.Bits[i])
        return false;
    return true;
  }

  bool operator!=(const SparseBitVectorElement &RHS) const {
    return !(*this == RHS);
  }

  // Return the bits that make up word Idx in our element.
  BitWord word(unsigned Idx) const {
    assert (Idx < BITWORDS_PER_ELEMENT);
    return Bits[Idx];
  }

  unsigned index() const {
    return ElementIndex;
  }

  bool empty() const {
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i)
      if (Bits[i])
        return false;
    return true;
  }

  void set(unsigned Idx) {
    Bits[Idx / BITWORD_SIZE] |= 1L << (Idx % BITWORD_SIZE);
  }

  bool test_and_set (unsigned Idx) {
    bool old = test(Idx);
    if (!old)
      set(Idx);
    return !old;
  }

  void reset(unsigned Idx) {
    Bits[Idx / BITWORD_SIZE] &= ~(1L << (Idx % BITWORD_SIZE));
  }

  bool test(unsigned Idx) const {
    return Bits[Idx / BITWORD_SIZE] & (1L << (Idx % BITWORD_SIZE));
  }

  unsigned count() const {
    unsigned NumBits = 0;
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i)
      if (sizeof(BitWord) == 4)
        NumBits += CountPopulation_32(Bits[i]);
      else if (sizeof(BitWord) == 8)
        NumBits += CountPopulation_64(Bits[i]);
      else
        assert(0 && "Unsupported!");
    return NumBits;
  }

  /// find_first - Returns the index of the first set bit.
  int find_first() const {
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i)
      if (Bits[i] != 0) {
        if (sizeof(BitWord) == 4)
          return i * BITWORD_SIZE + CountTrailingZeros_32(Bits[i]);
        else if (sizeof(BitWord) == 8)
          return i * BITWORD_SIZE + CountTrailingZeros_64(Bits[i]);
        else
          assert(0 && "Unsupported!");
      }
    assert(0 && "Illegal empty element");
  }

  /// find_next - Returns the index of the next set bit following the
  /// "Prev" bit. Returns -1 if the next set bit is not found.
  int find_next(unsigned Prev) const {
    ++Prev;
    if (Prev >= BITS_PER_ELEMENT)
      return -1;

    unsigned WordPos = Prev / BITWORD_SIZE;
    unsigned BitPos = Prev % BITWORD_SIZE;
    BitWord Copy = Bits[WordPos];
    assert (WordPos <= BITWORDS_PER_ELEMENT
            && "Word Position outside of element");

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
    for (unsigned i = WordPos+1; i < BITWORDS_PER_ELEMENT; ++i)
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

  // Union this element with RHS and return true if this one changed.
  bool unionWith(const SparseBitVectorElement &RHS) {
    bool changed = false;
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i) {
      BitWord old = changed ? 0 : Bits[i];

      Bits[i] |= RHS.Bits[i];
      if (old != Bits[i])
        changed = true;
    }
    return changed;
  }

  // Intersect this Element with RHS and return true if this one changed.
  // BecameZero is set to true if this element became all-zero bits.
  bool intersectWith(const SparseBitVectorElement &RHS,
                     bool &BecameZero) {
    bool changed = false;
    bool allzero = true;

    BecameZero = false;
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i) {
      BitWord old = changed ? 0 : Bits[i];

      Bits[i] &= RHS.Bits[i];
      if (Bits[i] != 0)
        allzero = false;

      if (old != Bits[i])
        changed = true;
    }
    BecameZero = !allzero;
    return changed;
  }
};

template <unsigned ElementSize = 128>
class SparseBitVector {
  typedef std::list<SparseBitVectorElement<ElementSize> *> ElementList;
  typedef typename ElementList::iterator ElementListIter;
  typedef typename ElementList::const_iterator ElementListConstIter;
  enum {
    BITWORD_SIZE = SparseBitVectorElement<ElementSize>::BITWORD_SIZE
  };

  // Pointer to our current Element.
  ElementListIter CurrElementIter;
  ElementList Elements;

  // This is like std::lower_bound, except we do linear searching from the
  // current position.
  ElementListIter FindLowerBound(unsigned ElementIndex) {

    if (Elements.empty()) {
      CurrElementIter = Elements.begin();
      return Elements.begin();
    }

    // Make sure our current iterator is valid.
    if (CurrElementIter == Elements.end())
      --CurrElementIter;

    // Search from our current iterator, either backwards or forwards,
    // depending on what element we are looking for.
    ElementListIter ElementIter = CurrElementIter;
    if ((*CurrElementIter)->index() == ElementIndex) {
      return ElementIter;
    } else if ((*CurrElementIter)->index() > ElementIndex) {
      while (ElementIter != Elements.begin()
             && (*ElementIter)->index() > ElementIndex)
        --ElementIter;
    } else {
      while (ElementIter != Elements.end() &&
             (*ElementIter)->index() <= ElementIndex)
        ++ElementIter;
      --ElementIter;
    }
    CurrElementIter = ElementIter;
    return ElementIter;
  }

  // Iterator to walk set bits in the bitmap.  This iterator is a lot uglier
  // than it would be, in order to be efficient.
  struct SparseBitVectorIterator {
  private:
    bool AtEnd;

    SparseBitVector<ElementSize> &BitVector;

    // Current element inside of bitmap.
    ElementListConstIter Iter;

    // Current bit number inside of our bitmap.
    unsigned BitNumber;

    // Current word number inside of our element.
    unsigned WordNumber;

    // Current bits from the element.
    typename SparseBitVectorElement<ElementSize>::BitWord Bits;

    // Move our iterator to the first non-zero bit in the bitmap.
    void AdvanceToFirstNonZero() {
      if (AtEnd)
        return;
      if (BitVector.Elements.empty()) {
        AtEnd = true;
        return;
      }
      Iter = BitVector.Elements.begin();
      BitNumber = (*Iter)->index() * ElementSize;
      unsigned BitPos = (*Iter)->find_first();
      BitNumber += BitPos;
      WordNumber = (BitNumber % ElementSize) / BITWORD_SIZE;
      Bits = (*Iter)->word(WordNumber);
      Bits >>= BitPos % BITWORD_SIZE;
    }

    // Move our iterator to the next non-zero bit.
    void AdvanceToNextNonZero() {
      if (AtEnd)
        return;

      while (Bits && !(Bits & 1)) {
        Bits >>= 1;
        BitNumber += 1;
      }

      // See if we ran out of Bits in this word.
      if (!Bits) {
        int NextSetBitNumber = (*Iter)->find_next(BitNumber % ElementSize) ;
        // If we ran out of set bits in this element, move to next element.
        if (NextSetBitNumber == -1 || (BitNumber % ElementSize == 0)) {
          Iter++;
          WordNumber = 0;

          // We may run out of elements in the bitmap.
          if (Iter == BitVector.Elements.end()) {
            AtEnd = true;
            return;
          }
          // Set up for next non zero word in bitmap.
          BitNumber = (*Iter)->index() * ElementSize;
          NextSetBitNumber = (*Iter)->find_first();
          BitNumber += NextSetBitNumber;
          WordNumber = (BitNumber % ElementSize) / BITWORD_SIZE;
          Bits = (*Iter)->word(WordNumber);
          Bits >>= NextSetBitNumber % BITWORD_SIZE;
        } else {
          WordNumber = (NextSetBitNumber % ElementSize) / BITWORD_SIZE;
          Bits = (*Iter)->word(WordNumber);
          Bits >>= NextSetBitNumber % BITWORD_SIZE;
        }
      }
    }
  public:
    // Preincrement.
    inline SparseBitVectorIterator& operator++() {
      BitNumber++;
      Bits >>= 1;
      AdvanceToNextNonZero();
      return *this;
    }

    // Postincrement.
    inline SparseBitVectorIterator operator++(int) {
      SparseBitVectorIterator tmp = *this;
      ++*this;
      return tmp;
    }

    // Return the current set bit number.
    unsigned operator*() const {
      return BitNumber;
    }

    bool operator==(const SparseBitVectorIterator &RHS) const {
      // If they are both at the end, ignore the rest of the fields.
      if (AtEnd == RHS.AtEnd)
        return true;
      // Otherwise they are the same if they have the same bit number and
      // bitmap.
      return AtEnd == RHS.AtEnd && RHS.BitNumber == BitNumber;
    }
    bool operator!=(const SparseBitVectorIterator &RHS) const {
      return !(*this == RHS);
    }

    explicit SparseBitVectorIterator(SparseBitVector<ElementSize> &RHS,
                                     bool end = false):BitVector(RHS) {
      Iter = BitVector.Elements.begin();
      BitNumber = 0;
      Bits = 0;
      WordNumber = ~0;
      AtEnd = end;
      AdvanceToFirstNonZero();
    }
  };
public:
  typedef SparseBitVectorIterator iterator;
  typedef const SparseBitVectorIterator const_iterator;

  SparseBitVector () {
    CurrElementIter = Elements.begin ();
  }

  ~SparseBitVector() {
    for_each(Elements.begin(), Elements.end(),
             deleter<SparseBitVectorElement<ElementSize> >);
  }

  // SparseBitVector copy ctor.
  SparseBitVector(const SparseBitVector &RHS) {
    ElementListConstIter ElementIter = RHS.Elements.begin();
    while (ElementIter != RHS.Elements.end()) {
      SparseBitVectorElement<ElementSize> *ElementCopy;
      ElementCopy = new SparseBitVectorElement<ElementSize>(*(*ElementIter));
      Elements.push_back(ElementCopy);
    }

    CurrElementIter = Elements.begin ();
  }

  // Test, Reset, and Set a bit in the bitmap.
  bool test(unsigned Idx) {
    if (Elements.empty())
      return false;

    unsigned ElementIndex = Idx / ElementSize;
    ElementListIter ElementIter = FindLowerBound(ElementIndex);

    // If we can't find an element that is supposed to contain this bit, there
    // is nothing more to do.
    if (ElementIter == Elements.end() ||
        (*ElementIter)->index() != ElementIndex)
      return false;
    return (*ElementIter)->test(Idx % ElementSize);
  }

  void reset(unsigned Idx) {
    if (Elements.empty())
      return;

    unsigned ElementIndex = Idx / ElementSize;
    ElementListIter ElementIter = FindLowerBound(ElementIndex);

    // If we can't find an element that is supposed to contain this bit, there
    // is nothing more to do.
    if (ElementIter == Elements.end() ||
        (*ElementIter)->index() != ElementIndex)
      return;
    (*ElementIter)->reset(Idx % ElementSize);

    // When the element is zeroed out, delete it.
    if ((*ElementIter)->empty()) {
      delete (*ElementIter);
      ++CurrElementIter;
      Elements.erase(ElementIter);
    }
  }

  void set(unsigned Idx) {
    SparseBitVectorElement<ElementSize> *Element;
    unsigned ElementIndex = Idx / ElementSize;

    if (Elements.empty()) {
      Element = new SparseBitVectorElement<ElementSize>(ElementIndex);
      Elements.push_back(Element);
    } else {
      ElementListIter ElementIter = FindLowerBound(ElementIndex);

      if (ElementIter != Elements.end() &&
          (*ElementIter)->index() == ElementIndex)
        Element = *ElementIter;
      else {
        Element = new SparseBitVectorElement<ElementSize>(ElementIndex);
        // Insert does insert before, and lower bound gives the one before.
        Elements.insert(++ElementIter, Element);
      }
    }
    Element->set(Idx % ElementSize);
  }

  // Union our bitmap with the RHS and return true if we changed.
  bool operator|=(const SparseBitVector &RHS) {
    bool changed = false;
    ElementListIter Iter1 = Elements.begin();
    ElementListConstIter Iter2 = RHS.Elements.begin();

    // IE They may both be end
    if (Iter1 == Iter2)
      return false;

    // See if the first bitmap element is the same in both.  This is only
    // possible if they are the same bitmap.
    if (Iter1 != Elements.end() && Iter2 != RHS.Elements.end())
      if (*Iter1 == *Iter2)
        return false;

    while (Iter2 != RHS.Elements.end()) {
      if (Iter1 == Elements.end() || (*Iter1)->index() > (*Iter2)->index()) {
        SparseBitVectorElement<ElementSize> *NewElem;

        NewElem = new SparseBitVectorElement<ElementSize>(*(*Iter2));
        Elements.insert(Iter1, NewElem);
        Iter2++;
        changed = true;
      } else if ((*Iter1)->index() == (*Iter2)->index()) {
        changed |= (*Iter1)->unionWith(*(*Iter2));
        Iter1++;
        Iter2++;
      } else {
        Iter1++;
      }
    }
    CurrElementIter = Elements.begin();
    return changed;
  }

  // Intersect our bitmap with the RHS and return true if ours changed.
  bool operator&=(const SparseBitVector &RHS) {
    bool changed = false;
    ElementListIter Iter1 = Elements.begin();
    ElementListConstIter Iter2 = RHS.Elements.begin();

    // IE They may both be end.
    if (Iter1 == Iter2)
      return false;

    // See if the first bitmap element is the same in both.  This is only
    // possible if they are the same bitmap.
    if (Iter1 != Elements.end() && Iter2 != RHS.Elements.end())
      if (*Iter1 == *Iter2)
        return false;

    // Loop through, intersecting as we go, erasing elements when necessary.
    while (Iter2 != RHS.Elements.end()) {
      if (Iter1 == Elements.end())
        return changed;

      if ((*Iter1)->index() > (*Iter2)->index()) {
        Iter2++;
      } else if ((*Iter1)->index() == (*Iter2)->index()) {
        bool BecameZero;
        changed |= (*Iter1)->intersectWith(*(*Iter2), BecameZero);
        if (BecameZero) {
          ElementListIter IterTmp = Iter1;
          delete *IterTmp;
          Elements.erase(IterTmp);
          Iter1++;
        } else {
          Iter1++;
        }
        Iter2++;
      } else {
        ElementListIter IterTmp = Iter1;
        Iter1++;
        delete *IterTmp;
        Elements.erase(IterTmp);
      }
    }
    CurrElementIter = Elements.begin();
    return changed;
  }

  iterator begin() const {
    return iterator(*this);
  }

  iterator end() const {
    return iterator(*this, ~0);
  }
};
}

#endif
