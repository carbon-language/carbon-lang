//===- llvm/ADT/SmallVector.h - 'Normally small' vectors --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SmallVector class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SMALLVECTOR_H
#define LLVM_ADT_SMALLVECTOR_H

#include <algorithm>
#include <iterator>
#include <memory>

namespace llvm {

/// SmallVectorImpl - This class consists of common code factored out of the
/// SmallVector class to reduce code duplication based on the SmallVector 'N'
/// template parameter.
template <typename T>
class SmallVectorImpl {
  T *Begin, *End, *Capacity;
  
  // Allocate raw space for N elements of type T.  If T has a ctor or dtor, we
  // don't want it to be automatically run, so we need to represent the space as
  // something else.  An array of char would work great, but might not be
  // aligned sufficiently.  Instead, we either use GCC extensions, or some
  // number of union instances for the space, which guarantee maximal alignment.
protected:
  union U {
    double D;
    long double LD;
    long long L;
    void *P;
  } FirstEl;
  // Space after 'FirstEl' is clobbered, do not add any instance vars after it.
public:
  // Default ctor - Initialize to empty.
  SmallVectorImpl(unsigned N)
    : Begin((T*)&FirstEl), End((T*)&FirstEl), Capacity((T*)&FirstEl+N) {
  }
  
  ~SmallVectorImpl() {
    // Destroy the constructed elements in the vector.
    for (iterator I = Begin, E = End; I != E; ++I)
      I->~T();

    // If this wasn't grown from the inline copy, deallocate the old space.
    if (!isSmall())
      delete[] (char*)Begin;
  }
  
  typedef size_t size_type;
  typedef T* iterator;
  typedef const T* const_iterator;
  typedef T& reference;
  typedef const T& const_reference;

  bool empty() const { return Begin == End; }
  size_type size() const { return End-Begin; }
  
  iterator begin() { return Begin; }
  const_iterator begin() const { return Begin; }

  iterator end() { return End; }
  const_iterator end() const { return End; }
  
  reference operator[](unsigned idx) {
    return Begin[idx];
  }
  const_reference operator[](unsigned idx) const {
    return Begin[idx];
  }
  
  reference back() {
    return end()[-1];
  }
  const_reference back() const {
    return end()[-1];
  }
  
  void push_back(const_reference Elt) {
    if (End < Capacity) {
  Retry:
      new (End) T(Elt);
      ++End;
      return;
    }
    grow();
    goto Retry;
  }
  
  void pop_back() {
    --End;
    End->~T();
  }
  
  void clear() {
    while (End != Begin) {
      End->~T();
      --End;
    }
  }
  
  /// append - Add the specified range to the end of the SmallVector.
  ///
  template<typename in_iter>
  void append(in_iter in_start, in_iter in_end) {
    unsigned NumInputs = std::distance(in_start, in_end);
    // Grow allocated space if needed.
    if (End+NumInputs > Capacity)
      grow(size()+NumInputs);

    // Copy the new elements over.
    std::uninitialized_copy(in_start, in_end, End);
    End += NumInputs;
  }
  
  void assign(unsigned NumElts, const T &Elt) {
    clear();
    if (Begin+NumElts > Capacity)
      grow(NumElts);
    End = Begin+NumElts;
    for (; NumElts; --NumElts)
      new (Begin+NumElts-1) T(Elt);
  }
  
  const SmallVectorImpl &operator=(const SmallVectorImpl &RHS) {
    // Avoid self-assignment.
    if (this == &RHS) return *this;
    
    // If we already have sufficient space, assign the common elements, then
    // destroy any excess.
    unsigned RHSSize = RHS.size();
    unsigned CurSize = size();
    if (CurSize >= RHSSize) {
      // Assign common elements.
      std::copy(RHS.Begin, RHS.Begin+RHSSize, Begin);
      
      // Destroy excess elements.
      for (unsigned i = RHSSize; i != CurSize; ++i)
        Begin[i].~T();
      
      // Trim.
      End = Begin + RHSSize;
      return *this;
    }
    
    // If we have to grow to have enough elements, destroy the current elements.
    // This allows us to avoid copying them during the grow.
    if (Capacity-Begin < RHSSize) {
      // Destroy current elements.
      for (iterator I = Begin, E = End; I != E; ++I)
        I->~T();
      End = Begin;
      CurSize = 0;
      grow(RHSSize);
    } else if (CurSize) {
      // Otherwise, use assignment for the already-constructed elements.
      std::copy(RHS.Begin, RHS.Begin+CurSize, Begin);
    }
    
    // Copy construct the new elements in place.
    std::uninitialized_copy(RHS.Begin+CurSize, RHS.End, Begin+CurSize);
    
    // Set end.
    End = Begin+RHSSize;
  }
  
private:
  /// isSmall - Return true if this is a smallvector which has not had dynamic
  /// memory allocated for it.
  bool isSmall() const {
    return (void*)Begin == (void*)&FirstEl;
  }

  /// grow - double the size of the allocated memory, guaranteeing space for at
  /// least one more element or MinSize if specified.
  void grow(unsigned MinSize = 0) {
    unsigned CurCapacity = Capacity-Begin;
    unsigned CurSize = size();
    unsigned NewCapacity = 2*CurCapacity;
    if (NewCapacity < MinSize)
      NewCapacity = MinSize;
    T *NewElts = reinterpret_cast<T*>(new char[NewCapacity*sizeof(T)]);

    // Copy the elements over.
    std::uninitialized_copy(Begin, End, NewElts);
    
    // Destroy the original elements.
    for (iterator I = Begin, E = End; I != E; ++I)
      I->~T();
    
    // If this wasn't grown from the inline copy, deallocate the old space.
    if (!isSmall())
      delete[] (char*)Begin;
    
    Begin = NewElts;
    End = NewElts+CurSize;
    Capacity = Begin+NewCapacity;
  }
};

  
/// SmallVector - This is a 'vector' (really, a variable-sized array), optimized
/// for the case when the array is small.  It contains some number of elements
/// in-place, which allows it to avoid heap allocation when the actual number of
/// elements is below that threshold.  This allows normal "small" cases to be
/// fast without losing generality for large inputs.
///
/// Note that this does not attempt to be exception safe.
///
template <typename T, unsigned N>
class SmallVector : public SmallVectorImpl<T> {
  /// InlineElts - These are 'N-1' elements that are stored inline in the body
  /// of the vector.  The extra '1' element is stored in SmallVectorImpl.
  typedef typename SmallVectorImpl<T>::U U;
  U InlineElts[(sizeof(T)*N+sizeof(U)-1)/sizeof(U) - 1];
public:  
  SmallVector() : SmallVectorImpl<T>(N) {
  }
  
  template<typename ItTy>
  SmallVector(ItTy S, ItTy E) : SmallVectorImpl<T>(N) {
    append(S, E);
  }
  
  SmallVector(const SmallVector &RHS) : SmallVectorImpl<T>(N) {
    operator=(RHS);
  }
};

} // End llvm namespace

#endif
