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
    destroy_range(Begin, End);

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
  
  reference front() {
    return begin()[0];
  }
  const_reference front() const {
    return begin()[0];
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
    destroy_range(Begin, End);
    End = Begin;
  }
  
  void resize(unsigned N) {
    if (N < size()) {
      destroy_range(Begin+N, End);
      End = Begin+N;
    } else if (N > size()) {
      if (Begin+N > Capacity)
        grow(N);
      construct_range(End, Begin+N, T());
      End = Begin+N;
    }
  }
  
  void swap(SmallVectorImpl &RHS);
  
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
    construct_range(Begin, End, Elt);
  }
  
  void erase(iterator I) {
    // Shift all elts down one.
    std::copy(I+1, End, I);
    // Drop the last elt.
    pop_back();
  }
  
  void erase(iterator S, iterator E) {
    // Shift all elts down.
    iterator I = std::copy(E, End, S);
    // Drop the last elts.
    destroy_range(I, End);
    End = I;
  }
  
  iterator insert(iterator I, const T &Elt) {
    if (I == End) {  // Important special case for empty vector.
      push_back(Elt);
      return end()-1;
    }
    
    if (End < Capacity) {
  Retry:
      new (End) T(back());
      ++End;
      // Push everything else over.
      std::copy_backward(I, End-1, End);
      *I = Elt;
      return I;
    }
    unsigned EltNo = I-Begin;
    grow();
    I = Begin+EltNo;
    goto Retry;
  }
  
  const SmallVectorImpl &operator=(const SmallVectorImpl &RHS);
  
private:
  /// isSmall - Return true if this is a smallvector which has not had dynamic
  /// memory allocated for it.
  bool isSmall() const {
    return (void*)Begin == (void*)&FirstEl;
  }

  /// grow - double the size of the allocated memory, guaranteeing space for at
  /// least one more element or MinSize if specified.
  void grow(unsigned MinSize = 0);

  void construct_range(T *S, T *E, const T &Elt) {
    for (; S != E; ++S)
      new (S) T(Elt);
  }

  
  void destroy_range(T *S, T *E) {
    while (S != E) {
      E->~T();
      --E;
    }
  }
};

// Define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T>
void SmallVectorImpl<T>::grow(unsigned MinSize) {
  unsigned CurCapacity = Capacity-Begin;
  unsigned CurSize = size();
  unsigned NewCapacity = 2*CurCapacity;
  if (NewCapacity < MinSize)
    NewCapacity = MinSize;
  T *NewElts = reinterpret_cast<T*>(new char[NewCapacity*sizeof(T)]);
  
  // Copy the elements over.
  std::uninitialized_copy(Begin, End, NewElts);
  
  // Destroy the original elements.
  destroy_range(Begin, End);
  
  // If this wasn't grown from the inline copy, deallocate the old space.
  if (!isSmall())
    delete[] (char*)Begin;
  
  Begin = NewElts;
  End = NewElts+CurSize;
  Capacity = Begin+NewCapacity;
}

template <typename T>
void SmallVectorImpl<T>::swap(SmallVectorImpl<T> &RHS) {
  if (this == &RHS) return;
  
  // We can only avoid copying elements if neither vector is small.
  if (!isSmall() && !RHS.isSmall()) {
    std::swap(Begin, RHS.Begin);
    std::swap(End, RHS.End);
    std::swap(Capacity, RHS.Capacity);
    return;
  }
  if (Begin+RHS.size() > Capacity)
    grow(RHS.size());
  if (RHS.begin()+size() > RHS.Capacity)
    RHS.grow(size());
  
  // Swap the shared elements.
  unsigned NumShared = size();
  if (NumShared > RHS.size()) NumShared = RHS.size();
  for (unsigned i = 0; i != NumShared; ++i)
    std::swap(Begin[i], RHS[i]);
  
  // Copy over the extra elts.
  if (size() > RHS.size()) {
    unsigned EltDiff = size() - RHS.size();
    std::uninitialized_copy(Begin+NumShared, End, RHS.End);
    RHS.End += EltDiff;
    destroy_range(Begin+NumShared, End);
    End = Begin+NumShared;
  } else if (RHS.size() > size()) {
    unsigned EltDiff = RHS.size() - size();
    std::uninitialized_copy(RHS.Begin+NumShared, RHS.End, End);
    End += EltDiff;
    destroy_range(RHS.Begin+NumShared, RHS.End);
    RHS.End = RHS.Begin+NumShared;
  }
}
  
template <typename T>
const SmallVectorImpl<T> &
SmallVectorImpl<T>::operator=(const SmallVectorImpl<T> &RHS) {
  // Avoid self-assignment.
  if (this == &RHS) return *this;
  
  // If we already have sufficient space, assign the common elements, then
  // destroy any excess.
  unsigned RHSSize = RHS.size();
  unsigned CurSize = size();
  if (CurSize >= RHSSize) {
    // Assign common elements.
    iterator NewEnd = std::copy(RHS.Begin, RHS.Begin+RHSSize, Begin);
    
    // Destroy excess elements.
    destroy_range(NewEnd, End);
    
    // Trim.
    End = NewEnd;
    return *this;
  }
  
  // If we have to grow to have enough elements, destroy the current elements.
  // This allows us to avoid copying them during the grow.
  if (unsigned(Capacity-Begin) < RHSSize) {
    // Destroy current elements.
    destroy_range(Begin, End);
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
  return *this;
}
  
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
  enum {
    // MinUs - The number of U's require to cover N T's.
    MinUs = (sizeof(T)*N+sizeof(U)-1)/sizeof(U),
    
    // NumInlineEltsElts - The number of elements actually in this array.  There
    // is already one in the parent class, and we have to round up to avoid
    // having a zero-element array.
    NumInlineEltsElts = (MinUs - 1) > 0 ? (MinUs - 1) : 1,
    
    // NumTsAvailable - The number of T's we actually have space for, which may
    // be more than N due to rounding.
    NumTsAvailable = (NumInlineEltsElts+1)*sizeof(U) / sizeof(T)
  };
  U InlineElts[NumInlineEltsElts];
public:  
  SmallVector() : SmallVectorImpl<T>(NumTsAvailable) {
  }
  
  template<typename ItTy>
  SmallVector(ItTy S, ItTy E) : SmallVectorImpl<T>(NumTsAvailable) {
    append(S, E);
  }
  
  SmallVector(const SmallVector &RHS) : SmallVectorImpl<T>(NumTsAvailable) {
    operator=(RHS);
  }
  
  const SmallVector &operator=(const SmallVector &RHS) {
    SmallVectorImpl<T>::operator=(RHS);
    return *this;
  }
};

} // End llvm namespace

namespace std {
  /// Implement std::swap in terms of SmallVector swap.
  template<typename T>
  inline void
  swap(llvm::SmallVectorImpl<T> &LHS, llvm::SmallVectorImpl<T> &RHS) {
    LHS.swap(RHS);
  }
  
  /// Implement std::swap in terms of SmallVector swap.
  template<typename T, unsigned N>
  inline void
  swap(llvm::SmallVector<T, N> &LHS, llvm::SmallVector<T, N> &RHS) {
    LHS.swap(RHS);
  }
}

#endif
