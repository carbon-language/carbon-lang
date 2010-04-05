//===- llvm/ADT/OwningPtr.h - Smart ptr that owns the pointee ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines and implements the OwningPtr class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_OWNING_PTR_H
#define LLVM_ADT_OWNING_PTR_H

#include <cassert>
#include <cstddef>

namespace llvm {

/// OwningPtr smart pointer - OwningPtr mimics a built-in pointer except that it
/// guarantees deletion of the object pointed to, either on destruction of the
/// OwningPtr or via an explicit reset().  Once created, ownership of the
/// pointee object can be taken away from OwningPtr by using the take method.
template<class T>
class OwningPtr {
  OwningPtr(OwningPtr const &);             // DO NOT IMPLEMENT
  OwningPtr &operator=(OwningPtr const &);  // DO NOT IMPLEMENT
  T *Ptr;
public:
  explicit OwningPtr(T *P = 0) : Ptr(P) {}

  ~OwningPtr() {
    delete Ptr;
  }

  /// reset - Change the current pointee to the specified pointer.  Note that
  /// calling this with any pointer (including a null pointer) deletes the
  /// current pointer.
  void reset(T *P = 0) {
    if (P == Ptr) return;
    T *Tmp = Ptr;
    Ptr = P;
    delete Tmp;
  }

  /// take - Reset the owning pointer to null and return its pointer.  This does
  /// not delete the pointer before returning it.
  T *take() {
    T *Tmp = Ptr;
    Ptr = 0;
    return Tmp;
  }

  T &operator*() const {
    assert(Ptr && "Cannot dereference null pointer");
    return *Ptr;
  }

  T *operator->() const { return Ptr; }
  T *get() const { return Ptr; }
  operator bool() const { return Ptr != 0; }
  bool operator!() const { return Ptr == 0; }

  void swap(OwningPtr &RHS) {
    T *Tmp = RHS.Ptr;
    RHS.Ptr = Ptr;
    Ptr = Tmp;
  }
};

template<class T>
inline void swap(OwningPtr<T> &a, OwningPtr<T> &b) {
  a.swap(b);
}

/// OwningArrayPtr smart pointer - OwningArrayPtr provides the same
///  functionality as OwningPtr, except that it works for array types.
template<class T>
class OwningArrayPtr {
  OwningArrayPtr(OwningArrayPtr const &);            // DO NOT IMPLEMENT
  OwningArrayPtr &operator=(OwningArrayPtr const &); // DO NOT IMPLEMENT
  T *Ptr;
public:
  explicit OwningArrayPtr(T *P = 0) : Ptr(P) {}

  ~OwningArrayPtr() {
    delete [] Ptr;
  }

  /// reset - Change the current pointee to the specified pointer.  Note that
  /// calling this with any pointer (including a null pointer) deletes the
  /// current pointer.
  void reset(T *P = 0) {
    if (P == Ptr) return;
    T *Tmp = Ptr;
    Ptr = P;
    delete [] Tmp;
  }

  /// take - Reset the owning pointer to null and return its pointer.  This does
  /// not delete the pointer before returning it.
  T *take() {
    T *Tmp = Ptr;
    Ptr = 0;
    return Tmp;
  }

  T &operator[](std::ptrdiff_t i) const {
    assert(Ptr && "Cannot dereference null pointer");
    return Ptr[i];
  }

  T *get() const { return Ptr; }
  operator bool() const { return Ptr != 0; }
  bool operator!() const { return Ptr == 0; }

  void swap(OwningArrayPtr &RHS) {
    T *Tmp = RHS.Ptr;
    RHS.Ptr = Ptr;
    Ptr = Tmp;
  }
};

template<class T>
inline void swap(OwningArrayPtr<T> &a, OwningArrayPtr<T> &b) {
  a.swap(b);
}

/// \brief A smart pointer that may own the object it points to.
///
/// An instance of \c MaybeOwningPtr may own the object it points to. If so,
/// it will guarantee that the object will be deleted either on destructin of
/// the OwningPtr or via an explicit reset(). Once created, ownership of the
/// pointee object can be taken away from OwningPtr by using the \c take()
/// method.
template<class T>
class MaybeOwningPtr {
  T *Ptr;
  bool Owned;
  
  struct MaybeOwningPtrRef {
    MaybeOwningPtrRef(T *Ptr, bool &Owned) : Ptr(Ptr), Owned(Owned) { }
    
    T *Ptr;
    bool &Owned;
  };
  
public:
  MaybeOwningPtr() : Ptr(0), Owned(false) { }
  
  explicit MaybeOwningPtr(T *P, bool OwnP) : Ptr(P), Owned(OwnP) {}
  
  /// \brief Take ownership of the pointer stored in \c Other.
  MaybeOwningPtr(MaybeOwningPtr& Other) : Ptr(Other.Ptr), Owned(Other.Owned) {
    Other.Owned = false;
  }

  MaybeOwningPtr(MaybeOwningPtrRef Other) : Ptr(Other.Ptr), Owned(Other.Owned) {
    Other.Owned = false;
  }
    
  /// \brief Take ownership of the ppinter stored in \c Other.
  MaybeOwningPtr &operator=(MaybeOwningPtr &Other) {
    reset(Other.Ptr, Other.Owned);
    Other.Owned = false;
    return *this;
  }

  ~MaybeOwningPtr() {
    if (Owned)
      delete Ptr;
  }
  
  operator MaybeOwningPtrRef() { return MaybeOwningPtrRef(Ptr, Owned); }
  
  /// reset - Change the current pointee to the specified pointer.  Note that
  /// calling this with any pointer (including a null pointer) deletes the
  /// current pointer.
  void reset(T *P, bool OwnP) {
    assert(P != Ptr);
    if (Owned)
      delete Ptr;
    
    Ptr = P;
    Owned = OwnP;
  }
  
  /// take - Return the underlying pointer and take ownership of it. This
  /// \c MaybeOwningPtr must have ownership before the call, and will 
  /// relinquish ownership as part of the call.
  T *take() {
    assert(Owned && "Cannot take ownership from a non-owning pointer");
    Owned = false;
    return Ptr;
  }
  
  T &operator*() const {
    assert(Ptr && "Cannot dereference null pointer");
    return *Ptr;
  }
  
  T *operator->() const { return Ptr; }
  T *get() const { return Ptr; }
  operator bool() const { return Ptr != 0; }
  bool operator!() const { return Ptr == 0; }
  
  void swap(MaybeOwningPtr &RHS) {
    T *Tmp = RHS.Ptr;
    RHS.Ptr = Ptr;
    Ptr = Tmp;
    bool TmpOwned = RHS.Owned;
    RHS.Owned = Owned;
    Owned = TmpOwned;
  }
};

template<class T>
inline void swap(MaybeOwningPtr<T> &a, MaybeOwningPtr<T> &b) {
  a.swap(b);
}
  
} // end namespace llvm

#endif
