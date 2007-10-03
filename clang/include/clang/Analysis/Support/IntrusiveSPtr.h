//===--- IntrusiveSPtr.h - Smart Reference Counting Pointers ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines two classes: RefCounted, a generic base class for objects
// that wish to have their lifetimes managed using reference counting, and
// IntrusiveSPtr, a template class that implements a "smart" pointer for
// objects that maintain their own internal reference count (e.g. RefCounted).
//
// IntrusiveSPtr is similar to Boost's intrusive_ptr with two main distinctions:
//
//  (1) We implement operator void*() instead of operator bool() so that
//      different pointer values may be accurately compared within an
//      expression.  This includes the comparison of smart pointers with their
//      "unsmart" cousins.
//
//  (2) We provide LLVM-style casting, via cast<> and dyn_cast<>, for
//      IntrusiveSPtrs.
//  
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTRUSIVESPTR
#define LLVM_CLANG_INTRUSIVESPTR

#include "llvm/Support/Casting.h"

namespace clang {

/// RefCounted - A generic base class for objects that wish to have
///  their lifetimes managed using reference counts.  Classes subclass
///  RefCounted to obtain such functionality, and are typically
///  handled with IntrusiveSPtr "smart pointers" (see below) which
///  automatically handle the management of reference counts.  Objects
///  that subclass RefCounted should not be allocated on the stack, as
///  invoking "delete" (which is called when the reference count hits
///  0) on such objects is an error.
class RefCounted {
  unsigned ref_cnt;
public:
  RefCounted() : ref_cnt(0) {}
  
  void Retain() { ++ref_cnt; }
  void Release() {
    assert (ref_cnt > 0 && "Reference count is already zero.");
    if (--ref_cnt == 0) delete this;
  }
  
protected:
  // Making the dstor protected (or private) prevents RefCounted
  // objects from being stack-allocated.  Subclasses should similarly
  // follow suit with this practice.
  virtual ~RefCounted() {}
};
  
} // end namespace clang

/// intrusive_ptr_add_ref - A utility function used by IntrusiveSPtr
///  to increment the reference count of a RefCounted object.  This
///  particular naming was chosen to be compatible with
///  boost::intrusive_ptr, which provides similar functionality to
///  IntrusiveSPtr.
static inline void intrusive_ptr_add_ref(clang::RefCounted* p) { p->Retain(); }

/// intrusive_ptr_release - The complement of intrusive_ptr_add_ref;
///  decrements the reference count of a RefCounted object.
static inline void intrusive_ptr_release(clang::RefCounted* p) { p->Release(); }

namespace clang {

/// IntrusiveSPtr - A template class that implements a "smart pointer"
///  that assumes the wrapped object has a reference count associated
///  with it that can be managed via calls to
///  intrusive_ptr_add_ref/intrusive_ptr_release.  The smart pointers
///  manage reference counts via the RAII idiom: upon creation of
///  smart pointer the reference count of the wrapped object is
///  incremented and upon destruction of the smart pointer the
///  reference count is decremented.  This class also safely handles
///  wrapping NULL pointers.
template <typename T>
class IntrusiveSPtr {
  T* Obj;
public:
  typedef T ObjType;
  
  explicit IntrusiveSPtr() : Obj(NULL) {}
  
  explicit IntrusiveSPtr(const T* obj) : Obj(const_cast<T*>(obj)) { 
    retain();
  }
  
  IntrusiveSPtr(const IntrusiveSPtr& S) : Obj(const_cast<T*>(S.Obj)) { 
    retain(); 
  }

  IntrusiveSPtr& operator=(const IntrusiveSPtr& S) {
    replace(static_cast<const T*>(S.getPtr()));
    return *this;
  }
  
  ~IntrusiveSPtr() { release(); }
  
  T& operator*() { return *Obj; }
  const T& operator*() const { return *Obj; }
  
  T* operator->() { return Obj; }
  const T* operator->() const { return Obj; }
  
  T* getPtr() { return Obj; }
  const T* getPtr() const { return Obj; }
  
  operator void*() { return Obj; }
  operator const void*() const { return Obj; }  

private:
  void retain() { if (Obj) intrusive_ptr_add_ref(Obj); }
  void release() { if (Obj) intrusive_ptr_release(Obj); }

  void replace(const T* o) {
    if (o == Obj)
      return;
    
    release();
    Obj = const_cast<T*>(o);
    retain();
  }
};
  
} // end namespace clang

//===----------------------------------------------------------------------===//
// LLVM-style downcasting support for IntrusiveSPtr objects
//===----------------------------------------------------------------------===//

namespace llvm {

/// cast<X> - Return the argument parameter (wrapped in an
/// IntrusiveSPtr smart pointer) to the specified type.  This casting
/// operator asserts that the type is correct, so it does not return a
/// NULL smart pointer on failure.  Note that the cast returns a
/// reference to an IntrusiveSPtr; thus no reference counts are
/// modified by the cast itself.  Assigning the result of the cast,
/// however, to a non-reference will obviously result in reference
/// counts being adjusted when the copy constructor or operator=
/// method for IntrusiveSPtr is invoked.
template <typename X, typename Y>
inline clang::IntrusiveSPtr<X>&
cast(const clang::IntrusiveSPtr<Y>& V) {
  assert (isa<X>(V.getPtr()) && 
    "cast<Ty>() (IntrusiveSPtr) argument of incompatible type!");
  clang::IntrusiveSPtr<Y>& W = const_cast<clang::IntrusiveSPtr<Y>&>(V);
  return reinterpret_cast<clang::IntrusiveSPtr<X>&>(W);
}

/// cast_or_null<X> - Functionally idential to cast, except that an
/// IntrusiveSPtr wrapping a NULL pointer is accepted.
template <typename X, typename Y>
inline clang::IntrusiveSPtr<X>&
cast_or_null(const clang::IntrusiveSPtr<Y>& V) {
  if (V == 0) return 0;
  assert (isa<X>(V.getPtr()) && 
    "cast_or_null<Ty>() (const IntrusiveSPtr) argument of incompatible type!");  
  clang::IntrusiveSPtr<Y>& W = const_cast<clang::IntrusiveSPtr<Y>&>(V);
  return reinterpret_cast<clang::IntrusiveSPtr<X>&>(W);
}  

/// dyn_cast<X> - Return the argument parameter (wrapped in an
/// IntrusiveSPtr smart pointer) to the specified type.  This casting
/// operator returns an IntrusiveSPtr wrapping a NULL pointer if the
/// argument is of the wrong type, so it can be used to test for a
/// type as well as cast if successful.  Unlike cast<>, a copy of the
/// IntrusiveSPtr is always made, resulting in the reference counts
/// being adjusted.
template <typename X, typename Y>
inline clang::IntrusiveSPtr<X>
dyn_cast(const clang::IntrusiveSPtr<Y>& V) {
  if (isa<X>(V.getPtr()))
    return clang::IntrusiveSPtr<X>(static_cast<const X*>(V.getPtr()));
  else
    return clang::IntrusiveSPtr<X>(NULL);
}

/// dyn_cast_or_null<X> - Functionally identical to dyn_cast, except
/// that an IntrusiveSPtr wrapping a NULL pointer is accepted.
template <typename X, typename Y>
inline clang::IntrusiveSPtr<X>
dyn_cast_or_null(const clang::IntrusiveSPtr<Y>& V) {
  if (V == 0) return clang::IntrusiveSPtr<X>();
  if (isa<X>(V.getPtr()))
    return clang::IntrusiveSPtr<X>(static_cast<const X*>(V.getPtr()));
  else
    return clang::IntrusiveSPtr<X>(NULL);
}
  
} // end namespace llvm

#endif
