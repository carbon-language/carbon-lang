//===-- NonCopyable.h - Disable copy ctor and op= in subclasses --*- C++ -*--=//
//
// This file defines the NonCopyable and NonCopyableV classes.  These mixin
// classes may be used to mark a class not being copyable.  You should derive
// from NonCopyable if you don't want to have a virtual dtor, or NonCopyableV
// if you do want polymorphic behavior in your class.
//
// No library is required when using these functinons.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_NONCOPYABLE_H
#define LLVM_SUPPORT_NONCOPYABLE_H

class NonCopyable {
  // Disable the copy constructor and the assignment operator
  // by making them both private:
  // 
  NonCopyable(const NonCopyable &);            // DO NOT IMPLEMENT
  NonCopyable &operator=(const NonCopyable &); // DO NOT IMPLEMENT
protected:
  inline NonCopyable() {}
  inline ~NonCopyable() {}
};

#endif
