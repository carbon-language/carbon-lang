//===- SJLJ-exception.h - SetJmp/LongJmp Exception Handling -----*- C++ -*-===//
//
// This file defines the data structures and API used by the Setjmp/Longjmp
// exception handling runtime library.
//
//===----------------------------------------------------------------------===//

#ifndef SJLJ_EXCEPTION_H
#define SJLJ_EXCEPTION_H

#include "exception.h"
#include <cassert>

struct llvm_sjlj_exception {
  // JmpBuffer - This is the buffer which was longjmp'd with.
  //
  void *JmpBuffer;

  // LongJmpValue - The value passed into longjmp, which the corresponding
  // setjmp should return.  Note that this value will never be equal to 0.
  //
  int LongJmpValue;

  // BaseException - The language independent portion of the exception state.
  // This is at the end of the record so that we can add additional members to
  // this structure without breaking binary compatibility.
  //
  llvm_exception BaseException;
};

extern "C" {
  // __llvm_sjljeh_throw_longjmp - This function creates the longjmp exception
  // and returns.  It takes care of mapping the longjmp value from 0 -> 1 as
  // appropriate.  The caller should immediately call llvm.unwind after this
  // function call.
  void __llvm_sjljeh_throw_longjmp(void *JmpBuffer, int Val) throw();

  // __llvm_sjljeh_init_setjmpmap - This funciton initializes the pointer
  // provided to an empty setjmp map, and should be called on entry to a
  // function which calls setjmp.
  void __llvm_sjljeh_init_setjmpmap(void **SetJmpMap) throw();

  // __llvm_sjljeh_destroy_setjmpmap - This function frees all memory associated
  // with the specified setjmpmap structure.  It should be called on all exits
  // (returns or unwinds) from the function which calls ...init_setjmpmap.
  void __llvm_sjljeh_destroy_setjmpmap(void **SetJmpMap) throw();

  // __llvm_sjljeh_add_setjmp_to_map - This function adds or updates an entry to
  // the map, to indicate which setjmp should be returned to if a longjmp
  // happens.
  void __llvm_sjljeh_add_setjmp_to_map(void **SetJmpMap, void *JmpBuf,
                                       unsigned SetJmpID) throw();

  // __llvm_sjljeh_is_longjmp_exception - This function returns true if the
  // current uncaught exception is a longjmp exception.  This is the first step
  // of catching a sjlj exception.
  bool __llvm_sjljeh_is_longjmp_exception() throw();
  
  // __llvm_sjljeh_get_longjmp_value - This function returns the value that the
  // setjmp call should "return".  This requires that the current uncaught
  // exception be a sjlj exception, though it does not require the exception to
  // be caught by this function.
  int __llvm_sjljeh_get_longjmp_value() throw();

  // __llvm_sjljeh_try_catching_longjmp_exception - This function checks to see
  // if the current uncaught longjmp exception matches any of the setjmps
  // collected in the setjmpmap structure.  If so, it catches and destroys the
  // exception, returning the index of the setjmp which caught the exception.
  // If not, it leaves the exception uncaught and returns a value of ~0.
  unsigned __llvm_sjljeh_try_catching_longjmp_exception(void **SetJmpMap)
    throw();
}

#endif
