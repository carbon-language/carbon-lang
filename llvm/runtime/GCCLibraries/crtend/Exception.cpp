//===- Exception.cpp - Generic language-independent exceptions ------------===//
//
// This file defines the the shared data structures used by all language
// specific exception handling runtime libraries.
//
//===----------------------------------------------------------------------===//

#include "Exception.h"
#include <cassert>

// Thread local state for exception handling.  FIXME: This should really be made
// thread-local!

// UncaughtExceptionStack - The stack of exceptions currently being thrown.
static llvm_exception *UncaughtExceptionStack = 0;

// __llvm_eh_has_uncaught_exception - This is used to implement
// std::uncaught_exception.
//
bool __llvm_eh_has_uncaught_exception() throw() {
  return UncaughtExceptionStack != 0;
}

// __llvm_eh_current_uncaught_exception - This function checks to see if the
// current uncaught exception is of the specified language type.  If so, it
// returns a pointer to the exception area data.
//
void *__llvm_eh_current_uncaught_exception_type(unsigned HandlerType) throw() {
  assert(UncaughtExceptionStack && "No uncaught exception!");
  if (UncaughtExceptionStack->ExceptionType == HandlerType)
    return UncaughtExceptionStack+1;
  return 0;
}

// __llvm_eh_add_uncaught_exception - This adds the specified exception to the
// top of the uncaught exception stack.  The exception should not already be on
// the stack!
void __llvm_eh_add_uncaught_exception(llvm_exception *E) throw() {
  E->Next = UncaughtExceptionStack;
  UncaughtExceptionStack = E;
}


// __llvm_eh_get_uncaught_exception - Returns the current uncaught exception.
// There must be an uncaught exception for this to work!
llvm_exception *__llvm_eh_get_uncaught_exception() throw() {
  assert(UncaughtExceptionStack && "There are no uncaught exceptions!?!?");
  return UncaughtExceptionStack;
}

// __llvm_eh_pop_from_uncaught_stack - Remove the current uncaught exception
// from the top of the stack.
llvm_exception *__llvm_eh_pop_from_uncaught_stack() throw() {
  llvm_exception *E = __llvm_eh_get_uncaught_exception();
  UncaughtExceptionStack = E->Next;
  return E;
}
