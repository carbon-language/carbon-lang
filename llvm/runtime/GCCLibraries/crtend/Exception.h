//===- exception.h - Generic language-independent exceptions ----*- C++ -*-===//
//
// This file defines the the shared data structures used by all language
// specific exception handling runtime libraries.
//
//===----------------------------------------------------------------------===//

#ifndef EXCEPTION_H
#define EXCEPTION_H

struct llvm_exception {
  // ExceptionDestructor - This call-back function is used to destroy the
  // current exception, without requiring the caller to know what the concrete
  // exception type is.
  //
  void (*ExceptionDestructor)(llvm_exception *);
  
  // ExceptionType - This field identifies what runtime library this exception
  // came from.  Currently defined values are:
  //     0 - Error
  //     1 - longjmp exception (see longjmp-exception.c)
  //     2 - C++ exception (see c++-exception.c)
  //
  unsigned ExceptionType;

  // Next - This points to the next exception in the current stack.
  llvm_exception *Next;

  // HandlerCount - This is a count of the number of handlers which have
  // currently caught this exception.  If the handler is caught and this number
  // falls to zero, the exception is destroyed.
  //
  unsigned HandlerCount;
};

enum {
  ErrorException   = 0,
  LongjmpException = 1,
  CXXException     = 2,
};

// Language independent exception handling API...
//
extern "C" {
  bool __llvm_eh_has_uncaught_exception(void);
  void *__llvm_eh_current_uncaught_exception_type(unsigned HandlerType);
}

#endif
