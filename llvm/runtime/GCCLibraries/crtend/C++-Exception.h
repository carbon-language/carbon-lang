//===- c++-exception.h - C++ Specific exception Handling --------*- C++ -*-===//
//
// This file defines the data structures and API used by the C++ exception
// handling runtime library.
//
//===----------------------------------------------------------------------===//

#ifndef CXX_EXCEPTION_H
#define CXX_EXCEPTION_H

#include "exception.h"
#include <typeinfo>
#include <cassert>

struct llvm_cxx_exception {
  /* TypeInfo - A pointer to the C++ std::type_info object for this exception
   * class.  This is required because the class may not be polymorphic.
   */
  const std::type_info *TypeInfo;

  /* ExceptionObjectDestructor - A pointer to the function which destroys the
   * object represented by this exception.  This is required because the class
   * may not be polymorphic.  This may be null if there is no cleanup required.
   */
  void (*ExceptionObjectDestructor)(void *);

  /* UnexpectedHandler - This contains a pointer to the "unexpected" handler
   * which may be registered by the user program with set_unexpected.  Calls to
   * unexpected which are a result of an exception throw are supposed to use the
   * value of the handler at the time of the throw, not the currently set value.
   */
  void (*UnexpectedHandler)();

  /* TerminateHandler - This contains a pointer to the "terminate" handler which
   * may be registered by the user program with set_terminate.  Calls to
   * unexpected which are a result of an exception throw are supposed to use the
   * value of the handler at the time of the throw, not the currently set value.
   */
  void (*TerminateHandler)();

  /* BaseException - The language independent portion of the exception state.
   * This is at the end of the record so that we can add additional members to
   * this structure without breaking binary compatibility.
   */
  llvm_exception BaseException;
};

inline llvm_cxx_exception *get_cxx_exception(llvm_exception *E) throw() {
  assert(E->ExceptionType == CXXException && "Not a C++ exception?");
  return (llvm_cxx_exception*)(E+1)-1;
}

// Interface to the C++ standard library to get to the terminate and unexpected
// handler stuff.
namespace __cxxabiv1 {
  // Invokes given handler, dying appropriately if the user handler was
  // so inconsiderate as to return.
  extern void __terminate(std::terminate_handler) __attribute__((noreturn));
  extern void __unexpected(std::unexpected_handler) __attribute__((noreturn));
  
  // The current installed user handlers.
  extern std::terminate_handler __terminate_handler;
  extern std::unexpected_handler __unexpected_handler;
}

extern "C" {
  void *__llvm_cxxeh_allocate_exception(unsigned NumBytes) throw();
  void __llvm_cxxeh_free_exception(void *ObjectPtr) throw();
  void __llvm_cxxeh_throw(void *ObjectPtr, void *TypeInfoPtr,
                          void (*DtorPtr)(void*)) throw();

  void __llvm_cxxeh_call_terminate() throw() __attribute__((noreturn));
  void * __llvm_cxxeh_current_uncaught_exception_isa(void *Ty)
    throw();
  void *__llvm_cxxeh_begin_catch() throw();
  void *__llvm_cxxeh_begin_catch_if_isa(void *CatchType) throw();
  void __llvm_cxxeh_end_catch() /* might throw */;
  void __llvm_cxxeh_rethrow() throw();
  void __llvm_cxxeh_check_eh_spec(void *Info, ...);
}

#endif
