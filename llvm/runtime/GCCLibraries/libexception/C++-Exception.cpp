//===- C++-Exception.cpp - Exception handling support for C++ exceptions --===//
//
// This file defines the methods used to implement C++ exception handling in
// terms of the invoke and %llvm.unwind intrinsic.  These primitives implement
// an exception handling ABI similar (but simpler and more efficient than) the
// Itanium C++ ABI exception handling standard.
//
//===----------------------------------------------------------------------===//

#include "C++-Exception.h"
#include <cstdlib>
#include <cstdarg>

//#define DEBUG

#ifdef DEBUG
#include <stdio.h>
#endif

//===----------------------------------------------------------------------===//
// Generic exception support
//

// Thread local state for exception handling.
// FIXME: This should really be made thread-local!
//

// LastCaughtException - The last exception caught by this handler.  This is for
// implementation of _rethrow and _get_last_caught.
//
static llvm_exception *LastCaughtException = 0;

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


//===----------------------------------------------------------------------===//
// C++ Specific exception handling support...
//
using namespace __cxxabiv1;

// __llvm_cxxeh_allocate_exception - This function allocates space for the
// specified number of bytes, plus a C++ exception object header.
//
void *__llvm_cxxeh_allocate_exception(unsigned NumBytes) throw() {
  // FIXME: This should eventually have back-up buffers for out-of-memory
  // situations.
  //
  llvm_cxx_exception *E =
    (llvm_cxx_exception *)malloc(NumBytes+sizeof(llvm_cxx_exception));
  E->BaseException.ExceptionType = 0; // intialize to invalid

  return E+1;   // return the pointer after the header
}

// __llvm_cxxeh_free_exception - Low-level function to free an exception.  This
// is called directly from generated C++ code if evaluating the exception value
// into the exception location throws.  Otherwise it is called from the C++
// exception object destructor.
//
void __llvm_cxxeh_free_exception(void *ObjectPtr) throw() {
  llvm_cxx_exception *E = (llvm_cxx_exception *)ObjectPtr - 1;
  free(E);
}

// cxx_destructor - This function is called through the generic
// exception->ExceptionDestructor function pointer to destroy a caught
// exception.
//
static void cxx_destructor(llvm_exception *LE) /* might throw */{
  assert(LE->Next == 0 && "On the uncaught stack??");
  llvm_cxx_exception *E = get_cxx_exception(LE);

  struct ExceptionFreer {
    void *Ptr;
    ExceptionFreer(void *P) : Ptr(P) {}
    ~ExceptionFreer() {
      // Free the memory for the exception, when the function is left, even if
      // the exception object dtor throws its own exception!
      __llvm_cxxeh_free_exception(Ptr);
    }
  } EF(E+1);
  
  // Run the exception object dtor if it exists. */
  if (E->ExceptionObjectDestructor)
    E->ExceptionObjectDestructor(E);
}

// __llvm_cxxeh_throw - Given a pointer to memory which has an exception object
// evaluated into it, this sets up all of the fields of the exception allowing
// it to be thrown.  After calling this, the code should call %llvm.unwind
//
void __llvm_cxxeh_throw(void *ObjectPtr, void *TypeInfoPtr,
                        void (*DtorPtr)(void*)) throw() {
  llvm_cxx_exception *E = (llvm_cxx_exception *)ObjectPtr - 1;
  E->BaseException.ExceptionDestructor = cxx_destructor;
  E->BaseException.ExceptionType = CXXException;
  E->BaseException.Next = UncaughtExceptionStack;
  UncaughtExceptionStack = &E->BaseException;
  E->BaseException.HandlerCount = 0;
  E->BaseException.isRethrown = 0;

  E->TypeInfo = (const std::type_info*)TypeInfoPtr;
  E->ExceptionObjectDestructor = DtorPtr;
  E->UnexpectedHandler = __unexpected_handler;
  E->TerminateHandler = __terminate_handler;
}


// CXXExceptionISA - use the type info object stored in the exception to see if
// TypeID matches and, if so, to adjust the exception object pointer.
//
static void *CXXExceptionISA(llvm_cxx_exception *E,
                             const std::type_info *Type) throw() {
  // ThrownPtr is a pointer to the object being thrown...
  void *ThrownPtr = E+1;
  const std::type_info *ThrownType = E->TypeInfo;

#if 0
  // FIXME: this code exists in the GCC exception handling library: I haven't
  // thought about this yet, so it should be verified at some point!

  // Pointer types need to adjust the actual pointer, not
  // the pointer to pointer that is the exception object.
  // This also has the effect of passing pointer types
  // "by value" through the __cxa_begin_catch return value.
  if (ThrownType->__is_pointer_p())
    ThrownPtr = *(void **)ThrownPtr;
#endif

  if (Type->__do_catch(ThrownType, &ThrownPtr, 1)) {
#ifdef DEBUG
    printf("isa<%s>(%s):  0x%p -> 0x%p\n", Type->name(), ThrownType->name(),
           E+1, ThrownPtr);
#endif
    return ThrownPtr;
  }

  return 0;
}

// __llvm_cxxeh_current_uncaught_exception_isa - This function checks to see if
// the current uncaught exception is a C++ exception, and if it is of the
// specified type id.  If so, it returns a pointer to the object adjusted as
// appropriate, otherwise it returns null.
//
void *__llvm_cxxeh_current_uncaught_exception_isa(void *CatchType) throw() {
  assert(UncaughtExceptionStack && "No uncaught exception!");
  if (UncaughtExceptionStack->ExceptionType != CXXException)
    return 0;     // If it's not a c++ exception, it doesn't match!

  // If it is a C++ exception, use the type info object stored in the exception
  // to see if TypeID matches and, if so, to adjust the exception object
  // pointer.
  //
  const std::type_info *Info = (const std::type_info *)CatchType;
  return CXXExceptionISA(get_cxx_exception(UncaughtExceptionStack), Info);
}


// __llvm_cxxeh_begin_catch - This function is called by "exception handlers",
// which transition an exception from being uncaught to being caught.  It
// returns a pointer to the exception object portion of the exception.  This
// function must work with foreign exceptions.
//
void *__llvm_cxxeh_begin_catch() throw() {
  llvm_exception *E = UncaughtExceptionStack;
  assert(UncaughtExceptionStack && "There are no uncaught exceptions!?!?");

  // The exception is now no longer uncaught.
  UncaughtExceptionStack = E->Next;
  
  // The exception is now caught.
  LastCaughtException = E;
  E->Next = 0;
  E->isRethrown = 0;

  // Increment the handler count for this exception.
  E->HandlerCount++;

#ifdef DEBUG
  printf("Exiting begin_catch Ex=0x%p HandlerCount=%d!\n", E+1,
         E->HandlerCount);
#endif
  
  // Return a pointer to the raw exception object.
  return E+1;
}

// __llvm_cxxeh_begin_catch_if_isa - This function checks to see if the current
// uncaught exception is of the specified type.  If not, it returns a null
// pointer, otherwise it 'catches' the exception and returns a pointer to the
// object of the specified type.  This function does never succeeds with foreign
// exceptions (because they can never be of type CatchType).
//
void *__llvm_cxxeh_begin_catch_if_isa(void *CatchType) throw() {
  void *ObjPtr = __llvm_cxxeh_current_uncaught_exception_isa(CatchType);
  if (!ObjPtr) return 0;
  
  // begin_catch, meaning that the object is now "caught", not "uncaught"
  __llvm_cxxeh_begin_catch();
  return ObjPtr;
}

// __llvm_cxxeh_get_last_caught - Return the last exception that was caught by
// ...begin_catch.
//
void *__llvm_cxxeh_get_last_caught() throw() {
  assert(LastCaughtException && "No exception caught!!");
  return LastCaughtException+1;
}

// __llvm_cxxeh_end_catch - This function decrements the HandlerCount of the
// top-level caught exception, destroying it if this is the last handler for the
// exception.
//
void __llvm_cxxeh_end_catch(void *Ex) /* might throw */ {
  llvm_exception *E = (llvm_exception*)Ex - 1;
  assert(E && "There are no caught exceptions!");
  
  // If this is the last handler using the exception, destroy it now!
  if (--E->HandlerCount == 0 && !E->isRethrown) {
#ifdef DEBUG
    printf("Destroying exception!\n");
#endif
    E->ExceptionDestructor(E);        // Release memory for the exception
  }
#ifdef DEBUG
  printf("Exiting end_catch Ex=0x%p HandlerCount=%d!\n", Ex, E->HandlerCount);
#endif
}

// __llvm_cxxeh_call_terminate - This function is called when the dtor for an
// object being destroyed due to an exception throw throws an exception.  This
// is illegal because it would cause multiple exceptions to be active at one
// time.
void __llvm_cxxeh_call_terminate() throw() {
  void (*Handler)(void) = __terminate_handler;
  if (UncaughtExceptionStack)
    if (UncaughtExceptionStack->ExceptionType == CXXException)
      Handler = get_cxx_exception(UncaughtExceptionStack)->TerminateHandler;
  __terminate(Handler);
}


// __llvm_cxxeh_rethrow - This function turns the top-level caught exception
// into an uncaught exception, in preparation for an llvm.unwind, which should
// follow immediately after the call to this function.  This function must be
// prepared to deal with foreign exceptions.
//
void __llvm_cxxeh_rethrow() throw() {
  llvm_exception *E = LastCaughtException;
  if (E == 0)
    // 15.1.8 - If there are no exceptions being thrown, 'throw;' should call
    // terminate.
    //
    __terminate(__terminate_handler);

  // Otherwise we have an exception to rethrow.  Mark the exception as such.
  E->isRethrown = 1;

  // Add the exception to the top of the uncaught stack, to preserve the
  // invariant that the top of the uncaught stack is the current exception.
  E->Next = UncaughtExceptionStack;
  UncaughtExceptionStack = E;

  // Return to the caller, which should perform the unwind now.
}

static bool ExceptionSpecificationPermitsException(llvm_exception *E,
                                                   const std::type_info *Info,
                                                   va_list Args) {
  // The only way it could match one of the types is if it is a C++ exception.
  if (E->ExceptionType != CXXException) return false;

  llvm_cxx_exception *Ex = get_cxx_exception(E);
  
  // Scan the list of accepted types, checking to see if the uncaught
  // exception is any of them.
  do {
    // Check to see if the exception matches one of the types allowed by the
    // exception specification.  If so, return to the caller to have the
    // exception rethrown.
    if (CXXExceptionISA(Ex, Info))
      return true;
    
    Info = va_arg(Args, std::type_info *);
  } while (Info);
  return false;
}


// __llvm_cxxeh_check_eh_spec - If a function with an exception specification is
// throwing an exception, this function gets called with the list of type_info
// objects that it is allowing to propagate.  Check to see if the current
// uncaught exception is one of these types, and if so, allow it to be thrown by
// returning to the caller, which should immediately follow this call with
// llvm.unwind.
//
// Note that this function does not throw any exceptions, but we can't put an
// exception specification on it or else we'll get infinite loops!
//
void __llvm_cxxeh_check_eh_spec(void *Info, ...) {
  const std::type_info *TypeInfo = (const std::type_info *)Info;
  llvm_exception *E = UncaughtExceptionStack;
  assert(E && "No uncaught exceptions!");

  if (TypeInfo == 0) {   // Empty exception specification
    // Whatever exception this is, it is not allowed by the (empty) spec, call
    // unexpected, according to 15.4.8.
    try {
      void *Ex = __llvm_cxxeh_begin_catch();   // Start the catch
      __llvm_cxxeh_end_catch(Ex);              // Free the exception
      __unexpected(__unexpected_handler);
    } catch (...) {
      // Any exception thrown by unexpected cannot match the ehspec.  Call
      // terminate, according to 15.4.9.
      __terminate(__terminate_handler);
    }
  }

  // Check to see if the exception matches one of the types allowed by the
  // exception specification.  If so, return to the caller to have the
  // exception rethrown.

  va_list Args;
  va_start(Args, Info);
  bool Ok = ExceptionSpecificationPermitsException(E, TypeInfo, Args);
  va_end(Args);
  if (Ok) return;

  // Ok, now we know that the exception is either not a C++ exception (thus not
  // permitted to pass through) or not a C++ exception that is allowed.  Kill
  // the exception and call the unexpected handler.
  try {
    void *Ex = __llvm_cxxeh_begin_catch();   // Start the catch
    __llvm_cxxeh_end_catch(Ex);              // Free the exception
  } catch (...) {
    __terminate(__terminate_handler);        // Exception dtor threw
  }

  try {
    __unexpected(__unexpected_handler);
  } catch (...) {
    // If the unexpected handler threw an exception, we will get here.  Since
    // entering the try block calls ..._begin_catch, we need to "rethrow" the
    // exception to make it uncaught again.  Exiting the catch will then leave
    // it in the uncaught state.
    __llvm_cxxeh_rethrow();
  }
  
  // Grab the newly caught exception.  If this exception is permitted by the
  // specification, allow it to be thrown.
  E = UncaughtExceptionStack;
  assert(E && "No uncaught exceptions!");

  va_start(Args, Info);
  Ok = ExceptionSpecificationPermitsException(E, TypeInfo, Args);
  va_end(Args);
  if (Ok) return;

  // Final case, check to see if we can throw an std::bad_exception.
  try {
    throw std::bad_exception();
  } catch (...) {
    __llvm_cxxeh_rethrow();
  }

  // Grab the new bad_exception...
  E = UncaughtExceptionStack;
  assert(E && "No uncaught exceptions!");

  // If it's permitted, allow it to be thrown instead.
  va_start(Args, Info);
  Ok = ExceptionSpecificationPermitsException(E, TypeInfo, Args);
  va_end(Args);
  if (Ok) return;

  // Otherwise, we are out of options, terminate, according to 15.5.2.2.
  __terminate(__terminate_handler);
}
