//===- c++-exception.cpp - Exception handling support for C++ exceptions --===//
//
// This file defines the methods used to implement C++ exception handling in
// terms of the invoke and %llvm.unwind intrinsic.  These primitives implement
// an exception handling ABI similar (but simpler and more efficient than) the
// Itanium C++ ABI exception handling standard.
//
//===----------------------------------------------------------------------===//

#include "c++-exception.h"
#include <cstdlib>
#include <cassert>

//===----------------------------------------------------------------------===//
// Generic exception support
//

// Thread local state for exception handling.
// FIXME: This should really be made thread-local!
//
static llvm_exception *CaughtExceptionStack = 0;

// UncaughtExceptionStack - The stack of exceptions currently being thrown.
static llvm_exception *UncaughtExceptionStack = 0;

// __llvm_eh_has_uncaught_exception - This is used to implement
// std::uncaught_exception.
//
bool __llvm_eh_has_uncaught_exception(void) {
  return UncaughtExceptionStack != 0;
}

// __llvm_eh_current_uncaught_exception - This function checks to see if the
// current uncaught exception is of the specified language type.  If so, it
// returns a pointer to the exception area data.
//
void *__llvm_eh_current_uncaught_exception_type(unsigned HandlerType) {
  assert(UncaughtExceptionStack && "No uncaught exception!");
  if (UncaughtExceptionStack->ExceptionType == HandlerType)
    return UncaughtExceptionStack+1;
  return 0;
}


//===----------------------------------------------------------------------===//
// C++ Specific exception handling support...
//

// __llvm_cxxeh_allocate_exception - This function allocates space for the
// specified number of bytes, plus a C++ exception object header.
//
void *__llvm_cxxeh_allocate_exception(unsigned NumBytes) {
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
void __llvm_cxxeh_free_exception(void *ObjectPtr) {
  llvm_cxx_exception *E = (llvm_cxx_exception *)ObjectPtr - 1;
  free(E);
}

// cxx_destructor - This function is called through the generic
// exception->ExceptionDestructor function pointer to destroy a caught
// exception.
//
static void cxx_destructor(llvm_exception *LE) {
  void *ObjectPtr = LE+1;
  llvm_cxx_exception *E = (llvm_cxx_exception *)ObjectPtr - 1;

  // The exception is no longer caught.
  assert(CaughtExceptionStack == LE &&
         "Destroying an exception which is not the current caught exception?");
  CaughtExceptionStack = LE->Next;

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
void __llvm_cxxeh_throw(void *ObjectPtr, const std::type_info *TypeInfoPtr,
                        void (*DtorPtr)(void*)) {
  llvm_cxx_exception *E = (llvm_cxx_exception *)ObjectPtr - 1;
  E->BaseException.ExceptionDestructor = cxx_destructor;
  E->BaseException.ExceptionType = CXXException;
  E->BaseException.Next = UncaughtExceptionStack;
  UncaughtExceptionStack = &E->BaseException;
  E->BaseException.HandlerCount = 0;

  E->TypeInfo = TypeInfoPtr;
  E->ExceptionObjectDestructor = DtorPtr;
  E->UnexpectedHandler = 0;  // FIXME
  E->TerminateHandler = 0;   // FIXME
}

// __llvm_cxxeh_current_uncaught_exception_isa - This function checks to see if
// the current uncaught exception is a C++ exception, and if it is of the
// specified type id.  If so, it returns a pointer to the object adjusted as
// appropriate, otherwise it returns null.
//
void *__llvm_cxxeh_current_uncaught_exception_isa(
                                         const std::type_info *CatchType) {
  assert(UncaughtExceptionStack && "No uncaught exception!");
  if (UncaughtExceptionStack->ExceptionType != CXXException)
    return 0;     // If it's not a c++ exception, it doesn't match!

  // If it is a C++ exception, use the type info object stored in the exception
  // to see if TypeID matches and, if so, to adjust the exception object
  // pointer.
  //
  llvm_cxx_exception *E = (llvm_cxx_exception*)UncaughtExceptionStack;

  // ThrownPtr is a pointer to the object being thrown...
  void *ThrownPtr = E+1;
  const std::type_info *ThrownType = E->TypeInfo;

  // FIXME: this code exists in the GCC exception handling library: I haven't
  // thought about this yet, so it should be verified at some point!
#if 1
  // Pointer types need to adjust the actual pointer, not
  // the pointer to pointer that is the exception object.
  // This also has the effect of passing pointer types
  // "by value" through the __cxa_begin_catch return value.
  if (ThrownType->__is_pointer_p())
    ThrownPtr = *(void **)ThrownPtr;
#endif

  if (CatchType->__do_catch(ThrownType, &ThrownPtr, 1))
    return ThrownPtr;

  return 0;
}


// __llvm_cxxeh_begin_catch - This function is called by "exception handlers",
// which transition an exception from being uncaught to being caught.  It
// returns a pointer to the exception object portion of the exception.  This
// function must work with foreign exceptions.
//
void *__llvm_cxxeh_begin_catch(void) {
  llvm_exception *E = UncaughtExceptionStack;
  assert(UncaughtExceptionStack && "There are no uncaught exceptions!?!?");

  // The exception is now no longer uncaught.
  UncaughtExceptionStack = E->Next;
  
  // The exception is now caught.
  E->Next = CaughtExceptionStack;
  CaughtExceptionStack = E->Next;

  // Increment the handler count for this exception.
  E->HandlerCount++;
  
  // Return a pointer to the raw exception object.
  return E+1;
}

// __llvm_cxxeh_begin_catch_if_isa - This function checks to see if the current
// uncaught exception is of the specified type.  If not, it returns a null
// pointer, otherwise it 'catches' the exception and returns a pointer to the
// object of the specified type.  This function does never succeeds with foreign
// exceptions (because they can never be of type CatchType).
//
void *__llvm_cxxeh_begin_catch_if_isa(const std::type_info *CatchType) {
  void *ObjPtr = __llvm_cxxeh_current_uncaught_exception_isa(CatchType);
  if (!ObjPtr) return 0;
  
  // begin_catch, meaning that the object is now "caught", not "uncaught"
  __llvm_cxxeh_begin_catch();
  return ObjPtr;
}


// __llvm_cxxeh_end_catch - This function decrements the HandlerCount of the
// top-level caught exception, destroying it if this is the last handler for the
// exception.
//
void __llvm_cxxeh_end_catch(void) {
  llvm_exception *E = CaughtExceptionStack;
  assert(E && "There are no caught exceptions!");
  
  // If this is the last handler using the exception, destroy it now!
  if (--E->HandlerCount == 0) {
    CaughtExceptionStack = E->Next;   // Unlink from the stack
    E->ExceptionDestructor(E);        // Release memory for the exception
  }
}

// __llvm_cxxeh_rethrow - This function turns the top-level caught exception
// into an uncaught exception, in preparation for an llvm.unwind, which should
// follow immediately after the call to this function.  This function must be
// prepared to deal with foreign exceptions.
//
void __llvm_cxxeh_rethrow(void) {
  llvm_exception *E = CaughtExceptionStack;
  if (E == 0) {
    // 15.1.8 - If there are no uncaught exceptions being thrown, 'throw;'
    // should call terminate.
    //
    assert(0 && "FIXME: this should call E->Terminate!"); // FIXME!
  }

  // Otherwise we have an exception to rethrow. Move it back to the uncaught
  // stack.
  CaughtExceptionStack = E->Next;
  E->Next = UncaughtExceptionStack;
  UncaughtExceptionStack = E;

  // Decrement the number of handlers which are using the exception.
  --E->HandlerCount;
  
  // Return to the caller, which should perform the unwind now.
}

