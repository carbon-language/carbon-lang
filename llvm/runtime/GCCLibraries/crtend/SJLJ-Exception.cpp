//===- SJLJ-Exception.cpp - SetJmp/LongJmp Exception Handling -------------===//
//
// This file implements the API used by the Setjmp/Longjmp exception handling
// runtime library.
//
//===----------------------------------------------------------------------===//

#include "SJLJ-Exception.h"
#include <cstdlib>
#include <cassert>

// Assert should only be used for debugging the runtime library.  Enabling it in
// CVS will break some platforms!
#undef assert
#define assert(X)

// get_sjlj_exception - Adjust the llvm_exception pointer to be an appropriate
// llvm_sjlj_exception pointer.
inline llvm_sjlj_exception *get_sjlj_exception(llvm_exception *E) {
  assert(E->ExceptionType == SJLJException);
  return (llvm_sjlj_exception*)(E+1) - 1;
}

// SetJmpMapEntry - One entry in a linked list of setjmps for the current
// function.
struct SetJmpMapEntry {
  void *JmpBuf;
  unsigned SetJmpID;
  SetJmpMapEntry *Next;
};

// SJLJDestructor - This function is used to free the exception when
// language-indent code needs to destroy the exception without knowing exactly
// what type it is.
static void SJLJDestructor(llvm_exception *E) {
  free(get_sjlj_exception(E));
}


// __llvm_sjljeh_throw_longjmp - This function creates the longjmp exception and
// returns.  It takes care of mapping the longjmp value from 0 -> 1 as
// appropriate.  The caller should immediately call llvm.unwind after this
// function call.
void __llvm_sjljeh_throw_longjmp(void *JmpBuffer, int Val) throw() {
  llvm_sjlj_exception *E =
    (llvm_sjlj_exception *)malloc(sizeof(llvm_sjlj_exception));
  E->BaseException.ExceptionDestructor = SJLJDestructor;
  E->BaseException.ExceptionType = SJLJException;
  E->BaseException.HandlerCount = 0;
  E->BaseException.isRethrown = 0;
  E->JmpBuffer = JmpBuffer;
  E->LongJmpValue = Val ? Val : 1;

  __llvm_eh_add_uncaught_exception(&E->BaseException);
}

// __llvm_sjljeh_init_setjmpmap - This funciton initializes the pointer provided
// to an empty setjmp map, and should be called on entry to a function which
// calls setjmp.
void __llvm_sjljeh_init_setjmpmap(void **SetJmpMap) throw() {
  *SetJmpMap = 0;
}

// __llvm_sjljeh_destroy_setjmpmap - This function frees all memory associated
// with the specified setjmpmap structure.  It should be called on all exits
// (returns or unwinds) from the function which calls ...init_setjmpmap.
void __llvm_sjljeh_destroy_setjmpmap(void **SetJmpMap) throw() {
  SetJmpMapEntry *Next;
  for (SetJmpMapEntry *SJE = *(SetJmpMapEntry**)SetJmpMap; SJE; SJE = Next) {
    Next = SJE->Next;
    free(SJE);
  }
}

// __llvm_sjljeh_add_setjmp_to_map - This function adds or updates an entry to
// the map, to indicate which setjmp should be returned to if a longjmp happens.
void __llvm_sjljeh_add_setjmp_to_map(void **SetJmpMap, void *JmpBuf,
                                     unsigned SetJmpID) throw() {
  SetJmpMapEntry **SJE = (SetJmpMapEntry**)SetJmpMap;

  // Scan for a pre-existing entry...
  for (; *SJE; SJE = &(*SJE)->Next)
    if ((*SJE)->JmpBuf == JmpBuf) {
      (*SJE)->SetJmpID = SetJmpID;
      return;
    }

  // No prexisting entry found, append to the end of the list...
  SetJmpMapEntry *New = (SetJmpMapEntry *)malloc(sizeof(SetJmpMapEntry));
  *SJE = New;
  New->JmpBuf = JmpBuf;
  New->SetJmpID = SetJmpID;
  New->Next = 0;
}

// __llvm_sjljeh_is_longjmp_exception - This function returns true if the
// current uncaught exception is a longjmp exception.  This is the first step of
// catching a sjlj exception.
bool __llvm_sjljeh_is_longjmp_exception() throw() {
  return __llvm_eh_current_uncaught_exception_type(SJLJException) != 0;
}
  
// __llvm_sjljeh_get_longjmp_value - This function returns the value that the
// setjmp call should "return".  This requires that the current uncaught
// exception be a sjlj exception, though it does not require the exception to be
// caught by this function.
int __llvm_sjljeh_get_longjmp_value() throw() {
  llvm_sjlj_exception *E =
    get_sjlj_exception(__llvm_eh_get_uncaught_exception());
  return E->LongJmpValue;
}

// __llvm_sjljeh_try_catching_longjmp_exception - This function checks to see if
// the current uncaught longjmp exception matches any of the setjmps collected
// in the setjmpmap structure.  If so, it catches and destroys the exception,
// returning the index of the setjmp which caught the exception.  If not, it
// leaves the exception uncaught and returns a value of ~0.
unsigned __llvm_sjljeh_try_catching_longjmp_exception(void **SetJmpMap) throw(){
  llvm_sjlj_exception *E =
    get_sjlj_exception(__llvm_eh_get_uncaught_exception());
 
  // Scan for a matching entry in the SetJmpMap...
  SetJmpMapEntry *SJE = *(SetJmpMapEntry**)SetJmpMap;
  for (; SJE; SJE = SJE->Next)
    if (SJE->JmpBuf == E->JmpBuffer) {
      // "Catch" and destroy the exception...
      __llvm_eh_pop_from_uncaught_stack();

      // We know it's a longjmp exception, so we can just free it instead of
      // calling the destructor.
      free(E);

      // Return the setjmp ID which we should branch to...
      return SJE->SetJmpID;
    }
  
  // No setjmp in this function catches the exception!
  return ~0;
}
