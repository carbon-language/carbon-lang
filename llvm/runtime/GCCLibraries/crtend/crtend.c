/*===- crtend.c - Initialization code for programs ------------------------===*\
 *
 * This file defines the __main function, which is used to run static
 * constructors and destructors in C++ programs, or with C programs that use GCC
 * extensions to accomplish the same effect.
 *
 * The main data structures used to implement this functionality is the
 * llvm.global_ctors and llvm.global_dtors lists, which are null terminated
 * lists of TorRec (defined below) structures.
 *
\*===----------------------------------------------------------------------===*/

#include <stdlib.h>

/* TorRec - The record type for each element of the ctor/dtor list */
typedef struct TorRec {
  int Priority;
  void (*FP)(void);
} TorRec;

/* __llvm_getGlobalCtors, __llvm_getGlobalDtors - Interface to the LLVM
 * listend.ll file to get access to the start of the ctor and dtor lists...
 */
TorRec *__llvm_getGlobalCtors(void);
TorRec *__llvm_getGlobalDtors(void);

static void run_destructors(void);

/* __main - A call to this function is automatically inserted into the top of
 * the "main" function in the program compiled.  This function is responsible
 * for calling static constructors before the program starts executing.
 */
void __main(void) {
  /* Loop over all of the constructor records, calling each function pointer. */
  TorRec *R = __llvm_getGlobalCtors();

  /* Recursively calling main is not legal C, but lots of people do it for
   * testing stuff.  We might as well work for them.
   */
  static _Bool Initialized = 0;
  if (Initialized) return;
  Initialized = 1;

  /* Only register the global dtor handler if there is at least one global
   * dtor!
   */
  if (__llvm_getGlobalDtors()[0].FP)
    if (atexit(run_destructors))
      abort();  /* Should be able to install ONE atexit handler! */

  /* FIXME: This should sort the list by priority! */
  if (R->FP)
    for (; R->FP; ++R)
      R->FP();
}

static void run_destructors(void) {
  /* Loop over all of the destructor records, calling each function pointer. */
  TorRec *R = __llvm_getGlobalDtors();

  /* FIXME: This should sort the list by priority! */
  for (; R->FP; ++R)
    R->FP();
}
