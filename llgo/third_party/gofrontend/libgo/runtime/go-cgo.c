/* go-cgo.c -- SWIG support routines for libgo.

   Copyright 2011 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-alloc.h"
#include "interface.h"
#include "go-panic.h"

/* Prepare to call from code written in Go to code written in C or
   C++.  This takes the current goroutine out of the Go scheduler, as
   though it were making a system call.  Otherwise the program can
   lock up if the C code goes to sleep on a mutex or for some other
   reason.  This idea is to call this function, then immediately call
   the C/C++ function.  After the C/C++ function returns, call
   syscall_cgocalldone.  The usual Go code would look like

       syscall.Cgocall()
       defer syscall.Cgocalldone()
       cfunction()

   */

/* We let Go code call these via the syscall package.  */
void syscall_cgocall(void) __asm__ (GOSYM_PREFIX "syscall.Cgocall");
void syscall_cgocalldone(void) __asm__ (GOSYM_PREFIX "syscall.CgocallDone");
void syscall_cgocallback(void) __asm__ (GOSYM_PREFIX "syscall.CgocallBack");
void syscall_cgocallbackdone(void) __asm__ (GOSYM_PREFIX "syscall.CgocallBackDone");

void
syscall_cgocall ()
{
  M* m;
  G* g;

  if (runtime_needextram && runtime_cas (&runtime_needextram, 1, 0))
    runtime_newextram ();

  m = runtime_m ();
  ++m->ncgocall;
  g = runtime_g ();
  ++g->ncgo;
  runtime_entersyscall ();
}

/* Prepare to return to Go code from C/C++ code.  */

void
syscall_cgocalldone ()
{
  G* g;

  g = runtime_g ();
  __go_assert (g != NULL);
  --g->ncgo;
  if (g->ncgo == 0)
    {
      /* We are going back to Go, and we are not in a recursive call.
	 Let the garbage collector clean up any unreferenced
	 memory.  */
      g->cgomal = NULL;
    }

  /* If we are invoked because the C function called _cgo_panic, then
     _cgo_panic will already have exited syscall mode.  */
  if (g->status == Gsyscall)
    runtime_exitsyscall ();
}

/* Call back from C/C++ code to Go code.  */

void
syscall_cgocallback ()
{
  M *mp;

  mp = runtime_m ();
  if (mp == NULL)
    {
      runtime_needm ();
      mp = runtime_m ();
      mp->dropextram = true;
    }

  runtime_exitsyscall ();

  mp = runtime_m ();
  if (mp->needextram)
    {
      mp->needextram = 0;
      runtime_newextram ();
    }
}

/* Prepare to return to C/C++ code from a callback to Go code.  */

void
syscall_cgocallbackdone ()
{
  M *mp;

  runtime_entersyscall ();
  mp = runtime_m ();
  if (mp->dropextram && runtime_g ()->ncgo == 0)
    {
      mp->dropextram = false;
      runtime_dropm ();
    }
}

/* Allocate memory and save it in a list visible to the Go garbage
   collector.  */

void *
alloc_saved (size_t n)
{
  void *ret;
  G *g;
  CgoMal *c;

  ret = __go_alloc (n);

  g = runtime_g ();
  c = (CgoMal *) __go_alloc (sizeof (CgoMal));
  c->next = g->cgomal;
  c->alloc = ret;
  g->cgomal = c;

  return ret;
}

/* These are routines used by SWIG.  The gc runtime library provides
   the same routines under the same name, though in that case the code
   is required to import runtime/cgo.  */

void *
_cgo_allocate (size_t n)
{
  void *ret;

  runtime_exitsyscall ();
  ret = alloc_saved (n);
  runtime_entersyscall ();
  return ret;
}

extern const struct __go_type_descriptor string_type_descriptor
  __asm__ (GOSYM_PREFIX "__go_tdn_string");

void
_cgo_panic (const char *p)
{
  intgo len;
  unsigned char *data;
  String *ps;
  struct __go_empty_interface e;

  runtime_exitsyscall ();
  len = __builtin_strlen (p);
  data = alloc_saved (len);
  __builtin_memcpy (data, p, len);
  ps = alloc_saved (sizeof *ps);
  ps->str = data;
  ps->len = len;
  e.__type_descriptor = &string_type_descriptor;
  e.__object = ps;

  /* We don't call runtime_entersyscall here, because normally what
     will happen is that we will walk up the stack to a Go deferred
     function that calls recover.  However, this will do the wrong
     thing if this panic is recovered and the stack unwinding is
     caught by a C++ exception handler.  It might be possible to
     handle this by calling runtime_entersyscall in the personality
     function in go-unwind.c.  FIXME.  */

  __go_panic (e);
}
