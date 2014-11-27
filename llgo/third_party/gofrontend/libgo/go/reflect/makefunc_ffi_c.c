// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "go-type.h"
#include "go-panic.h"

#ifdef USE_LIBFFI

#include "go-ffi.h"

#if FFI_CLOSURES
#define USE_LIBFFI_CLOSURES
#endif

#endif /* defined(USE_LIBFFI) */

/* Declare C functions with the names used to call from Go.  */

struct ffi_ret {
  void *code;
  void *data;
  void *cif;
};

struct ffi_ret ffi(const struct __go_func_type *ftyp, FuncVal *callback)
  __asm__ (GOSYM_PREFIX "reflect.ffi");

void ffiFree(void *data)
  __asm__ (GOSYM_PREFIX "reflect.ffiFree");

#ifdef USE_LIBFFI_CLOSURES

/* The function that we pass to ffi_prep_closure_loc.  This calls the
   Go callback function (passed in user_data) with the pointer to the
   arguments and the results area.  */

static void ffi_callback (ffi_cif *, void *, void **, void *)
  __asm__ ("reflect.ffi_callback");

static void
ffi_callback (ffi_cif* cif __attribute__ ((unused)), void *results,
	      void **args, void *user_data)
{
  Location locs[8];
  int n;
  int i;
  FuncVal *fv;
  void (*f) (void *, void *);

  /* This function is called from some series of FFI closure functions
     called by a Go function.  We want to see whether the caller of
     the closure functions can recover.  Look up the stack and skip
     the FFI functions.  */
  n = runtime_callers (1, &locs[0], sizeof locs / sizeof locs[0], true);
  for (i = 0; i < n; i++)
    {
      const byte *name;

      if (locs[i].function.len == 0)
	continue;
      if (locs[i].function.len < 4)
	break;
      name = locs[i].function.str;
      if (name[0] != 'f' || name[1] != 'f' || name[2] != 'i' || name[3] != '_')
	break;
    }
  if (i < n)
    __go_makefunc_ffi_can_recover (locs + i, n - i);

  fv = (FuncVal *) user_data;
  __go_set_closure (fv);
  f = (void *) fv->fn;
  f (args, results);

  if (i < n)
    __go_makefunc_returning ();
}

/* Allocate an FFI closure and arrange to call ffi_callback.  */

struct ffi_ret
ffi (const struct __go_func_type *ftyp, FuncVal *callback)
{
  ffi_cif *cif;
  void *code;
  void *data;
  struct ffi_ret ret;

  cif = (ffi_cif *) __go_alloc (sizeof (ffi_cif));
  __go_func_to_cif (ftyp, 0, 0, cif);
  data = ffi_closure_alloc (sizeof (ffi_closure), &code);
  if (data == NULL)
    runtime_panicstring ("ffi_closure_alloc failed");
  if (ffi_prep_closure_loc (data, cif, ffi_callback, callback, code)
      != FFI_OK)
    runtime_panicstring ("ffi_prep_closure_loc failed");
  ret.code = code;
  ret.data = data;
  ret.cif = cif;
  return ret;
}

/* Free the FFI closure.  */

void
ffiFree (void *data)
{
  ffi_closure_free (data);
}

#else /* !defined(USE_LIBFFI_CLOSURES) */

struct ffi_ret
ffi(const struct __go_func_type *ftyp, FuncVal *callback)
{
  runtime_panicstring ("libgo built without FFI does not support "
		       "reflect.MakeFunc");
}

void ffiFree(void *data)
{
  runtime_panicstring ("libgo built without FFI does not support "
		       "reflect.MakeFunc");
}

#endif
