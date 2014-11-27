/* -----------------------------------------------------------------*-C-*-
   ffitarget.h - Copyright (c) 2012  Anthony Green
                 Copyright (c) 1996-2003  Red Hat, Inc.
   Target configuration macros for SPARC.

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   ``Software''), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED ``AS IS'', WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.

   ----------------------------------------------------------------------- */

#ifndef LIBFFI_TARGET_H
#define LIBFFI_TARGET_H

#ifndef LIBFFI_H
#error "Please do not include ffitarget.h directly into your source.  Use ffi.h instead."
#endif

/* ---- System specific configurations ----------------------------------- */

#if defined(__arch64__) || defined(__sparcv9)
#ifndef SPARC64
#define SPARC64
#endif
#endif

#ifndef LIBFFI_ASM
typedef unsigned long          ffi_arg;
typedef signed long            ffi_sarg;

typedef enum ffi_abi {
  FFI_FIRST_ABI = 0,
  FFI_V8,
  FFI_V8PLUS,
  /* See below for the COMPAT_V9 rationale.  */
  FFI_COMPAT_V9,
  FFI_V9,
  FFI_LAST_ABI,
#ifdef SPARC64
  FFI_DEFAULT_ABI = FFI_V9
#else
  FFI_DEFAULT_ABI = FFI_V8
#endif
} ffi_abi;
#endif

#define V8_ABI_P(abi) ((abi) == FFI_V8 || (abi) == FFI_V8PLUS)
#define V9_ABI_P(abi) ((abi) == FFI_COMPAT_V9 || (abi) == FFI_V9)

#define FFI_TARGET_SPECIFIC_VARIADIC 1

/* The support of variadic functions was broken in the original implementation
   of the FFI_V9 ABI.  This has been fixed by adding one extra field to the
   CIF structure (nfixedargs field), which means that the ABI of libffi itself
   has changed.  In order to support applications using the original ABI, we
   have renamed FFI_V9 into FFI_COMPAT_V9 and defined a new FFI_V9 value.  */
#ifdef SPARC64
#define FFI_EXTRA_CIF_FIELDS unsigned int nfixedargs
#endif

/* ---- Definitions for closures ----------------------------------------- */

#define FFI_CLOSURES 1
#ifdef SPARC64
#define FFI_TRAMPOLINE_SIZE 24
#else
#define FFI_TRAMPOLINE_SIZE 16
#endif
#define FFI_NATIVE_RAW_API 0

#endif
