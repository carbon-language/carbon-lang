// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -DCONSTANT -cl-std=CL2.0
// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -DGLOBAL -cl-std=CL2.0
// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -DGENERIC -cl-std=CL2.0
// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -DCONSTANT -cl-std=clc++
// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -DGLOBAL -cl-std=clc++
// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -DGENERIC -cl-std=clc++
// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -DCONSTANT -cl-std=CL3.0 -cl-ext=+__opencl_c_generic_address_space
// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -DGLOBAL -cl-std=CL3.0 -cl-ext=+__opencl_c_generic_address_space
// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -DGENERIC -cl-std=CL3.0 -cl-ext=+__opencl_c_generic_address_space

/* OpenCLC v2.0 adds a set of restrictions for conversions between pointers to
*  different address spaces, mainly described in Sections 6.5.5 and 6.5.6.
*
*  It adds notion of overlapping address spaces. The main differention is that
*  an unnamed address space is added, called '__generic'. Pointers to the
*  generic address space can be interchangabley used with pointers to any
*  other address space except for __constant address space (Section 6.5.5).
*
*  Based on this there are 3 sets of tests: __generic, named (__global in this
*  case), and __constant, that should cover all program paths for CL address
*  space conversions used in initialisations, assignments, casts, comparisons
*  and arithmetic operations.
*
*  OpenCLC v3.0 supports generic address if __opencl_c_generic_address_space feature is supported
*/

#ifdef GENERIC
#define AS __generic
#define AS_COMP __local
#define AS_INCOMP __constant
#endif

#ifdef GLOBAL
#define AS __global
#define AS_COMP __global
#define AS_INCOMP __local
#endif

#ifdef CONSTANT
#define AS __constant
#define AS_COMP __constant
#define AS_INCOMP __global
#endif

void f_glob(__global int *arg_glob) {}
#ifndef GLOBAL
#if !__OPENCL_CPP_VERSION__
// expected-note@-3{{passing argument to parameter 'arg_glob' here}}
#else
// expected-note-re@-5{{candidate function not viable: cannot pass pointer to address space '__{{generic|constant}}' as a pointer to address space '__global' in 1st argument}}
#endif
#endif

void f_loc(__local int *arg_loc) {}
#if !__OPENCL_CPP_VERSION__
// expected-note@-2{{passing argument to parameter 'arg_loc' here}}
#else
// expected-note-re@-4{{candidate function not viable: cannot pass pointer to address space '__{{global|generic|constant}}' as a pointer to address space '__local' in 1st argument}}
#endif

void f_const(__constant int *arg_const) {}
#ifndef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-note@-3{{passing argument to parameter 'arg_const' here}}
#else
// expected-note-re@-5{{candidate function not viable: cannot pass pointer to address space '__{{global|generic}}' as a pointer to address space '__constant' in 1st argument}}
#endif
#endif

void f_priv(__private int *arg_priv) {}
#if !__OPENCL_CPP_VERSION__
// expected-note@-2{{passing argument to parameter 'arg_priv' here}}
#else
// expected-note-re@-4{{candidate function not viable: cannot pass pointer to address space '__{{global|generic|constant}}' as a pointer to address space '__private' in 1st argument}}
#endif

void f_gen(__generic int *arg_gen) {}
#ifdef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-note@-3{{passing argument to parameter 'arg_gen' here}}
#else
// expected-note@-5{{candidate function not viable: cannot pass pointer to address space '__constant' as a pointer to address space '__generic' in 1st argument}}
#endif
#endif

void test_conversion(__global int *arg_glob, __local int *arg_loc,
                     __constant int *arg_const, __private int *arg_priv,
                     __generic int *arg_gen) {

  AS int *var_init1 = arg_glob;
#ifdef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error@-3{{initializing '__constant int *__private' with an expression of type '__global int *__private' changes address space of pointer}}
#else
// expected-error@-5{{cannot initialize a variable of type '__constant int *__private' with an lvalue of type '__global int *__private'}}
#endif
#endif

  AS int *var_init2 = arg_loc;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{initializing '__{{global|constant}} int *__private' with an expression of type '__local int *__private' changes address space of pointer}}
#else
// expected-error-re@-5{{cannot initialize a variable of type '__{{global|constant}} int *__private' with an lvalue of type '__local int *__private'}}
#endif
#endif

  AS int *var_init3 = arg_const;
#ifndef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{initializing '__{{global|generic}} int *__private' with an expression of type '__constant int *__private' changes address space of pointer}}
#else
// expected-error-re@-5{{cannot initialize a variable of type '__{{global|generic}} int *__private' with an lvalue of type '__constant int *__private'}}
#endif
#endif

  AS int *var_init4 = arg_priv;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{initializing '__{{global|constant}} int *__private' with an expression of type '__private int *__private' changes address space of pointer}}
#else
// expected-error-re@-5{{cannot initialize a variable of type '__{{global|constant}} int *__private' with an lvalue of type '__private int *__private'}}
#endif
#endif

  AS int *var_init5 = arg_gen;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{initializing '__{{global|constant}} int *__private' with an expression of type '__generic int *__private' changes address space of pointer}}
#else
// expected-error-re@-5{{cannot initialize a variable of type '__{{global|constant}} int *__private' with an lvalue of type '__generic int *__private'}}
#endif
#endif

  AS int *var_cast1 = (AS int *)arg_glob;
#ifdef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error@-3{{casting '__global int *' to type '__constant int *' changes address space of pointer}}
#else
// expected-error@-5{{C-style cast from '__global int *' to '__constant int *' converts between mismatching address spaces}}
#endif
#endif

  AS int *var_cast2 = (AS int *)arg_loc;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{casting '__local int *' to type '__{{global|constant}} int *' changes address space of pointer}}
#else
// expected-error-re@-5{{C-style cast from '__local int *' to '__{{global|constant}} int *' converts between mismatching address spaces}}
#endif
#endif

  AS int *var_cast3 = (AS int *)arg_const;
#ifndef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{casting '__constant int *' to type '__{{global|generic}} int *' changes address space of pointer}}
#else
// expected-error-re@-5{{C-style cast from '__constant int *' to '__{{global|generic}} int *' converts between mismatching address spaces}}
#endif
#endif

  AS int *var_cast4 = (AS int *)arg_priv;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{casting '__private int *' to type '__{{global|constant}} int *' changes address space of pointer}}
#else
// expected-error-re@-5{{C-style cast from '__private int *' to '__{{global|constant}} int *' converts between mismatching address spaces}}
#endif
#endif

  AS int *var_cast5 = (AS int *)arg_gen;
#ifdef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error@-3{{casting '__generic int *' to type '__constant int *' changes address space of pointer}}
#else
// expected-error@-5{{C-style cast from '__generic int *' to '__constant int *' converts between mismatching address spaces}}
#endif
#endif

  AS int *var_impl;
  var_impl = arg_glob;
#ifdef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error@-3{{assigning '__global int *__private' to '__constant int *__private' changes address space of pointer}}
#else
// expected-error@-5{{assigning '__global int *__private' to '__constant int *' changes address space of pointer}}
#endif
#endif

  var_impl = arg_loc;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{assigning '__local int *__private' to '__{{global|constant}} int *__private' changes address space of pointer}}
#else
// expected-error-re@-5{{assigning '__local int *__private' to '__{{global|constant}} int *' changes address space of pointer}}
#endif
#endif

  var_impl = arg_const;
#ifndef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{assigning '__constant int *__private' to '__{{global|generic}} int *__private' changes address space of pointer}}
#else
// expected-error-re@-5{{assigning '__constant int *__private' to '__{{global|generic}} int *' changes address space of pointer}}
#endif
#endif

  var_impl = arg_priv;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{assigning '__private int *__private' to '__{{global|constant}} int *__private' changes address space of pointer}}
#else
// expected-error-re@-5{{assigning '__private int *__private' to '__{{global|constant}} int *' changes address space of pointer}}
#endif
#endif

  var_impl = arg_gen;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{assigning '__generic int *__private' to '__{{global|constant}} int *__private' changes address space of pointer}}
#else
// expected-error-re@-5{{assigning '__generic int *__private' to '__{{global|constant}} int *' changes address space of pointer}}
#endif
#endif

  var_cast1 = (AS int *)arg_glob;
#ifdef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error@-3{{casting '__global int *' to type '__constant int *' changes address space of pointer}}
#else
// expected-error@-5{{C-style cast from '__global int *' to '__constant int *' converts between mismatching address spaces}}
#endif
#endif

  var_cast2 = (AS int *)arg_loc;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{casting '__local int *' to type '__{{global|constant}} int *' changes address space of pointer}}
#else
// expected-error-re@-5{{C-style cast from '__local int *' to '__{{global|constant}} int *' converts between mismatching address spaces}}
#endif
#endif

  var_cast3 = (AS int *)arg_const;
#ifndef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{casting '__constant int *' to type '__{{global|generic}} int *' changes address space of pointer}}
#else
// expected-error-re@-5{{C-style cast from '__constant int *' to '__{{global|generic}} int *' converts between mismatching address spaces}}
#endif
#endif

  var_cast4 = (AS int *)arg_priv;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{casting '__private int *' to type '__{{global|constant}} int *' changes address space of pointer}}
#else
// expected-error-re@-5{{C-style cast from '__private int *' to '__{{global|constant}} int *' converts between mismatching address spaces}}
#endif
#endif

  var_cast5 = (AS int *)arg_gen;
#ifdef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error@-3{{casting '__generic int *' to type '__constant int *' changes address space of pointer}}
#else
// expected-error@-5{{C-style cast from '__generic int *' to '__constant int *' converts between mismatching address spaces}}
#endif
#endif

  AS int *var_cmp;
  int b = var_cmp != arg_glob;
#ifdef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error@-3{{comparison between  ('__constant int *' and '__global int *') which are pointers to non-overlapping address spaces}}
#else
// expected-error@-5{{comparison of distinct pointer types ('__constant int *' and '__global int *')}}
#endif
#endif

  b = var_cmp != arg_loc;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{comparison between  ('__{{global|constant}} int *' and '__local int *') which are pointers to non-overlapping address spaces}}
#else
// expected-error-re@-5{{comparison of distinct pointer types ('__{{global|constant}} int *' and '__local int *')}}
#endif
#endif

  b = var_cmp == arg_const;
#ifndef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{comparison between  ('__{{global|generic}} int *' and '__constant int *') which are pointers to non-overlapping address spaces}}
#else
// expected-error-re@-5{{comparison of distinct pointer types ('__{{global|generic}} int *' and '__constant int *')}}
#endif
#endif

  b = var_cmp <= arg_priv;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{comparison between  ('__{{global|constant}} int *' and '__private int *') which are pointers to non-overlapping address spaces}}
#else
// expected-error-re@-5{{comparison of distinct pointer types ('__{{global|constant}} int *' and '__private int *')}}
#endif
#endif

  b = var_cmp >= arg_gen;
#ifdef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error@-3{{comparison between  ('__constant int *' and '__generic int *') which are pointers to non-overlapping address spaces}}
#else
// expected-error@-5{{comparison of distinct pointer types ('__constant int *' and '__generic int *')}}
#endif
#endif

  AS int *var_sub;
  b = var_sub - arg_glob;
#ifdef CONSTANT
// expected-error@-2{{arithmetic operation with operands of type  ('__constant int *' and '__global int *') which are pointers to non-overlapping address spaces}}
#endif

  b = var_sub - arg_loc;
#ifndef GENERIC
// expected-error-re@-2{{arithmetic operation with operands of type  ('__{{global|constant}} int *' and '__local int *') which are pointers to non-overlapping address spaces}}
#endif

  b = var_sub - arg_const;
#ifndef CONSTANT
// expected-error-re@-2{{arithmetic operation with operands of type  ('__{{global|generic}} int *' and '__constant int *') which are pointers to non-overlapping address spaces}}
#endif

  b = var_sub - arg_priv;
#ifndef GENERIC
// expected-error-re@-2{{arithmetic operation with operands of type  ('__{{global|constant}} int *' and '__private int *') which are pointers to non-overlapping address spaces}}
#endif

  b = var_sub - arg_gen;
#ifdef CONSTANT
// expected-error@-2{{arithmetic operation with operands of type  ('__constant int *' and '__generic int *') which are pointers to non-overlapping address spaces}}
#endif

  f_glob(var_sub);
#ifndef GLOBAL
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{passing '__{{constant|generic}} int *__private' to parameter of type '__global int *' changes address space of pointer}}
#else
// expected-error@-5{{no matching function for call to 'f_glob'}}
#endif
#endif

  f_loc(var_sub);
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-2{{passing '__{{global|constant|generic}} int *__private' to parameter of type '__local int *' changes address space of pointer}}
#else
// expected-error@-4{{no matching function for call to 'f_loc'}}
#endif

  f_const(var_sub);
#ifndef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{passing '__{{global|generic}} int *__private' to parameter of type '__constant int *' changes address space of pointer}}
#else
// expected-error@-5{{no matching function for call to 'f_const'}}
#endif
#endif

  f_priv(var_sub);
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-2{{passing '__{{global|constant|generic}} int *__private' to parameter of type '__private int *' changes address space of pointer}}
#else
// expected-error@-4{{no matching function for call to 'f_priv'}}
#endif

  f_gen(var_sub);
#ifdef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error@-3{{passing '__constant int *__private' to parameter of type '__generic int *' changes address space of pointer}}
#else
// expected-error@-5{{no matching function for call to 'f_gen'}}
#endif
#endif
}

void test_ternary() {
  AS int *var_cond;
  __generic int *var_gen;
  __global int *var_glob;
  var_gen = 0 ? var_cond : var_glob;
#ifdef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error@-3{{conditional operator with the second and third operands of type  ('__constant int *' and '__global int *') which are pointers to non-overlapping address spaces}}
#else
// expected-error@-5{{incompatible operand types ('__constant int *' and '__global int *')}}
#endif
#endif

  __local int *var_loc;
  var_gen = 0 ? var_cond : var_loc;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{conditional operator with the second and third operands of type  ('__{{global|constant}} int *' and '__local int *') which are pointers to non-overlapping address spaces}}
#else
// expected-error-re@-5{{incompatible operand types ('__{{global|constant}} int *' and '__local int *')}}
#endif
#endif

  __constant int *var_const;
  var_cond = 0 ? var_cond : var_const;
#ifndef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{conditional operator with the second and third operands of type  ('__{{global|generic}} int *' and '__constant int *') which are pointers to non-overlapping address spaces}}
#else
// expected-error-re@-5{{incompatible operand types ('__{{global|generic}} int *' and '__constant int *')}}
#endif
#endif

  __private int *var_priv;
  var_gen = 0 ? var_cond : var_priv;
#ifndef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error-re@-3{{conditional operator with the second and third operands of type  ('__{{global|constant}} int *' and '__private int *') which are pointers to non-overlapping address spaces}}
#else
// expected-error-re@-5{{incompatible operand types ('__{{global|constant}} int *' and '__private int *')}}
#endif
#endif

  var_gen = 0 ? var_cond : var_gen;
#ifdef CONSTANT
#if !__OPENCL_CPP_VERSION__
// expected-error@-3{{conditional operator with the second and third operands of type  ('__constant int *' and '__generic int *') which are pointers to non-overlapping address spaces}}
#else
// expected-error@-5{{incompatible operand types ('__constant int *' and '__generic int *')}}
#endif
#endif

  void *var_void_gen;
  __global char *var_glob_ch;
  var_void_gen = 0 ? var_cond : var_glob_ch;
#if __OPENCL_CPP_VERSION__
// expected-error-re@-2{{incompatible operand types ('__{{constant|global|generic}} int *' and '__global char *')}}
#else
#ifdef CONSTANT
// expected-error@-5{{conditional operator with the second and third operands of type  ('__constant int *' and '__global char *') which are pointers to non-overlapping address spaces}}
#else
// expected-warning-re@-7{{pointer type mismatch ('__{{global|generic}} int *' and '__global char *')}}
#endif
#endif

  __local char *var_loc_ch;
  var_void_gen = 0 ? var_cond : var_loc_ch;
#if __OPENCL_CPP_VERSION__
// expected-error-re@-2{{incompatible operand types ('__{{constant|global|generic}} int *' and '__local char *')}}
#else
#ifndef GENERIC
// expected-error-re@-5{{conditional operator with the second and third operands of type  ('__{{global|constant}} int *' and '__local char *') which are pointers to non-overlapping address spaces}}
#else
// expected-warning@-7{{pointer type mismatch ('__generic int *' and '__local char *')}}
#endif
#endif

  __constant void *var_void_const;
  __constant char *var_const_ch;
  var_void_const = 0 ? var_cond : var_const_ch;
#if __OPENCL_CPP_VERSION__
// expected-error-re@-2{{incompatible operand types ('__{{constant|global|generic}} int *' and '__constant char *')}}
#else
#ifndef CONSTANT
// expected-error-re@-5{{conditional operator with the second and third operands of type  ('__{{global|generic}} int *' and '__constant char *') which are pointers to non-overlapping address spaces}}
#else
// expected-warning@-7{{pointer type mismatch ('__constant int *' and '__constant char *')}}
#endif
#endif

  __private char *var_priv_ch;
  var_void_gen = 0 ? var_cond : var_priv_ch;
#if __OPENCL_CPP_VERSION__
// expected-error-re@-2{{incompatible operand types ('__{{constant|global|generic}} int *' and '__private char *')}}
#else
#ifndef GENERIC
// expected-error-re@-5{{conditional operator with the second and third operands of type  ('__{{global|constant}} int *' and '__private char *') which are pointers to non-overlapping address spaces}}
#else
// expected-warning@-7{{pointer type mismatch ('__generic int *' and '__private char *')}}
#endif
#endif

  __generic char *var_gen_ch;
  var_void_gen = 0 ? var_cond : var_gen_ch;
#if __OPENCL_CPP_VERSION__
// expected-error-re@-2{{incompatible operand types ('__{{constant|global|generic}} int *' and '__generic char *')}}
#else
#ifdef CONSTANT
// expected-error@-5{{conditional operator with the second and third operands of type  ('__constant int *' and '__generic char *') which are pointers to non-overlapping address spaces}}
#else
// expected-warning-re@-7{{pointer type mismatch ('__{{global|generic}} int *' and '__generic char *')}}
#endif
#endif
}

void test_pointer_chains() {
  AS int *AS *var_as_as_int;
  AS int *AS_COMP *var_asc_as_int;
  AS_INCOMP int *AS_COMP *var_asc_asn_int;
  AS_COMP int *AS_COMP *var_asc_asc_int;

  // Case 1:
  //  * address spaces of corresponded most outer pointees overlaps, their canonical types are equal
  //  * CVR, address spaces and canonical types of the rest of pointees are equivalent.
  var_as_as_int = var_asc_as_int;
  var_as_as_int = 0 ? var_as_as_int : var_asc_as_int;

  // Case 2: Corresponded inner pointees has non-overlapping address spaces.
  var_as_as_int = 0 ? var_as_as_int : var_asc_asn_int;
#if !__OPENCL_CPP_VERSION__
// expected-warning-re@-2{{pointer type mismatch ('__{{(generic|global|constant)}} int *__{{(generic|global|constant)}} *' and '__{{(local|global|constant)}} int *__{{(constant|local|global)}} *')}}
#else
// expected-error-re@-4{{incompatible operand types ('__{{(generic|global|constant)}} int *__{{(generic|global|constant)}} *' and '__{{(local|global|constant)}} int *__{{(constant|local|global)}} *')}}
#endif

  // Case 3: Corresponded inner pointees has overlapping but not equivalent address spaces.
  var_as_as_int = var_asc_asc_int;
#ifdef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-error@-3 {{assigning '__local int *__local *__private' to '__generic int *__generic *__private' changes address space of nested pointer}}
#else
// expected-error@-5 {{assigning '__local int *__local *__private' to '__generic int *__generic *' changes address space of nested pointer}}
#endif
#endif

  var_as_as_int = (AS int *AS *)var_asc_asc_int;
#ifdef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-warning@-3 {{casting '__local int *__local *' to type '__generic int *__generic *' discards qualifiers in nested pointer types}}
#else
// expected-warning@-5 {{C-style cast from '__local int *__local *' to '__generic int *__generic *' changes address space of nested pointers}}
#endif
#endif

  var_as_as_int = (AS int *AS *)var_asc_asn_int;
#if !__OPENCL_CPP_VERSION__
// expected-warning-re@-2 {{casting '__{{global|local|constant}} int *__{{local|constant|global}} *' to type '__{{global|constant|generic}} int *__{{global|constant|generic}} *' discards qualifiers in nested pointer types}}
#else
// expected-warning-re@-4 {{C-style cast from '__{{global|local|constant}} int *__{{local|constant|global}} *' to '__{{global|constant|generic}} int *__{{global|constant|generic}} *' changes address space of nested pointers}}
#endif

  var_as_as_int = 0 ? var_as_as_int : var_asc_asc_int;
#ifdef GENERIC
#if !__OPENCL_CPP_VERSION__
// expected-warning@-3{{pointer type mismatch ('__generic int *__generic *' and '__local int *__local *')}}
#else
// expected-error@-5 {{incompatible operand types ('__generic int *__generic *' and '__local int *__local *')}}
#endif
#endif
}
