/* ===-- assembly.h - libUnwind assembler support macros -------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file defines macros for use in libUnwind assembler source.
 * This file is not part of the interface of this library.
 *
 * ===----------------------------------------------------------------------===
 */

#ifndef UNWIND_ASSEMBLY_H
#define UNWIND_ASSEMBLY_H

#if defined(__POWERPC__) || defined(__powerpc__) || defined(__ppc__)
#define SEPARATOR @
#else
#define SEPARATOR ;
#endif

#if defined(__APPLE__)
#define HIDDEN_DIRECTIVE .private_extern
#else
#define HIDDEN_DIRECTIVE .hidden
#endif

#define GLUE2(a, b) a ## b
#define GLUE(a, b) GLUE2(a, b)
#define SYMBOL_NAME(name) GLUE(__USER_LABEL_PREFIX__, name)

#define DEFINE_LIBUNWIND_FUNCTION(name)                   \
  .globl SYMBOL_NAME(name) SEPARATOR                      \
  SYMBOL_NAME(name):

#define DEFINE_LIBUNWIND_PRIVATE_FUNCTION(name)           \
  .globl SYMBOL_NAME(name) SEPARATOR                      \
  HIDDEN_DIRECTIVE SYMBOL_NAME(name) SEPARATOR            \
  SYMBOL_NAME(name):

#endif /* UNWIND_ASSEMBLY_H */
