/* ===-- assembly.h - compiler-rt assembler support macros -----------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file defines macros for use in compiler-rt assembler source.
 * This file is not part of the interface of this library.
 *
 * ===----------------------------------------------------------------------===
 */

#ifndef COMPILERRT_ASSEMBLY_H
#define COMPILERRT_ASSEMBLY_H

// Define SYMBOL_NAME to add the appropriate symbol prefix; we can't use
// USER_LABEL_PREFIX directly because of cpp brokenness.
#if defined(__POWERPC__) || defined(__powerpc__) || defined(__ppc__)

#define SYMBOL_NAME(name) name
#define SEPARATOR @

#else

#define SYMBOL_NAME(name) _##name
#define SEPARATOR ;

#endif

#define DEFINE_COMPILERRT_FUNCTION(name) \
  .globl SYMBOL_NAME(name) SEPARATOR     \
  SYMBOL_NAME(name):

#define DEFINE_COMPILERRT_PRIVATE_FUNCTION(name) \
  .globl SYMBOL_NAME(name) SEPARATOR             \
  .private_extern SYMBOL_NAME(name) SEPARATOR    \
  SYMBOL_NAME(name):

#endif /* COMPILERRT_ASSEMBLY_H */
