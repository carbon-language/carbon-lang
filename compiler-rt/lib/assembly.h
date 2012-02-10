/* ===-- assembly.h - compiler-rt assembler support macros -----------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
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

#if defined(__POWERPC__) || defined(__powerpc__) || defined(__ppc__)
#define SEPARATOR @
#else
#define SEPARATOR ;
#endif

#if defined(__APPLE__)
#define HIDDEN_DIRECTIVE .private_extern
#define LOCAL_LABEL(name) L_##name
#else
#define HIDDEN_DIRECTIVE .hidden
#define LOCAL_LABEL(name) .L_##name
#endif

#define GLUE2(a, b) a ## b
#define GLUE(a, b) GLUE2(a, b)
#define SYMBOL_NAME(name) GLUE(__USER_LABEL_PREFIX__, name)

#if defined(VISIBILITY_HIDDEN)
#define DECLARE_SYMBOL_VISIBILITY(name)                    \
  HIDDEN_DIRECTIVE name SEPARATOR
#else
#define DECLARE_SYMBOL_VISIBILITY(name)
#endif

#if defined(__APPLE__)
#define THUMB_FUNC(name) .thumb_func name
#else
#define THUMB_FUNC(name) .thumb_func
#endif

#if defined(__thumb__)
#define TARGET_FUNCTION_SETTINGS(name)                     \
  THUMB_FUNC(name) SEPARATOR                               \
  .code 16 SEPARATOR
#else
#define TARGET_FUNCTION_SETTINGS(name)
#endif

#define DEFINE_COMPILERRT_FUNCTION(name)                   \
  .globl SYMBOL_NAME(name) SEPARATOR                       \
  DECLARE_SYMBOL_VISIBILITY(SYMBOL_NAME(name))             \
  TARGET_FUNCTION_SETTINGS(SYMBOL_NAME(name))              \
  SYMBOL_NAME(name):

#define DEFINE_COMPILERRT_PRIVATE_FUNCTION(name)           \
  .globl SYMBOL_NAME(name) SEPARATOR                       \
  HIDDEN_DIRECTIVE SYMBOL_NAME(name) SEPARATOR             \
  TARGET_FUNCTION_SETTINGS(SYMBOL_NAME(name))              \
  SYMBOL_NAME(name):

#define DEFINE_COMPILERRT_PRIVATE_FUNCTION_UNMANGLED(name) \
  .globl name SEPARATOR                                    \
  HIDDEN_DIRECTIVE name SEPARATOR                          \
  TARGET_FUNCTION_SETTINGS(name)                           \
  name:

#define DEFINE_COMPILERRT_FUNCTION_ALIAS(name, target)     \
  .globl SYMBOL_NAME(name) SEPARATOR                       \
  .set SYMBOL_NAME(name), SYMBOL_NAME(target) SEPARATOR

#if defined (__ARM_EABI__)
# define DEFINE_AEABI_FUNCTION_ALIAS(aeabi_name, name)      \
  DEFINE_COMPILERRT_FUNCTION_ALIAS(aeabi_name, name)
#else
# define DEFINE_AEABI_FUNCTION_ALIAS(aeabi_name, name)
#endif

#endif /* COMPILERRT_ASSEMBLY_H */
