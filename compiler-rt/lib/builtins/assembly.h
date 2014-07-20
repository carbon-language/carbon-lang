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
#define HIDDEN(name) .private_extern name
#define LOCAL_LABEL(name) L_##name
// tell linker it can break up file at label boundaries
#define FILE_LEVEL_DIRECTIVE .subsections_via_symbols
#define SYMBOL_IS_FUNC(name)
#elif defined(__ELF__)
#define HIDDEN(name) .hidden name
#define LOCAL_LABEL(name) .L_##name
#define FILE_LEVEL_DIRECTIVE
#if defined(__arm__)
#define SYMBOL_IS_FUNC(name) .type name,%function
#else
#define SYMBOL_IS_FUNC(name) .type name,@function
#endif
#else
#define HIDDEN_DIRECTIVE(name)
#define LOCAL_LABEL(name) .L ## name
#define SYMBOL_IS_FUNC(name)                                                   \
  .def name SEPARATOR                                                          \
    .scl 2 SEPARATOR                                                           \
    .type 32 SEPARATOR                                                         \
  .endef
#define FILE_LEVEL_DIRECTIVE
#endif

#if defined(__arm__)
#ifndef __ARM_ARCH
#if defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) ||                     \
    defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) ||                    \
    defined(__ARM_ARCH_7EM__)
#define __ARM_ARCH 7
#endif
#endif

#ifndef __ARM_ARCH
#if defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) ||                     \
    defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6Z__) ||                    \
    defined(__ARM_ARCH_6ZK__) || defined(__ARM_ARCH_6ZM__)
#define __ARM_ARCH 6
#endif
#endif

#ifndef __ARM_ARCH
#if defined(__ARM_ARCH_5__) || defined(__ARM_ARCH_5T__) ||                     \
    defined(__ARM_ARCH_5TE__) || defined(__ARM_ARCH_5TEJ__)
#define __ARM_ARCH 5
#endif
#endif

#ifndef __ARM_ARCH
#define __ARM_ARCH 4
#endif

#if defined(__ARM_ARCH_4T__) || __ARM_ARCH >= 5
#define ARM_HAS_BX
#endif
#if !defined(__ARM_FEATURE_CLZ) &&                                             \
    (__ARM_ARCH >= 6 || (__ARM_ARCH == 5 && !defined(__ARM_ARCH_5__)))
#define __ARM_FEATURE_CLZ
#endif

#ifdef ARM_HAS_BX
#define JMP(r) bx r
#define JMPc(r, c) bx##c r
#else
#define JMP(r) mov pc, r
#define JMPc(r, c) mov##c pc, r
#endif
#endif

#define GLUE2(a, b) a##b
#define GLUE(a, b) GLUE2(a, b)
#define SYMBOL_NAME(name) GLUE(__USER_LABEL_PREFIX__, name)

#ifdef VISIBILITY_HIDDEN
#define DECLARE_SYMBOL_VISIBILITY(name)                                        \
  HIDDEN(SYMBOL_NAME(name)) SEPARATOR
#else
#define DECLARE_SYMBOL_VISIBILITY(name)
#endif

#define DEFINE_COMPILERRT_FUNCTION(name)                                       \
  FILE_LEVEL_DIRECTIVE SEPARATOR                                               \
  .globl SYMBOL_NAME(name) SEPARATOR                                           \
  SYMBOL_IS_FUNC(SYMBOL_NAME(name)) SEPARATOR                                  \
  DECLARE_SYMBOL_VISIBILITY(name)                                              \
  SYMBOL_NAME(name):

#define DEFINE_COMPILERRT_PRIVATE_FUNCTION(name)                               \
  FILE_LEVEL_DIRECTIVE SEPARATOR                                               \
  .globl SYMBOL_NAME(name) SEPARATOR                                           \
  SYMBOL_IS_FUNC(SYMBOL_NAME(name)) SEPARATOR                                  \
  HIDDEN(SYMBOL_NAME(name)) SEPARATOR                                          \
  SYMBOL_NAME(name):

#define DEFINE_COMPILERRT_PRIVATE_FUNCTION_UNMANGLED(name)                     \
  .globl name SEPARATOR                                                        \
  SYMBOL_IS_FUNC(name) SEPARATOR                                               \
  HIDDEN(name) SEPARATOR                                                       \
  name:

#define DEFINE_COMPILERRT_FUNCTION_ALIAS(name, target)                         \
  .globl SYMBOL_NAME(name) SEPARATOR                                           \
  SYMBOL_IS_FUNC(SYMBOL_NAME(name)) SEPARATOR                                  \
  .set SYMBOL_NAME(name), SYMBOL_NAME(target) SEPARATOR

#if defined(__ARM_EABI__)
#define DEFINE_AEABI_FUNCTION_ALIAS(aeabi_name, name)                          \
  DEFINE_COMPILERRT_FUNCTION_ALIAS(aeabi_name, name)
#else
#define DEFINE_AEABI_FUNCTION_ALIAS(aeabi_name, name)
#endif

#ifdef __ELF__
#define END_COMPILERRT_FUNCTION(name)                                          \
  .size SYMBOL_NAME(name), . - SYMBOL_NAME(name)
#else
#define END_COMPILERRT_FUNCTION(name)
#endif

#endif /* COMPILERRT_ASSEMBLY_H */
