//===-- llvm/Support/Compiler.h - Compiler abstraction support --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines several macros, based on the current compiler.  This allows
// use of compiler-specific features in a way that remains portable.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_COMPILER_H
#define LLVM_SUPPORT_COMPILER_H

/// LLVM_LIBRARY_VISIBILITY - If a class marked with this attribute is linked
/// into a shared library, then the class should be private to the library and
/// not accessible from outside it.  Can also be used to mark variables and
/// functions, making them private to any shared library they are linked into.

/// LLVM_GLOBAL_VISIBILITY - If a class marked with this attribute is linked
/// into a shared library, then the class will be accessible from outside the
/// the library.  Can also be used to mark variables and functions, making them
/// accessible from outside any shared library they are linked into.
#if defined(__MINGW32__) || defined(__CYGWIN__)
#define LLVM_LIBRARY_VISIBILITY
#define LLVM_GLOBAL_VISIBILITY __declspec(dllexport)
#elif (__GNUC__ >= 4)
#define LLVM_LIBRARY_VISIBILITY __attribute__ ((visibility("hidden")))
#define LLVM_GLOBAL_VISIBILITY __attribute__ ((visibility("default")))
#else
#define LLVM_LIBRARY_VISIBILITY
#define LLVM_GLOBAL_VISIBILITY
#endif

#if (__GNUC__ >= 4 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#define ATTRIBUTE_USED __attribute__((__used__))
#else
#define ATTRIBUTE_USED
#endif

#if (__GNUC__ >= 4 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#define ATTRIBUTE_UNUSED __attribute__((__unused__))
#else
#define ATTRIBUTE_UNUSED
#endif

#ifdef __GNUC__ // aka 'ATTRIBUTE_CONST' but following LLVM Conventions.
#define ATTRIBUTE_READNONE __attribute__((__const__))
#else
#define ATTRIBUTE_READNONE
#endif

#ifdef __GNUC__  // aka 'ATTRIBUTE_PURE' but following LLVM Conventions.
#define ATTRIBUTE_READONLY __attribute__((__pure__))
#else
#define ATTRIBUTE_READONLY
#endif

#if (__GNUC__ >= 4)
#define BUILTIN_EXPECT(EXPR, VALUE) __builtin_expect((EXPR), (VALUE))
#else
#define BUILTIN_EXPECT(EXPR, VALUE) (EXPR)
#endif

// C++ doesn't support 'extern template' of template specializations.  GCC does,
// but requires __extension__ before it.  In the header, use this:
//   EXTERN_TEMPLATE_INSTANTIATION(class foo<bar>);
// in the .cpp file, use this:
//   TEMPLATE_INSTANTIATION(class foo<bar>);
#ifdef __GNUC__
#define EXTERN_TEMPLATE_INSTANTIATION(X) __extension__ extern template X
#define TEMPLATE_INSTANTIATION(X) template X
#else
#define EXTERN_TEMPLATE_INSTANTIATION(X)
#define TEMPLATE_INSTANTIATION(X)
#endif

// DISABLE_INLINE - On compilers where we have a directive to do so, mark a
// method "not for inlining".
#if (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#define DISABLE_INLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define DISABLE_INLINE __declspec(noinline)
#else
#define DISABLE_INLINE
#endif

// ALWAYS_INLINE - On compilers where we have a directive to do so, mark a
// method "always inline" because it is performance sensitive.
// GCC 3.4 supported this but is buggy in various cases and produces
// unimplemented errors, just use it in GCC 4.0 and later.
#if __GNUC__ > 3
#define ALWAYS_INLINE __attribute__((always_inline))
#else
// TODO: No idea how to do this with MSVC.
#define ALWAYS_INLINE
#endif


#ifdef __GNUC__
#define NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define NORETURN __declspec(noreturn)
#else
#define NORETURN
#endif

#endif
