//===--- VariadicFunctions.h - Variadic Functions ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements compile-time type-safe variadic functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_VARIADIC_FUNCTION_H
#define LLVM_CLANG_TOOLING_VARIADIC_FUNCTION_H

#include <stddef.h>  // Defines NULL.

namespace clang {
namespace tooling {
namespace internal {

/// The VariadicFunction class template makes it easy to define
/// type-safe variadic functions where all arguments have the same
/// type.
///
/// Suppose we need a variadic function like this:
///
///   Result Foo(const Arg &a0, const Arg &a1, ..., const Arg &an);
///
/// Instead of many overloads of Foo(), we only need to define a helper
/// function that takes an array of arguments:
///
///   Result FooImpl(const Arg *const args[], int count) {
///     // 'count' is the number of values in the array; args[i] is a pointer
///     // to the i-th argument passed to Foo().  Therefore, write *args[i]
///     // to access the i-th argument.
///     ...
///   }
///
/// and then define Foo() like this:
///
///   const VariadicFunction<Result, Arg, FooImpl> Foo;
///
/// VariadicFunction takes care of defining the overloads of Foo().
///
/// Actually, Foo is a function object (i.e. functor) instead of a plain
/// function.  This object is stateless and its constructor/destructor
/// does nothing, so it's safe to call Foo(...) at any time.
///
/// Sometimes we need a variadic function to have some fixed leading
/// arguments whose types may be different from that of the optional
/// arguments.  For example:
///
///   bool FullMatch(const StringRef &s, const RE &regex,
///                  const Arg &a0, ..., const Arg &an);
///
/// VariadicFunctionN is for such cases, where N is the number of fixed
/// arguments.  It is like VariadicFunction, except that it takes N more
/// template arguments for the types of the fixed arguments:
///
///   bool FullMatchImpl(const StringRef &s, const RE &regex,
///                      const Arg *const args[], int count) { ... }
///   const VariadicFunction2<bool, const StringRef&,
///                           const RE&, Arg, FullMatchImpl>
///       FullMatch;
///
/// Currently VariadicFunction and friends support up-to 3
/// fixed leading arguments and up-to 32 optional arguments.
template <typename Result, typename Arg,
          Result (*Func)(const Arg *const [], int count)>
class VariadicFunction {
 public:
  VariadicFunction() {}

  Result operator()() const {
    return Func(NULL, 0);
  }

  Result operator()(const Arg &a0) const {
    const Arg *const args[] = { &a0 };
    return Func(args, 1);
  }

  Result operator()(const Arg &a0, const Arg &a1) const {
    const Arg *const args[] = { &a0, &a1 };
    return Func(args, 2);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2) const {
    const Arg *const args[] = { &a0, &a1, &a2 };
    return Func(args, 3);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3 };
    return Func(args, 4);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4 };
    return Func(args, 5);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5 };
    return Func(args, 6);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
    return Func(args, 7);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
    return Func(args, 8);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
    return Func(args, 9);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9 };
    return Func(args, 10);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10 };
    return Func(args, 11);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11 };
    return Func(args, 12);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12 };
    return Func(args, 13);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13 };
    return Func(args, 14);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14 };
    return Func(args, 15);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15 };
    return Func(args, 16);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16 };
    return Func(args, 17);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17 };
    return Func(args, 18);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18 };
    return Func(args, 19);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18, const Arg &a19) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19 };
    return Func(args, 20);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18, const Arg &a19,
      const Arg &a20) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19,
        &a20 };
    return Func(args, 21);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18, const Arg &a19,
      const Arg &a20, const Arg &a21) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21 };
    return Func(args, 22);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18, const Arg &a19,
      const Arg &a20, const Arg &a21, const Arg &a22) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22 };
    return Func(args, 23);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18, const Arg &a19,
      const Arg &a20, const Arg &a21, const Arg &a22, const Arg &a23) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23 };
    return Func(args, 24);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18, const Arg &a19,
      const Arg &a20, const Arg &a21, const Arg &a22, const Arg &a23,
      const Arg &a24) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24 };
    return Func(args, 25);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18, const Arg &a19,
      const Arg &a20, const Arg &a21, const Arg &a22, const Arg &a23,
      const Arg &a24, const Arg &a25) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25 };
    return Func(args, 26);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18, const Arg &a19,
      const Arg &a20, const Arg &a21, const Arg &a22, const Arg &a23,
      const Arg &a24, const Arg &a25, const Arg &a26) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26 };
    return Func(args, 27);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18, const Arg &a19,
      const Arg &a20, const Arg &a21, const Arg &a22, const Arg &a23,
      const Arg &a24, const Arg &a25, const Arg &a26, const Arg &a27) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27 };
    return Func(args, 28);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18, const Arg &a19,
      const Arg &a20, const Arg &a21, const Arg &a22, const Arg &a23,
      const Arg &a24, const Arg &a25, const Arg &a26, const Arg &a27,
      const Arg &a28) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28 };
    return Func(args, 29);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18, const Arg &a19,
      const Arg &a20, const Arg &a21, const Arg &a22, const Arg &a23,
      const Arg &a24, const Arg &a25, const Arg &a26, const Arg &a27,
      const Arg &a28, const Arg &a29) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28, &a29 };
    return Func(args, 30);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18, const Arg &a19,
      const Arg &a20, const Arg &a21, const Arg &a22, const Arg &a23,
      const Arg &a24, const Arg &a25, const Arg &a26, const Arg &a27,
      const Arg &a28, const Arg &a29, const Arg &a30) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28, &a29, &a30 };
    return Func(args, 31);
  }

  Result operator()(const Arg &a0, const Arg &a1, const Arg &a2, const Arg &a3,
      const Arg &a4, const Arg &a5, const Arg &a6, const Arg &a7,
      const Arg &a8, const Arg &a9, const Arg &a10, const Arg &a11,
      const Arg &a12, const Arg &a13, const Arg &a14, const Arg &a15,
      const Arg &a16, const Arg &a17, const Arg &a18, const Arg &a19,
      const Arg &a20, const Arg &a21, const Arg &a22, const Arg &a23,
      const Arg &a24, const Arg &a25, const Arg &a26, const Arg &a27,
      const Arg &a28, const Arg &a29, const Arg &a30, const Arg &a31) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28, &a29, &a30, &a31 };
    return Func(args, 32);
  }
};

template <typename Result, typename Param0, typename Arg,
          Result (*Func)(Param0, const Arg *const [], int count)>
class VariadicFunction1 {
 public:
  VariadicFunction1() {}

  Result operator()(Param0 p0) const {
    return Func(p0, NULL, 0);
  }

  Result operator()(Param0 p0, const Arg &a0) const {
    const Arg *const args[] = { &a0 };
    return Func(p0, args, 1);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1) const {
    const Arg *const args[] = { &a0, &a1 };
    return Func(p0, args, 2);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1,
      const Arg &a2) const {
    const Arg *const args[] = { &a0, &a1, &a2 };
    return Func(p0, args, 3);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3 };
    return Func(p0, args, 4);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4 };
    return Func(p0, args, 5);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5 };
    return Func(p0, args, 6);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
    return Func(p0, args, 7);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
    return Func(p0, args, 8);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
    return Func(p0, args, 9);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9 };
    return Func(p0, args, 10);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10 };
    return Func(p0, args, 11);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11 };
    return Func(p0, args, 12);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12 };
    return Func(p0, args, 13);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13 };
    return Func(p0, args, 14);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14 };
    return Func(p0, args, 15);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15 };
    return Func(p0, args, 16);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16 };
    return Func(p0, args, 17);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17 };
    return Func(p0, args, 18);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18 };
    return Func(p0, args, 19);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18,
      const Arg &a19) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19 };
    return Func(p0, args, 20);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18,
      const Arg &a19, const Arg &a20) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19,
        &a20 };
    return Func(p0, args, 21);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18,
      const Arg &a19, const Arg &a20, const Arg &a21) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21 };
    return Func(p0, args, 22);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18,
      const Arg &a19, const Arg &a20, const Arg &a21, const Arg &a22) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22 };
    return Func(p0, args, 23);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18,
      const Arg &a19, const Arg &a20, const Arg &a21, const Arg &a22,
      const Arg &a23) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23 };
    return Func(p0, args, 24);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18,
      const Arg &a19, const Arg &a20, const Arg &a21, const Arg &a22,
      const Arg &a23, const Arg &a24) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24 };
    return Func(p0, args, 25);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18,
      const Arg &a19, const Arg &a20, const Arg &a21, const Arg &a22,
      const Arg &a23, const Arg &a24, const Arg &a25) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25 };
    return Func(p0, args, 26);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18,
      const Arg &a19, const Arg &a20, const Arg &a21, const Arg &a22,
      const Arg &a23, const Arg &a24, const Arg &a25, const Arg &a26) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26 };
    return Func(p0, args, 27);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18,
      const Arg &a19, const Arg &a20, const Arg &a21, const Arg &a22,
      const Arg &a23, const Arg &a24, const Arg &a25, const Arg &a26,
      const Arg &a27) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27 };
    return Func(p0, args, 28);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18,
      const Arg &a19, const Arg &a20, const Arg &a21, const Arg &a22,
      const Arg &a23, const Arg &a24, const Arg &a25, const Arg &a26,
      const Arg &a27, const Arg &a28) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28 };
    return Func(p0, args, 29);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18,
      const Arg &a19, const Arg &a20, const Arg &a21, const Arg &a22,
      const Arg &a23, const Arg &a24, const Arg &a25, const Arg &a26,
      const Arg &a27, const Arg &a28, const Arg &a29) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28, &a29 };
    return Func(p0, args, 30);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18,
      const Arg &a19, const Arg &a20, const Arg &a21, const Arg &a22,
      const Arg &a23, const Arg &a24, const Arg &a25, const Arg &a26,
      const Arg &a27, const Arg &a28, const Arg &a29, const Arg &a30) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28, &a29, &a30 };
    return Func(p0, args, 31);
  }

  Result operator()(Param0 p0, const Arg &a0, const Arg &a1, const Arg &a2,
      const Arg &a3, const Arg &a4, const Arg &a5, const Arg &a6,
      const Arg &a7, const Arg &a8, const Arg &a9, const Arg &a10,
      const Arg &a11, const Arg &a12, const Arg &a13, const Arg &a14,
      const Arg &a15, const Arg &a16, const Arg &a17, const Arg &a18,
      const Arg &a19, const Arg &a20, const Arg &a21, const Arg &a22,
      const Arg &a23, const Arg &a24, const Arg &a25, const Arg &a26,
      const Arg &a27, const Arg &a28, const Arg &a29, const Arg &a30,
      const Arg &a31) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28, &a29, &a30, &a31 };
    return Func(p0, args, 32);
  }
};

template <typename Result, typename Param0, typename Param1, typename Arg,
          Result (*Func)(Param0, Param1, const Arg *const [], int count)>
class VariadicFunction2 {
 public:
  VariadicFunction2() {}

  Result operator()(Param0 p0, Param1 p1) const {
    return Func(p0, p1, NULL, 0);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0) const {
    const Arg *const args[] = { &a0 };
    return Func(p0, p1, args, 1);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1) const {
    const Arg *const args[] = { &a0, &a1 };
    return Func(p0, p1, args, 2);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2) const {
    const Arg *const args[] = { &a0, &a1, &a2 };
    return Func(p0, p1, args, 3);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3 };
    return Func(p0, p1, args, 4);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4 };
    return Func(p0, p1, args, 5);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5 };
    return Func(p0, p1, args, 6);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
    return Func(p0, p1, args, 7);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
    return Func(p0, p1, args, 8);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
    return Func(p0, p1, args, 9);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9 };
    return Func(p0, p1, args, 10);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10 };
    return Func(p0, p1, args, 11);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11 };
    return Func(p0, p1, args, 12);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12 };
    return Func(p0, p1, args, 13);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13 };
    return Func(p0, p1, args, 14);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14 };
    return Func(p0, p1, args, 15);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15 };
    return Func(p0, p1, args, 16);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16 };
    return Func(p0, p1, args, 17);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17 };
    return Func(p0, p1, args, 18);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18 };
    return Func(p0, p1, args, 19);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18, const Arg &a19) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19 };
    return Func(p0, p1, args, 20);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18, const Arg &a19, const Arg &a20) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19,
        &a20 };
    return Func(p0, p1, args, 21);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18, const Arg &a19, const Arg &a20, const Arg &a21) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21 };
    return Func(p0, p1, args, 22);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18, const Arg &a19, const Arg &a20, const Arg &a21,
      const Arg &a22) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22 };
    return Func(p0, p1, args, 23);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18, const Arg &a19, const Arg &a20, const Arg &a21,
      const Arg &a22, const Arg &a23) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23 };
    return Func(p0, p1, args, 24);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18, const Arg &a19, const Arg &a20, const Arg &a21,
      const Arg &a22, const Arg &a23, const Arg &a24) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24 };
    return Func(p0, p1, args, 25);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18, const Arg &a19, const Arg &a20, const Arg &a21,
      const Arg &a22, const Arg &a23, const Arg &a24, const Arg &a25) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25 };
    return Func(p0, p1, args, 26);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18, const Arg &a19, const Arg &a20, const Arg &a21,
      const Arg &a22, const Arg &a23, const Arg &a24, const Arg &a25,
      const Arg &a26) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26 };
    return Func(p0, p1, args, 27);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18, const Arg &a19, const Arg &a20, const Arg &a21,
      const Arg &a22, const Arg &a23, const Arg &a24, const Arg &a25,
      const Arg &a26, const Arg &a27) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27 };
    return Func(p0, p1, args, 28);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18, const Arg &a19, const Arg &a20, const Arg &a21,
      const Arg &a22, const Arg &a23, const Arg &a24, const Arg &a25,
      const Arg &a26, const Arg &a27, const Arg &a28) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28 };
    return Func(p0, p1, args, 29);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18, const Arg &a19, const Arg &a20, const Arg &a21,
      const Arg &a22, const Arg &a23, const Arg &a24, const Arg &a25,
      const Arg &a26, const Arg &a27, const Arg &a28, const Arg &a29) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28, &a29 };
    return Func(p0, p1, args, 30);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18, const Arg &a19, const Arg &a20, const Arg &a21,
      const Arg &a22, const Arg &a23, const Arg &a24, const Arg &a25,
      const Arg &a26, const Arg &a27, const Arg &a28, const Arg &a29,
      const Arg &a30) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28, &a29, &a30 };
    return Func(p0, p1, args, 31);
  }

  Result operator()(Param0 p0, Param1 p1, const Arg &a0, const Arg &a1,
      const Arg &a2, const Arg &a3, const Arg &a4, const Arg &a5,
      const Arg &a6, const Arg &a7, const Arg &a8, const Arg &a9,
      const Arg &a10, const Arg &a11, const Arg &a12, const Arg &a13,
      const Arg &a14, const Arg &a15, const Arg &a16, const Arg &a17,
      const Arg &a18, const Arg &a19, const Arg &a20, const Arg &a21,
      const Arg &a22, const Arg &a23, const Arg &a24, const Arg &a25,
      const Arg &a26, const Arg &a27, const Arg &a28, const Arg &a29,
      const Arg &a30, const Arg &a31) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28, &a29, &a30, &a31 };
    return Func(p0, p1, args, 32);
  }
};

template <typename Result, typename Param0, typename Param1, typename Param2,
    typename Arg,
          Result (*Func)(Param0, Param1, Param2, const Arg *const [],
              int count)>
class VariadicFunction3 {
 public:
  VariadicFunction3() {}

  Result operator()(Param0 p0, Param1 p1, Param2 p2) const {
    return Func(p0, p1, p2, NULL, 0);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0) const {
    const Arg *const args[] = { &a0 };
    return Func(p0, p1, p2, args, 1);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1) const {
    const Arg *const args[] = { &a0, &a1 };
    return Func(p0, p1, p2, args, 2);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2) const {
    const Arg *const args[] = { &a0, &a1, &a2 };
    return Func(p0, p1, p2, args, 3);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3 };
    return Func(p0, p1, p2, args, 4);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4 };
    return Func(p0, p1, p2, args, 5);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5 };
    return Func(p0, p1, p2, args, 6);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6 };
    return Func(p0, p1, p2, args, 7);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
    return Func(p0, p1, p2, args, 8);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
    return Func(p0, p1, p2, args, 9);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9 };
    return Func(p0, p1, p2, args, 10);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10 };
    return Func(p0, p1, p2, args, 11);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11 };
    return Func(p0, p1, p2, args, 12);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12 };
    return Func(p0, p1, p2, args, 13);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13 };
    return Func(p0, p1, p2, args, 14);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14 };
    return Func(p0, p1, p2, args, 15);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15 };
    return Func(p0, p1, p2, args, 16);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16 };
    return Func(p0, p1, p2, args, 17);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17 };
    return Func(p0, p1, p2, args, 18);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18 };
    return Func(p0, p1, p2, args, 19);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18, const Arg &a19) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19 };
    return Func(p0, p1, p2, args, 20);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18, const Arg &a19, const Arg &a20) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19,
        &a20 };
    return Func(p0, p1, p2, args, 21);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18, const Arg &a19, const Arg &a20,
      const Arg &a21) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21 };
    return Func(p0, p1, p2, args, 22);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18, const Arg &a19, const Arg &a20,
      const Arg &a21, const Arg &a22) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22 };
    return Func(p0, p1, p2, args, 23);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18, const Arg &a19, const Arg &a20,
      const Arg &a21, const Arg &a22, const Arg &a23) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23 };
    return Func(p0, p1, p2, args, 24);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18, const Arg &a19, const Arg &a20,
      const Arg &a21, const Arg &a22, const Arg &a23, const Arg &a24) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24 };
    return Func(p0, p1, p2, args, 25);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18, const Arg &a19, const Arg &a20,
      const Arg &a21, const Arg &a22, const Arg &a23, const Arg &a24,
      const Arg &a25) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25 };
    return Func(p0, p1, p2, args, 26);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18, const Arg &a19, const Arg &a20,
      const Arg &a21, const Arg &a22, const Arg &a23, const Arg &a24,
      const Arg &a25, const Arg &a26) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26 };
    return Func(p0, p1, p2, args, 27);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18, const Arg &a19, const Arg &a20,
      const Arg &a21, const Arg &a22, const Arg &a23, const Arg &a24,
      const Arg &a25, const Arg &a26, const Arg &a27) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27 };
    return Func(p0, p1, p2, args, 28);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18, const Arg &a19, const Arg &a20,
      const Arg &a21, const Arg &a22, const Arg &a23, const Arg &a24,
      const Arg &a25, const Arg &a26, const Arg &a27, const Arg &a28) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28 };
    return Func(p0, p1, p2, args, 29);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18, const Arg &a19, const Arg &a20,
      const Arg &a21, const Arg &a22, const Arg &a23, const Arg &a24,
      const Arg &a25, const Arg &a26, const Arg &a27, const Arg &a28,
      const Arg &a29) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28, &a29 };
    return Func(p0, p1, p2, args, 30);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18, const Arg &a19, const Arg &a20,
      const Arg &a21, const Arg &a22, const Arg &a23, const Arg &a24,
      const Arg &a25, const Arg &a26, const Arg &a27, const Arg &a28,
      const Arg &a29, const Arg &a30) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28, &a29, &a30 };
    return Func(p0, p1, p2, args, 31);
  }

  Result operator()(Param0 p0, Param1 p1, Param2 p2, const Arg &a0,
      const Arg &a1, const Arg &a2, const Arg &a3, const Arg &a4,
      const Arg &a5, const Arg &a6, const Arg &a7, const Arg &a8,
      const Arg &a9, const Arg &a10, const Arg &a11, const Arg &a12,
      const Arg &a13, const Arg &a14, const Arg &a15, const Arg &a16,
      const Arg &a17, const Arg &a18, const Arg &a19, const Arg &a20,
      const Arg &a21, const Arg &a22, const Arg &a23, const Arg &a24,
      const Arg &a25, const Arg &a26, const Arg &a27, const Arg &a28,
      const Arg &a29, const Arg &a30, const Arg &a31) const {
    const Arg *const args[] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8,
        &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17, &a18, &a19, &a20,
        &a21, &a22, &a23, &a24, &a25, &a26, &a27, &a28, &a29, &a30, &a31 };
    return Func(p0, p1, p2, args, 32);
  }
};

} // end namespace internal
} // end namespace tooling
} // end namespace clang

#endif  // LLVM_CLANG_TOOLING_VARIADIC_FUNCTION_H
