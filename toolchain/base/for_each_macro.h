// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_FOR_EACH_MACRO_H_
#define CARBON_TOOLCHAIN_BASE_FOR_EACH_MACRO_H_

/// CARBON_FOR_EACH() will apply `macro` to each argument in the variadic
/// argument list, putting the output of `sep()` between each one.
///
/// The `sep` should be a function macro that returns a separator. Premade
/// separataors are provided as CARBON_FOR_EACH_XYZ() macros.
#define CARBON_FOR_EACH(macro, sep, ...)      \
  __VA_OPT__(CARBON_INTERNAL_FOR_EACH_EXPAND( \
      CARBON_INTERNAL_FOR_EACH(macro, sep, __VA_ARGS__)))

#define CARBON_FOR_EACH_COMMA() ,
#define CARBON_FOR_EACH_SEMI() ;
#define CARBON_FOR_EACH_CONCAT()

// Internal helpers

#define CARBON_INTERNAL_FOR_EACH(macro, sep, a1, ...)                 \
  macro(a1) __VA_OPT__(sep()) __VA_OPT__(                             \
      CARBON_INTERNAL_FOR_EACH_AGAIN CARBON_INTERNAL_FOR_EACH_PARENS( \
          macro, sep, __VA_ARGS__))
#define CARBON_INTERNAL_FOR_EACH_PARENS ()
#define CARBON_INTERNAL_FOR_EACH_AGAIN() CARBON_INTERNAL_FOR_EACH

#define CARBON_INTERNAL_FOR_EACH_EXPAND(...)                             \
  CARBON_INTERNAL_FOR_EACH_EXPAND1(                                      \
      CARBON_INTERNAL_FOR_EACH_EXPAND1(CARBON_INTERNAL_FOR_EACH_EXPAND1( \
          CARBON_INTERNAL_FOR_EACH_EXPAND1(__VA_ARGS__))))
#define CARBON_INTERNAL_FOR_EACH_EXPAND1(...)                            \
  CARBON_INTERNAL_FOR_EACH_EXPAND2(                                      \
      CARBON_INTERNAL_FOR_EACH_EXPAND2(CARBON_INTERNAL_FOR_EACH_EXPAND2( \
          CARBON_INTERNAL_FOR_EACH_EXPAND2(__VA_ARGS__))))
#define CARBON_INTERNAL_FOR_EACH_EXPAND2(...)                            \
  CARBON_INTERNAL_FOR_EACH_EXPAND3(                                      \
      CARBON_INTERNAL_FOR_EACH_EXPAND3(CARBON_INTERNAL_FOR_EACH_EXPAND3( \
          CARBON_INTERNAL_FOR_EACH_EXPAND3(__VA_ARGS__))))
#define CARBON_INTERNAL_FOR_EACH_EXPAND3(...) __VA_ARGS__

#endif  // CARBON_TOOLCHAIN_BASE_FOR_EACH_MACRO_H_
