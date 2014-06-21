// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fms-compatibility -verify %s
// expected-no-diagnostics

static_assert(sizeof(0i8  ) == sizeof(__INT8_TYPE__ ), "");
static_assert(sizeof(0i16 ) == sizeof(__INT16_TYPE__), "");
static_assert(sizeof(0i32 ) == sizeof(__INT32_TYPE__), "");
static_assert(sizeof(0i64 ) == sizeof(__INT64_TYPE__), "");
static_assert(sizeof(0i128) >  sizeof(__INT64_TYPE__), "");
