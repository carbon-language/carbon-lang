// RUN: clang-cc -fsyntax-only %s

// FIXME: The Fibonacci/FibonacciEval dance is here to work around our
// inability to parse injected-class-name<template-argument-list>.
template<unsigned I>
struct FibonacciEval;

template<unsigned I>
struct Fibonacci {
  enum { value = FibonacciEval<I-1>::value + FibonacciEval<I-2>::value };
};

template<unsigned I>
struct FibonacciEval {
  enum { value = Fibonacci<I>::value };
};

template<> struct Fibonacci<0> {
  enum { value = 0 };
};

template<> struct Fibonacci<1> {
  enum { value = 1 };
};

int array5[Fibonacci<5>::value == 5? 1 : -1];
int array10[Fibonacci<10>::value == 55? 1 : -1];

template<unsigned I>
struct FibonacciEval2;

template<unsigned I>
struct Fibonacci2 {
  static const unsigned value 
    = FibonacciEval2<I-1>::value + FibonacciEval2<I-2>::value;
};

template<unsigned I>
struct FibonacciEval2 {
  static const unsigned value = Fibonacci2<I>::value;
};

template<> struct Fibonacci2<0> {
  static const unsigned value = 0;
};

template<> struct Fibonacci2<1> {
  static const unsigned value = 1;
};

int array5_2[Fibonacci2<5>::value == 5? 1 : -1];
int array10_2[Fibonacci2<10>::value == 55? 1 : -1];
