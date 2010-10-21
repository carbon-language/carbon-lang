// Suppress 'no run line' failure.
// RUN: %clang_cc1 -fsyntax-only -verify %s

template<template<> class C> class D; // expected-error{{template template parameter must have its own template parameters}}


