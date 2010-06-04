// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

// Check for template type parameter pack (mis-)matches with template
// type parameters.
template<typename ...T> struct X0t;
template<typename ...T> struct X0t;

template<typename ...T> struct X1t; // expected-note{{previous template type parameter pack declared here}}
template<typename T> struct X1t; // expected-error{{template type parameter conflicts with previous template type parameter pack}}

template<typename T> struct X2t; // expected-note{{previous template type parameter declared here}}
template<typename ...T> struct X2t; // expected-error{{template type parameter pack conflicts with previous template type parameter}}

template<template<typename ...T> class> struct X0tt; 
template<template<typename ...T> class> struct X0tt; 

template<template<typename ...T> class> struct X1tt; // expected-note{{previous template type parameter pack declared here}}
template<template<typename T> class> struct X1tt; // expected-error{{template type parameter conflicts with previous template type parameter pack}}

template<template<typename T> class> struct X2tt; // expected-note{{previous template type parameter declared here}}
template<template<typename ...T> class> struct X2tt; // expected-error{{template type parameter pack conflicts with previous template type parameter}}
