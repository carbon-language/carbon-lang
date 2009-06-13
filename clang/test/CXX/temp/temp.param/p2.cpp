// RUN: clang-cc -fsyntax-only -verify %s

// There is no semantic difference between class and typename in a
// template-parameter. typename followed by an unqualified-id names a
// template type parameter.
template<class T> struct X;
template<typename T> struct X;

// typename followed by aqualified-id denotes the type in a non-type
// parameter-declaration.
// FIXME: template<typename T, typename T::type Value> struct Y;

// A storage class shall not be specified in a template-parameter declaration.
template<static int Value> struct Z; // FIXME: expect an error

// FIXME: add the example from p2
