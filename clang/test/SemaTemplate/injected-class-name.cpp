// RUN: clang-cc -fsyntax-only -verify %s

template<typename T>
struct X {
  X<T*> *ptr;
};

X<int> x;

template<>
struct X<int***> {
  typedef X<int***> *ptr;
};

// FIXME: EDG rejects this in their strict-conformance mode, but I
// don't see any wording making this ill-formed.
X<float>::X<int> xi = x;
