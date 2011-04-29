// RUN: %clang_cc1 -fsyntax-only %s

// This is a test for an egregious hack in Clang that works around
// issues with GCC's evolution. libstdc++ 4.2.x uses __is_pod as an
// identifier (to declare a struct template like the one below), while
// GCC 4.3 and newer make __is_pod a keyword. Clang treats __is_pod as
// a keyword *unless* it is introduced following the struct keyword.

template<typename T>
struct __is_pod {
};

__is_pod<int> ipi;

// Ditto for __is_same.
template<typename T>
struct __is_same {
};

__is_same<int> ipi;

// Another, similar egregious hack for __is_signed, which is a type
// trait in Embarcadero's compiler but is used as an identifier in
// libstdc++.
struct test_is_signed {
  static const bool __is_signed = true;
};

bool check_signed = test_is_signed::__is_signed;
