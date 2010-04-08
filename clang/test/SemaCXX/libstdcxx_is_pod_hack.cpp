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
