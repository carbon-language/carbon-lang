// RUN: clang-cc -E %s | grep 'a:Y' &&
// RUN: clang-cc -E %s | grep 'b:Y' &&
// RUN: clang-cc -E %s | grep 'c:YY'

#define FOO(X) X ## Y
a:FOO()

#define FOO2(X) Y ## X
b:FOO2()

#define FOO3(X) X ## Y ## X ## Y ## X ## X
c:FOO3()

