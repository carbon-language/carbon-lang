// RUN: %clang -std=c++11 -o %t.o -c -no-integrated-as -fprofile-instr-generate %s

__attribute__((noinline)) static int bar() {
    return 1;
}

int foo(int a, int b)
{
    auto Func = [](int a, int b) { return a > b; };

    return Func(a,b) + bar();
}

