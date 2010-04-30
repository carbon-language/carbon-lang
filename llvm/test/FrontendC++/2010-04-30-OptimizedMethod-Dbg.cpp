// RUN: %llvmgcc -g -S -O2 %s -o %t
// RUN: grep "i1 false, i1 true. . . DW_TAG_subprogram" %t | count 2

class foo {
public:
      int bar(int x);
      static int baz(int x);
};

int foo::bar(int x) {
    return x*4 + 1;
}

int foo::baz(int x) {
    return x*4 + 1;
}

