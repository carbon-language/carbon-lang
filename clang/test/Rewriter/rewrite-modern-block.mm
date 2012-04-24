// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar://11230308

typedef struct {
    char byte0;
    char byte1;
} CFUUIDBytes;

void x(void *);

void y() {
    __block CFUUIDBytes bytes;
    
    void (^bar)() = ^{
        x(&bytes);
    };
}

// rdar://11236342
int foo() {
    __block int hello;
}

// rdar://7547630
// rewriting multiple __block decls on wintin same decl stmt.
int radar7547630() {
  __block int BI1, BI2;

  __block float FLOAT1, FT2, FFFFFFFF3,
   FFFXXX4;

  __block void (^B)(), (^BB)();
}

