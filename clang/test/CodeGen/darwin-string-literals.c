// RUN: clang-cc -triple i386-apple-darwin9 -emit-llvm %s -o %t &&

// RUN: grep -F '@"\01LC" = internal constant [8 x i8] c"string0\00"' %t &&
// RUN: grep -F '@"\01LC1" = internal constant [8 x i8] c"string1\00", section "__TEXT,__cstring,cstring_literals"' %t &&

// RUN: true

const char *g0 = "string0";
const void *g1 = __builtin___CFStringMakeConstantString("string1");
