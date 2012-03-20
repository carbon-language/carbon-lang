// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -fblocks -rewrite-objc -fms-extensions %t.mm -o %t-rw.cpp 
// RUN: FileCheck --input-file=%t-rw.cpp %s
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

struct S {
    int i1;
    double d1;
    void (^block1)();
};

@interface I
{
  struct S struct_ivar;

  struct S *pstruct_ivar;
}
@end

@implementation I
- (struct S) dMeth{ return struct_ivar; }
@end
  
// CHECK: return (*(struct S *)((char *)self + OBJC_IVAR_$_I$struct_ivar));
