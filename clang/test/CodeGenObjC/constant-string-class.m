// RUN: %clang_cc1 -triple i386-apple-darwin9 -fno-constant-cfstrings -fconstant-string-class Foo -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix CHECK-FRAGILE < %t %s

// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fno-constant-cfstrings -fconstant-string-class Foo -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix CHECK-NONFRAGILE < %t %s

// rdar: // 8564463
// PR6056

@interface Object {
  id isa;
}
@end

@interface Foo : Object{
  char *cString;
  unsigned int len;
}
- (char *)customString;
@end

id _FooClassReference[20];

@implementation Foo 
- (char *)customString { return cString ; }
@end

int main () {
  Foo *string = @"bla";
  return 0;
}

// CHECK-FRAGILE: @_FooClassReference = common global
// CHECK-NONFRAGILE: @"OBJC_CLASS_$_Object" = external global
