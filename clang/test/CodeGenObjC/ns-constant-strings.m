// RUN: %clang_cc1 -triple i386-apple-darwin9 -fno-constant-cfstrings -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix CHECK-FRAGILE < %t %s

// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fno-constant-cfstrings -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix CHECK-NONFRAGILE < %t %s

@interface NSString @end

@interface NSSimpleCString : NSString {
@protected
    char *bytes;
    unsigned int numBytes;
}
@end
    
@interface NSConstantString : NSSimpleCString
@end

#if OBJC_API_VERSION >= 2
extern Class _NSConstantStringClassReference;
#else
extern struct objc_class _NSConstantStringClassReference;
#endif

const NSConstantString *appKey =  @"MyApp";

int main() {
  const NSConstantString *appKey =  @"MyApp";
  const NSConstantString *appKey1 =  @"MyApp1";
}

// CHECK-FRAGILE: @_NSConstantStringClassReference = external global
// CHECK-NONFRAGILE: @"OBJC_CLASS_$_NSConstantString" = external global
