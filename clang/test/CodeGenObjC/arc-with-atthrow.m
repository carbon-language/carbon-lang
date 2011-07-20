// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fobjc-nonfragile-abi -fobjc-exceptions -o - %s | FileCheck %s
// $CLANG -cc1 -fobjc-arc -fobjc-nonfragile-abi -fobjc-exceptions
// pr10411

@interface NSException
+ (id)exception;
@end

void test() 
{ 
  @throw [NSException exception]; 
}

// CHECK: objc_retainAutoreleasedReturnValue
// CHECK:   call void @objc_release
// CHECK:   call void @objc_exception_throw
