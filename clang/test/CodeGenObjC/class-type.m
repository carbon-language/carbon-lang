// RUN: clang-cc -triple x86_64-unknown-unknown -emit-llvm -o - %s &&
// RUN: clang-cc -triple i386-apple-darwin9 -emit-llvm -o - %s &&
// RUN: clang-cc -triple x86_64-apple-darwin9 -emit-llvm -o - %s


@interface I0 {
  struct { int a; } a;
}
@end 

@class I2;

@interface I1 {
  I2 *_imageBrowser;
}
@end 

@implementation I1 
@end 

@interface I2 : I0 
@end 

@implementation I2 
@end 


// Implementations without interface declarations.
// rdar://6804402
@class foo;
@implementation foo 
@end

@implementation bar
@end

