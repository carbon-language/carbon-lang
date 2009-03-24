// RUN: clang-cc -triple x86_64-unknown-unknown -fobjc-gc-only -emit-llvm -o %t %s

@interface I0 {
  I0 *_f0;
}
@property (retain) id p0;
@end 

@implementation I0 
  @synthesize p0 = _f0;
@end

