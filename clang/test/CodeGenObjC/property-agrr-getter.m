// RUN: clang-cc -fnext-runtime -emit-llvm -o %t %s

typedef struct {
  unsigned f0;
} s0;

@interface A
- (s0) f0;
@end

@implementation A
-(s0) f0{}
- (unsigned) bar {
  return self.f0.f0;
}
@end

