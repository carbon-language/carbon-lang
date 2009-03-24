// RUN: clang-cc -fnext-runtime --emit-llvm -o %t %s

@interface A
@end
@implementation A
+(void) foo {
  printf("__func__: %s\n", __func__);
  printf("__FUNCTION__: %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__: %s\n", __PRETTY_FUNCTION__);
  return 0;
}
@end

int main() {
  [A foo];
  return 0;
}
