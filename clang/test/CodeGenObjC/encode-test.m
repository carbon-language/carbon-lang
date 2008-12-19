// RUN: clang -fnext-runtime -emit-llvm -o %t %s &&
// RUN: grep -e "\^{Innermost=CC}" %t | count 1

@class Int1;

struct Innermost {
  unsigned char a, b;
};

@interface Int1 {
  signed char a, b;
  struct Innermost *innermost;
}
@end

@implementation Int1
@end
