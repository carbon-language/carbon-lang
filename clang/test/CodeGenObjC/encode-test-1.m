// RUN: clang-cc -triple=i686-apple-darwin9 -emit-llvm -o %t %s
// RUN: grep -e "{Base=b2b3b4b5}" %t | count 1
// RUN: grep -e "{Derived=b2b3b4b5b5b4b3}" %t | count 1

enum Enum { one, two, three, four };

@interface Base {
  unsigned a: 2;
  int b: 3;
  enum Enum c: 4;
  unsigned d: 5;
} 
@end

@interface Derived: Base {
  signed e: 5;
  int f: 4;
  enum Enum g: 3;
} 
@end

@implementation Base @end

@implementation Derived @end
  
int main(void)
{

  const char *en = @encode(Base);
//  printf ("%s\n", en);

  const char *ed = @encode(Derived);
 // printf ("%s\n", ed);

  return 0;
}
