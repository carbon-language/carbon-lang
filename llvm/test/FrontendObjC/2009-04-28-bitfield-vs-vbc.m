// RUN: %llvmgcc -S -x objective-c -m32 %s -o %t
// This used to crash, 6831493.
#include <stdlib.h>

struct s0 {
  double x;
};

@interface I2 {
  struct s0 _iv1;
}
@end

@interface I3 : I2 {
  unsigned int _iv2 :1;
  unsigned : 0;
  unsigned int _iv3 : 3;
}
@end

@interface I4 : I3 {
  char _iv4;
}
@end

@interface I5 : I4 {
  char _iv5;
  int _iv6;
  int _iv7;
}

@property int P1;
@end

@implementation I2
@end

@implementation I3
@end

@implementation I4 
@end

@interface I5 ()
@property int P2;
@end

#if 0
int g2 = sizeof(I2);
int g3 = sizeof(I3);
int g4 = sizeof(I4);
int g5_0 = sizeof(I5);
#endif

@implementation I5
#ifdef __x86_64
@synthesize P1 = _MadeUpName;
@synthesize P2 = _AnotherMadeUpName;
#else
@synthesize P1 = _iv6;
@synthesize P2 = _iv7;
#endif
@end

#if 0
int g5_1 = sizeof(I5);
#endif

@interface T0_I0 {
  double iv_A_0;
  char iv_A_1;
}
@end

@interface T0_I1 : T0_I0 {
  char iv_B_0;
}
@end

@interface T0_I2 : T0_I1 {
  char iv_C_0;
}
@end

#if 0
int g6 = sizeof(T0_I0);
int g7 = sizeof(T0_I1);
int g8 = sizeof(T0_I2);
#endif
  
@implementation T0_I0 @end
@implementation T0_I1 @end  
@implementation T0_I2 @end

void f0(I2*i2,I3*i3,I4*i4,I5*i5,T0_I0*t0_i0,T0_I1*t0_i1,T0_I2*t0_i2) {
}

// Thomas Wang's ui32 hash.
unsigned hash_ui32_to_ui32(unsigned a) {
  a = (a ^ 61) ^ (a >> 16);
  a = a + (a << 3);
  a = a ^ (a >> 4);
  a = a * 0x27d4eb2d;
  a = a ^ (a >> 15);
  return a;
}

unsigned char hash_ui32_to_ui8(unsigned ui) {
  ui = hash_ui32_to_ui32(ui);
  ui ^= ui>>8;
  ui ^= ui>>8;
  ui ^= ui>>8;
  return (unsigned char) ui;
}

void *init() {
  unsigned i, N = 1024;
  unsigned char *p = malloc(N);
  for (i=0; i != N; ++i)
    p[i] = hash_ui32_to_ui8(i);
  return p;
}

int main(){
  void *p = init();
  f0(p,p,p,p,p,p,p);
}
