// RUN: %clang_cc1 -emit-llvm %s -o -

#ifdef PACKED
#define P __attribute__((packed))
#else
#define P
#endif

struct UnPacked {
 	int X;
	int Y;
};

struct P M_Packed {
  unsigned char A;
  struct UnPacked B;
};

struct M_Packed sM_Packed;

int testM_Packed (void) {
  struct M_Packed x;
  return (x.B.Y != 0);
}
