// RUN: ignore %llvmgcc -S %s -o - /dev/null |& not grep warning
// XFAIL: *

#define ATTR_BITS(N) __attribute__((bitwidth(N))) 

typedef int ATTR_BITS( 4) My04BitInt;
typedef int ATTR_BITS(16) My16BitInt;
typedef int ATTR_BITS(17) My17BitInt;
typedef int ATTR_BITS(37) My37BitInt;
typedef int ATTR_BITS(65) My65BitInt;

struct MyStruct {
  My04BitInt i4Field;
  short ATTR_BITS(12) i12Field;
  long ATTR_BITS(17) i17Field;
  My37BitInt i37Field;
};

My37BitInt doit( short ATTR_BITS(23) num) {
  My17BitInt i;
  struct MyStruct strct;
  int bitsize1 = sizeof(My17BitInt);
  int __attribute__((bitwidth(9))) j;
  int bitsize2 = sizeof(j);
  int result = bitsize1 + bitsize2;
  strct.i17Field = result;
  result += sizeof(struct MyStruct);
  return result;
}

int
main ( int argc, char** argv)
{
  return (int ATTR_BITS(32)) doit((short ATTR_BITS(23))argc);
}
