// RUN: %clang_cc1 -verify -Warray-bounds-pointer-arithmetic %s

// Test case from PR10615
struct ext2_super_block{
  unsigned char s_uuid[8]; // expected-note {{declared here}}
};
void* ext2_statfs (struct ext2_super_block *es,int a)
{
	 return (void *)es->s_uuid + sizeof(int); // no-warning
}
void* broken (struct ext2_super_block *es,int a)
{
	 return (void *)es->s_uuid + 80; // expected-warning {{refers past the end of the array}}
}

// Test case reduced from PR11594
struct S { int n; };
void pr11594(struct S *s) {
  int a[10];
  int *p = a - s->n;
}

// Test case reduced from <rdar://problem/11387038>.  This resulted in
// an assertion failure because of the typedef instead of an explicit
// constant array type.
struct RDar11387038 {};
typedef struct RDar11387038 RDar11387038Array[1];
struct RDar11387038_Table {
  RDar11387038Array z;
};
typedef struct RDar11387038_Table * TPtr;
typedef TPtr *TabHandle;
struct RDar11387038_B { TabHandle x; };
typedef struct RDar11387038_B RDar11387038_B;

void radar11387038(void) {
  RDar11387038_B *pRDar11387038_B;
  struct RDar11387038* y = &(*pRDar11387038_B->x)->z[4];
}

void pr51682 (void) {
  int arr [1];
  switch (0) {
    case 0:
      break;
    case 1:
      asm goto (""::"r"(arr[42] >> 1)::failed); // no-warning
      break;
  }
failed:;
}
