/* Testcase for when struct tag conflicts with typedef name... grr */

typedef struct foo {
  struct foo *X;
  int Y;
} * foo;

foo F;

