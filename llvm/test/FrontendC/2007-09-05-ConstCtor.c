// RUN: %llvmgcc -xc -Os -c %s -o /dev/null
// PR1641
// XFAIL: *
// See PR2452

struct A {
  unsigned long l;
};

void bar(struct A *a);

void bork() {
  const unsigned long vcgt = 'vcgt';
  struct A a = { vcgt };
  bar(&a);
}
