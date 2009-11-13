// RUN: clang-cc -analyze -analyzer-experimental-internal-checks -warn-dead-stores -verify %s
// RUN: clang-cc -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=basic -analyzer-constraints=basic -warn-dead-stores -verify %s
// RUN: clang-cc -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=basic -analyzer-constraints=range -warn-dead-stores -verify %s
// RUN: clang-cc -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -analyzer-constraints=basic -warn-dead-stores -verify %s
// RUN: clang-cc -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -analyzer-constraints=range -warn-dead-stores -verify %s

int j;
void f1() {
  int x = 4;

  ++x; // expected-warning{{never read}}

  switch (j) {
  case 1:
    throw 1;
    (void)x;
    break;
  }
}
