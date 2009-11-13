// RUN: clang-cc -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=basic -verify %s
// RUN: clang-cc -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -verify %s 

// This is a test case for the issue reported in PR 2819:
//  http://llvm.org/bugs/show_bug.cgi?id=2819
// The flow-sensitive dataflow solver should work even when no block in
// the CFG reaches the exit block.

int g(int x);
void h(int x);

int f(int x)
{
out_err:
  if (g(x)) {
    h(x);
  }
  goto out_err;
}
