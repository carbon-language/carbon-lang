// RUN: llvm-cov report %S/Inputs/report.covmapping -instr-profile %S/Inputs/report.profdata -no-colors 2>&1 | FileCheck %s

// CHECK: Filename                    Regions    Miss   Cover Functions  Executed
// CHECK: TOTAL                             5       2  60.00%         4    75.00%

void foo(bool cond) {
  if (cond) {
  }
}

void bar() {
}

void func() {
}

int main() {
  foo(false);
  bar();
  return 0;
}

// llvm-cov doesn't work on big endian yet
// XFAIL: powerpc64-, s390x, mips-, mips64-, sparc
