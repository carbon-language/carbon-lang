// Test that no data is collected without a runtime flag.
//
// RUN: %clangxx_asan -mllvm -asan-coverage=1 %s -o %t
//
// RUN: rm -rf %T/coverage-disabled
//
// RUN: mkdir -p %T/coverage-disabled/normal
// RUN: ASAN_OPTIONS=coverage_direct=0:coverage_dir=%T/coverage-disabled/normal:verbosity=1 %run %t
// RUN: not %sancov print %T/coverage-disabled/normal/*.sancov 2>&1
//
// RUN: mkdir -p %T/coverage-disabled/direct
// RUN: ASAN_OPTIONS=coverage_direct=1:coverage_dir=%T/coverage-disabled/direct:verbosity=1 %run %t
// RUN: cd %T/coverage-disabled/direct
// RUN: not %sancov rawunpack *.sancov

int main(int argc, char **argv) {
  return 0;
}
