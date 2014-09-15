// RUN: llvm-cov show %S/Inputs/regionMarkers.covmapping -instr-profile %S/Inputs/regionMarkers.profdata -show-regions -dump -filename-equivalence %s 2>&1 | FileCheck %s

int main() {                      // CHECK: Marker at [[@LINE]]:12 = 1
  int x = 0;

  if (x) {                        // CHECK: Marker at [[@LINE]]:10 = 0
    x = 0;
  } else {                        // CHECK: Marker at [[@LINE]]:10 = 1
    x = 1;
  }
                                  // CHECK: Marker at [[@LINE+2]]:19 = 101
                                  // CHECK: Marker at [[@LINE+1]]:28 = 100
  for (int i = 0; i < 100; ++i) { // CHECK: Marker at [[@LINE]]:33 = 100
    x = 1;
  }
                                  // CHECK: Marker at [[@LINE+1]]:16 = 1
  x = x < 10 ? x + 1 : x - 1;     // CHECK: Marker at [[@LINE]]:24 = 0
  x = x > 10 ?
        x - 1:                    // CHECK: Marker at [[@LINE]]:9 = 0
        x + 1;                    // CHECK: Marker at [[@LINE]]:9 = 1

  return 0;
}

// llvm-cov doesn't work on big endian yet
// XFAIL: powerpc64-, s390x, mips-, mips64-, sparc
