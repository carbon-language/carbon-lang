// RUN: %clangxx_tsan -O1 %s -o %t && not %t 2>&1 | FileCheck %s
extern "C" void AnnotateIgnoreWritesBegin(const char *f, int l);

int main() {
  AnnotateIgnoreWritesBegin("", 0);
}

// CHECK: ThreadSanitizer: main thread finished with ignores enabled
// CHECK:   Ignore was enabled at:
// CHECK:     #0 AnnotateIgnoreWritesBegin
// CHECK:     #1 main

