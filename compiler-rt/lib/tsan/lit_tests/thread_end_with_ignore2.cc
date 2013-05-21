// RUN: %clangxx_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
extern "C" void AnnotateIgnoreWritesBegin(const char *f, int l);

int main() {
  AnnotateIgnoreWritesBegin("", 0);
}

// CHECK: ThreadSanitizer: thread T0 finished with ignores enabled

