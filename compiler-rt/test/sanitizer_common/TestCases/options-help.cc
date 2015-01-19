// RUN: %clangxx -O0 %s -o %t
// RUN: %tool_options=help=1 %run %t 2>&1 | FileCheck %s

int main() {
}

// CHECK: Available flags for {{.*}}Sanitizer:
// CHECK: handle_segv
