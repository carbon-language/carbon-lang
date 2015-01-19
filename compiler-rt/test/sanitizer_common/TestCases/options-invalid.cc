// RUN: %clangxx -O0 %s -o %t
// RUN: %tool_options=invalid_option_name=10 not %run %t 2>&1 | FileCheck %s

int main() {
}

// CHECK: Unknown flag{{.*}}invalid_option_name
