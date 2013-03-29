// RUN: %clang --analyze %s -o /dev/null -Xclang -analyzer-checker=debug.ConfigDumper > %t 2>&1
// RUN: FileCheck --input-file=%t %s

void bar() {}
void foo() { bar(); }

// CHECK: [config]
// CHECK-NEXT: cfg-conditional-static-initializers = true
// CHECK-NEXT: cfg-temporary-dtors = false
// CHECK-NEXT: faux-bodies = true
// CHECK-NEXT: graph-trim-interval = 1000
// CHECK-NEXT: ipa = dynamic-bifurcate
// CHECK-NEXT: ipa-always-inline-size = 3
// CHECK-NEXT: max-inlinable-size = 50
// CHECK-NEXT: max-nodes = 150000
// CHECK-NEXT: max-times-inline-large = 32
// CHECK-NEXT: mode = deep
// CHECK-NEXT: [stats]
// CHECK-NEXT: num-entries = 10

