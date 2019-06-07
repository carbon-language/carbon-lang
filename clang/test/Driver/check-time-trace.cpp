// RUN: %clangxx -ftime-trace %s 2>&1 | grep "Time trace json-file dumped to" \
// RUN:   | awk '{print $NF}' | xargs cat \
// RUN:   | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN:   | FileCheck %s

// CHECK: "traceEvents": [
// CHECK: "args":
// CHECK: "detail":
// CHECK: "dur":
// CHECK: "name": "Source"
// CHECK-NEXT: "ph":
// CHECK-NEXT: "pid":
// CHECK-NEXT: "tid":
// CHECK-NEXT: "ts":
// CHECK: "name": "clang"
// CHECK: "name": "process_name"

#include <iostream>

int main() {
  std::cout << "Foo" << std::endl;
  return 0;
}
