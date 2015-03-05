// RUN: llgo -S -emit-llvm -o - %s | FileCheck %s

package foo

import _ "fmt"

var X interface{}

// CHECK: @"init$guard" = internal global i1 false

// CHECK: define void @foo..import()
// CHECK-NEXT: :
// CHECK-NEXT: %[[N:.*]] = load i1, i1* @"init$guard"
// CHECK-NEXT: br i1 %[[N]], label %{{.*}}, label %[[L:.*]]

// CHECK: ; <label>:[[L]]
// CHECK-NEXT: call void @__go_register_gc_roots
// CHECK-NEXT: store i1 true, i1* @"init$guard"
// CHECK-NEXT: call void @fmt..import()
// CHECK-NEXT: br label
