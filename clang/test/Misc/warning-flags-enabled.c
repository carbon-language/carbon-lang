// RUN: diagtool show-enabled 2>&1 | FileCheck %s
//
// This shows warnings which are on by default.
// We just check a few to make sure it's doing something sensible.
//
// CHECK: warn_condition_is_assignment
// CHECK: warn_null_arg
// CHECK: warn_unterminated_string


// RUN: diagtool show-enabled -Wno-everything 2>&1 | count 0


// RUN: diagtool show-enabled -Wno-everything -Wobjc-root-class 2>&1 | FileCheck -check-prefix CHECK-WARN %s
// RUN: diagtool show-enabled -Wno-everything -Werror=objc-root-class 2>&1 | FileCheck -check-prefix CHECK-ERROR %s
// RUN: diagtool show-enabled -Wno-everything -Wfatal-errors=objc-root-class 2>&1 | FileCheck -check-prefix CHECK-FATAL %s
//
// CHECK-WARN:  W  warn_objc_root_class_missing [-Wobjc-root-class]
// CHECK-ERROR: E  warn_objc_root_class_missing [-Wobjc-root-class]
// CHECK-FATAL: F  warn_objc_root_class_missing [-Wobjc-root-class]

// RUN: diagtool show-enabled --no-flags -Wno-everything -Wobjc-root-class 2>&1 | FileCheck -check-prefix CHECK-NO-FLAGS %s
//
// CHECK-NO-FLAGS-NOT: W
// CHECK-NO-FLAGS-NOT: E
// CHECK-NO-FLAGS-NOT: F
// CHECK-NO-FLAGS: warn_objc_root_class_missing [-Wobjc-root-class]
