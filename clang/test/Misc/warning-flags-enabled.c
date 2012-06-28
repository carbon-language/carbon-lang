// RUN: diagtool show-enabled %s | FileCheck %s
//
// This shows warnings which are on by default.
// We just check a few to make sure it's doing something sensible.
//
// CHECK: ext_unterminated_string
// CHECK: warn_condition_is_assignment
// CHECK: warn_null_arg


// RUN: diagtool show-enabled -Wno-everything %s | count 0


// RUN: diagtool show-enabled -Wno-everything -Wobjc-root-class %s | FileCheck -check-prefix CHECK-WARN %s
// RUN: diagtool show-enabled -Wno-everything -Werror=objc-root-class %s | FileCheck -check-prefix CHECK-ERROR %s
// RUN: diagtool show-enabled -Wno-everything -Wfatal-errors=objc-root-class %s | FileCheck -check-prefix CHECK-FATAL %s
//
// CHECK-WARN:  W  warn_objc_root_class_missing [-Wobjc-root-class]
// CHECK-ERROR: E  warn_objc_root_class_missing [-Wobjc-root-class]
// CHECK-FATAL: F  warn_objc_root_class_missing [-Wobjc-root-class]

// RUN: diagtool show-enabled --no-levels -Wno-everything -Wobjc-root-class %s | FileCheck -check-prefix CHECK-NO-LEVELS %s
//
// CHECK-NO-LEVELS-NOT: W
// CHECK-NO-LEVELS-NOT: E
// CHECK-NO-LEVELS-NOT: F
// CHECK-NO-LEVELS: warn_objc_root_class_missing [-Wobjc-root-class]
