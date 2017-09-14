// RUN: clang-refactor local-rename -selection=test:%s -v %s -- 2>&1 | FileCheck %s

/*range=*/int test;

/*range named=*/int test2;

/*range= +1*/int test3;

/* range = +100 */int test4;

/*range named =+0*/int test5;

// CHECK: Test selection group '':
// CHECK-NEXT:   95-95
// CHECK-NEXT:   148-148
// CHECK-NEXT:   187-187
// CHECK-NEXT: Test selection group 'named':
// CHECK-NEXT:   122-122
// CHECK-NEXT:   208-208

// The following invocations are in the default group:

// CHECK: invoking action 'local-rename':
// CHECK-NEXT: -selection={{.*}}tool-test-support.c:3:11

// CHECK: invoking action 'local-rename':
// CHECK-NEXT: -selection={{.*}}tool-test-support.c:7:15

// CHECK: invoking action 'local-rename':
// CHECK-NEXT: -selection={{.*}}tool-test-support.c:9:29


// The following invocations are in the 'named' group, and they follow
// the default invocation even if some of their ranges occur prior to the
// ranges from the default group because the groups are tested one-by-one:

// CHECK: invoking action 'local-rename':
// CHECK-NEXT: -selection={{.*}}tool-test-support.c:5:17

// CHECK: invoking action 'local-rename':
// CHECK-NEXT: -selection={{.*}}tool-test-support.c:11:20
