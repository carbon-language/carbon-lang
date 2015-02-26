// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s

void abort(void) __attribute__((noreturn));
void might_crash(void);
void cleanup(void);
int check_condition(void);
void basic_finally(void) {
  __try {
    might_crash();
  } __finally {
    cleanup();
  }
}

// CHECK-LABEL: define void @basic_finally()
// CHECK: invoke void @might_crash()
// CHECK:     to label %[[invoke_cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[invoke_cont]]
// CHECK: store i8 0, i8* %[[abnormal:[^ ]*]]
// CHECK: br label %[[finally:[^ ]*]]
//
// CHECK: [[finally]]
// CHECK: call void @cleanup()
// CHECK: load i8* %[[abnormal]]
// CHECK: icmp eq
// CHECK: br i1 %{{.*}}, label %[[finallycont:[^ ]*]], label %[[resumecont:[^ ]*]]
//
// CHECK: [[finallycont]]
// CHECK-NEXT: ret void
//
// CHECK: [[lpad]]
// CHECK-NEXT: landingpad
// CHECK-NEXT: cleanup
// CHECK: store i8 1, i8* %[[abnormal]]
// CHECK: br label %[[finally]]
//
// CHECK: [[resumecont]]
// CHECK: br label %[[ehresume:[^ ]*]]
//
// CHECK: [[ehresume]]
// CHECK: resume { i8*, i32 }

// Mostly check that we don't double emit 'r' which would crash.
void decl_in_finally(void) {
  __try {
    might_crash();
  } __finally {
    int r;
  }
}

// Ditto, don't crash double emitting 'l'.
void label_in_finally(void) {
  __try {
    might_crash();
  } __finally {
l:
    cleanup();
    if (check_condition())
      goto l;
  }
}

// CHECK-LABEL: define void @label_in_finally()
// CHECK: invoke void @might_crash()
// CHECK:     to label %[[invoke_cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[invoke_cont]]
// CHECK: store i8 0, i8* %[[abnormal:[^ ]*]]
// CHECK: br label %[[finally:[^ ]*]]
//
// CHECK: [[finally]]
// CHECK: br label %[[l:[^ ]*]]
//
// CHECK: [[l]]
// CHECK: call void @cleanup()
// CHECK: call i32 @check_condition()
// CHECK: br i1 {{.*}}, label
// CHECK: br label %[[l]]

int crashed;
void use_abnormal_termination(void) {
  __try {
    might_crash();
  } __finally {
    crashed = __abnormal_termination();
  }
}

// CHECK-LABEL: define void @use_abnormal_termination()
// CHECK: invoke void @might_crash()
// CHECK:     to label %[[invoke_cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[invoke_cont]]
// CHECK: store i8 0, i8* %[[abnormal:[^ ]*]]
// CHECK: br label %[[finally:[^ ]*]]
//
// CHECK: [[finally]]
// CHECK: load i8* %[[abnormal]]
// CHECK: zext i8 %{{.*}} to i32
// CHECK: store i32 %{{.*}}, i32* @crashed
// CHECK: load i8* %[[abnormal]]
// CHECK: icmp eq
// CHECK: br i1 %{{.*}}, label %[[finallycont:[^ ]*]], label %[[resumecont:[^ ]*]]
//
// CHECK: [[finallycont]]
// CHECK-NEXT: ret void
//
// CHECK: [[lpad]]
// CHECK-NEXT: landingpad
// CHECK-NEXT: cleanup
// CHECK: store i8 1, i8* %[[abnormal]]
// CHECK: br label %[[finally]]
//
// CHECK: [[resumecont]]
// CHECK: br label %[[ehresume:[^ ]*]]
//
// CHECK: [[ehresume]]
// CHECK: resume { i8*, i32 }

void noreturn_noop_finally() {
  __try {
    __noop();
  } __finally {
    abort();
  }
}

// CHECK-LABEL: define void @noreturn_noop_finally()
// CHECK: store i8 0, i8* %
// CHECK: br label %[[finally:[^ ]*]]
// CHECK: [[finally]]
// CHECK: call void @abort()
// CHECK-NEXT: unreachable
// CHECK-NOT: load

void noreturn_finally() {
  __try {
    might_crash();
  } __finally {
    abort();
  }
}

// CHECK-LABEL: define void @noreturn_finally()
// CHECK: invoke void @might_crash()
// CHECK:     to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: store i8 0, i8* %
// CHECK: br label %[[finally:[^ ]*]]
//
// CHECK: [[finally]]
// CHECK: call void @abort()
// CHECK-NEXT: unreachable
//
// CHECK: [[lpad]]
// CHECK: landingpad
// CHECK-NEXT: cleanup
// CHECK: store i8 1, i8* %
// CHECK: br label %[[finally]]

int finally_with_return() {
  __try {
    return 42;
  } __finally {
  }
}
// CHECK-LABEL: define i32 @finally_with_return()
// CHECK: store i8 0, i8* %
// CHECK-NEXT: br label %[[finally:[^ ]*]]
//
// CHECK: [[finally]]
// CHECK-NEXT: br label %[[finallycont:[^ ]*]]
//
// CHECK: [[finallycont]]
// CHECK-NEXT: ret i32 42

int nested___finally___finally() {
  __try {
    __try {
    } __finally {
      return 1;
    }
  } __finally {
    // Intentionally no return here.
  }
  return 0;
}
// CHECK-LABEL: define i32 @nested___finally___finally
// CHECK: store i8 0, i8* %
// CHECK-NEXT: br label %[[finally:[^ ]*]]
//
// CHECK: [[finally]]
// CHECK-NEXT: store i32 1, i32* %
// CHECK-NEXT: store i32 1, i32* %
// CHECK-NEXT: br label %[[cleanup:[^ ]*]]
//
// The finally's unreachable continuation block:
// CHECK: store i32 0, i32* %
// CHECK-NEXT: br label %[[cleanup]]
//
// CHECK: [[cleanup]]
// CHECK-NEXT: store i8 0, i8* %
// CHECK-NEXT: br label %[[outerfinally:[^ ]*]]
//
// CHECK: [[outerfinally]]
// CHECK-NEXT: br label %[[finallycont:[^ ]*]]
//
// CHECK: [[finallycont]]
// CHECK-NEXT: %[[dest:[^ ]*]] = load i32* %
// CHECK-NEXT: switch i32 %[[dest]]
// CHECK-NEXT: i32 0, label %[[cleanupcont:[^ ]*]]
//
// CHECK: [[cleanupcont]]
// CHECK-NEXT: store i32 0, i32* %
// CHECK-NEXT: br label %[[return:[^ ]*]]
//
// CHECK: [[return]]
// CHECK-NEXT: %[[reg:[^ ]*]] = load i32* %
// CHECK-NEXT: ret i32 %[[reg]]

int nested___finally___finally_with_eh_edge() {
  __try {
    __try {
      might_crash();
    } __finally {
      return 899;
    }
  } __finally {
    // Intentionally no return here.
  }
  return 912;
}
// CHECK-LABEL: define i32 @nested___finally___finally_with_eh_edge
// CHECK: invoke void @might_crash() #3
// CHECK-NEXT: to label %[[invokecont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[invokecont]]
// CHECK-NEXT: store i8 0, i8* %[[abnormal:[^ ]*]]
// CHECK-NEXT: br label %[[finally:[^ ]*]]

// CHECK: [[finally]]
// CHECK-NEXT: store i32 899, i32* %
// CHECK-NEXT: store i32 1, i32* %
// CHECK-NEXT: br label %[[cleanup:[^ ]*]]
//
// The inner finally's unreachable continuation block:
// CHECK: store i32 0, i32* %
// CHECK-NEXT: br label %[[cleanup]]
//
// CHECK: [[cleanup]]
// CHECK-NEXT: store i8 0, i8* %
// CHECK-NEXT: br label %[[outerfinally:[^ ]*]]
//
// CHECK: [[outerfinally]]
// CHECK-NEXT: %[[abnormallocal:[^ ]*]] = load i8* %[[abnormal]]
// CHECK-NEXT: %[[reg:[^ ]*]] = icmp eq i8 %[[abnormallocal]], 0
// CHECK-NEXT: br i1 %[[reg]], label %[[finallycont:[^ ]*]], label %[[finallyresume:[^ ]*]]
//
// CHECK: [[finallycont]]
// CHECK-NEXT: %[[dest:[^ ]*]] = load i32* %
// CHECK-NEXT: switch i32 %[[dest]]
// CHECK-NEXT: i32 0, label %[[cleanupcont:[^ ]*]]
//
// CHECK: [[cleanupcont]]
// CHECK-NEXT: store i32 912, i32* %
// CHECK-NEXT: br label %[[return:[^ ]*]]
//
//
// CHECK: [[lpad]]
// CHECK-NEXT: landingpad
// CHECK-NEXT: cleanup
// CHECK: store i8 1, i8* %[[abnormal]]
// CHECK: br label %[[finally]]
//
// The inner finally's unreachable resume block:
// CHECK: store i8 1, i8* %[[abnormal]]
// CHECK-NEXT: br label %[[outerfinally]]
//
// CHECK: [[finallyresume]]
// CHECK-NEXT: br label %[[ehresume:[^ ]*]]
//
// CHECK: [[return]]
// CHECK-NEXT: %[[reg:[^ ]*]] = load i32* %
// CHECK-NEXT: ret i32 %[[reg]]
//
// The ehresume block, not reachable either.
// CHECK: [[ehresume]]
// CHECK: resume
