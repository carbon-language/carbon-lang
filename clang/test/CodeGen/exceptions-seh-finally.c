// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s --check-prefix=CHECK --check-prefix=X64
// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s --check-prefix=CHECK --check-prefix=X86

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
// X64: %[[fp:[^ ]*]] = call i8* @llvm.frameaddress(i32 0)
// X64: call void @"\01?fin$0@0@basic_finally@@"(i8 0, i8* %[[fp]])
// X86: call void @"\01?fin$0@0@basic_finally@@"()
// CHECK-NEXT: ret void
//
// CHECK: [[lpad]]
// CHECK-NEXT: landingpad
// CHECK-NEXT: cleanup
// X64: %[[fp:[^ ]*]] = call i8* @llvm.frameaddress(i32 0)
// X64: call void @"\01?fin$0@0@basic_finally@@"(i8 1, i8* %[[fp]])
// X86: call void @"\01?fin$0@0@basic_finally@@"()
// CHECK: resume { i8*, i32 }

// CHECK: define internal void @"\01?fin$0@0@basic_finally@@"({{.*}})
// CHECK: call void @cleanup()

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
// X64: %[[fp:[^ ]*]] = call i8* @llvm.frameaddress(i32 0)
// X64: call void @"\01?fin$0@0@label_in_finally@@"(i8 0, i8* %[[fp]])
// X86: call void @"\01?fin$0@0@label_in_finally@@"()
// CHECK: ret void

// CHECK: define internal void @"\01?fin$0@0@label_in_finally@@"({{.*}})
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
// X86: call void (...) @llvm.frameescape(i32* %[[abnormal_termination:[^ ),]*]])
// X86: store i32 1, i32* %[[abnormal_termination]]
// CHECK: invoke void @might_crash()
// CHECK:     to label %[[invoke_cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[invoke_cont]]
// X64: %[[fp:[^ ]*]] = call i8* @llvm.frameaddress(i32 0)
// X64: call void @"\01?fin$0@0@use_abnormal_termination@@"(i8 0, i8* %[[fp]])
// X86: store i32 0, i32* %[[abnormal_termination]]
// X86: call void @"\01?fin$0@0@use_abnormal_termination@@"()
// CHECK: ret void
//
// CHECK: [[lpad]]
// CHECK-NEXT: landingpad
// CHECK-NEXT: cleanup
// X64: %[[fp:[^ ]*]] = call i8* @llvm.frameaddress(i32 0)
// X64: call void @"\01?fin$0@0@use_abnormal_termination@@"(i8 1, i8* %[[fp]])
// X86: call void @"\01?fin$0@0@use_abnormal_termination@@"()
// CHECK: resume { i8*, i32 }

// X64: define internal void @"\01?fin$0@0@use_abnormal_termination@@"(i8 %[[abnormal:abnormal_termination]], i8* %frame_pointer)
// X64: %[[abnormal_zext:[^ ]*]] = zext i8 %[[abnormal]] to i32
// X86: define internal void @"\01?fin$0@0@use_abnormal_termination@@"()
// X86: %[[ebp:[^ ]*]] = call i8* @llvm.frameaddress(i32 1)
// X86: %[[fp:[^ ]*]] = call i8* @llvm.x86.seh.recoverfp(i8* bitcast (void ()* @use_abnormal_termination to i8*), i8* %[[ebp]])
// X86: %[[abnormal_i8:[^ ]*]] = call i8* @llvm.framerecover(i8* bitcast (void ()* @use_abnormal_termination to i8*), i8* %[[fp]], i32 0)
// X86: %[[abnormal:[^ ]*]] = bitcast i8* %[[abnormal_i8]] to i32*
// X86: %[[abnormal_zext:[^ ]*]] = load i32, i32* %[[abnormal]]
// CHECK: store i32 %[[abnormal_zext]], i32* @crashed
// CHECK-NEXT: ret void

void noreturn_noop_finally() {
  __try {
    __noop();
  } __finally {
    abort();
  }
}

// CHECK-LABEL: define void @noreturn_noop_finally()
// CHECK: call void @"\01?fin$0@0@noreturn_noop_finally@@"({{.*}})
// CHECK: ret void

// CHECK: define internal void @"\01?fin$0@0@noreturn_noop_finally@@"({{.*}})
// CHECK: call void @abort()
// CHECK: unreachable

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
// CHECK: call void @"\01?fin$0@0@noreturn_finally@@"({{.*}})
// CHECK: ret void
//
// CHECK: [[lpad]]
// CHECK: landingpad
// CHECK-NEXT: cleanup
// CHECK: call void @"\01?fin$0@0@noreturn_finally@@"({{.*}})
// CHECK: resume { i8*, i32 }

// CHECK: define internal void @"\01?fin$0@0@noreturn_finally@@"({{.*}})
// CHECK: call void @abort()
// CHECK: unreachable

int finally_with_return() {
  __try {
    return 42;
  } __finally {
  }
}
// CHECK-LABEL: define i32 @finally_with_return()
// CHECK: call void @"\01?fin$0@0@finally_with_return@@"({{.*}})
// CHECK-NEXT: ret i32 42

// CHECK: define internal void @"\01?fin$0@0@finally_with_return@@"({{.*}})
// CHECK-NOT: br i1
// CHECK-NOT: br label
// CHECK: ret void

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
// CHECK: invoke void @"\01?fin$1@0@nested___finally___finally@@"({{.*}})
// CHECK:          to label %[[outercont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[outercont]]
// CHECK: call void @"\01?fin$0@0@nested___finally___finally@@"({{.*}})
// CHECK-NEXT: ret i32 0
//
// CHECK: [[lpad]]
// CHECK-NEXT: landingpad
// CHECK-NEXT: cleanup
// CHECK: call void @"\01?fin$0@0@nested___finally___finally@@"({{.*}})

// CHECK-LABEL: define internal void @"\01?fin$0@0@nested___finally___finally@@"({{.*}})
// CHECK: ret void

// CHECK-LABEL: define internal void @"\01?fin$1@0@nested___finally___finally@@"({{.*}})
// CHECK: unreachable

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
// CHECK: invoke void @might_crash()
// CHECK-NEXT: to label %[[invokecont:[^ ]*]] unwind label %[[lpad1:[^ ]*]]
//
// [[invokecont]]
// CHECK: invoke void @"\01?fin$1@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK:          to label %[[outercont:[^ ]*]] unwind label %[[lpad2:[^ ]*]]
//
// CHECK: [[outercont]]
// CHECK: call void @"\01?fin$0@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK-NEXT: ret i32 912
//
// CHECK: [[lpad1]]
// CHECK-NEXT: landingpad
// CHECK-NEXT: cleanup
// CHECK: invoke void @"\01?fin$1@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK:          to label %[[outercont:[^ ]*]] unwind label %[[lpad2]]
//
// CHECK: [[lpad2]]
// CHECK-NEXT: landingpad
// CHECK-NEXT: cleanup
// CHECK: call void @"\01?fin$0@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK: resume

// CHECK-LABEL: define internal void @"\01?fin$0@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK: ret void

// CHECK-LABEL: define internal void @"\01?fin$1@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK: unreachable
