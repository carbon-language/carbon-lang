// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s --check-prefix=CHECK --check-prefix=X64
// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s --check-prefix=CHECK --check-prefix=X86

void try_body(int numerator, int denominator, int *myres) {
  *myres = numerator / denominator;
}
// CHECK-LABEL: define void @try_body(i32 %numerator, i32 %denominator, i32* %myres)
// CHECK: sdiv i32
// CHECK: store i32 %{{.*}}, i32*
// CHECK: ret void

int safe_div(int numerator, int denominator, int *res) {
  int myres = 0;
  int success = 1;
  __try {
    try_body(numerator, denominator, &myres);
  } __except (1) {
    success = -42;
  }
  *res = myres;
  return success;
}

// X64-LABEL: define i32 @safe_div(i32 %numerator, i32 %denominator, i32* %res) {{.*}} personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
// X64: invoke void @try_body(i32 %{{.*}}, i32 %{{.*}}, i32* %{{.*}}) #[[NOINLINE:[0-9]+]]
// X64:       to label %{{.*}} unwind label %[[lpad:[^ ]*]]
//
// X64: [[lpad]]
// X64: landingpad { i8*, i32 }
// X64-NEXT: catch i8* null
// X64-NOT: br i1
// X64: br label %[[except:[^ ]*]]
// X64: [[except]]
// X64: store i32 -42, i32* %[[success:[^ ]*]]
//
// X64: %[[res:[^ ]*]] = load i32, i32* %[[success]]
// X64: ret i32 %[[res]]

// X86-LABEL: define i32 @safe_div(i32 %numerator, i32 %denominator, i32* %res) {{.*}} personality i8* bitcast (i32 (...)* @_except_handler3 to i8*)
// X86: invoke void @try_body(i32 %{{.*}}, i32 %{{.*}}, i32* %{{.*}}) #[[NOINLINE:[0-9]+]]
// X86:       to label %{{.*}} unwind label %[[lpad:[^ ]*]]
//
// X86: [[lpad]]
// X86: landingpad { i8*, i32 }
// X86-NEXT: catch i8* bitcast (i32 ()* @"\01?filt$0@0@safe_div@@" to i8*)
// X86-NOT: br i1
// X86: br label %[[except:[^ ]*]]
// X86: [[except]]
// X86: store i32 -42, i32* %[[success:[^ ]*]]
//
// X86: %[[res:[^ ]*]] = load i32, i32* %[[success]]
// X86: ret i32 %[[res]]

// X86-LABEL: define internal i32 @"\01?filt$0@0@safe_div@@"()
// X86: %[[ebp:[^ ]*]] = call i8* @llvm.frameaddress(i32 1)
// X86: %[[fp:[^ ]*]] = call i8* @llvm.x86.seh.recoverfp(i8* bitcast (i32 (i32, i32, i32*)* @safe_div to i8*), i8* %[[ebp]])
// X86: call i8* @llvm.localrecover(i8* bitcast (i32 (i32, i32, i32*)* @safe_div to i8*), i8* %[[fp]], i32 0)
// X86: load i8*, i8**
// X86: load i32*, i32**
// X86: load i32, i32*
// X86: store i32 %{{.*}}, i32*
// X86: ret i32 1

void j(void);

int filter_expr_capture(void) {
  int r = 42;
  __try {
    j();
  } __except(r = -1) {
    r = 13;
  }
  return r;
}

// CHECK-LABEL: define i32 @filter_expr_capture()
// X64-SAME: personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
// X86-SAME: personality i8* bitcast (i32 (...)* @_except_handler3 to i8*)
// X64: call void (...) @llvm.localescape(i32* %[[r:[^ ,]*]])
// X86: call void (...) @llvm.localescape(i32* %[[r:[^ ,]*]], i32* %[[code:[^ ,]*]])
// CHECK: store i32 42, i32* %[[r]]
// CHECK: invoke void @j() #[[NOINLINE]]
//
// CHECK: landingpad
// CHECK-NEXT: catch i8* bitcast (i32 ({{.*}})* @"\01?filt$0@0@filter_expr_capture@@" to i8*)
// CHECK: store i32 13, i32* %[[r]]
//
// CHECK: %[[rv:[^ ]*]] = load i32, i32* %[[r]]
// CHECK: ret i32 %[[rv]]

// X64-LABEL: define internal i32 @"\01?filt$0@0@filter_expr_capture@@"(i8* %exception_pointers, i8* %frame_pointer)
// X64: call i8* @llvm.localrecover(i8* bitcast (i32 ()* @filter_expr_capture to i8*), i8* %frame_pointer, i32 0)
//
// X86-LABEL: define internal i32 @"\01?filt$0@0@filter_expr_capture@@"()
// X86: %[[ebp:[^ ]*]] = call i8* @llvm.frameaddress(i32 1)
// X86: %[[fp:[^ ]*]] = call i8* @llvm.x86.seh.recoverfp(i8* bitcast (i32 ()* @filter_expr_capture to i8*), i8* %[[ebp]])
// X86: call i8* @llvm.localrecover(i8* bitcast (i32 ()* @filter_expr_capture to i8*), i8* %[[fp]], i32 0)
//
// CHECK: store i32 -1, i32* %{{.*}}
// CHECK: ret i32 -1

int nested_try(void) {
  int r = 42;
  __try {
    __try {
      j();
      r = 0;
    } __except(_exception_code() == 123) {
      r = 123;
    }
  } __except(_exception_code() == 456) {
    r = 456;
  }
  return r;
}
// CHECK-LABEL: define i32 @nested_try()
// X64-SAME: personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
// X86-SAME: personality i8* bitcast (i32 (...)* @_except_handler3 to i8*)
// CHECK: store i32 42, i32* %[[r:[^ ,]*]]
// CHECK: invoke void @j() #[[NOINLINE]]
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: store i32 0, i32* %[[r]]
// CHECK: br label %[[inner_try_cont:[^ ]*]]
//
// CHECK: [[lpad]]
// CHECK: landingpad { i8*, i32 }
// CHECK: catch i8* bitcast (i32 ({{.*}})* @"\01?filt$1@0@nested_try@@" to i8*)
// CHECK: catch i8* bitcast (i32 ({{.*}})* @"\01?filt$0@0@nested_try@@" to i8*)
// CHECK: store i8* %{{.*}}, i8** %[[ehptr_slot:[^ ]*]]
// CHECK: store i32 %{{.*}}, i32* %[[sel_slot:[^ ]*]]
//
// CHECK: load i32, i32* %[[sel_slot]]
// CHECK: call i32 @llvm.eh.typeid.for(i8* bitcast (i32 ({{.*}})* @"\01?filt$1@0@nested_try@@" to i8*))
// CHECK: icmp eq i32
// CHECK: br i1
//
// CHECK: load i32, i32* %[[sel_slot]]
// CHECK: call i32 @llvm.eh.typeid.for(i8* bitcast (i32 ({{.*}})* @"\01?filt$0@0@nested_try@@" to i8*))
// CHECK: icmp eq i32
// CHECK: br i1
//
// CHECK: store i32 456, i32* %[[r]]
// CHECK: br label %[[outer_try_cont:[^ ]*]]
//
// CHECK: [[outer_try_cont]]
// CHECK: %[[r_load:[^ ]*]] = load i32, i32* %[[r]]
// CHECK: ret i32 %[[r_load]]
//
// CHECK: store i32 123, i32* %[[r]]
// CHECK: br label %[[inner_try_cont]]
//
// CHECK: [[inner_try_cont]]
// CHECK: br label %[[outer_try_cont]]
//
// CHECK-LABEL: define internal i32 @"\01?filt$0@0@nested_try@@"({{.*}})
// X86: call i8* @llvm.x86.seh.recoverfp({{.*}})
// CHECK: load i32*, i32**
// CHECK: load i32, i32*
// CHECK: icmp eq i32 %{{.*}}, 456
//
// CHECK-LABEL: define internal i32 @"\01?filt$1@0@nested_try@@"({{.*}})
// X86: call i8* @llvm.x86.seh.recoverfp({{.*}})
// CHECK: load i32*, i32**
// CHECK: load i32, i32*
// CHECK: icmp eq i32 %{{.*}}, 123

int basic_finally(int g) {
  __try {
    j();
  } __finally {
    ++g;
  }
  return g;
}
// CHECK-LABEL: define i32 @basic_finally(i32 %g)
// X64-SAME: personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
// X86-SAME: personality i8* bitcast (i32 (...)* @_except_handler3 to i8*)
// CHECK: %[[g_addr:[^ ]*]] = alloca i32, align 4
// CHECK: call void (...) @llvm.localescape(i32* %[[g_addr]])
// CHECK: store i32 %g, i32* %[[g_addr]]
//
// CHECK: invoke void @j()
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.localaddress()
// CHECK: call void @"\01?fin$0@0@basic_finally@@"({{i8( zeroext)?}} 0, i8* %[[fp]])
// CHECK: load i32, i32* %[[g_addr]], align 4
// CHECK: ret i32
//
// CHECK: [[lpad]]
// CHECK: landingpad { i8*, i32 }
// CHECK-NEXT: cleanup
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.localaddress()
// CHECK: call void @"\01?fin$0@0@basic_finally@@"({{i8( zeroext)?}} 1, i8* %[[fp]])
// CHECK: resume

// CHECK: define internal void @"\01?fin$0@0@basic_finally@@"({{i8( zeroext)?}} %abnormal_termination, i8* %frame_pointer)
// CHECK:   call i8* @llvm.localrecover(i8* bitcast (i32 (i32)* @basic_finally to i8*), i8* %frame_pointer, i32 0)
// CHECK:   load i32, i32* %{{.*}}, align 4
// CHECK:   add nsw i32 %{{.*}}, 1
// CHECK:   store i32 %{{.*}}, i32* %{{.*}}, align 4
// CHECK:   ret void

int returns_int(void);
int except_return(void) {
  __try {
    return returns_int();
  } __except(1) {
    return 42;
  }
}
// CHECK-LABEL: define i32 @except_return()
// CHECK: %[[tmp:[^ ]*]] = invoke i32 @returns_int()
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: store i32 %[[tmp]], i32* %[[rv:[^ ]*]]
// CHECK: br label %[[retbb:[^ ]*]]
//
// CHECK: [[lpad]]
// CHECK: store i32 42, i32* %[[rv]]
// CHECK: br label %[[retbb]]
//
// CHECK: [[retbb]]
// CHECK: %[[r:[^ ]*]] = load i32, i32* %[[rv]]
// CHECK: ret i32 %[[r]]


// PR 24751: don't assert if a variable is used twice in a __finally block.
// Also, make sure we don't do redundant work to capture/project it.
void finally_capture_twice(int x) {
  __try {
  } __finally {
    int y = x;
    int z = x;
  }
}
//  
// CHECK-LABEL: define void @finally_capture_twice(
// CHECK:         [[X:%.*]] = alloca i32, align 4
// CHECK:         call void (...) @llvm.localescape(i32* [[X]])
// CHECK-NEXT:    store i32 {{.*}}, i32* [[X]], align 4
// CHECK-NEXT:    [[LOCAL:%.*]] = call i8* @llvm.localaddress()
// CHECK-NEXT:    call void [[FINALLY:@.*]](i8{{ zeroext | }}0, i8* [[LOCAL]])
// CHECK:       define internal void [[FINALLY]](
// CHECK:         [[LOCAL:%.*]] = call i8* @llvm.localrecover(
// CHECK:         [[X:%.*]] = bitcast i8* [[LOCAL]] to i32*
// CHECK-NEXT:    [[Y:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[Z:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store i8*
// CHECK-NEXT:    store i8
// CHECK-NEXT:    [[T0:%.*]] = load i32, i32* [[X]], align 4
// CHECK-NEXT:    store i32 [[T0]], i32* [[Y]], align 4
// CHECK-NEXT:    [[T0:%.*]] = load i32, i32* [[X]], align 4
// CHECK-NEXT:    store i32 [[T0]], i32* [[Z]], align 4
// CHECK-NEXT:    ret void

// CHECK: attributes #[[NOINLINE]] = { {{.*noinline.*}} }
