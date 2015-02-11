// RUN: %clang_cc1 -std=c++11 -fblocks -fms-extensions %s -triple=x86_64-windows-msvc -emit-llvm \
// RUN:         -o - -mconstructor-aliases -fcxx-exceptions -fexceptions | \
// RUN:         FileCheck %s --check-prefix=CHECK --check-prefix=CXXEH
// RUN: %clang_cc1 -std=c++11 -fblocks -fms-extensions %s -triple=x86_64-windows-msvc -emit-llvm \
// RUN:         -o - -mconstructor-aliases -O1 -disable-llvm-optzns | \
// RUN:         FileCheck %s --check-prefix=CHECK --check-prefix=NOCXX

extern "C" void might_throw();

struct HasCleanup {
  HasCleanup();
  ~HasCleanup();
  int padding;
};

extern "C" void use_cxx() {
  HasCleanup x;
  might_throw();
}

// Make sure we use __CxxFrameHandler3 for C++ EH.

// CXXEH-LABEL: define void @use_cxx()
// CXXEH: call %struct.HasCleanup* @"\01??0HasCleanup@@QEAA@XZ"(%struct.HasCleanup* %{{.*}})
// CXXEH: invoke void @might_throw()
// CXXEH:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CXXEH: [[cont]]
// CXXEH: call void @"\01??1HasCleanup@@QEAA@XZ"(%struct.HasCleanup* %{{.*}})
// CXXEH: ret void
//
// CXXEH: [[lpad]]
// CXXEH: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
// CXXEH-NEXT: cleanup
// CXXEH: call void @"\01??1HasCleanup@@QEAA@XZ"(%struct.HasCleanup* %{{.*}})
// CXXEH: br label %[[resume:[^ ]*]]
//
// CXXEH: [[resume]]
// CXXEH: resume

// NOCXX-LABEL: define void @use_cxx()
// NOCXX-NOT: invoke
// NOCXX: call %struct.HasCleanup* @"\01??0HasCleanup@@QEAA@XZ"(%struct.HasCleanup* %{{.*}})
// NOCXX-NOT: invoke
// NOCXX: call void @might_throw()
// NOCXX-NOT: invoke
// NOCXX: call void @"\01??1HasCleanup@@QEAA@XZ"(%struct.HasCleanup* %{{.*}})
// NOCXX-NOT: invoke
// NOCXX: ret void

extern "C" void use_seh() {
  __try {
    might_throw();
  } __except(1) {
  }
}

// Make sure we use __C_specific_handler for SEH.

// CHECK-LABEL: define void @use_seh()
// CHECK: invoke void @might_throw() #[[NOINLINE:[0-9]+]]
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: br label %[[ret:[^ ]*]]
//
// CHECK: [[lpad]]
// CHECK: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
// CHECK-NEXT: catch i8*
//
// CHECK: br label %[[ret]]
//
// CHECK: [[ret]]
// CHECK: ret void

void use_seh_in_lambda() {
  ([]() {
    __try {
      might_throw();
    } __except(1) {
    }
  })();
  HasCleanup x;
  might_throw();
}

// CXXEH-LABEL: define void @"\01?use_seh_in_lambda@@YAXXZ"()
// CXXEH: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)

// NOCXX-LABEL: define void @"\01?use_seh_in_lambda@@YAXXZ"()
// NOCXX-NOT: invoke
// NOCXX: ret void

// CHECK-LABEL: define internal void @"\01??R<lambda_0>@?use_seh_in_lambda@@YAXXZ@QEBAXXZ"(%class.anon* %this)
// CHECK: invoke void @might_throw() #[[NOINLINE]]
// CHECK: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)

// CHECK: attributes #[[NOINLINE]] = { {{.*noinline.*}} }
