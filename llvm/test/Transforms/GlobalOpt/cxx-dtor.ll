; RUN: opt < %s -globalopt -S | FileCheck %s

%0 = type { i32, void ()* }
%struct.A = type { i8 }

@a = global %struct.A zeroinitializer, align 1
@__dso_handle = external global i8*
@llvm.global_ctors = appending global [1 x %0] [%0 { i32 65535, void ()* @_GLOBAL__I_a }]

; CHECK-NOT: call i32 @__cxa_atexit

define internal void @__cxx_global_var_init() nounwind section "__TEXT,__StaticInit,regular,pure_instructions" {
  %1 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.A*)* @_ZN1AD1Ev to void (i8*)*), i8* getelementptr inbounds (%struct.A* @a, i32 0, i32 0), i8* bitcast (i8** @__dso_handle to i8*))
  ret void
}

define linkonce_odr void @_ZN1AD1Ev(%struct.A* %this) nounwind align 2 {
  call void @_ZN1AD2Ev(%struct.A* %this)
  ret void
}

declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*)

define linkonce_odr void @_ZN1AD2Ev(%struct.A* %this) nounwind align 2 {
  ret void
}

define internal void @_GLOBAL__I_a() nounwind section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @__cxx_global_var_init()
  ret void
}
