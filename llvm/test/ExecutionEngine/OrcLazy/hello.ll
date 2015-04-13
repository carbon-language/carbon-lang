; RUN: lli -jit-kind=orc-lazy -orc-lazy-debug=funcs-to-stderr %s 2>&1 | FileCheck %s
;
; CHECK: Hello
; CHECK: [ main$orc_body ]
; CHECK: Goodbye

%class.Foo = type { i8 }
%struct.__sFILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct.__sFILEX = type opaque
%struct.__sbuf = type { i8*, i32 }

@f = global %class.Foo zeroinitializer, align 1
@__dso_handle = external global i8
@__stderrp = external global %struct.__sFILE*
@.str = private unnamed_addr constant [7 x i8] c"Hello\0A\00", align 1
@.str1 = private unnamed_addr constant [9 x i8] c"Goodbye\0A\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_hello.cpp, i8* null }]

define internal void @_GLOBAL__sub_I_hello.cpp() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  call void @_ZN3FooC1Ev(%class.Foo* @f)
  %0 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%class.Foo*)* @_ZN3FooD1Ev to void (i8*)*), i8* getelementptr inbounds (%class.Foo, %class.Foo* @f, i32 0, i32 0), i8* @__dso_handle)
  ret void
}

define linkonce_odr void @_ZN3FooC1Ev(%class.Foo* %this) unnamed_addr align 2 {
entry:
  %0 = load %struct.__sFILE*, %struct.__sFILE** @__stderrp, align 8
  %call.i = call i32 (%struct.__sFILE*, i8*, ...)* @fprintf(%struct.__sFILE* %0, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str, i32 0, i32 0))
  ret void
}

define linkonce_odr void @_ZN3FooD1Ev(%class.Foo* %this) unnamed_addr align 2 {
entry:
  %0 = load %struct.__sFILE*, %struct.__sFILE** @__stderrp, align 8
  %call.i = call i32 (%struct.__sFILE*, i8*, ...)* @fprintf(%struct.__sFILE* %0, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str1, i32 0, i32 0))
  ret void
}


define i32 @main(i32 %argc, i8** %argv) {
entry:
  ret i32 0
}

declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*)
declare i32 @fprintf(%struct.__sFILE*, i8*, ...)
