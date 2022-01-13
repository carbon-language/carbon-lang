; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; rdar://12580965.
; ObjC++ test case.

; Make it easier to verify that alignment is correct/optimal:
; 16-bit pointers with 32-bit ABI alignment and 128-bit preferred alignment

target datalayout = "p:16:32:128" 

%struct.ButtonInitData = type { i8* }

@_ZL14buttonInitData = internal global [1 x %struct.ButtonInitData] zeroinitializer, align 4

@"\01L_OBJC_METH_VAR_NAME_40" = internal global [7 x i8] c"print:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_SELECTOR_REFERENCES_41" = internal externally_initialized  global i8* getelementptr inbounds ([7 x i8], [7 x i8]* @"\01L_OBJC_METH_VAR_NAME_40", i32 0, i32 0), section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_a, i8* null }]
@llvm.used = appending global [2 x i8*] [i8* getelementptr inbounds ([7 x i8], [7 x i8]* @"\01L_OBJC_METH_VAR_NAME_40", i32 0, i32 0),  i8* bitcast (i8** @"\01L_OBJC_SELECTOR_REFERENCES_41" to i8*)]

; Choose the preferred alignment.

; CHECK: @[[_ZL14BUTTONINITDATA_0_0:[a-zA-Z0-9_$"\\.-]+]] = internal unnamed_addr global i8* null, align 16
;.
define internal void @__cxx_global_var_init() section "__TEXT,__StaticInit,regular,pure_instructions" {
  %1 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_41", !invariant.load !2009
  store i8* %1, i8** getelementptr inbounds ([1 x %struct.ButtonInitData], [1 x %struct.ButtonInitData]* @_ZL14buttonInitData, i32 0, i32 0, i32 0), align 4
  ret void
}

define internal void @_GLOBAL__I_a() section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @__cxx_global_var_init()
  ret void
}

declare void @test(i8*)

; The preferred alignment is available.

define void @print() {
; CHECK-LABEL: @print(
; CHECK-NEXT:    [[TMP1:%.*]] = load i8*, i8** @_ZL14buttonInitData.0.0, align 16
; CHECK-NEXT:    call void @test(i8* [[TMP1]])
; CHECK-NEXT:    ret void
;
  %1 = load i8*, i8** getelementptr inbounds ([1 x %struct.ButtonInitData], [1 x %struct.ButtonInitData]* @_ZL14buttonInitData, i32 0, i32 0, i32 0), align 4
  call void @test(i8* %1)
  ret void
}

!2009 = !{}
