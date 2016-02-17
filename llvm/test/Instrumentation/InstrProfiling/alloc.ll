;; Ensure that SHF_ALLOC section flag is not set for the __llvm_covmap section on Linux.
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

@__profn_foo = private constant [3 x i8] c"foo"
@__llvm_coverage_mapping = internal constant { { i32, i32, i32, i32 }, [1 x <{ i64, i32, i64 }>], [24 x i8] } { { i32, i32, i32, i32 } { i32 1, i32 10, i32 14, i32 1 }, [1 x <{ i64, i32, i64 }>] [<{ i64, i32, i64 }> <{ i64 6699318081062747564, i32 9, i64 0 }>], [24 x i8] c"\01\08/tmp/t.c\01\00\00\01\01\01\0C\01\02\00\00\00\00\00" }, section "__llvm_covmap", align 8
@__profc_foo = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_foo = private global { i64, i64, i64*, i8*, i8*, i32, [1 x i16] } { i64 6699318081062747564, i64 0, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_foo, i32 0, i32 0), i8* bitcast (void ()* @foo to i8*), i8* null, i32 1, [1 x i16] zeroinitializer }, section "__llvm_prf_data", align 8
@__llvm_prf_nm = private constant [13 x i8] c"\03\0Bx\DAK\CB\CF\07\00\02\82\01E", section "__llvm_prf_names"
@llvm.used = appending global [3 x i8*] [i8* bitcast ({ { i32, i32, i32, i32 }, [1 x <{ i64, i32, i64 }>], [24 x i8] }* @__llvm_coverage_mapping to i8*), i8* bitcast ({ i64, i64, i64*, i8*, i8*, i32, [1 x i16] }* @__profd_foo to i8*), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @__llvm_prf_nm, i32 0, i32 0)], section "llvm.metadata"

; Function Attrs: nounwind uwtable
define void @foo() #0 {
  %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_foo, i64 0, i64 0)
  %1 = add i64 %pgocount, 1
  store i64 %1, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_foo, i64 0, i64 0)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.instrprof.increment(i8*, i64, i32, i32) #1

; CHECK-DAG: .section	__llvm_covmap,""
; CHECK-DAG: .section	__llvm_prf_cnts,"aw",@progbits
; CHECK-DAG: .section	__llvm_prf_data,"aw",@progbits

