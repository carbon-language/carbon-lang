; RUN: opt < %s -print-callgraph -disable-output 2>&1 | FileCheck %s
; CHECK: Call graph node <<null function>><<{{.*}}>>  #uses=0
; CHECK-NEXT:   CS<{{.*}}> calls function 'f'
; CHECK-NEXT:   CS<{{.*}}> calls function '__kmpc_fork_call'
; CHECK-EMPTY:

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @f() {
entry:
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @1, i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* @f..omp_par to void (i32*, i32*, ...)*))
  br label %omp.par.exit.split

omp.par.exit.split:                               ; preds = %omp_parallel
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @f..omp_par(i32* noalias %tid.addr, i32* noalias %zero.addr) {
omp.par.entry:
  %tid.addr.local = alloca i32, align 4
  %0 = load i32, i32* %tid.addr, align 4
  store i32 %0, i32* %tid.addr.local, align 4
  %tid = load i32, i32* %tid.addr.local, align 4
  br label %omp.par.region

omp.par.exit.split.exitStub:                      ; preds = %omp.par.outlined.exit
  ret void

omp.par.region:                                   ; preds = %omp.par.entry
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.par.region
  br label %omp.par.outlined.exit

omp.par.outlined.exit:                            ; preds = %omp.par.pre_finalize
  br label %omp.par.exit.split.exitStub
}

; Function Attrs: nounwind
declare !callback !2 void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) #2

!2 = !{!3}
!3 = !{i64 2, i64 -1, i64 -1, i1 true}
