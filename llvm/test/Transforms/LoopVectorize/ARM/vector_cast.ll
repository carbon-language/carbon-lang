; RUN: opt -loop-vectorize -tbaa -S < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--linux-gnueabi"

; This requires the loop vectorizer to create an interleaved access group
; for the stores to the struct. Here we need to perform a bitcast from a vector
; of pointers to a vector i32s.

%class.C = type { i8 }
%class.B = type { %class.A* }
%class.A = type { i8*, i32 }

; CHECK-LABEL: test0
define void @test0(%class.C* nocapture readnone %this, %class.B* dereferenceable(4) %p1) #0 align 2 {
entry:
  %call.i = tail call %class.A* @_ZN1B5m_fn2Ev(%class.B* nonnull %p1)
  %resize_I.i = getelementptr inbounds %class.B, %class.B* %p1, i32 0, i32 0
  %0 = load %class.A*, %class.A** %resize_I.i, align 4, !tbaa !3
  %cmp.6.i = icmp eq %class.A* %0, %call.i
  br i1 %cmp.6.i, label %_ZN1B5m_fn1Ev.exit, label %for.body.lr.ph.i

for.body.lr.ph.i:                                 ; preds = %entry
  %resize_I.promoted8.i = ptrtoint %class.A* %0 to i32
  %scevgep.i = getelementptr %class.A, %class.A* %call.i, i32 -1, i32 0
  %1 = ptrtoint i8** %scevgep.i to i32
  %2 = sub i32 %1, %resize_I.promoted8.i
  br label %for.body.i

for.cond.for.cond.cleanup_crit_edge.i:            ; preds = %for.body.i
  %3 = lshr i32 %2, 3
  %4 = add nuw nsw i32 %3, 1
  %scevgep10.i = getelementptr %class.A, %class.A* %0, i32 %4
  store %class.A* %scevgep10.i, %class.A** %resize_I.i, align 4, !tbaa !3
  br label %_ZN1B5m_fn1Ev.exit

for.body.i:                                       ; preds = %for.body.i, %for.body.lr.ph.i
  %5 = phi %class.A* [ %0, %for.body.lr.ph.i ], [ %incdec.ptr.i, %for.body.i ]
  %Data.i.i = getelementptr inbounds %class.A, %class.A* %5, i32 0, i32 0
  store i8* null, i8** %Data.i.i, align 4, !tbaa !8
  %Length.i.i = getelementptr inbounds %class.A, %class.A* %5, i32 0, i32 1
  store i32 0, i32* %Length.i.i, align 4, !tbaa !11
  %incdec.ptr.i = getelementptr inbounds %class.A, %class.A* %5, i32 1
  %cmp.i = icmp eq %class.A* %incdec.ptr.i, %call.i
  br i1 %cmp.i, label %for.cond.for.cond.cleanup_crit_edge.i, label %for.body.i

_ZN1B5m_fn1Ev.exit:                               ; preds = %for.cond.for.cond.cleanup_crit_edge.i, %entry
  ret void
}

declare %class.A* @_ZN1B5m_fn2Ev(%class.B*) #0

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a8" "target-features"="+neon,+vfp3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{!"clang version 3.8.0 (trunk 246510) (llvm/trunk 246509)"}
!3 = !{!4, !5, i64 0}
!4 = !{!"_ZTS1B", !5, i64 0}
!5 = !{!"any pointer", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!9, !5, i64 0}
!9 = !{!"_ZTS1A", !5, i64 0, !10, i64 4}
!10 = !{!"int", !6, i64 0}
!11 = !{!9, !10, i64 4}
