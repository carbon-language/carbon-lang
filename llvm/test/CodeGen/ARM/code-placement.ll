; RUN: llc < %s -mtriple=armv7-apple-darwin | FileCheck %s
; PHI elimination shouldn't break backedge.
; rdar://8263994

%struct.list_data_s = type { i16, i16 }
%struct.list_head = type { %struct.list_head*, %struct.list_data_s* }

define arm_apcscc %struct.list_head* @t(%struct.list_head* %list) nounwind {
entry:
  %0 = icmp eq %struct.list_head* %list, null
  br i1 %0, label %bb2, label %bb

bb:
; CHECK: LBB0_2:
; CHECK: bne LBB0_2
; CHECK-NOT: b LBB0_2
; CHECK: bx lr
  %list_addr.05 = phi %struct.list_head* [ %2, %bb ], [ %list, %entry ]
  %next.04 = phi %struct.list_head* [ %list_addr.05, %bb ], [ null, %entry ]
  %1 = getelementptr inbounds %struct.list_head* %list_addr.05, i32 0, i32 0
  %2 = load %struct.list_head** %1, align 4
  store %struct.list_head* %next.04, %struct.list_head** %1, align 4
  %3 = icmp eq %struct.list_head* %2, null
  br i1 %3, label %bb2, label %bb

bb2:
  %next.0.lcssa = phi %struct.list_head* [ null, %entry ], [ %list_addr.05, %bb ]
  ret %struct.list_head* %next.0.lcssa
}
