; RUN: opt -S -simplifycfg < %s | FileCheck %s

; SimplifyCFG should eliminate redundant indirectbr edges.

; CHECK: indbrtest0
; CHECK: indirectbr i8* %t, [label %BB0, label %BB1, label %BB2]
; CHECK: %x = phi i32 [ 0, %BB0 ], [ 1, %entry ]

declare void @foo()
declare void @A()
declare void @B(i32)
declare void @C()

define void @indbrtest0(i8** %P, i8** %Q) {
entry:
  store i8* blockaddress(@indbrtest0, %BB0), i8** %P
  store i8* blockaddress(@indbrtest0, %BB1), i8** %P
  store i8* blockaddress(@indbrtest0, %BB2), i8** %P
  call void @foo()
  %t = load i8** %Q
  indirectbr i8* %t, [label %BB0, label %BB1, label %BB2, label %BB0, label %BB1, label %BB2]
BB0:
  call void @A()
  br label %BB1
BB1:
  %x = phi i32 [ 0, %BB0 ], [ 1, %entry ], [ 1, %entry ]
  call void @B(i32 %x)
  ret void
BB2:
  call void @C()
  ret void
}

; SimplifyCFG should convert the indirectbr into a directbr. It would be even
; better if it removed the branch altogether, but simplifycfdg currently misses
; that because the predecessor is the entry block.

; CHECK: indbrtest1
; CHECK: br label %BB0

define void @indbrtest1(i8** %P, i8** %Q) {
entry:
  store i8* blockaddress(@indbrtest1, %BB0), i8** %P
  call void @foo()
  %t = load i8** %Q
  indirectbr i8* %t, [label %BB0, label %BB0]
BB0:
  call void @A()
  ret void
}

