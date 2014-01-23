; RUN: opt %s -argpromotion -S -o - | FileCheck %s
; PR14710

%pair = type { i32, i32 }

declare i8* @foo(%pair*)

define internal void @bar(%pair* byval %Data) {
; CHECK: define internal void @bar(i32 %Data.0, i32 %Data.1)
; CHECK: %Data = alloca %pair
; CHECK-NOT: tail
; CHECK: call i8* @foo(%pair* %Data)
  tail call i8* @foo(%pair* %Data)
  ret void
}

define void @zed(%pair* byval %Data) {
  call void @bar(%pair* byval %Data)
  ret void
}
