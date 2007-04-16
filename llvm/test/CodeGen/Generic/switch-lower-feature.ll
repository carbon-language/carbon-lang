; RUN: llvm-as < %s | llc -march=x86 -o - | grep \$7 | wc -l | grep 1
; RUN: llvm-as < %s | llc -march=x86 -o - | grep \$6 | wc -l | grep 1
; RUN: llvm-as < %s | llc -march=x86 -o - | grep 1024 | wc -l | grep 1
; RUN: llvm-as < %s | llc -march=x86 -o - | grep jb | wc -l | grep 2
; RUN: llvm-as < %s | llc -march=x86 -o - | grep je | wc -l | grep 1

define i32 @main(i32 %tmp158) {
entry:
        switch i32 %tmp158, label %bb336 [
	         i32 120, label %bb338
	         i32 121, label %bb338
                 i32 122, label %bb338
                 i32 123, label %bb338
                 i32 124, label %bb338
                 i32 125, label %bb338
                 i32 126, label %bb338
		 i32 1024, label %bb338
                 i32 0, label %bb338
                 i32 1, label %bb338
                 i32 2, label %bb338
                 i32 3, label %bb338
                 i32 4, label %bb338
		 i32 5, label %bb338
        ]
bb336:
  ret i32 10
bb338:
  ret i32 11
}
