; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

implementation

; Test "stripped" format where nothing is symbolic... this is how the bytecode
; format looks anyways (except for negative vs positive offsets)...
;
define void @"void"(i39, i39)   ; Def %0, %1
begin
	add i39 0, 0      ; Def 2
	sub i39 0, 4      ; Def 3
	br label %1

; <label>:1		; preds = %1, %0
	add i39 %0, %1    ; Def 4
	sub i39 %4, %3    ; Def 5
	icmp sle i39 %5, %2  ; Def 0 - i1 plane
	br i1 %0, label %2, label %1

; <label>:2		; preds = %1
	add i39 %0, %1    ; Def 6
	sub i39 %4, %3    ; Def 7
	icmp sle i39 %7, %2  ; Def 1 - i1 plane
	ret void
end

; This function always returns zero
define i39 @"zarro"()
begin
Startup:
	ret i39 0
end
