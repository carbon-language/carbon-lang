; RUN: llvm-as < %s | opt -preheaders

implementation

; This function should get a preheader inserted before BB3, that is jumped
; to by BB1 & BB2
;
void "test"()
begin
	br bool true, label %BB1, label %BB2
BB1:    br label %BB3
BB2:    br label %BB3


BB3:
	br label %BB3
end
