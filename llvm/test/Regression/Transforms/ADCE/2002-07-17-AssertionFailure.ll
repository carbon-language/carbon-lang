; This testcase fails because ADCE does not correctly delete the chain of 
; three instructions that are dead here.  Ironically there were a dead basic
; block in this function, it would work fine, but that would be the part we 
; have to fix now, wouldn't it....
;
; RUN: llvm-as < %s | opt -adce

void %foo(sbyte* %reg5481) {
        %cast611 = cast sbyte* %reg5481 to sbyte**              ; <sbyte**> [#uses=1]
        %reg162 = load sbyte** %cast611         ; <sbyte*> [#uses=0]
	cast sbyte*%reg162 to int
	ret void
}
