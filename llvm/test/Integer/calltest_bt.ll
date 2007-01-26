; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%FunTy = type i28(i28)

declare i28 @"test"(...)   ; Test differences of prototype
declare i28 @"test"()      ; Differ only by vararg

implementation

define void @"invoke"(%FunTy *%x) {
	%foo = call %FunTy* %x(i28 123)
	%foo2 = tail call %FunTy* %x(i28 123)
	ret void
}

define i28 @"main"(i28 %argc)   ; TODO: , sbyte **argv, sbyte **envp)
begin
        %retval = call i28 (i28) *@test(i28 %argc)
        %two    = add i28 %retval, %retval
	%retval2 = invoke i28 @test(i28 %argc)
		   to label %Next unwind label %Error
Next:
	%two2 = add i28 %two, %retval2
	call void @invoke (%FunTy* @test)
        ret i28 %two2
Error:
	ret i28 -1
end

define i28 @"test"(i28 %i0) {
    ret i28 %i0
}
