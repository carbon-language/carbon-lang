; RUN: as < %s | opt -instcombine -die | dis | grep call | not grep cast

implementation

; Simple case, argument translatable without changing the value
declare void %test1a(sbyte *%A) 
void %test1(int *%A) {
        call void(int*)* cast (void(sbyte*)* %test1a to void(int*)*)(int* %A)
        ret void
}

; More complex case, translate argument because of resolution.  This is safe 
; because we have the body of the function
void %test2a(sbyte %A) { ret void }
int %test2(int %A) {
	call void(int)* cast (void(sbyte)* %test2a to void(int)*)(int %A)
	ret int %A
}

; Resolving this should insert a cast from sbyte to int, following the C 
; promotion rules.
declare void %test3a(sbyte %A, ...)
void %test3(sbyte %A, sbyte %B) {
	call void(sbyte, sbyte)* cast (void(sbyte,...)* %test3a to void(sbyte,sbyte)*)(sbyte %A, sbyte %B)
        ret void
}

; test conversion of return value...
sbyte %test4a() { ret sbyte 0 }
int %test4() {
	%X = call int()* cast (sbyte()* %test4a to int()*)()
        ret int %X
}

; test conversion of return value... no value conversion occurs so we can do 
; this with just a prototype...
declare uint %test5a()
int %test5() {
	%X = call int()* cast (uint()* %test5a to int()*)()
        ret int %X
}

; test addition of new arguments...
declare int %test6a(int %X)
int %test6() {
	%X = call int()* cast (int(int)* %test6a to int()*)()
        ret int %X
}

; test removal of arguments, only can happen with a function body
void %test7a() { ret void } 
void %test7() {
	call void(int)* cast (void()* %test7a to void(int)*)(int 5)
        ret void
}

