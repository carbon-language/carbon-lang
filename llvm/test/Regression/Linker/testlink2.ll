; This file is used by testlink1.ll, so it doesn't actually do anything itself
;
; RUN: echo

%MyVar     = global int 4
%MyIntList = external global { \2 *, int }

%AConst    = constant int 123

%Intern1   = internal constant int 52 ;; Intern in both testlink[12].ll
%Intern2   = constant int 12345       ;; Intern in one but not in other

%MyIntListPtr = constant { {\2,int}* } { {\2,int}* %MyIntList }
%MyVarPtr  = global { int * }  { int * %MyVar }

constant int 412

implementation

int "foo"(int %blah)
begin
	store int %blah, int *%MyVar
	%idx = getelementptr { \2 *, int } * %MyIntList, long 0, ubyte 1
	store int 12, int* %idx

	%ack = load int * %0   ;; Load from the unnamed constant
	%fzo = add int %ack, %blah
	ret int %fzo
end

declare void "unimp"(float, double)

internal void "testintern"() begin ret void end
         void "Testintern"() begin ret void end
internal void "testIntern"() begin ret void end

