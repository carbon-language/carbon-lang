; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%x = type int

implementation

int "foo"(int %in) 
begin
label: 
  ret int 2
end

