; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

implementation

declare int "bar"(int %in) 

int "foo"(int %blah)
begin
  %xx = call int %bar(int %blah)
  ret int %xx
end

