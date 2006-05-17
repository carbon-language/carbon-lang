; RUN: llvm-as -f %s -o %t.bc
; RUN: lli %t.bc > /dev/null


%X = global int 7
%msg = internal global [13 x sbyte] c"Hello World\0A\00"


implementation

declare void %printf([13 x sbyte]*,...)

void %bar() {
  call void([13 x sbyte]*,...)* %printf([13 x sbyte]* %msg)
  ret void 
}

int %main() {
        call void %bar()
        ret int 0
}

