
%X = global int 7
%msg = internal global [13 x sbyte] c"Hello World\0A\00"


implementation

declare void %printf([13 x sbyte]*)

void %bar() {
  call void %printf([13 x sbyte]* %msg)
  ret void 
}

void %main() {
        call void %bar()
        ret void
}

