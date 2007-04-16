; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   opt -predsimplify -instcombine -simplifycfg | llvm-dis > %t
; RUN: grep -v declare %t | not grep fail 
; RUN: grep -v declare %t | grep -c pass | grep 4

void %test1(int %x) {
entry:
  %A = seteq int %x, 0
  br bool %A, label %then.1, label %else.1
then.1:
  %B = seteq int %x, 1
  br bool %B, label %then.2, label %else.1
then.2:
  call void (...)* %fail( )
  ret void
else.1:
  ret void
}

void %test2(int %x) {
entry:
  %A = seteq int %x, 0
  %B = seteq int %x, 1
  br bool %A, label %then.1, label %else.1
then.1:
  br bool %B, label %then.2, label %else.1
then.2:
  call void (...)* %fail( )
  ret void
else.1:
  ret void
}

void %test3(int %x) {
entry:
  %A = seteq int %x, 0
  %B = seteq int %x, 1
  br bool %A, label %then.1, label %else.1
then.1:
  br bool %B, label %then.2, label %else.1
then.2:
  call void (...)* %fail( )
  ret void
else.1:
  ret void
}

void %test4(int %x, int %y) {
entry:
  %A = seteq int %x, 0
  %B = seteq int %y, 0
  %C = and bool %A, %B
  br bool %C, label %then.1, label %else.1
then.1:
  %D = seteq int %x, 0
  br bool %D, label %then.2, label %else.2
then.2:
  %E = seteq int %y, 0
  br bool %E, label %else.1, label %else.2
else.1:
  ret void
else.2:
  call void (...)* %fail( )
  ret void
}

void %test5(int %x) {
entry:
  %A = seteq int %x, 0
  br bool %A, label %then.1, label %else.1
then.1:
  ret void
then.2:
  call void (...)* %fail( )
  ret void
else.1:
  %B = seteq int %x, 0
  br bool %B, label %then.2, label %then.1
}

void %test6(int %x, int %y) {
entry:
  %A = seteq int %x, 0
  %B = seteq int %y, 0
  %C = or bool %A, %B
  br bool %C, label %then.1, label %else.1
then.1:
  ret void
then.2:
  call void (...)* %fail( )
  ret void
else.1:
  %D = seteq int %x, 0
  br bool %D, label %then.2, label %else.2
else.2:
  %E = setne int %y, 0
  br bool %E, label %then.1, label %then.2
}

void %test7(int %x) {
entry:
  %A = setne int %x, 0
  %B = xor bool %A, true
  br bool %B, label %then.1, label %else.1
then.1:
  %C = seteq int %x, 1
  br bool %C, label %then.2, label %else.1
then.2:
  call void (...)* %fail( )
  ret void
else.1:
  ret void
}

void %test8(int %x) {
entry:
  %A = add int %x, 1
  %B = seteq int %x, 0
  br bool %B, label %then.1, label %then.2
then.1:
  %C = seteq int %A, 1
  br bool %C, label %then.2, label %else.2
then.2:
  ret void
else.2:
  call void (...)* %fail( )
  ret void
}

void %test9(int %y, int %z) {
entry:
  %x = add int %y, %z
  %A = seteq int %y, 3
  %B = seteq int %z, 5
  %C = and bool %A, %B
  br bool %C, label %cond_true, label %return

cond_true:
  %D = seteq int %x, 8
  br bool %D, label %then, label %oops

then:
  call void (...)* %pass( )
  ret void

oops:
  call void (...)* %fail( )
  ret void

return:
  ret void
}

void %test10()  {
entry:
  %A = alloca int
  %B = seteq int* %A, null
  br bool %B, label %cond_true, label %cond_false

cond_true:
  call void (...)* %fail ( )
  ret void

cond_false:
  call void (...)* %pass ( )
  ret void
}

void %switch1(int %x) {
entry:
  %A = seteq int %x, 10
  br bool %A, label %return, label %cond_false

cond_false:
  switch int %x, label %return [
    int 9, label %then1
    int 10, label %then2
  ]

then1:
  call void (...)* %pass( )
  ret void

then2:
  call void (...)* %fail( )
  ret void

return:
  ret void
}

void %switch2(int %x) {
entry:
  %A = seteq int %x, 10
  br bool %A, label %return, label %cond_false

cond_false:
  switch int %x, label %return [
    int 8, label %then1
    int 9, label %then1
    int 10, label %then1
  ]

then1:
  %B = setne int %x, 8
  br bool %B, label %then2, label %return

then2:
  call void (...)* %pass( )
  ret void

return:
  ret void
}

void %switch3(int %x) {
entry:
  %A = seteq int %x, 10
  br bool %A, label %return, label %cond_false

cond_false:
  switch int %x, label %return [
    int 9, label %then1
    int 10, label %then1
  ]

then1:
  %B = seteq int %x, 9
  br bool %B, label %return, label %oops

oops:
  call void (...)* %fail( )
  ret void

return:
  ret void
}

void %switch4(int %x) {
entry:
  %A = seteq int %x, 10
  br bool %A, label %then1, label %cond_false

cond_false:
  switch int %x, label %default [
    int 9, label %then1
    int 10, label %then2
  ]

then1:
  ret void

then2:
  ret void

default:
  %B = seteq int %x, 9
  br bool %B, label %oops, label %then1

oops:
  call void (...)* %fail( )
  ret void
}

void %select1(int %x) {
entry:
  %A = seteq int %x, 10
  %B = select bool %A, int 1, int 2
  %C = seteq int %B, 1
  br bool %C, label %then, label %else

then:
  br bool %A, label %return, label %oops

else:
  br bool %A, label %oops, label %return

oops:
  call void (...)* %fail( )
  ret void

return:
  ret void
}

void %select2(int %x) {
entry:
  %A = seteq int %x, 10
  %B = select bool %A, int 1, int 2
  %C = seteq int %B, 1
  br bool %A, label %then, label %else

then:
  br bool %C, label %return, label %oops

else:
  br bool %C, label %oops, label %return

oops:
  call void (...)* %fail( )
  ret void

return:
  ret void
}

declare void %fail(...)

declare void %pass(...)
