; RUN: llvm-as < %s | opt -predsimplify -instcombine -simplifycfg | llvm-dis | grep -v declare | not grep fail

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


declare void %fail(...)

declare void %pass(...)
