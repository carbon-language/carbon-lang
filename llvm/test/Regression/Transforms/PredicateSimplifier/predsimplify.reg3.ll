; RUN: llvm-as < %s | opt -predsimplify -simplifycfg | llvm-dis | grep pass

void %regtest(int %x) {
entry:
  %A = seteq int %x, 0
  br bool %A, label %middle, label %after
middle:
  br label %after
after:
  %B = seteq int %x, 0
  br bool %B, label %then, label %else
then:
  br label %end
else:
  call void (...)* %pass( )
  br label %end
end:
  ret void
}

declare void %pass(...)
