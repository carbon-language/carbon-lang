; FIXME: A should just be SM
; RUN: analyze %s -datastructure-gc -dsgc-dspass=bu -dsgc-check-flags=A:SIM
; Constant globals should not mark stuff incomplete.  This should allow the 
; bu pass to resolve the indirect call immediately in "test", allowing %A to
; be marked complete and the store to happen.

; This is the common case for handling vtables aggressively.

%G = constant void (int*)* %foo 

implementation

void %foo(int *%X) {
  store int 0, int* %X
  ret void
}

void %test() {
  %Fp = load void (int*)** %G
  %A = alloca int
  call void %Fp(int* %A)
  ret void
}
