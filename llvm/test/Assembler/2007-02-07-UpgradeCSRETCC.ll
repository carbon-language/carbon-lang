; For PR1187
; RUN: llvm-upgrade < %s > /dev/null

%mystruct = type { int, double }
%glob = global %mystruct { int 3, double 42.0 }
%fptr = external global void (i32)*

implementation

csretcc void %nada(%mystruct * %ptr, int %val) {
  ret void
}

int %main(int %argc, ubyte** %argv) {
  %astr = alloca %mystruct
  call void %nada(%mystruct* %astr, i32 7)
  %fptr = alloca void (%mystruct*, i32)*
  %f = load void (%mystruct*, i32)**%fptr
  call csretcc void %f(%mystruct* %astr, i32 7)
  ret int 0
}
