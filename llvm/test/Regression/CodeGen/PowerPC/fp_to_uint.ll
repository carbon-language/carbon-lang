; RUN: llvm-as < %s | llc -march=ppc32 | grep fctiwz | wc -l | grep 1

implementation

ushort %foo(float %a) {
entry:
        %tmp.1 = cast float %a to ushort
        ret ushort %tmp.1
}
