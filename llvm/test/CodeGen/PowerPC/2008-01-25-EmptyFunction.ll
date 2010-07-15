; RUN: llc < %s -march=ppc32 | grep trap
target triple = "powerpc-apple-darwin8"


define void @bork() noreturn nounwind  {
entry:
        unreachable
}
