; RUN: llvm-as < %s | llc -march=ppc32 | grep nop

define void @bork() noreturn nounwind  {
entry:
        unreachable
}
