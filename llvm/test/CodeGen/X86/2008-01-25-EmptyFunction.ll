; RUN: llvm-as < %s | llc -march=x86 | grep nop

define void @bork() noreturn nounwind  {
entry:
        unreachable
}
