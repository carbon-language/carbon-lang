; RUN: llc -O2 -print-after-all < %s 2>@1

define void @tester(){
  ret void
}

