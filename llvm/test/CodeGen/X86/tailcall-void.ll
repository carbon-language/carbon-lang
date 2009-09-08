; RUN: llc < %s -march=x86 -tailcallopt | grep TAILCALL
define fastcc void @i1test(i32, i32, i32, i32) {
  entry:
   tail call fastcc void @i1test( i32 %0, i32 %1, i32 %2, i32 %3)
   ret void 
}
