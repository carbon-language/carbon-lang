; RUN: llc < %s -march=x86 -disable-mmx |  FileCheck %s


define void @t(<2 x i64>* %dst, <2 x i64> %src1, <2 x i64> %src2) nounwind readonly {
; CHECK: andb
  %cmp1 = icmp ne <2 x i64> %src1, zeroinitializer
  %cmp2 = icmp ne <2 x i64> %src2, zeroinitializer
  %t1 = and <2 x i1> %cmp1, %cmp2
  %t2 = sext <2 x i1> %t1 to <2 x i64>
  store <2 x i64> %t2, <2 x i64>* %dst
  ret void
}

define void @t2(<3 x i64>* %dst, <3 x i64> %src1, <3 x i64> %src2) nounwind readonly {
; CHECK: andb
  %cmp1 = icmp ne <3 x i64> %src1, zeroinitializer
  %cmp2 = icmp ne <3 x i64> %src2, zeroinitializer
  %t1 = and <3 x i1> %cmp1, %cmp2
  %t2 = sext <3 x i1> %t1 to <3 x i64>
  store <3 x i64> %t2, <3 x i64>* %dst
  ret void
}
