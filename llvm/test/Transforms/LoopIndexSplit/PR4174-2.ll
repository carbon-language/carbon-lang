; RUN: llvm-as < %s | opt -loop-index-split | llvm-dis | not grep clone

declare void @f()

define fastcc i32 @main() nounwind {
entry:
        br label %bb1552

bb1552:
        %j295.0.reg2mem.0 = phi i32 [ %storemerge110, %bb1669 ], [ 0, %entry ]
        br label %bb1553

bb1553:
        call void @f()
        %tmp1628 = icmp sgt i32 %j295.0.reg2mem.0, -3
        br i1 %tmp1628, label %bb1588, label %bb1616

bb1588:
        br label %bb1616

bb1616:
        %tmp1629 = icmp sgt i32 %j295.0.reg2mem.0, -3
        br i1 %tmp1629, label %bb1649, label %bb1632

bb1632:
        br label %bb1669

bb1649:
        br label %bb1669

bb1669:
        %storemerge110 = add i32 %j295.0.reg2mem.0, 1
        %tmp1672 = icmp sgt i32 %storemerge110, 3
        br i1 %tmp1672, label %bb1678, label %bb1552

bb1678:
        ret i32 0
}
