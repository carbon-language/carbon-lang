; RUN:  llvm-as < %s |  opt -domtree -idom -gcse -etforest -constmerge -disable-output

define i32 @test1() {
       unreachable
}
