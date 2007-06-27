; RUN:  llvm-as < %s |  opt -domtree -gcse -domtree -constmerge -disable-output

define i32 @test1() {
       unreachable
}
