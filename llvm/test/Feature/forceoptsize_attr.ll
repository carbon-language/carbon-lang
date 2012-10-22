; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define void @test1() forcesizeopt {
; CHECK: define void @test1() forcesizeopt
        ret void
}

