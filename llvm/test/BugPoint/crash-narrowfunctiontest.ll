; Test that bugpoint can narrow down the testcase to the important function
; FIXME: This likely fails on windows
;
; RUN: bugpoint -load %llvmlibsdir/BugpointPasses%shlibext %s -output-prefix %t -bugpoint-crashcalls -silence-passes > /dev/null
; XFAIL: mingw

define i32 @foo() { ret i32 1 }

define i32 @test() {
	call i32 @test()
	ret i32 %1
}

define i32 @bar() { ret i32 2 }
