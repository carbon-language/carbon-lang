; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--

define i64 @test() {
        ret i64 undef
}
