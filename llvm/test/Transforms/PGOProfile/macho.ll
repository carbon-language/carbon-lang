; RUN: opt < %s -mtriple=x86_64-apple-macosx -pgo-instr-gen -instrprof -S | llc | FileCheck %s --check-prefix=MACHO-DIRECTIVE

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; MACHO-DIRECTIVE: .weak_definition        ___llvm_profile_raw_version
define i32 @test_macho(i32 %i) {
entry:
  ret i32 %i
}
