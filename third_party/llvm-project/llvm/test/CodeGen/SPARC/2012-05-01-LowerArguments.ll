; Just check that this doesn't crash:
; RUN: llc < %s
; PR2960

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f128:128:128"
target triple = "sparc-unknown-linux-gnu"
	%"5tango4core9Exception11IOException" = type { [5 x i8*]*, i8*, { i64, i8* }, { i64, i8* }, i64, %"6Object7Monitor"*, %"5tango4core9Exception11IOException"* }
	%"6Object7Monitor" = type { [3 x i8*]*, i8* }

define fastcc %"5tango4core9Exception11IOException"* @_D5tango4core9Exception13TextException5_ctorMFAaZC5tango4core9Exception13TextException(%"5tango4core9Exception11IOException"* %this, { i64, i8* } %msg) {
entry_tango.core.Exception.TextException.this:
	unreachable
}
