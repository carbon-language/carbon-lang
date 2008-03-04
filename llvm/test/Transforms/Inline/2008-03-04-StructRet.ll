; RUN: llvm-as < %s | opt -inline -sretpromotion -disable-output
	%struct.Benchmark = type { i32 (...)** }
	%struct.Complex = type { double, double }
	%struct.ComplexBenchmark = type { %struct.Benchmark }

define void @_Zml7ComplexS_(%struct.Complex* sret  %agg.result, double %a.0, double %a.1, double %b.0, double %b.1) nounwind  {
entry:
	ret void
}

define void @_ZNK16ComplexBenchmark9oop_styleEv(%struct.ComplexBenchmark* %this) nounwind  {
entry:
	%tmp = alloca %struct.Complex		; <%struct.Complex*> [#uses=2]
	br label %bb31

bb:		; preds = %bb31
	call void @_Zml7ComplexS_( %struct.Complex* sret  %tmp, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00 ) nounwind 
	%tmp21 = getelementptr %struct.Complex* %tmp, i32 0, i32 1		; <double*> [#uses=0]
	br label %bb31

bb31:		; preds = %bb, %entry
	br i1 false, label %bb, label %return

return:		; preds = %bb31
	ret void
}
