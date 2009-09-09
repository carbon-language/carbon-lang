; RUN: llc < %s -march=bfin -verify-machineinstrs

	type { i32, float, float, float, float, float, float, float, float, float, float }		; type %0
	%struct..s_segment_inf = type { float, i32, i16, i16, float, float, i32, float, float }

define i32 @main(i32 %argc.1, i8** %argv.1) {
entry:
	%tmp.218 = load float* null		; <float> [#uses=1]
	%tmp.219 = getelementptr %0* null, i64 0, i32 6		; <float*> [#uses=1]
	%tmp.220 = load float* %tmp.219		; <float> [#uses=1]
	%tmp.221 = getelementptr %0* null, i64 0, i32 7		; <float*> [#uses=1]
	%tmp.222 = load float* %tmp.221		; <float> [#uses=1]
	%tmp.223 = getelementptr %0* null, i64 0, i32 8		; <float*> [#uses=1]
	%tmp.224 = load float* %tmp.223		; <float> [#uses=1]
	%tmp.225 = getelementptr %0* null, i64 0, i32 9		; <float*> [#uses=1]
	%tmp.226 = load float* %tmp.225		; <float> [#uses=1]
	%tmp.227 = getelementptr %0* null, i64 0, i32 10		; <float*> [#uses=1]
	%tmp.228 = load float* %tmp.227		; <float> [#uses=1]
	call void @place_and_route(i32 0, i32 0, float 0.000000e+00, i32 0, i32 0, i8* null, i32 0, i32 0, i8* null, i8* null, i8* null, i8* null, i32 0, i32 0, i32 0, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, i32 0, i32 0, i32 0, i32 0, i32 0, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, i32 0, i32 0, i16 0, i16 0, i16 0, float 0.000000e+00, float 0.000000e+00, %struct..s_segment_inf* null, i32 0, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float %tmp.218, float %tmp.220, float %tmp.222, float %tmp.224, float %tmp.226, float %tmp.228)
	ret i32 0
}

declare void @place_and_route(i32, i32, float, i32, i32, i8*, i32, i32, i8*, i8*, i8*, i8*, i32, i32, i32, float, float, float, float, float, float, float, float, float, i32, i32, i32, i32, i32, float, float, float, i32, i32, i16, i16, i16, float, float, %struct..s_segment_inf*, i32, float, float, float, float, float, float, float, float, float, float)
