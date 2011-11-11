; RUN: llc < %s -march=ptx64 -mattr=+ptx20,+sm20 | FileCheck %s

declare i32 @printf(i8*, ...)

@str = private unnamed_addr constant [6 x i8] c"test\0A\00"

define ptx_device void @t1_printf() {
; CHECK: mov.u64 %rd{{[0-9]+}}, $L__str;
; CHECK: call.uni	(__localparam_{{[0-9]+}}), vprintf, (__localparam_{{[0-9]+}}, __localparam_{{[0-9]+}});
; CHECK: ret;
    %1 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([6 x i8]* @str, i64 0, i64 0))
	ret void
}

@str2 = private unnamed_addr constant [11 x i8] c"test = %f\0A\00"

define ptx_device void @t2_printf() {
; CHECK: .local .align 8 .b8 __local{{[0-9]+}}[{{[0-9]+}}];
; CHECK: mov.u64 %rd{{[0-9]+}}, $L__str2;
; CHECK: cvta.local.u64  %rd{{[0-9]+}}, __local{{[0-9+]}};
; CHECK: call.uni	(__localparam_{{[0-9]+}}), vprintf, (__localparam_{{[0-9]+}}, __localparam_{{[0-9]+}});
; CHECK: ret;
  %1 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @str2, i64 0, i64 0), double 0x3FF3333340000000)
  ret void
}
