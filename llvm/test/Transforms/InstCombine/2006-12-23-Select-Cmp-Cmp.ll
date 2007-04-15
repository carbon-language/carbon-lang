; For PR1065. This causes an assertion in instcombine if a select with two cmp
; operands is encountered.
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine -disable-output
; END.

; ModuleID = 'PR1065.bc'
target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"
	%struct.internal_state = type { int }
	%struct.mng_data = type { uint, sbyte*, uint, uint, uint, uint, uint, uint, uint, uint, uint, ubyte, uint, uint, uint, sbyte, uint, uint, uint, uint, ushort, ushort, ushort, sbyte, sbyte, double, double, double, sbyte, sbyte, sbyte, sbyte, uint, uint, uint, uint, int, sbyte, int, int, sbyte*, sbyte* (uint)*, void (sbyte*, uint)*, void (sbyte*, sbyte*, uint)*, sbyte (%struct.mng_data*)*, sbyte (%struct.mng_data*)*, sbyte (%struct.mng_data*, sbyte*, uint, uint*)*, sbyte (%struct.mng_data*, sbyte*, uint, uint*)*, sbyte (%struct.mng_data*, int, sbyte, int, uint, int, int, sbyte*)*, sbyte (%struct.mng_data*, int, int, sbyte*)*, sbyte (%struct.mng_data*, uint, uint)*, sbyte (%struct.mng_data*, ubyte, sbyte*, sbyte*, sbyte*, sbyte*)*, sbyte (%struct.mng_data*)*, sbyte (%struct.mng_data*, sbyte*)*, sbyte (%struct.mng_data*, sbyte*)*, sbyte (%struct.mng_data*, uint, uint)*, sbyte (%struct.mng_data*, int, uint, sbyte*)*, sbyte (%struct.mng_data*, ubyte, ubyte, uint, uint)*, sbyte* (%struct.mng_data*, uint)*, sbyte* (%struct.mng_data*, uint)*, sbyte* (%struct.mng_data*, uint)*, sbyte (%struct.mng_data*, uint, uint, uint, uint)*, uint (%struct.mng_data*)*, sbyte (%struct.mng_data*, uint)*, sbyte (%struct.mng_data*, uint)*, sbyte (%struct.mng_data*, uint, uint, uint, uint, uint, uint, uint, uint)*, sbyte (%struct.mng_data*, ubyte)*, sbyte (%struct.mng_data*, uint, sbyte*)*, sbyte (%struct.mng_data*, uint, sbyte, sbyte*)*, sbyte, int, uint, sbyte*, sbyte*, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, uint, uint, ubyte, ubyte, ubyte, ubyte, ubyte, uint, sbyte, sbyte, sbyte, uint, ubyte*, uint, ubyte*, uint, sbyte, ubyte, sbyte, uint, ubyte*, ubyte*, uint, uint, ubyte*, ubyte*, %struct.mng_pushdata*, %struct.mng_pushdata*, %struct.mng_pushdata*, %struct.mng_pushdata*, sbyte, sbyte, int, uint, ubyte*, sbyte, sbyte, uint, uint, uint, uint, uint, uint, sbyte, sbyte, sbyte, sbyte, int, int, sbyte*, uint, uint, uint, sbyte, sbyte, uint, uint, uint, uint, sbyte, sbyte, ubyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, uint, sbyte*, sbyte*, sbyte*, uint, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, %struct.mng_savedata*, uint, uint, uint, uint, sbyte, int, int, int, int, int, int, int, int, int, int, int, int, uint, uint, uint, uint, ubyte*, ubyte*, ubyte*, sbyte, sbyte, int, int, int, int, int, int, int, int, int, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, [256 x ubyte], double, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, ushort, sbyte, ubyte, sbyte, ubyte, sbyte, int, int, sbyte, int, int, int, int, ushort, ushort, ushort, ubyte, ushort, ubyte, int, int, uint, uint, ubyte, uint, uint, sbyte, int, int, int, int, ubyte, uint, uint, sbyte, int, int, int, int, uint, sbyte, uint, ubyte, ushort, ushort, ushort, short, uint, [256 x %struct.mng_palette8e], uint, [256 x ubyte], uint, uint, uint, uint, uint, uint, uint, uint, uint, ubyte, uint, sbyte*, ushort, ushort, ushort, sbyte*, ubyte, ubyte, uint, uint, uint, uint, sbyte, void ()*, void ()*, void ()*, void ()*, void ()*, void ()*, sbyte*, ubyte, ubyte, ubyte, uint, sbyte*, sbyte*, ushort, ushort, ushort, ushort, int, int, sbyte*, %struct.z_stream, int, int, int, int, int, uint, sbyte, sbyte, [256 x uint], sbyte }
	%struct.mng_palette8e = type { ubyte, ubyte, ubyte }
	%struct.mng_pushdata = type { sbyte*, sbyte*, uint, sbyte, ubyte*, uint }
	%struct.mng_savedata = type { sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, ushort, ushort, ushort, ubyte, ushort, ubyte, ubyte, uint, uint, sbyte, int, int, int, int, uint, [256 x %struct.mng_palette8e], uint, [256 x ubyte], uint, uint, uint, uint, uint, uint, uint, uint, uint, ubyte, uint, sbyte*, ushort, ushort, ushort }
	%struct.z_stream = type { ubyte*, uint, uint, ubyte*, uint, uint, sbyte*, %struct.internal_state*, sbyte* (sbyte*, uint, uint)*, void (sbyte*, sbyte*)*, sbyte*, int, uint, uint }

implementation   ; Functions:

void %mng_write_basi() {
entry:
	%tmp = load ubyte* null		; <ubyte> [#uses=1]
	%tmp = icmp ugt ubyte %tmp, 8		; <bool> [#uses=1]
	%tmp = load ushort* null		; <ushort> [#uses=2]
	%tmp3 = icmp eq ushort %tmp, 255		; <bool> [#uses=1]
	%tmp7 = icmp eq ushort %tmp, -1		; <bool> [#uses=1]
	%bOpaque.0.in = select bool %tmp, bool %tmp7, bool %tmp3		; <bool> [#uses=1]
	br bool %bOpaque.0.in, label %cond_next90, label %bb95

cond_next90:		; preds = %entry
	ret void

bb95:		; preds = %entry
	ret void
}
