	%lldb.compile_unit = type { uint, ushort, ushort, sbyte*, sbyte*, sbyte*, {  }* }
	%lldb.global = type { uint, %lldb.compile_unit*, sbyte*, {  }*, sbyte*, bool }
	%lldb.local = type { %lldb.global*, sbyte*, sbyte* }
%llvm.dbg.translation_units = linkonce global {  } { }		; <{  }*> [#uses=1]
%llvm.dbg.globals = linkonce global {  } { }		; <{  }*> [#uses=1]
%.str_1 = internal constant [11 x sbyte] c"funccall.c\00"		; <[11 x sbyte]*> [#uses=1]
%.str_2 = internal constant [12 x sbyte] c"/home/sabre\00"		; <[12 x sbyte]*> [#uses=1]
%.str_3 = internal constant [14 x sbyte] c"llvmgcc 3.4.x\00"		; <[14 x sbyte]*> [#uses=1]
%d.compile_unit = internal constant %lldb.compile_unit { uint 17, ushort 0, ushort 1, sbyte* getelementptr ([11 x sbyte]* %.str_1, long 0, long 0), sbyte* getelementptr ([12 x sbyte]* %.str_2, long 0, long 0), sbyte* getelementptr ([14 x sbyte]* %.str_3, long 0, long 0), {  }* %llvm.dbg.translation_units }		; <%lldb.compile_unit*> [#uses=9]
%.str_5 = internal global [5 x sbyte] c"main\00"		; <[5 x sbyte]*> [#uses=1]
%.str_6 = internal global [4 x sbyte] c"foo\00"		; <[4 x sbyte]*> [#uses=1]
%.str_7 = internal global [2 x sbyte] c"q\00"		; <[2 x sbyte]*> [#uses=1]
%d.main = global %lldb.global { uint 46, %lldb.compile_unit* %d.compile_unit, sbyte* getelementptr ([5 x sbyte]* %.str_5, long 0, long 0), {  }* %llvm.dbg.globals, sbyte* null, bool true }		; <%lldb.global*> [#uses=1]
%d.foo = global %lldb.global { uint 46, %lldb.compile_unit* %d.compile_unit, sbyte* getelementptr ([4 x sbyte]* %.str_6, long 0, long 0), {  }* %llvm.dbg.globals, sbyte* null, bool true }		; <%lldb.global*> [#uses=1]
%q = internal global int 0		; <int*> [#uses=7]
%d.q = internal global { %lldb.global, int* } { %lldb.global { uint 52, %lldb.compile_unit* %d.compile_unit, sbyte* getelementptr ([2 x sbyte]* %.str_7, long 0, long 0), {  }* %llvm.dbg.globals, sbyte* null, bool false }, int* %q }		; <{ %lldb.global, int* }*> [#uses=0]

implementation   ; Functions:

declare {  }* %llvm.dbg.stoppoint({  }*, uint, uint, %lldb.compile_unit*)

declare {  }* %llvm.dbg.func.start(%lldb.global*)

declare {  }* %llvm.dbg.region.start({  }*)

declare {  }* %llvm.dbg.region.end({  }*)

void %foo() {
	%t = alloca int		; <int*> [#uses=2]
	%.1 = call {  }* %llvm.dbg.func.start( %lldb.global* %d.foo )		; <{  }*> [#uses=1]
	%.2 = call {  }* %llvm.dbg.stoppoint( {  }* %.1, uint 5, uint 2, %lldb.compile_unit* %d.compile_unit )		; <{  }*> [#uses=1]
	%tmp.0 = load int* %q		; <int> [#uses=1]
	store int %tmp.0, int* %t
	%.3 = call {  }* %llvm.dbg.stoppoint( {  }* %.2, uint 6, uint 2, %lldb.compile_unit* %d.compile_unit )		; <{  }*> [#uses=1]
	%tmp.01 = load int* %t		; <int> [#uses=1]
	%tmp.1 = add int %tmp.01, 1		; <int> [#uses=1]
	store int %tmp.1, int* %q
	%.4 = call {  }* %llvm.dbg.stoppoint( {  }* %.3, uint 7, uint 1, %lldb.compile_unit* %d.compile_unit )		; <{  }*> [#uses=1]
	call {  }* %llvm.dbg.region.end( {  }* %.4 )		; <{  }*>:0 [#uses=0]
	ret void
}

int %main() {
entry:
	%.1 = call {  }* %llvm.dbg.func.start( %lldb.global* %d.main )		; <{  }*> [#uses=1]
	%result = alloca int		; <int*> [#uses=2]
	%.2 = call {  }* %llvm.dbg.stoppoint( {  }* %.1, uint 9, uint 2, %lldb.compile_unit* %d.compile_unit )		; <{  }*> [#uses=1]
	store int 0, int* %q
	%.3 = call {  }* %llvm.dbg.stoppoint( {  }* %.2, uint 10, uint 2, %lldb.compile_unit* %d.compile_unit )		; <{  }*> [#uses=1]
	call void %foo( )
	%.4 = call {  }* %llvm.dbg.stoppoint( {  }* %.3, uint 11, uint 2, %lldb.compile_unit* %d.compile_unit )		; <{  }*> [#uses=1]
	%tmp.2 = load int* %q		; <int> [#uses=1]
	%tmp.3 = sub int %tmp.2, 1		; <int> [#uses=1]
	store int %tmp.3, int* %q
	%.5 = call {  }* %llvm.dbg.stoppoint( {  }* %.4, uint 13, uint 2, %lldb.compile_unit* %d.compile_unit )		; <{  }*> [#uses=1]
	%tmp.4 = load int* %q		; <int> [#uses=1]
	store int %tmp.4, int* %result
	br label %return

after_ret:		; No predecessors!
	br label %return

return:		; preds = %entry, %after_ret
	%tmp.5 = load int* %result		; <int> [#uses=1]
	%.6 = call {  }* %llvm.dbg.stoppoint( {  }* %.5, uint 14, uint 1, %lldb.compile_unit* %d.compile_unit )		; <{  }*> [#uses=1]
	call {  }* %llvm.dbg.region.end( {  }* %.6 )		; <{  }*>:0 [#uses=0]
	ret int %tmp.5
}
