; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=att
; PR834

target endian = little
target pointersize = 32
target triple = "i386-unknown-freebsd6.1"

	%llvm.dbg.anchor.type = type { uint, uint }
	%llvm.dbg.basictype.type = type { uint, {  }*, sbyte*, {  }*, uint, ulong, ulong, ulong, uint, uint }
	%llvm.dbg.compile_unit.type = type { uint, {  }*, uint, sbyte*, sbyte*, sbyte* }
	%llvm.dbg.global_variable.type = type { uint, {  }*, {  }*, sbyte*, sbyte*, {  }*, uint, {  }*, bool, bool, {  }* }
%x = global int 0		; <int*> [#uses=1]
%llvm.dbg.global_variable = internal constant %llvm.dbg.global_variable.type {
    uint 327732, 
    {  }* cast (%llvm.dbg.anchor.type* %llvm.dbg.global_variables to {  }*), 
    {  }* cast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*), 
    sbyte* getelementptr ([2 x sbyte]* %str, int 0, int 0), 
    sbyte* null, 
    {  }* cast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*), 
    uint 1, 
    {  }* cast (%llvm.dbg.basictype.type* %llvm.dbg.basictype to {  }*), 
    bool false, 
    bool true, 
    {  }* cast (int* %x to {  }*) }, section "llvm.metadata"		; <%llvm.dbg.global_variable.type*> [#uses=0]
%llvm.dbg.global_variables = linkonce constant %llvm.dbg.anchor.type { uint 327680, uint 52 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
%llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type {
    uint 327697, 
    {  }* cast (%llvm.dbg.anchor.type* %llvm.dbg.compile_units to {  }*), 
    uint 4, 
    sbyte* getelementptr ([10 x sbyte]* %str, int 0, int 0), 
    sbyte* getelementptr ([32 x sbyte]* %str, int 0, int 0), 
    sbyte* getelementptr ([45 x sbyte]* %str, int 0, int 0) }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
%llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { uint 327680, uint 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
%str = internal constant [10 x sbyte] c"testb.cpp\00", section "llvm.metadata"		; <[10 x sbyte]*> [#uses=1]
%str = internal constant [32 x sbyte] c"/Sources/Projects/DwarfTesting/\00", section "llvm.metadata"		; <[32 x sbyte]*> [#uses=1]
%str = internal constant [45 x sbyte] c"4.0.1 LLVM (Apple Computer, Inc. build 5400)\00", section "llvm.metadata"		; <[45 x sbyte]*> [#uses=1]
%str = internal constant [2 x sbyte] c"x\00", section "llvm.metadata"		; <[2 x sbyte]*> [#uses=1]
%llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type {
    uint 327716, 
    {  }* cast (%llvm.dbg.compile_unit.type* %llvm.dbg.compile_unit to {  }*), 
    sbyte* getelementptr ([4 x sbyte]* %str, int 0, int 0), 
    {  }* null, 
    uint 0, 
    ulong 32, 
    ulong 32, 
    ulong 0, 
    uint 0, 
    uint 5 }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
%str = internal constant [4 x sbyte] c"int\00", section "llvm.metadata"		; <[4 x sbyte]*> [#uses=1]

implementation   ; Functions:
