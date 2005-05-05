; RUN: llvm-as < %s | opt -lowersetjmp
	%lldb.compile_unit = type { uint, ushort, ushort, sbyte*, sbyte*, sbyte*, {  }* }
%d.compile_unit = external global %lldb.compile_unit		; <%lldb.compile_unit*> [#uses=1]

implementation   ; Functions:

declare {  }* %llvm.dbg.stoppoint({  }*, uint, uint, %lldb.compile_unit*)

declare void %llvm.longjmp(int*, int)

void %rb_iterate() {
entry:
	br bool false, label %then.1, label %else.0

then.1:		; preds = %entry
	call void %llvm.longjmp( int* null, int 0 )
	%dbg.tmp.57 = call {  }* %llvm.dbg.stoppoint( {  }* null, uint 5056, uint 0, %lldb.compile_unit* %d.compile_unit )		; <{  }*> [#uses=1]
	store {  }* %dbg.tmp.57, {  }** null
	ret void

else.0:		; preds = %entry
	ret void
}
