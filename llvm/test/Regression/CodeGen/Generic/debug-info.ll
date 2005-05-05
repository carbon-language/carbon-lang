; RUN: llvm-as < %s | llc
%lldb.compile_unit = type { uint, ushort, ushort, sbyte*, sbyte*, sbyte*, {  }* }
%d.compile_unit7 = external global %lldb.compile_unit		; <%lldb.compile_unit*> [#uses=1]

implementation   ; Functions:

declare {  }* %llvm.dbg.stoppoint({  }*, uint, uint, %lldb.compile_unit*)

void %rb_raise(int, ...) {
entry:
	br bool false, label %strlen.exit, label %no_exit.i

no_exit.i:		; preds = %entry
	ret void

strlen.exit:		; preds = %entry
	%dbg.tmp.1.i2 = call {  }* %llvm.dbg.stoppoint( {  }* null, uint 4358, uint 0, %lldb.compile_unit* %d.compile_unit7 )		; <{  }*> [#uses=0]
	unreachable
}
