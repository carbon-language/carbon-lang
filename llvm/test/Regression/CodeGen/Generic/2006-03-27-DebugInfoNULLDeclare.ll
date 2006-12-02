; RUN: llvm-upgrade < %s | llvm-as | llc

implementation   ; Functions:

declare void %llvm.dbg.declare({  }*, {  }*)

void %foo() {
	call void %llvm.dbg.declare( {  }* null, {  }* null ) 
	ret void
}
