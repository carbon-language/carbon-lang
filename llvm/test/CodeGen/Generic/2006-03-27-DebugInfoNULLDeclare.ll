; RUN: llvm-as < %s | llc

declare void @llvm.dbg.declare({  }*, {  }*)

define void @foo() {
        call void @llvm.dbg.declare( {  }* null, {  }* null )
        ret void
}

