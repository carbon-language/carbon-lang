; RUN: llvm-as < %s | llc

	%Env = type opaque*

define void @.main(%Env) {
	call void @llvm.gcroot( %Env* null, %Env null )
	unreachable
}

declare void @llvm.gcroot(%Env*, %Env)
