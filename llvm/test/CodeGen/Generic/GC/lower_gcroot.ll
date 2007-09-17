; RUN: llvm-as < %s | llc

	%Env = type opaque*

define void @.main(%Env) {
	%Root = alloca %Env
	call void @llvm.gcroot( %Env* %Root, %Env null )
	unreachable
}

declare void @llvm.gcroot(%Env*, %Env)
