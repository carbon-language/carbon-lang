; RUN: llvm-upgrade < %s | llvm-as | llc -march=c

%JNIEnv = type %struct.JNINa*
%struct.JNINa = type { sbyte*, sbyte*, sbyte*, void (%JNIEnv*)* }

