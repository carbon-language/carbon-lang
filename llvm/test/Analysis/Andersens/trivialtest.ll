; RUN: llvm-upgrade < %s | llvm-as | opt -anders-aa -disable-output

void %foo() { ret void }
