; RUN: llvm-as < %s | opt -anders-aa -disable-output

void %foo() { ret void }
