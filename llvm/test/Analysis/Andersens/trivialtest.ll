; RUN: opt < %s -anders-aa -disable-output

define void @foo() { ret void }
