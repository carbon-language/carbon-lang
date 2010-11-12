; RUN: llc < %s PR8287: SelectionDag scheduling time. 
; Yes, some front end really produces this code. But that is a
; separate bug. This is more an example than a real test, because I
; don't know how give llvm-lit a timeout.

define void @foo([4096 x i8]* %arg1, [4096 x i8]* %arg2) {
  %buffer = alloca [4096 x i8]
  %pbuf = alloca [4096 x i8]*
  store [4096 x i8]* %buffer, [4096 x i8]** %pbuf

  %parg1 = alloca [4096 x i8]*
  store [4096 x i8]* %arg1, [4096 x i8]** %parg1

  %parg2 = alloca [4096 x i8]*
  store [4096 x i8]* %arg2, [4096 x i8]** %parg2

  ; The original test case has intermediate blocks.
  ; Presumably something fills in "buffer".

  %bufferCopy1 = load [4096 x i8]** %pbuf
  %dataCopy1 = load [4096 x i8]* %bufferCopy1
  %arg1Copy = load [4096 x i8]** %parg1
  store [4096 x i8] %dataCopy1, [4096 x i8]* %arg1Copy

  %bufferCopy2 = load [4096 x i8]** %pbuf
  %dataCopy2 = load [4096 x i8]* %bufferCopy2
  %arg2Copy = load [4096 x i8]** %parg2
  store [4096 x i8] %dataCopy2, [4096 x i8]* %arg2Copy

  ret void
}
