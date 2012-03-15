; RUN: opt -disable-output -objc-arc-contract %s
; test that we don't crash on unreachable code
%2 = type opaque

define void @_i_Test__foo(%2 *%x) {
entry:
  unreachable

return:                                           ; No predecessors!
  %bar = bitcast %2* %x to i8*
  %foo = call i8* @objc_autoreleaseReturnValue(i8* %bar) nounwind
  ret void
}

declare i8* @objc_autoreleaseReturnValue(i8*)
