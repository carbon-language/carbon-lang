; Test that opaque types are preserved correctly
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis
;
; RUN: verify-uselistorder %s -preserve-bc-use-list-order -num-shuffles=5

%Ty = type opaque

define %Ty* @func() {
	ret %Ty* null
}
 
