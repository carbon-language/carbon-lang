;; Date:     May 28, 2003.
;; From:     test/Programs/SingleSource/richards_benchmark.c
;; Function: struct task *handlerfn(struct packet *pkt)
;;
;; Error:    PreSelection puts the arguments of the Phi just before
;;           the Phi instead of in predecessor blocks.  This later
;;           causes llc to produces an invalid register <NULL VALUE>
;;           for the phi arguments.
;;
;; PreSelection Output:
;; *** LLVM code after pre-selection for function handlerfn:
;; 
;; 
;; %struct..task* %handlerfn(%struct..packet*) {
;; ; <label>:0             ; No predecessors!
;;         setne %struct..packet* %0, null         ; <bool>:0 [#uses=1]
;;         br bool %0, label %1, label %2
;; 
;; ; <label>:1             ; preds = %0
;;         br label %2
;; 
;; ; <label>:2             ; preds = %0, %1
;;         %addrOfGlobal = getelementptr int* %v2, long 0          ; <int*> [#uses=1]
;;         %addrOfGlobal1 = getelementptr int* %v1, long 0         ; <int*> [#uses=1]
;;         phi int* [ %addrOfGlobal, %1 ], [ %addrOfGlobal1, %0 ]          ; <int*>:0 [#uses=1]
;;         cast int* %0 to %struct..packet*                ; <%struct..packet*>:1 [#uses=1]
;;         call void %append( %struct..packet* %0, %struct..packet* %1 )
;;         ret %struct..task* null
;; }
;; llc: ../../../include/llvm/CodeGen/MachineInstr.h:294: int MachineOperand::getAllocatedRegNum() const: Assertion `hasAllocatedReg()' failed.
;; 


target endian = little
target pointersize = 32
	%struct..packet = type { %struct..packet*, int, int, int, [4 x sbyte] }
	%struct..task = type { %struct..task*, int, int, %struct..packet*, int, %struct..task* (%struct..packet*)*, int, int }
%v1 = external global int
%v2 = external global int

implementation   ; Functions:

%struct..task* %handlerfn(%struct..packet* %pkt.2) {
entry:		; No predecessors!
	%tmp.1 = setne %struct..packet* %pkt.2, null
	br bool %tmp.1, label %cond_false, label %cond_continue

cond_false:		; preds = %entry
	br label %cond_continue

cond_continue:		; preds = %entry, %cond_false
	%mem_tmp.0 = phi int* [ %v2, %cond_false ], [ %v1, %entry ]
	%tmp.12 = cast int* %mem_tmp.0 to %struct..packet*
	call void %append( %struct..packet* %pkt.2, %struct..packet* %tmp.12 )
	ret %struct..task* null
}

declare void %append(%struct..packet*, %struct..packet*)
