; RUN: llvm-as -f %s -o - | llc

;; Date:     May 28, 2003.
;; From:     test/Programs/SingleSource/richards_benchmark.c
;; Function: struct task *handlerfn(struct packet *pkt)
;;
;; Error:    PreSelection puts the arguments of the Phi just before
;;           the Phi instead of in predecessor blocks.  This later
;;           causes llc to produces an invalid register <NULL VALUE>
;;           for the phi arguments.

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
