; This test makes sure that these instructions are properly eliminated.
;

; RUN: if as < %s | opt -instcombine -die | dis | grep phi
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int "test1"(int %A, bool %b) {
BB0:    br bool %b, label %BB1, label %BB2
BB1:
        %B = phi int [%A, %BB0]     ; Combine away one argument PHI nodes
        ret int %B
BB2:
        ret int %A
}

