.386
.model flat

;; val1 should be linked normally. no_such_symbol1 should be ignored.
extern _no_such_symbol1 : PROC
extern _val1 (_no_such_symbol1): PROC

;; no_such_symbol2 should be linked as val2.
extern _val2 : PROC
extern _no_such_symbol2 (_val2) : PROC

;; no_such_symbol3 should fail to link.
extern _no_such_symbol3 : PROC

public _fn1
.code
_fn1:
	push	ebp
	mov	ebp, esp
	call	_val1
	call	_no_such_symbol2
	call	_no_such_symbol3
	pop	ebp
	ret	0
end _fn1
