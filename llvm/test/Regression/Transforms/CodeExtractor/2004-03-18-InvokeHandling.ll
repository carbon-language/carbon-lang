; RUN: llvm-as < %s | opt -loop-extract -disable-output

implementation   ; Functions:

declare int %_IO_getc()

declare void %__errno_location()

void %yylex() {
entry:
	switch uint 0, label %label.126 [
		 uint 0, label %return
		 uint 61, label %combine
		 uint 33, label %combine
		 uint 94, label %combine
		 uint 37, label %combine
		 uint 47, label %combine
		 uint 42, label %combine
		 uint 62, label %combine
		 uint 60, label %combine
		 uint 58, label %combine
		 uint 124, label %combine
		 uint 38, label %combine
		 uint 45, label %combine
		 uint 43, label %combine
		 uint 34, label %string_constant
		 uint 39, label %char_constant
		 uint 46, label %loopexit.2
		 uint 57, label %loopexit.2
		 uint 56, label %loopexit.2
		 uint 55, label %loopexit.2
		 uint 54, label %loopexit.2
		 uint 53, label %loopexit.2
		 uint 52, label %loopexit.2
		 uint 51, label %loopexit.2
		 uint 50, label %loopexit.2
		 uint 49, label %loopexit.2
		 uint 48, label %loopexit.2
		 uint 95, label %letter
		 uint 122, label %letter
		 uint 121, label %letter
		 uint 120, label %letter
		 uint 119, label %letter
		 uint 118, label %letter
		 uint 117, label %letter
		 uint 116, label %letter
		 uint 115, label %letter
		 uint 114, label %letter
		 uint 113, label %letter
		 uint 112, label %letter
		 uint 111, label %letter
		 uint 110, label %letter
		 uint 109, label %letter
		 uint 108, label %letter
		 uint 107, label %letter
		 uint 106, label %letter
		 uint 105, label %letter
		 uint 104, label %letter
		 uint 103, label %letter
		 uint 102, label %letter
		 uint 101, label %letter
		 uint 100, label %letter
		 uint 99, label %letter
		 uint 98, label %letter
		 uint 97, label %letter
		 uint 90, label %letter
		 uint 89, label %letter
		 uint 88, label %letter
		 uint 87, label %letter
		 uint 86, label %letter
		 uint 85, label %letter
		 uint 84, label %letter
		 uint 83, label %letter
		 uint 82, label %letter
		 uint 81, label %letter
		 uint 80, label %letter
		 uint 79, label %letter
		 uint 78, label %letter
		 uint 77, label %letter
		 uint 75, label %letter
		 uint 74, label %letter
		 uint 73, label %letter
		 uint 72, label %letter
		 uint 71, label %letter
		 uint 70, label %letter
		 uint 69, label %letter
		 uint 68, label %letter
		 uint 67, label %letter
		 uint 66, label %letter
		 uint 65, label %letter
		 uint 64, label %label.13
		 uint 76, label %label.12
		 uint 36, label %label.11
		 uint 4294967295, label %label.10
	]

label.10:		; preds = %entry
	ret void

label.11:		; preds = %entry
	ret void

label.12:		; preds = %entry
	ret void

label.13:		; preds = %entry
	ret void

letter:		; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry
	ret void

loopexit.2:		; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry
	switch int 0, label %shortcirc_next.14 [
		 int 48, label %then.20
		 int 46, label %endif.38
	]

then.20:		; preds = %loopexit.2
	switch int 0, label %else.4 [
		 int 120, label %then.21
		 int 88, label %then.21
	]

then.21:		; preds = %then.20, %then.20
	ret void

else.4:		; preds = %then.20
	ret void

shortcirc_next.14:		; preds = %loopexit.2
	ret void

endif.38:		; preds = %loopexit.2
	br bool false, label %then.40, label %then.39

then.39:		; preds = %endif.38
	ret void

then.40:		; preds = %endif.38
	invoke void %__errno_location( )
			to label %switchexit.2 unwind label %LongJmpBlkPre

loopentry.6:		; preds = %endif.52
	switch uint 0, label %switchexit.2 [
		 uint 73, label %label.82
		 uint 105, label %label.82
		 uint 76, label %label.80
		 uint 108, label %label.80
		 uint 70, label %label.78
		 uint 102, label %label.78
	]

label.78:		; preds = %loopentry.6, %loopentry.6
	ret void

label.80:		; preds = %loopentry.6, %loopentry.6
	ret void

label.82:		; preds = %loopentry.6, %loopentry.6
	%c.0.15.5 = phi int [ %tmp.79417, %loopentry.6 ], [ %tmp.79417, %loopentry.6 ]		; <int> [#uses=0]
	ret void

switchexit.2:		; preds = %then.40, %loopentry.6
	br bool false, label %endif.51, label %loopexit.6

endif.51:		; preds = %switchexit.2
	br bool false, label %endif.52, label %then.52

then.52:		; preds = %endif.51
	ret void

endif.52:		; preds = %endif.51
	%tmp.79417 = invoke int %_IO_getc( )
			to label %loopentry.6 unwind label %LongJmpBlkPre		; <int> [#uses=2]

loopexit.6:		; preds = %switchexit.2
	ret void

char_constant:		; preds = %entry
	ret void

string_constant:		; preds = %entry
	ret void

combine:		; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry
	ret void

label.126:		; preds = %entry
	ret void

return:		; preds = %entry
	ret void

LongJmpBlkPre:		; preds = %then.40, %endif.52
	ret void
}
