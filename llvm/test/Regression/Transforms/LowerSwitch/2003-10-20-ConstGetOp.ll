void %main() {
entry:		; No predecessors!
	switch uint 1, label %exit [
		 uint 0, label %label.0
     uint 1, label %label.1
	]

label.0:		; preds = %endif.1
	br label %exit

label.1:		; preds = %endif.1
  br label %exit

exit:
	ret void
}
