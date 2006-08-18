; RUN: opt -analyze %s -domset -disable-verify
;
int %re_match_2() {
ENTRY:
	br label %loopexit.20
loopentry.20:
	br label %loopexit.20

loopexit.20:
	ret int 0

endif.46:   ; UNREACHABLE
	br label %loopentry.20

}
