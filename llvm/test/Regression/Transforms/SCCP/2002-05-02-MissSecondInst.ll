; RUN: if as < %s | opt -sccp | dis | grep sub
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

void "test3"(int, int)
begin
	add int 0, 0
        sub int 0, 4
        ret void
end

