; RUN: as < %s | opt -sccp | dis | not grep sub

void %test3(int, int) {
	add int 0, 0
        sub int 0, 4
        ret void
}
