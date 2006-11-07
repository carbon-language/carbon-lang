;test all the shifted and signextending adds and subs with and without consts

; RUN: llvm-as < %s | llc -march=alpha | grep '	addl' |wc -l |grep 2 &&
; RUN: llvm-as < %s | llc -march=alpha | grep '	addq' |wc -l |grep 2 &&
; RUN: llvm-as < %s | llc -march=alpha | grep '	subl' |wc -l |grep 2 &&
; RUN: llvm-as < %s | llc -march=alpha | grep '	subq' |wc -l |grep 1 &&
; RUN: llvm-as < %s | llc -march=alpha | grep 'lda $0,-100($16)' |wc -l |grep 1 &&

; RUN: llvm-as < %s | llc -march=alpha | grep 's4addl' |wc -l |grep 2 &&
; RUN: llvm-as < %s | llc -march=alpha | grep 's8addl' |wc -l |grep 2 &&
; RUN: llvm-as < %s | llc -march=alpha | grep 's4addq' |wc -l |grep 2 &&
; RUN: llvm-as < %s | llc -march=alpha | grep 's8addq' |wc -l |grep 2 &&

; RUN: llvm-as < %s | llc -march=alpha | grep 's4subl' |wc -l |grep 2 &&
; RUN: llvm-as < %s | llc -march=alpha | grep 's8subl' |wc -l |grep 2 &&
; RUN: llvm-as < %s | llc -march=alpha | grep 's4subq' |wc -l |grep 2 &&
; RUN: llvm-as < %s | llc -march=alpha | grep 's8subq' |wc -l |grep 2

implementation   ; Functions:

int %al(int %x, int %y) {
entry:
        %tmp.3 = add int %y, %x
        ret int %tmp.3
}

int %ali(int %x) {
entry:
        %tmp.3 = add int 100, %x
        ret int %tmp.3
}

long %aq(long %x, long %y) {
entry:
        %tmp.3 = add long %y, %x
        ret long %tmp.3
}
long %aqi(long %x) {
entry:
        %tmp.3 = add long 100, %x
        ret long %tmp.3
}

int %sl(int %x, int %y) {
entry:
        %tmp.3 = sub int %y, %x
        ret int %tmp.3
}

int %sli(int %x) {
entry:
        %tmp.3 = sub int %x, 100
        ret int %tmp.3
}

long %sq(long %x, long %y) {
entry:
        %tmp.3 = sub long %y, %x
        ret long %tmp.3
}
long %sqi(long %x) {
entry:
        %tmp.3 = sub long %x, 100
        ret long %tmp.3
}



int %a4l(int %x, int %y) {
entry:
        %tmp.1 = shl int %y, ubyte 2
        %tmp.3 = add int %tmp.1, %x
        ret int %tmp.3
}

int %a8l(int %x, int %y) {
entry:
        %tmp.1 = shl int %y, ubyte 3
        %tmp.3 = add int %tmp.1, %x
        ret int %tmp.3
}

long %a4q(long %x, long %y) {
entry:
        %tmp.1 = shl long %y, ubyte 2
        %tmp.3 = add long %tmp.1, %x
        ret long %tmp.3
}

long %a8q(long %x, long %y) {
entry:
        %tmp.1 = shl long %y, ubyte 3
        %tmp.3 = add long %tmp.1, %x
        ret long %tmp.3
}

int %a4li(int %y) {
entry:
        %tmp.1 = shl int %y, ubyte 2
        %tmp.3 = add int 100, %tmp.1
        ret int %tmp.3
}

int %a8li(int %y) {
entry:
        %tmp.1 = shl int %y, ubyte 3
        %tmp.3 = add int 100, %tmp.1
        ret int %tmp.3
}

long %a4qi(long %y) {
entry:
        %tmp.1 = shl long %y, ubyte 2
        %tmp.3 = add long 100, %tmp.1
        ret long %tmp.3
}

long %a8qi(long %y) {
entry:
        %tmp.1 = shl long %y, ubyte 3
        %tmp.3 = add long 100, %tmp.1
        ret long %tmp.3
}




int %s4l(int %x, int %y) {
entry:
        %tmp.1 = shl int %y, ubyte 2
        %tmp.3 = sub int %tmp.1, %x
        ret int %tmp.3
}

int %s8l(int %x, int %y) {
entry:
        %tmp.1 = shl int %y, ubyte 3
        %tmp.3 = sub int %tmp.1, %x
        ret int %tmp.3
}

long %s4q(long %x, long %y) {
entry:
        %tmp.1 = shl long %y, ubyte 2
        %tmp.3 = sub long %tmp.1, %x
        ret long %tmp.3
}

long %s8q(long %x, long %y) {
entry:
        %tmp.1 = shl long %y, ubyte 3
        %tmp.3 = sub long %tmp.1, %x
        ret long %tmp.3
}

int %s4li(int %y) {
entry:
        %tmp.1 = shl int %y, ubyte 2
        %tmp.3 = sub int %tmp.1, 100
        ret int %tmp.3
}

int %s8li(int %y) {
entry:
        %tmp.1 = shl int %y, ubyte 3
        %tmp.3 = sub int %tmp.1, 100
        ret int %tmp.3
}

long %s4qi(long %y) {
entry:
        %tmp.1 = shl long %y, ubyte 2
        %tmp.3 = sub long %tmp.1, 100
        ret long %tmp.3
}

long %s8qi(long %y) {
entry:
        %tmp.1 = shl long %y, ubyte 3
        %tmp.3 = sub long %tmp.1, 100
        ret long %tmp.3
}

