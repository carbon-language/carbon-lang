; Make sure that the constant propagator doesn't cause a sigfpe
;
; RUN: as < %s | opt -constprop
;

int "test"() {
        %R = div int -2147483648, -1
        ret int %R
}

int "test2"() {
        %R = rem int -2147483648, -1
        ret int %R
}

