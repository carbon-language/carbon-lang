; Ensure constant propogation of logical instructions is working correctly.

; RUN: if as < %s | opt -constprop -die | dis | ggrep -E 'and|or|xor'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi


int  "test1"() { %R = and int 4,1234          ret int  %R }
bool "test1"() { %R = and bool true, false    ret bool %R }

int  "test2"() { %R = or int 4,1234          ret int  %R }
bool "test2"() { %R = or bool true, false    ret bool %R }

int  "test3"() { %R = xor int 4,1234          ret int  %R }
bool "test3"() { %R = xor bool true, false    ret bool %R }
