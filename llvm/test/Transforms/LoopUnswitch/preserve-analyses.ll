; RUN: opt -loop-unswitch %s -disable-output

; Loop unswitch should be able to unswitch these loops and
; preserve LCSSA and LoopSimplify forms.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv6-apple-darwin9"

%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct._RuneCharClass = type { [14 x i8], i32 }
%struct._RuneEntry = type { i32, i32, i32, i32* }
%struct._RuneLocale = type { [8 x i8], [32 x i8], i32 (i8*, i32, i8**)*, i32 (i32, i8*, i32, i8**)*, i32, [256 x i32], [256 x i32], [256 x i32], %struct._RuneRange, %struct._RuneRange, %struct._RuneRange, i8*, i32, i32, %struct._RuneCharClass* }
%struct._RuneRange = type { i32, %struct._RuneEntry* }
%struct.__sFILEX = type opaque
%struct.__sbuf = type { i8*, i32 }
%struct.colstr = type { i8*, i8* }
%struct.optstr = type { i8*, i32* }

@expflg = external global i32                     ; <i32*> [#uses=0]
@ctrflg = external global i32                     ; <i32*> [#uses=0]
@boxflg = external global i32                     ; <i32*> [#uses=0]
@dboxflg = external global i32                    ; <i32*> [#uses=0]
@tab = external global i32                        ; <i32*> [#uses=0]
@F1 = external global i32                         ; <i32*> [#uses=0]
@F2 = external global i32                         ; <i32*> [#uses=0]
@allflg = external global i32                     ; <i32*> [#uses=0]
@leftover = external global i32                   ; <i32*> [#uses=0]
@textflg = external global i32                    ; <i32*> [#uses=0]
@left1flg = external global i32                   ; <i32*> [#uses=0]
@rightl = external global i32                     ; <i32*> [#uses=0]
@iline = external global i32                      ; <i32*> [#uses=0]
@ifile = external global i8*                      ; <i8**> [#uses=0]
@.str = external constant [6 x i8], align 1       ; <[6 x i8]*> [#uses=0]
@texname = external global i32                    ; <i32*> [#uses=0]
@texct = external global i32                      ; <i32*> [#uses=0]
@texstr = external global [63 x i8], align 4      ; <[63 x i8]*> [#uses=0]
@nlin = external global i32                       ; <i32*> [#uses=0]
@ncol = external global i32                       ; <i32*> [#uses=0]
@nclin = external global i32                      ; <i32*> [#uses=0]
@nslin = external global i32                      ; <i32*> [#uses=0]
@style = external global [100 x [20 x i32]]       ; <[100 x [20 x i32]]*> [#uses=0]
@ctop = external global [100 x [20 x i32]]        ; <[100 x [20 x i32]]*> [#uses=0]
@font = external global [100 x [20 x [2 x i8]]]   ; <[100 x [20 x [2 x i8]]]*> [#uses=0]
@csize = external global [100 x [20 x [4 x i8]]]  ; <[100 x [20 x [4 x i8]]]*> [#uses=0]
@vsize = external global [100 x [20 x [4 x i8]]]  ; <[100 x [20 x [4 x i8]]]*> [#uses=0]
@cll = external global [20 x [10 x i8]]           ; <[20 x [10 x i8]]*> [#uses=0]
@stynum = external global [201 x i32]             ; <[201 x i32]*> [#uses=0]
@lefline = external global [100 x [20 x i32]]     ; <[100 x [20 x i32]]*> [#uses=0]
@fullbot = external global [200 x i32]            ; <[200 x i32]*> [#uses=0]
@instead = external global [200 x i8*]            ; <[200 x i8*]*> [#uses=0]
@evenflg = external global i32                    ; <i32*> [#uses=0]
@evenup = external global [20 x i32]              ; <[20 x i32]*> [#uses=0]
@linsize = external global i32                    ; <i32*> [#uses=0]
@pr1403 = external global i32                     ; <i32*> [#uses=0]
@delim1 = external global i32                     ; <i32*> [#uses=1]
@delim2 = external global i32                     ; <i32*> [#uses=1]
@table = external global [200 x %struct.colstr*]  ; <[200 x %struct.colstr*]*> [#uses=0]
@cspace = external global i8*                     ; <i8**> [#uses=0]
@cstore = external global i8*                     ; <i8**> [#uses=0]
@exstore = external global i8*                    ; <i8**> [#uses=0]
@exlim = external global i8*                      ; <i8**> [#uses=0]
@sep = external global [20 x i32]                 ; <[20 x i32]*> [#uses=0]
@used = external global [20 x i32]                ; <[20 x i32]*> [#uses=0]
@lused = external global [20 x i32]               ; <[20 x i32]*> [#uses=0]
@rused = external global [20 x i32]               ; <[20 x i32]*> [#uses=0]
@linestop = external global [200 x i32]           ; <[200 x i32]*> [#uses=0]
@last = external global i8*                       ; <i8**> [#uses=0]
@linstart = external global i32                   ; <i32*> [#uses=0]
@tabin = external global %struct.FILE*            ; <%struct.FILE**> [#uses=0]
@tabout = external global %struct.FILE*           ; <%struct.FILE**> [#uses=0]
@sargc = external global i32                      ; <i32*> [#uses=0]
@sargv = external global i8**                     ; <i8***> [#uses=0]
@.str1 = external constant [4 x i8], align 1      ; <[4 x i8]*> [#uses=0]
@.str12 = external constant [16 x i8], align 1    ; <[16 x i8]*> [#uses=0]
@.str2 = external constant [4 x i8], align 1      ; <[4 x i8]*> [#uses=0]
@.str3 = external constant [16 x i8], align 1     ; <[16 x i8]*> [#uses=0]
@.str4 = external constant [4 x i8], align 1      ; <[4 x i8]*> [#uses=0]
@.str5 = external constant [2 x i8], align 1      ; <[2 x i8]*> [#uses=0]
@.str6 = external constant [16 x i8], align 1     ; <[16 x i8]*> [#uses=0]
@__stdinp = external global %struct.FILE*         ; <%struct.FILE**> [#uses=0]
@__stdoutp = external global %struct.FILE*        ; <%struct.FILE**> [#uses=0]
@.str7 = external constant [4 x i8], align 1      ; <[4 x i8]*> [#uses=0]
@.str8 = external constant [4 x i8], align 1      ; <[4 x i8]*> [#uses=0]
@options = external global [21 x %struct.optstr]  ; <[21 x %struct.optstr]*> [#uses=0]
@.str9 = external constant [7 x i8], align 1      ; <[7 x i8]*> [#uses=0]
@.str110 = external constant [7 x i8], align 1    ; <[7 x i8]*> [#uses=0]
@.str211 = external constant [7 x i8], align 1    ; <[7 x i8]*> [#uses=0]
@.str312 = external constant [7 x i8], align 1    ; <[7 x i8]*> [#uses=0]
@.str413 = external constant [4 x i8], align 1    ; <[4 x i8]*> [#uses=0]
@.str514 = external constant [4 x i8], align 1    ; <[4 x i8]*> [#uses=0]
@.str615 = external constant [7 x i8], align 1    ; <[7 x i8]*> [#uses=0]
@.str716 = external constant [7 x i8], align 1    ; <[7 x i8]*> [#uses=0]
@.str817 = external constant [10 x i8], align 1   ; <[10 x i8]*> [#uses=0]
@.str918 = external constant [10 x i8], align 1   ; <[10 x i8]*> [#uses=0]
@.str10 = external constant [6 x i8], align 1     ; <[6 x i8]*> [#uses=0]
@.str11 = external constant [6 x i8], align 1     ; <[6 x i8]*> [#uses=0]
@.str1219 = external constant [12 x i8], align 1  ; <[12 x i8]*> [#uses=0]
@.str13 = external constant [12 x i8], align 1    ; <[12 x i8]*> [#uses=0]
@.str14 = external constant [4 x i8], align 1     ; <[4 x i8]*> [#uses=0]
@.str15 = external constant [4 x i8], align 1     ; <[4 x i8]*> [#uses=0]
@.str16 = external constant [9 x i8], align 1     ; <[9 x i8]*> [#uses=0]
@.str17 = external constant [9 x i8], align 1     ; <[9 x i8]*> [#uses=0]
@.str18 = external constant [6 x i8], align 1     ; <[6 x i8]*> [#uses=0]
@.str19 = external constant [6 x i8], align 1     ; <[6 x i8]*> [#uses=0]
@.str20 = external constant [14 x i8], align 1    ; <[14 x i8]*> [#uses=0]
@.str21 = external constant [25 x i8], align 1    ; <[25 x i8]*> [#uses=0]
@.str22 = external constant [11 x i8], align 1    ; <[11 x i8]*> [#uses=0]
@.str23 = external constant [15 x i8], align 1    ; <[15 x i8]*> [#uses=0]
@.str24 = external constant [34 x i8], align 1    ; <[34 x i8]*> [#uses=0]
@.str125 = external constant [32 x i8], align 1   ; <[32 x i8]*> [#uses=0]
@.str226 = external constant [17 x i8], align 1   ; <[17 x i8]*> [#uses=0]
@.str327 = external constant [38 x i8], align 1   ; <[38 x i8]*> [#uses=0]
@oncol = external global i32                      ; <i32*> [#uses=0]
@.str428 = external constant [40 x i8], align 1   ; <[40 x i8]*> [#uses=0]
@.str529 = external constant [31 x i8], align 1   ; <[31 x i8]*> [#uses=0]
@.str630 = external constant [51 x i8], align 1   ; <[51 x i8]*> [#uses=0]
@.str731 = external constant [51 x i8], align 1   ; <[51 x i8]*> [#uses=0]
@.str832 = external constant [40 x i8], align 1   ; <[40 x i8]*> [#uses=0]
@.str933 = external constant [26 x i8], align 1   ; <[26 x i8]*> [#uses=0]
@.str1034 = external constant [24 x i8], align 1  ; <[24 x i8]*> [#uses=0]
@.str1135 = external constant [21 x i8], align 1  ; <[21 x i8]*> [#uses=0]
@.str1236 = external constant [24 x i8], align 1  ; <[24 x i8]*> [#uses=0]
@.str1337 = external constant [33 x i8], align 1  ; <[33 x i8]*> [#uses=0]
@.str1438 = external constant [22 x i8], align 1  ; <[22 x i8]*> [#uses=0]
@.str1539 = external constant [32 x i8], align 1  ; <[32 x i8]*> [#uses=0]
@.str1640 = external constant [4 x i8], align 1   ; <[4 x i8]*> [#uses=0]
@.str1741 = external constant [6 x i8], align 1   ; <[6 x i8]*> [#uses=0]
@_DefaultRuneLocale = external global %struct._RuneLocale ; <%struct._RuneLocale*> [#uses=0]
@.str43 = external constant [3 x i8], align 1     ; <[3 x i8]*> [#uses=0]
@.str144 = external constant [43 x i8], align 1   ; <[43 x i8]*> [#uses=0]
@.str245 = external constant [4 x i8], align 1    ; <[4 x i8]*> [#uses=0]
@.str346 = external constant [4 x i8], align 1    ; <[4 x i8]*> [#uses=0]
@.str447 = external constant [4 x i8], align 1    ; <[4 x i8]*> [#uses=0]
@.str548 = external constant [3 x i8], align 1    ; <[3 x i8]*> [#uses=0]
@.str649 = external constant [1 x i8], align 1    ; <[1 x i8]*> [#uses=0]
@.str51 = external constant [5 x i8], align 1     ; <[5 x i8]*> [#uses=0]
@.str152 = external constant [2 x i8], align 1    ; <[2 x i8]*> [#uses=0]
@.str253 = external constant [2 x i8], align 1    ; <[2 x i8]*> [#uses=0]
@.str354 = external constant [7 x i8], align 1    ; <[7 x i8]*> [#uses=0]
@.str455 = external constant [10 x i8], align 1   ; <[10 x i8]*> [#uses=0]
@.str556 = external constant [16 x i8], align 1   ; <[16 x i8]*> [#uses=0]
@.str657 = external constant [19 x i8], align 1   ; <[19 x i8]*> [#uses=0]
@.str758 = external constant [32 x i8], align 1   ; <[32 x i8]*> [#uses=0]
@.str859 = external constant [8 x i8], align 1    ; <[8 x i8]*> [#uses=0]
@.str960 = external constant [30 x i8], align 1   ; <[30 x i8]*> [#uses=0]
@.str1061 = external constant [17 x i8], align 1  ; <[17 x i8]*> [#uses=0]
@.str1162 = external constant [35 x i8], align 1  ; <[35 x i8]*> [#uses=0]
@.str1263 = external constant [14 x i8], align 1  ; <[14 x i8]*> [#uses=0]
@.str1364 = external constant [20 x i8], align 1  ; <[20 x i8]*> [#uses=0]
@.str1465 = external constant [30 x i8], align 1  ; <[30 x i8]*> [#uses=0]
@.str1566 = external constant [41 x i8], align 1  ; <[41 x i8]*> [#uses=0]
@.str1667 = external constant [12 x i8], align 1  ; <[12 x i8]*> [#uses=0]
@.str1768 = external constant [7 x i8], align 1   ; <[7 x i8]*> [#uses=0]
@.str1869 = external constant [5 x i8], align 1   ; <[5 x i8]*> [#uses=0]
@.str1970 = external constant [29 x i8], align 1  ; <[29 x i8]*> [#uses=0]
@.str2071 = external constant [22 x i8], align 1  ; <[22 x i8]*> [#uses=0]
@.str2172 = external constant [17 x i8], align 1  ; <[17 x i8]*> [#uses=0]
@.str2273 = external constant [15 x i8], align 1  ; <[15 x i8]*> [#uses=0]
@.str2374 = external constant [36 x i8], align 1  ; <[36 x i8]*> [#uses=0]
@.str2475 = external constant [9 x i8], align 1   ; <[9 x i8]*> [#uses=0]
@.str25 = external constant [7 x i8], align 1     ; <[7 x i8]*> [#uses=0]
@.str26 = external constant [20 x i8], align 1    ; <[20 x i8]*> [#uses=0]
@.str27 = external constant [17 x i8], align 1    ; <[17 x i8]*> [#uses=0]
@.str28 = external constant [11 x i8], align 1    ; <[11 x i8]*> [#uses=0]
@.str29 = external constant [25 x i8], align 1    ; <[25 x i8]*> [#uses=0]
@.str30 = external constant [24 x i8], align 1    ; <[24 x i8]*> [#uses=0]
@.str31 = external constant [14 x i8], align 1    ; <[14 x i8]*> [#uses=0]
@.str32 = external constant [18 x i8], align 1    ; <[18 x i8]*> [#uses=0]
@.str33 = external constant [79 x i8], align 1    ; <[79 x i8]*> [#uses=0]
@.str77 = external constant [13 x i8], align 1    ; <[13 x i8]*> [#uses=0]
@.str178 = external constant [13 x i8], align 1   ; <[13 x i8]*> [#uses=0]
@.str279 = external constant [12 x i8], align 1   ; <[12 x i8]*> [#uses=0]
@.str380 = external constant [5 x i8], align 1    ; <[5 x i8]*> [#uses=0]
@.str481 = external constant [8 x i8], align 1    ; <[8 x i8]*> [#uses=0]
@.str582 = external constant [11 x i8], align 1   ; <[11 x i8]*> [#uses=0]
@.str683 = external constant [33 x i8], align 1   ; <[33 x i8]*> [#uses=0]
@.str784 = external constant [8 x i8], align 1    ; <[8 x i8]*> [#uses=0]
@.str885 = external constant [12 x i8], align 1   ; <[12 x i8]*> [#uses=0]
@.str986 = external constant [7 x i8], align 1    ; <[7 x i8]*> [#uses=0]
@.str1087 = external constant [28 x i8], align 1  ; <[28 x i8]*> [#uses=0]
@.str1188 = external constant [29 x i8], align 1  ; <[29 x i8]*> [#uses=0]
@.str1289 = external constant [11 x i8], align 1  ; <[11 x i8]*> [#uses=0]
@.str1390 = external constant [16 x i8], align 1  ; <[16 x i8]*> [#uses=0]
@.str1491 = external constant [22 x i8], align 1  ; <[22 x i8]*> [#uses=0]
@.str1592 = external constant [15 x i8], align 1  ; <[15 x i8]*> [#uses=0]
@.str1693 = external constant [13 x i8], align 1  ; <[13 x i8]*> [#uses=0]
@.str1794 = external constant [21 x i8], align 1  ; <[21 x i8]*> [#uses=0]
@.str1895 = external constant [25 x i8], align 1  ; <[25 x i8]*> [#uses=0]
@.str1996 = external constant [5 x i8], align 1   ; <[5 x i8]*> [#uses=0]
@.str2097 = external constant [4 x i8], align 1   ; <[4 x i8]*> [#uses=0]
@.str2198 = external constant [5 x i8], align 1   ; <[5 x i8]*> [#uses=0]
@.str2299 = external constant [5 x i8], align 1   ; <[5 x i8]*> [#uses=0]
@.str23100 = external constant [8 x i8], align 1  ; <[8 x i8]*> [#uses=0]
@.str24101 = external constant [14 x i8], align 1 ; <[14 x i8]*> [#uses=0]
@.str25102 = external constant [32 x i8], align 1 ; <[32 x i8]*> [#uses=0]
@.str26103 = external constant [11 x i8], align 1 ; <[11 x i8]*> [#uses=0]
@.str27104 = external constant [12 x i8], align 1 ; <[12 x i8]*> [#uses=0]
@.str28105 = external constant [5 x i8], align 1  ; <[5 x i8]*> [#uses=0]
@.str29106 = external constant [10 x i8], align 1 ; <[10 x i8]*> [#uses=0]
@.str30107 = external constant [7 x i8], align 1  ; <[7 x i8]*> [#uses=0]
@.str31108 = external constant [12 x i8], align 1 ; <[12 x i8]*> [#uses=0]
@.str111 = external constant [5 x i8], align 1    ; <[5 x i8]*> [#uses=0]
@.str1112 = external constant [8 x i8], align 1   ; <[8 x i8]*> [#uses=0]
@.str2113 = external constant [7 x i8], align 1   ; <[7 x i8]*> [#uses=0]
@.str3114 = external constant [8 x i8], align 1   ; <[8 x i8]*> [#uses=0]
@.str4115 = external constant [14 x i8], align 1  ; <[14 x i8]*> [#uses=0]
@.str5116 = external constant [16 x i8], align 1  ; <[16 x i8]*> [#uses=0]
@.str6117 = external constant [8 x i8], align 1   ; <[8 x i8]*> [#uses=0]
@.str7118 = external constant [28 x i8], align 1  ; <[28 x i8]*> [#uses=0]
@.str8119 = external constant [8 x i8], align 1   ; <[8 x i8]*> [#uses=0]
@.str9120 = external constant [16 x i8], align 1  ; <[16 x i8]*> [#uses=0]
@.str10121 = external constant [13 x i8], align 1 ; <[13 x i8]*> [#uses=0]
@.str11122 = external constant [14 x i8], align 1 ; <[14 x i8]*> [#uses=0]
@.str12123 = external constant [32 x i8], align 1 ; <[32 x i8]*> [#uses=0]
@.str13124 = external constant [27 x i8], align 1 ; <[27 x i8]*> [#uses=0]
@.str14125 = external constant [6 x i8], align 1  ; <[6 x i8]*> [#uses=0]
@.str15126 = external constant [13 x i8], align 1 ; <[13 x i8]*> [#uses=0]
@.str16127 = external constant [2 x i8], align 1  ; <[2 x i8]*> [#uses=0]
@.str17128 = external constant [8 x i8], align 1  ; <[8 x i8]*> [#uses=0]
@.str18129 = external constant [30 x i8], align 1 ; <[30 x i8]*> [#uses=0]
@.str19130 = external constant [13 x i8], align 1 ; <[13 x i8]*> [#uses=0]
@.str20131 = external constant [8 x i8], align 1  ; <[8 x i8]*> [#uses=0]
@.str21132 = external constant [9 x i8], align 1  ; <[9 x i8]*> [#uses=0]
@.str22133 = external constant [2 x i8], align 1  ; <[2 x i8]*> [#uses=0]
@watchout = external global i32                   ; <i32*> [#uses=0]
@once = external global i32                       ; <i32*> [#uses=0]
@.str23134 = external constant [20 x i8], align 1 ; <[20 x i8]*> [#uses=0]
@.str24135 = external constant [9 x i8], align 1  ; <[9 x i8]*> [#uses=0]
@.str25136 = external constant [18 x i8], align 1 ; <[18 x i8]*> [#uses=0]
@.str26137 = external constant [14 x i8], align 1 ; <[14 x i8]*> [#uses=0]
@.str27138 = external constant [63 x i8], align 1 ; <[63 x i8]*> [#uses=0]
@.str28139 = external constant [61 x i8], align 1 ; <[61 x i8]*> [#uses=0]
@.str29140 = external constant [14 x i8], align 1 ; <[14 x i8]*> [#uses=0]
@.str30141 = external constant [19 x i8], align 1 ; <[19 x i8]*> [#uses=0]
@.str31142 = external constant [15 x i8], align 1 ; <[15 x i8]*> [#uses=0]
@.str32143 = external constant [11 x i8], align 1 ; <[11 x i8]*> [#uses=0]
@.str33144 = external constant [3 x i8], align 1  ; <[3 x i8]*> [#uses=0]
@.str34 = external constant [12 x i8], align 1    ; <[12 x i8]*> [#uses=0]
@.str35 = external constant [23 x i8], align 1    ; <[23 x i8]*> [#uses=0]
@.str36 = external constant [23 x i8], align 1    ; <[23 x i8]*> [#uses=0]
@.str37 = external constant [5 x i8], align 1     ; <[5 x i8]*> [#uses=0]
@__stderrp = external global %struct.FILE*        ; <%struct.FILE**> [#uses=0]
@.str38 = external constant [44 x i8], align 1    ; <[44 x i8]*> [#uses=0]
@.str39 = external constant [16 x i8], align 1    ; <[16 x i8]*> [#uses=0]
@topat = external global [20 x i32]               ; <[20 x i32]*> [#uses=0]
@.str40 = external constant [22 x i8], align 1    ; <[22 x i8]*> [#uses=0]
@.str41 = external constant [10 x i8], align 1    ; <[10 x i8]*> [#uses=0]
@.str42 = external constant [12 x i8], align 1    ; <[12 x i8]*> [#uses=0]
@.str43145 = external constant [16 x i8], align 1 ; <[16 x i8]*> [#uses=0]
@.str149 = external constant [4 x i8], align 1    ; <[4 x i8]*> [#uses=0]
@useln = external global i32                      ; <i32*> [#uses=0]
@.str1150 = external constant [1 x i8], align 1   ; <[1 x i8]*> [#uses=0]
@.str2151 = external constant [26 x i8], align 1  ; <[26 x i8]*> [#uses=0]
@.str3152 = external constant [32 x i8], align 1  ; <[32 x i8]*> [#uses=0]
@spcount = external global i32                    ; <i32*> [#uses=0]
@tpcount = external global i32                    ; <i32*> [#uses=0]
@thisvec = external global i8*                    ; <i8**> [#uses=0]
@tpvecs = external global [50 x i8*]              ; <[50 x i8*]*> [#uses=0]
@.str156 = external constant [21 x i8], align 1   ; <[21 x i8]*> [#uses=0]
@spvecs = external global [20 x i8*]              ; <[20 x i8*]*> [#uses=0]
@.str1157 = external constant [29 x i8], align 1  ; <[29 x i8]*> [#uses=0]
@.str2158 = external constant [24 x i8], align 1  ; <[24 x i8]*> [#uses=0]
@.str164 = external constant [71 x i8], align 1   ; <[71 x i8]*> [#uses=0]
@.str1165 = external constant [71 x i8], align 1  ; <[71 x i8]*> [#uses=0]
@.str2166 = external constant [47 x i8], align 1  ; <[47 x i8]*> [#uses=0]
@.str169 = external constant [18 x i8], align 1   ; <[18 x i8]*> [#uses=0]
@backp = external global i8*                      ; <i8**> [#uses=0]
@backup = external global [500 x i8]              ; <[500 x i8]*> [#uses=0]
@.str1170 = external constant [15 x i8], align 1  ; <[15 x i8]*> [#uses=0]
@.str2171 = external constant [16 x i8], align 1  ; <[16 x i8]*> [#uses=0]
@.str176 = external constant [5 x i8], align 1    ; <[5 x i8]*> [#uses=0]
@.str1177 = external constant [35 x i8], align 1  ; <[35 x i8]*> [#uses=0]
@.str2178 = external constant [11 x i8], align 1  ; <[11 x i8]*> [#uses=0]
@.str3179 = external constant [33 x i8], align 1  ; <[33 x i8]*> [#uses=0]
@.str4180 = external constant [36 x i8], align 1  ; <[36 x i8]*> [#uses=0]
@.str5181 = external constant [11 x i8], align 1  ; <[11 x i8]*> [#uses=0]
@.str6182 = external constant [9 x i8], align 1   ; <[9 x i8]*> [#uses=0]
@.str7183 = external constant [4 x i8], align 1   ; <[4 x i8]*> [#uses=0]
@.str8184 = external constant [5 x i8], align 1   ; <[5 x i8]*> [#uses=0]
@.str9185 = external constant [8 x i8], align 1   ; <[8 x i8]*> [#uses=0]
@.str10186 = external constant [11 x i8], align 1 ; <[11 x i8]*> [#uses=0]
@.str11187 = external constant [12 x i8], align 1 ; <[12 x i8]*> [#uses=0]
@.str12188 = external constant [12 x i8], align 1 ; <[12 x i8]*> [#uses=0]
@.str13189 = external constant [15 x i8], align 1 ; <[15 x i8]*> [#uses=0]
@.str14190 = external constant [15 x i8], align 1 ; <[15 x i8]*> [#uses=0]
@.str15191 = external constant [17 x i8], align 1 ; <[17 x i8]*> [#uses=0]
@.str16192 = external constant [4 x i8], align 1  ; <[4 x i8]*> [#uses=0]
@.str17193 = external constant [5 x i8], align 1  ; <[5 x i8]*> [#uses=0]
@.str18194 = external constant [10 x i8], align 1 ; <[10 x i8]*> [#uses=0]
@.str19195 = external constant [19 x i8], align 1 ; <[19 x i8]*> [#uses=0]
@.str203 = external constant [5 x i8], align 1    ; <[5 x i8]*> [#uses=0]
@.str1204 = external constant [12 x i8], align 1  ; <[12 x i8]*> [#uses=0]
@.str2205 = external constant [31 x i8], align 1  ; <[31 x i8]*> [#uses=0]
@.str3206 = external constant [15 x i8], align 1  ; <[15 x i8]*> [#uses=0]
@.str4207 = external constant [5 x i8], align 1   ; <[5 x i8]*> [#uses=0]
@.str5208 = external constant [10 x i8], align 1  ; <[10 x i8]*> [#uses=0]
@.str6209 = external constant [5 x i8], align 1   ; <[5 x i8]*> [#uses=0]
@.str7210 = external constant [9 x i8], align 1   ; <[9 x i8]*> [#uses=0]
@.str8211 = external constant [21 x i8], align 1  ; <[21 x i8]*> [#uses=0]
@.str9212 = external constant [11 x i8], align 1  ; <[11 x i8]*> [#uses=0]
@.str10213 = external constant [14 x i8], align 1 ; <[14 x i8]*> [#uses=0]
@.str11214 = external constant [8 x i8], align 1  ; <[8 x i8]*> [#uses=0]
@.str12215 = external constant [8 x i8], align 1  ; <[8 x i8]*> [#uses=0]
@.str13216 = external constant [8 x i8], align 1  ; <[8 x i8]*> [#uses=0]
@.str14217 = external constant [37 x i8], align 1 ; <[37 x i8]*> [#uses=0]
@.str15218 = external constant [9 x i8], align 1  ; <[9 x i8]*> [#uses=0]
@.str16219 = external constant [20 x i8], align 1 ; <[20 x i8]*> [#uses=0]
@.str17220 = external constant [28 x i8], align 1 ; <[28 x i8]*> [#uses=0]
@.str18221 = external constant [9 x i8], align 1  ; <[9 x i8]*> [#uses=0]
@.str19222 = external constant [7 x i8], align 1  ; <[7 x i8]*> [#uses=0]
@.str20223 = external constant [3 x i8], align 1  ; <[3 x i8]*> [#uses=0]
@.str21224 = external constant [4 x i8], align 1  ; <[4 x i8]*> [#uses=0]
@.str22225 = external constant [11 x i8], align 1 ; <[11 x i8]*> [#uses=0]
@.str23226 = external constant [13 x i8], align 1 ; <[13 x i8]*> [#uses=0]
@.str24227 = external constant [5 x i8], align 1  ; <[5 x i8]*> [#uses=0]
@.str25228 = external constant [15 x i8], align 1 ; <[15 x i8]*> [#uses=0]
@.str26229 = external constant [15 x i8], align 1 ; <[15 x i8]*> [#uses=0]
@.str27230 = external constant [4 x i8], align 1  ; <[4 x i8]*> [#uses=0]
@.str28231 = external constant [7 x i8], align 1  ; <[7 x i8]*> [#uses=0]
@.str242 = external constant [7 x i8], align 1    ; <[7 x i8]*> [#uses=0]
@.str1243 = external constant [25 x i8], align 1  ; <[25 x i8]*> [#uses=0]
@.str252 = external constant [4 x i8], align 1    ; <[4 x i8]*> [#uses=0]
@.str1253 = external constant [1 x i8], align 1   ; <[1 x i8]*> [#uses=0]
@.str2254 = external constant [9 x i8], align 1   ; <[9 x i8]*> [#uses=0]
@.str3255 = external constant [8 x i8], align 1   ; <[8 x i8]*> [#uses=0]
@.str4256 = external constant [3 x i8], align 1   ; <[3 x i8]*> [#uses=0]
@.str5257 = external constant [4 x i8], align 1   ; <[4 x i8]*> [#uses=0]
@.str6258 = external constant [7 x i8], align 1   ; <[7 x i8]*> [#uses=0]
@.str7259 = external constant [4 x i8], align 1   ; <[4 x i8]*> [#uses=0]
@.str8260 = external constant [12 x i8], align 1  ; <[12 x i8]*> [#uses=0]
@.str9261 = external constant [8 x i8], align 1   ; <[8 x i8]*> [#uses=0]
@.str10262 = external constant [15 x i8], align 1 ; <[15 x i8]*> [#uses=0]
@.str11263 = external constant [12 x i8], align 1 ; <[12 x i8]*> [#uses=0]
@.str12264 = external constant [5 x i8], align 1  ; <[5 x i8]*> [#uses=0]
@.str13265 = external constant [2 x i8], align 1  ; <[2 x i8]*> [#uses=0]
@.str14266 = external constant [5 x i8], align 1  ; <[5 x i8]*> [#uses=0]
@.str15267 = external constant [16 x i8], align 1 ; <[16 x i8]*> [#uses=0]
@.str16268 = external constant [29 x i8], align 1 ; <[29 x i8]*> [#uses=0]
@.str17269 = external constant [14 x i8], align 1 ; <[14 x i8]*> [#uses=0]
@.str18270 = external constant [4 x i8], align 1  ; <[4 x i8]*> [#uses=0]
@.str19271 = external constant [9 x i8], align 1  ; <[9 x i8]*> [#uses=0]
@.str20272 = external constant [32 x i8], align 1 ; <[32 x i8]*> [#uses=0]
@.str21273 = external constant [12 x i8], align 1 ; <[12 x i8]*> [#uses=0]
@.str282 = external constant [8 x i8], align 1    ; <[8 x i8]*> [#uses=0]
@.str1283 = external constant [7 x i8], align 1   ; <[7 x i8]*> [#uses=0]
@.str2284 = external constant [4 x i8], align 1   ; <[4 x i8]*> [#uses=0]
@.str3285 = external constant [7 x i8], align 1   ; <[7 x i8]*> [#uses=0]
@.str4286 = external constant [8 x i8], align 1   ; <[8 x i8]*> [#uses=0]
@.str5287 = external constant [8 x i8], align 1   ; <[8 x i8]*> [#uses=0]
@.str6288 = external constant [15 x i8], align 1  ; <[15 x i8]*> [#uses=0]
@.str7289 = external constant [12 x i8], align 1  ; <[12 x i8]*> [#uses=0]
@.str8290 = external constant [3 x i8], align 1   ; <[3 x i8]*> [#uses=0]
@.str9291 = external constant [7 x i8], align 1   ; <[7 x i8]*> [#uses=0]
@.str10292 = external constant [15 x i8], align 1 ; <[15 x i8]*> [#uses=0]
@.str11293 = external constant [6 x i8], align 1  ; <[6 x i8]*> [#uses=0]
@.str12294 = external constant [2 x i8], align 1  ; <[2 x i8]*> [#uses=0]
@.str13295 = external constant [1 x i8], align 1  ; <[1 x i8]*> [#uses=0]
@.str14296 = external constant [6 x i8], align 1  ; <[6 x i8]*> [#uses=0]
@.str15297 = external constant [28 x i8], align 1 ; <[28 x i8]*> [#uses=0]
@.str16298 = external constant [4 x i8], align 1  ; <[4 x i8]*> [#uses=0]
@.str17299 = external constant [14 x i8], align 1 ; <[14 x i8]*> [#uses=0]

declare arm_apcscc void @main(i32, i8**) noreturn nounwind

declare arm_apcscc i32 @swapin() nounwind

declare arm_apcscc %struct.FILE* @"\01_fopen"(i8*, i8*)

declare arm_apcscc void @setinp(i32, i8**) nounwind

declare arm_apcscc i32 @tbl(i32, i8**) nounwind

declare arm_apcscc i32 @fprintf(%struct.FILE* nocapture, i8* nocapture, ...) nounwind

declare arm_apcscc i32 @fclose(%struct.FILE* nocapture) nounwind

declare arm_apcscc void @exit(i32) noreturn nounwind

declare arm_apcscc void @tableput() nounwind

declare arm_apcscc void @init_options() nounwind

declare arm_apcscc void @backrest(i8*) nounwind

declare arm_apcscc void @getcomm() nounwind

declare arm_apcscc i32 @printf(i8* nocapture, ...) nounwind

declare arm_apcscc i8* @strchr(i8*, i32) nounwind readonly

declare arm_apcscc i32 @strlen(i8* nocapture) nounwind readonly

declare arm_apcscc void @getspec() nounwind

declare arm_apcscc void @readspec() nounwind

declare arm_apcscc i32 @"\01_fwrite"(i8*, i32, i32, i8*)

declare arm_apcscc i32 @atoi(i8* nocapture) nounwind readonly

declare arm_apcscc i32 @fputc(i32, i8* nocapture) nounwind

declare arm_apcscc void @gettbl() nounwind

declare arm_apcscc i32 @vspen(i8*) nounwind readonly

declare arm_apcscc i32 @vspand(i32, i32, i32) nounwind readonly

declare arm_apcscc i32 @oneh(i32) nounwind readonly

declare arm_apcscc i32 @nodata(i32) nounwind readonly

declare arm_apcscc i32 @permute() nounwind

declare arm_apcscc void @maktab() nounwind

declare arm_apcscc i32 @filler(i8*) nounwind readonly

declare arm_apcscc void @wide(i8*, i8*, i8*) nounwind

declare arm_apcscc i32 @"\01_fputs"(i8*, i8*)

declare arm_apcscc void @runout() nounwind

declare arm_apcscc void @need() nounwind

declare arm_apcscc void @deftail() nounwind

declare arm_apcscc i32 @ifline(i8*) nounwind readonly

declare arm_apcscc void @runtabs(i32, i32) nounwind

declare arm_apcscc void @putline(i32, i32) nounwind

declare arm_apcscc void @putsize(i8*) nounwind

declare arm_apcscc void @putfont(i8*) nounwind

declare arm_apcscc i32 @__maskrune(i32, i32)

declare arm_apcscc void @funnies(i32, i32) nounwind

declare arm_apcscc void @puttext(i8*, i8*, i8*) nounwind

declare arm_apcscc i32 @puts(i8* nocapture) nounwind

declare arm_apcscc void @yetmore() nounwind

declare arm_apcscc i32 @domore(i8*) nounwind

declare arm_apcscc void @checkuse() nounwind

declare arm_apcscc void @release() nounwind

declare arm_apcscc i32* @alocv(i32) nounwind

declare arm_apcscc i8* @calloc(...)

declare arm_apcscc i8* @chspace() nounwind

declare arm_apcscc i32 @real(i8*) nounwind readonly

declare arm_apcscc void @choochar() nounwind

declare arm_apcscc i32 @point(i32) nounwind readnone

declare arm_apcscc void @error(i8*) nounwind

declare arm_apcscc i8* @gets1(i8*) nounwind

declare arm_apcscc i8* @fgets(i8*, i32, %struct.FILE* nocapture) nounwind

declare arm_apcscc i32 @get1char() nounwind

declare arm_apcscc i32 @getc(%struct.FILE* nocapture) nounwind

declare arm_apcscc void @un1getc(i32) nounwind

declare arm_apcscc void @savefill() nounwind

declare arm_apcscc void @cleanfc() nounwind

declare arm_apcscc void @saveline() nounwind

declare arm_apcscc void @ifdivert() nounwind

declare arm_apcscc void @restline() nounwind

declare arm_apcscc void @endoff() nounwind

declare arm_apcscc void @rstofill() nounwind

declare arm_apcscc i32 @gettext(i8* nocapture, i32, i32, i8*, i8*) nounwind

declare arm_apcscc void @untext() nounwind

declare arm_apcscc i32 @interv(i32, i32) nounwind readonly

declare arm_apcscc i32 @up1(i32) nounwind readonly

declare arm_apcscc i32 @interh(i32, i32) nounwind readonly

declare arm_apcscc i32 @maknew(i8*) nounwind

define arm_apcscc i32 @ineqn(i8* %s, i8* %p) nounwind readonly {
entry:
  %0 = load i32* @delim1, align 4                 ; <i32> [#uses=1]
  %1 = load i32* @delim2, align 4                 ; <i32> [#uses=1]
  br label %bb8.outer

bb:                                               ; preds = %bb8
  %2 = icmp eq i8* %p_addr.0, %s                  ; <i1> [#uses=1]
  br i1 %2, label %bb10, label %bb2

bb2:                                              ; preds = %bb
  %3 = getelementptr inbounds i8* %p_addr.0, i32 1 ; <i8*> [#uses=3]
  switch i32 %ineq.0.ph, label %bb8.backedge [
    i32 0, label %bb3
    i32 1, label %bb6
  ]

bb8.backedge:                                     ; preds = %bb6, %bb5, %bb2
  br label %bb8

bb3:                                              ; preds = %bb2
  %4 = icmp eq i32 %8, %0                         ; <i1> [#uses=1]
  br i1 %4, label %bb8.outer.loopexit, label %bb5

bb5:                                              ; preds = %bb3
  br i1 %6, label %bb6, label %bb8.backedge

bb6:                                              ; preds = %bb5, %bb2
  %5 = icmp eq i32 %8, %1                         ; <i1> [#uses=1]
  br i1 %5, label %bb7, label %bb8.backedge

bb7:                                              ; preds = %bb6
  %.lcssa1 = phi i8* [ %3, %bb6 ]                 ; <i8*> [#uses=1]
  br label %bb8.outer.backedge

bb8.outer.backedge:                               ; preds = %bb8.outer.loopexit, %bb7
  %.lcssa2 = phi i8* [ %.lcssa1, %bb7 ], [ %.lcssa, %bb8.outer.loopexit ] ; <i8*> [#uses=1]
  %ineq.0.ph.be = phi i32 [ 0, %bb7 ], [ 1, %bb8.outer.loopexit ] ; <i32> [#uses=1]
  br label %bb8.outer

bb8.outer.loopexit:                               ; preds = %bb3
  %.lcssa = phi i8* [ %3, %bb3 ]                  ; <i8*> [#uses=1]
  br label %bb8.outer.backedge

bb8.outer:                                        ; preds = %bb8.outer.backedge, %entry
  %ineq.0.ph = phi i32 [ 0, %entry ], [ %ineq.0.ph.be, %bb8.outer.backedge ] ; <i32> [#uses=3]
  %p_addr.0.ph = phi i8* [ %p, %entry ], [ %.lcssa2, %bb8.outer.backedge ] ; <i8*> [#uses=1]
  %6 = icmp eq i32 %ineq.0.ph, 1                  ; <i1> [#uses=1]
  br label %bb8

bb8:                                              ; preds = %bb8.outer, %bb8.backedge
  %p_addr.0 = phi i8* [ %p_addr.0.ph, %bb8.outer ], [ %3, %bb8.backedge ] ; <i8*> [#uses=3]
  %7 = load i8* %p_addr.0, align 1                ; <i8> [#uses=2]
  %8 = sext i8 %7 to i32                          ; <i32> [#uses=2]
  %9 = icmp eq i8 %7, 0                           ; <i1> [#uses=1]
  br i1 %9, label %bb10, label %bb

bb10:                                             ; preds = %bb8, %bb
  %.0 = phi i32 [ %ineq.0.ph, %bb ], [ 0, %bb8 ]  ; <i32> [#uses=1]
  ret i32 %.0
}

declare arm_apcscc i32 @match(i8* nocapture, i8* nocapture) nounwind readonly

declare arm_apcscc i32 @prefix(i8* nocapture, i8* nocapture) nounwind readonly

declare arm_apcscc i32 @letter(i32) nounwind readnone

declare arm_apcscc i32 @numb(i8* nocapture) nounwind readonly

declare arm_apcscc i32 @digit(i32) nounwind readnone

declare arm_apcscc i32 @max(i32, i32) nounwind readnone

declare arm_apcscc void @tcopy(i8* nocapture, i8* nocapture) nounwind

declare arm_apcscc i32 @ctype(i32, i32) nounwind readonly

declare arm_apcscc i32 @min(i32, i32) nounwind readnone

declare arm_apcscc i32 @fspan(i32, i32) nounwind readonly

declare arm_apcscc i32 @lspan(i32, i32) nounwind readonly

declare arm_apcscc i32 @ctspan(i32, i32) nounwind readonly

declare arm_apcscc i32 @thish(i32, i32) nounwind readonly

declare arm_apcscc i32 @allh(i32) nounwind readonly

declare arm_apcscc void @tohcol(i32) nounwind

declare arm_apcscc void @makeline(i32, i32, i32) nounwind

declare arm_apcscc i32 @next(i32) nounwind readonly

declare arm_apcscc i32 @prev(i32) nounwind readonly

declare arm_apcscc i32 @lefdata(i32, i32) nounwind readonly

declare arm_apcscc i32 @left(i32, i32, i32* nocapture) nounwind

declare arm_apcscc i32 @strcmp(i8* nocapture, i8* nocapture) nounwind readonly

declare arm_apcscc void @getstop() nounwind

declare arm_apcscc void @drawline(i32, i32, i32, i32, i32, i32) nounwind

declare arm_apcscc void @fullwide(i32, i32) nounwind

declare arm_apcscc void @drawvert(i32, i32, i32, i32) nounwind

declare arm_apcscc i32 @barent(i8*) nounwind readonly

declare arm_apcscc i32 @midbcol(i32, i32) nounwind readonly

declare arm_apcscc i32 @midbar(i32, i32) nounwind readonly


; This is a simplified form of ineqn from above. It triggers some
; different cases in the loop-unswitch code.

define void @simplified_ineqn() nounwind readonly {
entry:
  br label %bb8.outer

bb8.outer:                                        ; preds = %bb6, %bb2, %entry
  %x = phi i32 [ 0, %entry ], [ 0, %bb6 ], [ 1, %bb2 ] ; <i32> [#uses=1]
  br i1 undef, label %return, label %bb2

bb2:                                              ; preds = %bb
  switch i32 %x, label %bb6 [
    i32 0, label %bb8.outer
  ]

bb6:                                              ; preds = %bb2
  br i1 undef, label %bb8.outer, label %bb2

return:                                             ; preds = %bb8, %bb
  ret void
}
