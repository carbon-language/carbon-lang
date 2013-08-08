; RUN: opt -objc-arc -S < %s
; rdar://12277446
; rdar://12480535

; The total number of paths grows exponentially with the number of branches, and a
; computation of this number can overflow any reasonable fixed-sized
; integer. This can occur in both the addition phase when we are adding up the
; total bottomup/topdown paths and when we multiply them together at the end.

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios5.0.0"

%struct.NSConstantString.11.33.55.77.99.121.143.332.1130.1340.2768 = type { i32*, i32, i8*, i32 }

@_unnamed_cfstring_591 = external constant %struct.NSConstantString.11.33.55.77.99.121.143.332.1130.1340.2768, section "__DATA,__cfstring"

declare i8* @objc_retain(i8*) nonlazybind
declare i8* @objc_retainAutoreleasedReturnValue(i8*) nonlazybind
declare void @objc_release(i8*) nonlazybind
declare i8* @returner()

define hidden void @test1() {
entry:
  br i1 undef, label %msgSend.nullinit, label %msgSend.call

msgSend.call:                                     ; preds = %entry
  br label %msgSend.cont

msgSend.nullinit:                                 ; preds = %entry
  br label %msgSend.cont

msgSend.cont:                                     ; preds = %msgSend.nullinit, %msgSend.call
  %0 = bitcast %struct.NSConstantString.11.33.55.77.99.121.143.332.1130.1340.2768* @_unnamed_cfstring_591 to i8*
  %1 = call i8* @objc_retain(i8* %0) nounwind
  br i1 undef, label %msgSend.nullinit33, label %msgSend.call32

msgSend.call32:                                   ; preds = %if.end10
  br label %msgSend.cont34

msgSend.nullinit33:                               ; preds = %if.end10
  br label %msgSend.cont34

msgSend.cont34:                                   ; preds = %msgSend.nullinit33, %msgSend.call32
  br i1 undef, label %msgSend.nullinit38, label %msgSend.call37

msgSend.call37:                                   ; preds = %msgSend.cont34
  br label %msgSend.cont39

msgSend.nullinit38:                               ; preds = %msgSend.cont34
  br label %msgSend.cont39

msgSend.cont39:                                   ; preds = %msgSend.nullinit38, %msgSend.call37
  br i1 undef, label %msgSend.nullinit49, label %msgSend.call48

msgSend.call48:                                   ; preds = %msgSend.cont39
  br label %msgSend.cont50

msgSend.nullinit49:                               ; preds = %msgSend.cont39
  br label %msgSend.cont50

msgSend.cont50:                                   ; preds = %msgSend.nullinit49, %msgSend.call48
  br i1 undef, label %msgSend.nullinit61, label %msgSend.call60

msgSend.call60:                                   ; preds = %msgSend.cont50
  br label %msgSend.cont62

msgSend.nullinit61:                               ; preds = %msgSend.cont50
  br label %msgSend.cont62

msgSend.cont62:                                   ; preds = %msgSend.nullinit61, %msgSend.call60
  br i1 undef, label %msgSend.nullinit67, label %msgSend.call66

msgSend.call66:                                   ; preds = %msgSend.cont62
  br label %msgSend.cont68

msgSend.nullinit67:                               ; preds = %msgSend.cont62
  br label %msgSend.cont68

msgSend.cont68:                                   ; preds = %msgSend.nullinit67, %msgSend.call66
  br i1 undef, label %msgSend.nullinit84, label %msgSend.call83

msgSend.call83:                                   ; preds = %msgSend.cont68
  br label %msgSend.cont85

msgSend.nullinit84:                               ; preds = %msgSend.cont68
  br label %msgSend.cont85

msgSend.cont85:                                   ; preds = %msgSend.nullinit84, %msgSend.call83
  br i1 undef, label %msgSend.nullinit90, label %msgSend.call89

msgSend.call89:                                   ; preds = %msgSend.cont85
  br label %msgSend.cont91

msgSend.nullinit90:                               ; preds = %msgSend.cont85
  br label %msgSend.cont91

msgSend.cont91:                                   ; preds = %msgSend.nullinit90, %msgSend.call89
  br i1 undef, label %msgSend.nullinit104, label %msgSend.call103

msgSend.call103:                                  ; preds = %msgSend.cont91
  br label %msgSend.cont105

msgSend.nullinit104:                              ; preds = %msgSend.cont91
  br label %msgSend.cont105

msgSend.cont105:                                  ; preds = %msgSend.nullinit104, %msgSend.call103
  br i1 undef, label %land.lhs.true, label %if.end123

land.lhs.true:                                    ; preds = %msgSend.cont105
  br i1 undef, label %if.then117, label %if.end123

if.then117:                                       ; preds = %land.lhs.true
  br label %if.end123

if.end123:                                        ; preds = %if.then117, %land.lhs.true, %msgSend.cont105
  br i1 undef, label %msgSend.nullinit132, label %msgSend.call131

msgSend.call131:                                  ; preds = %if.end123
  br label %msgSend.cont133

msgSend.nullinit132:                              ; preds = %if.end123
  br label %msgSend.cont133

msgSend.cont133:                                  ; preds = %msgSend.nullinit132, %msgSend.call131
  br i1 undef, label %msgSend.nullinit139, label %msgSend.call138

msgSend.call138:                                  ; preds = %msgSend.cont133
  br label %msgSend.cont140

msgSend.nullinit139:                              ; preds = %msgSend.cont133
  br label %msgSend.cont140

msgSend.cont140:                                  ; preds = %msgSend.nullinit139, %msgSend.call138
  br i1 undef, label %if.then151, label %if.end157

if.then151:                                       ; preds = %msgSend.cont140
  br label %if.end157

if.end157:                                        ; preds = %if.then151, %msgSend.cont140
  br i1 undef, label %msgSend.nullinit164, label %msgSend.call163

msgSend.call163:                                  ; preds = %if.end157
  br label %msgSend.cont165

msgSend.nullinit164:                              ; preds = %if.end157
  br label %msgSend.cont165

msgSend.cont165:                                  ; preds = %msgSend.nullinit164, %msgSend.call163
  br i1 undef, label %msgSend.nullinit176, label %msgSend.call175

msgSend.call175:                                  ; preds = %msgSend.cont165
  br label %msgSend.cont177

msgSend.nullinit176:                              ; preds = %msgSend.cont165
  br label %msgSend.cont177

msgSend.cont177:                                  ; preds = %msgSend.nullinit176, %msgSend.call175
  br i1 undef, label %land.lhs.true181, label %if.end202

land.lhs.true181:                                 ; preds = %msgSend.cont177
  br i1 undef, label %if.then187, label %if.end202

if.then187:                                       ; preds = %land.lhs.true181
  br i1 undef, label %msgSend.nullinit199, label %msgSend.call198

msgSend.call198:                                  ; preds = %if.then187
  br label %msgSend.cont200

msgSend.nullinit199:                              ; preds = %if.then187
  br label %msgSend.cont200

msgSend.cont200:                                  ; preds = %msgSend.nullinit199, %msgSend.call198
  br label %if.end202

if.end202:                                        ; preds = %msgSend.cont200, %land.lhs.true181, %msgSend.cont177
  br i1 undef, label %msgSend.nullinit236, label %msgSend.call235

msgSend.call235:                                  ; preds = %if.end202
  br label %msgSend.cont237

msgSend.nullinit236:                              ; preds = %if.end202
  br label %msgSend.cont237

msgSend.cont237:                                  ; preds = %msgSend.nullinit236, %msgSend.call235
  br i1 undef, label %msgSend.nullinit254, label %msgSend.call253

msgSend.call253:                                  ; preds = %msgSend.cont237
  br label %msgSend.cont255

msgSend.nullinit254:                              ; preds = %msgSend.cont237
  br label %msgSend.cont255

msgSend.cont255:                                  ; preds = %msgSend.nullinit254, %msgSend.call253
  br i1 undef, label %msgSend.nullinit269, label %msgSend.call268

msgSend.call268:                                  ; preds = %msgSend.cont255
  br label %msgSend.cont270

msgSend.nullinit269:                              ; preds = %msgSend.cont255
  br label %msgSend.cont270

msgSend.cont270:                                  ; preds = %msgSend.nullinit269, %msgSend.call268
  br i1 undef, label %msgSend.nullinit281, label %msgSend.call280

msgSend.call280:                                  ; preds = %msgSend.cont270
  br label %msgSend.cont282

msgSend.nullinit281:                              ; preds = %msgSend.cont270
  br label %msgSend.cont282

msgSend.cont282:                                  ; preds = %msgSend.nullinit281, %msgSend.call280
  br i1 undef, label %msgSend.nullinit287, label %msgSend.call286

msgSend.call286:                                  ; preds = %msgSend.cont282
  br label %msgSend.cont288

msgSend.nullinit287:                              ; preds = %msgSend.cont282
  br label %msgSend.cont288

msgSend.cont288:                                  ; preds = %msgSend.nullinit287, %msgSend.call286
  br i1 undef, label %msgSend.nullinit303, label %msgSend.call302

msgSend.call302:                                  ; preds = %msgSend.cont288
  br label %msgSend.cont304

msgSend.nullinit303:                              ; preds = %msgSend.cont288
  br label %msgSend.cont304

msgSend.cont304:                                  ; preds = %msgSend.nullinit303, %msgSend.call302
  br i1 undef, label %msgSend.nullinit344, label %msgSend.call343

msgSend.call343:                                  ; preds = %msgSend.cont304
  br label %msgSend.cont345

msgSend.nullinit344:                              ; preds = %msgSend.cont304
  br label %msgSend.cont345

msgSend.cont345:                                  ; preds = %msgSend.nullinit344, %msgSend.call343
  br i1 undef, label %msgSend.nullinit350, label %msgSend.call349

msgSend.call349:                                  ; preds = %msgSend.cont345
  br label %msgSend.cont351

msgSend.nullinit350:                              ; preds = %msgSend.cont345
  br label %msgSend.cont351

msgSend.cont351:                                  ; preds = %msgSend.nullinit350, %msgSend.call349
  br i1 undef, label %msgSend.nullinit366, label %msgSend.call365

msgSend.call365:                                  ; preds = %msgSend.cont351
  br label %msgSend.cont367

msgSend.nullinit366:                              ; preds = %msgSend.cont351
  br label %msgSend.cont367

msgSend.cont367:                                  ; preds = %msgSend.nullinit366, %msgSend.call365
  br i1 undef, label %msgSend.nullinit376, label %msgSend.call375

msgSend.call375:                                  ; preds = %msgSend.cont367
  br label %msgSend.cont377

msgSend.nullinit376:                              ; preds = %msgSend.cont367
  br label %msgSend.cont377

msgSend.cont377:                                  ; preds = %msgSend.nullinit376, %msgSend.call375
  br i1 undef, label %if.then384, label %if.else401

if.then384:                                       ; preds = %msgSend.cont377
  br i1 undef, label %msgSend.nullinit392, label %msgSend.call391

msgSend.call391:                                  ; preds = %if.then384
  br label %msgSend.cont393

msgSend.nullinit392:                              ; preds = %if.then384
  br label %msgSend.cont393

msgSend.cont393:                                  ; preds = %msgSend.nullinit392, %msgSend.call391
  br label %if.end418

if.else401:                                       ; preds = %msgSend.cont377
  br i1 undef, label %msgSend.nullinit409, label %msgSend.call408

msgSend.call408:                                  ; preds = %if.else401
  br label %msgSend.cont410

msgSend.nullinit409:                              ; preds = %if.else401
  br label %msgSend.cont410

msgSend.cont410:                                  ; preds = %msgSend.nullinit409, %msgSend.call408
  br label %if.end418

if.end418:                                        ; preds = %msgSend.cont410, %msgSend.cont393
  br i1 undef, label %msgSend.nullinit470, label %msgSend.call469

msgSend.call469:                                  ; preds = %if.end418
  br label %msgSend.cont471

msgSend.nullinit470:                              ; preds = %if.end418
  br label %msgSend.cont471

msgSend.cont471:                                  ; preds = %msgSend.nullinit470, %msgSend.call469
  br i1 undef, label %msgSend.nullinit484, label %msgSend.call483

msgSend.call483:                                  ; preds = %msgSend.cont471
  br label %msgSend.cont485

msgSend.nullinit484:                              ; preds = %msgSend.cont471
  br label %msgSend.cont485

msgSend.cont485:                                  ; preds = %msgSend.nullinit484, %msgSend.call483
  br i1 undef, label %msgSend.nullinit500, label %msgSend.call499

msgSend.call499:                                  ; preds = %msgSend.cont485
  br label %msgSend.cont501

msgSend.nullinit500:                              ; preds = %msgSend.cont485
  br label %msgSend.cont501

msgSend.cont501:                                  ; preds = %msgSend.nullinit500, %msgSend.call499
  br i1 undef, label %msgSend.nullinit506, label %msgSend.call505

msgSend.call505:                                  ; preds = %msgSend.cont501
  br label %msgSend.cont507

msgSend.nullinit506:                              ; preds = %msgSend.cont501
  br label %msgSend.cont507

msgSend.cont507:                                  ; preds = %msgSend.nullinit506, %msgSend.call505
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Function Attrs: optsize ssp uwtable
define void @test2() unnamed_addr align 2 {
bb:
  br i1 undef, label %bb3, label %bb2

bb2:                                              ; preds = %bb
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  br i1 undef, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  br label %bb5

bb5:                                              ; preds = %bb4, %bb3
  br i1 undef, label %bb7, label %bb6

bb6:                                              ; preds = %bb5
  br label %bb7

bb7:                                              ; preds = %bb6, %bb5
  br i1 undef, label %bb9, label %bb8

bb8:                                              ; preds = %bb7
  unreachable

bb9:                                              ; preds = %bb7
  br i1 undef, label %bb11, label %bb10

bb10:                                             ; preds = %bb9
  br label %bb11

bb11:                                             ; preds = %bb10, %bb9
  br i1 undef, label %bb13, label %bb12

bb12:                                             ; preds = %bb11
  br label %bb13

bb13:                                             ; preds = %bb12, %bb11
  br i1 undef, label %bb15, label %bb14

bb14:                                             ; preds = %bb13
  br label %bb15

bb15:                                             ; preds = %bb14, %bb13
  br i1 undef, label %bb17, label %bb16

bb16:                                             ; preds = %bb15
  br label %bb17

bb17:                                             ; preds = %bb16, %bb15
  br i1 undef, label %bb19, label %bb18

bb18:                                             ; preds = %bb17
  br label %bb19

bb19:                                             ; preds = %bb18, %bb17
  br i1 undef, label %bb222, label %bb20

bb20:                                             ; preds = %bb19
  br i1 undef, label %bb222, label %bb21

bb21:                                             ; preds = %bb20
  br i1 undef, label %bb22, label %bb30

bb22:                                             ; preds = %bb21
  br i1 undef, label %bb23, label %bb32

bb23:                                             ; preds = %bb22
  br i1 undef, label %bb24, label %bb34

bb24:                                             ; preds = %bb23
  br i1 undef, label %bb26, label %bb25

bb25:                                             ; preds = %bb24
  br label %bb27

bb26:                                             ; preds = %bb24
  br label %bb27

bb27:                                             ; preds = %bb26, %bb25
  br i1 undef, label %bb28, label %bb42

bb28:                                             ; preds = %bb27
  br i1 undef, label %bb36, label %bb29

bb29:                                             ; preds = %bb28
  br label %bb36

bb30:                                             ; preds = %bb210, %bb207, %bb203, %bb199, %bb182, %bb176, %bb174, %bb171, %bb136, %bb132, %bb21
  br label %bb213

bb32:                                             ; preds = %bb22
  unreachable

bb34:                                             ; preds = %bb23
  unreachable

bb36:                                             ; preds = %bb29, %bb28
  br i1 undef, label %bb38, label %bb37

bb37:                                             ; preds = %bb36
  br label %bb39

bb38:                                             ; preds = %bb36
  br label %bb39

bb39:                                             ; preds = %bb38, %bb37
  br i1 undef, label %bb41, label %bb40

bb40:                                             ; preds = %bb39
  unreachable

bb41:                                             ; preds = %bb39
  br label %bb42

bb42:                                             ; preds = %bb41, %bb27
  br i1 undef, label %bb43, label %bb214

bb43:                                             ; preds = %bb42
  br i1 undef, label %bb47, label %bb45

bb45:                                             ; preds = %bb130, %bb128, %bb126, %bb124, %bb122, %bb120, %bb118, %bb116, %bb114, %bb112, %bb110, %bb108, %bb105, %bb102, %bb100, %bb96, %bb94, %bb90, %bb88, %bb84, %bb82, %bb78, %bb76, %bb72, %bb70, %bb66, %bb64, %bb60, %bb58, %bb54, %bb51, %bb43
  unreachable

bb47:                                             ; preds = %bb43
  br i1 undef, label %bb48, label %bb106

bb48:                                             ; preds = %bb47
  br i1 undef, label %bb50, label %bb49

bb49:                                             ; preds = %bb48
  br label %bb51

bb50:                                             ; preds = %bb48
  br label %bb51

bb51:                                             ; preds = %bb50, %bb49
  br i1 undef, label %bb53, label %bb45

bb53:                                             ; preds = %bb51
  br i1 undef, label %bb54, label %bb134

bb54:                                             ; preds = %bb53
  br i1 undef, label %bb55, label %bb45

bb55:                                             ; preds = %bb54
  br i1 undef, label %bb57, label %bb56

bb56:                                             ; preds = %bb55
  br label %bb58

bb57:                                             ; preds = %bb55
  br label %bb58

bb58:                                             ; preds = %bb57, %bb56
  br i1 undef, label %bb60, label %bb45

bb60:                                             ; preds = %bb58
  br i1 undef, label %bb61, label %bb45

bb61:                                             ; preds = %bb60
  br i1 undef, label %bb63, label %bb62

bb62:                                             ; preds = %bb61
  br label %bb64

bb63:                                             ; preds = %bb61
  br label %bb64

bb64:                                             ; preds = %bb63, %bb62
  br i1 undef, label %bb66, label %bb45

bb66:                                             ; preds = %bb64
  br i1 undef, label %bb67, label %bb45

bb67:                                             ; preds = %bb66
  br i1 undef, label %bb69, label %bb68

bb68:                                             ; preds = %bb67
  br label %bb70

bb69:                                             ; preds = %bb67
  br label %bb70

bb70:                                             ; preds = %bb69, %bb68
  br i1 undef, label %bb72, label %bb45

bb72:                                             ; preds = %bb70
  br i1 undef, label %bb73, label %bb45

bb73:                                             ; preds = %bb72
  br i1 undef, label %bb75, label %bb74

bb74:                                             ; preds = %bb73
  br label %bb76

bb75:                                             ; preds = %bb73
  br label %bb76

bb76:                                             ; preds = %bb75, %bb74
  br i1 undef, label %bb78, label %bb45

bb78:                                             ; preds = %bb76
  br i1 undef, label %bb79, label %bb45

bb79:                                             ; preds = %bb78
  br i1 undef, label %bb81, label %bb80

bb80:                                             ; preds = %bb79
  br label %bb82

bb81:                                             ; preds = %bb79
  br label %bb82

bb82:                                             ; preds = %bb81, %bb80
  br i1 undef, label %bb84, label %bb45

bb84:                                             ; preds = %bb82
  br i1 undef, label %bb85, label %bb45

bb85:                                             ; preds = %bb84
  br i1 undef, label %bb87, label %bb86

bb86:                                             ; preds = %bb85
  br label %bb88

bb87:                                             ; preds = %bb85
  br label %bb88

bb88:                                             ; preds = %bb87, %bb86
  br i1 undef, label %bb90, label %bb45

bb90:                                             ; preds = %bb88
  br i1 undef, label %bb91, label %bb45

bb91:                                             ; preds = %bb90
  br i1 undef, label %bb93, label %bb92

bb92:                                             ; preds = %bb91
  br label %bb94

bb93:                                             ; preds = %bb91
  br label %bb94

bb94:                                             ; preds = %bb93, %bb92
  br i1 undef, label %bb96, label %bb45

bb96:                                             ; preds = %bb94
  br i1 undef, label %bb97, label %bb45

bb97:                                             ; preds = %bb96
  br i1 undef, label %bb99, label %bb98

bb98:                                             ; preds = %bb97
  br label %bb100

bb99:                                             ; preds = %bb97
  br label %bb100

bb100:                                            ; preds = %bb99, %bb98
  br i1 undef, label %bb102, label %bb45

bb102:                                            ; preds = %bb100
  br i1 undef, label %bb104, label %bb45

bb104:                                            ; preds = %bb102
  br i1 undef, label %bb108, label %bb105

bb105:                                            ; preds = %bb104
  br i1 undef, label %bb108, label %bb45

bb106:                                            ; preds = %bb47
  unreachable

bb108:                                            ; preds = %bb105, %bb104
  br i1 undef, label %bb110, label %bb45

bb110:                                            ; preds = %bb108
  br i1 undef, label %bb112, label %bb45

bb112:                                            ; preds = %bb110
  br i1 undef, label %bb114, label %bb45

bb114:                                            ; preds = %bb112
  br i1 undef, label %bb116, label %bb45

bb116:                                            ; preds = %bb114
  br i1 undef, label %bb118, label %bb45

bb118:                                            ; preds = %bb116
  br i1 undef, label %bb120, label %bb45

bb120:                                            ; preds = %bb118
  br i1 undef, label %bb122, label %bb45

bb122:                                            ; preds = %bb120
  br i1 undef, label %bb124, label %bb45

bb124:                                            ; preds = %bb122
  br i1 undef, label %bb126, label %bb45

bb126:                                            ; preds = %bb124
  br i1 undef, label %bb128, label %bb45

bb128:                                            ; preds = %bb126
  br i1 undef, label %bb130, label %bb45

bb130:                                            ; preds = %bb128
  br i1 undef, label %bb132, label %bb45

bb132:                                            ; preds = %bb130
  br i1 undef, label %bb135, label %bb30

bb134:                                            ; preds = %bb53
  unreachable

bb135:                                            ; preds = %bb132
  br i1 undef, label %bb139, label %bb136

bb136:                                            ; preds = %bb135
  br i1 undef, label %bb138, label %bb30

bb138:                                            ; preds = %bb136
  br label %bb139

bb139:                                            ; preds = %bb138, %bb135
  br i1 undef, label %bb140, label %bb141

bb140:                                            ; preds = %bb139
  unreachable

bb141:                                            ; preds = %bb139
  br i1 undef, label %bb142, label %bb215

bb142:                                            ; preds = %bb141
  br i1 undef, label %bb144, label %bb143

bb143:                                            ; preds = %bb142
  br label %bb145

bb144:                                            ; preds = %bb142
  br label %bb145

bb145:                                            ; preds = %bb144, %bb143
  br i1 undef, label %bb146, label %bb151

bb146:                                            ; preds = %bb145
  br i1 undef, label %bb148, label %bb153

bb148:                                            ; preds = %bb146
  br i1 undef, label %bb155, label %bb149

bb149:                                            ; preds = %bb148
  br i1 undef, label %bb150, label %bb153

bb150:                                            ; preds = %bb149
  br label %bb155

bb151:                                            ; preds = %bb145
  unreachable

bb153:                                            ; preds = %bb158, %bb149, %bb146
  unreachable

bb155:                                            ; preds = %bb150, %bb148
  br i1 undef, label %bb157, label %bb156

bb156:                                            ; preds = %bb155
  br label %bb158

bb157:                                            ; preds = %bb155
  br label %bb158

bb158:                                            ; preds = %bb157, %bb156
  br i1 undef, label %bb160, label %bb153

bb160:                                            ; preds = %bb158
  br i1 undef, label %bb162, label %bb161

bb161:                                            ; preds = %bb160
  br label %bb163

bb162:                                            ; preds = %bb160
  br label %bb163

bb163:                                            ; preds = %bb162, %bb161
  br i1 undef, label %bb165, label %bb164

bb164:                                            ; preds = %bb163
  br label %bb165

bb165:                                            ; preds = %bb164, %bb163
  br i1 undef, label %bb170, label %bb166

bb166:                                            ; preds = %bb165
  br i1 undef, label %bb167, label %bb168

bb167:                                            ; preds = %bb166
  unreachable

bb168:                                            ; preds = %bb166
  unreachable

bb170:                                            ; preds = %bb165
  br i1 undef, label %bb215, label %bb171

bb171:                                            ; preds = %bb170
  br i1 undef, label %bb173, label %bb30

bb173:                                            ; preds = %bb171
  br i1 undef, label %bb174, label %bb215

bb174:                                            ; preds = %bb173
  br i1 undef, label %bb176, label %bb30

bb176:                                            ; preds = %bb174
  br i1 undef, label %bb178, label %bb30

bb178:                                            ; preds = %bb176
  br i1 undef, label %bb179, label %bb193

bb179:                                            ; preds = %bb178
  br i1 undef, label %bb181, label %bb180

bb180:                                            ; preds = %bb179
  br label %bb182

bb181:                                            ; preds = %bb179
  br label %bb182

bb182:                                            ; preds = %bb181, %bb180
  br i1 undef, label %bb184, label %bb30

bb184:                                            ; preds = %bb182
  %tmp185 = call i8* @returner()
  br i1 undef, label %bb186, label %bb195

bb186:                                            ; preds = %bb184
  %tmp188 = call i8* @objc_retainAutoreleasedReturnValue(i8* %tmp185)
  %tmp189 = call i8* @objc_retain(i8* %tmp188)
  call void @objc_release(i8* %tmp189), !clang.imprecise_release !0
  br i1 undef, label %bb197, label %bb190

bb190:                                            ; preds = %bb186
  br i1 undef, label %bb192, label %bb195

bb192:                                            ; preds = %bb190
  br i1 undef, label %bb197, label %bb195

bb193:                                            ; preds = %bb178
  br label %bb213

bb195:                                            ; preds = %bb192, %bb190, %bb184
  unreachable

bb197:                                            ; preds = %bb192, %bb186
  br i1 undef, label %bb198, label %bb215

bb198:                                            ; preds = %bb197
  br i1 undef, label %bb202, label %bb199

bb199:                                            ; preds = %bb198
  br i1 undef, label %bb201, label %bb30

bb201:                                            ; preds = %bb199
  br label %bb202

bb202:                                            ; preds = %bb201, %bb198
  br i1 undef, label %bb206, label %bb203

bb203:                                            ; preds = %bb202
  br i1 undef, label %bb205, label %bb30

bb205:                                            ; preds = %bb203
  br label %bb206

bb206:                                            ; preds = %bb205, %bb202
  br i1 undef, label %bb210, label %bb207

bb207:                                            ; preds = %bb206
  br i1 undef, label %bb209, label %bb30

bb209:                                            ; preds = %bb207
  br label %bb210

bb210:                                            ; preds = %bb209, %bb206
  br i1 undef, label %bb212, label %bb30

bb212:                                            ; preds = %bb210
  unreachable

bb213:                                            ; preds = %bb193, %bb30
  resume { i8*, i32 } undef

bb214:                                            ; preds = %bb42
  br label %bb219

bb215:                                            ; preds = %bb197, %bb173, %bb170, %bb141
  br i1 undef, label %bb217, label %bb216

bb216:                                            ; preds = %bb215
  br label %bb217

bb217:                                            ; preds = %bb216, %bb215
  br i1 undef, label %bb219, label %bb218

bb218:                                            ; preds = %bb217
  br label %bb219

bb219:                                            ; preds = %bb218, %bb217, %bb214
  br i1 undef, label %bb221, label %bb220

bb220:                                            ; preds = %bb219
  unreachable

bb221:                                            ; preds = %bb219
  unreachable

bb222:                                            ; preds = %bb20, %bb19
  ret void
}


!0 = metadata !{}
