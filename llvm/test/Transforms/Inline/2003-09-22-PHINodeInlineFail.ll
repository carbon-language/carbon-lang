; RUN: opt < %s -inline -disable-output

define i32 @main() {
entry:
        invoke void @__main( )
                        to label %LongJmpBlkPost unwind label %LongJmpBlkPre

LongJmpBlkPost:
        ret i32 0

LongJmpBlkPre:
        %i.3 = phi i32 [ 0, %entry ], [ 0, %entry ]             ; <i32> [#uses=0]
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 cleanup
        ret i32 0
}

define void @__main() {
        ret void
}

declare i32 @__gxx_personality_v0(...)
