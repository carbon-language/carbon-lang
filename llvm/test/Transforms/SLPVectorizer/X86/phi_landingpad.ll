; RUN: opt < %s -slp-vectorizer -mtriple=x86_64-apple-macosx10.9.0 -disable-output

target datalayout = "f64:64:64-v64:64:64"

define void @test_phi_in_landingpad() {
entry:
  invoke void @foo()
          to label %inner unwind label %lpad

inner:
  %x0 = fsub double undef, undef
  %y0 = fsub double undef, undef
  invoke void @foo()
          to label %done unwind label %lpad

lpad:
  %x1 = phi double [ undef, %entry ], [ undef, %inner ]
  %y1 = phi double [ undef, %entry ], [ undef, %inner ]
  landingpad { i8*, i32 } personality i8*
          bitcast (i32 (...)* @__gxx_personality_v0 to i8*) catch i8* null
  br label %done

done:
  phi double [ %x0, %inner ], [ %x1, %lpad ]
  phi double [ %y0, %inner ], [ %y1, %lpad ]
  ret void
}

declare void @foo()

declare i32 @__gxx_personality_v0(...)
