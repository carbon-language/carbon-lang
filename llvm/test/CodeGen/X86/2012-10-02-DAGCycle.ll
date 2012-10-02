; RUN: llc -mtriple=i386-apple-macosx -relocation-model=pic < %s
; rdar://12393897

%TRp = type { i32, %TRH*, i32, i32 }
%TRH = type { i8*, i8*, i8*, i8*, {}* }

define i32 @t(%TRp* inreg %rp) nounwind optsize ssp {
entry:
  %handler = getelementptr inbounds %TRp* %rp, i32 0, i32 1
  %0 = load %TRH** %handler, align 4
  %sync = getelementptr inbounds %TRH* %0, i32 0, i32 4
  %sync12 = load {}** %sync, align 4
  %1 = bitcast {}* %sync12 to i32 (%TRp*)*
  %call = tail call i32 %1(%TRp* inreg %rp) nounwind optsize
  ret i32 %call
}
