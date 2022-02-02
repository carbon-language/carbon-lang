; RUN: llc < %s -mtriple=arm64-apple-ios
; rdar://11849816

@shlib_path_substitutions = external hidden unnamed_addr global i8**, align 8

declare i64 @llvm.objectsize.i64(i8*, i1) nounwind readnone

declare noalias i8* @xmalloc(i64) optsize

declare i64 @strlen(i8* nocapture) nounwind readonly optsize

declare i8* @__strcpy_chk(i8*, i8*, i64) nounwind optsize

declare i8* @__strcat_chk(i8*, i8*, i64) nounwind optsize

declare noalias i8* @xstrdup(i8*) optsize

define i8* @dyld_fix_path(i8* %path) nounwind optsize ssp {
entry:
  br i1 undef, label %if.end56, label %for.cond

for.cond:                                         ; preds = %entry
  br i1 undef, label %for.cond10, label %for.body

for.body:                                         ; preds = %for.cond
  unreachable

for.cond10:                                       ; preds = %for.cond
  br i1 undef, label %if.end56, label %for.body14

for.body14:                                       ; preds = %for.cond10
  %call22 = tail call i64 @strlen(i8* undef) nounwind optsize
  %sext = shl i64 %call22, 32
  %conv30 = ashr exact i64 %sext, 32
  %add29 = sub i64 0, %conv30
  %sub = add i64 %add29, 0
  %add31 = shl i64 %sub, 32
  %sext59 = add i64 %add31, 4294967296
  %conv33 = ashr exact i64 %sext59, 32
  %call34 = tail call noalias i8* @xmalloc(i64 %conv33) nounwind optsize
  br i1 undef, label %cond.false45, label %cond.true43

cond.true43:                                      ; preds = %for.body14
  unreachable

cond.false45:                                     ; preds = %for.body14
  %add.ptr = getelementptr inbounds i8, i8* %path, i64 %conv30
  unreachable

if.end56:                                         ; preds = %for.cond10, %entry
  ret i8* null
}

declare i32 @strncmp(i8* nocapture, i8* nocapture, i64) nounwind readonly optsize

declare i8* @strcpy(i8*, i8* nocapture) nounwind
