; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct._exon_t.12.103.220.363.480.649.740.857.1039.1065.1078.1091.1117.1130.1156.1169.1195.1221.1234.1286.1299.1312.1338.1429.1455.1468.1494.1520.1884.1897.1975.2066.2105.2170.2171 = type { i32, i32, i32, i32, i32, i32, [8 x i8] }

define void @SIM4() {
entry:
  br i1 undef, label %return, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  br i1 undef, label %return, label %if.end

if.end:                                           ; preds = %lor.lhs.false
  br i1 undef, label %for.end605, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %if.end
  br label %for.body

for.body:                                         ; preds = %for.inc603, %for.body.lr.ph
  br i1 undef, label %for.inc603, label %if.end12

if.end12:                                         ; preds = %for.body
  br i1 undef, label %land.lhs.true, label %land.lhs.true167

land.lhs.true:                                    ; preds = %if.end12
  br i1 undef, label %if.then17, label %land.lhs.true167

if.then17:                                        ; preds = %land.lhs.true
  br i1 undef, label %if.end98, label %land.rhs.lr.ph

land.rhs.lr.ph:                                   ; preds = %if.then17
  unreachable

if.end98:                                         ; preds = %if.then17
  %from299 = getelementptr inbounds %struct._exon_t.12.103.220.363.480.649.740.857.1039.1065.1078.1091.1117.1130.1156.1169.1195.1221.1234.1286.1299.1312.1338.1429.1455.1468.1494.1520.1884.1897.1975.2066.2105.2170.2171* undef, i64 0, i32 1
  br i1 undef, label %land.lhs.true167, label %if.then103

if.then103:                                       ; preds = %if.end98
  %.sub100 = select i1 undef, i32 250, i32 undef
  %mul114 = shl nsw i32 %.sub100, 2
  %from1115 = getelementptr inbounds %struct._exon_t.12.103.220.363.480.649.740.857.1039.1065.1078.1091.1117.1130.1156.1169.1195.1221.1234.1286.1299.1312.1338.1429.1455.1468.1494.1520.1884.1897.1975.2066.2105.2170.2171* undef, i64 0, i32 0
  %cond125 = select i1 undef, i32 undef, i32 %mul114
  br label %for.cond.i

for.cond.i:                                       ; preds = %land.rhs.i874, %if.then103
  %row.0.i = phi i32 [ undef, %land.rhs.i874 ], [ %.sub100, %if.then103 ]
  %col.0.i = phi i32 [ undef, %land.rhs.i874 ], [ %cond125, %if.then103 ]
  br i1 undef, label %land.rhs.i874, label %for.end.i

land.rhs.i874:                                    ; preds = %for.cond.i
  br i1 undef, label %for.cond.i, label %for.end.i

for.end.i:                                        ; preds = %land.rhs.i874, %for.cond.i
  br i1 undef, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %for.end.i
  %add14.i = add nsw i32 %row.0.i, undef
  %add15.i = add nsw i32 %col.0.i, undef
  br label %extend_bw.exit

if.end.i:                                         ; preds = %for.end.i
  %add16.i = add i32 %cond125, %.sub100
  %cmp26514.i = icmp slt i32 %add16.i, 0
  br i1 %cmp26514.i, label %for.end33.i, label %for.body28.lr.ph.i

for.body28.lr.ph.i:                               ; preds = %if.end.i
  br label %for.end33.i

for.end33.i:                                      ; preds = %for.body28.lr.ph.i, %if.end.i
  br i1 undef, label %for.end58.i, label %for.body52.lr.ph.i

for.body52.lr.ph.i:                               ; preds = %for.end33.i
  br label %for.end58.i

for.end58.i:                                      ; preds = %for.body52.lr.ph.i, %for.end33.i
  br label %while.cond260.i

while.cond260.i:                                  ; preds = %land.rhs263.i, %for.end58.i
  br i1 undef, label %land.rhs263.i, label %while.end275.i

land.rhs263.i:                                    ; preds = %while.cond260.i
  br i1 undef, label %while.cond260.i, label %while.end275.i

while.end275.i:                                   ; preds = %land.rhs263.i, %while.cond260.i
  br label %extend_bw.exit

extend_bw.exit:                                   ; preds = %while.end275.i, %if.then.i
  %add14.i1262 = phi i32 [ %add14.i, %if.then.i ], [ undef, %while.end275.i ]
  %add15.i1261 = phi i32 [ %add15.i, %if.then.i ], [ undef, %while.end275.i ]
  br i1 false, label %if.then157, label %land.lhs.true167

if.then157:                                       ; preds = %extend_bw.exit
  %add158 = add nsw i32 %add14.i1262, 1
  store i32 %add158, i32* %from299, align 4
  %add160 = add nsw i32 %add15.i1261, 1
  store i32 %add160, i32* %from1115, align 4
  br label %land.lhs.true167

land.lhs.true167:                                 ; preds = %if.then157, %extend_bw.exit, %if.end98, %land.lhs.true, %if.end12
  unreachable

for.inc603:                                       ; preds = %for.body
  br i1 undef, label %for.body, label %for.end605

for.end605:                                       ; preds = %for.inc603, %if.end
  unreachable

return:                                           ; preds = %lor.lhs.false, %entry
  ret void
}

