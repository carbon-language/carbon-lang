; RUN: llc -march=hexagon -mcpu=hexagonv5 -O3 < %s | FileCheck %s

; Test that we do not generate a hardware loop due to a potential underflow.

; CHECK-NOT: loop0

%struct.3 = type { i8*, i8, i8, i32, i32, i16, i16, i16, i16, i16, i16, i16, %struct.2* }
%struct.2 = type { i16, i16, i16, i16, %struct.1* }
%struct.1 = type { %struct.1*, %struct.0*, i32, i32, i16, [2 x i16], [2 x i16], i16 }
%struct.0 = type { %struct.0*, i32, i32, i32, i32, i32, i32, i16, i16, i16, i8, i8, i8, i8 }

@pairArray = external global i32**
@carray = external global %struct.3**

define void @test() #0 {
entry:
  %0 = load i32**, i32*** @pairArray, align 4
  %1 = load %struct.3**, %struct.3*** @carray, align 4
  br i1 undef, label %for.end110, label %for.body

for.body:
  %row.0199 = phi i32 [ %inc109, %for.inc108 ], [ 1, %entry ]
  %arrayidx = getelementptr inbounds i32*, i32** %0, i32 %row.0199
  %2 = load i32*, i32** %arrayidx, align 4
  br i1 undef, label %for.body48, label %for.inc108

for.cond45:
  %cmp46 = icmp sgt i32 %dec58, 0
  br i1 %cmp46, label %for.body48, label %for.inc108

for.body48:
  %i.1190 = phi i32 [ %dec58, %for.cond45 ], [ 0, %for.body ]
  %arrayidx50 = getelementptr inbounds i32, i32* %2, i32 %i.1190
  %3 = load i32, i32* %arrayidx50, align 4
  %cmp53 = icmp slt i32 %3, 0
  %dec58 = add nsw i32 %i.1190, -1
  br i1 %cmp53, label %for.end59, label %for.cond45

for.end59:
  %cmp60 = icmp slt i32 %i.1190, 0
  br i1 %cmp60, label %if.then65, label %for.inc108

if.then65:
  br label %for.body80

for.body80:
  %j.1196.in = phi i32 [ %j.1196, %for.body80 ], [ %i.1190, %if.then65 ]
  %j.1196 = add nsw i32 %j.1196.in, 1
  %arrayidx81 = getelementptr inbounds i32, i32* %2, i32 %j.1196
  %4 = load i32, i32* %arrayidx81, align 4
  %arrayidx82 = getelementptr inbounds %struct.3*, %struct.3** %1, i32 %4
  %5 = load %struct.3*, %struct.3** %arrayidx82, align 4
  %cxcenter83 = getelementptr inbounds %struct.3, %struct.3* %5, i32 0, i32 3
  store i32 0, i32* %cxcenter83, align 4
  %6 = load i32, i32* %arrayidx81, align 4
  %arrayidx87 = getelementptr inbounds i32, i32* %2, i32 %j.1196.in
  store i32 %6, i32* %arrayidx87, align 4
  %exitcond = icmp eq i32 %j.1196, 0
  br i1 %exitcond, label %for.inc108, label %for.body80

for.inc108:
  %inc109 = add nsw i32 %row.0199, 1
  br i1 undef, label %for.body, label %for.end110

for.end110:
  ret void
}
