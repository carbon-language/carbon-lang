; RUN: opt -gvn -disable-output %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

%struct.attribute_spec = type { i8*, i32, i32, i8, i8, i8 }

@attribute_tables = external global [4 x %struct.attribute_spec*] ; <[4 x %struct.attribute_spec*]*> [#uses=2]

define void @decl_attributes() nounwind {
entry:
  br label %bb69.i

bb69.i:                                           ; preds = %bb57.i.preheader
  %tmp4 = getelementptr inbounds [4 x %struct.attribute_spec*]* @attribute_tables, i32 0, i32 undef ; <%struct.attribute_spec**> [#uses=1]
  %tmp5 = getelementptr inbounds [4 x %struct.attribute_spec*]* @attribute_tables, i32 0, i32 undef ; <%struct.attribute_spec**> [#uses=1]
  %tmp3 = load %struct.attribute_spec** %tmp4, align 4 ; <%struct.attribute_spec*> [#uses=1]
  br label %bb65.i

bb65.i:                                           ; preds = %bb65.i.preheader, %bb64.i
  %storemerge6.i = phi i32 [ 1, %bb64.i ], [ 0, %bb69.i ] ; <i32> [#uses=3]
  %scevgep14 = getelementptr inbounds %struct.attribute_spec* %tmp3, i32 %storemerge6.i, i32 0 ; <i8**> [#uses=1]
  %tmp2 = load i8** %scevgep14, align 4           ; <i8*> [#uses=0]
  %tmp = load %struct.attribute_spec** %tmp5, align 4 ; <%struct.attribute_spec*> [#uses=1]
  %scevgep1516 = getelementptr inbounds %struct.attribute_spec* %tmp, i32 %storemerge6.i, i32 0 ; <i8**> [#uses=0]
  unreachable

bb64.i:                                           ; Unreachable
  br label %bb65.i

bb66.i:                                           ; Unreachable
  br label %bb69.i
}
