; RUN: llc %s -o -

; PR6332
%struct.AVCodecTag = type opaque
@ff_codec_bmp_tags = external global [0 x %struct.AVCodecTag]
@tags = global [1 x %struct.AVCodecTag*] [%struct.AVCodecTag* getelementptr
inbounds ([0 x %struct.AVCodecTag]* @ff_codec_bmp_tags, i32 0, i32 0)]


; rdar://8878965

%struct.CAMERA = type { [3 x double], [3 x double], [3 x double], [3 x double], [3 x double], [3 x double], double, double, i32, double, double, i32, double, i32* }

define void @Parse_Camera(%struct.CAMERA** nocapture %Camera_Ptr) nounwind {
entry:
%.pre = load %struct.CAMERA** %Camera_Ptr, align 4
%0 = getelementptr inbounds %struct.CAMERA* %.pre, i32 0, i32 1, i32 0
%1 = getelementptr inbounds %struct.CAMERA* %.pre, i32 0, i32 1, i32 2
br label %bb32

bb32:                                             ; preds = %bb6
%2 = load double* %0, align 4
%3 = load double* %1, align 4
%4 = load double* %0, align 4
call void @Parse_Vector(double* %0) nounwind
%5 = call i32 @llvm.objectsize.i32(i8* undef, i1 false)
%6 = icmp eq i32 %5, -1
br i1 %6, label %bb34, label %bb33

bb33:                                             ; preds = %bb32
unreachable

bb34:                                             ; preds = %bb32
unreachable

}

declare void @Parse_Vector(double*)
declare i32 @llvm.objectsize.i32(i8*, i1)

