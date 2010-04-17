; RUN: llc %s -o -

; PR6332
%struct.AVCodecTag = type opaque
@ff_codec_bmp_tags = external global [0 x %struct.AVCodecTag]
@tags = global [1 x %struct.AVCodecTag*] [%struct.AVCodecTag* getelementptr
inbounds ([0 x %struct.AVCodecTag]* @ff_codec_bmp_tags, i32 0, i32 0)]

