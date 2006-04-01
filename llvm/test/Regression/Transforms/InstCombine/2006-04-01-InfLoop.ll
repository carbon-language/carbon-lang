; RUN: llvm-as < %s | opt -instcombine -disable-output

	%struct.DecRefPicMarking_s = type { int, int, int, int, int, %struct.DecRefPicMarking_s* }
	%struct.datapartition = type { %typedef.Bitstream*, %typedef.DecodingEnvironment, int (%struct.syntaxelement*, %struct.img_par*, %struct.inp_par*, %struct.datapartition*)* }
	%struct.img_par = type { int, uint, uint, int, int*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, [16 x [16 x ushort]], [6 x [32 x int]], [16 x [16 x int]], [4 x [12 x [4 x [4 x int]]]], [16 x int], int**, int*, int***, int**, int, int, int, int, %typedef.Slice*, %struct.macroblock*, int, int, int, int, int, int, int**, %struct.DecRefPicMarking_s*, int, int, int, int, int, int, int, uint, int, int, int, uint, uint, uint, uint, int, [3 x int], int, uint, int, uint, int, int, int, uint, uint, int, int, int, int, uint, uint, int***, int***, int****, int, int, uint, int, int, int, int, uint, uint, uint, uint, uint, uint, uint, int, int, int, int, int, int, int, int, int, int, int, uint, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, %struct.timeb, %struct.timeb, int, int, int, int, int, uint, int, int }
	%struct.inp_par = type { [100 x sbyte], [100 x sbyte], [100 x sbyte], int, int, int, int, int, int, int }
	%struct.macroblock = type { int, int, int, %struct.macroblock*, %struct.macroblock*, int, [2 x [4 x [4 x [2 x int]]]], int, long, long, int, int, [4 x int], [4 x int], int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int }
	%struct.pix_pos = type { int, int, int, int, int, int }
	%struct.storable_picture = type { uint, int, int, int, int, [50 x [6 x [33 x long]]], [50 x [6 x [33 x long]]], [50 x [6 x [33 x long]]], [50 x [6 x [33 x long]]], uint, int, int, int, int, int, int, int, short, int, int, int, int, int, int, int, uint, uint, ushort**, ushort***, ubyte*, short**, sbyte***, long***, long***, short****, ubyte**, ubyte**, %struct.storable_picture*, %struct.storable_picture*, %struct.storable_picture*, int, int, int, int, int, int, int, int, int, int, int, int, int, [2 x int], int, %struct.DecRefPicMarking_s*, int }
	%struct.syntaxelement = type { int, int, int, int, int, uint, int, int, void (int, int, int*, int*)*, void (%struct.syntaxelement*, %struct.inp_par*, %struct.img_par*, %typedef.DecodingEnvironment*)* }
	%struct.timeb = type { int, ushort, short, short }
	%typedef.BiContextType = type { ushort, ubyte }
	%typedef.Bitstream = type { int, int, int, int, ubyte*, int }
	%typedef.DecodingEnvironment = type { uint, uint, uint, uint, int, ubyte*, int* }
	%typedef.MotionInfoContexts = type { [4 x [11 x %typedef.BiContextType]], [2 x [9 x %typedef.BiContextType]], [2 x [10 x %typedef.BiContextType]], [2 x [6 x %typedef.BiContextType]], [4 x %typedef.BiContextType], [4 x %typedef.BiContextType], [3 x %typedef.BiContextType] }
	%typedef.Slice = type { int, int, int, int, uint, int, int, int, int, %struct.datapartition*, %typedef.MotionInfoContexts*, %typedef.TextureInfoContexts*, int, int*, int*, int*, int, int*, int*, int*, int (%struct.img_par*, %struct.inp_par*)*, int, int, int, int }
	%typedef.TextureInfoContexts = type { [2 x %typedef.BiContextType], [4 x %typedef.BiContextType], [3 x [4 x %typedef.BiContextType]], [10 x [4 x %typedef.BiContextType]], [10 x [15 x %typedef.BiContextType]], [10 x [15 x %typedef.BiContextType]], [10 x [5 x %typedef.BiContextType]], [10 x [5 x %typedef.BiContextType]], [10 x [15 x %typedef.BiContextType]], [10 x [15 x %typedef.BiContextType]] }
%dec_picture = external global %struct.storable_picture*		; <%struct.storable_picture**> [#uses=1]
%last_dquant = external global int		; <int*> [#uses=1]

implementation   ; Functions:

void %readCBP_CABAC(%struct.syntaxelement* %se, %struct.inp_par* %inp, %struct.img_par* %img.1, %typedef.DecodingEnvironment* %dep_dp) {
entry:
	%block_a = alloca %struct.pix_pos		; <%struct.pix_pos*> [#uses=5]
	%tmp.1 = getelementptr %struct.img_par* %img.1, int 0, uint 37		; <%typedef.Slice**> [#uses=1]
	%tmp.2 = load %typedef.Slice** %tmp.1		; <%typedef.Slice*> [#uses=1]
	%tmp.3 = getelementptr %typedef.Slice* %tmp.2, int 0, uint 11		; <%typedef.TextureInfoContexts**> [#uses=1]
	%tmp.4 = load %typedef.TextureInfoContexts** %tmp.3		; <%typedef.TextureInfoContexts*> [#uses=3]
	%tmp.6 = getelementptr %struct.img_par* %img.1, int 0, uint 38		; <%struct.macroblock**> [#uses=1]
	%tmp.7 = load %struct.macroblock** %tmp.6		; <%struct.macroblock*> [#uses=1]
	%tmp.9 = getelementptr %struct.img_par* %img.1, int 0, uint 1		; <uint*> [#uses=1]
	%tmp.10 = load uint* %tmp.9		; <uint> [#uses=1]
	%tmp.11 = cast uint %tmp.10 to int		; <int> [#uses=1]
	%tmp.12 = getelementptr %struct.macroblock* %tmp.7, int %tmp.11		; <%struct.macroblock*> [#uses=18]
	br label %loopentry.0

loopentry.0:		; preds = %loopexit.1, %entry
	%mask.1 = phi int [ undef, %entry ], [ %mask.0, %loopexit.1 ]		; <int> [#uses=1]
	%cbp_bit.1 = phi int [ undef, %entry ], [ %cbp_bit.0, %loopexit.1 ]		; <int> [#uses=1]
	%cbp.2 = phi int [ 0, %entry ], [ %cbp.1, %loopexit.1 ]		; <int> [#uses=5]
	%curr_cbp_ctx.1 = phi int [ undef, %entry ], [ %curr_cbp_ctx.0, %loopexit.1 ]		; <int> [#uses=1]
	%b.2 = phi int [ undef, %entry ], [ %b.1, %loopexit.1 ]		; <int> [#uses=1]
	%a.2 = phi int [ undef, %entry ], [ %a.1, %loopexit.1 ]		; <int> [#uses=1]
	%mb_y.0 = phi int [ 0, %entry ], [ %tmp.152, %loopexit.1 ]		; <int> [#uses=7]
	%mb_x.0 = phi int [ undef, %entry ], [ %mb_x.1, %loopexit.1 ]		; <int> [#uses=0]
	%tmp.14 = setle int %mb_y.0, 3		; <bool> [#uses=2]
	%tmp.15 = cast bool %tmp.14 to int		; <int> [#uses=0]
	br bool %tmp.14, label %no_exit.0, label %loopexit.0

no_exit.0:		; preds = %loopentry.0
	br label %loopentry.1

loopentry.1:		; preds = %endif.7, %no_exit.0
	%mask.0 = phi int [ %mask.1, %no_exit.0 ], [ %tmp.131, %endif.7 ]		; <int> [#uses=1]
	%cbp_bit.0 = phi int [ %cbp_bit.1, %no_exit.0 ], [ %tmp.142, %endif.7 ]		; <int> [#uses=1]
	%cbp.1 = phi int [ %cbp.2, %no_exit.0 ], [ %cbp.0, %endif.7 ]		; <int> [#uses=5]
	%curr_cbp_ctx.0 = phi int [ %curr_cbp_ctx.1, %no_exit.0 ], [ %tmp.125, %endif.7 ]		; <int> [#uses=1]
	%b.1 = phi int [ %b.2, %no_exit.0 ], [ %b.0, %endif.7 ]		; <int> [#uses=1]
	%a.1 = phi int [ %a.2, %no_exit.0 ], [ %a.0, %endif.7 ]		; <int> [#uses=1]
	%mb_x.1 = phi int [ 0, %no_exit.0 ], [ %tmp.150, %endif.7 ]		; <int> [#uses=9]
	%tmp.17 = setle int %mb_x.1, 3		; <bool> [#uses=2]
	%tmp.18 = cast bool %tmp.17 to int		; <int> [#uses=0]
	br bool %tmp.17, label %no_exit.1, label %loopexit.1

no_exit.1:		; preds = %loopentry.1
	%tmp.20 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 12		; <[4 x int]*> [#uses=1]
	%tmp.22 = div int %mb_x.1, 2		; <int> [#uses=1]
	%tmp.24 = add int %tmp.22, %mb_y.0		; <int> [#uses=1]
	%tmp.25 = getelementptr [4 x int]* %tmp.20, int 0, int %tmp.24		; <int*> [#uses=1]
	%tmp.26 = load int* %tmp.25		; <int> [#uses=1]
	%tmp.27 = seteq int %tmp.26, 11		; <bool> [#uses=2]
	%tmp.28 = cast bool %tmp.27 to int		; <int> [#uses=0]
	br bool %tmp.27, label %then.0, label %else.0

then.0:		; preds = %no_exit.1
	br label %endif.0

else.0:		; preds = %no_exit.1
	br label %endif.0

endif.0:		; preds = %else.0, %then.0
	%tmp.30 = seteq int %mb_y.0, 0		; <bool> [#uses=2]
	%tmp.31 = cast bool %tmp.30 to int		; <int> [#uses=0]
	br bool %tmp.30, label %then.1, label %else.1

then.1:		; preds = %endif.0
	%tmp.33 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 3		; <%struct.macroblock**> [#uses=1]
	%tmp.34 = load %struct.macroblock** %tmp.33		; <%struct.macroblock*> [#uses=1]
	%tmp.35 = cast %struct.macroblock* %tmp.34 to sbyte*		; <sbyte*> [#uses=1]
	%tmp.36 = seteq sbyte* %tmp.35, null		; <bool> [#uses=2]
	%tmp.37 = cast bool %tmp.36 to int		; <int> [#uses=0]
	br bool %tmp.36, label %then.2, label %else.2

then.2:		; preds = %then.1
	br label %endif.1

else.2:		; preds = %then.1
	%tmp.39 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 3		; <%struct.macroblock**> [#uses=1]
	%tmp.40 = load %struct.macroblock** %tmp.39		; <%struct.macroblock*> [#uses=1]
	%tmp.41 = getelementptr %struct.macroblock* %tmp.40, int 0, uint 5		; <int*> [#uses=1]
	%tmp.42 = load int* %tmp.41		; <int> [#uses=1]
	%tmp.43 = seteq int %tmp.42, 14		; <bool> [#uses=2]
	%tmp.44 = cast bool %tmp.43 to int		; <int> [#uses=0]
	br bool %tmp.43, label %then.3, label %else.3

then.3:		; preds = %else.2
	br label %endif.1

else.3:		; preds = %else.2
	%tmp.46 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 3		; <%struct.macroblock**> [#uses=1]
	%tmp.47 = load %struct.macroblock** %tmp.46		; <%struct.macroblock*> [#uses=1]
	%tmp.48 = getelementptr %struct.macroblock* %tmp.47, int 0, uint 7		; <int*> [#uses=1]
	%tmp.49 = load int* %tmp.48		; <int> [#uses=1]
	%tmp.51 = div int %mb_x.1, 2		; <int> [#uses=1]
	%tmp.52 = add int %tmp.51, 2		; <int> [#uses=1]
	%tmp.53 = cast int %tmp.52 to ubyte		; <ubyte> [#uses=1]
	%tmp.54 = shr int %tmp.49, ubyte %tmp.53		; <int> [#uses=1]
	%tmp.55 = cast int %tmp.54 to uint		; <uint> [#uses=1]
	%tmp.57 = xor uint %tmp.55, 1		; <uint> [#uses=1]
	%tmp.58 = cast uint %tmp.57 to int		; <int> [#uses=1]
	%tmp.59 = and int %tmp.58, 1		; <int> [#uses=1]
	br label %endif.1

else.1:		; preds = %endif.0
	%tmp.62 = div int %mb_x.1, 2		; <int> [#uses=1]
	%tmp.63 = cast int %tmp.62 to ubyte		; <ubyte> [#uses=1]
	%tmp.64 = shr int %cbp.1, ubyte %tmp.63		; <int> [#uses=1]
	%tmp.65 = cast int %tmp.64 to uint		; <uint> [#uses=1]
	%tmp.67 = xor uint %tmp.65, 1		; <uint> [#uses=1]
	%tmp.68 = cast uint %tmp.67 to int		; <int> [#uses=1]
	%tmp.69 = and int %tmp.68, 1		; <int> [#uses=1]
	br label %endif.1

endif.1:		; preds = %else.1, %else.3, %then.3, %then.2
	%b.0 = phi int [ 0, %then.2 ], [ 0, %then.3 ], [ %tmp.59, %else.3 ], [ %tmp.69, %else.1 ]		; <int> [#uses=2]
	%tmp.71 = seteq int %mb_x.1, 0		; <bool> [#uses=2]
	%tmp.72 = cast bool %tmp.71 to int		; <int> [#uses=0]
	br bool %tmp.71, label %then.4, label %else.4

then.4:		; preds = %endif.1
	%tmp.74 = getelementptr %struct.img_par* %img.1, int 0, uint 1		; <uint*> [#uses=1]
	%tmp.75 = load uint* %tmp.74		; <uint> [#uses=1]
	%tmp.76 = cast uint %tmp.75 to int		; <int> [#uses=1]
	call void %getLuma4x4Neighbour( int %tmp.76, int %mb_x.1, int %mb_y.0, int -1, int 0, %struct.pix_pos* %block_a )
	%tmp.79 = getelementptr %struct.pix_pos* %block_a, int 0, uint 0		; <int*> [#uses=1]
	%tmp.80 = load int* %tmp.79		; <int> [#uses=1]
	%tmp.81 = setne int %tmp.80, 0		; <bool> [#uses=2]
	%tmp.82 = cast bool %tmp.81 to int		; <int> [#uses=0]
	br bool %tmp.81, label %then.5, label %else.5

then.5:		; preds = %then.4
	%tmp.84 = getelementptr %struct.img_par* %img.1, int 0, uint 38		; <%struct.macroblock**> [#uses=1]
	%tmp.85 = load %struct.macroblock** %tmp.84		; <%struct.macroblock*> [#uses=1]
	%tmp.86 = getelementptr %struct.pix_pos* %block_a, int 0, uint 1		; <int*> [#uses=1]
	%tmp.87 = load int* %tmp.86		; <int> [#uses=1]
	%tmp.88 = getelementptr %struct.macroblock* %tmp.85, int %tmp.87		; <%struct.macroblock*> [#uses=1]
	%tmp.89 = getelementptr %struct.macroblock* %tmp.88, int 0, uint 5		; <int*> [#uses=1]
	%tmp.90 = load int* %tmp.89		; <int> [#uses=1]
	%tmp.91 = seteq int %tmp.90, 14		; <bool> [#uses=2]
	%tmp.92 = cast bool %tmp.91 to int		; <int> [#uses=0]
	br bool %tmp.91, label %then.6, label %else.6

then.6:		; preds = %then.5
	br label %endif.4

else.6:		; preds = %then.5
	%tmp.94 = getelementptr %struct.img_par* %img.1, int 0, uint 38		; <%struct.macroblock**> [#uses=1]
	%tmp.95 = load %struct.macroblock** %tmp.94		; <%struct.macroblock*> [#uses=1]
	%tmp.96 = getelementptr %struct.pix_pos* %block_a, int 0, uint 1		; <int*> [#uses=1]
	%tmp.97 = load int* %tmp.96		; <int> [#uses=1]
	%tmp.98 = getelementptr %struct.macroblock* %tmp.95, int %tmp.97		; <%struct.macroblock*> [#uses=1]
	%tmp.99 = getelementptr %struct.macroblock* %tmp.98, int 0, uint 7		; <int*> [#uses=1]
	%tmp.100 = load int* %tmp.99		; <int> [#uses=1]
	%tmp.101 = getelementptr %struct.pix_pos* %block_a, int 0, uint 3		; <int*> [#uses=1]
	%tmp.102 = load int* %tmp.101		; <int> [#uses=1]
	%tmp.103 = div int %tmp.102, 2		; <int> [#uses=1]
	%tmp.104 = mul int %tmp.103, 2		; <int> [#uses=1]
	%tmp.105 = add int %tmp.104, 1		; <int> [#uses=1]
	%tmp.106 = cast int %tmp.105 to ubyte		; <ubyte> [#uses=1]
	%tmp.107 = shr int %tmp.100, ubyte %tmp.106		; <int> [#uses=1]
	%tmp.108 = cast int %tmp.107 to uint		; <uint> [#uses=1]
	%tmp.110 = xor uint %tmp.108, 1		; <uint> [#uses=1]
	%tmp.111 = cast uint %tmp.110 to int		; <int> [#uses=1]
	%tmp.112 = and int %tmp.111, 1		; <int> [#uses=1]
	br label %endif.4

else.5:		; preds = %then.4
	br label %endif.4

else.4:		; preds = %endif.1
	%tmp.115 = cast int %mb_y.0 to ubyte		; <ubyte> [#uses=1]
	%tmp.116 = shr int %cbp.1, ubyte %tmp.115		; <int> [#uses=1]
	%tmp.117 = cast int %tmp.116 to uint		; <uint> [#uses=1]
	%tmp.119 = xor uint %tmp.117, 1		; <uint> [#uses=1]
	%tmp.120 = cast uint %tmp.119 to int		; <int> [#uses=1]
	%tmp.121 = and int %tmp.120, 1		; <int> [#uses=1]
	br label %endif.4

endif.4:		; preds = %else.4, %else.5, %else.6, %then.6
	%a.0 = phi int [ 0, %then.6 ], [ %tmp.112, %else.6 ], [ 0, %else.5 ], [ %tmp.121, %else.4 ]		; <int> [#uses=2]
	%tmp.123 = mul int %b.0, 2		; <int> [#uses=1]
	%tmp.125 = add int %tmp.123, %a.0		; <int> [#uses=2]
	%tmp.127 = div int %mb_x.1, 2		; <int> [#uses=1]
	%tmp.129 = add int %tmp.127, %mb_y.0		; <int> [#uses=1]
	%tmp.130 = cast int %tmp.129 to ubyte		; <ubyte> [#uses=1]
	%tmp.131 = shl int 1, ubyte %tmp.130		; <int> [#uses=2]
	%tmp.135 = getelementptr %typedef.TextureInfoContexts* %tmp.4, int 0, uint 2		; <[3 x [4 x %typedef.BiContextType]]*> [#uses=1]
	%tmp.136 = getelementptr [3 x [4 x %typedef.BiContextType]]* %tmp.135, int 0, int 0		; <[4 x %typedef.BiContextType]*> [#uses=1]
	%tmp.137 = getelementptr [4 x %typedef.BiContextType]* %tmp.136, int 0, int 0		; <%typedef.BiContextType*> [#uses=1]
	%tmp.139 = cast int %tmp.125 to uint		; <uint> [#uses=1]
	%tmp.140 = cast uint %tmp.139 to int		; <int> [#uses=1]
	%tmp.141 = getelementptr %typedef.BiContextType* %tmp.137, int %tmp.140		; <%typedef.BiContextType*> [#uses=1]
	%tmp.132 = call uint %biari_decode_symbol( %typedef.DecodingEnvironment* %dep_dp, %typedef.BiContextType* %tmp.141 )		; <uint> [#uses=1]
	%tmp.142 = cast uint %tmp.132 to int		; <int> [#uses=2]
	%tmp.144 = setne int %tmp.142, 0		; <bool> [#uses=2]
	%tmp.145 = cast bool %tmp.144 to int		; <int> [#uses=0]
	br bool %tmp.144, label %then.7, label %endif.7

then.7:		; preds = %endif.4
	%tmp.148 = add int %cbp.1, %tmp.131		; <int> [#uses=1]
	br label %endif.7

endif.7:		; preds = %then.7, %endif.4
	%cbp.0 = phi int [ %tmp.148, %then.7 ], [ %cbp.1, %endif.4 ]		; <int> [#uses=1]
	%tmp.150 = add int %mb_x.1, 2		; <int> [#uses=1]
	br label %loopentry.1

loopexit.1:		; preds = %loopentry.1
	%tmp.152 = add int %mb_y.0, 2		; <int> [#uses=1]
	br label %loopentry.0

loopexit.0:		; preds = %loopentry.0
	%tmp.153 = load %struct.storable_picture** %dec_picture		; <%struct.storable_picture*> [#uses=1]
	%tmp.154 = getelementptr %struct.storable_picture* %tmp.153, int 0, uint 45		; <int*> [#uses=1]
	%tmp.155 = load int* %tmp.154		; <int> [#uses=1]
	%tmp.156 = setne int %tmp.155, 0		; <bool> [#uses=2]
	%tmp.157 = cast bool %tmp.156 to int		; <int> [#uses=0]
	br bool %tmp.156, label %then.8, label %endif.8

then.8:		; preds = %loopexit.0
	%tmp.159 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 3		; <%struct.macroblock**> [#uses=1]
	%tmp.160 = load %struct.macroblock** %tmp.159		; <%struct.macroblock*> [#uses=1]
	%tmp.161 = cast %struct.macroblock* %tmp.160 to sbyte*		; <sbyte*> [#uses=1]
	%tmp.162 = setne sbyte* %tmp.161, null		; <bool> [#uses=2]
	%tmp.163 = cast bool %tmp.162 to int		; <int> [#uses=0]
	br bool %tmp.162, label %then.9, label %endif.9

then.9:		; preds = %then.8
	%tmp.165 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 3		; <%struct.macroblock**> [#uses=1]
	%tmp.166 = load %struct.macroblock** %tmp.165		; <%struct.macroblock*> [#uses=1]
	%tmp.167 = getelementptr %struct.macroblock* %tmp.166, int 0, uint 5		; <int*> [#uses=1]
	%tmp.168 = load int* %tmp.167		; <int> [#uses=1]
	%tmp.169 = seteq int %tmp.168, 14		; <bool> [#uses=2]
	%tmp.170 = cast bool %tmp.169 to int		; <int> [#uses=0]
	br bool %tmp.169, label %then.10, label %else.7

then.10:		; preds = %then.9
	br label %endif.9

else.7:		; preds = %then.9
	%tmp.172 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 3		; <%struct.macroblock**> [#uses=1]
	%tmp.173 = load %struct.macroblock** %tmp.172		; <%struct.macroblock*> [#uses=1]
	%tmp.174 = getelementptr %struct.macroblock* %tmp.173, int 0, uint 7		; <int*> [#uses=1]
	%tmp.175 = load int* %tmp.174		; <int> [#uses=1]
	%tmp.176 = setgt int %tmp.175, 15		; <bool> [#uses=1]
	%tmp.177 = cast bool %tmp.176 to int		; <int> [#uses=1]
	br label %endif.9

endif.9:		; preds = %else.7, %then.10, %then.8
	%b.4 = phi int [ 1, %then.10 ], [ %tmp.177, %else.7 ], [ 0, %then.8 ]		; <int> [#uses=1]
	%tmp.179 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 4		; <%struct.macroblock**> [#uses=1]
	%tmp.180 = load %struct.macroblock** %tmp.179		; <%struct.macroblock*> [#uses=1]
	%tmp.181 = cast %struct.macroblock* %tmp.180 to sbyte*		; <sbyte*> [#uses=1]
	%tmp.182 = setne sbyte* %tmp.181, null		; <bool> [#uses=2]
	%tmp.183 = cast bool %tmp.182 to int		; <int> [#uses=0]
	br bool %tmp.182, label %then.11, label %endif.11

then.11:		; preds = %endif.9
	%tmp.185 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 4		; <%struct.macroblock**> [#uses=1]
	%tmp.186 = load %struct.macroblock** %tmp.185		; <%struct.macroblock*> [#uses=1]
	%tmp.187 = getelementptr %struct.macroblock* %tmp.186, int 0, uint 5		; <int*> [#uses=1]
	%tmp.188 = load int* %tmp.187		; <int> [#uses=1]
	%tmp.189 = seteq int %tmp.188, 14		; <bool> [#uses=2]
	%tmp.190 = cast bool %tmp.189 to int		; <int> [#uses=0]
	br bool %tmp.189, label %then.12, label %else.8

then.12:		; preds = %then.11
	br label %endif.11

else.8:		; preds = %then.11
	%tmp.192 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 4		; <%struct.macroblock**> [#uses=1]
	%tmp.193 = load %struct.macroblock** %tmp.192		; <%struct.macroblock*> [#uses=1]
	%tmp.194 = getelementptr %struct.macroblock* %tmp.193, int 0, uint 7		; <int*> [#uses=1]
	%tmp.195 = load int* %tmp.194		; <int> [#uses=1]
	%tmp.196 = setgt int %tmp.195, 15		; <bool> [#uses=1]
	%tmp.197 = cast bool %tmp.196 to int		; <int> [#uses=1]
	br label %endif.11

endif.11:		; preds = %else.8, %then.12, %endif.9
	%a.4 = phi int [ 1, %then.12 ], [ %tmp.197, %else.8 ], [ 0, %endif.9 ]		; <int> [#uses=1]
	%tmp.199 = mul int %b.4, 2		; <int> [#uses=1]
	%tmp.201 = add int %tmp.199, %a.4		; <int> [#uses=1]
	%tmp.205 = getelementptr %typedef.TextureInfoContexts* %tmp.4, int 0, uint 2		; <[3 x [4 x %typedef.BiContextType]]*> [#uses=1]
	%tmp.206 = getelementptr [3 x [4 x %typedef.BiContextType]]* %tmp.205, int 0, int 1		; <[4 x %typedef.BiContextType]*> [#uses=1]
	%tmp.207 = getelementptr [4 x %typedef.BiContextType]* %tmp.206, int 0, int 0		; <%typedef.BiContextType*> [#uses=1]
	%tmp.209 = cast int %tmp.201 to uint		; <uint> [#uses=1]
	%tmp.210 = cast uint %tmp.209 to int		; <int> [#uses=1]
	%tmp.211 = getelementptr %typedef.BiContextType* %tmp.207, int %tmp.210		; <%typedef.BiContextType*> [#uses=1]
	%tmp.202 = call uint %biari_decode_symbol( %typedef.DecodingEnvironment* %dep_dp, %typedef.BiContextType* %tmp.211 )		; <uint> [#uses=1]
	%tmp.212 = cast uint %tmp.202 to int		; <int> [#uses=1]
	%tmp.214 = setne int %tmp.212, 0		; <bool> [#uses=2]
	%tmp.215 = cast bool %tmp.214 to int		; <int> [#uses=0]
	br bool %tmp.214, label %then.13, label %endif.8

then.13:		; preds = %endif.11
	%tmp.217 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 3		; <%struct.macroblock**> [#uses=1]
	%tmp.218 = load %struct.macroblock** %tmp.217		; <%struct.macroblock*> [#uses=1]
	%tmp.219 = cast %struct.macroblock* %tmp.218 to sbyte*		; <sbyte*> [#uses=1]
	%tmp.220 = setne sbyte* %tmp.219, null		; <bool> [#uses=2]
	%tmp.221 = cast bool %tmp.220 to int		; <int> [#uses=0]
	br bool %tmp.220, label %then.14, label %endif.14

then.14:		; preds = %then.13
	%tmp.223 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 3		; <%struct.macroblock**> [#uses=1]
	%tmp.224 = load %struct.macroblock** %tmp.223		; <%struct.macroblock*> [#uses=1]
	%tmp.225 = getelementptr %struct.macroblock* %tmp.224, int 0, uint 5		; <int*> [#uses=1]
	%tmp.226 = load int* %tmp.225		; <int> [#uses=1]
	%tmp.227 = seteq int %tmp.226, 14		; <bool> [#uses=2]
	%tmp.228 = cast bool %tmp.227 to int		; <int> [#uses=0]
	br bool %tmp.227, label %then.15, label %else.9

then.15:		; preds = %then.14
	br label %endif.14

else.9:		; preds = %then.14
	%tmp.230 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 3		; <%struct.macroblock**> [#uses=1]
	%tmp.231 = load %struct.macroblock** %tmp.230		; <%struct.macroblock*> [#uses=1]
	%tmp.232 = getelementptr %struct.macroblock* %tmp.231, int 0, uint 7		; <int*> [#uses=1]
	%tmp.233 = load int* %tmp.232		; <int> [#uses=1]
	%tmp.234 = setgt int %tmp.233, 15		; <bool> [#uses=2]
	%tmp.235 = cast bool %tmp.234 to int		; <int> [#uses=0]
	br bool %tmp.234, label %then.16, label %endif.14

then.16:		; preds = %else.9
	%tmp.237 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 3		; <%struct.macroblock**> [#uses=1]
	%tmp.238 = load %struct.macroblock** %tmp.237		; <%struct.macroblock*> [#uses=1]
	%tmp.239 = getelementptr %struct.macroblock* %tmp.238, int 0, uint 7		; <int*> [#uses=1]
	%tmp.240 = load int* %tmp.239		; <int> [#uses=1]
	%tmp.242 = shr int %tmp.240, ubyte 4		; <int> [#uses=1]
	%tmp.243 = seteq int %tmp.242, 2		; <bool> [#uses=1]
	%tmp.244 = cast bool %tmp.243 to int		; <int> [#uses=1]
	br label %endif.14

endif.14:		; preds = %then.16, %else.9, %then.15, %then.13
	%b.5 = phi int [ 1, %then.15 ], [ %tmp.244, %then.16 ], [ 0, %else.9 ], [ 0, %then.13 ]		; <int> [#uses=1]
	%tmp.246 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 4		; <%struct.macroblock**> [#uses=1]
	%tmp.247 = load %struct.macroblock** %tmp.246		; <%struct.macroblock*> [#uses=1]
	%tmp.248 = cast %struct.macroblock* %tmp.247 to sbyte*		; <sbyte*> [#uses=1]
	%tmp.249 = setne sbyte* %tmp.248, null		; <bool> [#uses=2]
	%tmp.250 = cast bool %tmp.249 to int		; <int> [#uses=0]
	br bool %tmp.249, label %then.17, label %endif.17

then.17:		; preds = %endif.14
	%tmp.252 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 4		; <%struct.macroblock**> [#uses=1]
	%tmp.253 = load %struct.macroblock** %tmp.252		; <%struct.macroblock*> [#uses=1]
	%tmp.254 = getelementptr %struct.macroblock* %tmp.253, int 0, uint 5		; <int*> [#uses=1]
	%tmp.255 = load int* %tmp.254		; <int> [#uses=1]
	%tmp.256 = seteq int %tmp.255, 14		; <bool> [#uses=2]
	%tmp.257 = cast bool %tmp.256 to int		; <int> [#uses=0]
	br bool %tmp.256, label %then.18, label %else.10

then.18:		; preds = %then.17
	br label %endif.17

else.10:		; preds = %then.17
	%tmp.259 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 4		; <%struct.macroblock**> [#uses=1]
	%tmp.260 = load %struct.macroblock** %tmp.259		; <%struct.macroblock*> [#uses=1]
	%tmp.261 = getelementptr %struct.macroblock* %tmp.260, int 0, uint 7		; <int*> [#uses=1]
	%tmp.262 = load int* %tmp.261		; <int> [#uses=1]
	%tmp.263 = setgt int %tmp.262, 15		; <bool> [#uses=2]
	%tmp.264 = cast bool %tmp.263 to int		; <int> [#uses=0]
	br bool %tmp.263, label %then.19, label %endif.17

then.19:		; preds = %else.10
	%tmp.266 = getelementptr %struct.macroblock* %tmp.12, int 0, uint 4		; <%struct.macroblock**> [#uses=1]
	%tmp.267 = load %struct.macroblock** %tmp.266		; <%struct.macroblock*> [#uses=1]
	%tmp.268 = getelementptr %struct.macroblock* %tmp.267, int 0, uint 7		; <int*> [#uses=1]
	%tmp.269 = load int* %tmp.268		; <int> [#uses=1]
	%tmp.271 = shr int %tmp.269, ubyte 4		; <int> [#uses=1]
	%tmp.272 = seteq int %tmp.271, 2		; <bool> [#uses=1]
	%tmp.273 = cast bool %tmp.272 to int		; <int> [#uses=1]
	br label %endif.17

endif.17:		; preds = %then.19, %else.10, %then.18, %endif.14
	%a.5 = phi int [ 1, %then.18 ], [ %tmp.273, %then.19 ], [ 0, %else.10 ], [ 0, %endif.14 ]		; <int> [#uses=1]
	%tmp.275 = mul int %b.5, 2		; <int> [#uses=1]
	%tmp.277 = add int %tmp.275, %a.5		; <int> [#uses=1]
	%tmp.281 = getelementptr %typedef.TextureInfoContexts* %tmp.4, int 0, uint 2		; <[3 x [4 x %typedef.BiContextType]]*> [#uses=1]
	%tmp.282 = getelementptr [3 x [4 x %typedef.BiContextType]]* %tmp.281, int 0, int 2		; <[4 x %typedef.BiContextType]*> [#uses=1]
	%tmp.283 = getelementptr [4 x %typedef.BiContextType]* %tmp.282, int 0, int 0		; <%typedef.BiContextType*> [#uses=1]
	%tmp.285 = cast int %tmp.277 to uint		; <uint> [#uses=1]
	%tmp.286 = cast uint %tmp.285 to int		; <int> [#uses=1]
	%tmp.287 = getelementptr %typedef.BiContextType* %tmp.283, int %tmp.286		; <%typedef.BiContextType*> [#uses=1]
	%tmp.278 = call uint %biari_decode_symbol( %typedef.DecodingEnvironment* %dep_dp, %typedef.BiContextType* %tmp.287 )		; <uint> [#uses=1]
	%tmp.288 = cast uint %tmp.278 to int		; <int> [#uses=1]
	%tmp.290 = seteq int %tmp.288, 1		; <bool> [#uses=2]
	%tmp.291 = cast bool %tmp.290 to int		; <int> [#uses=0]
	br bool %tmp.290, label %cond_true, label %cond_false

cond_true:		; preds = %endif.17
	%tmp.293 = add int %cbp.2, 32		; <int> [#uses=1]
	br label %cond_continue

cond_false:		; preds = %endif.17
	%tmp.295 = add int %cbp.2, 16		; <int> [#uses=1]
	br label %cond_continue

cond_continue:		; preds = %cond_false, %cond_true
	%mem_tmp.0 = phi int [ %tmp.293, %cond_true ], [ %tmp.295, %cond_false ]		; <int> [#uses=1]
	br label %endif.8

endif.8:		; preds = %cond_continue, %endif.11, %loopexit.0
	%cbp.3 = phi int [ %mem_tmp.0, %cond_continue ], [ %cbp.2, %endif.11 ], [ %cbp.2, %loopexit.0 ]		; <int> [#uses=2]
	%tmp.298 = getelementptr %struct.syntaxelement* %se, int 0, uint 1		; <int*> [#uses=1]
	store int %cbp.3, int* %tmp.298
	%tmp.301 = seteq int %cbp.3, 0		; <bool> [#uses=2]
	%tmp.302 = cast bool %tmp.301 to int		; <int> [#uses=0]
	br bool %tmp.301, label %then.20, label %return

then.20:		; preds = %endif.8
	store int 0, int* %last_dquant
	ret void

return:		; preds = %endif.8
	ret void
}

declare uint %biari_decode_symbol(%typedef.DecodingEnvironment*, %typedef.BiContextType*)

declare void %getLuma4x4Neighbour(int, int, int, int, int, %struct.pix_pos*)
