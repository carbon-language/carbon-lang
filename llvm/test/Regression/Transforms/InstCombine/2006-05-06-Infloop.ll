; RUN: llvm-as < %s | opt -instcombine -disable-output

	%struct.gs_matrix = type { float, int, float, int, float, int, float, int, float, int, float, int }
	%struct.gx_bitmap = type { ubyte*, int, int, int }
	%struct.gx_device = type { int, %struct.gx_device_procs*, sbyte*, int, int, float, float, int, ushort, int, int }
	%struct.gx_device_memory = type { int, %struct.gx_device_procs*, sbyte*, int, int, float, float, int, ushort, int, int, %struct.gs_matrix, int, ubyte*, ubyte**, int (%struct.gx_device_memory*, int, int, int, int, int)*, int, int, ubyte* }
	%struct.gx_device_procs = type { int (%struct.gx_device*)*, void (%struct.gx_device*, %struct.gs_matrix*)*, int (%struct.gx_device*)*, int (%struct.gx_device*)*, int (%struct.gx_device*)*, uint (%struct.gx_device*, ushort, ushort, ushort)*, int (%struct.gx_device*, uint, ushort*)*, int (%struct.gx_device*, int, int, int, int, uint)*, int (%struct.gx_device*, %struct.gx_bitmap*, int, int, int, int, uint, uint)*, int (%struct.gx_device*, ubyte*, int, int, int, int, int, int, uint, uint)*, int (%struct.gx_device*, ubyte*, int, int, int, int, int, int)*, int (%struct.gx_device*, int, int, int, int, uint)*, int (%struct.gx_device*, int, int, int, int, int, int, uint)*, int (%struct.gx_device*, %struct.gx_bitmap*, int, int, int, int, int, int, uint, uint)* }

implementation   ; Functions:

int %mem_mono_copy_mono(%struct.gx_device* %dev, ubyte* %base, int %sourcex, int %raster, int %x, int %y, int %w, int %h, uint %zero, uint %one) {
entry:
	%raster = cast int %raster to uint		; <uint> [#uses=3]
	%tmp = seteq uint %one, %zero		; <bool> [#uses=1]
	br bool %tmp, label %cond_true, label %cond_next

cond_true:		; preds = %entry
	%tmp6 = tail call int %mem_mono_fill_rectangle( %struct.gx_device* %dev, int %x, int %y, int %w, int %h, uint %zero )		; <int> [#uses=1]
	ret int %tmp6

cond_next:		; preds = %entry
	%tmp8 = cast %struct.gx_device* %dev to %struct.gx_device_memory*		; <%struct.gx_device_memory*> [#uses=6]
	%tmp = getelementptr %struct.gx_device_memory* %tmp8, int 0, uint 15		; <int (%struct.gx_device_memory*, int, int, int, int, int)**> [#uses=1]
	%tmp = load int (%struct.gx_device_memory*, int, int, int, int, int)** %tmp		; <int (%struct.gx_device_memory*, int, int, int, int, int)*> [#uses=2]
	%tmp9 = seteq int (%struct.gx_device_memory*, int, int, int, int, int)* %tmp, %mem_no_fault_proc		; <bool> [#uses=1]
	br bool %tmp9, label %cond_next46, label %cond_true10

cond_true10:		; preds = %cond_next
	%tmp16 = add int %x, 7		; <int> [#uses=1]
	%tmp17 = add int %tmp16, %w		; <int> [#uses=1]
	%tmp18 = shr int %tmp17, ubyte 3		; <int> [#uses=1]
	%tmp20 = shr int %x, ubyte 3		; <int> [#uses=2]
	%tmp21 = sub int %tmp18, %tmp20		; <int> [#uses=1]
	%tmp27 = tail call int %tmp( %struct.gx_device_memory* %tmp8, int %tmp20, int %y, int %tmp21, int %h, int 1 )		; <int> [#uses=2]
	%tmp29 = setlt int %tmp27, 0		; <bool> [#uses=1]
	br bool %tmp29, label %cond_true30, label %cond_next46

cond_true30:		; preds = %cond_true10
	%tmp41 = tail call int %mem_copy_mono_recover( %struct.gx_device* %dev, ubyte* %base, int %sourcex, int %raster, int %x, int %y, int %w, int %h, uint %zero, uint %one, int %tmp27 )		; <int> [#uses=1]
	ret int %tmp41

cond_next46:		; preds = %cond_true10, %cond_next
	%tmp48 = setgt int %w, 0		; <bool> [#uses=1]
	%tmp53 = setgt int %h, 0		; <bool> [#uses=1]
	%bothcond = and bool %tmp53, %tmp48		; <bool> [#uses=1]
	br bool %bothcond, label %bb58, label %return

bb58:		; preds = %cond_next46
	%tmp60 = setlt int %x, 0		; <bool> [#uses=1]
	br bool %tmp60, label %return, label %cond_next63

cond_next63:		; preds = %bb58
	%tmp65 = getelementptr %struct.gx_device_memory* %tmp8, int 0, uint 3		; <int*> [#uses=1]
	%tmp66 = load int* %tmp65		; <int> [#uses=1]
	%tmp68 = sub int %tmp66, %w		; <int> [#uses=1]
	%tmp70 = setlt int %tmp68, %x		; <bool> [#uses=1]
	%tmp75 = setlt int %y, 0		; <bool> [#uses=1]
	%bothcond1 = or bool %tmp70, %tmp75		; <bool> [#uses=1]
	br bool %bothcond1, label %return, label %cond_next78

cond_next78:		; preds = %cond_next63
	%tmp80 = getelementptr %struct.gx_device_memory* %tmp8, int 0, uint 4		; <int*> [#uses=1]
	%tmp81 = load int* %tmp80		; <int> [#uses=1]
	%tmp83 = sub int %tmp81, %h		; <int> [#uses=1]
	%tmp85 = setlt int %tmp83, %y		; <bool> [#uses=1]
	br bool %tmp85, label %return, label %bb91

bb91:		; preds = %cond_next78
	%tmp93 = shr int %x, ubyte 3		; <int> [#uses=4]
	%tmp = getelementptr %struct.gx_device_memory* %tmp8, int 0, uint 14		; <ubyte***> [#uses=1]
	%tmp = load ubyte*** %tmp		; <ubyte**> [#uses=1]
	%tmp96 = getelementptr ubyte** %tmp, int %y		; <ubyte**> [#uses=4]
	%tmp98 = load ubyte** %tmp96		; <ubyte*> [#uses=1]
	%tmp100 = getelementptr ubyte* %tmp98, int %tmp93		; <ubyte*> [#uses=3]
	%tmp102 = shr int %sourcex, ubyte 3		; <int> [#uses=3]
	%tmp106 = and int %sourcex, 7		; <int> [#uses=1]
	%tmp107 = sub int 8, %tmp106		; <int> [#uses=4]
	%tmp109 = and int %x, 7		; <int> [#uses=3]
	%tmp110 = sub int 8, %tmp109		; <int> [#uses=8]
	%tmp112 = sub int 8, %tmp110		; <int> [#uses=1]
	%tmp112 = cast int %tmp112 to ubyte		; <ubyte> [#uses=1]
	%tmp113464 = shr uint 255, ubyte %tmp112		; <uint> [#uses=4]
	%tmp116 = setgt int %tmp110, %w		; <bool> [#uses=1]
	%tmp132 = getelementptr %struct.gx_device_memory* %tmp8, int 0, uint 16		; <int*> [#uses=2]
	br bool %tmp116, label %cond_true117, label %cond_false123

cond_true117:		; preds = %bb91
	%tmp119 = cast int %w to ubyte		; <ubyte> [#uses=1]
	%tmp120 = shr uint %tmp113464, ubyte %tmp119		; <uint> [#uses=1]
	%tmp122 = sub uint %tmp113464, %tmp120		; <uint> [#uses=2]
	%tmp13315 = load int* %tmp132		; <int> [#uses=1]
	%tmp13416 = seteq int %tmp13315, 0		; <bool> [#uses=1]
	br bool %tmp13416, label %cond_next151, label %cond_true135

cond_false123:		; preds = %bb91
	%tmp126 = sub int %w, %tmp110		; <int> [#uses=1]
	%tmp126 = cast int %tmp126 to ubyte		; <ubyte> [#uses=1]
	%tmp127 = and ubyte %tmp126, 7		; <ubyte> [#uses=1]
	%tmp128 = shr uint 255, ubyte %tmp127		; <uint> [#uses=1]
	%tmp1295 = sub uint 255, %tmp128		; <uint> [#uses=2]
	%tmp133 = load int* %tmp132		; <int> [#uses=1]
	%tmp134 = seteq int %tmp133, 0		; <bool> [#uses=1]
	br bool %tmp134, label %cond_next151, label %cond_true135

cond_true135:		; preds = %cond_false123, %cond_true117
	%rmask.0.0 = phi uint [ undef, %cond_true117 ], [ %tmp1295, %cond_false123 ]		; <uint> [#uses=2]
	%mask.1.0 = phi uint [ %tmp122, %cond_true117 ], [ %tmp113464, %cond_false123 ]		; <uint> [#uses=2]
	%not.tmp137 = setne uint %zero, 4294967295		; <bool> [#uses=1]
	%tmp140 = cast bool %not.tmp137 to uint		; <uint> [#uses=1]
	%zero_addr.0 = xor uint %tmp140, %zero		; <uint> [#uses=2]
	%tmp144 = seteq uint %one, 4294967295		; <bool> [#uses=1]
	br bool %tmp144, label %cond_next151, label %cond_true145

cond_true145:		; preds = %cond_true135
	%tmp147 = xor uint %one, 1		; <uint> [#uses=1]
	br label %cond_next151

cond_next151:		; preds = %cond_true145, %cond_true135, %cond_false123, %cond_true117
	%rmask.0.1 = phi uint [ %rmask.0.0, %cond_true145 ], [ undef, %cond_true117 ], [ %tmp1295, %cond_false123 ], [ %rmask.0.0, %cond_true135 ]		; <uint> [#uses=4]
	%mask.1.1 = phi uint [ %mask.1.0, %cond_true145 ], [ %tmp122, %cond_true117 ], [ %tmp113464, %cond_false123 ], [ %mask.1.0, %cond_true135 ]		; <uint> [#uses=4]
	%one_addr.0 = phi uint [ %tmp147, %cond_true145 ], [ %one, %cond_true117 ], [ %one, %cond_false123 ], [ %one, %cond_true135 ]		; <uint> [#uses=2]
	%zero_addr.1 = phi uint [ %zero_addr.0, %cond_true145 ], [ %zero, %cond_true117 ], [ %zero, %cond_false123 ], [ %zero_addr.0, %cond_true135 ]		; <uint> [#uses=2]
	%tmp153 = seteq uint %zero_addr.1, 1		; <bool> [#uses=2]
	%tmp158 = seteq uint %one_addr.0, 0		; <bool> [#uses=2]
	%bothcond2 = or bool %tmp153, %tmp158		; <bool> [#uses=1]
	%iftmp.35.0 = select bool %bothcond2, uint 4294967295, uint 0		; <uint> [#uses=8]
	%tmp167 = seteq uint %zero_addr.1, 0		; <bool> [#uses=1]
	%bothcond3 = or bool %tmp167, %tmp158		; <bool> [#uses=1]
	%iftmp.36.0 = select bool %bothcond3, uint 0, uint 4294967295		; <uint> [#uses=4]
	%tmp186 = seteq uint %one_addr.0, 1		; <bool> [#uses=1]
	%bothcond4 = or bool %tmp153, %tmp186		; <bool> [#uses=1]
	%iftmp.37.0 = select bool %bothcond4, uint 4294967295, uint 0		; <uint> [#uses=6]
	%tmp196 = seteq int %tmp107, %tmp110		; <bool> [#uses=1]
	br bool %tmp196, label %cond_true197, label %cond_false299

cond_true197:		; preds = %cond_next151
	%tmp29222 = add int %h, -1		; <int> [#uses=3]
	%tmp29424 = setlt int %tmp29222, 0		; <bool> [#uses=1]
	br bool %tmp29424, label %return, label %cond_true295.preheader

cond_true249.preheader:		; preds = %cond_true295
	br label %cond_true249

cond_true249:		; preds = %cond_true249, %cond_true249.preheader
	%indvar = phi uint [ 0, %cond_true249.preheader ], [ %indvar.next, %cond_true249 ]		; <uint> [#uses=2]
	%optr.3.2 = phi ubyte* [ %tmp232, %cond_true249 ], [ %dest.1.0, %cond_true249.preheader ]		; <ubyte*> [#uses=1]
	%bptr.3.2 = phi ubyte* [ %tmp226, %cond_true249 ], [ %line.1.0, %cond_true249.preheader ]		; <ubyte*> [#uses=1]
	%tmp. = add int %tmp109, %w		; <int> [#uses=1]
	%indvar = cast uint %indvar to int		; <int> [#uses=1]
	%tmp.58 = mul int %indvar, -8		; <int> [#uses=1]
	%tmp.57 = add int %tmp., -16		; <int> [#uses=1]
	%tmp246.2 = add int %tmp.58, %tmp.57		; <int> [#uses=1]
	%tmp225 = cast ubyte* %bptr.3.2 to uint		; <uint> [#uses=1]
	%tmp226 = add uint %tmp225, 1		; <uint> [#uses=1]
	%tmp226 = cast uint %tmp226 to ubyte*		; <ubyte*> [#uses=3]
	%tmp228 = load ubyte* %tmp226		; <ubyte> [#uses=1]
	%tmp228 = cast ubyte %tmp228 to uint		; <uint> [#uses=1]
	%tmp230 = xor uint %tmp228, %iftmp.35.0		; <uint> [#uses=2]
	%tmp231 = cast ubyte* %optr.3.2 to uint		; <uint> [#uses=1]
	%tmp232 = add uint %tmp231, 1		; <uint> [#uses=1]
	%tmp232 = cast uint %tmp232 to ubyte*		; <ubyte*> [#uses=4]
	%tmp235 = or uint %tmp230, %iftmp.36.0		; <uint> [#uses=1]
	%tmp235 = cast uint %tmp235 to ubyte		; <ubyte> [#uses=1]
	%tmp237 = load ubyte* %tmp232		; <ubyte> [#uses=1]
	%tmp238 = and ubyte %tmp235, %tmp237		; <ubyte> [#uses=1]
	%tmp241 = and uint %tmp230, %iftmp.37.0		; <uint> [#uses=1]
	%tmp241 = cast uint %tmp241 to ubyte		; <ubyte> [#uses=1]
	%tmp242 = or ubyte %tmp238, %tmp241		; <ubyte> [#uses=1]
	store ubyte %tmp242, ubyte* %tmp232
	%tmp24629 = add int %tmp246.2, -8		; <int> [#uses=2]
	%tmp24831 = setlt int %tmp24629, 0		; <bool> [#uses=1]
	%indvar.next = add uint %indvar, 1		; <uint> [#uses=1]
	br bool %tmp24831, label %bb252.loopexit, label %cond_true249

bb252.loopexit:		; preds = %cond_true249
	br label %bb252

bb252:		; preds = %cond_true295, %bb252.loopexit
	%optr.3.3 = phi ubyte* [ %dest.1.0, %cond_true295 ], [ %tmp232, %bb252.loopexit ]		; <ubyte*> [#uses=1]
	%bptr.3.3 = phi ubyte* [ %line.1.0, %cond_true295 ], [ %tmp226, %bb252.loopexit ]		; <ubyte*> [#uses=1]
	%tmp246.3 = phi int [ %tmp246, %cond_true295 ], [ %tmp24629, %bb252.loopexit ]		; <int> [#uses=1]
	%tmp254 = setgt int %tmp246.3, -8		; <bool> [#uses=1]
	br bool %tmp254, label %cond_true255, label %cond_next280

cond_true255:		; preds = %bb252
	%tmp256 = cast ubyte* %bptr.3.3 to uint		; <uint> [#uses=1]
	%tmp257 = add uint %tmp256, 1		; <uint> [#uses=1]
	%tmp257 = cast uint %tmp257 to ubyte*		; <ubyte*> [#uses=1]
	%tmp259 = load ubyte* %tmp257		; <ubyte> [#uses=1]
	%tmp259 = cast ubyte %tmp259 to uint		; <uint> [#uses=1]
	%tmp261 = xor uint %tmp259, %iftmp.35.0		; <uint> [#uses=2]
	%tmp262 = cast ubyte* %optr.3.3 to uint		; <uint> [#uses=1]
	%tmp263 = add uint %tmp262, 1		; <uint> [#uses=1]
	%tmp263 = cast uint %tmp263 to ubyte*		; <ubyte*> [#uses=2]
	%tmp265 = cast uint %tmp261 to ubyte		; <ubyte> [#uses=1]
	%tmp268 = or ubyte %tmp266, %tmp265		; <ubyte> [#uses=1]
	%tmp270 = load ubyte* %tmp263		; <ubyte> [#uses=1]
	%tmp271 = and ubyte %tmp268, %tmp270		; <ubyte> [#uses=1]
	%tmp276 = and uint %tmp274, %tmp261		; <uint> [#uses=1]
	%tmp276 = cast uint %tmp276 to ubyte		; <ubyte> [#uses=1]
	%tmp277 = or ubyte %tmp271, %tmp276		; <ubyte> [#uses=1]
	store ubyte %tmp277, ubyte* %tmp263
	br label %cond_next280

cond_next280:		; preds = %cond_true255, %bb252
	%tmp281 = cast ubyte** %dest_line.1.0 to uint		; <uint> [#uses=1]
	%tmp282 = add uint %tmp281, 4		; <uint> [#uses=1]
	%tmp282 = cast uint %tmp282 to ubyte**		; <ubyte**> [#uses=2]
	%tmp284 = load ubyte** %tmp282		; <ubyte*> [#uses=1]
	%tmp286 = getelementptr ubyte* %tmp284, int %tmp93		; <ubyte*> [#uses=1]
	%tmp292 = add int %tmp292.0, -1		; <int> [#uses=1]
	%tmp294 = setlt int %tmp292, 0		; <bool> [#uses=1]
	%indvar.next61 = add uint %indvar60, 1		; <uint> [#uses=1]
	br bool %tmp294, label %return.loopexit, label %cond_true295

cond_true295.preheader:		; preds = %cond_true197
	%tmp200 = sub int %w, %tmp110		; <int> [#uses=1]
	%tmp209 = cast uint %mask.1.1 to ubyte		; <ubyte> [#uses=1]
	%tmp209not = xor ubyte %tmp209, 255		; <ubyte> [#uses=1]
	%tmp212 = cast uint %iftmp.36.0 to ubyte		; <ubyte> [#uses=2]
	%tmp211 = or ubyte %tmp212, %tmp209not		; <ubyte> [#uses=2]
	%tmp219 = and uint %iftmp.37.0, %mask.1.1		; <uint> [#uses=2]
	%tmp246 = add int %tmp200, -8		; <int> [#uses=3]
	%tmp248 = setlt int %tmp246, 0		; <bool> [#uses=1]
	%tmp264 = cast uint %rmask.0.1 to ubyte		; <ubyte> [#uses=1]
	%tmp264not = xor ubyte %tmp264, 255		; <ubyte> [#uses=1]
	%tmp266 = or ubyte %tmp212, %tmp264not		; <ubyte> [#uses=2]
	%tmp274 = and uint %iftmp.37.0, %rmask.0.1		; <uint> [#uses=2]
	br bool %tmp248, label %cond_true295.preheader.split.us, label %cond_true295.preheader.split

cond_true295.preheader.split.us:		; preds = %cond_true295.preheader
	br label %cond_true295.us

cond_true295.us:		; preds = %cond_next280.us, %cond_true295.preheader.split.us
	%indvar86 = phi uint [ 0, %cond_true295.preheader.split.us ], [ %indvar.next87, %cond_next280.us ]		; <uint> [#uses=3]
	%dest.1.0.us = phi ubyte* [ %tmp286.us, %cond_next280.us ], [ %tmp100, %cond_true295.preheader.split.us ]		; <ubyte*> [#uses=3]
	%dest_line.1.0.us = phi ubyte** [ %tmp282.us, %cond_next280.us ], [ %tmp96, %cond_true295.preheader.split.us ]		; <ubyte**> [#uses=1]
	%tmp.89 = sub uint 0, %indvar86		; <uint> [#uses=1]
	%tmp.89 = cast uint %tmp.89 to int		; <int> [#uses=1]
	%tmp292.0.us = add int %tmp.89, %tmp29222		; <int> [#uses=1]
	%tmp.91 = mul uint %indvar86, %raster		; <uint> [#uses=1]
	%tmp.91 = cast uint %tmp.91 to int		; <int> [#uses=1]
	%tmp104.sum101 = add int %tmp102, %tmp.91		; <int> [#uses=1]
	%line.1.0.us = getelementptr ubyte* %base, int %tmp104.sum101		; <ubyte*> [#uses=2]
	%tmp.us = load ubyte* %line.1.0.us		; <ubyte> [#uses=1]
	%tmp206.us = cast ubyte %tmp.us to uint		; <uint> [#uses=1]
	%tmp208.us = xor uint %tmp206.us, %iftmp.35.0		; <uint> [#uses=2]
	%tmp210.us = cast uint %tmp208.us to ubyte		; <ubyte> [#uses=1]
	%tmp213.us = or ubyte %tmp211, %tmp210.us		; <ubyte> [#uses=1]
	%tmp215.us = load ubyte* %dest.1.0.us		; <ubyte> [#uses=1]
	%tmp216.us = and ubyte %tmp213.us, %tmp215.us		; <ubyte> [#uses=1]
	%tmp221.us = and uint %tmp219, %tmp208.us		; <uint> [#uses=1]
	%tmp221.us = cast uint %tmp221.us to ubyte		; <ubyte> [#uses=1]
	%tmp222.us = or ubyte %tmp216.us, %tmp221.us		; <ubyte> [#uses=1]
	store ubyte %tmp222.us, ubyte* %dest.1.0.us
	br bool true, label %bb252.us, label %cond_true249.preheader.us

cond_next280.us:		; preds = %bb252.us, %cond_true255.us
	%tmp281.us = cast ubyte** %dest_line.1.0.us to uint		; <uint> [#uses=1]
	%tmp282.us = add uint %tmp281.us, 4		; <uint> [#uses=1]
	%tmp282.us = cast uint %tmp282.us to ubyte**		; <ubyte**> [#uses=2]
	%tmp284.us = load ubyte** %tmp282.us		; <ubyte*> [#uses=1]
	%tmp286.us = getelementptr ubyte* %tmp284.us, int %tmp93		; <ubyte*> [#uses=1]
	%tmp292.us = add int %tmp292.0.us, -1		; <int> [#uses=1]
	%tmp294.us = setlt int %tmp292.us, 0		; <bool> [#uses=1]
	%indvar.next87 = add uint %indvar86, 1		; <uint> [#uses=1]
	br bool %tmp294.us, label %return.loopexit.us, label %cond_true295.us

cond_true255.us:		; preds = %bb252.us
	%tmp256.us = cast ubyte* %bptr.3.3.us to uint		; <uint> [#uses=1]
	%tmp257.us = add uint %tmp256.us, 1		; <uint> [#uses=1]
	%tmp257.us = cast uint %tmp257.us to ubyte*		; <ubyte*> [#uses=1]
	%tmp259.us = load ubyte* %tmp257.us		; <ubyte> [#uses=1]
	%tmp259.us = cast ubyte %tmp259.us to uint		; <uint> [#uses=1]
	%tmp261.us = xor uint %tmp259.us, %iftmp.35.0		; <uint> [#uses=2]
	%tmp262.us = cast ubyte* %optr.3.3.us to uint		; <uint> [#uses=1]
	%tmp263.us = add uint %tmp262.us, 1		; <uint> [#uses=1]
	%tmp263.us = cast uint %tmp263.us to ubyte*		; <ubyte*> [#uses=2]
	%tmp265.us = cast uint %tmp261.us to ubyte		; <ubyte> [#uses=1]
	%tmp268.us = or ubyte %tmp266, %tmp265.us		; <ubyte> [#uses=1]
	%tmp270.us = load ubyte* %tmp263.us		; <ubyte> [#uses=1]
	%tmp271.us = and ubyte %tmp268.us, %tmp270.us		; <ubyte> [#uses=1]
	%tmp276.us = and uint %tmp274, %tmp261.us		; <uint> [#uses=1]
	%tmp276.us = cast uint %tmp276.us to ubyte		; <ubyte> [#uses=1]
	%tmp277.us = or ubyte %tmp271.us, %tmp276.us		; <ubyte> [#uses=1]
	store ubyte %tmp277.us, ubyte* %tmp263.us
	br label %cond_next280.us

bb252.us:		; preds = %bb252.loopexit.us, %cond_true295.us
	%optr.3.3.us = phi ubyte* [ %dest.1.0.us, %cond_true295.us ], [ undef, %bb252.loopexit.us ]		; <ubyte*> [#uses=1]
	%bptr.3.3.us = phi ubyte* [ %line.1.0.us, %cond_true295.us ], [ undef, %bb252.loopexit.us ]		; <ubyte*> [#uses=1]
	%tmp246.3.us = phi int [ %tmp246, %cond_true295.us ], [ undef, %bb252.loopexit.us ]		; <int> [#uses=1]
	%tmp254.us = setgt int %tmp246.3.us, -8		; <bool> [#uses=1]
	br bool %tmp254.us, label %cond_true255.us, label %cond_next280.us

cond_true249.us:		; preds = %cond_true249.preheader.us, %cond_true249.us
	br bool undef, label %bb252.loopexit.us, label %cond_true249.us

cond_true249.preheader.us:		; preds = %cond_true295.us
	br label %cond_true249.us

bb252.loopexit.us:		; preds = %cond_true249.us
	br label %bb252.us

return.loopexit.us:		; preds = %cond_next280.us
	br label %return.loopexit.split

cond_true295.preheader.split:		; preds = %cond_true295.preheader
	br label %cond_true295

cond_true295:		; preds = %cond_true295.preheader.split, %cond_next280
	%indvar60 = phi uint [ 0, %cond_true295.preheader.split ], [ %indvar.next61, %cond_next280 ]		; <uint> [#uses=3]
	%dest.1.0 = phi ubyte* [ %tmp286, %cond_next280 ], [ %tmp100, %cond_true295.preheader.split ]		; <ubyte*> [#uses=4]
	%dest_line.1.0 = phi ubyte** [ %tmp282, %cond_next280 ], [ %tmp96, %cond_true295.preheader.split ]		; <ubyte**> [#uses=1]
	%tmp.63 = sub uint 0, %indvar60		; <uint> [#uses=1]
	%tmp.63 = cast uint %tmp.63 to int		; <int> [#uses=1]
	%tmp292.0 = add int %tmp.63, %tmp29222		; <int> [#uses=1]
	%tmp.65 = mul uint %indvar60, %raster		; <uint> [#uses=1]
	%tmp.65 = cast uint %tmp.65 to int		; <int> [#uses=1]
	%tmp104.sum97 = add int %tmp102, %tmp.65		; <int> [#uses=1]
	%line.1.0 = getelementptr ubyte* %base, int %tmp104.sum97		; <ubyte*> [#uses=3]
	%tmp = load ubyte* %line.1.0		; <ubyte> [#uses=1]
	%tmp206 = cast ubyte %tmp to uint		; <uint> [#uses=1]
	%tmp208 = xor uint %tmp206, %iftmp.35.0		; <uint> [#uses=2]
	%tmp210 = cast uint %tmp208 to ubyte		; <ubyte> [#uses=1]
	%tmp213 = or ubyte %tmp211, %tmp210		; <ubyte> [#uses=1]
	%tmp215 = load ubyte* %dest.1.0		; <ubyte> [#uses=1]
	%tmp216 = and ubyte %tmp213, %tmp215		; <ubyte> [#uses=1]
	%tmp221 = and uint %tmp219, %tmp208		; <uint> [#uses=1]
	%tmp221 = cast uint %tmp221 to ubyte		; <ubyte> [#uses=1]
	%tmp222 = or ubyte %tmp216, %tmp221		; <ubyte> [#uses=1]
	store ubyte %tmp222, ubyte* %dest.1.0
	br bool false, label %bb252, label %cond_true249.preheader

cond_false299:		; preds = %cond_next151
	%tmp302 = sub int %tmp107, %tmp110		; <int> [#uses=1]
	%tmp303 = and int %tmp302, 7		; <int> [#uses=3]
	%tmp305 = sub int 8, %tmp303		; <int> [#uses=1]
	%tmp45438 = add int %h, -1		; <int> [#uses=2]
	%tmp45640 = setlt int %tmp45438, 0		; <bool> [#uses=1]
	br bool %tmp45640, label %return, label %cond_true457.preheader

cond_true316:		; preds = %cond_true457
	%tmp318 = cast ubyte %tmp318 to uint		; <uint> [#uses=1]
	%tmp320 = shr uint %tmp318, ubyte %tmp319		; <uint> [#uses=1]
	br label %cond_next340

cond_false321:		; preds = %cond_true457
	%tmp3188 = cast ubyte %tmp318 to uint		; <uint> [#uses=1]
	%tmp325 = shl uint %tmp3188, ubyte %tmp324		; <uint> [#uses=2]
	%tmp326 = cast ubyte* %line.3.0 to uint		; <uint> [#uses=1]
	%tmp327 = add uint %tmp326, 1		; <uint> [#uses=1]
	%tmp327 = cast uint %tmp327 to ubyte*		; <ubyte*> [#uses=3]
	br bool %tmp330, label %cond_true331, label %cond_next340

cond_true331:		; preds = %cond_false321
	%tmp333 = load ubyte* %tmp327		; <ubyte> [#uses=1]
	%tmp333 = cast ubyte %tmp333 to uint		; <uint> [#uses=1]
	%tmp335 = shr uint %tmp333, ubyte %tmp319		; <uint> [#uses=1]
	%tmp337 = add uint %tmp335, %tmp325		; <uint> [#uses=1]
	br label %cond_next340

cond_next340:		; preds = %cond_true331, %cond_false321, %cond_true316
	%bits.0 = phi uint [ %tmp320, %cond_true316 ], [ %tmp337, %cond_true331 ], [ %tmp325, %cond_false321 ]		; <uint> [#uses=1]
	%bptr307.3 = phi ubyte* [ %line.3.0, %cond_true316 ], [ %tmp327, %cond_true331 ], [ %tmp327, %cond_false321 ]		; <ubyte*> [#uses=2]
	%tmp343 = xor uint %bits.0, %iftmp.35.0		; <uint> [#uses=2]
	%tmp345 = cast uint %tmp343 to ubyte		; <ubyte> [#uses=1]
	%tmp348 = or ubyte %tmp346, %tmp345		; <ubyte> [#uses=1]
	%tmp350 = load ubyte* %dest.3.0		; <ubyte> [#uses=1]
	%tmp351 = and ubyte %tmp348, %tmp350		; <ubyte> [#uses=1]
	%tmp356 = and uint %tmp354, %tmp343		; <uint> [#uses=1]
	%tmp356 = cast uint %tmp356 to ubyte		; <ubyte> [#uses=1]
	%tmp357 = or ubyte %tmp351, %tmp356		; <ubyte> [#uses=1]
	store ubyte %tmp357, ubyte* %dest.3.0
	%tmp362 = cast ubyte* %dest.3.0 to uint		; <uint> [#uses=1]
	%optr309.3.in51 = add uint %tmp362, 1		; <uint> [#uses=2]
	%optr309.353 = cast uint %optr309.3.in51 to ubyte*		; <ubyte*> [#uses=2]
	br bool %tmp39755, label %cond_true398.preheader, label %bb401

cond_true398.preheader:		; preds = %cond_next340
	br label %cond_true398

cond_true398:		; preds = %cond_true398, %cond_true398.preheader
	%indvar66 = phi uint [ 0, %cond_true398.preheader ], [ %indvar.next67, %cond_true398 ]		; <uint> [#uses=3]
	%bptr307.4.0 = phi ubyte* [ %tmp370, %cond_true398 ], [ %bptr307.3, %cond_true398.preheader ]		; <ubyte*> [#uses=2]
	%optr309.3.0 = phi ubyte* [ %optr309.3, %cond_true398 ], [ %optr309.353, %cond_true398.preheader ]		; <ubyte*> [#uses=2]
	%optr309.3.in.0 = add uint %indvar66, %optr309.3.in51		; <uint> [#uses=1]
	%tmp.70 = add int %tmp109, %w		; <int> [#uses=1]
	%indvar66 = cast uint %indvar66 to int		; <int> [#uses=1]
	%tmp.72 = mul int %indvar66, -8		; <int> [#uses=1]
	%tmp.71 = add int %tmp.70, -8		; <int> [#uses=1]
	%count308.3.0 = add int %tmp.72, %tmp.71		; <int> [#uses=1]
	%tmp366 = load ubyte* %bptr307.4.0		; <ubyte> [#uses=1]
	%tmp366 = cast ubyte %tmp366 to uint		; <uint> [#uses=1]
	%tmp369 = cast ubyte* %bptr307.4.0 to uint		; <uint> [#uses=1]
	%tmp370 = add uint %tmp369, 1		; <uint> [#uses=1]
	%tmp370 = cast uint %tmp370 to ubyte*		; <ubyte*> [#uses=3]
	%tmp372 = load ubyte* %tmp370		; <ubyte> [#uses=1]
	%tmp372 = cast ubyte %tmp372 to uint		; <uint> [#uses=1]
	%tmp374463 = shr uint %tmp372, ubyte %tmp319		; <uint> [#uses=1]
	%tmp368 = shl uint %tmp366, ubyte %tmp324		; <uint> [#uses=1]
	%tmp377 = add uint %tmp374463, %tmp368		; <uint> [#uses=1]
	%tmp379 = xor uint %tmp377, %iftmp.35.0		; <uint> [#uses=2]
	%tmp382 = or uint %tmp379, %iftmp.36.0		; <uint> [#uses=1]
	%tmp382 = cast uint %tmp382 to ubyte		; <ubyte> [#uses=1]
	%tmp384 = load ubyte* %optr309.3.0		; <ubyte> [#uses=1]
	%tmp385 = and ubyte %tmp382, %tmp384		; <ubyte> [#uses=1]
	%tmp388 = and uint %tmp379, %iftmp.37.0		; <uint> [#uses=1]
	%tmp388 = cast uint %tmp388 to ubyte		; <ubyte> [#uses=1]
	%tmp389 = or ubyte %tmp385, %tmp388		; <ubyte> [#uses=1]
	store ubyte %tmp389, ubyte* %optr309.3.0
	%tmp392 = add int %count308.3.0, -8		; <int> [#uses=2]
	%optr309.3.in = add uint %optr309.3.in.0, 1		; <uint> [#uses=1]
	%optr309.3 = cast uint %optr309.3.in to ubyte*		; <ubyte*> [#uses=2]
	%tmp397 = setgt int %tmp392, 7		; <bool> [#uses=1]
	%indvar.next67 = add uint %indvar66, 1		; <uint> [#uses=1]
	br bool %tmp397, label %cond_true398, label %bb401.loopexit

bb401.loopexit:		; preds = %cond_true398
	br label %bb401

bb401:		; preds = %bb401.loopexit, %cond_next340
	%count308.3.1 = phi int [ %tmp361, %cond_next340 ], [ %tmp392, %bb401.loopexit ]		; <int> [#uses=2]
	%bptr307.4.1 = phi ubyte* [ %bptr307.3, %cond_next340 ], [ %tmp370, %bb401.loopexit ]		; <ubyte*> [#uses=2]
	%optr309.3.1 = phi ubyte* [ %optr309.353, %cond_next340 ], [ %optr309.3, %bb401.loopexit ]		; <ubyte*> [#uses=2]
	%tmp403 = setgt int %count308.3.1, 0		; <bool> [#uses=1]
	br bool %tmp403, label %cond_true404, label %cond_next442

cond_true404:		; preds = %bb401
	%tmp406 = load ubyte* %bptr307.4.1		; <ubyte> [#uses=1]
	%tmp406 = cast ubyte %tmp406 to int		; <int> [#uses=1]
	%tmp408 = shl int %tmp406, ubyte %tmp324		; <int> [#uses=2]
	%tmp413 = setgt int %count308.3.1, %tmp303		; <bool> [#uses=1]
	br bool %tmp413, label %cond_true414, label %cond_next422

cond_true414:		; preds = %cond_true404
	%tmp409 = cast ubyte* %bptr307.4.1 to uint		; <uint> [#uses=1]
	%tmp410 = add uint %tmp409, 1		; <uint> [#uses=1]
	%tmp410 = cast uint %tmp410 to ubyte*		; <ubyte*> [#uses=1]
	%tmp416 = load ubyte* %tmp410		; <ubyte> [#uses=1]
	%tmp416 = cast ubyte %tmp416 to uint		; <uint> [#uses=1]
	%tmp418 = shr uint %tmp416, ubyte %tmp319		; <uint> [#uses=1]
	%tmp418 = cast uint %tmp418 to int		; <int> [#uses=1]
	%tmp420 = add int %tmp418, %tmp408		; <int> [#uses=1]
	br label %cond_next422

cond_next422:		; preds = %cond_true414, %cond_true404
	%bits.6 = phi int [ %tmp420, %cond_true414 ], [ %tmp408, %cond_true404 ]		; <int> [#uses=1]
	%tmp425 = xor int %bits.6, %iftmp.35.0		; <int> [#uses=1]
	%tmp427 = cast int %tmp425 to ubyte		; <ubyte> [#uses=2]
	%tmp430 = or ubyte %tmp428, %tmp427		; <ubyte> [#uses=1]
	%tmp432 = load ubyte* %optr309.3.1		; <ubyte> [#uses=1]
	%tmp433 = and ubyte %tmp430, %tmp432		; <ubyte> [#uses=1]
	%tmp438 = and ubyte %tmp436, %tmp427		; <ubyte> [#uses=1]
	%tmp439 = or ubyte %tmp433, %tmp438		; <ubyte> [#uses=1]
	store ubyte %tmp439, ubyte* %optr309.3.1
	br label %cond_next442

cond_next442:		; preds = %cond_next422, %bb401
	%tmp443 = cast ubyte** %dest_line.3.0 to uint		; <uint> [#uses=1]
	%tmp444 = add uint %tmp443, 4		; <uint> [#uses=1]
	%tmp444 = cast uint %tmp444 to ubyte**		; <ubyte**> [#uses=2]
	%tmp446 = load ubyte** %tmp444		; <ubyte*> [#uses=1]
	%tmp448 = getelementptr ubyte* %tmp446, int %tmp93		; <ubyte*> [#uses=1]
	%tmp454 = add int %tmp454.0, -1		; <int> [#uses=1]
	%tmp456 = setlt int %tmp454, 0		; <bool> [#uses=1]
	%indvar.next75 = add uint %indvar74, 1		; <uint> [#uses=1]
	br bool %tmp456, label %return.loopexit56, label %cond_true457

cond_true457.preheader:		; preds = %cond_false299
	%tmp315 = setlt int %tmp107, %tmp110		; <bool> [#uses=1]
	%tmp319 = cast int %tmp303 to ubyte		; <ubyte> [#uses=4]
	%tmp324 = cast int %tmp305 to ubyte		; <ubyte> [#uses=3]
	%tmp330 = setlt int %tmp107, %w		; <bool> [#uses=1]
	%tmp344 = cast uint %mask.1.1 to ubyte		; <ubyte> [#uses=1]
	%tmp344not = xor ubyte %tmp344, 255		; <ubyte> [#uses=1]
	%tmp347 = cast uint %iftmp.36.0 to ubyte		; <ubyte> [#uses=2]
	%tmp346 = or ubyte %tmp347, %tmp344not		; <ubyte> [#uses=1]
	%tmp354 = and uint %iftmp.37.0, %mask.1.1		; <uint> [#uses=1]
	%tmp361 = sub int %w, %tmp110		; <int> [#uses=2]
	%tmp39755 = setgt int %tmp361, 7		; <bool> [#uses=1]
	%iftmp.35.0 = cast uint %iftmp.35.0 to int		; <int> [#uses=1]
	%tmp426 = cast uint %rmask.0.1 to ubyte		; <ubyte> [#uses=1]
	%tmp426not = xor ubyte %tmp426, 255		; <ubyte> [#uses=1]
	%tmp428 = or ubyte %tmp347, %tmp426not		; <ubyte> [#uses=1]
	%tmp436 = and uint %iftmp.37.0, %rmask.0.1		; <uint> [#uses=1]
	%tmp436 = cast uint %tmp436 to ubyte		; <ubyte> [#uses=1]
	br label %cond_true457

cond_true457:		; preds = %cond_true457.preheader, %cond_next442
	%indvar74 = phi uint [ 0, %cond_true457.preheader ], [ %indvar.next75, %cond_next442 ]		; <uint> [#uses=3]
	%dest.3.0 = phi ubyte* [ %tmp448, %cond_next442 ], [ %tmp100, %cond_true457.preheader ]		; <ubyte*> [#uses=3]
	%dest_line.3.0 = phi ubyte** [ %tmp444, %cond_next442 ], [ %tmp96, %cond_true457.preheader ]		; <ubyte**> [#uses=1]
	%tmp.77 = sub uint 0, %indvar74		; <uint> [#uses=1]
	%tmp.77 = cast uint %tmp.77 to int		; <int> [#uses=1]
	%tmp454.0 = add int %tmp.77, %tmp45438		; <int> [#uses=1]
	%tmp.79 = mul uint %indvar74, %raster		; <uint> [#uses=1]
	%tmp.79 = cast uint %tmp.79 to int		; <int> [#uses=1]
	%tmp104.sum = add int %tmp102, %tmp.79		; <int> [#uses=1]
	%line.3.0 = getelementptr ubyte* %base, int %tmp104.sum		; <ubyte*> [#uses=3]
	%tmp318 = load ubyte* %line.3.0		; <ubyte> [#uses=2]
	br bool %tmp315, label %cond_false321, label %cond_true316

return.loopexit:		; preds = %cond_next280
	br label %return.loopexit.split

return.loopexit.split:		; preds = %return.loopexit, %return.loopexit.us
	br label %return

return.loopexit56:		; preds = %cond_next442
	br label %return

return:		; preds = %return.loopexit56, %return.loopexit.split, %cond_false299, %cond_true197, %cond_next78, %cond_next63, %bb58, %cond_next46
	%retval.0 = phi int [ 0, %cond_next46 ], [ -1, %bb58 ], [ -1, %cond_next63 ], [ -1, %cond_next78 ], [ 0, %cond_true197 ], [ 0, %cond_false299 ], [ 0, %return.loopexit.split ], [ 0, %return.loopexit56 ]		; <int> [#uses=1]
	ret int %retval.0
}

declare int %mem_no_fault_proc(%struct.gx_device_memory*, int, int, int, int, int)

declare int %mem_mono_fill_rectangle(%struct.gx_device*, int, int, int, int, uint)

declare int %mem_copy_mono_recover(%struct.gx_device*, ubyte*, int, int, int, int, int, int, uint, uint, int)
