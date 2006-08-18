; RUN: llvm-as < %s | opt -analyze -datastructure

target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"
deplibs = [ "c", "crtend", "stdc++" ]
	%struct.Blend_Map_Entry = type { float, ubyte, { [2 x double], [4 x ubyte] } }
	%struct.Blend_Map_Struct = type { short, short, short, int, %struct.Blend_Map_Entry* }
	%struct.Image_Colour_Struct = type { ushort, ushort, ushort, ushort, ushort }
	%struct.Image_Struct = type { int, int, int, int, int, short, short, [3 x double], float, float, int, int, short, %struct.Image_Colour_Struct*, { ubyte** } }
	%struct.Pattern_Struct = type { ushort, ushort, ushort, int, float, float, float, %struct.Warps_Struct*, %struct.Pattern_Struct*, %struct.Blend_Map_Struct*, { [3 x double], [4 x ubyte] } }
	%struct.Tnormal_Struct = type { ushort, ushort, ushort, int, float, float, float, %struct.Warps_Struct*, %struct.Pattern_Struct*, %struct.Blend_Map_Struct*, { [3 x double], [4 x ubyte] }, float }
	%struct.Warps_Struct = type { ushort, %struct.Warps_Struct* }

implementation   ; Functions:

declare fastcc %struct.Image_Struct* %Parse_Image()

fastcc void %Parse_Bump_Map(%struct.Tnormal_Struct* %Tnormal) {
entry:
	%tmp.0 = tail call fastcc %struct.Image_Struct* %Parse_Image( )		; <%struct.Image_Struct*> [#uses=1]
	%tmp.28 = getelementptr %struct.Tnormal_Struct* %Tnormal, int 0, uint 10		; <{ [3 x double], [4 x ubyte] }*> [#uses=1]
	%tmp.32 = cast { [3 x double], [4 x ubyte] }* %tmp.28 to %struct.Image_Struct**		; <%struct.Image_Struct**> [#uses=1]
	store %struct.Image_Struct* %tmp.0, %struct.Image_Struct** %tmp.32
	ret void
}
