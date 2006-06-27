; RUN: llvm-as < %s | opt -globalsmodref-aa -dse -disable-output
target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin8"
	%struct.ECacheType = type { uint, uint, int }
	%struct.FILE = type { ubyte*, int, int, short, short, %struct.__sbuf, int, sbyte*, int (sbyte*)*, int (sbyte*, sbyte*, int)*, long (sbyte*, long, int)*, int (sbyte*, sbyte*, int)*, %struct.__sbuf, %struct.__sFILEX*, int, [3 x ubyte], [1 x ubyte], %struct.__sbuf, int, long }
	%struct.QTType = type { sbyte, sbyte, ushort, uint, uint, int }
	%struct.TType = type { sbyte, sbyte, sbyte, sbyte, ushort, uint, uint, int }
	%struct._RuneEntry = type { int, int, int, uint* }
	%struct._RuneLocale = type { [8 x sbyte], [32 x sbyte], int (sbyte*, uint, sbyte**)*, int (int, sbyte*, uint, sbyte**)*, int, [256 x uint], [256 x int], [256 x int], %struct._RuneRange, %struct._RuneRange, %struct._RuneRange, sbyte*, int }
	%struct._RuneRange = type { int, %struct._RuneEntry* }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { ubyte*, int }
	%struct.move_s = type { int, int, int, int, int, int }
	%struct.move_x = type { int, int, int, int }
	%struct.node_t = type { ubyte, ubyte, ubyte, ubyte, int, int, %struct.node_t**, %struct.node_t*, %struct.move_s }
	%struct.see_data = type { int, int }
%rook_o.2925 = internal global [4 x int] [ int 12, int -12, int 1, int -1 ]		; <[4 x int]*> [#uses=0]
%bishop_o.2926 = internal global [4 x int] [ int 11, int -11, int 13, int -13 ]		; <[4 x int]*> [#uses=0]
%knight_o.2927 = internal global [8 x int] [ int 10, int -10, int 14, int -14, int 23, int -23, int 25, int -25 ]		; <[8 x int]*> [#uses=0]
%board = internal global [144 x int] zeroinitializer		; <[144 x int]*> [#uses=0]
%holding = internal global [2 x [16 x int]] zeroinitializer		; <[2 x [16 x int]]*> [#uses=0]
%hold_hash = internal global uint 0		; <uint*> [#uses=0]
%white_hand_eval = internal global int 0		; <int*> [#uses=0]
%black_hand_eval = internal global int 0		; <int*> [#uses=0]
%num_holding = internal global [2 x int] zeroinitializer		; <[2 x int]*> [#uses=0]
%zobrist = internal global [14 x [144 x uint]] zeroinitializer		; <[14 x [144 x uint]]*> [#uses=0]
%Variant = internal global int 0		; <int*> [#uses=7]
%userealholdings.b = internal global bool false		; <bool*> [#uses=1]
%realholdings = internal global [255 x sbyte] zeroinitializer		; <[255 x sbyte]*> [#uses=0]
%comp_color = internal global int 0		; <int*> [#uses=0]
%C.97.3177 = internal global [13 x int] [ int 0, int 2, int 1, int 4, int 3, int 0, int 0, int 8, int 7, int 10, int 9, int 12, int 11 ]		; <[13 x int]*> [#uses=0]
%str = internal global [30 x sbyte] c"%s:%u: failed assertion `%s'\0A\00"		; <[30 x sbyte]*> [#uses=0]
%str = internal global [81 x sbyte] c"/Volumes/Stuff/src/speccpu2006-091-llvm/benchspec//CPU2006/458.sjeng/src/crazy.c\00"		; <[81 x sbyte]*> [#uses=0]
%str = internal global [32 x sbyte] c"piece > frame && piece < npiece\00"		; <[32 x sbyte]*> [#uses=0]
%C.101.3190 = internal global [13 x int] [ int 0, int 2, int 1, int 2, int 1, int 0, int 0, int 2, int 1, int 2, int 1, int 2, int 1 ]		; <[13 x int]*> [#uses=0]
%hand_value = internal global [13 x int] [ int 0, int 100, int -100, int 210, int -210, int 0, int 0, int 250, int -250, int 450, int -450, int 230, int -230 ]		; <[13 x int]*> [#uses=0]
%material = internal global [14 x int] zeroinitializer		; <[14 x int]*> [#uses=0]
%Material = internal global int 0		; <int*> [#uses=0]
%str = internal global [23 x sbyte] c"holding[who][what] > 0\00"		; <[23 x sbyte]*> [#uses=0]
%str = internal global [24 x sbyte] c"holding[who][what] < 20\00"		; <[24 x sbyte]*> [#uses=0]
%fifty = internal global int 0		; <int*> [#uses=0]
%move_number = internal global int 0		; <int*> [#uses=1]
%ply = internal global int 0		; <int*> [#uses=2]
%hash_history = internal global [600 x uint] zeroinitializer		; <[600 x uint]*> [#uses=1]
%hash = internal global uint 0		; <uint*> [#uses=1]
%ECacheSize.b = internal global bool false		; <bool*> [#uses=1]
%ECache = internal global %struct.ECacheType* null		; <%struct.ECacheType**> [#uses=1]
%ECacheProbes = internal global uint 0		; <uint*> [#uses=1]
%ECacheHits = internal global uint 0		; <uint*> [#uses=1]
%str = internal global [34 x sbyte] c"Out of memory allocating ECache.\0A\00"		; <[34 x sbyte]*> [#uses=0]
%rankoffsets.2930 = internal global [8 x int] [ int 110, int 98, int 86, int 74, int 62, int 50, int 38, int 26 ]		; <[8 x int]*> [#uses=0]
%white_castled = internal global int 0		; <int*> [#uses=0]
%black_castled = internal global int 0		; <int*> [#uses=0]
%book_ply = internal global int 0		; <int*> [#uses=0]
%bking_loc = internal global int 0		; <int*> [#uses=1]
%wking_loc = internal global int 0		; <int*> [#uses=1]
%white_to_move = internal global int 0		; <int*> [#uses=3]
%moved = internal global [144 x int] zeroinitializer		; <[144 x int]*> [#uses=0]
%ep_square = internal global int 0		; <int*> [#uses=0]
%_DefaultRuneLocale = external global %struct._RuneLocale		; <%struct._RuneLocale*> [#uses=0]
%str = internal global [3 x sbyte] c"bm\00"		; <[3 x sbyte]*> [#uses=0]
%str1 = internal global [3 x sbyte] c"am\00"		; <[3 x sbyte]*> [#uses=0]
%str1 = internal global [34 x sbyte] c"No best-move or avoid-move found!\00"		; <[34 x sbyte]*> [#uses=0]
%str = internal global [25 x sbyte] c"\0AName of EPD testsuite: \00"		; <[25 x sbyte]*> [#uses=0]
%__sF = external global [0 x %struct.FILE]		; <[0 x %struct.FILE]*> [#uses=0]
%str = internal global [21 x sbyte] c"\0ATime per move (s): \00"		; <[21 x sbyte]*> [#uses=0]
%str = internal global [2 x sbyte] c"\0A\00"		; <[2 x sbyte]*> [#uses=0]
%str2 = internal global [2 x sbyte] c"r\00"		; <[2 x sbyte]*> [#uses=0]
%root_to_move = internal global int 0		; <int*> [#uses=1]
%forcedwin.b = internal global bool false		; <bool*> [#uses=2]
%fixed_time = internal global int 0		; <int*> [#uses=1]
%nodes = internal global int 0		; <int*> [#uses=1]
%qnodes = internal global int 0		; <int*> [#uses=1]
%str = internal global [29 x sbyte] c"\0ANodes: %i (%0.2f%% qnodes)\0A\00"		; <[29 x sbyte]*> [#uses=0]
%str = internal global [54 x sbyte] c"ECacheProbes : %u   ECacheHits : %u   HitRate : %f%%\0A\00"		; <[54 x sbyte]*> [#uses=0]
%TTStores = internal global uint 0		; <uint*> [#uses=1]
%TTProbes = internal global uint 0		; <uint*> [#uses=1]
%TTHits = internal global uint 0		; <uint*> [#uses=1]
%str = internal global [60 x sbyte] c"TTStores : %u TTProbes : %u   TTHits : %u   HitRate : %f%%\0A\00"		; <[60 x sbyte]*> [#uses=0]
%NTries = internal global uint 0		; <uint*> [#uses=1]
%NCuts = internal global uint 0		; <uint*> [#uses=1]
%TExt = internal global uint 0		; <uint*> [#uses=1]
%str = internal global [51 x sbyte] c"NTries : %u  NCuts : %u  CutRate : %f%%  TExt: %u\0A\00"		; <[51 x sbyte]*> [#uses=0]
%ext_check = internal global uint 0		; <uint*> [#uses=1]
%razor_drop = internal global uint 0		; <uint*> [#uses=1]
%razor_material = internal global uint 0		; <uint*> [#uses=1]
%str = internal global [61 x sbyte] c"Check extensions: %u  Razor drops : %u  Razor Material : %u\0A\00"		; <[61 x sbyte]*> [#uses=0]
%FHF = internal global uint 0		; <uint*> [#uses=1]
%FH = internal global uint 0		; <uint*> [#uses=1]
%str = internal global [22 x sbyte] c"Move ordering : %f%%\0A\00"		; <[22 x sbyte]*> [#uses=0]
%maxposdiff = internal global int 0		; <int*> [#uses=1]
%str = internal global [47 x sbyte] c"Material score: %d  Eval : %d  MaxPosDiff: %d\0A\00"		; <[47 x sbyte]*> [#uses=0]
%str = internal global [17 x sbyte] c"Solution found.\0A\00"		; <[17 x sbyte]*> [#uses=0]
%str3 = internal global [21 x sbyte] c"Solution not found.\0A\00"		; <[21 x sbyte]*> [#uses=0]
%str = internal global [15 x sbyte] c"Solved: %d/%d\0A\00"		; <[15 x sbyte]*> [#uses=0]
%str = internal global [9 x sbyte] c"EPD: %s\0A\00"		; <[9 x sbyte]*> [#uses=0]
%str4 = internal global [21 x sbyte] c"Searching to %d ply\0A\00"		; <[21 x sbyte]*> [#uses=0]
%maxdepth = internal global int 0		; <int*> [#uses=0]
%std_material = internal global [14 x int] [ int 0, int 100, int -100, int 310, int -310, int 4000, int -4000, int 500, int -500, int 900, int -900, int 325, int -325, int 0 ]		; <[14 x int]*> [#uses=0]
%zh_material = internal global [14 x int] [ int 0, int 100, int -100, int 210, int -210, int 4000, int -4000, int 250, int -250, int 450, int -450, int 230, int -230, int 0 ]		; <[14 x int]*> [#uses=0]
%suicide_material = internal global [14 x int] [ int 0, int 15, int -15, int 150, int -150, int 500, int -500, int 150, int -150, int 50, int -50, int 0, int 0, int 0 ]		; <[14 x int]*> [#uses=0]
%losers_material = internal global [14 x int] [ int 0, int 80, int -80, int 320, int -320, int 1000, int -1000, int 350, int -350, int 400, int -400, int 270, int -270, int 0 ]		; <[14 x int]*> [#uses=0]
%Xfile = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 1, int 2, int 3, int 4, int 5, int 6, int 7, int 8, int 0, int 0, int 0, int 0, int 1, int 2, int 3, int 4, int 5, int 6, int 7, int 8, int 0, int 0, int 0, int 0, int 1, int 2, int 3, int 4, int 5, int 6, int 7, int 8, int 0, int 0, int 0, int 0, int 1, int 2, int 3, int 4, int 5, int 6, int 7, int 8, int 0, int 0, int 0, int 0, int 1, int 2, int 3, int 4, int 5, int 6, int 7, int 8, int 0, int 0, int 0, int 0, int 1, int 2, int 3, int 4, int 5, int 6, int 7, int 8, int 0, int 0, int 0, int 0, int 1, int 2, int 3, int 4, int 5, int 6, int 7, int 8, int 0, int 0, int 0, int 0, int 1, int 2, int 3, int 4, int 5, int 6, int 7, int 8, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%Xrank = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 1, int 1, int 1, int 1, int 1, int 1, int 1, int 1, int 0, int 0, int 0, int 0, int 2, int 2, int 2, int 2, int 2, int 2, int 2, int 2, int 0, int 0, int 0, int 0, int 3, int 3, int 3, int 3, int 3, int 3, int 3, int 3, int 0, int 0, int 0, int 0, int 4, int 4, int 4, int 4, int 4, int 4, int 4, int 4, int 0, int 0, int 0, int 0, int 5, int 5, int 5, int 5, int 5, int 5, int 5, int 5, int 0, int 0, int 0, int 0, int 6, int 6, int 6, int 6, int 6, int 6, int 6, int 6, int 0, int 0, int 0, int 0, int 7, int 7, int 7, int 7, int 7, int 7, int 7, int 7, int 0, int 0, int 0, int 0, int 8, int 8, int 8, int 8, int 8, int 8, int 8, int 8, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%Xdiagl = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 1, int 2, int 3, int 4, int 5, int 6, int 7, int 8, int 0, int 0, int 0, int 0, int 9, int 1, int 2, int 3, int 4, int 5, int 6, int 7, int 0, int 0, int 0, int 0, int 10, int 9, int 1, int 2, int 3, int 4, int 5, int 6, int 0, int 0, int 0, int 0, int 11, int 10, int 9, int 1, int 2, int 3, int 4, int 5, int 0, int 0, int 0, int 0, int 12, int 11, int 10, int 9, int 1, int 2, int 3, int 4, int 0, int 0, int 0, int 0, int 13, int 12, int 11, int 10, int 9, int 1, int 2, int 3, int 0, int 0, int 0, int 0, int 14, int 13, int 12, int 11, int 10, int 9, int 1, int 2, int 0, int 0, int 0, int 0, int 15, int 14, int 13, int 12, int 11, int 10, int 9, int 1, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%Xdiagr = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 15, int 14, int 13, int 12, int 11, int 10, int 9, int 1, int 0, int 0, int 0, int 0, int 14, int 13, int 12, int 11, int 10, int 9, int 1, int 2, int 0, int 0, int 0, int 0, int 13, int 12, int 11, int 10, int 9, int 1, int 2, int 3, int 0, int 0, int 0, int 0, int 12, int 11, int 10, int 9, int 1, int 2, int 3, int 4, int 0, int 0, int 0, int 0, int 11, int 10, int 9, int 1, int 2, int 3, int 4, int 5, int 0, int 0, int 0, int 0, int 10, int 9, int 1, int 2, int 3, int 4, int 5, int 6, int 0, int 0, int 0, int 0, int 9, int 1, int 2, int 3, int 4, int 5, int 6, int 7, int 0, int 0, int 0, int 0, int 1, int 2, int 3, int 4, int 5, int 6, int 7, int 8, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%sqcolor = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 0, int 0, int 0, int 0, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 0, int 0, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 0, int 0, int 0, int 0, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 0, int 0, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 0, int 0, int 0, int 0, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 0, int 0, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 0, int 0, int 0, int 0, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 1, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%pcsqbishop = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int -5, int -5, int -10, int -5, int -5, int -10, int -5, int -5, int 0, int 0, int 0, int 0, int -5, int 10, int 5, int 10, int 10, int 5, int 10, int -5, int 0, int 0, int 0, int 0, int -5, int 5, int 6, int 15, int 15, int 6, int 5, int -5, int 0, int 0, int 0, int 0, int -5, int 3, int 15, int 10, int 10, int 15, int 3, int -5, int 0, int 0, int 0, int 0, int -5, int 3, int 15, int 10, int 10, int 15, int 3, int -5, int 0, int 0, int 0, int 0, int -5, int 5, int 6, int 15, int 15, int 6, int 5, int -5, int 0, int 0, int 0, int 0, int -5, int 10, int 5, int 10, int 10, int 5, int 10, int -5, int 0, int 0, int 0, int 0, int -5, int -5, int -10, int -5, int -5, int -10, int -5, int -5, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%black_knight = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int -20, int -10, int -10, int -10, int -10, int -10, int -10, int -20, int 0, int 0, int 0, int 0, int -10, int 15, int 25, int 25, int 25, int 25, int 15, int -10, int 0, int 0, int 0, int 0, int -10, int 15, int 25, int 35, int 35, int 35, int 15, int -10, int 0, int 0, int 0, int 0, int -10, int 10, int 25, int 20, int 25, int 25, int 10, int -10, int 0, int 0, int 0, int 0, int -10, int 0, int 20, int 20, int 20, int 20, int 0, int -10, int 0, int 0, int 0, int 0, int -10, int 0, int 15, int 15, int 15, int 15, int 0, int -10, int 0, int 0, int 0, int 0, int -10, int 0, int 0, int 3, int 3, int 0, int 0, int -10, int 0, int 0, int 0, int 0, int -20, int -35, int -10, int -10, int -10, int -10, int -35, int -20, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%white_knight = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int -20, int -35, int -10, int -10, int -10, int -10, int -35, int -20, int 0, int 0, int 0, int 0, int -10, int 0, int 0, int 3, int 3, int 0, int 0, int -10, int 0, int 0, int 0, int 0, int -10, int 0, int 15, int 15, int 15, int 15, int 0, int -10, int 0, int 0, int 0, int 0, int -10, int 0, int 20, int 20, int 20, int 20, int 0, int -10, int 0, int 0, int 0, int 0, int -10, int 10, int 25, int 20, int 25, int 25, int 10, int -10, int 0, int 0, int 0, int 0, int -10, int 15, int 25, int 35, int 35, int 35, int 15, int -10, int 0, int 0, int 0, int 0, int -10, int 15, int 25, int 25, int 25, int 25, int 15, int -10, int 0, int 0, int 0, int 0, int -20, int -10, int -10, int -10, int -10, int -10, int -10, int -20, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%white_pawn = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 25, int 25, int 35, int 5, int 5, int 50, int 45, int 30, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 7, int 7, int 5, int 5, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 14, int 14, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 10, int 20, int 20, int 10, int 5, int 5, int 0, int 0, int 0, int 0, int 12, int 18, int 18, int 27, int 27, int 18, int 18, int 18, int 0, int 0, int 0, int 0, int 25, int 30, int 30, int 35, int 35, int 35, int 30, int 25, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%black_pawn = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 30, int 30, int 30, int 35, int 35, int 35, int 30, int 25, int 0, int 0, int 0, int 0, int 12, int 18, int 18, int 27, int 27, int 18, int 18, int 18, int 0, int 0, int 0, int 0, int 0, int 0, int 10, int 20, int 20, int 10, int 5, int 5, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 14, int 14, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 7, int 7, int 5, int 5, int 0, int 0, int 0, int 0, int 0, int 25, int 25, int 35, int 5, int 5, int 50, int 45, int 30, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%white_king = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int -100, int 7, int 4, int 0, int 10, int 4, int 7, int -100, int 0, int 0, int 0, int 0, int -250, int -200, int -150, int -100, int -100, int -150, int -200, int -250, int 0, int 0, int 0, int 0, int -350, int -300, int -300, int -250, int -250, int -300, int -300, int -350, int 0, int 0, int 0, int 0, int -400, int -400, int -400, int -350, int -350, int -400, int -400, int -400, int 0, int 0, int 0, int 0, int -450, int -450, int -450, int -450, int -450, int -450, int -450, int -450, int 0, int 0, int 0, int 0, int -500, int -500, int -500, int -500, int -500, int -500, int -500, int -500, int 0, int 0, int 0, int 0, int -500, int -500, int -500, int -500, int -500, int -500, int -500, int -500, int 0, int 0, int 0, int 0, int -500, int -500, int -500, int -500, int -500, int -500, int -500, int -500, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%black_king = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int -500, int -500, int -500, int -500, int -500, int -500, int -500, int -500, int 0, int 0, int 0, int 0, int -500, int -500, int -500, int -500, int -500, int -500, int -500, int -500, int 0, int 0, int 0, int 0, int -500, int -500, int -500, int -500, int -500, int -500, int -500, int -500, int 0, int 0, int 0, int 0, int -450, int -450, int -450, int -450, int -450, int -450, int -450, int -450, int 0, int 0, int 0, int 0, int -400, int -400, int -400, int -350, int -350, int -400, int -400, int -400, int 0, int 0, int 0, int 0, int -350, int -300, int -300, int -250, int -250, int -300, int -300, int -350, int 0, int 0, int 0, int 0, int -250, int -200, int -150, int -100, int -100, int -150, int -200, int -250, int 0, int 0, int 0, int 0, int -100, int 7, int 4, int 0, int 10, int 4, int 7, int -100, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%black_queen = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 5, int 5, int 5, int 10, int 10, int 5, int 5, int 5, int 0, int 0, int 0, int 0, int 0, int 0, int 3, int 3, int 3, int 3, int 3, int 0, int 0, int 0, int 0, int 0, int -30, int -30, int -30, int -30, int -30, int -30, int -30, int -30, int 0, int 0, int 0, int 0, int -60, int -40, int -40, int -60, int -60, int -40, int -40, int -60, int 0, int 0, int 0, int 0, int -40, int -40, int -40, int -40, int -40, int -40, int -40, int -40, int 0, int 0, int 0, int 0, int -15, int -15, int -15, int -10, int -10, int -15, int -15, int -15, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 7, int 10, int 5, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 5, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%white_queen = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 5, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 7, int 10, int 5, int 0, int 0, int 0, int 0, int 0, int 0, int -15, int -15, int -15, int -10, int -10, int -15, int -15, int -15, int 0, int 0, int 0, int 0, int -40, int -40, int -40, int -40, int -40, int -40, int -40, int -40, int 0, int 0, int 0, int 0, int -60, int -40, int -40, int -60, int -60, int -40, int -40, int -60, int 0, int 0, int 0, int 0, int -30, int -30, int -30, int -30, int -30, int -30, int -30, int -30, int 0, int 0, int 0, int 0, int 0, int 0, int 3, int 3, int 3, int 3, int 3, int 0, int 0, int 0, int 0, int 0, int 5, int 5, int 5, int 10, int 10, int 5, int 5, int 5, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%black_rook = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 10, int 15, int 20, int 25, int 25, int 20, int 15, int 10, int 0, int 0, int 0, int 0, int 0, int 10, int 15, int 20, int 20, int 15, int 10, int 0, int 0, int 0, int 0, int 0, int -20, int -20, int -20, int -20, int -20, int -20, int -20, int -20, int 0, int 0, int 0, int 0, int -20, int -20, int -20, int -30, int -30, int -20, int -20, int -20, int 0, int 0, int 0, int 0, int -20, int -20, int -20, int -20, int -20, int -20, int -20, int -20, int 0, int 0, int 0, int 0, int -15, int -15, int -15, int -10, int -10, int -15, int -15, int -15, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 7, int 10, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 2, int 2, int 2, int 2, int 2, int 2, int 2, int 2, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%white_rook = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 2, int 2, int 2, int 2, int 2, int 2, int 2, int 2, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 7, int 10, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int -15, int -15, int -15, int -10, int -10, int -15, int -15, int -15, int 0, int 0, int 0, int 0, int -20, int -20, int -20, int -20, int -20, int -20, int -20, int -20, int 0, int 0, int 0, int 0, int -20, int -20, int -20, int -30, int -30, int -20, int -20, int -20, int 0, int 0, int 0, int 0, int -20, int -20, int -20, int -20, int -20, int -20, int -20, int -20, int 0, int 0, int 0, int 0, int 0, int 10, int 15, int 20, int 20, int 15, int 10, int 0, int 0, int 0, int 0, int 0, int 10, int 15, int 20, int 25, int 25, int 20, int 15, int 10, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%upscale = internal global [64 x int] [ int 26, int 27, int 28, int 29, int 30, int 31, int 32, int 33, int 38, int 39, int 40, int 41, int 42, int 43, int 44, int 45, int 50, int 51, int 52, int 53, int 54, int 55, int 56, int 57, int 62, int 63, int 64, int 65, int 66, int 67, int 68, int 69, int 74, int 75, int 76, int 77, int 78, int 79, int 80, int 81, int 86, int 87, int 88, int 89, int 90, int 91, int 92, int 93, int 98, int 99, int 100, int 101, int 102, int 103, int 104, int 105, int 110, int 111, int 112, int 113, int 114, int 115, int 116, int 117 ]		; <[64 x int]*> [#uses=0]
%pre_p_tropism = internal global [9 x int] [ int 9999, int 40, int 20, int 10, int 3, int 1, int 1, int 0, int 9999 ]		; <[9 x int]*> [#uses=0]
%pre_r_tropism = internal global [9 x int] [ int 9999, int 50, int 40, int 15, int 5, int 1, int 1, int 0, int 9999 ]		; <[9 x int]*> [#uses=0]
%pre_n_tropism = internal global [9 x int] [ int 9999, int 50, int 70, int 35, int 10, int 2, int 1, int 0, int 9999 ]		; <[9 x int]*> [#uses=0]
%pre_q_tropism = internal global [9 x int] [ int 9999, int 100, int 60, int 20, int 5, int 2, int 0, int 0, int 9999 ]		; <[9 x int]*> [#uses=0]
%pre_b_tropism = internal global [9 x int] [ int 9999, int 50, int 25, int 15, int 5, int 2, int 2, int 2, int 9999 ]		; <[9 x int]*> [#uses=0]
%rookdistance = internal global [144 x [144 x int]] zeroinitializer		; <[144 x [144 x int]]*> [#uses=0]
%distance = internal global [144 x [144 x int]] zeroinitializer		; <[144 x [144 x int]]*> [#uses=0]
%p_tropism = internal global [144 x [144 x ubyte]] zeroinitializer		; <[144 x [144 x ubyte]]*> [#uses=0]
%b_tropism = internal global [144 x [144 x ubyte]] zeroinitializer		; <[144 x [144 x ubyte]]*> [#uses=0]
%n_tropism = internal global [144 x [144 x ubyte]] zeroinitializer		; <[144 x [144 x ubyte]]*> [#uses=0]
%r_tropism = internal global [144 x [144 x ubyte]] zeroinitializer		; <[144 x [144 x ubyte]]*> [#uses=0]
%q_tropism = internal global [144 x [144 x ubyte]] zeroinitializer		; <[144 x [144 x ubyte]]*> [#uses=0]
%cfg_devscale.b = internal global bool false		; <bool*> [#uses=0]
%pieces = internal global [62 x int] zeroinitializer		; <[62 x int]*> [#uses=0]
%piece_count = internal global int 0		; <int*> [#uses=1]
%cfg_smarteval.b = internal global bool false		; <bool*> [#uses=0]
%lcentral = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int -20, int -15, int -15, int -15, int -15, int -15, int -15, int -20, int 0, int 0, int 0, int 0, int -15, int 0, int 3, int 5, int 5, int 3, int 0, int -15, int 0, int 0, int 0, int 0, int -15, int 0, int 15, int 15, int 15, int 15, int 0, int -15, int 0, int 0, int 0, int 0, int -15, int 0, int 15, int 30, int 30, int 15, int 0, int -15, int 0, int 0, int 0, int 0, int -15, int 0, int 15, int 30, int 30, int 15, int 0, int -15, int 0, int 0, int 0, int 0, int -15, int 0, int 15, int 15, int 15, int 15, int 0, int -15, int 0, int 0, int 0, int 0, int -15, int 0, int 3, int 5, int 5, int 3, int 0, int -15, int 0, int 0, int 0, int 0, int -20, int -15, int -15, int -15, int -15, int -15, int -15, int -20, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%str3 = internal global [81 x sbyte] c"/Volumes/Stuff/src/speccpu2006-091-llvm/benchspec//CPU2006/458.sjeng/src/leval.c\00"		; <[81 x sbyte]*> [#uses=0]
%str5 = internal global [21 x sbyte] c"(i > 0) && (i < 145)\00"		; <[21 x sbyte]*> [#uses=0]
%kingcap.b = internal global bool false		; <bool*> [#uses=0]
%numb_moves = internal global int 0		; <int*> [#uses=2]
%genfor = internal global %struct.move_s* null		; <%struct.move_s**> [#uses=0]
%captures = internal global uint 0		; <uint*> [#uses=1]
%fcaptures.b = internal global bool false		; <bool*> [#uses=0]
%gfrom = internal global int 0		; <int*> [#uses=0]
%Giveaway.b = internal global bool false		; <bool*> [#uses=0]
%path_x = internal global [300 x %struct.move_x] zeroinitializer		; <[300 x %struct.move_x]*> [#uses=0]
%str7 = internal global [81 x sbyte] c"/Volumes/Stuff/src/speccpu2006-091-llvm/benchspec//CPU2006/458.sjeng/src/moves.c\00"		; <[81 x sbyte]*> [#uses=0]
%str8 = internal global [15 x sbyte] c"find_slot < 63\00"		; <[15 x sbyte]*> [#uses=0]
%is_promoted = internal global [62 x int] zeroinitializer		; <[62 x int]*> [#uses=0]
%squares = internal global [144 x int] zeroinitializer		; <[144 x int]*> [#uses=0]
%str = internal global [38 x sbyte] c"promoted > frame && promoted < npiece\00"		; <[38 x sbyte]*> [#uses=0]
%str1 = internal global [38 x sbyte] c"promoted < npiece && promoted > frame\00"		; <[38 x sbyte]*> [#uses=0]
%evalRoutines = internal global [7 x int (int, int)*] [ int (int, int)* %ErrorIt, int (int, int)* %Pawn, int (int, int)* %Knight, int (int, int)* %King, int (int, int)* %Rook, int (int, int)* %Queen, int (int, int)* %Bishop ]		; <[7 x int (int, int)*]*> [#uses=0]
%sbishop = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int -2, int -2, int -2, int -2, int -2, int -2, int -2, int -2, int 0, int 0, int 0, int 0, int -2, int 8, int 5, int 5, int 5, int 5, int 8, int -2, int 0, int 0, int 0, int 0, int -2, int 3, int 3, int 5, int 5, int 3, int 3, int -2, int 0, int 0, int 0, int 0, int -2, int 2, int 5, int 4, int 4, int 5, int 2, int -2, int 0, int 0, int 0, int 0, int -2, int 2, int 5, int 4, int 4, int 5, int 2, int -2, int 0, int 0, int 0, int 0, int -2, int 3, int 3, int 5, int 5, int 3, int 3, int -2, int 0, int 0, int 0, int 0, int -2, int 8, int 5, int 5, int 5, int 5, int 8, int -2, int 0, int 0, int 0, int 0, int -2, int -2, int -2, int -2, int -2, int -2, int -2, int -2, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%sknight = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int -20, int -10, int -10, int -10, int -10, int -10, int -10, int -20, int 0, int 0, int 0, int 0, int -10, int 0, int 0, int 3, int 3, int 0, int 0, int -10, int 0, int 0, int 0, int 0, int -10, int 0, int 5, int 5, int 5, int 5, int 0, int -10, int 0, int 0, int 0, int 0, int -10, int 0, int 5, int 10, int 10, int 5, int 0, int -10, int 0, int 0, int 0, int 0, int -10, int 0, int 5, int 10, int 10, int 5, int 0, int -10, int 0, int 0, int 0, int 0, int -10, int 0, int 5, int 5, int 5, int 5, int 0, int -10, int 0, int 0, int 0, int 0, int -10, int 0, int 0, int 3, int 3, int 0, int 0, int -10, int 0, int 0, int 0, int 0, int -20, int -10, int -10, int -10, int -10, int -10, int -10, int -20, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%swhite_pawn = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 1, int 2, int 3, int 10, int 10, int 3, int 2, int 1, int 0, int 0, int 0, int 0, int 2, int 4, int 6, int 12, int 12, int 6, int 4, int 2, int 0, int 0, int 0, int 0, int 3, int 6, int 9, int 14, int 14, int 9, int 6, int 3, int 0, int 0, int 0, int 0, int 10, int 12, int 14, int 16, int 16, int 14, int 12, int 10, int 0, int 0, int 0, int 0, int 20, int 22, int 24, int 26, int 26, int 24, int 22, int 20, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%sblack_pawn = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 20, int 22, int 24, int 26, int 26, int 24, int 22, int 20, int 0, int 0, int 0, int 0, int 10, int 12, int 14, int 16, int 16, int 14, int 12, int 10, int 0, int 0, int 0, int 0, int 3, int 6, int 9, int 14, int 14, int 9, int 6, int 3, int 0, int 0, int 0, int 0, int 2, int 4, int 6, int 12, int 12, int 6, int 4, int 2, int 0, int 0, int 0, int 0, int 1, int 2, int 3, int 10, int 10, int 3, int 2, int 1, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%swhite_king = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 2, int 14, int 0, int 0, int 0, int 9, int 14, int 2, int 0, int 0, int 0, int 0, int -3, int -5, int -6, int -6, int -6, int -6, int -5, int -3, int 0, int 0, int 0, int 0, int -5, int -5, int -8, int -8, int -8, int -8, int -5, int -5, int 0, int 0, int 0, int 0, int -8, int -8, int -13, int -13, int -13, int -13, int -8, int -8, int 0, int 0, int 0, int 0, int -13, int -13, int -21, int -21, int -21, int -21, int -13, int -13, int 0, int 0, int 0, int 0, int -21, int -21, int -34, int -34, int -34, int -34, int -21, int -21, int 0, int 0, int 0, int 0, int -34, int -34, int -55, int -55, int -55, int -55, int -34, int -34, int 0, int 0, int 0, int 0, int -55, int -55, int -89, int -89, int -89, int -89, int -55, int -55, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%sblack_king = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int -55, int -55, int -89, int -89, int -89, int -89, int -55, int -55, int 0, int 0, int 0, int 0, int -34, int -34, int -55, int -55, int -55, int -55, int -34, int -34, int 0, int 0, int 0, int 0, int -21, int -21, int -34, int -34, int -34, int -34, int -21, int -21, int 0, int 0, int 0, int 0, int -13, int -13, int -21, int -21, int -21, int -21, int -13, int -13, int 0, int 0, int 0, int 0, int -8, int -8, int -13, int -13, int -13, int -13, int -8, int -8, int 0, int 0, int 0, int 0, int -5, int -5, int -8, int -8, int -8, int -8, int -5, int -5, int 0, int 0, int 0, int 0, int -3, int -5, int -6, int -6, int -6, int -6, int -5, int -3, int 0, int 0, int 0, int 0, int 2, int 14, int 0, int 0, int 0, int 9, int 14, int 2, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%send_king = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int -5, int -3, int -1, int 0, int 0, int -1, int -3, int -5, int 0, int 0, int 0, int 0, int -3, int 10, int 10, int 10, int 10, int 10, int 10, int -3, int 0, int 0, int 0, int 0, int -1, int 10, int 25, int 25, int 25, int 25, int 10, int -1, int 0, int 0, int 0, int 0, int 0, int 10, int 25, int 50, int 50, int 25, int 10, int 0, int 0, int 0, int 0, int 0, int 0, int 10, int 25, int 50, int 50, int 25, int 10, int 0, int 0, int 0, int 0, int 0, int -1, int 10, int 25, int 25, int 25, int 25, int 10, int -1, int 0, int 0, int 0, int 0, int -3, int 10, int 10, int 10, int 10, int 10, int 10, int -3, int 0, int 0, int 0, int 0, int -5, int -3, int -1, int 0, int 0, int -1, int -3, int -5, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%srev_rank = internal global [9 x int] [ int 0, int 8, int 7, int 6, int 5, int 4, int 3, int 2, int 1 ]		; <[9 x int]*> [#uses=0]
%std_p_tropism = internal global [8 x int] [ int 9999, int 15, int 10, int 7, int 2, int 0, int 0, int 0 ]		; <[8 x int]*> [#uses=0]
%std_own_p_tropism = internal global [8 x int] [ int 9999, int 30, int 10, int 2, int 0, int 0, int 0, int 0 ]		; <[8 x int]*> [#uses=0]
%std_r_tropism = internal global [16 x int] [ int 9999, int 0, int 15, int 5, int 2, int 1, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[16 x int]*> [#uses=0]
%std_n_tropism = internal global [8 x int] [ int 9999, int 14, int 9, int 6, int 1, int 0, int 0, int 0 ]		; <[8 x int]*> [#uses=0]
%std_q_tropism = internal global [8 x int] [ int 9999, int 200, int 50, int 15, int 3, int 2, int 1, int 0 ]		; <[8 x int]*> [#uses=0]
%std_b_tropism = internal global [8 x int] [ int 9999, int 12, int 7, int 5, int 0, int 0, int 0, int 0 ]		; <[8 x int]*> [#uses=0]
%phase = internal global int 0		; <int*> [#uses=1]
%dir.3001 = internal global [4 x int] [ int -13, int -11, int 11, int 13 ]		; <[4 x int]*> [#uses=0]
%dir.3021 = internal global [4 x int] [ int -1, int 1, int 12, int -12 ]		; <[4 x int]*> [#uses=0]
%king_locs = internal global [2 x int] zeroinitializer		; <[2 x int]*> [#uses=0]
%square_d1.3081 = internal global [2 x int] [ int 29, int 113 ]		; <[2 x int]*> [#uses=0]
%wmat = internal global int 0		; <int*> [#uses=0]
%bmat = internal global int 0		; <int*> [#uses=0]
%str = internal global [35 x sbyte] c"Illegal piece detected sq=%i c=%i\0A\00"		; <[35 x sbyte]*> [#uses=0]
%str10 = internal global [81 x sbyte] c"/Volumes/Stuff/src/speccpu2006-091-llvm/benchspec//CPU2006/458.sjeng/src/neval.c\00"		; <[81 x sbyte]*> [#uses=0]
%std_hand_value = internal global [13 x int] [ int 0, int 100, int -100, int 210, int -210, int 0, int 0, int 250, int -250, int 450, int -450, int 230, int -230 ]		; <[13 x int]*> [#uses=0]
%xb_mode = internal global int 0		; <int*> [#uses=0]
%str = internal global [69 x sbyte] c"tellics ptell Hello! I am Sjeng and hope you enjoy playing with me.\0A\00"		; <[69 x sbyte]*> [#uses=0]
%str = internal global [76 x sbyte] c"tellics ptell For help on some commands that I understand, ptell me 'help'\0A\00"		; <[76 x sbyte]*> [#uses=0]
%str12 = internal global [3 x sbyte] c"%s\00"		; <[3 x sbyte]*> [#uses=0]
%my_partner = internal global [256 x sbyte] zeroinitializer		; <[256 x sbyte]*> [#uses=0]
%str13 = internal global [25 x sbyte] c"tellics set f5 bughouse\0A\00"		; <[25 x sbyte]*> [#uses=0]
%str = internal global [16 x sbyte] c"tellics unseek\0A\00"		; <[16 x sbyte]*> [#uses=0]
%str = internal global [20 x sbyte] c"tellics set f5 1=1\0A\00"		; <[20 x sbyte]*> [#uses=0]
%str = internal global [80 x sbyte] c"is...uh...what did you say?\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"		; <[80 x sbyte]*> [#uses=0]
%str = internal global [5 x sbyte] c"help\00"		; <[5 x sbyte]*> [#uses=0]
%str = internal global [147 x sbyte] c"tellics ptell Commands that I understand are : sit, go, fast, slow, abort, flag, +/++/+++/-/--/---{p,n,b,r,q,d,h,trades}, x, dead, formula, help.\0A\00"		; <[147 x sbyte]*> [#uses=0]
%str = internal global [6 x sbyte] c"sorry\00"		; <[6 x sbyte]*> [#uses=0]
%str = internal global [59 x sbyte] c"tellics ptell Sorry, but I'm not playing a bughouse game.\0A\00"		; <[59 x sbyte]*> [#uses=0]
%str = internal global [4 x sbyte] c"sit\00"		; <[4 x sbyte]*> [#uses=0]
%str = internal global [56 x sbyte] c"tellics ptell Ok, I sit next move. Tell me when to go.\0A\00"		; <[56 x sbyte]*> [#uses=0]
%must_sit.b = internal global bool false		; <bool*> [#uses=0]
%str114 = internal global [3 x sbyte] c"go\00"		; <[3 x sbyte]*> [#uses=0]
%str2 = internal global [5 x sbyte] c"move\00"		; <[5 x sbyte]*> [#uses=0]
%str = internal global [31 x sbyte] c"tellics ptell Ok, I'm moving.\0A\00"		; <[31 x sbyte]*> [#uses=0]
%str3 = internal global [5 x sbyte] c"fast\00"		; <[5 x sbyte]*> [#uses=0]
%str4 = internal global [5 x sbyte] c"time\00"		; <[5 x sbyte]*> [#uses=0]
%str15 = internal global [35 x sbyte] c"tellics ptell Ok, I'm going FAST!\0A\00"		; <[35 x sbyte]*> [#uses=0]
%go_fast.b = internal global bool false		; <bool*> [#uses=0]
%str5 = internal global [5 x sbyte] c"slow\00"		; <[5 x sbyte]*> [#uses=0]
%str16 = internal global [36 x sbyte] c"tellics ptell Ok, moving normally.\0A\00"		; <[36 x sbyte]*> [#uses=0]
%str6 = internal global [6 x sbyte] c"abort\00"		; <[6 x sbyte]*> [#uses=0]
%str7 = internal global [35 x sbyte] c"tellics ptell Requesting abort...\0A\00"		; <[35 x sbyte]*> [#uses=0]
%str17 = internal global [15 x sbyte] c"tellics abort\0A\00"		; <[15 x sbyte]*> [#uses=0]
%str8 = internal global [5 x sbyte] c"flag\00"		; <[5 x sbyte]*> [#uses=0]
%str = internal global [27 x sbyte] c"tellics ptell Flagging...\0A\00"		; <[27 x sbyte]*> [#uses=0]
%str = internal global [14 x sbyte] c"tellics flag\0A\00"		; <[14 x sbyte]*> [#uses=0]
%str18 = internal global [2 x sbyte] c"+\00"		; <[2 x sbyte]*> [#uses=0]
%str9 = internal global [6 x sbyte] c"trade\00"		; <[6 x sbyte]*> [#uses=0]
%str10 = internal global [35 x sbyte] c"tellics ptell Ok, trading is GOOD\0A\00"		; <[35 x sbyte]*> [#uses=0]
%str11 = internal global [4 x sbyte] c"+++\00"		; <[4 x sbyte]*> [#uses=0]
%str12 = internal global [6 x sbyte] c"mates\00"		; <[6 x sbyte]*> [#uses=0]
%str13 = internal global [3 x sbyte] c"++\00"		; <[3 x sbyte]*> [#uses=0]
%str = internal global [49 x sbyte] c"is VERY good (ptell me 'x' to play normal again)\00"		; <[49 x sbyte]*> [#uses=0]
%str = internal global [44 x sbyte] c"is good (ptell me 'x' to play normal again)\00"		; <[44 x sbyte]*> [#uses=0]
%str19 = internal global [29 x sbyte] c"tellics ptell Ok, Knight %s\0A\00"		; <[29 x sbyte]*> [#uses=0]
%str14 = internal global [29 x sbyte] c"tellics ptell Ok, Bishop %s\0A\00"		; <[29 x sbyte]*> [#uses=0]
%str15 = internal global [27 x sbyte] c"tellics ptell Ok, Rook %s\0A\00"		; <[27 x sbyte]*> [#uses=0]
%str = internal global [28 x sbyte] c"tellics ptell Ok, Queen %s\0A\00"		; <[28 x sbyte]*> [#uses=0]
%str16 = internal global [27 x sbyte] c"tellics ptell Ok, Pawn %s\0A\00"		; <[27 x sbyte]*> [#uses=0]
%str17 = internal global [31 x sbyte] c"tellics ptell Ok, Diagonal %s\0A\00"		; <[31 x sbyte]*> [#uses=0]
%str18 = internal global [28 x sbyte] c"tellics ptell Ok, Heavy %s\0A\00"		; <[28 x sbyte]*> [#uses=0]
%str20 = internal global [34 x sbyte] c"tellics ptell Ok, trading is BAD\0A\00"		; <[34 x sbyte]*> [#uses=0]
%str20 = internal global [4 x sbyte] c"---\00"		; <[4 x sbyte]*> [#uses=0]
%str = internal global [53 x sbyte] c"mates you (ptell me 'x' when it no longer mates you)\00"		; <[53 x sbyte]*> [#uses=0]
%str21 = internal global [3 x sbyte] c"--\00"		; <[3 x sbyte]*> [#uses=0]
%str = internal global [52 x sbyte] c"is VERY bad (ptell me 'x' when it is no longer bad)\00"		; <[52 x sbyte]*> [#uses=0]
%str21 = internal global [47 x sbyte] c"is bad (ptell me 'x' when it is no longer bad)\00"		; <[47 x sbyte]*> [#uses=0]
%str23 = internal global [16 x sbyte] c"mate me anymore\00"		; <[16 x sbyte]*> [#uses=0]
%str24 = internal global [6 x sbyte] c"never\00"		; <[6 x sbyte]*> [#uses=0]
%str25 = internal global [5 x sbyte] c"mind\00"		; <[5 x sbyte]*> [#uses=0]
%str22 = internal global [9 x sbyte] c"ptell me\00"		; <[9 x sbyte]*> [#uses=0]
%str = internal global [55 x sbyte] c"tellics ptell Ok, reverting to STANDARD piece values!\0A\00"		; <[55 x sbyte]*> [#uses=0]
%partnerdead.b = internal global bool false		; <bool*> [#uses=0]
%piecedead.b = internal global bool false		; <bool*> [#uses=0]
%str = internal global [26 x sbyte] c"i'll have to sit...(dead)\00"		; <[26 x sbyte]*> [#uses=0]
%str27 = internal global [5 x sbyte] c"dead\00"		; <[5 x sbyte]*> [#uses=0]
%str28 = internal global [27 x sbyte] c"i'll have to sit...(piece)\00"		; <[27 x sbyte]*> [#uses=0]
%str29 = internal global [3 x sbyte] c"ok\00"		; <[3 x sbyte]*> [#uses=0]
%str30 = internal global [3 x sbyte] c"hi\00"		; <[3 x sbyte]*> [#uses=0]
%str31 = internal global [6 x sbyte] c"hello\00"		; <[6 x sbyte]*> [#uses=0]
%str32 = internal global [26 x sbyte] c"tellics ptell Greetings.\0A\00"		; <[26 x sbyte]*> [#uses=0]
%str = internal global [8 x sbyte] c"formula\00"		; <[8 x sbyte]*> [#uses=0]
%str = internal global [87 x sbyte] c"tellics ptell Setting formula, if you are still interrupted, complain to my operator.\0A\00"		; <[87 x sbyte]*> [#uses=0]
%str33 = internal global [59 x sbyte] c"tellics ptell Sorry, but I don't understand that command.\0A\00"		; <[59 x sbyte]*> [#uses=0]
%pawnmated.3298 = internal global int 0		; <int*> [#uses=0]
%knightmated.3299 = internal global int 0		; <int*> [#uses=0]
%bishopmated.3300 = internal global int 0		; <int*> [#uses=0]
%rookmated.3301 = internal global int 0		; <int*> [#uses=0]
%queenmated.3302 = internal global int 0		; <int*> [#uses=0]
%str = internal global [41 x sbyte] c"tellics ptell p doesn't mate me anymore\0A\00"		; <[41 x sbyte]*> [#uses=0]
%str34 = internal global [41 x sbyte] c"tellics ptell n doesn't mate me anymore\0A\00"		; <[41 x sbyte]*> [#uses=0]
%str35 = internal global [41 x sbyte] c"tellics ptell b doesn't mate me anymore\0A\00"		; <[41 x sbyte]*> [#uses=0]
%str36 = internal global [41 x sbyte] c"tellics ptell r doesn't mate me anymore\0A\00"		; <[41 x sbyte]*> [#uses=0]
%str37 = internal global [41 x sbyte] c"tellics ptell q doesn't mate me anymore\0A\00"		; <[41 x sbyte]*> [#uses=0]
%str38 = internal global [20 x sbyte] c"tellics ptell ---p\0A\00"		; <[20 x sbyte]*> [#uses=0]
%str39 = internal global [20 x sbyte] c"tellics ptell ---n\0A\00"		; <[20 x sbyte]*> [#uses=0]
%str40 = internal global [20 x sbyte] c"tellics ptell ---b\0A\00"		; <[20 x sbyte]*> [#uses=0]
%str41 = internal global [20 x sbyte] c"tellics ptell ---r\0A\00"		; <[20 x sbyte]*> [#uses=0]
%str42 = internal global [20 x sbyte] c"tellics ptell ---q\0A\00"		; <[20 x sbyte]*> [#uses=0]
%str23 = internal global [17 x sbyte] c"tellics ptell x\0A\00"		; <[17 x sbyte]*> [#uses=0]
%str = internal global [18 x sbyte] c"tellics ptell go\0A\00"		; <[18 x sbyte]*> [#uses=0]
%bufftop = internal global int 0		; <int*> [#uses=2]
%membuff = internal global ubyte* null		; <ubyte**> [#uses=3]
%maxply = internal global int 0		; <int*> [#uses=1]
%forwards = internal global int 0		; <int*> [#uses=1]
%nodecount = internal global int 0		; <int*> [#uses=1]
%frees = internal global int 0		; <int*> [#uses=0]
%PBSize.b = internal global bool false		; <bool*> [#uses=1]
%alllosers.b = internal global bool false		; <bool*> [#uses=1]
%rootlosers = internal global [300 x int] zeroinitializer		; <[300 x int]*> [#uses=1]
%pn_move = internal global %struct.move_s zeroinitializer		; <%struct.move_s*> [#uses=7]
%iters = internal global int 0		; <int*> [#uses=1]
%kibitzed.b = internal global bool false		; <bool*> [#uses=0]
%str24 = internal global [28 x sbyte] c"tellics kibitz Forced win!\0A\00"		; <[28 x sbyte]*> [#uses=0]
%str25 = internal global [34 x sbyte] c"tellics kibitz Forced win! (alt)\0A\00"		; <[34 x sbyte]*> [#uses=0]
%pn_time = internal global int 0		; <int*> [#uses=1]
%post = internal global uint 0		; <uint*> [#uses=0]
%str = internal global [94 x sbyte] c"tellics whisper proof %d, disproof %d, %d losers, highest depth %d, primary %d, secondary %d\0A\00"		; <[94 x sbyte]*> [#uses=0]
%str26 = internal global [30 x sbyte] c"tellics whisper Forced reply\0A\00"		; <[30 x sbyte]*> [#uses=0]
%str27 = internal global [60 x sbyte] c"P: %d D: %d N: %d S: %d Mem: %2.2fM Iters: %d MaxDepth: %d\0A\00"		; <[60 x sbyte]*> [#uses=0]
%str = internal global [90 x sbyte] c"tellics whisper proof %d, disproof %d, %d nodes, %d forwards, %d iters, highest depth %d\0A\00"		; <[90 x sbyte]*> [#uses=0]
%str = internal global [11 x sbyte] c"Time : %f\0A\00"		; <[11 x sbyte]*> [#uses=0]
%str28 = internal global [23 x sbyte] c"This position is WON.\0A\00"		; <[23 x sbyte]*> [#uses=0]
%str29 = internal global [5 x sbyte] c"PV: \00"		; <[5 x sbyte]*> [#uses=0]
%str30 = internal global [4 x sbyte] c"%s \00"		; <[4 x sbyte]*> [#uses=0]
%str31 = internal global [2 x sbyte] c" \00"		; <[2 x sbyte]*> [#uses=0]
%str32 = internal global [41 x sbyte] c"\0Atellics kibitz Forced win in %d moves.\0A\00"		; <[41 x sbyte]*> [#uses=0]
%str33 = internal global [20 x sbyte] c"\0A1-0 {White mates}\0A\00"		; <[20 x sbyte]*> [#uses=0]
%result = internal global int 0		; <int*> [#uses=4]
%str1 = internal global [20 x sbyte] c"\0A0-1 {Black mates}\0A\00"		; <[20 x sbyte]*> [#uses=0]
%str35 = internal global [24 x sbyte] c"This position is LOST.\0A\00"		; <[24 x sbyte]*> [#uses=0]
%str36 = internal global [27 x sbyte] c"This position is UNKNOWN.\0A\00"		; <[27 x sbyte]*> [#uses=0]
%str37 = internal global [47 x sbyte] c"P: %d D: %d N: %d S: %d Mem: %2.2fM Iters: %d\0A\00"		; <[47 x sbyte]*> [#uses=0]
%s_threat.b = internal global bool false		; <bool*> [#uses=0]
%TTSize.b = internal global bool false		; <bool*> [#uses=3]
%cfg_razordrop.b = internal global bool false		; <bool*> [#uses=0]
%cfg_futprune.b = internal global bool false		; <bool*> [#uses=0]
%cfg_onerep.b = internal global bool false		; <bool*> [#uses=0]
%setcode = internal global [30 x sbyte] zeroinitializer		; <[30 x sbyte]*> [#uses=0]
%str38 = internal global [3 x sbyte] c"%u\00"		; <[3 x sbyte]*> [#uses=0]
%searching_pv.b = internal global bool false		; <bool*> [#uses=0]
%pv = internal global [300 x [300 x %struct.move_s]] zeroinitializer		; <[300 x [300 x %struct.move_s]]*> [#uses=0]
%i_depth = internal global int 0		; <int*> [#uses=0]
%history_h = internal global [144 x [144 x uint]] zeroinitializer		; <[144 x [144 x uint]]*> [#uses=0]
%killer1 = internal global [300 x %struct.move_s] zeroinitializer		; <[300 x %struct.move_s]*> [#uses=0]
%killer2 = internal global [300 x %struct.move_s] zeroinitializer		; <[300 x %struct.move_s]*> [#uses=0]
%killer3 = internal global [300 x %struct.move_s] zeroinitializer		; <[300 x %struct.move_s]*> [#uses=0]
%rootnodecount = internal global [512 x uint] zeroinitializer		; <[512 x uint]*> [#uses=0]
%raw_nodes = internal global int 0		; <int*> [#uses=0]
%pv_length = internal global [300 x int] zeroinitializer		; <[300 x int]*> [#uses=0]
%time_exit.b = internal global bool false		; <bool*> [#uses=0]
%time_for_move = internal global int 0		; <int*> [#uses=3]
%failed = internal global int 0		; <int*> [#uses=0]
%extendedtime.b = internal global bool false		; <bool*> [#uses=1]
%time_left = internal global int 0		; <int*> [#uses=0]
%str39 = internal global [38 x sbyte] c"Extended from %d to %d, time left %d\0A\00"		; <[38 x sbyte]*> [#uses=0]
%checks = internal global [300 x uint] zeroinitializer		; <[300 x uint]*> [#uses=0]
%singular = internal global [300 x uint] zeroinitializer		; <[300 x uint]*> [#uses=0]
%recaps = internal global [300 x uint] zeroinitializer		; <[300 x uint]*> [#uses=0]
%ext_onerep = internal global uint 0		; <uint*> [#uses=1]
%FULL = internal global uint 0		; <uint*> [#uses=1]
%PVS = internal global uint 0		; <uint*> [#uses=1]
%PVSF = internal global uint 0		; <uint*> [#uses=1]
%killer_scores = internal global [300 x int] zeroinitializer		; <[300 x int]*> [#uses=0]
%killer_scores2 = internal global [300 x int] zeroinitializer		; <[300 x int]*> [#uses=0]
%killer_scores3 = internal global [300 x int] zeroinitializer		; <[300 x int]*> [#uses=0]
%time_failure.b = internal global bool false		; <bool*> [#uses=0]
%cur_score = internal global int 0		; <int*> [#uses=0]
%legals = internal global int 0		; <int*> [#uses=3]
%movetotal = internal global int 0		; <int*> [#uses=0]
%searching_move = internal global [20 x sbyte] zeroinitializer		; <[20 x sbyte]*> [#uses=0]
%is_pondering.b = internal global bool false		; <bool*> [#uses=6]
%true_i_depth = internal global sbyte 0		; <sbyte*> [#uses=1]
%is_analyzing.b = internal global bool false		; <bool*> [#uses=0]
%inc = internal global int 0		; <int*> [#uses=1]
%time_cushion = internal global int 0		; <int*> [#uses=2]
%str40 = internal global [16 x sbyte] c"Opening phase.\0A\00"		; <[16 x sbyte]*> [#uses=1]
%str = internal global [19 x sbyte] c"Middlegame phase.\0A\00"		; <[19 x sbyte]*> [#uses=1]
%str1 = internal global [16 x sbyte] c"Endgame phase.\0A\00"		; <[16 x sbyte]*> [#uses=1]
%str43 = internal global [20 x sbyte] c"Time for move : %d\0A\00"		; <[20 x sbyte]*> [#uses=1]
%postpv = internal global [256 x sbyte] zeroinitializer		; <[256 x sbyte]*> [#uses=0]
%str44 = internal global [49 x sbyte] c"tellics whisper %d restart(s), ended up with %s\0A\00"		; <[49 x sbyte]*> [#uses=0]
%moves_to_tc = internal global int 0		; <int*> [#uses=0]
%str45 = internal global [27 x sbyte] c"tellics kibitz Mate in %d\0A\00"		; <[27 x sbyte]*> [#uses=0]
%str46 = internal global [52 x sbyte] c"tellics ptell Mate in %d, give him no more pieces.\0A\00"		; <[52 x sbyte]*> [#uses=0]
%tradefreely.b = internal global bool false		; <bool*> [#uses=0]
%str = internal global [37 x sbyte] c"tellics ptell You can trade freely.\0A\00"		; <[37 x sbyte]*> [#uses=0]
%str47 = internal global [25 x sbyte] c"tellics ptell ---trades\0A\00"		; <[25 x sbyte]*> [#uses=0]
%str2 = internal global [49 x sbyte] c"tellics kibitz Both players dead...resigning...\0A\00"		; <[49 x sbyte]*> [#uses=0]
%str3 = internal global [16 x sbyte] c"tellics resign\0A\00"		; <[16 x sbyte]*> [#uses=0]
%str48 = internal global [81 x sbyte] c"tellics ptell I am forcedly mated (dead). Tell me 'go' to start moving into it.\0A\00"		; <[81 x sbyte]*> [#uses=0]
%str = internal global [62 x sbyte] c"tellics ptell I'll have to sit...(lose piece that mates you)\0A\00"		; <[62 x sbyte]*> [#uses=0]
%see_num_attackers = internal global [2 x int] zeroinitializer		; <[2 x int]*> [#uses=0]
%see_attackers = internal global [2 x [16 x %struct.see_data]] zeroinitializer		; <[2 x [16 x %struct.see_data]]*> [#uses=0]
%scentral = internal global [144 x int] [ int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int -20, int -10, int -10, int -10, int -10, int -10, int -10, int -20, int 0, int 0, int 0, int 0, int -10, int 0, int 3, int 5, int 5, int 3, int 0, int -10, int 0, int 0, int 0, int 0, int -10, int 2, int 15, int 15, int 15, int 15, int 2, int -10, int 0, int 0, int 0, int 0, int -10, int 7, int 15, int 25, int 25, int 15, int 7, int -10, int 0, int 0, int 0, int 0, int -10, int 7, int 15, int 25, int 25, int 15, int 7, int -10, int 0, int 0, int 0, int 0, int -10, int 2, int 15, int 15, int 15, int 15, int 2, int -10, int 0, int 0, int 0, int 0, int -10, int 0, int 3, int 5, int 5, int 3, int 0, int -10, int 0, int 0, int 0, int 0, int -20, int -10, int -10, int -10, int -10, int -10, int -10, int -20, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0, int 0 ]		; <[144 x int]*> [#uses=0]
%str51 = internal global [81 x sbyte] c"/Volumes/Stuff/src/speccpu2006-091-llvm/benchspec//CPU2006/458.sjeng/src/seval.c\00"		; <[81 x sbyte]*> [#uses=0]
%divider = internal global [50 x sbyte] c"-------------------------------------------------\00"		; <[50 x sbyte]*> [#uses=0]
%min_per_game = internal global int 0		; <int*> [#uses=0]
%opp_rating = internal global int 0		; <int*> [#uses=0]
%my_rating = internal global int 0		; <int*> [#uses=0]
%str53 = internal global [15 x sbyte] c"SPEC Workload\0A\00"		; <[15 x sbyte]*> [#uses=0]
%opening_history = internal global [256 x sbyte] zeroinitializer		; <[256 x sbyte]*> [#uses=0]
%str60 = internal global [81 x sbyte] c"Material score: %d   Eval : %d  MaxPosDiff: %d  White hand: %d  Black hand : %d\0A\00"		; <[81 x sbyte]*> [#uses=0]
%str61 = internal global [26 x sbyte] c"Hash : %X  HoldHash : %X\0A\00"		; <[26 x sbyte]*> [#uses=0]
%str62 = internal global [9 x sbyte] c"move %s\0A\00"		; <[9 x sbyte]*> [#uses=0]
%str63 = internal global [5 x sbyte] c"\0A%s\0A\00"		; <[5 x sbyte]*> [#uses=0]
%str64 = internal global [19 x sbyte] c"0-1 {Black Mates}\0A\00"		; <[19 x sbyte]*> [#uses=0]
%str1 = internal global [19 x sbyte] c"1-0 {White Mates}\0A\00"		; <[19 x sbyte]*> [#uses=0]
%str65 = internal global [27 x sbyte] c"1/2-1/2 {Fifty move rule}\0A\00"		; <[27 x sbyte]*> [#uses=0]
%str2 = internal global [29 x sbyte] c"1/2-1/2 {3 fold repetition}\0A\00"		; <[29 x sbyte]*> [#uses=0]
%str66 = internal global [16 x sbyte] c"1/2-1/2 {Draw}\0A\00"		; <[16 x sbyte]*> [#uses=0]
%str68 = internal global [8 x sbyte] c"Sjeng: \00"		; <[8 x sbyte]*> [#uses=0]
%str69 = internal global [18 x sbyte] c"Illegal move: %s\0A\00"		; <[18 x sbyte]*> [#uses=0]
%str3 = internal global [9 x sbyte] c"setboard\00"		; <[9 x sbyte]*> [#uses=0]
%str470 = internal global [5 x sbyte] c"quit\00"		; <[5 x sbyte]*> [#uses=0]
%str571 = internal global [5 x sbyte] c"exit\00"		; <[5 x sbyte]*> [#uses=0]
%str6 = internal global [8 x sbyte] c"diagram\00"		; <[8 x sbyte]*> [#uses=0]
%str7 = internal global [2 x sbyte] c"d\00"		; <[2 x sbyte]*> [#uses=0]
%str72 = internal global [6 x sbyte] c"perft\00"		; <[6 x sbyte]*> [#uses=0]
%str73 = internal global [3 x sbyte] c"%d\00"		; <[3 x sbyte]*> [#uses=0]
%str74 = internal global [28 x sbyte] c"Raw nodes for depth %d: %i\0A\00"		; <[28 x sbyte]*> [#uses=0]
%str = internal global [13 x sbyte] c"Time : %.2f\0A\00"		; <[13 x sbyte]*> [#uses=0]
%str75 = internal global [4 x sbyte] c"new\00"		; <[4 x sbyte]*> [#uses=0]
%str = internal global [40 x sbyte] c"tellics set 1 Sjeng SPEC 1.0 (SPEC/%s)\0A\00"		; <[40 x sbyte]*> [#uses=0]
%str = internal global [7 x sbyte] c"xboard\00"		; <[7 x sbyte]*> [#uses=0]
%str8 = internal global [6 x sbyte] c"nodes\00"		; <[6 x sbyte]*> [#uses=0]
%str77 = internal global [38 x sbyte] c"Number of nodes: %i (%0.2f%% qnodes)\0A\00"		; <[38 x sbyte]*> [#uses=0]
%str9 = internal global [5 x sbyte] c"post\00"		; <[5 x sbyte]*> [#uses=0]
%str10 = internal global [7 x sbyte] c"nopost\00"		; <[7 x sbyte]*> [#uses=0]
%str11 = internal global [7 x sbyte] c"random\00"		; <[7 x sbyte]*> [#uses=0]
%str12 = internal global [5 x sbyte] c"hard\00"		; <[5 x sbyte]*> [#uses=0]
%str13 = internal global [5 x sbyte] c"easy\00"		; <[5 x sbyte]*> [#uses=0]
%str14 = internal global [2 x sbyte] c"?\00"		; <[2 x sbyte]*> [#uses=0]
%str15 = internal global [6 x sbyte] c"white\00"		; <[6 x sbyte]*> [#uses=0]
%str16 = internal global [6 x sbyte] c"black\00"		; <[6 x sbyte]*> [#uses=0]
%str17 = internal global [6 x sbyte] c"force\00"		; <[6 x sbyte]*> [#uses=0]
%str18 = internal global [5 x sbyte] c"eval\00"		; <[5 x sbyte]*> [#uses=0]
%str = internal global [10 x sbyte] c"Eval: %d\0A\00"		; <[10 x sbyte]*> [#uses=0]
%str2178 = internal global [3 x sbyte] c"%i\00"		; <[3 x sbyte]*> [#uses=0]
%str22 = internal global [5 x sbyte] c"otim\00"		; <[5 x sbyte]*> [#uses=0]
%opp_time = internal global int 0		; <int*> [#uses=0]
%str23 = internal global [6 x sbyte] c"level\00"		; <[6 x sbyte]*> [#uses=0]
%str = internal global [12 x sbyte] c"%i %i:%i %i\00"		; <[12 x sbyte]*> [#uses=0]
%sec_per_game = internal global int 0		; <int*> [#uses=0]
%str24 = internal global [9 x sbyte] c"%i %i %i\00"		; <[9 x sbyte]*> [#uses=0]
%str25 = internal global [7 x sbyte] c"rating\00"		; <[7 x sbyte]*> [#uses=0]
%str26 = internal global [6 x sbyte] c"%i %i\00"		; <[6 x sbyte]*> [#uses=0]
%str27 = internal global [8 x sbyte] c"holding\00"		; <[8 x sbyte]*> [#uses=0]
%str28 = internal global [8 x sbyte] c"variant\00"		; <[8 x sbyte]*> [#uses=0]
%str29 = internal global [7 x sbyte] c"normal\00"		; <[7 x sbyte]*> [#uses=0]
%str79 = internal global [11 x sbyte] c"crazyhouse\00"		; <[11 x sbyte]*> [#uses=0]
%str30 = internal global [9 x sbyte] c"bughouse\00"		; <[9 x sbyte]*> [#uses=0]
%str31 = internal global [8 x sbyte] c"suicide\00"		; <[8 x sbyte]*> [#uses=0]
%str32 = internal global [9 x sbyte] c"giveaway\00"		; <[9 x sbyte]*> [#uses=0]
%str33 = internal global [7 x sbyte] c"losers\00"		; <[7 x sbyte]*> [#uses=0]
%str34 = internal global [8 x sbyte] c"analyze\00"		; <[8 x sbyte]*> [#uses=0]
%str35 = internal global [5 x sbyte] c"undo\00"		; <[5 x sbyte]*> [#uses=0]
%str36 = internal global [18 x sbyte] c"Move number : %d\0A\00"		; <[18 x sbyte]*> [#uses=0]
%str37 = internal global [7 x sbyte] c"remove\00"		; <[7 x sbyte]*> [#uses=0]
%str38 = internal global [5 x sbyte] c"edit\00"		; <[5 x sbyte]*> [#uses=0]
%str41 = internal global [2 x sbyte] c"#\00"		; <[2 x sbyte]*> [#uses=0]
%str42 = internal global [8 x sbyte] c"partner\00"		; <[8 x sbyte]*> [#uses=0]
%str43 = internal global [9 x sbyte] c"$partner\00"		; <[9 x sbyte]*> [#uses=0]
%str44 = internal global [6 x sbyte] c"ptell\00"		; <[6 x sbyte]*> [#uses=0]
%str45 = internal global [5 x sbyte] c"test\00"		; <[5 x sbyte]*> [#uses=0]
%str46 = internal global [3 x sbyte] c"st\00"		; <[3 x sbyte]*> [#uses=0]
%str47 = internal global [7 x sbyte] c"result\00"		; <[7 x sbyte]*> [#uses=0]
%str48 = internal global [6 x sbyte] c"prove\00"		; <[6 x sbyte]*> [#uses=0]
%str49 = internal global [26 x sbyte] c"\0AMax time to search (s): \00"		; <[26 x sbyte]*> [#uses=0]
%str50 = internal global [5 x sbyte] c"ping\00"		; <[5 x sbyte]*> [#uses=0]
%str51 = internal global [9 x sbyte] c"pong %d\0A\00"		; <[9 x sbyte]*> [#uses=0]
%str52 = internal global [6 x sbyte] c"fritz\00"		; <[6 x sbyte]*> [#uses=0]
%str53 = internal global [6 x sbyte] c"reset\00"		; <[6 x sbyte]*> [#uses=0]
%str54 = internal global [3 x sbyte] c"sd\00"		; <[3 x sbyte]*> [#uses=0]
%str55 = internal global [26 x sbyte] c"New max depth set to: %d\0A\00"		; <[26 x sbyte]*> [#uses=0]
%str56 = internal global [5 x sbyte] c"auto\00"		; <[5 x sbyte]*> [#uses=0]
%str57 = internal global [9 x sbyte] c"protover\00"		; <[9 x sbyte]*> [#uses=0]
%str = internal global [63 x sbyte] c"feature ping=0 setboard=1 playother=0 san=0 usermove=0 time=1\0A\00"		; <[63 x sbyte]*> [#uses=0]
%str80 = internal global [53 x sbyte] c"feature draw=0 sigint=0 sigterm=0 reuse=1 analyze=0\0A\00"		; <[53 x sbyte]*> [#uses=0]
%str = internal global [33 x sbyte] c"feature myname=\22Sjeng SPEC 1.0\22\0A\00"		; <[33 x sbyte]*> [#uses=0]
%str = internal global [71 x sbyte] c"feature variants=\22normal,bughouse,crazyhouse,suicide,giveaway,losers\22\0A\00"		; <[71 x sbyte]*> [#uses=0]
%str = internal global [46 x sbyte] c"feature colors=1 ics=0 name=0 pause=0 done=1\0A\00"		; <[46 x sbyte]*> [#uses=0]
%str58 = internal global [9 x sbyte] c"accepted\00"		; <[9 x sbyte]*> [#uses=0]
%str59 = internal global [9 x sbyte] c"rejected\00"		; <[9 x sbyte]*> [#uses=0]
%str = internal global [65 x sbyte] c"Interface does not support a required feature...expect trouble.\0A\00"		; <[65 x sbyte]*> [#uses=0]
%str61 = internal global [6 x sbyte] c"\0A%s\0A\0A\00"		; <[6 x sbyte]*> [#uses=0]
%str81 = internal global [41 x sbyte] c"diagram/d:       toggle diagram display\0A\00"		; <[41 x sbyte]*> [#uses=0]
%str82 = internal global [34 x sbyte] c"exit/quit:       terminate Sjeng\0A\00"		; <[34 x sbyte]*> [#uses=0]
%str62 = internal global [51 x sbyte] c"go:              make Sjeng play the side to move\0A\00"		; <[51 x sbyte]*> [#uses=0]
%str83 = internal global [35 x sbyte] c"new:             start a new game\0A\00"		; <[35 x sbyte]*> [#uses=0]
%str84 = internal global [55 x sbyte] c"level <x>:       the xboard style command to set time\0A\00"		; <[55 x sbyte]*> [#uses=0]
%str85 = internal global [49 x sbyte] c"  <x> should be in the form: <a> <b> <c> where:\0A\00"		; <[49 x sbyte]*> [#uses=0]
%str63 = internal global [49 x sbyte] c"  a -> moves to TC (0 if using an ICS style TC)\0A\00"		; <[49 x sbyte]*> [#uses=0]
%str86 = internal global [25 x sbyte] c"  b -> minutes per game\0A\00"		; <[25 x sbyte]*> [#uses=0]
%str64 = internal global [29 x sbyte] c"  c -> increment in seconds\0A\00"		; <[29 x sbyte]*> [#uses=0]
%str65 = internal global [55 x sbyte] c"nodes:           outputs the number of nodes searched\0A\00"		; <[55 x sbyte]*> [#uses=0]
%str87 = internal global [47 x sbyte] c"perft <x>:       compute raw nodes to depth x\0A\00"		; <[47 x sbyte]*> [#uses=0]
%str = internal global [42 x sbyte] c"post:            toggles thinking output\0A\00"		; <[42 x sbyte]*> [#uses=0]
%str = internal global [45 x sbyte] c"xboard:          put Sjeng into xboard mode\0A\00"		; <[45 x sbyte]*> [#uses=0]
%str = internal global [39 x sbyte] c"test:            run an EPD testsuite\0A\00"		; <[39 x sbyte]*> [#uses=0]
%str88 = internal global [52 x sbyte] c"speed:           test movegen and evaluation speed\0A\00"		; <[52 x sbyte]*> [#uses=0]
%str89 = internal global [59 x sbyte] c"proof:           try to prove or disprove the current pos\0A\00"		; <[59 x sbyte]*> [#uses=0]
%str90 = internal global [44 x sbyte] c"sd <x>:          limit thinking to depth x\0A\00"		; <[44 x sbyte]*> [#uses=0]
%str66 = internal global [51 x sbyte] c"st <x>:          limit thinking to x centiseconds\0A\00"		; <[51 x sbyte]*> [#uses=0]
%str67 = internal global [54 x sbyte] c"setboard <FEN>:  set board to a specified FEN string\0A\00"		; <[54 x sbyte]*> [#uses=0]
%str68 = internal global [38 x sbyte] c"undo:            back up a half move\0A\00"		; <[38 x sbyte]*> [#uses=0]
%str69 = internal global [38 x sbyte] c"remove:          back up a full move\0A\00"		; <[38 x sbyte]*> [#uses=0]
%str70 = internal global [42 x sbyte] c"force:           disable computer moving\0A\00"		; <[42 x sbyte]*> [#uses=0]
%str71 = internal global [44 x sbyte] c"auto:            computer plays both sides\0A\00"		; <[44 x sbyte]*> [#uses=0]
%DP_TTable = internal global %struct.TType* null		; <%struct.TType**> [#uses=1]
%AS_TTable = internal global %struct.TType* null		; <%struct.TType**> [#uses=1]
%QS_TTable = internal global %struct.QTType* null		; <%struct.QTType**> [#uses=1]
%str93 = internal global [38 x sbyte] c"Out of memory allocating hashtables.\0A\00"		; <[38 x sbyte]*> [#uses=0]
%type_to_char.3058 = internal global [14 x int] [ int 70, int 80, int 80, int 78, int 78, int 75, int 75, int 82, int 82, int 81, int 81, int 66, int 66, int 69 ]		; <[14 x int]*> [#uses=0]
%str94 = internal global [8 x sbyte] c"%c@%c%d\00"		; <[8 x sbyte]*> [#uses=0]
%str95 = internal global [5 x sbyte] c"%c%d\00"		; <[5 x sbyte]*> [#uses=0]
%str1 = internal global [8 x sbyte] c"%c%d=%c\00"		; <[8 x sbyte]*> [#uses=0]
%str2 = internal global [8 x sbyte] c"%cx%c%d\00"		; <[8 x sbyte]*> [#uses=0]
%str96 = internal global [11 x sbyte] c"%cx%c%d=%c\00"		; <[11 x sbyte]*> [#uses=0]
%str97 = internal global [4 x sbyte] c"O-O\00"		; <[4 x sbyte]*> [#uses=0]
%str98 = internal global [6 x sbyte] c"O-O-O\00"		; <[6 x sbyte]*> [#uses=0]
%str99 = internal global [9 x sbyte] c"%c%c%c%d\00"		; <[9 x sbyte]*> [#uses=0]
%str3100 = internal global [9 x sbyte] c"%c%d%c%d\00"		; <[9 x sbyte]*> [#uses=0]
%str101 = internal global [10 x sbyte] c"%c%cx%c%d\00"		; <[10 x sbyte]*> [#uses=0]
%str4 = internal global [10 x sbyte] c"%c%dx%c%d\00"		; <[10 x sbyte]*> [#uses=0]
%str102 = internal global [7 x sbyte] c"%c%c%d\00"		; <[7 x sbyte]*> [#uses=0]
%str5103 = internal global [5 x sbyte] c"illg\00"		; <[5 x sbyte]*> [#uses=0]
%type_to_char.3190 = internal global [14 x int] [ int 70, int 80, int 112, int 78, int 110, int 75, int 107, int 82, int 114, int 81, int 113, int 66, int 98, int 69 ]		; <[14 x int]*> [#uses=0]
%str7 = internal global [10 x sbyte] c"%c%d%c%dn\00"		; <[10 x sbyte]*> [#uses=0]
%str8 = internal global [10 x sbyte] c"%c%d%c%dr\00"		; <[10 x sbyte]*> [#uses=0]
%str9 = internal global [10 x sbyte] c"%c%d%c%db\00"		; <[10 x sbyte]*> [#uses=0]
%str10 = internal global [10 x sbyte] c"%c%d%c%dk\00"		; <[10 x sbyte]*> [#uses=0]
%str11 = internal global [10 x sbyte] c"%c%d%c%dq\00"		; <[10 x sbyte]*> [#uses=0]
%C.88.3251 = internal global [14 x sbyte*] [ sbyte* getelementptr ([3 x sbyte]* %str105, int 0, int 0), sbyte* getelementptr ([3 x sbyte]* %str12106, int 0, int 0), sbyte* getelementptr ([3 x sbyte]* %str13107, int 0, int 0), sbyte* getelementptr ([3 x sbyte]* %str14, int 0, int 0), sbyte* getelementptr ([3 x sbyte]* %str15, int 0, int 0), sbyte* getelementptr ([3 x sbyte]* %str16, int 0, int 0), sbyte* getelementptr ([3 x sbyte]* %str17, int 0, int 0), sbyte* getelementptr ([3 x sbyte]* %str18, int 0, int 0), sbyte* getelementptr ([3 x sbyte]* %str19108, int 0, int 0), sbyte* getelementptr ([3 x sbyte]* %str20, int 0, int 0), sbyte* getelementptr ([3 x sbyte]* %str21109, int 0, int 0), sbyte* getelementptr ([3 x sbyte]* %str22, int 0, int 0), sbyte* getelementptr ([3 x sbyte]* %str23, int 0, int 0), sbyte* getelementptr ([3 x sbyte]* %str24, int 0, int 0) ]		; <[14 x sbyte*]*> [#uses=0]
%str105 = internal global [3 x sbyte] c"!!\00"		; <[3 x sbyte]*> [#uses=1]
%str12106 = internal global [3 x sbyte] c" P\00"		; <[3 x sbyte]*> [#uses=1]
%str13107 = internal global [3 x sbyte] c"*P\00"		; <[3 x sbyte]*> [#uses=1]
%str14 = internal global [3 x sbyte] c" N\00"		; <[3 x sbyte]*> [#uses=1]
%str15 = internal global [3 x sbyte] c"*N\00"		; <[3 x sbyte]*> [#uses=1]
%str16 = internal global [3 x sbyte] c" K\00"		; <[3 x sbyte]*> [#uses=1]
%str17 = internal global [3 x sbyte] c"*K\00"		; <[3 x sbyte]*> [#uses=1]
%str18 = internal global [3 x sbyte] c" R\00"		; <[3 x sbyte]*> [#uses=1]
%str19108 = internal global [3 x sbyte] c"*R\00"		; <[3 x sbyte]*> [#uses=1]
%str20 = internal global [3 x sbyte] c" Q\00"		; <[3 x sbyte]*> [#uses=1]
%str21109 = internal global [3 x sbyte] c"*Q\00"		; <[3 x sbyte]*> [#uses=1]
%str22 = internal global [3 x sbyte] c" B\00"		; <[3 x sbyte]*> [#uses=1]
%str23 = internal global [3 x sbyte] c"*B\00"		; <[3 x sbyte]*> [#uses=1]
%str24 = internal global [3 x sbyte] c"  \00"		; <[3 x sbyte]*> [#uses=1]
%str110 = internal global [42 x sbyte] c"+----+----+----+----+----+----+----+----+\00"		; <[42 x sbyte]*> [#uses=0]
%str25 = internal global [6 x sbyte] c"  %s\0A\00"		; <[6 x sbyte]*> [#uses=0]
%str26 = internal global [5 x sbyte] c"%d |\00"		; <[5 x sbyte]*> [#uses=0]
%str27 = internal global [6 x sbyte] c" %s |\00"		; <[6 x sbyte]*> [#uses=0]
%str28 = internal global [7 x sbyte] c"\0A  %s\0A\00"		; <[7 x sbyte]*> [#uses=0]
%str111 = internal global [45 x sbyte] c"\0A     a    b    c    d    e    f    g    h\0A\0A\00"		; <[45 x sbyte]*> [#uses=0]
%str29 = internal global [45 x sbyte] c"\0A     h    g    f    e    d    c    b    a\0A\0A\00"		; <[45 x sbyte]*> [#uses=0]
%str33 = internal global [2 x sbyte] c"<\00"		; <[2 x sbyte]*> [#uses=0]
%str34 = internal global [3 x sbyte] c"> \00"		; <[3 x sbyte]*> [#uses=0]
%str114 = internal global [18 x sbyte] c"%2i %7i %5i %8i  \00"		; <[18 x sbyte]*> [#uses=0]
%str115 = internal global [20 x sbyte] c"%2i %c%1i.%02i %9i \00"		; <[20 x sbyte]*> [#uses=0]
%str39 = internal global [5 x sbyte] c"%s !\00"		; <[5 x sbyte]*> [#uses=0]
%str40 = internal global [6 x sbyte] c"%s !!\00"		; <[6 x sbyte]*> [#uses=0]
%str41 = internal global [6 x sbyte] c"%s ??\00"		; <[6 x sbyte]*> [#uses=0]
%str124 = internal global [71 x sbyte] c"\0ASjeng version SPEC 1.0, Copyright (C) 2000-2005 Gian-Carlo Pascutto\0A\0A\00"		; <[71 x sbyte]*> [#uses=0]
%state = internal global [625 x uint] zeroinitializer		; <[625 x uint]*> [#uses=0]

implementation   ; Functions:

declare fastcc int %calc_attackers(int, int)

declare fastcc uint %is_attacked(int, int)

declare fastcc void %ProcessHoldings(sbyte*)

declare void %llvm.memset.i32(sbyte*, ubyte, uint, uint)

declare sbyte* %strncpy(sbyte*, sbyte*, uint)

declare void %llvm.memcpy.i32(sbyte*, sbyte*, uint, uint)

declare void %__eprintf(sbyte*, sbyte*, uint, sbyte*)

declare fastcc void %addHolding(int, int)

declare fastcc void %removeHolding(int, int)

declare fastcc void %DropremoveHolding(int, int)

declare int %printf(sbyte*, ...)

declare fastcc uint %is_draw()

declare void %exit(int)

declare fastcc void %setup_epd_line(sbyte*)

declare int %atoi(sbyte*)

declare fastcc void %reset_piece_square()

declare fastcc void %initialize_hash()

declare int %__maskrune(int, uint)

declare fastcc void %comp_to_san(long, long, long, sbyte*)

declare sbyte* %strstr(sbyte*, sbyte*)

declare int %atol(sbyte*)

declare %struct.FILE* %fopen(sbyte*, sbyte*)

declare fastcc void %display_board(int)

internal csretcc void %think(%struct.move_s* %agg.result) {
entry:
	%output.i = alloca [8 x sbyte], align 8		; <[8 x sbyte]*> [#uses=0]
	%comp_move = alloca %struct.move_s, align 16		; <%struct.move_s*> [#uses=7]
	%temp_move = alloca %struct.move_s, align 16		; <%struct.move_s*> [#uses=6]
	%moves = alloca [512 x %struct.move_s], align 16		; <[512 x %struct.move_s]*> [#uses=7]
	%output = alloca [8 x sbyte], align 8		; <[8 x sbyte]*> [#uses=1]
	store bool false, bool* %userealholdings.b
	%tmp = getelementptr [512 x %struct.move_s]* %moves, int 0, int 0		; <%struct.move_s*> [#uses=3]
	%tmp362 = getelementptr %struct.move_s* %comp_move, int 0, uint 0		; <int*> [#uses=0]
	%tmp365 = getelementptr %struct.move_s* %comp_move, int 0, uint 1		; <int*> [#uses=0]
	%tmp368 = getelementptr %struct.move_s* %comp_move, int 0, uint 2		; <int*> [#uses=0]
	%tmp371 = getelementptr %struct.move_s* %comp_move, int 0, uint 3		; <int*> [#uses=0]
	%tmp374 = getelementptr %struct.move_s* %comp_move, int 0, uint 4		; <int*> [#uses=0]
	%tmp377 = getelementptr %struct.move_s* %comp_move, int 0, uint 5		; <int*> [#uses=0]
	%tmp = cast %struct.move_s* %comp_move to { long, long, long }*		; <{ long, long, long }*> [#uses=3]
	%tmp = getelementptr { long, long, long }* %tmp, int 0, uint 0		; <long*> [#uses=0]
	%tmp829 = getelementptr { long, long, long }* %tmp, int 0, uint 1		; <long*> [#uses=0]
	%tmp832 = getelementptr { long, long, long }* %tmp, int 0, uint 2		; <long*> [#uses=0]
	%output = getelementptr [8 x sbyte]* %output, int 0, int 0		; <sbyte*> [#uses=0]
	%tmp573 = getelementptr %struct.move_s* %temp_move, int 0, uint 0		; <int*> [#uses=0]
	%tmp576 = getelementptr %struct.move_s* %temp_move, int 0, uint 1		; <int*> [#uses=0]
	%tmp579 = getelementptr %struct.move_s* %temp_move, int 0, uint 2		; <int*> [#uses=0]
	%tmp582 = getelementptr %struct.move_s* %temp_move, int 0, uint 3		; <int*> [#uses=0]
	%tmp585 = getelementptr %struct.move_s* %temp_move, int 0, uint 4		; <int*> [#uses=0]
	%tmp588 = getelementptr %struct.move_s* %temp_move, int 0, uint 5		; <int*> [#uses=0]
	%pn_restart.0.ph = cast uint 0 to int		; <int> [#uses=2]
	%tmp21362 = seteq uint 0, 0		; <bool> [#uses=2]
	%tmp216 = cast int %pn_restart.0.ph to float		; <float> [#uses=1]
	%tmp216 = cast float %tmp216 to double		; <double> [#uses=1]
	%tmp217 = add double %tmp216, 1.000000e+00		; <double> [#uses=1]
	%tmp835 = setgt int %pn_restart.0.ph, 9		; <bool> [#uses=0]
	store int 0, int* %nodes
	store int 0, int* %qnodes
	store int 1, int* %ply
	store uint 0, uint* %ECacheProbes
	store uint 0, uint* %ECacheHits
	store uint 0, uint* %TTProbes
	store uint 0, uint* %TTHits
	store uint 0, uint* %TTStores
	store uint 0, uint* %NCuts
	store uint 0, uint* %NTries
	store uint 0, uint* %TExt
	store uint 0, uint* %FH
	store uint 0, uint* %FHF
	store uint 0, uint* %PVS
	store uint 0, uint* %FULL
	store uint 0, uint* %PVSF
	store uint 0, uint* %ext_check
	store uint 0, uint* %ext_onerep
	store uint 0, uint* %razor_drop
	store uint 0, uint* %razor_material
	store bool false, bool* %extendedtime.b
	store bool false, bool* %forcedwin.b
	store int 200, int* %maxposdiff
	store sbyte 0, sbyte* %true_i_depth
	store int 0, int* %legals
	%tmp48 = load int* %Variant		; <int> [#uses=1]
	%tmp49 = seteq int %tmp48, 4		; <bool> [#uses=1]
	%storemerge = cast bool %tmp49 to uint		; <uint> [#uses=1]
	store uint %storemerge, uint* %captures
	call fastcc void %gen( %struct.move_s* %tmp )
	%tmp53 = load int* %numb_moves		; <int> [#uses=1]
	%tmp.i = load int* %Variant		; <int> [#uses=1]
	%tmp.i = seteq int %tmp.i, 3		; <bool> [#uses=1]
	br bool %tmp.i, label %in_check.exit, label %cond_next.i

cond_next.i:		; preds = %entry
	%tmp2.i5 = load int* %white_to_move		; <int> [#uses=1]
	%tmp3.i = seteq int %tmp2.i5, 1		; <bool> [#uses=0]
	ret void

in_check.exit:		; preds = %entry
	%tmp7637 = setgt int %tmp53, 0		; <bool> [#uses=1]
	br bool %tmp7637, label %cond_true77, label %bb80

cond_true77:		; preds = %in_check.exit
	%l.1.0 = cast uint 0 to int		; <int> [#uses=2]
	call fastcc void %make( %struct.move_s* %tmp, int %l.1.0 )
	%tmp61 = call fastcc uint %check_legal( %struct.move_s* %tmp, int %l.1.0, int 0 )		; <uint> [#uses=1]
	%tmp62 = seteq uint %tmp61, 0		; <bool> [#uses=0]
	ret void

bb80:		; preds = %in_check.exit
	%tmp81 = load int* %Variant		; <int> [#uses=1]
	%tmp82 = seteq int %tmp81, 4		; <bool> [#uses=1]
	br bool %tmp82, label %cond_true83, label %cond_next118

cond_true83:		; preds = %bb80
	%tmp84 = load int* %legals		; <int> [#uses=1]
	%tmp85 = seteq int %tmp84, 0		; <bool> [#uses=0]
	ret void

cond_next118:		; preds = %bb80
	%tmp119 = load int* %Variant		; <int> [#uses=1]
	%tmp120 = seteq int %tmp119, 1		; <bool> [#uses=1]
	br bool %tmp120, label %cond_next176, label %cond_true121

cond_true121:		; preds = %cond_next118
	%tmp122.b = load bool* %is_pondering.b		; <bool> [#uses=1]
	br bool %tmp122.b, label %cond_next176, label %cond_true124

cond_true124:		; preds = %cond_true121
	%tmp125 = load int* %legals		; <int> [#uses=1]
	%tmp126 = seteq int %tmp125, 1		; <bool> [#uses=1]
	br bool %tmp126, label %cond_true127, label %cond_next176

cond_true127:		; preds = %cond_true124
	%tmp128 = load int* %inc		; <int> [#uses=1]
	%tmp129 = mul int %tmp128, 100		; <int> [#uses=1]
	%tmp130 = load int* %time_cushion		; <int> [#uses=1]
	%tmp131 = add int %tmp129, %tmp130		; <int> [#uses=1]
	store int %tmp131, int* %time_cushion
	%tmp134 = getelementptr %struct.move_s* %agg.result, int 0, uint 0		; <int*> [#uses=1]
	%tmp135 = getelementptr [512 x %struct.move_s]* %moves, int 0, int 0, uint 0		; <int*> [#uses=1]
	%tmp136 = load int* %tmp135		; <int> [#uses=1]
	store int %tmp136, int* %tmp134
	%tmp137 = getelementptr %struct.move_s* %agg.result, int 0, uint 1		; <int*> [#uses=1]
	%tmp138 = getelementptr [512 x %struct.move_s]* %moves, int 0, int 0, uint 1		; <int*> [#uses=1]
	%tmp139 = load int* %tmp138		; <int> [#uses=1]
	store int %tmp139, int* %tmp137
	%tmp140 = getelementptr %struct.move_s* %agg.result, int 0, uint 2		; <int*> [#uses=1]
	%tmp141 = getelementptr [512 x %struct.move_s]* %moves, int 0, int 0, uint 2		; <int*> [#uses=1]
	%tmp142 = load int* %tmp141		; <int> [#uses=1]
	store int %tmp142, int* %tmp140
	%tmp143 = getelementptr %struct.move_s* %agg.result, int 0, uint 3		; <int*> [#uses=1]
	%tmp144 = getelementptr [512 x %struct.move_s]* %moves, int 0, int 0, uint 3		; <int*> [#uses=1]
	%tmp145 = load int* %tmp144		; <int> [#uses=1]
	store int %tmp145, int* %tmp143
	%tmp146 = getelementptr %struct.move_s* %agg.result, int 0, uint 4		; <int*> [#uses=1]
	%tmp147 = getelementptr [512 x %struct.move_s]* %moves, int 0, int 0, uint 4		; <int*> [#uses=1]
	%tmp148 = load int* %tmp147		; <int> [#uses=1]
	store int %tmp148, int* %tmp146
	%tmp149 = getelementptr %struct.move_s* %agg.result, int 0, uint 5		; <int*> [#uses=1]
	%tmp150 = getelementptr [512 x %struct.move_s]* %moves, int 0, int 0, uint 5		; <int*> [#uses=1]
	%tmp151 = load int* %tmp150		; <int> [#uses=1]
	store int %tmp151, int* %tmp149
	ret void

cond_next176:		; preds = %cond_true124, %cond_true121, %cond_next118
	call fastcc void %check_phase( )
	%tmp177 = load int* %phase		; <int> [#uses=1]
	switch int %tmp177, label %bb187 [
		 int 0, label %bb178
		 int 1, label %bb180
		 int 2, label %bb183
	]

bb178:		; preds = %cond_next176
	%tmp179 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([16 x sbyte]* %str40, int 0, uint 0) )		; <int> [#uses=0]
	%tmp18854.b = load bool* %is_pondering.b		; <bool> [#uses=1]
	br bool %tmp18854.b, label %cond_false210, label %cond_true190

bb180:		; preds = %cond_next176
	%tmp182 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([19 x sbyte]* %str, int 0, uint 0) )		; <int> [#uses=0]
	%tmp18856.b = load bool* %is_pondering.b		; <bool> [#uses=0]
	ret void

bb183:		; preds = %cond_next176
	%tmp185 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([16 x sbyte]* %str1, int 0, uint 0) )		; <int> [#uses=0]
	%tmp18858.b = load bool* %is_pondering.b		; <bool> [#uses=0]
	ret void

bb187:		; preds = %cond_next176
	%tmp188.b = load bool* %is_pondering.b		; <bool> [#uses=0]
	ret void

cond_true190:		; preds = %bb178
	%tmp191 = load int* %fixed_time		; <int> [#uses=1]
	%tmp192 = seteq int %tmp191, 0		; <bool> [#uses=0]
	ret void

cond_false210:		; preds = %bb178
	store int 999999, int* %time_for_move
	br bool %tmp21362, label %cond_true226.critedge, label %bb287.critedge

cond_true226.critedge:		; preds = %cond_false210
	%tmp223.c = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([20 x sbyte]* %str43, int 0, uint 0), int 999999 )		; <int> [#uses=0]
	%tmp.i = load %struct.TType** %DP_TTable		; <%struct.TType*> [#uses=1]
	%tmp.i7.b = load bool* %TTSize.b		; <bool> [#uses=1]
	%tmp1.i = select bool %tmp.i7.b, uint 60000000, uint 0		; <uint> [#uses=1]
	%tmp.i = getelementptr %struct.TType* %tmp.i, int 0, uint 0		; <sbyte*> [#uses=1]
	call void %llvm.memset.i32( sbyte* %tmp.i, ubyte 0, uint %tmp1.i, uint 4 )
	%tmp2.i = load %struct.TType** %AS_TTable		; <%struct.TType*> [#uses=1]
	%tmp3.i8.b = load bool* %TTSize.b		; <bool> [#uses=1]
	%tmp4.i = select bool %tmp3.i8.b, uint 60000000, uint 0		; <uint> [#uses=1]
	%tmp2.i = getelementptr %struct.TType* %tmp2.i, int 0, uint 0		; <sbyte*> [#uses=1]
	call void %llvm.memset.i32( sbyte* %tmp2.i, ubyte 0, uint %tmp4.i, uint 4 )
	%tmp.i = load %struct.QTType** %QS_TTable		; <%struct.QTType*> [#uses=1]
	%tmp5.i9.b = load bool* %TTSize.b		; <bool> [#uses=1]
	%tmp6.i10 = select bool %tmp5.i9.b, uint 48000000, uint 0		; <uint> [#uses=1]
	%tmp7.i = getelementptr %struct.QTType* %tmp.i, int 0, uint 0		; <sbyte*> [#uses=1]
	call void %llvm.memset.i32( sbyte* %tmp7.i, ubyte 0, uint %tmp6.i10, uint 4 )
	%tmp.i = load %struct.ECacheType** %ECache		; <%struct.ECacheType*> [#uses=1]
	%tmp.i14.b = load bool* %ECacheSize.b		; <bool> [#uses=1]
	%tmp1.i16 = select bool %tmp.i14.b, uint 12000000, uint 0		; <uint> [#uses=1]
	%tmp.i17 = cast %struct.ECacheType* %tmp.i to sbyte*		; <sbyte*> [#uses=1]
	call void %llvm.memset.i32( sbyte* %tmp.i17, ubyte 0, uint %tmp1.i16, uint 4 )
	call void %llvm.memset.i32( sbyte* cast ([300 x int]* %rootlosers to sbyte*), ubyte 0, uint 1200, uint 4 )
	%tmp234.b = load bool* %is_pondering.b		; <bool> [#uses=1]
	br bool %tmp234.b, label %bb263, label %cond_next238

cond_next238:		; preds = %cond_true226.critedge
	%tmp239 = load int* %Variant		; <int> [#uses=2]
	switch int %tmp239, label %bb263 [
		 int 3, label %bb249
		 int 4, label %bb249
	]

bb249:		; preds = %cond_next238, %cond_next238
	%tmp250 = load int* %piece_count		; <int> [#uses=1]
	%tmp251 = setgt int %tmp250, 3		; <bool> [#uses=1]
	%tmp240.not = setne int %tmp239, 3		; <bool> [#uses=1]
	%brmerge = or bool %tmp251, %tmp240.not		; <bool> [#uses=1]
	br bool %brmerge, label %bb260, label %bb263

bb260:		; preds = %bb249
	%tmp261 = load int* %time_for_move		; <int> [#uses=1]
	%tmp261 = cast int %tmp261 to float		; <float> [#uses=1]
	%tmp261 = cast float %tmp261 to double		; <double> [#uses=1]
	%tmp262 = div double %tmp261, 3.000000e+00		; <double> [#uses=1]
	%tmp262 = cast double %tmp262 to int		; <int> [#uses=1]
	store int %tmp262, int* %pn_time
	%tmp1.b.i = load bool* %PBSize.b		; <bool> [#uses=1]
	%tmp1.i1 = select bool %tmp1.b.i, uint 200000, uint 0		; <uint> [#uses=1]
	%tmp.i2 = call sbyte* %calloc( uint %tmp1.i1, uint 44 )		; <sbyte*> [#uses=1]
	%tmp.i = cast sbyte* %tmp.i2 to ubyte*		; <ubyte*> [#uses=1]
	store ubyte* %tmp.i, ubyte** %membuff
	%tmp2.i3 = call sbyte* %calloc( uint 1, uint 44 )		; <sbyte*> [#uses=3]
	%tmp2.i = cast sbyte* %tmp2.i3 to %struct.node_t*		; <%struct.node_t*> [#uses=6]
	%tmp.i = getelementptr [512 x %struct.move_s]* null, int 0, int 0		; <%struct.move_s*> [#uses=3]
	call fastcc void %gen( %struct.move_s* %tmp.i )
	%tmp3.i4 = load int* %numb_moves		; <int> [#uses=4]
	%tmp3.i5 = cast int %tmp3.i4 to uint		; <uint> [#uses=0]
	store bool false, bool* %alllosers.b
	call void %llvm.memset.i32( sbyte* cast ([300 x int]* %rootlosers to sbyte*), ubyte 0, uint 1200, uint 4 )
	%nodesspent.i = cast [512 x int]* null to sbyte*		; <sbyte*> [#uses=1]
	call void %llvm.memset.i32( sbyte* %nodesspent.i, ubyte 0, uint 2048, uint 16 )
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 0)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 1)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 2)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 3)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 4)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 5)
	%tmp.i.i = load int* %Variant		; <int> [#uses=1]
	%tmp.i.i = seteq int %tmp.i.i, 3		; <bool> [#uses=1]
	br bool %tmp.i.i, label %in_check.exit.i, label %cond_next.i.i

cond_next.i.i:		; preds = %bb260
	%tmp2.i.i = load int* %white_to_move		; <int> [#uses=1]
	%tmp3.i.i = seteq int %tmp2.i.i, 1		; <bool> [#uses=1]
	br bool %tmp3.i.i, label %cond_true4.i.i, label %cond_false12.i.i

cond_true4.i.i:		; preds = %cond_next.i.i
	%tmp5.i.i = load int* %wking_loc		; <int> [#uses=1]
	%tmp6.i.i = call fastcc uint %is_attacked( int %tmp5.i.i, int 0 )		; <uint> [#uses=1]
	%not.tmp7.i.i = setne uint %tmp6.i.i, 0		; <bool> [#uses=1]
	%tmp217.i = cast bool %not.tmp7.i.i to int		; <int> [#uses=1]
	%tmp4219.i = setgt int %tmp3.i4, 0		; <bool> [#uses=1]
	br bool %tmp4219.i, label %cond_true43.i, label %bb46.i

cond_false12.i.i:		; preds = %cond_next.i.i
	%tmp13.i.i = load int* %bking_loc		; <int> [#uses=1]
	%tmp14.i.i = call fastcc uint %is_attacked( int %tmp13.i.i, int 1 )		; <uint> [#uses=1]
	%not.tmp15.i.i = setne uint %tmp14.i.i, 0		; <bool> [#uses=1]
	%tmp2120.i = cast bool %not.tmp15.i.i to int		; <int> [#uses=1]
	%tmp4222.i = setgt int %tmp3.i4, 0		; <bool> [#uses=1]
	br bool %tmp4222.i, label %cond_true43.i, label %bb46.i

in_check.exit.i:		; preds = %bb260
	%tmp4224.i = setgt int %tmp3.i4, 0		; <bool> [#uses=0]
	ret void

cond_true43.i:		; preds = %cond_false12.i.i, %cond_true4.i.i
	%tmp21.0.ph.i = phi int [ %tmp217.i, %cond_true4.i.i ], [ %tmp2120.i, %cond_false12.i.i ]		; <int> [#uses=1]
	%i.0.0.i = cast uint 0 to int		; <int> [#uses=2]
	call fastcc void %make( %struct.move_s* %tmp.i, int %i.0.0.i )
	%tmp27.i = call fastcc uint %check_legal( %struct.move_s* %tmp.i, int %i.0.0.i, int %tmp21.0.ph.i )		; <uint> [#uses=1]
	%tmp.i6 = seteq uint %tmp27.i, 0		; <bool> [#uses=0]
	ret void

bb46.i:		; preds = %cond_false12.i.i, %cond_true4.i.i
	%tmp48.i = seteq int 0, 0		; <bool> [#uses=1]
	br bool %tmp48.i, label %cond_true49.i, label %cond_next53.i

cond_true49.i:		; preds = %bb46.i
	store int 0, int* %bufftop
	%tmp50.i = load ubyte** %membuff		; <ubyte*> [#uses=1]
	free ubyte* %tmp50.i
	free sbyte* %tmp2.i3
	ret void

cond_next53.i:		; preds = %bb46.i
	store int 1, int* %nodecount
	store int 0, int* %iters
	store int 0, int* %maxply
	store int 0, int* %forwards
	%tmp54.i = load int* %move_number		; <int> [#uses=1]
	%tmp55.i = load int* %ply		; <int> [#uses=1]
	%tmp56.i = add int %tmp54.i, -1		; <int> [#uses=1]
	%tmp57.i = add int %tmp56.i, %tmp55.i		; <int> [#uses=1]
	%tmp58.i = load uint* %hash		; <uint> [#uses=1]
	%tmp.i = getelementptr [600 x uint]* %hash_history, int 0, int %tmp57.i		; <uint*> [#uses=1]
	store uint %tmp58.i, uint* %tmp.i
	%tmp59.i = load int* %white_to_move		; <int> [#uses=1]
	%tmp60.i = seteq int %tmp59.i, 0		; <bool> [#uses=1]
	%tmp60.i = cast bool %tmp60.i to int		; <int> [#uses=1]
	store int %tmp60.i, int* %root_to_move
	%tmp.i4.i = load int* %Variant		; <int> [#uses=2]
	%tmp.i5.i = seteq int %tmp.i4.i, 3		; <bool> [#uses=1]
	br bool %tmp.i5.i, label %cond_true.i.i, label %cond_false.i.i

cond_true.i.i:		; preds = %cond_next53.i
	call fastcc void %suicide_pn_eval( %struct.node_t* %tmp2.i )
	%tmp6328.i = getelementptr %struct.node_t* %tmp2.i, int 0, uint 0		; <ubyte*> [#uses=1]
	%tmp29.i = load ubyte* %tmp6328.i		; <ubyte> [#uses=1]
	%tmp6430.i = seteq ubyte %tmp29.i, 1		; <bool> [#uses=0]
	ret void

cond_false.i.i:		; preds = %cond_next53.i
	%tmp2.i.i = seteq int %tmp.i4.i, 4		; <bool> [#uses=1]
	%tmp63.i = getelementptr %struct.node_t* %tmp2.i, int 0, uint 0		; <ubyte*> [#uses=2]
	br bool %tmp2.i.i, label %cond_true3.i.i, label %cond_false5.i.i

cond_true3.i.i:		; preds = %cond_false.i.i
	call fastcc void %losers_pn_eval( %struct.node_t* %tmp2.i )
	%tmp31.i = load ubyte* %tmp63.i		; <ubyte> [#uses=1]
	%tmp6432.i = seteq ubyte %tmp31.i, 1		; <bool> [#uses=1]
	br bool %tmp6432.i, label %bb75.i, label %cond_next67.i

cond_false5.i.i:		; preds = %cond_false.i.i
	call fastcc void %std_pn_eval( %struct.node_t* %tmp2.i )
	%tmp.i = load ubyte* %tmp63.i		; <ubyte> [#uses=1]
	%tmp64.i = seteq ubyte %tmp.i, 1		; <bool> [#uses=0]
	ret void

cond_next67.i:		; preds = %cond_true3.i.i
	%tmp69.i = getelementptr %struct.node_t* %tmp2.i, int 0, uint 0		; <ubyte*> [#uses=1]
	%tmp70.i = load ubyte* %tmp69.i		; <ubyte> [#uses=1]
	%tmp71.i = seteq ubyte %tmp70.i, 0		; <bool> [#uses=0]
	ret void

bb75.i:		; preds = %cond_true3.i.i
	store int 0, int* %bufftop
	%tmp76.i = load ubyte** %membuff		; <ubyte*> [#uses=1]
	free ubyte* %tmp76.i
	free sbyte* %tmp2.i3
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 0)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 1)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 2)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 3)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 4)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 5)
	%tmp28869 = load int* %result		; <int> [#uses=1]
	%tmp28970 = seteq int %tmp28869, 0		; <bool> [#uses=1]
	br bool %tmp28970, label %cond_next337, label %cond_true290

bb263:		; preds = %bb249, %cond_next238, %cond_true226.critedge
	br bool %tmp21362, label %cond_true266, label %bb287

cond_true266:		; preds = %bb263
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 0)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 1)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 2)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 3)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 4)
	store int 0, int* getelementptr (%struct.move_s* %pn_move, uint 0, uint 5)
	%tmp28871 = load int* %result		; <int> [#uses=1]
	%tmp28972 = seteq int %tmp28871, 0		; <bool> [#uses=0]
	ret void

bb287.critedge:		; preds = %cond_false210
	%tmp218.c = div double 1.999998e+06, %tmp217		; <double> [#uses=1]
	%tmp218.c = cast double %tmp218.c to int		; <int> [#uses=2]
	store int %tmp218.c, int* %time_for_move
	%tmp22367.c = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([20 x sbyte]* %str43, int 0, uint 0), int %tmp218.c )		; <int> [#uses=0]
	ret void

bb287:		; preds = %bb263
	%tmp288 = load int* %result		; <int> [#uses=1]
	%tmp289 = seteq int %tmp288, 0		; <bool> [#uses=0]
	ret void

cond_true290:		; preds = %bb75.i
	%tmp292 = load int* getelementptr (%struct.move_s* %pn_move, int 0, uint 1)		; <int> [#uses=1]
	%tmp295 = seteq int %tmp292, 0		; <bool> [#uses=0]
	ret void

cond_next337:		; preds = %bb75.i
	%tmp338.b = load bool* %forcedwin.b		; <bool> [#uses=1]
	br bool %tmp338.b, label %bb348, label %cond_next342

cond_next342:		; preds = %cond_next337
	%tmp343 = load int* %result		; <int> [#uses=1]
	%tmp344 = seteq int %tmp343, 0		; <bool> [#uses=0]
	ret void

bb348:		; preds = %cond_next337
	%tmp350 = load int* getelementptr (%struct.move_s* %pn_move, int 0, uint 1)		; <int> [#uses=1]
	%tmp353 = seteq int %tmp350, 0		; <bool> [#uses=0]
	ret void
}

declare fastcc int %eval(int, int)

declare sbyte* %fgets(sbyte*, int, %struct.FILE*)

declare int %fclose(%struct.FILE*)

declare fastcc int %losers_eval()

declare fastcc int %l_bishop_mobility(int)

declare fastcc int %l_rook_mobility(int)

declare fastcc uint %check_legal(%struct.move_s*, int, int)

declare fastcc void %gen(%struct.move_s*)

declare fastcc void %push_pawn(int, uint)

declare fastcc void %push_knighT(int)

declare fastcc void %push_slidE(int)

declare fastcc void %push_king(int)

declare fastcc uint %f_in_check(%struct.move_s*, int)

declare fastcc void %make(%struct.move_s*, int)

declare fastcc void %add_capture(int, int, int)

declare fastcc void %unmake(%struct.move_s*, int)

declare int %ErrorIt(int, int)

declare int %Pawn(int, int)

declare int %Knight(int, int)

declare int %King(int, int)

declare int %Rook(int, int)

declare int %Queen(int, int)

declare int %Bishop(int, int)

declare fastcc void %check_phase()

declare fastcc int %bishop_mobility(int)

declare fastcc int %rook_mobility(int)

declare int %sscanf(sbyte*, sbyte*, ...)

declare int %strncmp(sbyte*, sbyte*, uint)

declare sbyte* %strchr(sbyte*, int)

declare fastcc void %CheckBadFlow(uint)

declare fastcc void %suicide_pn_eval(%struct.node_t*)

declare fastcc void %losers_pn_eval(%struct.node_t*)

declare fastcc void %std_pn_eval(%struct.node_t*)

declare fastcc %struct.node_t* %select_most_proving(%struct.node_t*)

declare fastcc void %set_proof_and_disproof_numbers(%struct.node_t*)

declare fastcc void %StoreTT(int, int, int, int, int, int)

declare fastcc void %develop_node(%struct.node_t*)

declare fastcc void %update_ancestors(%struct.node_t*)

declare sbyte* %calloc(uint, uint)

declare fastcc void %comp_to_coord(long, long, long, sbyte*)

declare sbyte* %strcat(sbyte*, sbyte*)

declare int %sprintf(sbyte*, sbyte*, ...)

declare fastcc void %order_moves(%struct.move_s*, int*, int*, int, int)

declare fastcc int %see(int, int, int)

declare fastcc void %perft(int)

declare fastcc int %qsearch(int, int, int)

declare fastcc int %allocate_time()

declare fastcc void %QStoreTT(int, int, int, int)

declare fastcc int %search(int, int, int, int)

declare fastcc int %ProbeTT(int*, int, int*, int*, int*, int)

declare csretcc void %search_root(%struct.move_s*, int, int, int)

declare fastcc void %post_fh_thinking(int, %struct.move_s*)

declare fastcc void %post_thinking(int)

declare int %fprintf(%struct.FILE*, sbyte*, ...)

declare fastcc int %s_bishop_mobility(int)

declare fastcc int %s_rook_mobility(int)

declare fastcc int %suicide_mid_eval()

declare int %main(int, sbyte**)

declare fastcc void %init_game()

declare void %setbuf(%struct.FILE*, sbyte*)

declare sbyte* %strcpy(sbyte*, sbyte*)

declare int %__tolower(int)

declare int %strcmp(sbyte*, sbyte*)

declare void (int)* %signal(int, void (int)*)

declare bool %llvm.isunordered.f64(double, double)

declare fastcc void %hash_extract_pv(int, sbyte*)

declare double %difftime(int, int)

declare int %getc(%struct.FILE*)

declare uint %strlen(sbyte*)

declare uint %fwrite(sbyte*, uint, uint, %struct.FILE*)
