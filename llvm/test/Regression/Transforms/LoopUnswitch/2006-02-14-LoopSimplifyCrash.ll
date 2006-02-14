; RUN: llvm-as < %s | opt -loop-unswitch -disable-output

target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin8.2.0"
deplibs = [ "c", "crtend" ]
	%struct.__sFILE = type { ubyte*, int, int, short, short, %struct.__sbuf, int, sbyte*, int (sbyte*)*, int (sbyte*, sbyte*, int)*, long (sbyte*, long, int)*, int (sbyte*, sbyte*, int)*, %struct.__sbuf, %struct.__sFILEX*, int, [3 x ubyte], [1 x ubyte], %struct.__sbuf, int, long }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { ubyte*, int }
	%struct.fd_set = type { [32 x int] }
	%struct.timeval = type { int, int }
	%struct.tm = type { int, int, int, int, int, int, int, int, int, int, sbyte* }
	%typedef.CHESS_PATH = type { [65 x int], ubyte, ubyte, ubyte }
	%typedef.CHESS_POSITION = type { ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, uint, int, sbyte, sbyte, [64 x sbyte], sbyte, sbyte, sbyte, sbyte, sbyte }
	%typedef.HASH_ENTRY = type { ulong, ulong }
	%typedef.NEXT_MOVE = type { int, int, int* }
	%typedef.PAWN_HASH_ENTRY = type { uint, short, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte }
	%typedef.SEARCH_POSITION = type { ubyte, sbyte, sbyte, ubyte }
	%union.doub0. = type { ulong }
%search = external global %typedef.CHESS_POSITION		; <%typedef.CHESS_POSITION*> [#uses=1]
%w_pawn_attacks = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%b_pawn_attacks = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%knight_attacks = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%bishop_attacks_rl45 = external global [64 x [256 x ulong]]		; <[64 x [256 x ulong]]*> [#uses=0]
%bishop_shift_rl45 = external global [64 x int]		; <[64 x int]*> [#uses=0]
%bishop_attacks_rr45 = external global [64 x [256 x ulong]]		; <[64 x [256 x ulong]]*> [#uses=0]
%bishop_shift_rr45 = external global [64 x int]		; <[64 x int]*> [#uses=0]
%rook_attacks_r0 = external global [64 x [256 x ulong]]		; <[64 x [256 x ulong]]*> [#uses=0]
%rook_attacks_rl90 = external global [64 x [256 x ulong]]		; <[64 x [256 x ulong]]*> [#uses=0]
%king_attacks = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%set_mask = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%first_ones = external global [65536 x ubyte]		; <[65536 x ubyte]*> [#uses=0]
%last_ones = external global [65536 x ubyte]		; <[65536 x ubyte]*> [#uses=0]
%draw_score_is_zero = external global int		; <int*> [#uses=0]
%default_draw_score = external global int		; <int*> [#uses=0]
%opening = external global int		; <int*> [#uses=0]
%middle_game = external global int		; <int*> [#uses=0]
%tc_increment = external global int		; <int*> [#uses=0]
%tc_time_remaining_opponent = external global int		; <int*> [#uses=0]
%.ctor_1 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%input_stream = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
%__sF = external global [0 x %struct.__sFILE]		; <[0 x %struct.__sFILE]*> [#uses=1]
%xboard = external global int		; <int*> [#uses=0]
%.str_1 = external global [14 x sbyte]		; <[14 x sbyte]*> [#uses=0]
%.str_2 = external global [14 x sbyte]		; <[14 x sbyte]*> [#uses=0]
%buffer = external global [512 x sbyte]		; <[512 x sbyte]*> [#uses=0]
%nargs = external global int		; <int*> [#uses=0]
%args = external global [32 x sbyte*]		; <[32 x sbyte*]*> [#uses=0]
%.str_3 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_4 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.str_5 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_6 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_7 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_8 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_9 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_10 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_11 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_12 = external global [24 x sbyte]		; <[24 x sbyte]*> [#uses=0]
%.str_14 = external global [23 x sbyte]		; <[23 x sbyte]*> [#uses=0]
%position = external global [67 x %typedef.SEARCH_POSITION]		; <[67 x %typedef.SEARCH_POSITION]*> [#uses=0]
%log_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
%move_number = external global int		; <int*> [#uses=0]
%rephead_b = external global ulong*		; <ulong**> [#uses=0]
%replist_b = external global [82 x ulong]		; <[82 x ulong]*> [#uses=0]
%rephead_w = external global ulong*		; <ulong**> [#uses=0]
%replist_w = external global [82 x ulong]		; <[82 x ulong]*> [#uses=0]
%moves_out_of_book = external global int		; <int*> [#uses=0]
%largest_positional_score = external global int		; <int*> [#uses=0]
%end_game = external global int		; <int*> [#uses=0]
%p_values = external global [15 x int]		; <[15 x int]*> [#uses=0]
%clear_mask = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%directions = external global [64 x [64 x sbyte]]		; <[64 x [64 x sbyte]]*> [#uses=0]
%root_wtm = external global int		; <int*> [#uses=0]
%all_pawns = external global ulong		; <ulong*> [#uses=0]
%pawn_score = external global %typedef.PAWN_HASH_ENTRY		; <%typedef.PAWN_HASH_ENTRY*> [#uses=0]
%pawn_probes = external global int		; <int*> [#uses=0]
%pawn_hits = external global int		; <int*> [#uses=0]
%outside_passed = external global [128 x int]		; <[128 x int]*> [#uses=0]
%root_total_black_pieces = external global int		; <int*> [#uses=0]
%root_total_white_pawns = external global int		; <int*> [#uses=0]
%root_total_white_pieces = external global int		; <int*> [#uses=0]
%root_total_black_pawns = external global int		; <int*> [#uses=0]
%mask_A7H7 = external global ulong		; <ulong*> [#uses=0]
%mask_B6B7 = external global ulong		; <ulong*> [#uses=0]
%mask_G6G7 = external global ulong		; <ulong*> [#uses=0]
%mask_A2H2 = external global ulong		; <ulong*> [#uses=0]
%mask_B2B3 = external global ulong		; <ulong*> [#uses=0]
%mask_G2G3 = external global ulong		; <ulong*> [#uses=0]
%king_defects_w = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%good_bishop_kw = external global ulong		; <ulong*> [#uses=0]
%mask_F3H3 = external global ulong		; <ulong*> [#uses=0]
%file_mask = external global [8 x ulong]		; <[8 x ulong]*> [#uses=0]
%good_bishop_qw = external global ulong		; <ulong*> [#uses=0]
%mask_A3C3 = external global ulong		; <ulong*> [#uses=0]
%king_defects_b = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%good_bishop_kb = external global ulong		; <ulong*> [#uses=0]
%mask_F6H6 = external global ulong		; <ulong*> [#uses=0]
%good_bishop_qb = external global ulong		; <ulong*> [#uses=0]
%mask_A6C6 = external global ulong		; <ulong*> [#uses=0]
%square_color = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%evaluations = external global uint		; <uint*> [#uses=0]
%king_value_w = external global [64 x int]		; <[64 x int]*> [#uses=0]
%rank_mask = external global [8 x ulong]		; <[8 x ulong]*> [#uses=0]
%mask_kr_trapped_w = external global [3 x ulong]		; <[3 x ulong]*> [#uses=0]
%mask_qr_trapped_w = external global [3 x ulong]		; <[3 x ulong]*> [#uses=0]
%king_value_b = external global [64 x int]		; <[64 x int]*> [#uses=0]
%mask_kr_trapped_b = external global [3 x ulong]		; <[3 x ulong]*> [#uses=0]
%mask_qr_trapped_b = external global [3 x ulong]		; <[3 x ulong]*> [#uses=0]
%white_outpost = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%mask_no_pawn_attacks_b = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%knight_value_w = external global [64 x int]		; <[64 x int]*> [#uses=0]
%black_outpost = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%mask_no_pawn_attacks_w = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%knight_value_b = external global [64 x int]		; <[64 x int]*> [#uses=0]
%bishop_value_w = external global [64 x int]		; <[64 x int]*> [#uses=0]
%bishop_mobility_rl45 = external global [64 x [256 x int]]		; <[64 x [256 x int]]*> [#uses=0]
%bishop_mobility_rr45 = external global [64 x [256 x int]]		; <[64 x [256 x int]]*> [#uses=0]
%bishop_value_b = external global [64 x int]		; <[64 x int]*> [#uses=0]
%rook_value_w = external global [64 x int]		; <[64 x int]*> [#uses=0]
%plus8dir = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%mask_abs7_w = external global ulong		; <ulong*> [#uses=0]
%rook_value_b = external global [64 x int]		; <[64 x int]*> [#uses=0]
%minus8dir = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%mask_abs7_b = external global ulong		; <ulong*> [#uses=0]
%queen_value_w = external global [64 x int]		; <[64 x int]*> [#uses=0]
%queen_value_b = external global [64 x int]		; <[64 x int]*> [#uses=0]
%white_minor_pieces = external global ulong		; <ulong*> [#uses=0]
%black_minor_pieces = external global ulong		; <ulong*> [#uses=0]
%not_rook_pawns = external global ulong		; <ulong*> [#uses=0]
%dark_squares = external global ulong		; <ulong*> [#uses=0]
%b_n_mate_dark_squares = external global [64 x int]		; <[64 x int]*> [#uses=0]
%b_n_mate_light_squares = external global [64 x int]		; <[64 x int]*> [#uses=0]
%mate = external global [64 x int]		; <[64 x int]*> [#uses=0]
%first_ones_8bit = external global [256 x ubyte]		; <[256 x ubyte]*> [#uses=0]
%reduced_material_passer = external global [20 x int]		; <[20 x int]*> [#uses=0]
%supported_passer = external global [8 x int]		; <[8 x int]*> [#uses=0]
%passed_pawn_value = external global [8 x int]		; <[8 x int]*> [#uses=0]
%connected_passed = external global [256 x ubyte]		; <[256 x ubyte]*> [#uses=0]
%black_pawn_race_btm = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%white_pawn_race_wtm = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%black_pawn_race_wtm = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%white_pawn_race_btm = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%obstructed = external global [64 x [64 x ulong]]		; <[64 x [64 x ulong]]*> [#uses=0]
%pawn_hash_table = external global %typedef.PAWN_HASH_ENTRY*		; <%typedef.PAWN_HASH_ENTRY**> [#uses=0]
%pawn_hash_mask = external global uint		; <uint*> [#uses=0]
%pawn_value_w = external global [64 x int]		; <[64 x int]*> [#uses=0]
%mask_pawn_isolated = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%mask_pawn_passed_w = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%mask_pawn_protected_w = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%pawn_value_b = external global [64 x int]		; <[64 x int]*> [#uses=0]
%mask_pawn_passed_b = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%mask_pawn_protected_b = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%unblocked_pawns = external global [9 x int]		; <[9 x int]*> [#uses=0]
%mask_wk_4th = external global ulong		; <ulong*> [#uses=0]
%mask_wk_5th = external global ulong		; <ulong*> [#uses=0]
%mask_wq_4th = external global ulong		; <ulong*> [#uses=0]
%mask_wq_5th = external global ulong		; <ulong*> [#uses=0]
%stonewall_white = external global ulong		; <ulong*> [#uses=0]
%mask_bk_4th = external global ulong		; <ulong*> [#uses=0]
%mask_bk_5th = external global ulong		; <ulong*> [#uses=0]
%mask_bq_5th = external global ulong		; <ulong*> [#uses=0]
%mask_bq_4th = external global ulong		; <ulong*> [#uses=0]
%stonewall_black = external global ulong		; <ulong*> [#uses=0]
%last_ones_8bit = external global [256 x ubyte]		; <[256 x ubyte]*> [#uses=0]
%right_side_mask = external global [8 x ulong]		; <[8 x ulong]*> [#uses=0]
%left_side_empty_mask = external global [8 x ulong]		; <[8 x ulong]*> [#uses=0]
%left_side_mask = external global [8 x ulong]		; <[8 x ulong]*> [#uses=0]
%right_side_empty_mask = external global [8 x ulong]		; <[8 x ulong]*> [#uses=0]
%pv = external global [65 x %typedef.CHESS_PATH]		; <[65 x %typedef.CHESS_PATH]*> [#uses=0]
%history_w = external global [4096 x int]		; <[4096 x int]*> [#uses=0]
%history_b = external global [4096 x int]		; <[4096 x int]*> [#uses=0]
%killer_move1 = external global [65 x int]		; <[65 x int]*> [#uses=0]
%killer_count1 = external global [65 x int]		; <[65 x int]*> [#uses=0]
%killer_move2 = external global [65 x int]		; <[65 x int]*> [#uses=0]
%killer_count2 = external global [65 x int]		; <[65 x int]*> [#uses=0]
%current_move = external global [65 x int]		; <[65 x int]*> [#uses=0]
%init_r90 = external global [64 x int]		; <[64 x int]*> [#uses=0]
%init_l90 = external global [64 x int]		; <[64 x int]*> [#uses=0]
%init_l45 = external global [64 x int]		; <[64 x int]*> [#uses=0]
%init_ul45 = external global [64 x int]		; <[64 x int]*> [#uses=0]
%init_r45 = external global [64 x int]		; <[64 x int]*> [#uses=0]
%init_ur45 = external global [64 x int]		; <[64 x int]*> [#uses=0]
%diagonal_length = external global [64 x int]		; <[64 x int]*> [#uses=0]
%last = external global [65 x int*]		; <[65 x int*]*> [#uses=0]
%move_list = external global [5120 x int]		; <[5120 x int]*> [#uses=0]
%history_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
%.str_1 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_2 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_3 = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%.str_5 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_6 = external global [14 x sbyte]		; <[14 x sbyte]*> [#uses=0]
%trans_ref_wa = external global %typedef.HASH_ENTRY*		; <%typedef.HASH_ENTRY**> [#uses=0]
%hash_table_size = external global int		; <int*> [#uses=0]
%trans_ref_wb = external global %typedef.HASH_ENTRY*		; <%typedef.HASH_ENTRY**> [#uses=0]
%trans_ref_ba = external global %typedef.HASH_ENTRY*		; <%typedef.HASH_ENTRY**> [#uses=0]
%trans_ref_bb = external global %typedef.HASH_ENTRY*		; <%typedef.HASH_ENTRY**> [#uses=0]
%pawn_hash_table_size = external global int		; <int*> [#uses=0]
%.str_9 = external global [37 x sbyte]		; <[37 x sbyte]*> [#uses=0]
%log_hash = external global int		; <int*> [#uses=0]
%log_pawn_hash = external global int		; <int*> [#uses=0]
%hash_maska = external global int		; <int*> [#uses=0]
%hash_maskb = external global int		; <int*> [#uses=0]
%mask_1 = external global ulong		; <ulong*> [#uses=0]
%bishop_attacks = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%queen_attacks = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%plus7dir = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%plus9dir = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%minus7dir = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%minus9dir = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%plus1dir = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%minus1dir = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%rook_attacks = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%king_attacks_1 = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%king_attacks_2 = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%.ctor_1 = external global [64 x int]		; <[64 x int]*> [#uses=0]
%.ctor_2 = external global [64 x int]		; <[64 x int]*> [#uses=0]
%rook_mobility_r0 = external global [64 x [256 x int]]		; <[64 x [256 x int]]*> [#uses=0]
%rook_mobility_rl90 = external global [64 x [256 x int]]		; <[64 x [256 x int]]*> [#uses=0]
%initial_position = external global [80 x sbyte]		; <[80 x sbyte]*> [#uses=5]
"a1.0__" = external global [80 x sbyte]		; <[80 x sbyte]*> [#uses=0]
"a2.1__" = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
"a3.2__" = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
"a4.3__" = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
"a5.4__" = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
"args.5__" = external global [16 x sbyte*]		; <[16 x sbyte*]*> [#uses=0]
%.str_10 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%w_pawn_random = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%w_pawn_random32 = external global [64 x uint]		; <[64 x uint]*> [#uses=0]
%b_pawn_random = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%b_pawn_random32 = external global [64 x uint]		; <[64 x uint]*> [#uses=0]
%w_knight_random = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%b_knight_random = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%w_bishop_random = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%b_bishop_random = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%w_rook_random = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%b_rook_random = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%w_queen_random = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%b_queen_random = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%w_king_random = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%b_king_random = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%enpassant_random = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%castle_random_w = external global [2 x ulong]		; <[2 x ulong]*> [#uses=0]
%castle_random_b = external global [2 x ulong]		; <[2 x ulong]*> [#uses=0]
%set_mask_rl90 = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%set_mask_rl45 = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%set_mask_rr45 = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%transposition_id = external global sbyte		; <sbyte*> [#uses=0]
%mask_2 = external global ulong		; <ulong*> [#uses=0]
%mask_3 = external global ulong		; <ulong*> [#uses=0]
%mask_4 = external global ulong		; <ulong*> [#uses=0]
%mask_8 = external global ulong		; <ulong*> [#uses=0]
%mask_16 = external global ulong		; <ulong*> [#uses=0]
%mask_32 = external global ulong		; <ulong*> [#uses=0]
%mask_72 = external global ulong		; <ulong*> [#uses=0]
%mask_80 = external global ulong		; <ulong*> [#uses=0]
%mask_85 = external global ulong		; <ulong*> [#uses=0]
%mask_96 = external global ulong		; <ulong*> [#uses=0]
%mask_107 = external global ulong		; <ulong*> [#uses=0]
%mask_108 = external global ulong		; <ulong*> [#uses=0]
%mask_112 = external global ulong		; <ulong*> [#uses=0]
%mask_118 = external global ulong		; <ulong*> [#uses=0]
%mask_120 = external global ulong		; <ulong*> [#uses=0]
%mask_121 = external global ulong		; <ulong*> [#uses=0]
%mask_127 = external global ulong		; <ulong*> [#uses=0]
%mask_clear_entry = external global ulong		; <ulong*> [#uses=0]
%clear_mask_rl45 = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%clear_mask_rr45 = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%clear_mask_rl90 = external global [65 x ulong]		; <[65 x ulong]*> [#uses=0]
%right_half_mask = external global ulong		; <ulong*> [#uses=0]
%left_half_mask = external global ulong		; <ulong*> [#uses=0]
%mask_not_rank8 = external global ulong		; <ulong*> [#uses=0]
%mask_not_rank1 = external global ulong		; <ulong*> [#uses=0]
%center = external global ulong		; <ulong*> [#uses=0]
%mask_pawn_connected = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%mask_eptest = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%mask_kingside_attack_w1 = external global ulong		; <ulong*> [#uses=0]
%mask_kingside_attack_w2 = external global ulong		; <ulong*> [#uses=0]
%mask_queenside_attack_w1 = external global ulong		; <ulong*> [#uses=0]
%mask_queenside_attack_w2 = external global ulong		; <ulong*> [#uses=0]
%mask_kingside_attack_b1 = external global ulong		; <ulong*> [#uses=0]
%mask_kingside_attack_b2 = external global ulong		; <ulong*> [#uses=0]
%mask_queenside_attack_b1 = external global ulong		; <ulong*> [#uses=0]
%mask_queenside_attack_b2 = external global ulong		; <ulong*> [#uses=0]
%pawns_cramp_black = external global ulong		; <ulong*> [#uses=0]
%pawns_cramp_white = external global ulong		; <ulong*> [#uses=0]
%light_squares = external global ulong		; <ulong*> [#uses=0]
%mask_left_edge = external global ulong		; <ulong*> [#uses=0]
%mask_right_edge = external global ulong		; <ulong*> [#uses=0]
%mask_advance_2_w = external global ulong		; <ulong*> [#uses=0]
%mask_advance_2_b = external global ulong		; <ulong*> [#uses=0]
%mask_corner_squares = external global ulong		; <ulong*> [#uses=0]
%mask_promotion_threat_w = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%mask_promotion_threat_b = external global [64 x ulong]		; <[64 x ulong]*> [#uses=0]
%promote_mask_w = external global ulong		; <ulong*> [#uses=0]
%promote_mask_b = external global ulong		; <ulong*> [#uses=0]
%mask_a1_corner = external global ulong		; <ulong*> [#uses=0]
%mask_h1_corner = external global ulong		; <ulong*> [#uses=0]
%mask_a8_corner = external global ulong		; <ulong*> [#uses=0]
%mask_h8_corner = external global ulong		; <ulong*> [#uses=0]
%white_center_pawns = external global ulong		; <ulong*> [#uses=0]
%black_center_pawns = external global ulong		; <ulong*> [#uses=0]
%wtm_random = external global [2 x ulong]		; <[2 x ulong]*> [#uses=0]
%endgame_random_w = external global ulong		; <ulong*> [#uses=0]
%endgame_random_b = external global ulong		; <ulong*> [#uses=0]
%w_rooks_random = external global ulong		; <ulong*> [#uses=0]
%b_rooks_random = external global ulong		; <ulong*> [#uses=0]
%.ctor_11 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.ctor_2 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.str_1 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_2 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_32 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_4 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_5 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_6 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_7 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_8 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_9 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_10 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_11 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_12 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_13 = external global [15 x sbyte]		; <[15 x sbyte]*> [#uses=0]
%num_ponder_moves = external global int		; <int*> [#uses=0]
%ponder_moves = external global [220 x int]		; <[220 x int]*> [#uses=0]
%.str_14 = external global [15 x sbyte]		; <[15 x sbyte]*> [#uses=0]
%.str_15 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_16 = external global [20 x sbyte]		; <[20 x sbyte]*> [#uses=0]
%auto232 = external global int		; <int*> [#uses=0]
%puzzling = external global sbyte		; <sbyte*> [#uses=0]
%abort_search = external global sbyte		; <sbyte*> [#uses=0]
%.str_24 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%wtm = external global int		; <int*> [#uses=0]
%.str_3 = external global [15 x sbyte]		; <[15 x sbyte]*> [#uses=0]
%.str_4 = external global [15 x sbyte]		; <[15 x sbyte]*> [#uses=0]
%end_time = external global uint		; <uint*> [#uses=0]
%time_type = external global uint		; <uint*> [#uses=0]
%start_time = external global uint		; <uint*> [#uses=0]
%.str_6 = external global [12 x sbyte]		; <[12 x sbyte]*> [#uses=0]
%.str_7 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%nodes_searched = external global uint		; <uint*> [#uses=0]
%iteration_depth = external global int		; <int*> [#uses=0]
%searched_this_root_move = external global [256 x sbyte]		; <[256 x sbyte]*> [#uses=0]
%.str_9 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_10 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_11 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_12 = external global [12 x sbyte]		; <[12 x sbyte]*> [#uses=0]
%.str_14 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_16 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%thinking = external global sbyte		; <sbyte*> [#uses=0]
%time_abort = external global int		; <int*> [#uses=0]
%.str_17 = external global [24 x sbyte]		; <[24 x sbyte]*> [#uses=0]
%analyze_move_read = external global int		; <int*> [#uses=0]
%analyze_mode = external global int		; <int*> [#uses=0]
%pondering = external global sbyte		; <sbyte*> [#uses=0]
%auto232_delay = external global int		; <int*> [#uses=0]
%auto_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
%.str_19 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_20 = external global [11 x sbyte]		; <[11 x sbyte]*> [#uses=0]
%.str_21 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%ponder_move = external global int		; <int*> [#uses=0]
%predicted = external global int		; <int*> [#uses=0]
%made_predicted_move = external global int		; <int*> [#uses=0]
%opponent_end_time = external global uint		; <uint*> [#uses=0]
%program_start_time = external global uint		; <uint*> [#uses=0]
%.str_23 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_24 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_25 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_26 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_28 = external global [31 x sbyte]		; <[31 x sbyte]*> [#uses=0]
%book_move = external global int		; <int*> [#uses=0]
%elapsed_start = external global uint		; <uint*> [#uses=0]
%burp = external global int		; <int*> [#uses=0]
%cpu_percent = external global int		; <int*> [#uses=0]
%next_time_check = external global int		; <int*> [#uses=0]
%nodes_between_time_checks = external global int		; <int*> [#uses=0]
%transposition_hits = external global int		; <int*> [#uses=0]
%transposition_probes = external global int		; <int*> [#uses=0]
%tb_probes = external global int		; <int*> [#uses=0]
%tb_probes_successful = external global int		; <int*> [#uses=0]
%check_extensions_done = external global int		; <int*> [#uses=0]
%recapture_extensions_done = external global int		; <int*> [#uses=0]
%passed_pawn_extensions_done = external global int		; <int*> [#uses=0]
%one_reply_extensions_done = external global int		; <int*> [#uses=0]
%program_end_time = external global uint		; <uint*> [#uses=0]
%root_value = external global int		; <int*> [#uses=0]
%last_search_value = external global int		; <int*> [#uses=0]
%.str_1 = external global [48 x sbyte]		; <[48 x sbyte]*> [#uses=0]
%.str_2 = external global [48 x sbyte]		; <[48 x sbyte]*> [#uses=0]
%booking = external global sbyte		; <sbyte*> [#uses=0]
%annotate_mode = external global int		; <int*> [#uses=0]
%.str_4 = external global [38 x sbyte]		; <[38 x sbyte]*> [#uses=0]
%.str_5 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%last_pv = external global %typedef.CHESS_PATH		; <%typedef.CHESS_PATH*> [#uses=0]
%.str_8 = external global [53 x sbyte]		; <[53 x sbyte]*> [#uses=0]
%root_alpha = external global int		; <int*> [#uses=0]
%last_value = external global int		; <int*> [#uses=0]
%root_beta = external global int		; <int*> [#uses=0]
%root_nodes = external global [256 x uint]		; <[256 x uint]*> [#uses=0]
%trace_level = external global int		; <int*> [#uses=0]
%.str_9 = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%.str_10 = external global [37 x sbyte]		; <[37 x sbyte]*> [#uses=0]
%search_failed_high = external global int		; <int*> [#uses=0]
%search_failed_low = external global int		; <int*> [#uses=0]
%nodes_per_second = external global int		; <int*> [#uses=0]
%time_limit = external global int		; <int*> [#uses=0]
%easy_move = external global int		; <int*> [#uses=0]
%noise_level = external global uint		; <uint*> [#uses=0]
%.str_12 = external global [34 x sbyte]		; <[34 x sbyte]*> [#uses=0]
%.str_136 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%solution_type = external global int		; <int*> [#uses=0]
%number_of_solutions = external global int		; <int*> [#uses=0]
%solutions = external global [10 x int]		; <[10 x int]*> [#uses=0]
%early_exit = external global int		; <int*> [#uses=0]
%.str_14 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_15 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_16 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%whisper_value = external global int		; <int*> [#uses=0]
%.str_17 = external global [29 x sbyte]		; <[29 x sbyte]*> [#uses=0]
%.str_19 = external global [37 x sbyte]		; <[37 x sbyte]*> [#uses=0]
%last_mate_score = external global int		; <int*> [#uses=0]
%search_depth = external global int		; <int*> [#uses=0]
%elapsed_end = external global uint		; <uint*> [#uses=0]
%.str_20 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_21 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_22 = external global [13 x sbyte]		; <[13 x sbyte]*> [#uses=0]
%.str_23 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_24 = external global [48 x sbyte]		; <[48 x sbyte]*> [#uses=0]
%.str_25 = external global [67 x sbyte]		; <[67 x sbyte]*> [#uses=0]
%.str_26 = external global [69 x sbyte]		; <[69 x sbyte]*> [#uses=0]
%hash_move = external global [65 x int]		; <[65 x int]*> [#uses=0]
%version = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%mode = external global uint		; <uint*> [#uses=0]
%batch_mode = external global int		; <int*> [#uses=0]
%crafty_rating = external global int		; <int*> [#uses=0]
%opponent_rating = external global int		; <int*> [#uses=0]
%pgn_event = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%pgn_site = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%pgn_date = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%pgn_round = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%pgn_white = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%pgn_white_elo = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%pgn_black = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%pgn_black_elo = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%pgn_result = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%number_auto_kibitzers = external global int		; <int*> [#uses=0]
%auto_kibitz_list = external global [100 x [20 x sbyte]]		; <[100 x [20 x sbyte]]*> [#uses=0]
%number_of_computers = external global int		; <int*> [#uses=0]
%computer_list = external global [100 x [20 x sbyte]]		; <[100 x [20 x sbyte]]*> [#uses=0]
%number_of_GMs = external global int		; <int*> [#uses=0]
%GM_list = external global [100 x [20 x sbyte]]		; <[100 x [20 x sbyte]]*> [#uses=0]
%number_of_IMs = external global int		; <int*> [#uses=0]
%IM_list = external global [100 x [20 x sbyte]]		; <[100 x [20 x sbyte]]*> [#uses=0]
%ics = external global int		; <int*> [#uses=0]
%output_format = external global int		; <int*> [#uses=0]
%EGTBlimit = external global int		; <int*> [#uses=0]
%whisper = external global int		; <int*> [#uses=0]
%channel = external global int		; <int*> [#uses=0]
%new_game = external global int		; <int*> [#uses=0]
%channel_title = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%initialized = external global int		; <int*> [#uses=0]
%kibitz = external global int		; <int*> [#uses=0]
%post = external global int		; <int*> [#uses=0]
%log_id = external global int		; <int*> [#uses=0]
%crafty_is_white = external global int		; <int*> [#uses=0]
%last_opponent_move = external global int		; <int*> [#uses=0]
%search_move = external global int		; <int*> [#uses=0]
%time_used = external global int		; <int*> [#uses=0]
%time_used_opponent = external global int		; <int*> [#uses=0]
%auto_kibitzing = external global int		; <int*> [#uses=0]
%test_mode = external global int		; <int*> [#uses=0]
%resign = external global sbyte		; <sbyte*> [#uses=0]
%resign_counter = external global sbyte		; <sbyte*> [#uses=0]
%resign_count = external global sbyte		; <sbyte*> [#uses=0]
%draw_counter = external global sbyte		; <sbyte*> [#uses=0]
%draw_count = external global sbyte		; <sbyte*> [#uses=0]
%tc_moves = external global int		; <int*> [#uses=0]
%tc_time = external global int		; <int*> [#uses=0]
%tc_time_remaining = external global int		; <int*> [#uses=0]
%tc_moves_remaining = external global int		; <int*> [#uses=0]
%tc_secondary_moves = external global int		; <int*> [#uses=0]
%tc_secondary_time = external global int		; <int*> [#uses=0]
%tc_sudden_death = external global int		; <int*> [#uses=0]
%tc_operator_time = external global int		; <int*> [#uses=0]
%tc_safety_margin = external global int		; <int*> [#uses=0]
%force = external global int		; <int*> [#uses=0]
%over = external global int		; <int*> [#uses=0]
%usage_level = external global int		; <int*> [#uses=0]
%audible_alarm = external global sbyte		; <sbyte*> [#uses=0]
%ansi = external global int		; <int*> [#uses=0]
%book_accept_mask = external global int		; <int*> [#uses=0]
%book_reject_mask = external global int		; <int*> [#uses=0]
%book_random = external global int		; <int*> [#uses=0]
%book_search_trigger = external global int		; <int*> [#uses=0]
%learning = external global int		; <int*> [#uses=0]
%show_book = external global int		; <int*> [#uses=0]
%book_selection_width = external global int		; <int*> [#uses=0]
%ponder = external global int		; <int*> [#uses=0]
%verbosity_level = external global int		; <int*> [#uses=0]
%push_extensions = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%.str_28 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_3 = external global [43 x sbyte]		; <[43 x sbyte]*> [#uses=0]
%display = external global %typedef.CHESS_POSITION		; <%typedef.CHESS_POSITION*> [#uses=0]
%.str_4 = external global [14 x sbyte]		; <[14 x sbyte]*> [#uses=0]
%opponent_start_time = external global uint		; <uint*> [#uses=0]
%.str_8 = external global [12 x sbyte]		; <[12 x sbyte]*> [#uses=0]
%.str_9 = external global [12 x sbyte]		; <[12 x sbyte]*> [#uses=0]
%.str_18 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_19 = external global [35 x sbyte]		; <[35 x sbyte]*> [#uses=0]
%.str_2013 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_21 = external global [41 x sbyte]		; <[41 x sbyte]*> [#uses=0]
%.str_22 = external global [29 x sbyte]		; <[29 x sbyte]*> [#uses=0]
%.str_23 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%whisper_text = external global [500 x sbyte]		; <[500 x sbyte]*> [#uses=0]
%.str_24 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_25 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_26 = external global [11 x sbyte]		; <[11 x sbyte]*> [#uses=0]
%.str_28 = external global [13 x sbyte]		; <[13 x sbyte]*> [#uses=0]
%.str_29 = external global [13 x sbyte]		; <[13 x sbyte]*> [#uses=0]
%.str_30 = external global [33 x sbyte]		; <[33 x sbyte]*> [#uses=0]
%.str_31 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_32 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_36 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=1]
%.str_37 = external global [15 x sbyte]		; <[15 x sbyte]*> [#uses=0]
%.str_44 = external global [12 x sbyte]		; <[12 x sbyte]*> [#uses=0]
%.str_45 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_49 = external global [15 x sbyte]		; <[15 x sbyte]*> [#uses=0]
%.str_52 = external global [12 x sbyte]		; <[12 x sbyte]*> [#uses=0]
%previous_search_value = external global int		; <int*> [#uses=0]
%.str_64 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%whisper_depth = external global int		; <int*> [#uses=0]
%.str_65 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_66 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%total_moves = external global int		; <int*> [#uses=0]
%book_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
%books_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
%book_lrn_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
%position_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
%position_lrn_file = external global %struct.__sFILE*		; <%struct.__sFILE**> [#uses=0]
%log_filename = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%history_filename = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%book_path = external global [128 x sbyte]		; <[128 x sbyte]*> [#uses=0]
%log_path = external global [128 x sbyte]		; <[128 x sbyte]*> [#uses=0]
%tb_path = external global [128 x sbyte]		; <[128 x sbyte]*> [#uses=0]
%cmd_buffer = external global [512 x sbyte]		; <[512 x sbyte]*> [#uses=0]
%root_move = external global int		; <int*> [#uses=0]
%hint = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
%absolute_time_limit = external global int		; <int*> [#uses=0]
%search_time_limit = external global int		; <int*> [#uses=0]
%in_check = external global [65 x sbyte]		; <[65 x sbyte]*> [#uses=0]
%extended_reason = external global [65 x sbyte]		; <[65 x sbyte]*> [#uses=0]
%current_phase = external global [65 x sbyte]		; <[65 x sbyte]*> [#uses=0]
%sort_value = external global [256 x int]		; <[256 x int]*> [#uses=0]
%next_status = external global [65 x %typedef.NEXT_MOVE]		; <[65 x %typedef.NEXT_MOVE]*> [#uses=0]
%save_hash_key = external global [67 x ulong]		; <[67 x ulong]*> [#uses=0]
%save_pawn_hash_key = external global [67 x uint]		; <[67 x uint]*> [#uses=0]
%pawn_advance = external global [8 x int]		; <[8 x int]*> [#uses=0]
%bit_move = external global ulong		; <ulong*> [#uses=0]
%.str_1 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.str_2 = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%.str_3 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_1 = external global [34 x sbyte]		; <[34 x sbyte]*> [#uses=0]
%.str_2 = external global [46 x sbyte]		; <[46 x sbyte]*> [#uses=0]
%.str_2 = external global [47 x sbyte]		; <[47 x sbyte]*> [#uses=0]
%.str_1 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_2 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_3 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_4 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_5 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_615 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_7 = external global [21 x sbyte]		; <[21 x sbyte]*> [#uses=0]
%.str_10 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_11 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_12 = external global [18 x sbyte]		; <[18 x sbyte]*> [#uses=0]
%.str_1318 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_1419 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_15 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.str_16 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_19 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_20 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_2222 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_2323 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_25 = external global [29 x sbyte]		; <[29 x sbyte]*> [#uses=0]
%.str_27 = external global [48 x sbyte]		; <[48 x sbyte]*> [#uses=0]
%.str_28 = external global [42 x sbyte]		; <[42 x sbyte]*> [#uses=0]
%.str_29 = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%.str_30 = external global [27 x sbyte]		; <[27 x sbyte]*> [#uses=0]
%.str_31 = external global [18 x sbyte]		; <[18 x sbyte]*> [#uses=0]
%.str_32 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_33 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_34 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_35 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_36 = external global [46 x sbyte]		; <[46 x sbyte]*> [#uses=0]
%.str_37 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_38 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_41 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_42 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_43 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_44 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_4525 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_46 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_47 = external global [19 x sbyte]		; <[19 x sbyte]*> [#uses=0]
%.str_48 = external global [35 x sbyte]		; <[35 x sbyte]*> [#uses=0]
%.str_49 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_50 = external global [19 x sbyte]		; <[19 x sbyte]*> [#uses=0]
%.str_51 = external global [25 x sbyte]		; <[25 x sbyte]*> [#uses=0]
%.str_52 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_53 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_54 = external global [23 x sbyte]		; <[23 x sbyte]*> [#uses=0]
%.str_55 = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%.str_56 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_57 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_58 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_59 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_60 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_61 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_62 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_63 = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
%.str_64 = external global [37 x sbyte]		; <[37 x sbyte]*> [#uses=0]
%.str_66 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_67 = external global [21 x sbyte]		; <[21 x sbyte]*> [#uses=0]
%.str_68 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_69 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_71 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_72 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_73 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_74 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_75 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_81 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_83 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_84 = external global [23 x sbyte]		; <[23 x sbyte]*> [#uses=0]
%.str_86 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_87 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_89 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_90 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_91 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_92 = external global [47 x sbyte]		; <[47 x sbyte]*> [#uses=0]
%.str_94 = external global [37 x sbyte]		; <[37 x sbyte]*> [#uses=0]
%.str_95 = external global [33 x sbyte]		; <[33 x sbyte]*> [#uses=0]
%.str_96 = external global [34 x sbyte]		; <[34 x sbyte]*> [#uses=0]
%.str_97 = external global [33 x sbyte]		; <[33 x sbyte]*> [#uses=0]
%.str_98 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_100 = external global [42 x sbyte]		; <[42 x sbyte]*> [#uses=0]
%.str_101 = external global [38 x sbyte]		; <[38 x sbyte]*> [#uses=0]
%.str_102 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_103 = external global [38 x sbyte]		; <[38 x sbyte]*> [#uses=0]
%.str_104 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_105 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_106 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_107 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_108 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_109 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_110 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_111 = external global [54 x sbyte]		; <[54 x sbyte]*> [#uses=0]
%.str_112 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_113 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_114 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_115 = external global [48 x sbyte]		; <[48 x sbyte]*> [#uses=0]
%.str_116 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_117 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_118 = external global [63 x sbyte]		; <[63 x sbyte]*> [#uses=0]
%.str_119 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_120 = external global [63 x sbyte]		; <[63 x sbyte]*> [#uses=0]
%.str_121 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_122 = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%.str_123 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_124 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_125 = external global [65 x sbyte]		; <[65 x sbyte]*> [#uses=0]
%.str_126 = external global [65 x sbyte]		; <[65 x sbyte]*> [#uses=0]
%.str_127 = external global [69 x sbyte]		; <[69 x sbyte]*> [#uses=0]
%.str_128 = external global [66 x sbyte]		; <[66 x sbyte]*> [#uses=0]
%.str_129 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_130 = external global [63 x sbyte]		; <[63 x sbyte]*> [#uses=0]
%.str_131 = external global [67 x sbyte]		; <[67 x sbyte]*> [#uses=0]
%.str_132 = external global [63 x sbyte]		; <[63 x sbyte]*> [#uses=0]
%.str_133 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_134 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_135 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_136 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_137 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_138 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_139 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_140 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_141 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_142 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_143 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_144 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_145 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_146 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_147 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_148 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_149 = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%.str_150 = external global [65 x sbyte]		; <[65 x sbyte]*> [#uses=0]
%.str_151 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_152 = external global [46 x sbyte]		; <[46 x sbyte]*> [#uses=0]
%.str_153 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_154 = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%.str_156 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_157 = external global [31 x sbyte]		; <[31 x sbyte]*> [#uses=0]
%.str_158 = external global [71 x sbyte]		; <[71 x sbyte]*> [#uses=0]
%.str_159 = external global [72 x sbyte]		; <[72 x sbyte]*> [#uses=0]
%.str_160 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_161 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_162 = external global [63 x sbyte]		; <[63 x sbyte]*> [#uses=0]
%.str_163 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_164 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_165 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_166 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_167 = external global [63 x sbyte]		; <[63 x sbyte]*> [#uses=0]
%.str_168 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_169 = external global [65 x sbyte]		; <[65 x sbyte]*> [#uses=0]
%.str_170 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_171 = external global [63 x sbyte]		; <[63 x sbyte]*> [#uses=0]
%.str_172 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_173 = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%.str_174 = external global [51 x sbyte]		; <[51 x sbyte]*> [#uses=0]
%.str_175 = external global [70 x sbyte]		; <[70 x sbyte]*> [#uses=0]
%.str_176 = external global [67 x sbyte]		; <[67 x sbyte]*> [#uses=0]
%.str_177 = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%.str_178 = external global [48 x sbyte]		; <[48 x sbyte]*> [#uses=0]
%.str_180 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_181 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_182 = external global [53 x sbyte]		; <[53 x sbyte]*> [#uses=0]
%.str_183 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_184 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_185 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_186 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_187 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_188 = external global [35 x sbyte]		; <[35 x sbyte]*> [#uses=0]
%.str_189 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_190 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_191 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_192 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_193 = external global [55 x sbyte]		; <[55 x sbyte]*> [#uses=0]
%.str_194 = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%.str_195 = external global [33 x sbyte]		; <[33 x sbyte]*> [#uses=0]
%.str_196 = external global [46 x sbyte]		; <[46 x sbyte]*> [#uses=0]
%.str_197 = external global [11 x sbyte]		; <[11 x sbyte]*> [#uses=0]
%.str_198 = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
%.str_201 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_202 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_203 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_204 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_206 = external global [48 x sbyte]		; <[48 x sbyte]*> [#uses=0]
%.str_207 = external global [46 x sbyte]		; <[46 x sbyte]*> [#uses=0]
%.str_208 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_209 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_210 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_211 = external global [43 x sbyte]		; <[43 x sbyte]*> [#uses=0]
%.str_213 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_214 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_215 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_216 = external global [14 x sbyte]		; <[14 x sbyte]*> [#uses=0]
%.str_218 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_219 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_220 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_221 = external global [37 x sbyte]		; <[37 x sbyte]*> [#uses=0]
%.str_222 = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%.str_223 = external global [66 x sbyte]		; <[66 x sbyte]*> [#uses=0]
%.str_224 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_225 = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%.str_226 = external global [63 x sbyte]		; <[63 x sbyte]*> [#uses=0]
%.str_227 = external global [46 x sbyte]		; <[46 x sbyte]*> [#uses=0]
%.str_228 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_229 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_230 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_231 = external global [19 x sbyte]		; <[19 x sbyte]*> [#uses=0]
%.str_232 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_233 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_234 = external global [54 x sbyte]		; <[54 x sbyte]*> [#uses=0]
%.str_235 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_236 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.str_237 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_238 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_239 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_240 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.str_241 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_242 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_243 = external global [35 x sbyte]		; <[35 x sbyte]*> [#uses=0]
%.str_245 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_246 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_247 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_248 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_249 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_250 = external global [45 x sbyte]		; <[45 x sbyte]*> [#uses=0]
%.str_253 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_254 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.str_256 = external global [43 x sbyte]		; <[43 x sbyte]*> [#uses=0]
%.str_258 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_259 = external global [27 x sbyte]		; <[27 x sbyte]*> [#uses=0]
%.str_261 = external global [43 x sbyte]		; <[43 x sbyte]*> [#uses=0]
%.str_262 = external global [43 x sbyte]		; <[43 x sbyte]*> [#uses=0]
%.str_263 = external global [43 x sbyte]		; <[43 x sbyte]*> [#uses=0]
%.str_266 = external global [43 x sbyte]		; <[43 x sbyte]*> [#uses=0]
%.str_267 = external global [43 x sbyte]		; <[43 x sbyte]*> [#uses=0]
%.str_268 = external global [43 x sbyte]		; <[43 x sbyte]*> [#uses=0]
%.str_270 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_271 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_272 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_273 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_274 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_275 = external global [44 x sbyte]		; <[44 x sbyte]*> [#uses=0]
%.str_276 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_277 = external global [21 x sbyte]		; <[21 x sbyte]*> [#uses=0]
%.str_278 = external global [48 x sbyte]		; <[48 x sbyte]*> [#uses=0]
%.str_279 = external global [51 x sbyte]		; <[51 x sbyte]*> [#uses=0]
%.str_280 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_281 = external global [53 x sbyte]		; <[53 x sbyte]*> [#uses=0]
%.str_282 = external global [50 x sbyte]		; <[50 x sbyte]*> [#uses=0]
%.str_283 = external global [53 x sbyte]		; <[53 x sbyte]*> [#uses=0]
%.str_284 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_285 = external global [55 x sbyte]		; <[55 x sbyte]*> [#uses=0]
%.str_286 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_287 = external global [54 x sbyte]		; <[54 x sbyte]*> [#uses=0]
%.str_288 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_289 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_290 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_291 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_292 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_293 = external global [54 x sbyte]		; <[54 x sbyte]*> [#uses=0]
%.str_294 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_295 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_296 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_297 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_298 = external global [51 x sbyte]		; <[51 x sbyte]*> [#uses=0]
%.str_299 = external global [54 x sbyte]		; <[54 x sbyte]*> [#uses=0]
%.str_300 = external global [49 x sbyte]		; <[49 x sbyte]*> [#uses=0]
%.str_301 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_302 = external global [18 x sbyte]		; <[18 x sbyte]*> [#uses=0]
%.str_304 = external global [50 x sbyte]		; <[50 x sbyte]*> [#uses=0]
%.str_305 = external global [50 x sbyte]		; <[50 x sbyte]*> [#uses=0]
%.str_306 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_308 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_310 = external global [50 x sbyte]		; <[50 x sbyte]*> [#uses=0]
%.str_311 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_312 = external global [53 x sbyte]		; <[53 x sbyte]*> [#uses=0]
%.str_313 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_314 = external global [53 x sbyte]		; <[53 x sbyte]*> [#uses=0]
%.str_315 = external global [54 x sbyte]		; <[54 x sbyte]*> [#uses=0]
%.str_316 = external global [51 x sbyte]		; <[51 x sbyte]*> [#uses=0]
%.str_317 = external global [26 x sbyte]		; <[26 x sbyte]*> [#uses=0]
%.str_319 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_320 = external global [53 x sbyte]		; <[53 x sbyte]*> [#uses=0]
%.str_321 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_322 = external global [55 x sbyte]		; <[55 x sbyte]*> [#uses=0]
%.str_323 = external global [20 x sbyte]		; <[20 x sbyte]*> [#uses=0]
%.str_325 = external global [26 x sbyte]		; <[26 x sbyte]*> [#uses=0]
%.str_327 = external global [51 x sbyte]		; <[51 x sbyte]*> [#uses=0]
%.str_328 = external global [51 x sbyte]		; <[51 x sbyte]*> [#uses=0]
%.str_329 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_330 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_331 = external global [53 x sbyte]		; <[53 x sbyte]*> [#uses=0]
%.str_332 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_333 = external global [50 x sbyte]		; <[50 x sbyte]*> [#uses=0]
%.str_334 = external global [54 x sbyte]		; <[54 x sbyte]*> [#uses=0]
%.str_335 = external global [48 x sbyte]		; <[48 x sbyte]*> [#uses=0]
%.str_336 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_337 = external global [50 x sbyte]		; <[50 x sbyte]*> [#uses=0]
%.str_338 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_339 = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%.str_340 = external global [55 x sbyte]		; <[55 x sbyte]*> [#uses=0]
%.str_341 = external global [54 x sbyte]		; <[54 x sbyte]*> [#uses=0]
%.str_342 = external global [49 x sbyte]		; <[49 x sbyte]*> [#uses=0]
%.str_343 = external global [63 x sbyte]		; <[63 x sbyte]*> [#uses=0]
%.str_344 = external global [49 x sbyte]		; <[49 x sbyte]*> [#uses=0]
%.str_345 = external global [47 x sbyte]		; <[47 x sbyte]*> [#uses=0]
%.str_346 = external global [49 x sbyte]		; <[49 x sbyte]*> [#uses=0]
%.str_347 = external global [43 x sbyte]		; <[43 x sbyte]*> [#uses=0]
%.str_348 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_349 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_350 = external global [55 x sbyte]		; <[55 x sbyte]*> [#uses=0]
%.str_351 = external global [47 x sbyte]		; <[47 x sbyte]*> [#uses=0]
%.str_352 = external global [49 x sbyte]		; <[49 x sbyte]*> [#uses=0]
%.str_353 = external global [49 x sbyte]		; <[49 x sbyte]*> [#uses=0]
%.str_354 = external global [42 x sbyte]		; <[42 x sbyte]*> [#uses=0]
%.str_355 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_356 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_357 = external global [53 x sbyte]		; <[53 x sbyte]*> [#uses=0]
%.str_358 = external global [47 x sbyte]		; <[47 x sbyte]*> [#uses=0]
%.str_359 = external global [63 x sbyte]		; <[63 x sbyte]*> [#uses=0]
%.str_360 = external global [54 x sbyte]		; <[54 x sbyte]*> [#uses=0]
%.str_362 = external global [53 x sbyte]		; <[53 x sbyte]*> [#uses=0]
%.str_363 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_364 = external global [56 x sbyte]		; <[56 x sbyte]*> [#uses=0]
%.str_365 = external global [51 x sbyte]		; <[51 x sbyte]*> [#uses=0]
%.str_366 = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=0]
%.str_367 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_368 = external global [48 x sbyte]		; <[48 x sbyte]*> [#uses=0]
%.str_369 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_370 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_371 = external global [49 x sbyte]		; <[49 x sbyte]*> [#uses=0]
%.str_372 = external global [58 x sbyte]		; <[58 x sbyte]*> [#uses=0]
%.str_373 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_374 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_375 = external global [63 x sbyte]		; <[63 x sbyte]*> [#uses=0]
%.str_376 = external global [63 x sbyte]		; <[63 x sbyte]*> [#uses=0]
%.str_377 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_378 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_379 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_380 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_381 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_382 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_383 = external global [49 x sbyte]		; <[49 x sbyte]*> [#uses=0]
%.str_384 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_385 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_387 = external global [53 x sbyte]		; <[53 x sbyte]*> [#uses=0]
%.str_388 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_389 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_390 = external global [55 x sbyte]		; <[55 x sbyte]*> [#uses=0]
%.str_391 = external global [55 x sbyte]		; <[55 x sbyte]*> [#uses=0]
%.str_392 = external global [71 x sbyte]		; <[71 x sbyte]*> [#uses=0]
%.str_393 = external global [46 x sbyte]		; <[46 x sbyte]*> [#uses=0]
%.str_394 = external global [51 x sbyte]		; <[51 x sbyte]*> [#uses=0]
%.str_395 = external global [54 x sbyte]		; <[54 x sbyte]*> [#uses=0]
%.str_396 = external global [60 x sbyte]		; <[60 x sbyte]*> [#uses=0]
%.str_397 = external global [49 x sbyte]		; <[49 x sbyte]*> [#uses=0]
%.str_398 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_399 = external global [67 x sbyte]		; <[67 x sbyte]*> [#uses=0]
%.str_400 = external global [55 x sbyte]		; <[55 x sbyte]*> [#uses=0]
%.str_401 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_402 = external global [62 x sbyte]		; <[62 x sbyte]*> [#uses=0]
%.str_403 = external global [47 x sbyte]		; <[47 x sbyte]*> [#uses=0]
%.str_404 = external global [59 x sbyte]		; <[59 x sbyte]*> [#uses=0]
%.str_405 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_406 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_407 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_408 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_409 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_410 = external global [26 x sbyte]		; <[26 x sbyte]*> [#uses=0]
%.str_411 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_412 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_413 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_414 = external global [19 x sbyte]		; <[19 x sbyte]*> [#uses=0]
%.str_421 = external global [53 x sbyte]		; <[53 x sbyte]*> [#uses=0]
%.str_422 = external global [55 x sbyte]		; <[55 x sbyte]*> [#uses=0]
%.str_423 = external global [23 x sbyte]		; <[23 x sbyte]*> [#uses=0]
%.str_424 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_426 = external global [42 x sbyte]		; <[42 x sbyte]*> [#uses=0]
%.str_427 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_429 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_430 = external global [24 x sbyte]		; <[24 x sbyte]*> [#uses=0]
%.str_431 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_432 = external global [23 x sbyte]		; <[23 x sbyte]*> [#uses=0]
%.str_433 = external global [23 x sbyte]		; <[23 x sbyte]*> [#uses=0]
%.str_434 = external global [24 x sbyte]		; <[24 x sbyte]*> [#uses=0]
%.str_435 = external global [27 x sbyte]		; <[27 x sbyte]*> [#uses=0]
%.str_436 = external global [28 x sbyte]		; <[28 x sbyte]*> [#uses=0]
%.str_437 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_438 = external global [38 x sbyte]		; <[38 x sbyte]*> [#uses=0]
%.str_440 = external global [44 x sbyte]		; <[44 x sbyte]*> [#uses=0]
%.str_445 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_446 = external global [26 x sbyte]		; <[26 x sbyte]*> [#uses=0]
%.str_447 = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%.str_448 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_449 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_450 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_451 = external global [44 x sbyte]		; <[44 x sbyte]*> [#uses=0]
%.str_452 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_453 = external global [26 x sbyte]		; <[26 x sbyte]*> [#uses=0]
%.str_454 = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%.str_455 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_456 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_459 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_460 = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%.str_461 = external global [42 x sbyte]		; <[42 x sbyte]*> [#uses=0]
%.str_462 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_463 = external global [28 x sbyte]		; <[28 x sbyte]*> [#uses=0]
%.str_466 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_467 = external global [35 x sbyte]		; <[35 x sbyte]*> [#uses=0]
%.str_468 = external global [45 x sbyte]		; <[45 x sbyte]*> [#uses=0]
%.str_469 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_470 = external global [31 x sbyte]		; <[31 x sbyte]*> [#uses=0]
%.str_474 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_477 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_480 = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
%.str_483 = external global [19 x sbyte]		; <[19 x sbyte]*> [#uses=0]
%.str_485 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_487 = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%.str_490 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_494 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_495 = external global [33 x sbyte]		; <[33 x sbyte]*> [#uses=0]
%.str_497 = external global [12 x sbyte]		; <[12 x sbyte]*> [#uses=0]
%.str_498 = external global [13 x sbyte]		; <[13 x sbyte]*> [#uses=0]
%.str_507 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_508 = external global [11 x sbyte]		; <[11 x sbyte]*> [#uses=0]
%.str_509 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_510 = external global [37 x sbyte]		; <[37 x sbyte]*> [#uses=0]
%.str_511 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_512 = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%.str_513 = external global [18 x sbyte]		; <[18 x sbyte]*> [#uses=0]
%.str_514 = external global [14 x sbyte]		; <[14 x sbyte]*> [#uses=0]
%.str_515 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_516 = external global [21 x sbyte]		; <[21 x sbyte]*> [#uses=0]
%.str_517 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_519 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.str_520 = external global [21 x sbyte]		; <[21 x sbyte]*> [#uses=0]
%.str_521 = external global [15 x sbyte]		; <[15 x sbyte]*> [#uses=0]
%.str_522 = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
%.str_523 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_524 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_525 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_526 = external global [19 x sbyte]		; <[19 x sbyte]*> [#uses=0]
%.str_527 = external global [24 x sbyte]		; <[24 x sbyte]*> [#uses=0]
%.str_528 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_529 = external global [28 x sbyte]		; <[28 x sbyte]*> [#uses=0]
%.str_530 = external global [54 x sbyte]		; <[54 x sbyte]*> [#uses=0]
%.str_531 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_532 = external global [38 x sbyte]		; <[38 x sbyte]*> [#uses=0]
%.str_533 = external global [32 x sbyte]		; <[32 x sbyte]*> [#uses=0]
%.str_534 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_535 = external global [27 x sbyte]		; <[27 x sbyte]*> [#uses=0]
%.str_536 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_537 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_539 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_540 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_541 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_542 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_543 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_544 = external global [49 x sbyte]		; <[49 x sbyte]*> [#uses=0]
%.str_546 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_550 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_551 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_552 = external global [31 x sbyte]		; <[31 x sbyte]*> [#uses=0]
%.str_553 = external global [52 x sbyte]		; <[52 x sbyte]*> [#uses=0]
%.str_554 = external global [43 x sbyte]		; <[43 x sbyte]*> [#uses=0]
%.str_555 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_556 = external global [26 x sbyte]		; <[26 x sbyte]*> [#uses=0]
%.str_557 = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
%.str_559 = external global [27 x sbyte]		; <[27 x sbyte]*> [#uses=0]
%.str_560 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_562 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_564 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_565 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_567 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_568 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_570 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_571 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_572 = external global [30 x sbyte]		; <[30 x sbyte]*> [#uses=0]
%.str_574 = external global [20 x sbyte]		; <[20 x sbyte]*> [#uses=0]
%.str_576 = external global [21 x sbyte]		; <[21 x sbyte]*> [#uses=0]
%.str_577 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_578 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_579 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_580 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_581 = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%.str_582 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_583 = external global [24 x sbyte]		; <[24 x sbyte]*> [#uses=0]
%.str_584 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_586 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_587 = external global [28 x sbyte]		; <[28 x sbyte]*> [#uses=0]
%.str_589 = external global [28 x sbyte]		; <[28 x sbyte]*> [#uses=0]
%.str_590 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_591 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_592 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_596 = external global [25 x sbyte]		; <[25 x sbyte]*> [#uses=0]
%.str_597 = external global [27 x sbyte]		; <[27 x sbyte]*> [#uses=0]
%.str_598 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.str_599 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.str_605 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_610 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_613 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_616 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_621 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_622 = external global [24 x sbyte]		; <[24 x sbyte]*> [#uses=0]
%.str_623 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_624 = external global [24 x sbyte]		; <[24 x sbyte]*> [#uses=0]
%.str_625 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_626 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_628 = external global [30 x sbyte]		; <[30 x sbyte]*> [#uses=0]
%.str_629 = external global [14 x sbyte]		; <[14 x sbyte]*> [#uses=0]
%.str_630 = external global [13 x sbyte]		; <[13 x sbyte]*> [#uses=0]
%.str_631 = external global [24 x sbyte]		; <[24 x sbyte]*> [#uses=0]
%.str_632 = external global [14 x sbyte]		; <[14 x sbyte]*> [#uses=0]
%.str_633 = external global [14 x sbyte]		; <[14 x sbyte]*> [#uses=0]
%.str_634 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.str_635 = external global [14 x sbyte]		; <[14 x sbyte]*> [#uses=0]
%.str_636 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.str_637 = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
%.str_639 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_643 = external global [20 x sbyte]		; <[20 x sbyte]*> [#uses=0]
%.str_644 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_645 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_646 = external global [1 x sbyte]		; <[1 x sbyte]*> [#uses=0]
%.str_649 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_654 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=1]
%.str_656 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_658 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_660 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_662 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_664 = external global [2 x sbyte]		; <[2 x sbyte]*> [#uses=0]
%.str_666 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_667 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_669 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_670 = external global [20 x sbyte]		; <[20 x sbyte]*> [#uses=0]
%.str_671 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_672 = external global [23 x sbyte]		; <[23 x sbyte]*> [#uses=0]
%.str_674 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_675 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_676 = external global [30 x sbyte]		; <[30 x sbyte]*> [#uses=0]
%.str_680 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_682 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_683 = external global [37 x sbyte]		; <[37 x sbyte]*> [#uses=0]
%.str_684 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_685 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_686 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_687 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_688 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_689 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_690 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_691 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_692 = external global [20 x sbyte]		; <[20 x sbyte]*> [#uses=0]
%.str_694 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_695 = external global [19 x sbyte]		; <[19 x sbyte]*> [#uses=0]
%.str_697 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_698 = external global [28 x sbyte]		; <[28 x sbyte]*> [#uses=0]
%.str_700 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_701 = external global [19 x sbyte]		; <[19 x sbyte]*> [#uses=0]
%.str_702 = external global [26 x sbyte]		; <[26 x sbyte]*> [#uses=0]
%.str_703 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_704 = external global [25 x sbyte]		; <[25 x sbyte]*> [#uses=0]
%.str_707 = external global [4 x sbyte]		; <[4 x sbyte]*> [#uses=0]
%.str_708 = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
%.str_709 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_710 = external global [20 x sbyte]		; <[20 x sbyte]*> [#uses=0]
%.str_711 = external global [37 x sbyte]		; <[37 x sbyte]*> [#uses=0]
%.str_722 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_723 = external global [34 x sbyte]		; <[34 x sbyte]*> [#uses=0]
%.str_726 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_727 = external global [23 x sbyte]		; <[23 x sbyte]*> [#uses=0]
%.str_728 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_729 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_730 = external global [12 x sbyte]		; <[12 x sbyte]*> [#uses=0]
%.str_732 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_734 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_735 = external global [28 x sbyte]		; <[28 x sbyte]*> [#uses=0]
%.str_736 = external global [61 x sbyte]		; <[61 x sbyte]*> [#uses=0]
%.str_738 = external global [25 x sbyte]		; <[25 x sbyte]*> [#uses=0]
%.str_739 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_740 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_741 = external global [25 x sbyte]		; <[25 x sbyte]*> [#uses=0]
%.str_742 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_743 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_744 = external global [31 x sbyte]		; <[31 x sbyte]*> [#uses=0]
%.str_745 = external global [42 x sbyte]		; <[42 x sbyte]*> [#uses=0]
%.str_747 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_748 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_750 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
"text_move.0__" = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
"new_text.1__" = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_1 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
"text_move.2__" = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_1 = external global [15 x sbyte]		; <[15 x sbyte]*> [#uses=0]
%.str_2 = external global [19 x sbyte]		; <[19 x sbyte]*> [#uses=0]
%.str_3 = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
%.str_130 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_231 = external global [47 x sbyte]		; <[47 x sbyte]*> [#uses=0]
%.str_3 = external global [27 x sbyte]		; <[27 x sbyte]*> [#uses=0]
%.str_4 = external global [27 x sbyte]		; <[27 x sbyte]*> [#uses=0]
%.str_7 = external global [30 x sbyte]		; <[30 x sbyte]*> [#uses=0]
"hashing_pawns.0__" = external global int		; <int*> [#uses=0]
"hashing_opening.1__" = external global int		; <int*> [#uses=0]
"hashing_middle_game.2__" = external global int		; <int*> [#uses=0]
"hashing_end_game.3__" = external global int		; <int*> [#uses=0]
"last_wtm.4__" = external global int		; <int*> [#uses=0]
%.str_1 = external global [37 x sbyte]		; <[37 x sbyte]*> [#uses=0]
%.str_1 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_1 = external global [19 x sbyte]		; <[19 x sbyte]*> [#uses=0]
%.str_2 = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
%.str_3 = external global [30 x sbyte]		; <[30 x sbyte]*> [#uses=0]
%.str_4 = external global [30 x sbyte]		; <[30 x sbyte]*> [#uses=0]
%.str_5 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_6 = external global [25 x sbyte]		; <[25 x sbyte]*> [#uses=0]
%.str_7 = external global [14 x sbyte]		; <[14 x sbyte]*> [#uses=0]
%.str_934 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_1 = external global [28 x sbyte]		; <[28 x sbyte]*> [#uses=0]
%.str_3 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_1 = external global [38 x sbyte]		; <[38 x sbyte]*> [#uses=0]
%.str_2 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_4 = external global [28 x sbyte]		; <[28 x sbyte]*> [#uses=0]
%.str_5 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_1 = external global [11 x sbyte]		; <[11 x sbyte]*> [#uses=0]
%.str_2 = external global [27 x sbyte]		; <[27 x sbyte]*> [#uses=0]
%.str_7 = external global [29 x sbyte]		; <[29 x sbyte]*> [#uses=0]
%.str_10 = external global [34 x sbyte]		; <[34 x sbyte]*> [#uses=0]
%.str_1141 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_12 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_14 = external global [20 x sbyte]		; <[20 x sbyte]*> [#uses=0]
%.str_1542 = external global [17 x sbyte]		; <[17 x sbyte]*> [#uses=0]
%.ctor_1 = external global [25 x sbyte]		; <[25 x sbyte]*> [#uses=0]
%.str_1 = external global [33 x sbyte]		; <[33 x sbyte]*> [#uses=0]
%.str_3 = external global [21 x sbyte]		; <[21 x sbyte]*> [#uses=0]
%.str_4 = external global [25 x sbyte]		; <[25 x sbyte]*> [#uses=0]
%.str_5 = external global [26 x sbyte]		; <[26 x sbyte]*> [#uses=0]
%.str_6 = external global [55 x sbyte]		; <[55 x sbyte]*> [#uses=0]
%.str_143 = external global [33 x sbyte]		; <[33 x sbyte]*> [#uses=0]
%.str_2 = external global [15 x sbyte]		; <[15 x sbyte]*> [#uses=0]
%.str_1 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_2 = external global [51 x sbyte]		; <[51 x sbyte]*> [#uses=0]
%bit_move44 = external global ulong		; <ulong*> [#uses=0]
%.str_1 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_248 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_349 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.ctor_1 = external global [46 x sbyte]		; <[46 x sbyte]*> [#uses=0]
%.str_5 = external global [43 x sbyte]		; <[43 x sbyte]*> [#uses=0]
%.str_6 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_751 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_852 = external global [3 x sbyte]		; <[3 x sbyte]*> [#uses=0]
%.str_9 = external global [42 x sbyte]		; <[42 x sbyte]*> [#uses=0]
%.str_10 = external global [41 x sbyte]		; <[41 x sbyte]*> [#uses=0]
"out.0__" = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_1153 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_12 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_13 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_14 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_15 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_16 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_17 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
"out.1__" = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_18 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_19 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_20 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_21 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_2254 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_2355 = external global [8 x sbyte]		; <[8 x sbyte]*> [#uses=0]
%.str_24 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]
%.str_25 = external global [45 x sbyte]		; <[45 x sbyte]*> [#uses=0]
%.str_26 = external global [24 x sbyte]		; <[24 x sbyte]*> [#uses=0]
"out.2__" = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_31 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
"out.3__" = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
"out.4__" = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_3457 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_35 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_36 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_37 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_41 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_45 = external global [55 x sbyte]		; <[55 x sbyte]*> [#uses=0]
"save_book_selection_width.5__" = external global int		; <int*> [#uses=0]
"save_book_random.6__" = external global int		; <int*> [#uses=0]
"save_whisper.7__" = external global int		; <int*> [#uses=0]
"save_kibitz.8__" = external global int		; <int*> [#uses=0]
"save_channel.9__" = external global int		; <int*> [#uses=0]
"save_resign.10" = external global int		; <int*> [#uses=0]
"save_resign_count.11" = external global int		; <int*> [#uses=0]
"save_draw_count.12" = external global int		; <int*> [#uses=0]
"save_learning.13" = external global int		; <int*> [#uses=0]
%.str_49 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_51 = external global [44 x sbyte]		; <[44 x sbyte]*> [#uses=0]
"x.14" = external global [55 x uint]		; <[55 x uint]*> [#uses=0]
"init.15.b" = external global bool		; <bool*> [#uses=0]
"y.16" = external global [55 x uint]		; <[55 x uint]*> [#uses=0]
"j.17" = external global int		; <int*> [#uses=0]
"k.18" = external global int		; <int*> [#uses=0]
%.str_52 = external global [50 x sbyte]		; <[50 x sbyte]*> [#uses=0]
"text.19" = external global [128 x sbyte]		; <[128 x sbyte]*> [#uses=0]
%.str_5659 = external global [12 x sbyte]		; <[12 x sbyte]*> [#uses=0]
%.str_62 = external global [14 x sbyte]		; <[14 x sbyte]*> [#uses=0]
%.str_6662 = external global [5 x sbyte]		; <[5 x sbyte]*> [#uses=0]
%.str_68 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_69 = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
%.str_70 = external global [16 x sbyte]		; <[16 x sbyte]*> [#uses=0]
%.str_72 = external global [35 x sbyte]		; <[35 x sbyte]*> [#uses=0]
%.str_74 = external global [23 x sbyte]		; <[23 x sbyte]*> [#uses=0]
%.str_76 = external global [24 x sbyte]		; <[24 x sbyte]*> [#uses=0]
%.str_78 = external global [57 x sbyte]		; <[57 x sbyte]*> [#uses=0]
%.str_80 = external global [45 x sbyte]		; <[45 x sbyte]*> [#uses=0]
%.str_82 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_84 = external global [10 x sbyte]		; <[10 x sbyte]*> [#uses=0]
%.str_86 = external global [19 x sbyte]		; <[19 x sbyte]*> [#uses=0]
%.str_88 = external global [7 x sbyte]		; <[7 x sbyte]*> [#uses=0]
%.str_90 = external global [31 x sbyte]		; <[31 x sbyte]*> [#uses=0]
%.str_92 = external global [19 x sbyte]		; <[19 x sbyte]*> [#uses=0]
%.str_94 = external global [30 x sbyte]		; <[30 x sbyte]*> [#uses=0]
%.str_95 = external global [48 x sbyte]		; <[48 x sbyte]*> [#uses=0]
%.str_97 = external global [18 x sbyte]		; <[18 x sbyte]*> [#uses=0]
%.str_98 = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%.str_100 = external global [22 x sbyte]		; <[22 x sbyte]*> [#uses=0]
%.str_163 = external global [38 x sbyte]		; <[38 x sbyte]*> [#uses=0]
%.str_2 = external global [38 x sbyte]		; <[38 x sbyte]*> [#uses=0]
%.str_3 = external global [50 x sbyte]		; <[50 x sbyte]*> [#uses=0]
%.str_4 = external global [50 x sbyte]		; <[50 x sbyte]*> [#uses=0]
%.str_5 = external global [51 x sbyte]		; <[51 x sbyte]*> [#uses=0]
%.str_6 = external global [30 x sbyte]		; <[30 x sbyte]*> [#uses=0]
%.str_7 = external global [28 x sbyte]		; <[28 x sbyte]*> [#uses=0]
%.str_8 = external global [33 x sbyte]		; <[33 x sbyte]*> [#uses=0]
%.str_9 = external global [54 x sbyte]		; <[54 x sbyte]*> [#uses=0]
%.str_10 = external global [47 x sbyte]		; <[47 x sbyte]*> [#uses=0]
%.str_11 = external global [46 x sbyte]		; <[46 x sbyte]*> [#uses=0]
%.str_12 = external global [47 x sbyte]		; <[47 x sbyte]*> [#uses=0]
%.str_13 = external global [46 x sbyte]		; <[46 x sbyte]*> [#uses=0]
%.str_14 = external global [35 x sbyte]		; <[35 x sbyte]*> [#uses=0]
%.str_15 = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%.str_16 = external global [35 x sbyte]		; <[35 x sbyte]*> [#uses=0]
%.str_17 = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%.str_18 = external global [35 x sbyte]		; <[35 x sbyte]*> [#uses=0]
%.str_19 = external global [41 x sbyte]		; <[41 x sbyte]*> [#uses=0]
%.str_20 = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%.str_22 = external global [35 x sbyte]		; <[35 x sbyte]*> [#uses=0]
%.str_23 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_24 = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%.str_26 = external global [35 x sbyte]		; <[35 x sbyte]*> [#uses=0]
%.str_27 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_28 = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%.str_30 = external global [35 x sbyte]		; <[35 x sbyte]*> [#uses=0]
%.str_31 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_32 = external global [36 x sbyte]		; <[36 x sbyte]*> [#uses=0]
%.str_33 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_34 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_3565 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_36 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_37 = external global [41 x sbyte]		; <[41 x sbyte]*> [#uses=0]
%.str_38 = external global [41 x sbyte]		; <[41 x sbyte]*> [#uses=0]
%.str_39 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_40 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_41 = external global [40 x sbyte]		; <[40 x sbyte]*> [#uses=0]
%.str_42 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_43 = external global [41 x sbyte]		; <[41 x sbyte]*> [#uses=0]
%.str_44 = external global [41 x sbyte]		; <[41 x sbyte]*> [#uses=0]
%.str_45 = external global [39 x sbyte]		; <[39 x sbyte]*> [#uses=0]
%.str_46 = external global [35 x sbyte]		; <[35 x sbyte]*> [#uses=0]
%.str_47 = external global [50 x sbyte]		; <[50 x sbyte]*> [#uses=0]
%.str_48 = external global [26 x sbyte]		; <[26 x sbyte]*> [#uses=0]
%.str_49 = external global [31 x sbyte]		; <[31 x sbyte]*> [#uses=0]
%.str_50 = external global [15 x sbyte]		; <[15 x sbyte]*> [#uses=0]
%.str_51 = external global [6 x sbyte]		; <[6 x sbyte]*> [#uses=0]
%.str_52 = external global [24 x sbyte]		; <[24 x sbyte]*> [#uses=0]
%.str_53 = external global [9 x sbyte]		; <[9 x sbyte]*> [#uses=0]

implementation   ; Functions:

declare ulong %AttacksFrom(int, int)

declare ulong %AttacksTo(int)

declare int %Attacked(int, int)

declare ulong %Mask(int)

declare int %PopCnt(ulong)

declare int %FirstOne(ulong)

declare int %LastOne(ulong)

declare int %DrawScore()

declare int %Drawn(int)

declare sbyte* %strchr(sbyte*, int)

declare int %strcmp(sbyte*, sbyte*)

declare uint %strlen(sbyte*)

declare int %printf(sbyte*, ...)

declare void %Edit()

declare void %llvm.memcpy(sbyte*, sbyte*, uint, uint)

declare int %fflush(%struct.__sFILE*)

declare int %Read(int, sbyte*)

declare int %ReadParse(sbyte*, sbyte**, sbyte*)

declare void %DisplayChessBoard(%struct.__sFILE*, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, uint, int, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte)

declare void %SetChessBitBoards(%typedef.SEARCH_POSITION*)

declare int %EnPrise(int, int)

declare ulong %SwapXray(ulong, int, int)

declare int %Evaluate(int, int, int, int)

declare int %EvaluateMate()

declare int %EvaluatePawns()

declare int %EvaluatePassedPawns()

declare int %EvaluatePassedPawnRaces(int)

declare int %Swap(int, int, int)

declare int %EvaluateDevelopment(int)

declare int %EvaluateDraws()

declare int %HasOpposition(int, int, int)

declare void %HistoryBest(int, int, int)

declare void %HistoryRefutation(int, int, int)

declare int %sprintf(sbyte*, sbyte*, ...)

declare void %Initialize(int)

declare void %InitializeZeroMasks()

declare void %InitializeMasks()

declare void %InitializeRandomHash()

declare void %InitializeAttackBoards()

declare void %InitializePawnMasks()

declare void %InitializePieceMasks()

declare void %InitializeChessBoard(%typedef.SEARCH_POSITION*)

declare %struct.__sFILE* %fopen(sbyte*, sbyte*)

int %Option() {
no_exit.53.outer:
	%tmp.4747 = shl int 7, ubyte 3		; <int> [#uses=1]
	%tmp.4779 = seteq %struct.__sFILE* getelementptr ([0 x %struct.__sFILE]* %__sF, int 0, int 1), null		; <bool> [#uses=2]
	br label %no_exit.53

no_exit.53:		; preds = %else.166, %else.168, %then.360, %no_exit.53.outer
	%file.2.3.3.ph = phi int [ 0, %no_exit.53.outer ], [ %inc.551688, %then.360 ], [ %inc.551701, %else.168 ], [ %file.2.3.3.ph, %else.166 ]		; <int> [#uses=2]
	%nempty.5.3.ph = phi int [ 0, %no_exit.53.outer ], [ %nempty.5.3, %then.360 ], [ %nempty.5.3, %else.168 ], [ %nempty.5.3.ph, %else.166 ]		; <int> [#uses=2]
	%indvar2053 = phi uint [ 0, %no_exit.53.outer ], [ 0, %then.360 ], [ 0, %else.168 ], [ %indvar.next2054, %else.166 ]		; <uint> [#uses=2]
	%indvar2053 = cast uint %indvar2053 to int		; <int> [#uses=2]
	%file.2.3.3 = add int %indvar2053, %file.2.3.3.ph		; <int> [#uses=4]
	%nempty.5.3 = add int %indvar2053, %nempty.5.3.ph		; <int> [#uses=3]
	%tmp.4749 = add int %file.2.3.3, %tmp.4747		; <int> [#uses=1]
	%tmp.4750 = getelementptr %typedef.CHESS_POSITION* %search, int 0, uint 22, int %tmp.4749		; <sbyte*> [#uses=3]
	%tmp.4751 = load sbyte* %tmp.4750		; <sbyte> [#uses=1]
	%tmp.4752 = seteq sbyte %tmp.4751, 0		; <bool> [#uses=1]
	br bool %tmp.4752, label %else.166, label %then.357

then.357:		; preds = %no_exit.53
	%tmp.4755 = seteq int %nempty.5.3, 0		; <bool> [#uses=1]
	br bool %tmp.4755, label %endif.358, label %then.358

then.358:		; preds = %then.357
	ret int 0

endif.358:		; preds = %then.357
	br bool %tmp.4779, label %else.168, label %then.360

then.360:		; preds = %endif.358
	%tmp.4791 = load sbyte* %tmp.4750		; <sbyte> [#uses=1]
	%tmp.4792 = cast sbyte %tmp.4791 to int		; <int> [#uses=1]
	%tmp.4793 = add int %tmp.4792, 7		; <int> [#uses=1]
	%tmp.4794 = getelementptr [15 x sbyte]* null, int 0, int %tmp.4793		; <sbyte*> [#uses=1]
	%tmp.4795 = load sbyte* %tmp.4794		; <sbyte> [#uses=1]
	%tmp.4796 = cast sbyte %tmp.4795 to int		; <int> [#uses=1]
	%tmp.4781 = call int (%struct.__sFILE*, sbyte*, ...)* %fprintf( %struct.__sFILE* getelementptr ([0 x %struct.__sFILE]* %__sF, int 0, int 1), sbyte* getelementptr ([3 x sbyte]* %.str_36, int 0, int 0), int %tmp.4796 )		; <int> [#uses=0]
	%inc.551688 = add int %file.2.3.3, 1		; <int> [#uses=2]
	%tmp.47421699 = setlt int %inc.551688, 8		; <bool> [#uses=1]
	br bool %tmp.47421699, label %no_exit.53, label %loopexit.56

else.168:		; preds = %endif.358
	%tmp.4799 = call uint %strlen( sbyte* getelementptr ([80 x sbyte]* %initial_position, int 0, int 0) )		; <uint> [#uses=2]
	%tmp.4802 = getelementptr [80 x sbyte]* %initial_position, int 0, uint %tmp.4799		; <sbyte*> [#uses=1]
	%tmp.4811 = load sbyte* %tmp.4750		; <sbyte> [#uses=1]
	%tmp.4812 = cast sbyte %tmp.4811 to int		; <int> [#uses=1]
	%tmp.4813 = add int %tmp.4812, 7		; <int> [#uses=1]
	%tmp.4814 = getelementptr [15 x sbyte]* null, int 0, int %tmp.4813		; <sbyte*> [#uses=1]
	%tmp.4815 = load sbyte* %tmp.4814		; <sbyte> [#uses=1]
	store sbyte %tmp.4815, sbyte* %tmp.4802
	%tmp.4802.sum = add uint %tmp.4799, 1		; <uint> [#uses=1]
	%tmp.4802.end = getelementptr [80 x sbyte]* %initial_position, int 0, uint %tmp.4802.sum		; <sbyte*> [#uses=1]
	store sbyte 0, sbyte* %tmp.4802.end
	%inc.551701 = add int %file.2.3.3, 1		; <int> [#uses=2]
	%tmp.47421703 = setlt int %inc.551701, 8		; <bool> [#uses=1]
	br bool %tmp.47421703, label %no_exit.53, label %loopexit.56

else.166:		; preds = %no_exit.53
	%inc.55 = add int %file.2.3.3, 1		; <int> [#uses=1]
	%tmp.47421705 = setlt int %inc.55, 8		; <bool> [#uses=1]
	%indvar.next2054 = add uint %indvar2053, 1		; <uint> [#uses=1]
	br bool %tmp.47421705, label %no_exit.53, label %loopexit.56

loopexit.56:		; preds = %else.166, %else.168, %then.360
	br bool %tmp.4779, label %else.169, label %then.361

then.361:		; preds = %loopexit.56
	%tmp.4822 = call uint %fwrite( sbyte* getelementptr ([2 x sbyte]* %.str_654, int 0, int 0), uint 1, uint 1, %struct.__sFILE* getelementptr ([0 x %struct.__sFILE]* %__sF, int 0, int 1) )		; <uint> [#uses=0]
	%dec.101707 = add int 7, -1		; <int> [#uses=1]
	%tmp.47391709 = setgt int %dec.101707, -1		; <bool> [#uses=0]
	ret int 0

else.169:		; preds = %loopexit.56
	%tmp.4827 = call uint %strlen( sbyte* getelementptr ([80 x sbyte]* %initial_position, int 0, int 0) )		; <uint> [#uses=2]
	%tmp.4830 = getelementptr [80 x sbyte]* %initial_position, int 0, uint %tmp.4827		; <sbyte*> [#uses=1]
	store sbyte 47, sbyte* %tmp.4830
	%tmp.4830.sum = add uint %tmp.4827, 1		; <uint> [#uses=1]
	%tmp.4830.end = getelementptr [80 x sbyte]* %initial_position, int 0, uint %tmp.4830.sum		; <sbyte*> [#uses=1]
	store sbyte 0, sbyte* %tmp.4830.end
	%dec.10 = add int 7, -1		; <int> [#uses=1]
	%tmp.47391711 = setgt int %dec.10, -1		; <bool> [#uses=0]
	ret int 0
}

declare void %InitializeHashTables()

declare int %InitializeFindAttacks(int, int, int)

declare void %SetBoard(int, sbyte**, int)

declare int %KingPawnSquare(int, int, int, int)

declare ulong %Random64()

declare uint %Random32()

declare sbyte* %strcpy(sbyte*, sbyte*)

declare int %InputMove(sbyte*, int, int, int, int)

declare int %InputMoveICS(sbyte*, int, int, int, int)

declare int* %GenerateCaptures(int, int, int*)

declare int* %GenerateNonCaptures(int, int, int*)

declare void %MakeMove(int, int, int)

declare void %UnMakeMove(int, int, int)

declare void %Interrupt(int)

declare uint %GetTime(uint)

declare sbyte* %DisplayTime(uint)

declare sbyte* %OutputMoveICS(int*)

declare void %Delay(int)

declare int %fprintf(%struct.__sFILE*, sbyte*, ...)

declare void %SignalInterrupt(int)

declare void (int)* %signal(int, void (int)*)

declare int %Iterate(int, int, int)

declare void %PreEvaluate(int)

declare void %RootMoveList(int)

declare sbyte* %OutputMove(int*, int, int)

declare void %TimeSet(int)

declare void %StorePV(int, int)

declare int %SearchRoot(int, int, int, int)

declare void %Whisper(int, int, int, int, uint, int, sbyte*)

declare sbyte* %DisplayEvaluation(int)

declare int %LookUp(int, int, int, int*, int*)

declare sbyte* %strstr(sbyte*, sbyte*)

declare int %main(int, sbyte**)

declare void %__main()

declare int %atoi(sbyte*)

declare void %NewGame(int)

declare int %Ponder(int)

declare int %fseek(%struct.__sFILE*, int, int)

declare void %MakeMoveRoot(int, int)

declare int %RepetitionDraw(int)

declare sbyte* %Reverse()

declare sbyte* %Normal()

declare void %TimeAdjust(int, uint)

declare void %ValidatePosition(int, int, sbyte*)

declare int %ValidMove(int, int, int)

declare int* %GenerateCheckEvasions(int, int, int*)

declare ulong %InterposeSquares(int, int, int)

declare int %PinnedOnKing(int, int)

declare int %NextMove(int, int)

declare int %NextEvasion(int, int)

declare int %NextRootMove(int)

declare int %TimeCheck(int)

declare int %strncmp(sbyte*, sbyte*, uint)

declare void %exit(int)

declare int %OptionMatch(sbyte*, sbyte*)

declare int %fclose(%struct.__sFILE*)

declare int %ParseTime(sbyte*)

declare sbyte* %DisplayHHMM(uint)

declare void %DisplayPieceBoards(int*, int*)

declare int %fscanf(%struct.__sFILE*, sbyte*, ...)

declare int %feof(%struct.__sFILE*)

declare sbyte* %fgets(sbyte*, int, %struct.__sFILE*)

declare int %remove(sbyte*)

declare int %__tolower(int)

declare uint %clock()

declare void %OptionPerft(int, int, int)

declare void %Phase()

declare int %ReadNextMove(sbyte*, int, int)

declare int %time(int*)

declare %struct.tm* %localtime(int*)

declare sbyte* %gets(sbyte*)

declare int %OutputGood(sbyte*, int, int)

declare int %CheckInput()

declare void %ClearHashTables()

declare int %Quiesce(int, int, int, int)

declare void %SearchTrace(int, int, int, int, int, sbyte*, int)

declare int %RepetitionCheck(int, int)

declare void %ResignOrDraw(int, int)

declare int %Search(int, int, int, int, int, int)

declare void %StoreRefutation(int, int, int, int)

declare void %StoreBest(int, int, int, int, int)

declare void %SearchOutput(int, int)

declare uint %strspn(sbyte*, sbyte*)

declare int %isatty(int)

declare int %fileno(%struct.__sFILE*)

declare void %llvm.memset(sbyte*, ubyte, uint, uint)

declare int %select(int, %struct.fd_set*, %struct.fd_set*, %struct.fd_set*, %struct.timeval*)

declare void %DisplayBitBoard(ulong)

declare sbyte* %DisplayEvaluationWhisper(int)

declare sbyte* %DisplayTimeWhisper(uint)

declare void %Display64bitWord(ulong)

declare void %Display2BitBoards(ulong, ulong)

declare void %DisplayChessMove(sbyte*, int)

declare void %llvm.memmove(sbyte*, sbyte*, uint, uint)

declare void %ReadClear()

declare sbyte* %strtok(sbyte*, sbyte*)

declare int %SpecReadRaw()

declare int %read(int, sbyte*, uint)

declare int* %__error()

declare int %ReadChessMove(%struct.__sFILE*, int, int)

declare ulong %ValidateComputeBishopAttacks(int)

declare ulong %ValidateComputeRookAttacks(int)

declare sbyte* %memchr(sbyte*, int, uint)

declare uint %fwrite(sbyte*, uint, uint, %struct.__sFILE*)
