; RUN: llc -march=mipsel -mcpu=mips16 -relocation-model=pic < %s | FileCheck %s -check-prefix=cmp16

%union.yystype = type { i32 }
%struct.LIST_HELP = type { %struct.LIST_HELP*, i8* }
%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i32, i32, [40 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.MEMORY_RESOURCEHELP = type { i8*, i8*, i8*, i8*, i32, i32, i32 }
%struct.signature = type { i8*, i32, i32, i32, i32, i32, %struct.LIST_HELP* }
%union.yyalloc = type { %union.yystype }
%struct.term = type { i32, %union.anon, %struct.LIST_HELP*, i32, i32 }
%union.anon = type { %struct.LIST_HELP* }
%struct.CLAUSE_HELP = type { i32, i32, i32, i32, i32*, i32, %struct.LIST_HELP*, %struct.LIST_HELP*, i32, i32, %struct.LITERAL_HELP**, i32, i32, i32, i32 }
%struct.LITERAL_HELP = type { i32, i32, i32, %struct.CLAUSE_HELP*, %struct.term* }
%struct.DFG_VARENTRY = type { i8*, i32 }

@dfg_nerrs = common global i32 0, align 4
@dfg_char = common global i32 0, align 4
@yypact = internal unnamed_addr constant [477 x i16] [i16 9, i16 -32, i16 35, i16 232, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -6, i16 13, i16 67, i16 20, i16 45, i16 53, i16 30, i16 -356, i16 110, i16 46, i16 118, i16 121, i16 -12, i16 73, i16 -356, i16 91, i16 84, i16 113, i16 112, i16 141, i16 123, i16 128, i16 132, i16 -356, i16 -356, i16 175, i16 152, i16 161, i16 155, i16 191, i16 2, i16 162, i16 180, i16 -356, i16 204, i16 232, i16 214, i16 173, i16 -356, i16 252, i16 176, i16 206, i16 209, i16 213, i16 226, i16 232, i16 47, i16 -356, i16 -356, i16 80, i16 218, i16 254, i16 224, i16 -14, i16 -356, i16 -356, i16 230, i16 233, i16 -356, i16 234, i16 241, i16 232, i16 242, i16 -356, i16 -356, i16 -356, i16 243, i16 237, i16 21, i16 244, i16 -356, i16 260, i16 -356, i16 246, i16 245, i16 250, i16 251, i16 294, i16 247, i16 248, i16 2, i16 232, i16 93, i16 -356, i16 -356, i16 232, i16 255, i16 272, i16 232, i16 253, i16 -356, i16 256, i16 -356, i16 232, i16 257, i16 232, i16 290, i16 232, i16 232, i16 -356, i16 -356, i16 -356, i16 258, i16 21, i16 261, i16 -356, i16 271, i16 -356, i16 262, i16 264, i16 14, i16 263, i16 317, i16 108, i16 -356, i16 -356, i16 265, i16 266, i16 80, i16 119, i16 -356, i16 85, i16 268, i16 312, i16 -356, i16 124, i16 -356, i16 270, i16 273, i16 269, i16 -356, i16 274, i16 -356, i16 309, i16 275, i16 -356, i16 -52, i16 276, i16 277, i16 232, i16 279, i16 -356, i16 -356, i16 281, i16 -356, i16 -356, i16 -356, i16 284, i16 287, i16 288, i16 321, i16 -356, i16 -356, i16 286, i16 108, i16 -356, i16 -356, i16 289, i16 232, i16 232, i16 138, i16 -356, i16 -356, i16 156, i16 291, i16 293, i16 232, i16 -17, i16 232, i16 232, i16 232, i16 232, i16 346, i16 232, i16 -356, i16 232, i16 -356, i16 40, i16 296, i16 -356, i16 -356, i16 297, i16 299, i16 302, i16 300, i16 -356, i16 303, i16 -356, i16 -356, i16 285, i16 301, i16 85, i16 232, i16 143, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 337, i16 16, i16 304, i16 298, i16 306, i16 -356, i16 32, i16 -356, i16 311, i16 305, i16 -356, i16 56, i16 308, i16 314, i16 310, i16 -356, i16 -356, i16 315, i16 318, i16 -356, i16 -356, i16 108, i16 -356, i16 -356, i16 313, i16 319, i16 156, i16 -2, i16 320, i16 -356, i16 -356, i16 232, i16 232, i16 316, i16 322, i16 232, i16 232, i16 323, i16 324, i16 307, i16 325, i16 326, i16 -356, i16 240, i16 -356, i16 327, i16 329, i16 108, i16 -356, i16 -356, i16 -356, i16 331, i16 332, i16 334, i16 333, i16 -356, i16 335, i16 -356, i16 336, i16 -356, i16 -356, i16 145, i16 -356, i16 -356, i16 -356, i16 96, i16 -356, i16 -356, i16 -356, i16 338, i16 340, i16 -356, i16 -356, i16 342, i16 232, i16 163, i16 339, i16 -356, i16 -356, i16 239, i16 343, i16 232, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 344, i16 -356, i16 -356, i16 341, i16 347, i16 348, i16 350, i16 -356, i16 3, i16 -356, i16 -15, i16 -356, i16 -356, i16 -356, i16 42, i16 232, i16 -356, i16 43, i16 -356, i16 349, i16 351, i16 -356, i16 -356, i16 96, i16 232, i16 352, i16 96, i16 96, i16 353, i16 355, i16 357, i16 57, i16 358, i16 361, i16 -356, i16 359, i16 -356, i16 163, i16 108, i16 360, i16 362, i16 -356, i16 363, i16 364, i16 -356, i16 44, i16 -356, i16 -13, i16 -356, i16 366, i16 365, i16 -356, i16 168, i16 372, i16 -356, i16 369, i16 -356, i16 -356, i16 -356, i16 96, i16 -356, i16 96, i16 232, i16 371, i16 373, i16 341, i16 -356, i16 -356, i16 0, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 367, i16 -356, i16 370, i16 -356, i16 375, i16 -356, i16 306, i16 374, i16 228, i16 377, i16 379, i16 380, i16 341, i16 -356, i16 -356, i16 50, i16 381, i16 376, i16 382, i16 -356, i16 383, i16 -356, i16 384, i16 66, i16 -356, i16 -356, i16 386, i16 228, i16 387, i16 385, i16 -356, i16 -356, i16 388, i16 7, i16 -356, i16 -356, i16 -356, i16 389, i16 232, i16 239, i16 -356, i16 228, i16 -356, i16 69, i16 239, i16 393, i16 232, i16 232, i16 90, i16 96, i16 306, i16 390, i16 -356, i16 -356, i16 153, i16 -356, i16 -356, i16 391, i16 179, i16 -356, i16 396, i16 395, i16 -356, i16 397, i16 239, i16 398, i16 401, i16 -356, i16 402, i16 399, i16 -356, i16 168, i16 96, i16 409, i16 408, i16 185, i16 -356, i16 410, i16 411, i16 -356, i16 405, i16 168, i16 -356, i16 -356, i16 400, i16 412, i16 -356, i16 168, i16 413, i16 198, i16 345, i16 -356, i16 -356, i16 168, i16 168, i16 394, i16 -356, i16 168, i16 -356], align 2
@yytranslate = internal unnamed_addr constant [319 x i8] c"\00\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02@A\02\02E\02B\02\02\02\02\02\02\02\02\02\02\02F\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02C\02D\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\01\02\03\04\05\06\07\08\09\0A\0B\0C\0D\0E\0F\10\11\12\13\14\15\16\17\18\19\1A\1B\1C\1D\1E\1F !\22#$%&'()*+,-./0123456789:;<=>?", align 1
@yycheck = internal unnamed_addr constant [507 x i16] [i16 3, i16 46, i16 103, i16 3, i16 19, i16 19, i16 361, i16 9, i16 20, i16 276, i16 3, i16 8, i16 164, i16 65, i16 27, i16 6, i16 18, i16 69, i16 32, i16 31, i16 18, i16 38, i16 19, i16 23, i16 364, i16 18, i16 5, i16 40, i16 21, i16 22, i16 23, i16 24, i16 64, i16 12, i16 27, i16 0, i16 48, i16 40, i16 53, i16 41, i16 33, i16 34, i16 45, i16 41, i16 37, i16 47, i16 63, i16 40, i16 41, i16 47, i16 390, i16 11, i16 55, i16 3, i16 47, i16 322, i16 59, i16 17, i16 325, i16 65, i16 46, i16 416, i16 60, i16 56, i16 62, i16 8, i16 64, i16 60, i16 71, i16 62, i16 425, i16 57, i16 58, i16 23, i16 67, i16 19, i16 19, i16 27, i16 179, i16 66, i16 232, i16 65, i16 26, i16 16, i16 185, i16 69, i16 66, i16 90, i16 91, i16 36, i16 40, i16 358, i16 95, i16 360, i16 49, i16 98, i16 66, i16 65, i16 18, i16 3, i16 103, i16 69, i16 105, i16 18, i16 107, i16 108, i16 258, i16 65, i16 65, i16 65, i16 64, i16 69, i16 69, i16 69, i16 18, i16 68, i16 69, i16 21, i16 22, i16 23, i16 24, i16 41, i16 4, i16 27, i16 14, i16 128, i16 41, i16 47, i16 131, i16 33, i16 34, i16 65, i16 47, i16 37, i16 65, i16 69, i16 40, i16 41, i16 69, i16 66, i16 60, i16 449, i16 62, i16 47, i16 64, i16 60, i16 149, i16 62, i16 64, i16 64, i16 29, i16 459, i16 56, i16 254, i16 63, i16 65, i16 60, i16 465, i16 62, i16 69, i16 427, i16 68, i16 69, i16 471, i16 472, i16 168, i16 169, i16 475, i16 60, i16 61, i16 173, i16 3, i16 59, i16 176, i16 18, i16 178, i16 179, i16 180, i16 181, i16 67, i16 183, i16 18, i16 185, i16 450, i16 43, i16 337, i16 18, i16 68, i16 69, i16 66, i16 22, i16 23, i16 68, i16 69, i16 66, i16 27, i16 64, i16 41, i16 201, i16 202, i16 25, i16 33, i16 34, i16 47, i16 41, i16 37, i16 68, i16 69, i16 40, i16 41, i16 47, i16 68, i16 69, i16 68, i16 69, i16 47, i16 60, i16 65, i16 62, i16 64, i16 64, i16 68, i16 69, i16 60, i16 63, i16 62, i16 35, i16 64, i16 60, i16 67, i16 62, i16 51, i16 18, i16 19, i16 237, i16 21, i16 32, i16 64, i16 24, i16 242, i16 243, i16 27, i16 66, i16 246, i16 247, i16 290, i16 18, i16 68, i16 69, i16 21, i16 18, i16 254, i16 24, i16 68, i16 69, i16 41, i16 42, i16 18, i16 18, i16 19, i16 21, i16 47, i16 10, i16 24, i16 50, i16 37, i16 68, i16 69, i16 54, i16 41, i16 56, i16 65, i16 63, i16 41, i16 60, i16 47, i16 62, i16 64, i16 52, i16 47, i16 41, i16 41, i16 285, i16 286, i16 56, i16 67, i16 47, i16 47, i16 60, i16 292, i16 62, i16 67, i16 60, i16 39, i16 62, i16 56, i16 66, i16 64, i16 64, i16 60, i16 60, i16 62, i16 62, i16 66, i16 44, i16 64, i16 64, i16 64, i16 64, i16 15, i16 65, i16 314, i16 66, i16 63, i16 66, i16 64, i16 69, i16 45, i16 28, i16 66, i16 323, i16 65, i16 67, i16 65, i16 67, i16 64, i16 55, i16 64, i16 66, i16 65, i16 13, i16 19, i16 66, i16 336, i16 69, i16 67, i16 66, i16 64, i16 69, i16 30, i16 19, i16 386, i16 66, i16 69, i16 69, i16 66, i16 69, i16 66, i16 351, i16 65, i16 63, i16 65, i16 64, i16 7, i16 69, i16 66, i16 19, i16 66, i16 361, i16 66, i16 405, i16 65, i16 64, i16 66, i16 65, i16 63, i16 66, i16 66, i16 65, i16 65, i16 65, i16 64, i16 417, i16 65, i16 419, i16 69, i16 66, i16 422, i16 67, i16 65, i16 113, i16 66, i16 69, i16 65, i16 68, i16 66, i16 66, i16 128, i16 66, i16 66, i16 66, i16 66, i16 66, i16 66, i16 65, i16 64, i16 64, i16 442, i16 64, i16 66, i16 65, i16 62, i16 3, i16 201, i16 66, i16 69, i16 66, i16 65, i16 64, i16 66, i16 69, i16 64, i16 64, i16 416, i16 64, i16 70, i16 65, i16 65, i16 69, i16 64, i16 67, i16 424, i16 425, i16 66, i16 66, i16 64, i16 66, i16 65, i16 69, i16 66, i16 64, i16 66, i16 60, i16 69, i16 65, i16 64, i16 69, i16 64, i16 62, i16 69, i16 65, i16 67, i16 65, i16 64, i16 64, i16 64, i16 449, i16 65, i16 64, i16 40, i16 65, i16 68, i16 66, i16 237, i16 67, i16 65, i16 459, i16 69, i16 69, i16 66, i16 69, i16 65, i16 465, i16 68, i16 70, i16 67, i16 69, i16 67, i16 471, i16 472, i16 69, i16 69, i16 475, i16 65, i16 69, i16 65, i16 65, i16 65, i16 243, i16 66, i16 393, i16 411, i16 90, i16 405, i16 451, i16 393, i16 447, i16 419, i16 63, i16 -1, i16 336, i16 -1, i16 285, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 178], align 2
@yytable = internal unnamed_addr constant [507 x i16] [i16 10, i16 77, i16 139, i16 388, i16 331, i16 99, i16 384, i16 261, i16 30, i16 301, i16 293, i16 328, i16 196, i16 184, i16 362, i16 1, i16 262, i16 185, i16 62, i16 31, i16 5, i16 209, i16 329, i16 389, i16 387, i16 5, i16 110, i16 363, i16 67, i16 294, i16 295, i16 68, i16 3, i16 111, i16 296, i16 4, i16 32, i16 58, i16 332, i16 263, i16 297, i16 298, i16 65, i16 6, i16 299, i16 264, i16 210, i16 300, i16 6, i16 7, i16 407, i16 223, i16 88, i16 388, i16 7, i16 339, i16 94, i16 224, i16 342, i16 11, i16 154, i16 428, i16 8, i16 73, i16 9, i16 348, i16 55, i16 8, i16 106, i16 9, i16 436, i16 155, i16 156, i16 389, i16 425, i16 249, i16 349, i16 362, i16 213, i16 12, i16 257, i16 240, i16 250, i16 13, i16 221, i16 185, i16 15, i16 58, i16 126, i16 19, i16 363, i16 382, i16 130, i16 383, i16 16, i16 134, i16 21, i16 245, i16 5, i16 293, i16 138, i16 246, i16 141, i16 5, i16 144, i16 138, i16 284, i16 333, i16 335, i16 359, i16 25, i16 176, i16 336, i16 360, i16 5, i16 89, i16 90, i16 67, i16 294, i16 295, i16 68, i16 6, i16 26, i16 296, i16 23, i16 94, i16 6, i16 7, i16 172, i16 297, i16 298, i16 418, i16 7, i16 299, i16 431, i16 419, i16 300, i16 6, i16 419, i16 34, i16 8, i16 454, i16 9, i16 7, i16 91, i16 8, i16 188, i16 9, i16 37, i16 169, i16 28, i16 464, i16 73, i16 281, i16 36, i16 437, i16 8, i16 468, i16 9, i16 360, i16 438, i16 127, i16 128, i16 473, i16 474, i16 198, i16 199, i16 476, i16 161, i16 162, i16 205, i16 369, i16 38, i16 208, i16 5, i16 138, i16 138, i16 214, i16 218, i16 40, i16 220, i16 5, i16 138, i16 455, i16 41, i16 354, i16 5, i16 167, i16 168, i16 43, i16 370, i16 371, i16 175, i16 176, i16 44, i16 372, i16 45, i16 6, i16 172, i16 235, i16 47, i16 373, i16 374, i16 7, i16 6, i16 375, i16 200, i16 201, i16 376, i16 6, i16 7, i16 236, i16 237, i16 291, i16 292, i16 7, i16 8, i16 50, i16 9, i16 52, i16 202, i16 441, i16 442, i16 8, i16 51, i16 9, i16 53, i16 314, i16 8, i16 59, i16 9, i16 60, i16 5, i16 66, i16 205, i16 67, i16 62, i16 78, i16 68, i16 267, i16 214, i16 69, i16 82, i16 271, i16 273, i16 319, i16 5, i16 444, i16 243, i16 67, i16 5, i16 138, i16 68, i16 458, i16 459, i16 6, i16 70, i16 5, i16 5, i16 280, i16 67, i16 7, i16 79, i16 68, i16 71, i16 400, i16 470, i16 471, i16 72, i16 6, i16 73, i16 83, i16 84, i16 6, i16 8, i16 7, i16 9, i16 85, i16 86, i16 7, i16 6, i16 6, i16 134, i16 315, i16 73, i16 95, i16 7, i16 7, i16 8, i16 321, i16 9, i16 98, i16 8, i16 96, i16 9, i16 73, i16 102, i16 103, i16 104, i16 8, i16 8, i16 9, i16 9, i16 109, i16 114, i16 105, i16 107, i16 108, i16 113, i16 121, i16 118, i16 334, i16 117, i16 119, i16 124, i16 120, i16 123, i16 132, i16 142, i16 136, i16 340, i16 137, i16 131, i16 147, i16 140, i16 149, i16 150, i16 158, i16 152, i16 153, i16 159, i16 174, i16 165, i16 315, i16 164, i16 173, i16 177, i16 178, i16 179, i16 182, i16 194, i16 403, i16 186, i16 181, i16 183, i16 189, i16 187, i16 190, i16 377, i16 191, i16 192, i16 195, i16 193, i16 219, i16 232, i16 197, i16 239, i16 206, i16 214, i16 207, i16 403, i16 225, i16 242, i16 227, i16 228, i16 229, i16 230, i16 233, i16 231, i16 241, i16 248, i16 276, i16 429, i16 251, i16 403, i16 243, i16 253, i16 433, i16 247, i16 252, i16 148, i16 255, i16 258, i16 256, i16 269, i16 259, i16 266, i16 166, i16 270, i16 274, i16 275, i16 277, i16 278, i16 282, i16 283, i16 285, i16 286, i16 448, i16 287, i16 288, i16 289, i16 318, i16 388, i16 234, i16 310, i16 290, i16 311, i16 312, i16 322, i16 320, i16 323, i16 325, i16 326, i16 214, i16 327, i16 472, i16 338, i16 341, i16 337, i16 346, i16 345, i16 435, i16 214, i16 347, i16 350, i16 351, i16 355, i16 356, i16 352, i16 357, i16 367, i16 368, i16 380, i16 358, i16 381, i16 385, i16 393, i16 386, i16 413, i16 395, i16 397, i16 399, i16 404, i16 405, i16 406, i16 411, i16 377, i16 414, i16 417, i16 363, i16 423, i16 415, i16 420, i16 260, i16 422, i16 440, i16 377, i16 424, i16 427, i16 434, i16 443, i16 445, i16 377, i16 446, i16 475, i16 449, i16 447, i16 465, i16 377, i16 377, i16 450, i16 451, i16 377, i16 457, i16 462, i16 460, i16 461, i16 466, i16 268, i16 469, i16 408, i16 426, i16 125, i16 421, i16 456, i16 409, i16 452, i16 430, i16 100, i16 0, i16 353, i16 0, i16 313, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 212], align 2
@dfg_lval = common global %union.yystype zeroinitializer, align 4
@yydefact = internal unnamed_addr constant [477 x i8] c"\00\00\00\00\01bca`_\00\00\00\00\12\00\00\AE2\00\00\14\00\00G\00\00\08\00\1A\00\00\00\AF4h\00\00\00\0A\00\00 \02\00\00\00\00H\8C\00\00\00\00\00\00\00\16\18\00\00$\00\00\C0\B1\00\00~\00\00\00\00}56|\00\00\00i\A8\04\00\00\00\00\00\00\00\00\00\00\1C\1E\00\00*\00\00\C1\003\00\00\00>\00\008EF\00\00\00\8D\11\05\00\00\00\00\0C\00\15\17\00\00\00\00\22\00\00\00\C3\00\B0\00\88\00:\00?\00\00\8A\00\00\00\00\00\A9\09\00\0E\10\0F\00\00\00\0010\00\00\1B\1D\00\00\00\00&(\00\00\00\00\00\00\00\00\00\00\00\80\00I\00\00\AA\0B\00\00\00\00\19\00!#\00\00\00\00\00,.\13\C2\C4\B5\B3\00\00\00f;d\00B\00\00\8B\00\00\00\00\AC\06\00\00\03\1F\00%'\00\00\00\00\00\89\7F\00\00\00\00\00\00\00\00\00\00\00\8E\00\07\00\00\00+-\B4\00\00\00\00\B2\00e\009C\00@7DTmk\90\00\00\0D)\00\00\00\00\B6g\00\00\00[X]^YZ\00\5CUK\00\00\00M\00\81\00\AB\AD/\00\00\BC\00\BA\00\00=A\00\00\00\00\00\00\00\00\00\00\00\B8\00\B7\00\00\00\00L\00\00V\00Qoj\00\00\8F\00\00\BB\00<NJ\00P\00\00\00\00Kpq\83l\9F\9C\A2\A1\9D\9E\9B\A0\9A\00\98\BE\B9\00WR\00\00\00\00\00K\84\85\00\A3\00\00O\00r\00\00vx\00\00\00\00\A6\A7\00\00\99\BF\BD\00\00\00u\00n\00\00\00\00\00\00\00s\00w\87\00z\82\00\00\A4\00\00y\00\00\00\00S\00\00{\00\00\00\00\00\92\00\00\86\94\00\A5t\00\00\93\00\00\00\00\91\95\00\00\00\96\00\97", align 1
@yyr2 = internal unnamed_addr constant [197 x i8] c"\00\02\0A\0B\05\05\05\05\00\05\00\05\00\05\01\01\01\06\00\09\00\05\01\03\01\05\00\05\01\03\01\05\00\05\01\03\00\05\01\03\01\05\00\05\01\03\01\05\01\01\00\05\00\02\01\07\02\07\00\00\0B\09\00\01\01\03\01\03\08\01\01\00\02\00\07\00\02\01\04\06\04\00\00\0A\00\01\01\03\01\01\01\01\01\01\01\01\01\01\01\01\01\03\01\04\00\02\0A\00\0B\00\07\00\01\01\00\00\0A\04\01\03\01\04\01\03\01\01\01\06\04\00\07\00\01\01\08\04\01\04\01\03\00\02\00\09\00\0F\01\03\00\04\03\05\00\03\01\01\01\01\01\01\01\01\01\00\03\07\01\01\00\02\00\06\00\03\00\02\05\00\09\01\03\00\03\04\04\06\01\03\01\06\00\02\01\02\05\01\03", align 1
@dfg_DESC.0 = internal unnamed_addr global i8* null
@dfg_DESC.1 = internal unnamed_addr global i8* null
@dfg_DESC.2 = internal unnamed_addr global i8* null
@dfg_DESC.3 = internal unnamed_addr global i8* null
@dfg_DESC.4 = internal unnamed_addr global i32 0
@dfg_DESC.5 = internal unnamed_addr global i8* null
@dfg_DESC.6 = internal unnamed_addr global i8* null
@dfg_SORTDECLLIST = internal unnamed_addr global %struct.LIST_HELP* null, align 4
@dfg_AXIOMLIST = internal unnamed_addr global %struct.LIST_HELP* null, align 4
@dfg_CONJECLIST = internal unnamed_addr global %struct.LIST_HELP* null, align 4
@dfg_IGNORE = internal unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [9 x i8] c"set_flag\00", align 1
@.str1 = private unnamed_addr constant [12 x i8] c"set_DomPred\00", align 1
@.str2 = private unnamed_addr constant [15 x i8] c"set_precedence\00", align 1
@stdout = external global %struct._IO_FILE*
@.str3 = private unnamed_addr constant [38 x i8] c"\0A Line %d: Symbol is not a variable.\0A\00", align 1
@dfg_LINENUMBER = common global i32 0, align 4
@.str4 = private unnamed_addr constant [39 x i8] c"\0A Line %d: Symbol is not a predicate.\0A\00", align 1
@dfg_AXCLAUSES = internal unnamed_addr global %struct.LIST_HELP* null, align 4
@dfg_CONCLAUSES = internal unnamed_addr global %struct.LIST_HELP* null, align 4
@.str5 = private unnamed_addr constant [6 x i8] c"SPASS\00", align 1
@dfg_PROOFLIST = internal unnamed_addr global %struct.LIST_HELP* null, align 4
@.str6 = private unnamed_addr constant [11 x i8] c"splitlevel\00", align 1
@dfg_TERMLIST = internal unnamed_addr global %struct.LIST_HELP* null, align 4
@dfg_IGNORETEXT = common global i32 0, align 4
@.str7 = private unnamed_addr constant [22 x i8] c"\0A Undefined symbol %s\00", align 1
@.str8 = private unnamed_addr constant [19 x i8] c" in DomPred list.\0A\00", align 1
@.str9 = private unnamed_addr constant [30 x i8] c"\0A Symbol %s isn't a predicate\00", align 1
@.str10 = private unnamed_addr constant [24 x i8] c"\0A Found unknown flag %s\00", align 1
@dfg_FLAGS = internal unnamed_addr global i32* null, align 4
@.str11 = private unnamed_addr constant [23 x i8] c"\0A Undefined symbol %s \00", align 1
@.str12 = private unnamed_addr constant [22 x i8] c" in precedence list.\0A\00", align 1
@dfg_PRECEDENCE = internal unnamed_addr global i32* null, align 4
@dfg_USERPRECEDENCE = internal unnamed_addr global %struct.LIST_HELP* null, align 4
@.str13 = private unnamed_addr constant [21 x i8] c"in precedence list.\0A\00", align 1
@.str14 = private unnamed_addr constant [27 x i8] c"\0A Invalid symbol status %s\00", align 1
@.str15 = private unnamed_addr constant [21 x i8] c" in precedence list.\00", align 1
@yyr1 = internal unnamed_addr constant [197 x i8] c"\00GHIJKLMNNOOPPQQQRSSTTUUVVWWXXYYZZ[[\5C\5C]]^^__``aabbccddeeeefgehiijjkklmmnnooppqqqqrsqttuuvvvwwxxyyyyyzz{{||}~}\7F\7F\80\80\81\82\83\81\84\85\85\86\86\87\87\88\88\88\88\88\89\89\8A\8A\8B\8B\8C\8D\8D\8E\8E\8F\8F\91\90\92\92\93\93\94\94\95\95\97\96\98\98\98\98\98\98\98\98\98\99\99\99\9A\9A\9B\9B\9D\9C\9E\9E\9F\9F\A0\A1\A0\A2\A2\A3\A3\A4\A4\A4\A5\A5\A6\A6\A7\A7\A8\A8\A9\AA\AA", align 1
@yypgoto = internal unnamed_addr constant [100 x i16] [i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 392, i16 -356, i16 -356, i16 259, i16 -356, i16 -356, i16 -356, i16 -356, i16 202, i16 -356, i16 -356, i16 216, i16 -152, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 267, i16 -356, i16 -356, i16 -340, i16 -267, i16 -356, i16 -356, i16 -356, i16 70, i16 -356, i16 -356, i16 -356, i16 -3, i16 -355, i16 235, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 87, i16 -356, i16 -356, i16 33, i16 78, i16 68, i16 -356, i16 -45, i16 -356, i16 -356, i16 92, i16 39, i16 -101, i16 328, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -308, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 -356, i16 154, i16 -356, i16 -356, i16 425, i16 207], align 2
@yydefgoto = internal unnamed_addr constant [100 x i16] [i16 -1, i16 2, i16 14, i16 20, i16 27, i16 87, i16 122, i16 39, i16 54, i16 160, i16 157, i16 17, i16 18, i16 29, i16 56, i16 57, i16 42, i16 92, i16 93, i16 61, i16 129, i16 97, i16 170, i16 171, i16 133, i16 203, i16 204, i16 163, i16 24, i16 46, i16 74, i16 180, i16 244, i16 75, i16 143, i16 272, i16 217, i16 48, i16 112, i16 35, i16 222, i16 324, i16 343, i16 361, i16 398, i16 302, i16 344, i16 303, i16 304, i16 305, i16 76, i16 215, i16 216, i16 49, i16 80, i16 308, i16 307, i16 364, i16 365, i16 416, i16 439, i16 366, i16 401, i16 402, i16 432, i16 306, i16 330, i16 390, i16 391, i16 392, i16 145, i16 146, i16 81, i16 115, i16 279, i16 309, i16 453, i16 463, i16 467, i16 378, i16 394, i16 379, i16 412, i16 410, i16 116, i16 151, i16 226, i16 254, i16 22, i16 33, i16 101, i16 211, i16 238, i16 265, i16 316, i16 317, i16 396, i16 63, i16 64, i16 135], align 2
@yytname = internal unnamed_addr constant [172 x i8*] [i8* getelementptr inbounds ([5 x i8]* @.str60, i32 0, i32 0), i8* getelementptr inbounds ([6 x i8]* @.str61, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str62, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str63, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str64, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str65, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str66, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8]* @.str67, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str68, i32 0, i32 0), i8* getelementptr inbounds ([15 x i8]* @.str69, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str70, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str71, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str72, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str73, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8]* @.str74, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str75, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8]* @.str76, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str77, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str78, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str79, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str80, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str81, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str82, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str83, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str84, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8]* @.str85, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str86, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str87, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str88, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str89, i32 0, i32 0), i8* getelementptr inbounds ([14 x i8]* @.str90, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str91, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str92, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str93, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str94, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str95, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str96, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str97, i32 0, i32 0), i8* getelementptr inbounds ([14 x i8]* @.str98, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str99, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8]* @.str100, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str101, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str102, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str103, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str104, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str105, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str106, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str107, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8]* @.str108, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str109, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str110, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str111, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str112, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str113, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str114, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8]* @.str115, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str116, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str117, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str118, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str119, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str120, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str121, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8]* @.str122, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str123, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8]* @.str124, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8]* @.str125, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8]* @.str126, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8]* @.str127, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8]* @.str128, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8]* @.str129, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8]* @.str130, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str131, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str132, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str133, i32 0, i32 0), i8* getelementptr inbounds ([5 x i8]* @.str134, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8]* @.str135, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8]* @.str136, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str137, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str138, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str139, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str140, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str141, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str142, i32 0, i32 0), i8* getelementptr inbounds ([14 x i8]* @.str143, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8]* @.str144, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8]* @.str145, i32 0, i32 0), i8* getelementptr inbounds ([5 x i8]* @.str146, i32 0, i32 0), i8* getelementptr inbounds ([14 x i8]* @.str147, i32 0, i32 0), i8* getelementptr inbounds ([14 x i8]* @.str148, i32 0, i32 0), i8* getelementptr inbounds ([5 x i8]* @.str149, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str150, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str151, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8]* @.str152, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8]* @.str153, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8]* @.str154, i32 0, i32 0), i8* getelementptr inbounds ([15 x i8]* @.str155, i32 0, i32 0), i8* getelementptr inbounds ([15 x i8]* @.str156, i32 0, i32 0), i8* getelementptr inbounds ([6 x i8]* @.str157, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8]* @.str158, i32 0, i32 0), i8* getelementptr inbounds ([19 x i8]* @.str159, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str160, i32 0, i32 0), i8* getelementptr inbounds ([5 x i8]* @.str161, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8]* @.str162, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8]* @.str163, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str164, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str165, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str166, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str167, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str168, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8]* @.str169, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8]* @.str170, i32 0, i32 0), i8* getelementptr inbounds ([15 x i8]* @.str171, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str172, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str173, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8]* @.str174, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8]* @.str175, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str176, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str177, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str178, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str179, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str180, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8]* @.str181, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str182, i32 0, i32 0), i8* getelementptr inbounds ([6 x i8]* @.str183, i32 0, i32 0), i8* getelementptr inbounds ([15 x i8]* @.str184, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str185, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8]* @.str186, i32 0, i32 0), i8* getelementptr inbounds ([14 x i8]* @.str187, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8]* @.str188, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str189, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8]* @.str190, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8]* @.str191, i32 0, i32 0), i8* getelementptr inbounds ([14 x i8]* @.str192, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str193, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8]* @.str194, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str195, i32 0, i32 0), i8* getelementptr inbounds ([5 x i8]* @.str196, i32 0, i32 0), i8* getelementptr inbounds ([14 x i8]* @.str197, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8]* @.str198, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str199, i32 0, i32 0), i8* getelementptr inbounds ([14 x i8]* @.str200, i32 0, i32 0), i8* getelementptr inbounds ([5 x i8]* @.str201, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str202, i32 0, i32 0), i8* getelementptr inbounds ([14 x i8]* @.str203, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str204, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8]* @.str205, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8]* @.str206, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str207, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8]* @.str208, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str209, i32 0, i32 0), i8* getelementptr inbounds ([14 x i8]* @.str210, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8]* @.str211, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str212, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str213, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8]* @.str214, i32 0, i32 0), i8* getelementptr inbounds ([15 x i8]* @.str215, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str216, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8]* @.str217, i32 0, i32 0), i8* getelementptr inbounds ([6 x i8]* @.str218, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8]* @.str219, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str220, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8]* @.str221, i32 0, i32 0), i8* getelementptr inbounds ([6 x i8]* @.str222, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8]* @.str223, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str224, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str225, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str226, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str227, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str228, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str229, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8]* @.str230, i32 0, i32 0), i8* null], align 4
@.str16 = private unnamed_addr constant [25 x i8] c"parse error, unexpected \00", align 1
@.str17 = private unnamed_addr constant [13 x i8] c", expecting \00", align 1
@.str18 = private unnamed_addr constant [5 x i8] c" or \00", align 1
@.str20 = private unnamed_addr constant [12 x i8] c"parse error\00", align 1
@.str21 = private unnamed_addr constant [22 x i8] c"parser stack overflow\00", align 1
@.str22 = private unnamed_addr constant [15 x i8] c"\0A Line %i: %s\0A\00", align 1
@.str24 = private unnamed_addr constant [12 x i8] c"satisfiable\00", align 1
@.str25 = private unnamed_addr constant [14 x i8] c"unsatisfiable\00", align 1
@.str26 = private unnamed_addr constant [8 x i8] c"unknown\00", align 1
@stderr = external global %struct._IO_FILE*
@.str27 = private unnamed_addr constant [31 x i8] c"\0A\09Error in file %s at line %d\0A\00", align 1
@.str28 = private unnamed_addr constant [12 x i8] c"dfgparser.y\00", align 1
@.str29 = private unnamed_addr constant [47 x i8] c"\0A In dfg_ProblemStatusString: Invalid status.\0A\00", align 1
@.str30 = private unnamed_addr constant [133 x i8] c"\0A Please report this error via email to spass@mpi-sb.mpg.de including\0A the SPASS version, input problem, options, operating system.\0A\00", align 1
@.str31 = private unnamed_addr constant [30 x i8] c"list_of_descriptions.\0A  name(\00", align 1
@.str32 = private unnamed_addr constant [6 x i8] c"{* *}\00", align 1
@.str33 = private unnamed_addr constant [13 x i8] c").\0A  author(\00", align 1
@.str34 = private unnamed_addr constant [4 x i8] c").\0A\00", align 1
@.str35 = private unnamed_addr constant [11 x i8] c"  version(\00", align 1
@.str36 = private unnamed_addr constant [9 x i8] c"  logic(\00", align 1
@.str37 = private unnamed_addr constant [10 x i8] c"  status(\00", align 1
@.str38 = private unnamed_addr constant [18 x i8] c").\0A  description(\00", align 1
@.str39 = private unnamed_addr constant [8 x i8] c"  date(\00", align 1
@.str40 = private unnamed_addr constant [13 x i8] c"end_of_list.\00", align 1
@dfg_VARLIST = internal unnamed_addr global %struct.LIST_HELP* null, align 4
@.str41 = private unnamed_addr constant [55 x i8] c"\0A In dfg_VarCheck: List of variables should be empty!\0A\00", align 1
@symbol_STANDARDVARCOUNTER = external global i32
@memory_FREEDBYTES = external global i32
@memory_ARRAY = external global [0 x %struct.MEMORY_RESOURCEHELP*]
@dfg_VARDECL = internal unnamed_addr global i1 false
@dfg_SYMBOLLIST = internal unnamed_addr global %struct.LIST_HELP* null, align 4
@symbol_TYPESTATBITS = external constant i32
@symbol_SIGNATURE = external global %struct.signature**
@.str42 = private unnamed_addr constant [44 x i8] c"\0A Line %d: Symbol is not a sort predicate.\0A\00", align 1
@.str43 = private unnamed_addr constant [33 x i8] c"\0A Line %d: undefined symbol %s.\0A\00", align 1
@.str44 = private unnamed_addr constant [38 x i8] c"\0A Line %d: Symbol is not a function.\0A\00", align 1
@symbol_TYPEMASK = external constant i32
@fol_FALSE = external global i32
@fol_TRUE = external global i32
@.str45 = private unnamed_addr constant [33 x i8] c"\0A Line %d: Undefined symbol %s.\0A\00", align 1
@.str46 = private unnamed_addr constant [30 x i8] c"\0A Line %u: Free Variable %s.\0A\00", align 1
@.str47 = private unnamed_addr constant [11 x i8] c"\0A Line %u:\00", align 1
@.str48 = private unnamed_addr constant [21 x i8] c" The actual arity %u\00", align 1
@.str49 = private unnamed_addr constant [22 x i8] c" of symbol %s differs\00", align 1
@.str50 = private unnamed_addr constant [30 x i8] c" from the previous arity %u.\0A\00", align 1
@.str51 = private unnamed_addr constant [50 x i8] c"\0A Line %u: Symbol %s was declared with arity %u.\0A\00", align 1
@.str52 = private unnamed_addr constant [58 x i8] c"\0A Line %u: symbols with arbitrary arity are not allowed.\0A\00", align 1
@.str53 = private unnamed_addr constant [46 x i8] c"\0A Line %u: symbol %s was already declared as \00", align 1
@.str54 = private unnamed_addr constant [11 x i8] c"function.\0A\00", align 1
@.str55 = private unnamed_addr constant [12 x i8] c"predicate.\0A\00", align 1
@.str56 = private unnamed_addr constant [10 x i8] c"junctor.\0A\00", align 1
@.str57 = private unnamed_addr constant [15 x i8] c"unknown type.\0A\00", align 1
@.str58 = private unnamed_addr constant [57 x i8] c"\0A Line %u: symbol %s was already declared with arity %d\0A\00", align 1
@stack_POINTER = external global i32
@dfg_in = external global %struct._IO_FILE*
@.str59 = private unnamed_addr constant [3 x i8] c"\0A\0A\00", align 1
@.str60 = private unnamed_addr constant [5 x i8] c"$end\00", align 1
@.str61 = private unnamed_addr constant [6 x i8] c"error\00", align 1
@.str62 = private unnamed_addr constant [11 x i8] c"$undefined\00", align 1
@.str63 = private unnamed_addr constant [8 x i8] c"DFG_AND\00", align 1
@.str64 = private unnamed_addr constant [11 x i8] c"DFG_AUTHOR\00", align 1
@.str65 = private unnamed_addr constant [11 x i8] c"DFG_AXIOMS\00", align 1
@.str66 = private unnamed_addr constant [12 x i8] c"DFG_BEGPROB\00", align 1
@.str67 = private unnamed_addr constant [7 x i8] c"DFG_BY\00", align 1
@.str68 = private unnamed_addr constant [11 x i8] c"DFG_CLAUSE\00", align 1
@.str69 = private unnamed_addr constant [15 x i8] c"DFG_CLOSEBRACE\00", align 1
@.str70 = private unnamed_addr constant [12 x i8] c"DFG_CLSLIST\00", align 1
@.str71 = private unnamed_addr constant [8 x i8] c"DFG_CNF\00", align 1
@.str72 = private unnamed_addr constant [12 x i8] c"DFG_CONJECS\00", align 1
@.str73 = private unnamed_addr constant [9 x i8] c"DFG_DATE\00", align 1
@.str74 = private unnamed_addr constant [13 x i8] c"DFG_DECLLIST\00", align 1
@.str75 = private unnamed_addr constant [9 x i8] c"DFG_DESC\00", align 1
@.str76 = private unnamed_addr constant [13 x i8] c"DFG_DESCLIST\00", align 1
@.str77 = private unnamed_addr constant [8 x i8] c"DFG_DNF\00", align 1
@.str78 = private unnamed_addr constant [12 x i8] c"DFG_DOMPRED\00", align 1
@.str79 = private unnamed_addr constant [12 x i8] c"DFG_ENDLIST\00", align 1
@.str80 = private unnamed_addr constant [12 x i8] c"DFG_ENDPROB\00", align 1
@.str81 = private unnamed_addr constant [10 x i8] c"DFG_EQUAL\00", align 1
@.str82 = private unnamed_addr constant [10 x i8] c"DFG_EQUIV\00", align 1
@.str83 = private unnamed_addr constant [11 x i8] c"DFG_EXISTS\00", align 1
@.str84 = private unnamed_addr constant [10 x i8] c"DFG_FALSE\00", align 1
@.str85 = private unnamed_addr constant [13 x i8] c"DFG_FORMLIST\00", align 1
@.str86 = private unnamed_addr constant [12 x i8] c"DFG_FORMULA\00", align 1
@.str87 = private unnamed_addr constant [11 x i8] c"DFG_FORALL\00", align 1
@.str88 = private unnamed_addr constant [11 x i8] c"DFG_FREELY\00", align 1
@.str89 = private unnamed_addr constant [9 x i8] c"DFG_FUNC\00", align 1
@.str90 = private unnamed_addr constant [14 x i8] c"DFG_GENERATED\00", align 1
@.str91 = private unnamed_addr constant [11 x i8] c"DFG_GENSET\00", align 1
@.str92 = private unnamed_addr constant [11 x i8] c"DFG_HYPOTH\00", align 1
@.str93 = private unnamed_addr constant [12 x i8] c"DFG_IMPLIED\00", align 1
@.str94 = private unnamed_addr constant [12 x i8] c"DFG_IMPLIES\00", align 1
@.str95 = private unnamed_addr constant [10 x i8] c"DFG_LOGIC\00", align 1
@.str96 = private unnamed_addr constant [9 x i8] c"DFG_NAME\00", align 1
@.str97 = private unnamed_addr constant [8 x i8] c"DFG_NOT\00", align 1
@.str98 = private unnamed_addr constant [14 x i8] c"DFG_OPENBRACE\00", align 1
@.str99 = private unnamed_addr constant [11 x i8] c"DFG_OPERAT\00", align 1
@.str100 = private unnamed_addr constant [7 x i8] c"DFG_OR\00", align 1
@.str101 = private unnamed_addr constant [9 x i8] c"DFG_PREC\00", align 1
@.str102 = private unnamed_addr constant [9 x i8] c"DFG_PRED\00", align 1
@.str103 = private unnamed_addr constant [12 x i8] c"DFG_PRDICAT\00", align 1
@.str104 = private unnamed_addr constant [12 x i8] c"DFG_PRFLIST\00", align 1
@.str105 = private unnamed_addr constant [12 x i8] c"DFG_QUANTIF\00", align 1
@.str106 = private unnamed_addr constant [10 x i8] c"DFG_SATIS\00", align 1
@.str107 = private unnamed_addr constant [12 x i8] c"DFG_SETFLAG\00", align 1
@.str108 = private unnamed_addr constant [13 x i8] c"DFG_SETTINGS\00", align 1
@.str109 = private unnamed_addr constant [12 x i8] c"DFG_SYMLIST\00", align 1
@.str110 = private unnamed_addr constant [9 x i8] c"DFG_SORT\00", align 1
@.str111 = private unnamed_addr constant [10 x i8] c"DFG_SORTS\00", align 1
@.str112 = private unnamed_addr constant [11 x i8] c"DFG_STATUS\00", align 1
@.str113 = private unnamed_addr constant [9 x i8] c"DFG_STEP\00", align 1
@.str114 = private unnamed_addr constant [12 x i8] c"DFG_SUBSORT\00", align 1
@.str115 = private unnamed_addr constant [13 x i8] c"DFG_TERMLIST\00", align 1
@.str116 = private unnamed_addr constant [9 x i8] c"DFG_TRUE\00", align 1
@.str117 = private unnamed_addr constant [12 x i8] c"DFG_UNKNOWN\00", align 1
@.str118 = private unnamed_addr constant [12 x i8] c"DFG_UNSATIS\00", align 1
@.str119 = private unnamed_addr constant [12 x i8] c"DFG_VERSION\00", align 1
@.str120 = private unnamed_addr constant [8 x i8] c"DFG_NUM\00", align 1
@.str121 = private unnamed_addr constant [11 x i8] c"DFG_MINUS1\00", align 1
@.str122 = private unnamed_addr constant [7 x i8] c"DFG_ID\00", align 1
@.str123 = private unnamed_addr constant [9 x i8] c"DFG_TEXT\00", align 1
@.str124 = private unnamed_addr constant [4 x i8] c"'('\00", align 1
@.str125 = private unnamed_addr constant [4 x i8] c"')'\00", align 1
@.str126 = private unnamed_addr constant [4 x i8] c"'.'\00", align 1
@.str127 = private unnamed_addr constant [4 x i8] c"'['\00", align 1
@.str128 = private unnamed_addr constant [4 x i8] c"']'\00", align 1
@.str129 = private unnamed_addr constant [4 x i8] c"','\00", align 1
@.str130 = private unnamed_addr constant [4 x i8] c"':'\00", align 1
@.str131 = private unnamed_addr constant [8 x i8] c"$accept\00", align 1
@.str132 = private unnamed_addr constant [8 x i8] c"problem\00", align 1
@.str133 = private unnamed_addr constant [12 x i8] c"description\00", align 1
@.str134 = private unnamed_addr constant [5 x i8] c"name\00", align 1
@.str135 = private unnamed_addr constant [7 x i8] c"author\00", align 1
@.str136 = private unnamed_addr constant [7 x i8] c"status\00", align 1
@.str137 = private unnamed_addr constant [9 x i8] c"desctext\00", align 1
@.str138 = private unnamed_addr constant [11 x i8] c"versionopt\00", align 1
@.str139 = private unnamed_addr constant [9 x i8] c"logicopt\00", align 1
@.str140 = private unnamed_addr constant [8 x i8] c"dateopt\00", align 1
@.str141 = private unnamed_addr constant [10 x i8] c"log_state\00", align 1
@.str142 = private unnamed_addr constant [12 x i8] c"logicalpart\00", align 1
@.str143 = private unnamed_addr constant [14 x i8] c"symbollistopt\00", align 1
@.str144 = private unnamed_addr constant [13 x i8] c"functionsopt\00", align 1
@.str145 = private unnamed_addr constant [13 x i8] c"functionlist\00", align 1
@.str146 = private unnamed_addr constant [5 x i8] c"func\00", align 1
@.str147 = private unnamed_addr constant [14 x i8] c"predicatesopt\00", align 1
@.str148 = private unnamed_addr constant [14 x i8] c"predicatelist\00", align 1
@.str149 = private unnamed_addr constant [5 x i8] c"pred\00", align 1
@.str150 = private unnamed_addr constant [9 x i8] c"sortsopt\00", align 1
@.str151 = private unnamed_addr constant [9 x i8] c"sortlist\00", align 1
@.str152 = private unnamed_addr constant [13 x i8] c"operatorsopt\00", align 1
@.str153 = private unnamed_addr constant [13 x i8] c"operatorlist\00", align 1
@.str154 = private unnamed_addr constant [3 x i8] c"op\00", align 1
@.str155 = private unnamed_addr constant [15 x i8] c"quantifiersopt\00", align 1
@.str156 = private unnamed_addr constant [15 x i8] c"quantifierlist\00", align 1
@.str157 = private unnamed_addr constant [6 x i8] c"quant\00", align 1
@.str158 = private unnamed_addr constant [7 x i8] c"number\00", align 1
@.str159 = private unnamed_addr constant [19 x i8] c"declarationlistopt\00", align 1
@.str160 = private unnamed_addr constant [12 x i8] c"decllistopt\00", align 1
@.str161 = private unnamed_addr constant [5 x i8] c"decl\00", align 1
@.str162 = private unnamed_addr constant [3 x i8] c"@1\00", align 1
@.str163 = private unnamed_addr constant [3 x i8] c"@2\00", align 1
@.str164 = private unnamed_addr constant [8 x i8] c"gendecl\00", align 1
@.str165 = private unnamed_addr constant [10 x i8] c"freelyopt\00", align 1
@.str166 = private unnamed_addr constant [9 x i8] c"funclist\00", align 1
@.str167 = private unnamed_addr constant [9 x i8] c"sortdecl\00", align 1
@.str168 = private unnamed_addr constant [12 x i8] c"formulalist\00", align 1
@.str169 = private unnamed_addr constant [7 x i8] c"origin\00", align 1
@.str170 = private unnamed_addr constant [16 x i8] c"formulalistsopt\00", align 1
@.str171 = private unnamed_addr constant [15 x i8] c"formulalistopt\00", align 1
@.str172 = private unnamed_addr constant [9 x i8] c"labelopt\00", align 1
@.str173 = private unnamed_addr constant [8 x i8] c"formula\00", align 1
@.str174 = private unnamed_addr constant [3 x i8] c"@3\00", align 1
@.str175 = private unnamed_addr constant [3 x i8] c"@4\00", align 1
@.str176 = private unnamed_addr constant [11 x i8] c"formulaopt\00", align 1
@.str177 = private unnamed_addr constant [8 x i8] c"arglist\00", align 1
@.str178 = private unnamed_addr constant [10 x i8] c"binsymbol\00", align 1
@.str179 = private unnamed_addr constant [8 x i8] c"nsymbol\00", align 1
@.str180 = private unnamed_addr constant [12 x i8] c"quantsymbol\00", align 1
@.str181 = private unnamed_addr constant [3 x i8] c"id\00", align 1
@.str182 = private unnamed_addr constant [10 x i8] c"qtermlist\00", align 1
@.str183 = private unnamed_addr constant [6 x i8] c"qterm\00", align 1
@.str184 = private unnamed_addr constant [15 x i8] c"clauselistsopt\00", align 1
@.str185 = private unnamed_addr constant [11 x i8] c"clauselist\00", align 1
@.str186 = private unnamed_addr constant [3 x i8] c"@5\00", align 1
@.str187 = private unnamed_addr constant [14 x i8] c"cnfclausesopt\00", align 1
@.str188 = private unnamed_addr constant [13 x i8] c"cnfclauseopt\00", align 1
@.str189 = private unnamed_addr constant [10 x i8] c"cnfclause\00", align 1
@.str190 = private unnamed_addr constant [3 x i8] c"@6\00", align 1
@.str191 = private unnamed_addr constant [3 x i8] c"@7\00", align 1
@.str192 = private unnamed_addr constant [14 x i8] c"cnfclausebody\00", align 1
@.str193 = private unnamed_addr constant [8 x i8] c"litlist\00", align 1
@.str194 = private unnamed_addr constant [4 x i8] c"lit\00", align 1
@.str195 = private unnamed_addr constant [9 x i8] c"atomlist\00", align 1
@.str196 = private unnamed_addr constant [5 x i8] c"atom\00", align 1
@.str197 = private unnamed_addr constant [14 x i8] c"dnfclausesopt\00", align 1
@.str198 = private unnamed_addr constant [13 x i8] c"dnfclauseopt\00", align 1
@.str199 = private unnamed_addr constant [10 x i8] c"dnfclause\00", align 1
@.str200 = private unnamed_addr constant [14 x i8] c"dnfclausebody\00", align 1
@.str201 = private unnamed_addr constant [5 x i8] c"term\00", align 1
@.str202 = private unnamed_addr constant [9 x i8] c"termlist\00", align 1
@.str203 = private unnamed_addr constant [14 x i8] c"prooflistsopt\00", align 1
@.str204 = private unnamed_addr constant [10 x i8] c"prooflist\00", align 1
@.str205 = private unnamed_addr constant [3 x i8] c"@8\00", align 1
@.str206 = private unnamed_addr constant [13 x i8] c"prooflistopt\00", align 1
@.str207 = private unnamed_addr constant [11 x i8] c"parentlist\00", align 1
@.str208 = private unnamed_addr constant [13 x i8] c"assoclistopt\00", align 1
@.str209 = private unnamed_addr constant [10 x i8] c"assoclist\00", align 1
@.str210 = private unnamed_addr constant [14 x i8] c"id_or_formula\00", align 1
@.str211 = private unnamed_addr constant [3 x i8] c"@9\00", align 1
@.str212 = private unnamed_addr constant [10 x i8] c"anysymbol\00", align 1
@.str213 = private unnamed_addr constant [8 x i8] c"optargs\00", align 1
@.str214 = private unnamed_addr constant [7 x i8] c"clause\00", align 1
@.str215 = private unnamed_addr constant [15 x i8] c"listOfTermsopt\00", align 1
@.str216 = private unnamed_addr constant [12 x i8] c"listOfTerms\00", align 1
@.str217 = private unnamed_addr constant [4 x i8] c"@10\00", align 1
@.str218 = private unnamed_addr constant [6 x i8] c"terms\00", align 1
@.str219 = private unnamed_addr constant [16 x i8] c"settinglistsopt\00", align 1
@.str220 = private unnamed_addr constant [12 x i8] c"settinglist\00", align 1
@.str221 = private unnamed_addr constant [4 x i8] c"@11\00", align 1
@.str222 = private unnamed_addr constant [6 x i8] c"flags\00", align 1
@.str223 = private unnamed_addr constant [11 x i8] c"spassflags\00", align 1
@.str224 = private unnamed_addr constant [10 x i8] c"spassflag\00", align 1
@.str225 = private unnamed_addr constant [9 x i8] c"preclist\00", align 1
@.str226 = private unnamed_addr constant [9 x i8] c"precitem\00", align 1
@.str227 = private unnamed_addr constant [8 x i8] c"statopt\00", align 1
@.str228 = private unnamed_addr constant [10 x i8] c"gsettings\00", align 1
@.str229 = private unnamed_addr constant [9 x i8] c"gsetting\00", align 1
@.str230 = private unnamed_addr constant [10 x i8] c"labellist\00", align 1
@.str231 = private unnamed_addr constant [50 x i8] c"\0A Error: Flag value %d is too small for flag %s.\0A\00", align 1
@.str232 = private unnamed_addr constant [50 x i8] c"\0A Error: Flag value %d is too large for flag %s.\0A\00", align 1
@.str233 = private unnamed_addr constant [31 x i8] c"\0A Line %d: is not a function.\0A\00", align 1
@fol_EQUALITY = external global i32
@stack_STACK = external global [10000 x i8*]
@fol_EXIST = external global i32
@fol_OR = external global i32
@fol_AND = external global i32
@fol_IMPLIES = external global i32
@fol_IMPLIED = external global i32
@fol_EQUIV = external global i32
@fol_NOT = external global i32
@fol_ALL = external global i32

; Function Attrs: nounwind
define i32 @dfg_parse() #0 {
entry:
  %yyssa = alloca [200 x i16], align 2
  %yyvsa = alloca [200 x %union.yystype], align 4
  %yyval = alloca %union.yystype, align 4
  %0 = bitcast [200 x i16]* %yyssa to i8*
  call void @llvm.lifetime.start(i64 400, i8* %0) #1
  %arraydecay = getelementptr inbounds [200 x i16]* %yyssa, i32 0, i32 0
  %1 = bitcast [200 x %union.yystype]* %yyvsa to i8*
  call void @llvm.lifetime.start(i64 800, i8* %1) #1
  %arraydecay1 = getelementptr inbounds [200 x %union.yystype]* %yyvsa, i32 0, i32 0
  store i32 0, i32* @dfg_nerrs, align 4
  store i32 -2, i32* @dfg_char, align 4
  %2 = getelementptr inbounds %union.yystype* %yyval, i32 0, i32 0
  %3 = load i32* @symbol_TYPEMASK, align 4
  %4 = load i32* @symbol_TYPESTATBITS, align 4
  br label %yysetstate

yynewstate:                                       ; preds = %if.then1223, %if.else1226, %if.end100
  %yyvsp.0 = phi %union.yystype* [ %incdec.ptr1204, %if.then1223 ], [ %incdec.ptr1204, %if.else1226 ], [ %incdec.ptr101, %if.end100 ]
  %yyssp.0 = phi i16* [ %add.ptr1203, %if.then1223 ], [ %add.ptr1203, %if.else1226 ], [ %yyssp.2, %if.end100 ]
  %yystate.0 = phi i32 [ %conv1225, %if.then1223 ], [ %conv1229, %if.else1226 ], [ %conv81, %if.end100 ]
  %incdec.ptr = getelementptr inbounds i16* %yyssp.0, i32 1
  br label %yysetstate

yysetstate:                                       ; preds = %yynewstate, %entry
  %yystacksize.0 = phi i32 [ 200, %entry ], [ %yystacksize.1, %yynewstate ]
  %yyvsp.1 = phi %union.yystype* [ %arraydecay1, %entry ], [ %yyvsp.0, %yynewstate ]
  %yyvs.0 = phi %union.yystype* [ %arraydecay1, %entry ], [ %yyvs.1, %yynewstate ]
  %yyssp.1 = phi i16* [ %arraydecay, %entry ], [ %incdec.ptr, %yynewstate ]
  %yyss.0 = phi i16* [ %arraydecay, %entry ], [ %yyss.1, %yynewstate ]
  %yystate.1 = phi i32 [ 0, %entry ], [ %yystate.0, %yynewstate ]
  %conv = trunc i32 %yystate.1 to i16
  store i16 %conv, i16* %yyssp.1, align 2
  %add.ptr.sum = add i32 %yystacksize.0, -1
  %add.ptr2 = getelementptr inbounds i16* %yyss.0, i32 %add.ptr.sum
  %cmp = icmp ult i16* %yyssp.1, %add.ptr2
  br i1 %cmp, label %yybackup, label %if.then

if.then:                                          ; preds = %yysetstate
  %sub.ptr.lhs.cast = ptrtoint i16* %yyssp.1 to i32
  %sub.ptr.rhs.cast = ptrtoint i16* %yyss.0 to i32
  %sub.ptr.sub = sub i32 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %sub.ptr.div = ashr exact i32 %sub.ptr.sub, 1
  %add = add nsw i32 %sub.ptr.div, 1
  %cmp4 = icmp ugt i32 %yystacksize.0, 9999
  br i1 %cmp4, label %yyoverflowlab, label %if.end

if.end:                                           ; preds = %if.then
  %mul = shl i32 %yystacksize.0, 1
  %cmp7 = icmp ugt i32 %mul, 10000
  %.mul = select i1 %cmp7, i32 10000, i32 %mul
  %mul11 = mul i32 %.mul, 6
  %add121756 = or i32 %mul11, 3
  %5 = alloca i8, i32 %add121756, align 4
  %6 = bitcast i8* %5 to %union.yyalloc*
  %yyss15 = bitcast i8* %5 to i16*
  %7 = bitcast i16* %yyss.0 to i8*
  %mul16 = shl i32 %add, 1
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %5, i8* %7, i32 %mul16, i32 2, i1 false)
  %8 = lshr exact i32 %.mul, 1
  %div = and i32 %8, 1073741823
  %yyvs23 = getelementptr inbounds %union.yyalloc* %6, i32 %div, i32 0
  %9 = bitcast %union.yystype* %yyvs23 to i8*
  %10 = bitcast %union.yystype* %yyvs.0 to i8*
  %mul24 = shl i32 %add, 2
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %9, i8* %10, i32 %mul24, i32 4, i1 false)
  %add.ptr41 = getelementptr inbounds i16* %yyss15, i32 %sub.ptr.div
  %add.ptr43 = getelementptr inbounds %union.yystype* %yyvs23, i32 %sub.ptr.div
  %add.ptr44.sum = add i32 %.mul, -1
  %cmp46 = icmp slt i32 %sub.ptr.div, %add.ptr44.sum
  br i1 %cmp46, label %yybackup, label %yyreturn

yybackup:                                         ; preds = %if.end, %yysetstate
  %yystacksize.1 = phi i32 [ %.mul, %if.end ], [ %yystacksize.0, %yysetstate ]
  %yyvsp.2 = phi %union.yystype* [ %add.ptr43, %if.end ], [ %yyvsp.1, %yysetstate ]
  %yyvs.1 = phi %union.yystype* [ %yyvs23, %if.end ], [ %yyvs.0, %yysetstate ]
  %yyssp.2 = phi i16* [ %add.ptr41, %if.end ], [ %yyssp.1, %yysetstate ]
  %yyss.1 = phi i16* [ %yyss15, %if.end ], [ %yyss.0, %yysetstate ]
  %arrayidx = getelementptr inbounds [477 x i16]* @yypact, i32 0, i32 %yystate.1
  %11 = load i16* %arrayidx, align 2
  %conv51 = sext i16 %11 to i32
  %cmp52 = icmp eq i16 %11, -356
  br i1 %cmp52, label %yydefault, label %if.end55

if.end55:                                         ; preds = %yybackup
  %12 = load i32* @dfg_char, align 4
  %cmp56 = icmp eq i32 %12, -2
  br i1 %cmp56, label %if.then58, label %if.end59

if.then58:                                        ; preds = %if.end55
  %call = call i32 @dfg_lex() #1
  store i32 %call, i32* @dfg_char, align 4
  br label %if.end59

if.end59:                                         ; preds = %if.then58, %if.end55
  %13 = phi i32 [ %call, %if.then58 ], [ %12, %if.end55 ]
  %cmp60 = icmp slt i32 %13, 1
  br i1 %cmp60, label %if.then62, label %if.else

if.then62:                                        ; preds = %if.end59
  store i32 0, i32* @dfg_char, align 4
  br label %if.end67

if.else:                                          ; preds = %if.end59
  %cmp63 = icmp ult i32 %13, 319
  br i1 %cmp63, label %cond.true, label %if.end67

cond.true:                                        ; preds = %if.else
  %arrayidx65 = getelementptr inbounds [319 x i8]* @yytranslate, i32 0, i32 %13
  %14 = load i8* %arrayidx65, align 1
  %conv66 = zext i8 %14 to i32
  br label %if.end67

if.end67:                                         ; preds = %cond.true, %if.else, %if.then62
  %15 = phi i32 [ 0, %if.then62 ], [ %13, %cond.true ], [ %13, %if.else ]
  %yychar1.2 = phi i32 [ 0, %if.then62 ], [ %conv66, %cond.true ], [ 2, %if.else ]
  %add68 = add nsw i32 %yychar1.2, %conv51
  %16 = icmp ugt i32 %add68, 506
  br i1 %16, label %yydefault, label %lor.lhs.false73

lor.lhs.false73:                                  ; preds = %if.end67
  %arrayidx74 = getelementptr inbounds [507 x i16]* @yycheck, i32 0, i32 %add68
  %17 = load i16* %arrayidx74, align 2
  %conv75 = sext i16 %17 to i32
  %cmp76 = icmp eq i32 %conv75, %yychar1.2
  br i1 %cmp76, label %if.end79, label %yydefault

if.end79:                                         ; preds = %lor.lhs.false73
  %arrayidx80 = getelementptr inbounds [507 x i16]* @yytable, i32 0, i32 %add68
  %18 = load i16* %arrayidx80, align 2
  %conv81 = zext i16 %18 to i32
  %cmp82 = icmp eq i16 %18, 0
  br i1 %cmp82, label %if.then1232, label %if.end92

if.end92:                                         ; preds = %if.end79
  %cmp93 = icmp eq i32 %add68, 35
  br i1 %cmp93, label %yyreturn, label %if.end96

if.end96:                                         ; preds = %if.end92
  %cmp97 = icmp eq i32 %15, 0
  br i1 %cmp97, label %if.end100, label %if.then99

if.then99:                                        ; preds = %if.end96
  store i32 -2, i32* @dfg_char, align 4
  br label %if.end100

if.end100:                                        ; preds = %if.end96, %if.then99
  %incdec.ptr101 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 1
  %19 = load i32* getelementptr inbounds (%union.yystype* @dfg_lval, i32 0, i32 0), align 4
  %20 = getelementptr inbounds %union.yystype* %incdec.ptr101, i32 0, i32 0
  store i32 %19, i32* %20, align 4
  br label %yynewstate

yydefault:                                        ; preds = %lor.lhs.false73, %if.end67, %yybackup
  %arrayidx105 = getelementptr inbounds [477 x i8]* @yydefact, i32 0, i32 %yystate.1
  %21 = load i8* %arrayidx105, align 1
  %conv106 = zext i8 %21 to i32
  %cmp107 = icmp eq i8 %21, 0
  br i1 %cmp107, label %if.then1232, label %yyreduce

yyreduce:                                         ; preds = %yydefault
  %arrayidx111 = getelementptr inbounds [197 x i8]* @yyr2, i32 0, i32 %conv106
  %22 = load i8* %arrayidx111, align 1
  %conv112 = zext i8 %22 to i32
  %sub113 = sub i32 1, %conv112
  %23 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 %sub113, i32 0
  %24 = load i32* %23, align 4
  store i32 %24, i32* %2, align 4
  switch i32 %conv106, label %sw.epilog1200 [
    i32 2, label %sw.bb
    i32 4, label %sw.bb116
    i32 5, label %sw.bb119
    i32 6, label %sw.bb122
    i32 7, label %sw.bb124
    i32 9, label %sw.bb127
    i32 11, label %sw.bb130
    i32 13, label %sw.bb133
    i32 14, label %sw.bb136
    i32 15, label %sw.bb138
    i32 16, label %sw.bb140
    i32 24, label %sw.bb142
    i32 25, label %sw.bb145
    i32 30, label %sw.bb149
    i32 31, label %sw.bb152
    i32 34, label %sw.bb157
    i32 35, label %sw.bb160
    i32 40, label %sw.bb163
    i32 41, label %sw.bb166
    i32 46, label %sw.bb171
    i32 47, label %sw.bb174
    i32 48, label %sw.bb179
    i32 49, label %sw.bb181
    i32 55, label %sw.bb185
    i32 56, label %sw.bb190
    i32 57, label %sw.bb195
    i32 58, label %sw.bb198
    i32 59, label %sw.bb199
    i32 60, label %sw.bb200
    i32 61, label %sw.bb210
    i32 62, label %sw.bb217
    i32 63, label %sw.bb219
    i32 64, label %sw.bb221
    i32 65, label %sw.bb226
    i32 66, label %sw.bb233
    i32 67, label %sw.bb236
    i32 68, label %sw.bb239
    i32 69, label %sw.bb255
    i32 70, label %sw.bb257
    i32 73, label %sw.bb259
    i32 74, label %sw.bb262
    i32 75, label %sw.bb290
    i32 76, label %sw.bb292
    i32 77, label %sw.bb296
    i32 78, label %sw.bb300
    i32 79, label %sw.bb312
    i32 80, label %sw.bb327
    i32 81, label %sw.bb339
    i32 82, label %sw.bb340
    i32 83, label %sw.bb341
    i32 84, label %sw.bb355
    i32 85, label %sw.bb357
    i32 86, label %sw.bb361
    i32 87, label %sw.bb372
    i32 88, label %sw.bb387
    i32 89, label %sw.bb390
    i32 90, label %sw.bb393
    i32 91, label %sw.bb396
    i32 92, label %sw.bb399
    i32 93, label %sw.bb402
    i32 94, label %sw.bb405
    i32 95, label %sw.bb408
    i32 96, label %sw.bb419
    i32 97, label %sw.bb429
    i32 98, label %sw.bb437
    i32 99, label %sw.bb445
    i32 100, label %sw.bb453
    i32 101, label %sw.bb464
    i32 102, label %sw.bb479
    i32 103, label %sw.bb494
    i32 106, label %sw.bb519
    i32 107, label %sw.bb535
    i32 108, label %sw.bb536
    i32 109, label %sw.bb538
    i32 110, label %sw.bb541
    i32 111, label %sw.bb570
    i32 112, label %sw.bb572
    i32 113, label %sw.bb576
    i32 114, label %sw.bb580
    i32 115, label %sw.bb581
    i32 116, label %sw.bb582
    i32 117, label %sw.bb595
    i32 118, label %sw.bb606
    i32 119, label %sw.bb617
    i32 120, label %sw.bb632
    i32 121, label %sw.bb636
    i32 122, label %sw.bb650
    i32 123, label %sw.bb655
    i32 124, label %sw.bb663
    i32 125, label %sw.bb674
    i32 126, label %sw.bb684
    i32 127, label %sw.bb694
    i32 128, label %sw.bb709
    i32 136, label %sw.bb721
    i32 137, label %sw.bb732
    i32 138, label %sw.bb744
    i32 139, label %sw.bb755
    i32 142, label %sw.bb770
    i32 143, label %sw.bb777
    i32 145, label %sw.bb787
    i32 146, label %sw.bb851
    i32 147, label %sw.bb867
    i32 148, label %sw.bb887
    i32 149, label %sw.bb889
    i32 150, label %sw.bb893
    i32 151, label %sw.bb934
    i32 152, label %sw.bb977
    i32 153, label %sw.bb978
    i32 154, label %sw.bb998
    i32 155, label %sw.bb1002
    i32 156, label %sw.bb1004
    i32 157, label %sw.bb1006
    i32 158, label %sw.bb1008
    i32 159, label %sw.bb1010
    i32 160, label %sw.bb1012
    i32 161, label %sw.bb1014
    i32 162, label %sw.bb1016
    i32 163, label %sw.bb1018
    i32 164, label %sw.bb1020
    i32 165, label %sw.bb1022
    i32 166, label %sw.bb1024
    i32 167, label %sw.bb1028
    i32 170, label %sw.bb1030
    i32 171, label %sw.bb1031
    i32 173, label %sw.bb1032
    i32 177, label %sw.bb1037
    i32 178, label %sw.bb1046
    i32 179, label %sw.bb1047
    i32 184, label %for.cond.preheader
    i32 185, label %sw.bb1084
    i32 188, label %sw.bb1099
    i32 189, label %sw.bb1114
    i32 190, label %sw.bb1138
    i32 191, label %sw.bb1140
    i32 194, label %sw.bb1184
    i32 195, label %sw.bb1187
    i32 196, label %sw.bb1192
  ]

for.cond.preheader:                               ; preds = %yyreduce
  %arrayidx1052 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %list1053 = bitcast %union.yystype* %arrayidx1052 to %struct.LIST_HELP**
  %25 = load %struct.LIST_HELP** %list1053, align 4
  %cmp.i21612226 = icmp eq %struct.LIST_HELP* %25, null
  br i1 %cmp.i21612226, label %sw.epilog1200, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %for.cond.preheader
  %26 = getelementptr inbounds %union.yystype* %arrayidx1052, i32 0, i32 0
  br label %for.body

sw.bb:                                            ; preds = %yyreduce
  %arrayidx115 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -7
  %string = bitcast %union.yystype* %arrayidx115 to i8**
  %27 = load i8** %string, align 4
  call void @string_StringFree(i8* %27) #1
  br label %yyreturn

sw.bb116:
  %arrayidx117 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %string118 = bitcast %union.yystype* %arrayidx117 to i8**
  %28 = load i8** %string118, align 4
  store i8* %28, i8** @dfg_DESC.0, align 4
  br label %sw.epilog1200

sw.bb119:                                         ; preds = %yyreduce
  %arrayidx120 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %string121 = bitcast %union.yystype* %arrayidx120 to i8**
  %29 = load i8** %string121, align 4
  store i8* %29, i8** @dfg_DESC.1, align 4
  br label %sw.epilog1200

sw.bb122:                                         ; preds = %yyreduce
  %state = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2, i32 0
  %30 = load i32* %state, align 4
  store i32 %30, i32* @dfg_DESC.4, align 4
  br label %sw.epilog1200

sw.bb124:                                         ; preds = %yyreduce
  %arrayidx125 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %string126 = bitcast %union.yystype* %arrayidx125 to i8**
  %31 = load i8** %string126, align 4
  store i8* %31, i8** @dfg_DESC.5, align 4
  br label %sw.epilog1200

sw.bb127:                                         ; preds = %yyreduce
  %arrayidx128 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %string129 = bitcast %union.yystype* %arrayidx128 to i8**
  %32 = load i8** %string129, align 4
  store i8* %32, i8** @dfg_DESC.2, align 4
  br label %sw.epilog1200

sw.bb130:                                         ; preds = %yyreduce
  %arrayidx131 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %string132 = bitcast %union.yystype* %arrayidx131 to i8**
  %33 = load i8** %string132, align 4
  store i8* %33, i8** @dfg_DESC.3, align 4
  br label %sw.epilog1200

sw.bb133:                                         ; preds = %yyreduce
  %arrayidx134 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %string135 = bitcast %union.yystype* %arrayidx134 to i8**
  %34 = load i8** %string135, align 4
  store i8* %34, i8** @dfg_DESC.6, align 4
  br label %sw.epilog1200

sw.bb136:                                         ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb138:                                         ; preds = %yyreduce
  store i32 1, i32* %2, align 4
  br label %sw.epilog1200

sw.bb140:                                         ; preds = %yyreduce
  store i32 2, i32* %2, align 4
  br label %sw.epilog1200

sw.bb142:                                         ; preds = %yyreduce
  %string144 = bitcast %union.yystype* %yyvsp.2 to i8**
  %35 = load i8** %string144, align 4
  call fastcc void @dfg_SymbolDecl(i32 284, i8* %35, i32 -2)
  br label %sw.epilog1200

sw.bb145:                                         ; preds = %yyreduce
  %arrayidx146 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -3
  %string147 = bitcast %union.yystype* %arrayidx146 to i8**
  %36 = load i8** %string147, align 4
  %number = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1, i32 0
  %37 = load i32* %number, align 4
  call fastcc void @dfg_SymbolDecl(i32 284, i8* %36, i32 %37)
  br label %sw.epilog1200

sw.bb149:                                         ; preds = %yyreduce
  %string151 = bitcast %union.yystype* %yyvsp.2 to i8**
  %38 = load i8** %string151, align 4
  call fastcc void @dfg_SymbolDecl(i32 298, i8* %38, i32 -2)
  br label %sw.epilog1200

sw.bb152:                                         ; preds = %yyreduce
  %arrayidx153 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -3
  %string154 = bitcast %union.yystype* %arrayidx153 to i8**
  %39 = load i8** %string154, align 4
  %number156 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1, i32 0
  %40 = load i32* %number156, align 4
  call fastcc void @dfg_SymbolDecl(i32 298, i8* %39, i32 %40)
  br label %sw.epilog1200

sw.bb157:                                         ; preds = %yyreduce
  %string159 = bitcast %union.yystype* %yyvsp.2 to i8**
  %41 = load i8** %string159, align 4
  call fastcc void @dfg_SymbolDecl(i32 298, i8* %41, i32 1)
  br label %sw.epilog1200

sw.bb160:                                         ; preds = %yyreduce
  %string162 = bitcast %union.yystype* %yyvsp.2 to i8**
  %42 = load i8** %string162, align 4
  call fastcc void @dfg_SymbolDecl(i32 298, i8* %42, i32 1)
  br label %sw.epilog1200

sw.bb163:                                         ; preds = %yyreduce
  %string165 = bitcast %union.yystype* %yyvsp.2 to i8**
  %43 = load i8** %string165, align 4
  call fastcc void @dfg_SymbolDecl(i32 294, i8* %43, i32 -2)
  br label %sw.epilog1200

sw.bb166:                                         ; preds = %yyreduce
  %arrayidx167 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -3
  %string168 = bitcast %union.yystype* %arrayidx167 to i8**
  %44 = load i8** %string168, align 4
  %number170 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1, i32 0
  %45 = load i32* %number170, align 4
  call fastcc void @dfg_SymbolDecl(i32 294, i8* %44, i32 %45)
  br label %sw.epilog1200

sw.bb171:                                         ; preds = %yyreduce
  %string173 = bitcast %union.yystype* %yyvsp.2 to i8**
  %46 = load i8** %string173, align 4
  call fastcc void @dfg_SymbolDecl(i32 300, i8* %46, i32 -2)
  br label %sw.epilog1200

sw.bb174:                                         ; preds = %yyreduce
  %arrayidx175 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -3
  %string176 = bitcast %union.yystype* %arrayidx175 to i8**
  %47 = load i8** %string176, align 4
  %number178 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1, i32 0
  %48 = load i32* %number178, align 4
  call fastcc void @dfg_SymbolDecl(i32 300, i8* %47, i32 %48)
  br label %sw.epilog1200

sw.bb179:                                         ; preds = %yyreduce
  store i32 -1, i32* %2, align 4
  br label %sw.epilog1200

sw.bb181:                                         ; preds = %yyreduce
  %number183 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 0, i32 0
  %49 = load i32* %number183, align 4
  store i32 %49, i32* %2, align 4
  br label %sw.epilog1200

sw.bb185:                                         ; preds = %yyreduce
  %arrayidx186 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -4
  %string187 = bitcast %union.yystype* %arrayidx186 to i8**
  %50 = load i8** %string187, align 4
  %arrayidx188 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %string189 = bitcast %union.yystype* %arrayidx188 to i8**
  %51 = load i8** %string189, align 4
  %call.i = call fastcc i32 @dfg_Symbol(i8* %50, i32 1) #1
  %call1.i = call fastcc i32 @dfg_Symbol(i8* %51, i32 1) #1
  %tobool.i.i = icmp sgt i32 %call.i, -1
  br i1 %tobool.i.i, label %if.then.i, label %land.rhs.i.i

land.rhs.i.i:                                     ; preds = %sw.bb185
  %sub.i.i.i = sub nsw i32 0, %call.i
  %and.i.i.i = and i32 %3, %sub.i.i.i
  %cmp.i.i = icmp eq i32 %and.i.i.i, 2
  br i1 %cmp.i.i, label %if.end.i, label %if.then.i

if.then.i:                                        ; preds = %land.rhs.i.i, %sw.bb185
  %52 = load %struct._IO_FILE** @stdout, align 4
  %call3.i = call i32 @fflush(%struct._IO_FILE* %52) #1
  %53 = load i32* @dfg_LINENUMBER, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([44 x i8]* @.str42, i32 0, i32 0), i32 %53) #1
  call fastcc void @misc_Error() #1
  unreachable

if.end.i:                                         ; preds = %land.rhs.i.i
  %tobool.i34.i = icmp sgt i32 %call1.i, -1
  br i1 %tobool.i34.i, label %if.then6.i, label %land.rhs.i38.i

land.rhs.i38.i:                                   ; preds = %if.end.i
  %sub.i.i35.i = sub nsw i32 0, %call1.i
  %and.i.i36.i = and i32 %3, %sub.i.i35.i
  %cmp.i37.i = icmp eq i32 %and.i.i36.i, 2
  br i1 %cmp.i37.i, label %if.end8.i, label %if.then6.i

if.then6.i:                                       ; preds = %land.rhs.i38.i, %if.end.i
  %54 = load %struct._IO_FILE** @stdout, align 4
  %call7.i = call i32 @fflush(%struct._IO_FILE* %54) #1
  %55 = load i32* @dfg_LINENUMBER, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([44 x i8]* @.str42, i32 0, i32 0), i32 %55) #1
  call fastcc void @misc_Error() #1
  unreachable

if.end8.i:                                        ; preds = %land.rhs.i38.i
  %56 = load i32* @symbol_STANDARDVARCOUNTER, align 4
  %inc.i.i = add nsw i32 %56, 1
  store i32 %inc.i.i, i32* @symbol_STANDARDVARCOUNTER, align 4
  %call11.i = call %struct.term* @term_Create(i32 %inc.i.i, %struct.LIST_HELP* null) #1
  store i32 0, i32* @symbol_STANDARDVARCOUNTER, align 4
  %57 = bitcast %struct.term* %call11.i to i8*
  %call.i.i50.i = call i8* @memory_Malloc(i32 8) #1
  %58 = bitcast i8* %call.i.i50.i to %struct.LIST_HELP*
  %car.i.i51.i = getelementptr inbounds i8* %call.i.i50.i, i32 4
  %59 = bitcast i8* %car.i.i51.i to i8**
  store i8* %57, i8** %59, align 4
  %cdr.i.i52.i = bitcast i8* %call.i.i50.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i52.i, align 4
  %call13.i = call %struct.term* @term_Create(i32 %call.i, %struct.LIST_HELP* %58) #1
  %call14.i = call %struct.term* @term_Copy(%struct.term* %call11.i) #1
  %60 = bitcast %struct.term* %call14.i to i8*
  %call.i.i53.i = call i8* @memory_Malloc(i32 8) #1
  %61 = bitcast i8* %call.i.i53.i to %struct.LIST_HELP*
  %car.i.i54.i = getelementptr inbounds i8* %call.i.i53.i, i32 4
  %62 = bitcast i8* %car.i.i54.i to i8**
  store i8* %60, i8** %62, align 4
  %cdr.i.i55.i = bitcast i8* %call.i.i53.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i55.i, align 4
  %call16.i = call %struct.term* @term_Create(i32 %call1.i, %struct.LIST_HELP* %61) #1
  %63 = load i32* @fol_IMPLIES, align 4
  %64 = bitcast %struct.term* %call13.i to i8*
  %65 = bitcast %struct.term* %call16.i to i8*
  %call.i.i56.i = call i8* @memory_Malloc(i32 8) #1
  %66 = bitcast i8* %call.i.i56.i to %struct.LIST_HELP*
  %car.i.i57.i = getelementptr inbounds i8* %call.i.i56.i, i32 4
  %67 = bitcast i8* %car.i.i57.i to i8**
  store i8* %65, i8** %67, align 4
  %cdr.i.i58.i = bitcast i8* %call.i.i56.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i58.i, align 4
  %call.i.i = call i8* @memory_Malloc(i32 8) #1
  %68 = bitcast i8* %call.i.i to %struct.LIST_HELP*
  %car.i.i = getelementptr inbounds i8* %call.i.i, i32 4
  %69 = bitcast i8* %car.i.i to i8**
  store i8* %64, i8** %69, align 4
  %cdr.i.i = bitcast i8* %call.i.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* %66, %struct.LIST_HELP** %cdr.i.i, align 4
  %call20.i = call %struct.term* @term_Create(i32 %63, %struct.LIST_HELP* %68) #1
  %70 = load i32* @fol_ALL, align 4
  %call22.i = call %struct.term* @term_Copy(%struct.term* %call11.i) #1
  %71 = bitcast %struct.term* %call22.i to i8*
  %call.i.i47.i = call i8* @memory_Malloc(i32 8) #1
  %72 = bitcast i8* %call.i.i47.i to %struct.LIST_HELP*
  %car.i.i48.i = getelementptr inbounds i8* %call.i.i47.i, i32 4
  %73 = bitcast i8* %car.i.i48.i to i8**
  store i8* %71, i8** %73, align 4
  %cdr.i.i49.i = bitcast i8* %call.i.i47.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i49.i, align 4
  %74 = bitcast %struct.term* %call20.i to i8*
  %call.i.i44.i = call i8* @memory_Malloc(i32 8) #1
  %75 = bitcast i8* %call.i.i44.i to %struct.LIST_HELP*
  %car.i.i45.i = getelementptr inbounds i8* %call.i.i44.i, i32 4
  %76 = bitcast i8* %car.i.i45.i to i8**
  store i8* %74, i8** %76, align 4
  %cdr.i.i46.i = bitcast i8* %call.i.i44.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i46.i, align 4
  %call25.i = call %struct.term* @fol_CreateQuantifier(i32 %70, %struct.LIST_HELP* %72, %struct.LIST_HELP* %75) #1
  %77 = load %struct.LIST_HELP** @dfg_SORTDECLLIST, align 4
  %78 = bitcast %struct.term* %call25.i to %struct.LIST_HELP*
  %call.i.i41.i = call i8* @memory_Malloc(i32 8) #1
  %car.i.i42.i = getelementptr inbounds i8* %call.i.i41.i, i32 4
  %79 = bitcast i8* %car.i.i42.i to i8**
  store i8* null, i8** %79, align 4
  %cdr.i.i43.i = bitcast i8* %call.i.i41.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* %78, %struct.LIST_HELP** %cdr.i.i43.i, align 4
  %call.i.i.i = call i8* @memory_Malloc(i32 8) #1
  %80 = bitcast i8* %call.i.i.i to %struct.LIST_HELP*
  %car.i.i.i = getelementptr inbounds i8* %call.i.i.i, i32 4
  %81 = bitcast i8* %car.i.i.i to i8**
  store i8* %call.i.i41.i, i8** %81, align 4
  %cdr.i.i.i = bitcast i8* %call.i.i.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i.i, align 4
  %cmp.i.i.i = icmp eq %struct.LIST_HELP* %77, null
  br i1 %cmp.i.i.i, label %dfg_SubSort.exit, label %if.end.i.i

if.end.i.i:                                       ; preds = %if.end8.i
  %cmp.i18.i.i = icmp eq i8* %call.i.i.i, null
  br i1 %cmp.i18.i.i, label %dfg_SubSort.exit, label %for.cond.i.i

for.cond.i.i:                                     ; preds = %if.end.i.i, %for.cond.i.i
  %List1.addr.0.i.i = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i.i, %for.cond.i.i ], [ %77, %if.end.i.i ]
  %List1.addr.0.idx15.i.i = getelementptr %struct.LIST_HELP* %List1.addr.0.i.i, i32 0, i32 0
  %List1.addr.0.idx15.val.i.i = load %struct.LIST_HELP** %List1.addr.0.idx15.i.i, align 4
  %cmp.i16.i.i = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i.i, null
  br i1 %cmp.i16.i.i, label %for.end.i.i, label %for.cond.i.i

for.end.i.i:                                      ; preds = %for.cond.i.i
  store %struct.LIST_HELP* %80, %struct.LIST_HELP** %List1.addr.0.idx15.i.i, align 4
  br label %dfg_SubSort.exit

dfg_SubSort.exit:                                 ; preds = %if.end8.i, %if.end.i.i, %for.end.i.i
  %retval.0.i.i = phi %struct.LIST_HELP* [ %77, %for.end.i.i ], [ %80, %if.end8.i ], [ %77, %if.end.i.i ]
  store %struct.LIST_HELP* %retval.0.i.i, %struct.LIST_HELP** @dfg_SORTDECLLIST, align 4
  br label %sw.epilog1200

sw.bb190:                                         ; preds = %yyreduce
  %82 = load %struct.LIST_HELP** @dfg_SORTDECLLIST, align 4
  %arrayidx191 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %term = bitcast %union.yystype* %arrayidx191 to %struct.term**
  %83 = load %struct.term** %term, align 4
  %84 = bitcast %struct.term* %83 to %struct.LIST_HELP*
  %call.i.i1761 = call i8* @memory_Malloc(i32 8) #1
  %car.i.i1762 = getelementptr inbounds i8* %call.i.i1761, i32 4
  %85 = bitcast i8* %car.i.i1762 to i8**
  store i8* null, i8** %85, align 4
  %cdr.i.i1763 = bitcast i8* %call.i.i1761 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %84, %struct.LIST_HELP** %cdr.i.i1763, align 4
  %call.i.i1764 = call i8* @memory_Malloc(i32 8) #1
  %86 = bitcast i8* %call.i.i1764 to %struct.LIST_HELP*
  %car.i.i1765 = getelementptr inbounds i8* %call.i.i1764, i32 4
  %87 = bitcast i8* %car.i.i1765 to i8**
  store i8* %call.i.i1761, i8** %87, align 4
  %cdr.i.i1766 = bitcast i8* %call.i.i1764 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1766, align 4
  %cmp.i.i1767 = icmp eq %struct.LIST_HELP* %82, null
  br i1 %cmp.i.i1767, label %list_Nconc.exit, label %if.end.i1768

if.end.i1768:                                     ; preds = %sw.bb190
  %cmp.i18.i = icmp eq i8* %call.i.i1764, null
  br i1 %cmp.i18.i, label %list_Nconc.exit, label %for.cond.i

for.cond.i:                                       ; preds = %if.end.i1768, %for.cond.i
  %List1.addr.0.i = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i, %for.cond.i ], [ %82, %if.end.i1768 ]
  %List1.addr.0.idx15.i = getelementptr %struct.LIST_HELP* %List1.addr.0.i, i32 0, i32 0
  %List1.addr.0.idx15.val.i = load %struct.LIST_HELP** %List1.addr.0.idx15.i, align 4
  %cmp.i16.i = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i, null
  br i1 %cmp.i16.i, label %for.end.i, label %for.cond.i

for.end.i:                                        ; preds = %for.cond.i
  store %struct.LIST_HELP* %86, %struct.LIST_HELP** %List1.addr.0.idx15.i, align 4
  br label %list_Nconc.exit

list_Nconc.exit:                                  ; preds = %sw.bb190, %if.end.i1768, %for.end.i
  %retval.0.i = phi %struct.LIST_HELP* [ %82, %for.end.i ], [ %86, %sw.bb190 ], [ %82, %if.end.i1768 ]
  store %struct.LIST_HELP* %retval.0.i, %struct.LIST_HELP** @dfg_SORTDECLLIST, align 4
  br label %sw.epilog1200

sw.bb195:                                         ; preds = %yyreduce
  %arrayidx196 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -4
  %string197 = bitcast %union.yystype* %arrayidx196 to i8**
  %88 = load i8** %string197, align 4
  call void @string_StringFree(i8* %88) #1
  br label %sw.epilog1200

sw.bb198:                                         ; preds = %yyreduce
  %89 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %call.i.i.i1769 = call i8* @memory_Malloc(i32 8) #1
  %90 = bitcast i8* %call.i.i.i1769 to %struct.LIST_HELP*
  %car.i.i.i1770 = getelementptr inbounds i8* %call.i.i.i1769, i32 4
  %91 = bitcast i8* %car.i.i.i1770 to i8**
  store i8* null, i8** %91, align 4
  %cdr.i.i.i1771 = bitcast i8* %call.i.i.i1769 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %89, %struct.LIST_HELP** %cdr.i.i.i1771, align 4
  store %struct.LIST_HELP* %90, %struct.LIST_HELP** @dfg_VARLIST, align 4
  store i1 true, i1* @dfg_VARDECL, align 1
  br label %sw.epilog1200

sw.bb199:                                         ; preds = %yyreduce
  store i1 false, i1* @dfg_VARDECL, align 1
  br label %sw.epilog1200

sw.bb200:                                         ; preds = %yyreduce
  %92 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %.idx.i = getelementptr %struct.LIST_HELP* %92, i32 0, i32 1
  %.idx.val.i = load i8** %.idx.i, align 4
  %93 = bitcast i8* %.idx.val.i to %struct.LIST_HELP*
  call void @list_DeleteWithElement(%struct.LIST_HELP* %93, void (i8*)* bitcast (void (%struct.DFG_VARENTRY*)* @dfg_VarFree to void (i8*)*)) #1
  %94 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %L.idx.i.i = getelementptr %struct.LIST_HELP* %94, i32 0, i32 0
  %L.idx.val.i.i = load %struct.LIST_HELP** %L.idx.i.i, align 4
  %95 = bitcast %struct.LIST_HELP* %94 to i8*
  %96 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %96, i32 0, i32 4
  %97 = load i32* %total_size.i.i.i.i, align 4
  %98 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i = add i32 %98, %97
  store i32 %add24.i.i.i.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %96, i32 0, i32 0
  %99 = load i8** %free.i.i.i.i, align 4
  %.c.i.i.i = bitcast i8* %99 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i, %struct.LIST_HELP** %L.idx.i.i, align 4
  %100 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %100, i32 0, i32 0
  store i8* %95, i8** %free27.i.i.i.i, align 4
  store %struct.LIST_HELP* %L.idx.val.i.i, %struct.LIST_HELP** @dfg_VARLIST, align 4
  %cmp.i.i1772 = icmp eq %struct.LIST_HELP* %L.idx.val.i.i, null
  br i1 %cmp.i.i1772, label %dfg_VarCheck.exit, label %if.then.i1774

if.then.i1774:                                    ; preds = %sw.bb200
  %101 = load %struct._IO_FILE** @stdout, align 4
  %call1.i1773 = call i32 @fflush(%struct._IO_FILE* %101) #1
  %102 = load %struct._IO_FILE** @stderr, align 4
  %call2.i = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %102, i8* getelementptr inbounds ([31 x i8]* @.str27, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str28, i32 0, i32 0), i32 1881) #1
  call void (i8*, ...)* @misc_ErrorReport(i8* getelementptr inbounds ([55 x i8]* @.str41, i32 0, i32 0)) #1
  %103 = load %struct._IO_FILE** @stderr, align 4
  %104 = call i32 @fwrite(i8* getelementptr inbounds ([133 x i8]* @.str30, i32 0, i32 0), i32 132, i32 1, %struct._IO_FILE* %103) #1
  call fastcc void @misc_DumpCore() #1
  unreachable

dfg_VarCheck.exit:                                ; preds = %sw.bb200
  store i32 0, i32* @symbol_STANDARDVARCOUNTER, align 4
  %105 = load i32* @fol_ALL, align 4
  %arrayidx203 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -6
  %list = bitcast %union.yystype* %arrayidx203 to %struct.LIST_HELP**
  %106 = load %struct.LIST_HELP** %list, align 4
  %arrayidx204 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %term205 = bitcast %union.yystype* %arrayidx204 to %struct.term**
  %107 = load %struct.term** %term205, align 4
  %call206 = call %struct.term* @dfg_CreateQuantifier(i32 %105, %struct.LIST_HELP* %106, %struct.term* %107)
  %108 = load %struct.LIST_HELP** @dfg_SORTDECLLIST, align 4
  %109 = bitcast %struct.term* %call206 to %struct.LIST_HELP*
  %call.i.i1776 = call i8* @memory_Malloc(i32 8) #1
  %car.i.i1777 = getelementptr inbounds i8* %call.i.i1776, i32 4
  %110 = bitcast i8* %car.i.i1777 to i8**
  store i8* null, i8** %110, align 4
  %cdr.i.i1778 = bitcast i8* %call.i.i1776 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %109, %struct.LIST_HELP** %cdr.i.i1778, align 4
  %call.i.i1779 = call i8* @memory_Malloc(i32 8) #1
  %111 = bitcast i8* %call.i.i1779 to %struct.LIST_HELP*
  %car.i.i1780 = getelementptr inbounds i8* %call.i.i1779, i32 4
  %112 = bitcast i8* %car.i.i1780 to i8**
  store i8* %call.i.i1776, i8** %112, align 4
  %cdr.i.i1781 = bitcast i8* %call.i.i1779 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1781, align 4
  %cmp.i.i1782 = icmp eq %struct.LIST_HELP* %108, null
  br i1 %cmp.i.i1782, label %list_Nconc.exit1792, label %if.end.i1784

if.end.i1784:                                     ; preds = %dfg_VarCheck.exit
  %cmp.i18.i1783 = icmp eq i8* %call.i.i1779, null
  br i1 %cmp.i18.i1783, label %list_Nconc.exit1792, label %for.cond.i1789

for.cond.i1789:                                   ; preds = %if.end.i1784, %for.cond.i1789
  %List1.addr.0.i1785 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i1787, %for.cond.i1789 ], [ %108, %if.end.i1784 ]
  %List1.addr.0.idx15.i1786 = getelementptr %struct.LIST_HELP* %List1.addr.0.i1785, i32 0, i32 0
  %List1.addr.0.idx15.val.i1787 = load %struct.LIST_HELP** %List1.addr.0.idx15.i1786, align 4
  %cmp.i16.i1788 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i1787, null
  br i1 %cmp.i16.i1788, label %for.end.i1790, label %for.cond.i1789

for.end.i1790:                                    ; preds = %for.cond.i1789
  store %struct.LIST_HELP* %111, %struct.LIST_HELP** %List1.addr.0.idx15.i1786, align 4
  br label %list_Nconc.exit1792

list_Nconc.exit1792:                              ; preds = %dfg_VarCheck.exit, %if.end.i1784, %for.end.i1790
  %retval.0.i1791 = phi %struct.LIST_HELP* [ %108, %for.end.i1790 ], [ %111, %dfg_VarCheck.exit ], [ %108, %if.end.i1784 ]
  store %struct.LIST_HELP* %retval.0.i1791, %struct.LIST_HELP** @dfg_SORTDECLLIST, align 4
  br label %sw.epilog1200

sw.bb210:                                         ; preds = %yyreduce
  %arrayidx211 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -7
  %string212 = bitcast %union.yystype* %arrayidx211 to i8**
  %113 = load i8** %string212, align 4
  %call213 = call fastcc i32 @dfg_Symbol(i8* %113, i32 1)
  %bool = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -6, i32 0
  %114 = load i32* %bool, align 4
  %arrayidx215 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %list216 = bitcast %union.yystype* %arrayidx215 to %struct.LIST_HELP**
  %115 = load %struct.LIST_HELP** %list216, align 4
  %tobool.i.i1793 = icmp sgt i32 %call213, -1
  br i1 %tobool.i.i1793, label %if.then.i1799, label %land.rhs.i.i1797

land.rhs.i.i1797:                                 ; preds = %sw.bb210
  %sub.i.i.i1794 = sub nsw i32 0, %call213
  %and.i.i.i1795 = and i32 %3, %sub.i.i.i1794
  %cmp.i.i1796 = icmp eq i32 %and.i.i.i1795, 2
  br i1 %cmp.i.i1796, label %if.end.i1800, label %if.then.i1799

if.then.i1799:                                    ; preds = %land.rhs.i.i1797, %sw.bb210
  %116 = load %struct._IO_FILE** @stdout, align 4
  %call1.i1798 = call i32 @fflush(%struct._IO_FILE* %116) #1
  %117 = load i32* @dfg_LINENUMBER, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([44 x i8]* @.str42, i32 0, i32 0), i32 %117) #1
  call fastcc void @misc_Error() #1
  unreachable

if.end.i1800:                                     ; preds = %land.rhs.i.i1797
  %shr.i.i54.i = ashr i32 %sub.i.i.i1794, %4
  %118 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i55.i = getelementptr inbounds %struct.signature** %118, i32 %shr.i.i54.i
  %119 = load %struct.signature** %arrayidx.i.i55.i, align 4
  %props.i56.i = getelementptr inbounds %struct.signature* %119, i32 0, i32 4
  %120 = load i32* %props.i56.i, align 4
  %and.i.i = and i32 %120, 512
  %tobool.i57.i = icmp eq i32 %and.i.i, 0
  br i1 %tobool.i57.i, label %symbol_RemoveProperty.exit.i, label %if.then.i.i

if.then.i.i:                                      ; preds = %if.end.i1800
  %sub.i.i = add i32 %120, -512
  store i32 %sub.i.i, i32* %props.i56.i, align 4
  %.pre.i = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i63.phi.trans.insert.i = getelementptr inbounds %struct.signature** %.pre.i, i32 %shr.i.i54.i
  %.pre93.i = load %struct.signature** %arrayidx.i.i63.phi.trans.insert.i, align 4
  %props.i64.phi.trans.insert.i = getelementptr inbounds %struct.signature* %.pre93.i, i32 0, i32 4
  %.pre94.i = load i32* %props.i64.phi.trans.insert.i, align 4
  br label %symbol_RemoveProperty.exit.i

symbol_RemoveProperty.exit.i:                     ; preds = %if.then.i.i, %if.end.i1800
  %121 = phi i32 [ %120, %if.end.i1800 ], [ %.pre94.i, %if.then.i.i ]
  %122 = phi %struct.signature* [ %119, %if.end.i1800 ], [ %.pre93.i, %if.then.i.i ]
  %and.i65.i = and i32 %121, 256
  %tobool.i66.i = icmp eq i32 %and.i65.i, 0
  br i1 %tobool.i66.i, label %symbol_RemoveProperty.exit69.i, label %if.then.i68.i

if.then.i68.i:                                    ; preds = %symbol_RemoveProperty.exit.i
  %props.i64.i = getelementptr inbounds %struct.signature* %122, i32 0, i32 4
  %sub.i67.i = add i32 %121, -256
  store i32 %sub.i67.i, i32* %props.i64.i, align 4
  %.pre95.i = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i83.phi.trans.insert.i = getelementptr inbounds %struct.signature** %.pre95.i, i32 %shr.i.i54.i
  %.pre96.i = load %struct.signature** %arrayidx.i.i83.phi.trans.insert.i, align 4
  br label %symbol_RemoveProperty.exit69.i

symbol_RemoveProperty.exit69.i:                   ; preds = %if.then.i68.i, %symbol_RemoveProperty.exit.i
  %123 = phi %struct.signature* [ %122, %symbol_RemoveProperty.exit.i ], [ %.pre96.i, %if.then.i68.i ]
  %generatedBy.i84.i = getelementptr inbounds %struct.signature* %123, i32 0, i32 6
  %124 = load %struct.LIST_HELP** %generatedBy.i84.i, align 4
  %cmp.i5.i.i = icmp eq %struct.LIST_HELP* %124, null
  br i1 %cmp.i5.i.i, label %list_Delete.exit.i, label %while.body.i.i

while.body.i.i:                                   ; preds = %symbol_RemoveProperty.exit69.i, %while.body.i.i
  %L.addr.06.i.i = phi %struct.LIST_HELP* [ %L.addr.0.idx.val.i.i, %while.body.i.i ], [ %124, %symbol_RemoveProperty.exit69.i ]
  %L.addr.0.idx.i.i = getelementptr %struct.LIST_HELP* %L.addr.06.i.i, i32 0, i32 0
  %L.addr.0.idx.val.i.i = load %struct.LIST_HELP** %L.addr.0.idx.i.i, align 4
  %125 = bitcast %struct.LIST_HELP* %L.addr.06.i.i to i8*
  %126 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i1801 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %126, i32 0, i32 4
  %127 = load i32* %total_size.i.i.i.i1801, align 4
  %128 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i1802 = add i32 %128, %127
  store i32 %add24.i.i.i.i1802, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i1803 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %126, i32 0, i32 0
  %129 = load i8** %free.i.i.i.i1803, align 4
  %.c.i.i.i1804 = bitcast i8* %129 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i1804, %struct.LIST_HELP** %L.addr.0.idx.i.i, align 4
  %130 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i1805 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %130, i32 0, i32 0
  store i8* %125, i8** %free27.i.i.i.i1805, align 4
  %cmp.i.i.i1806 = icmp eq %struct.LIST_HELP* %L.addr.0.idx.val.i.i, null
  br i1 %cmp.i.i.i1806, label %list_Delete.exit.loopexit.i, label %while.body.i.i

list_Delete.exit.loopexit.i:                      ; preds = %while.body.i.i
  %.pre97.i = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i78.phi.trans.insert.i = getelementptr inbounds %struct.signature** %.pre97.i, i32 %shr.i.i54.i
  %.pre98.i = load %struct.signature** %arrayidx.i.i78.phi.trans.insert.i, align 4
  br label %list_Delete.exit.i

list_Delete.exit.i:                               ; preds = %list_Delete.exit.loopexit.i, %symbol_RemoveProperty.exit69.i
  %131 = phi %struct.signature* [ %.pre98.i, %list_Delete.exit.loopexit.i ], [ %123, %symbol_RemoveProperty.exit69.i ]
  %props.i79.i = getelementptr inbounds %struct.signature* %131, i32 0, i32 4
  %132 = load i32* %props.i79.i, align 4
  %or.i80.i = or i32 %132, 512
  store i32 %or.i80.i, i32* %props.i79.i, align 4
  %tobool3.i = icmp ne i32 %114, 0
  br i1 %tobool3.i, label %if.then4.i, label %for.cond.preheader.i

if.then4.i:                                       ; preds = %list_Delete.exit.i
  %133 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i73.i = getelementptr inbounds %struct.signature** %133, i32 %shr.i.i54.i
  %134 = load %struct.signature** %arrayidx.i.i73.i, align 4
  %props.i74.i = getelementptr inbounds %struct.signature* %134, i32 0, i32 4
  %135 = load i32* %props.i74.i, align 4
  %or.i75.i = or i32 %135, 256
  store i32 %or.i75.i, i32* %props.i74.i, align 4
  br label %for.cond.preheader.i

for.cond.preheader.i:                             ; preds = %if.then4.i, %list_Delete.exit.i
  %cmp.i7086.i = icmp eq %struct.LIST_HELP* %115, null
  br i1 %cmp.i7086.i, label %dfg_SymbolGenerated.exit, label %for.body.lr.ph.i

for.body.lr.ph.i:                                 ; preds = %for.cond.preheader.i
  br i1 %tobool3.i, label %for.body.us.i, label %for.body.i

for.body.us.i:                                    ; preds = %for.body.lr.ph.i, %for.inc.us.i
  %scan.087.us.i = phi %struct.LIST_HELP* [ %scan.0.idx43.val.us.i, %for.inc.us.i ], [ %115, %for.body.lr.ph.i ]
  %scan.0.idx42.us.i = getelementptr %struct.LIST_HELP* %scan.087.us.i, i32 0, i32 1
  %scan.0.idx42.val.us.i = load i8** %scan.0.idx42.us.i, align 4
  %call9.us.i = call i32 @symbol_Lookup(i8* %scan.0.idx42.val.us.i) #1
  %cmp.us.i = icmp eq i32 %call9.us.i, 0
  br i1 %cmp.us.i, label %if.then10.i, label %if.else.us.i

if.else.us.i:                                     ; preds = %for.body.us.i
  %tobool.i58.us.i = icmp sgt i32 %call9.us.i, -1
  br i1 %tobool.i58.us.i, label %if.then15.i, label %land.rhs.i59.us.i

land.rhs.i59.us.i:                                ; preds = %if.else.us.i
  %sub.i6.i.us.i = sub nsw i32 0, %call9.us.i
  %and.i7.i.us.i = and i32 %3, %sub.i6.i.us.i
  %136 = icmp ult i32 %and.i7.i.us.i, 2
  br i1 %136, label %for.inc.us.i, label %if.then15.i

for.inc.us.i:                                     ; preds = %land.rhs.i59.us.i
  %scan.0.idx.val.us.i = load i8** %scan.0.idx42.us.i, align 4
  call void @string_StringFree(i8* %scan.0.idx.val.us.i) #1
  %137 = inttoptr i32 %call9.us.i to i8*
  store i8* %137, i8** %scan.0.idx42.us.i, align 4
  %shr.i.i49.us.i = ashr i32 %sub.i6.i.us.i, %4
  %138 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i50.us.i = getelementptr inbounds %struct.signature** %138, i32 %shr.i.i49.us.i
  %139 = load %struct.signature** %arrayidx.i.i50.us.i, align 4
  %props.i51.us.i = getelementptr inbounds %struct.signature* %139, i32 0, i32 4
  %140 = load i32* %props.i51.us.i, align 4
  %or.i52.us.i = or i32 %140, 512
  store i32 %or.i52.us.i, i32* %props.i51.us.i, align 4
  %141 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i47.us.i = getelementptr inbounds %struct.signature** %141, i32 %shr.i.i49.us.i
  %142 = load %struct.signature** %arrayidx.i.i47.us.i, align 4
  %props.i.us.i = getelementptr inbounds %struct.signature* %142, i32 0, i32 4
  %143 = load i32* %props.i.us.i, align 4
  %or.i.us.i = or i32 %143, 256
  store i32 %or.i.us.i, i32* %props.i.us.i, align 4
  %scan.0.idx43.us.i = getelementptr %struct.LIST_HELP* %scan.087.us.i, i32 0, i32 0
  %scan.0.idx43.val.us.i = load %struct.LIST_HELP** %scan.0.idx43.us.i, align 4
  %cmp.i70.us.i = icmp eq %struct.LIST_HELP* %scan.0.idx43.val.us.i, null
  br i1 %cmp.i70.us.i, label %dfg_SymbolGenerated.exit, label %for.body.us.i

for.body.i:                                       ; preds = %for.body.lr.ph.i, %for.inc.i
  %scan.087.i = phi %struct.LIST_HELP* [ %scan.0.idx43.val.i, %for.inc.i ], [ %115, %for.body.lr.ph.i ]
  %scan.0.idx42.i = getelementptr %struct.LIST_HELP* %scan.087.i, i32 0, i32 1
  %scan.0.idx42.val.i = load i8** %scan.0.idx42.i, align 4
  %call9.i = call i32 @symbol_Lookup(i8* %scan.0.idx42.val.i) #1
  %cmp.i = icmp eq i32 %call9.i, 0
  br i1 %cmp.i, label %if.then10.i, label %if.else.i

if.then10.i:                                      ; preds = %for.body.us.i, %for.body.i
  %scan.0.idx42.lcssa.i = phi i8** [ %scan.0.idx42.i, %for.body.i ], [ %scan.0.idx42.us.i, %for.body.us.i ]
  %144 = load %struct._IO_FILE** @stdout, align 4
  %call11.i1807 = call i32 @fflush(%struct._IO_FILE* %144) #1
  %145 = load i32* @dfg_LINENUMBER, align 4
  %scan.0.idx41.val.i = load i8** %scan.0.idx42.lcssa.i, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([33 x i8]* @.str43, i32 0, i32 0), i32 %145, i8* %scan.0.idx41.val.i) #1
  call fastcc void @misc_Error() #1
  unreachable

if.else.i:                                        ; preds = %for.body.i
  %tobool.i58.i = icmp sgt i32 %call9.i, -1
  br i1 %tobool.i58.i, label %if.then15.i, label %land.rhs.i59.i

land.rhs.i59.i:                                   ; preds = %if.else.i
  %sub.i6.i.i = sub nsw i32 0, %call9.i
  %and.i7.i.i = and i32 %3, %sub.i6.i.i
  %146 = icmp ult i32 %and.i7.i.i, 2
  br i1 %146, label %for.inc.i, label %if.then15.i

if.then15.i:                                      ; preds = %land.rhs.i59.us.i, %if.else.us.i, %land.rhs.i59.i, %if.else.i
  %147 = load %struct._IO_FILE** @stdout, align 4
  %call16.i1808 = call i32 @fflush(%struct._IO_FILE* %147) #1
  %148 = load i32* @dfg_LINENUMBER, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([38 x i8]* @.str44, i32 0, i32 0), i32 %148) #1
  call fastcc void @misc_Error() #1
  unreachable

for.inc.i:                                        ; preds = %land.rhs.i59.i
  %scan.0.idx.val.i = load i8** %scan.0.idx42.i, align 4
  call void @string_StringFree(i8* %scan.0.idx.val.i) #1
  %149 = inttoptr i32 %call9.i to i8*
  store i8* %149, i8** %scan.0.idx42.i, align 4
  %shr.i.i49.i = ashr i32 %sub.i6.i.i, %4
  %150 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i50.i = getelementptr inbounds %struct.signature** %150, i32 %shr.i.i49.i
  %151 = load %struct.signature** %arrayidx.i.i50.i, align 4
  %props.i51.i = getelementptr inbounds %struct.signature* %151, i32 0, i32 4
  %152 = load i32* %props.i51.i, align 4
  %or.i52.i = or i32 %152, 512
  store i32 %or.i52.i, i32* %props.i51.i, align 4
  %scan.0.idx43.i = getelementptr %struct.LIST_HELP* %scan.087.i, i32 0, i32 0
  %scan.0.idx43.val.i = load %struct.LIST_HELP** %scan.0.idx43.i, align 4
  %cmp.i70.i = icmp eq %struct.LIST_HELP* %scan.0.idx43.val.i, null
  br i1 %cmp.i70.i, label %dfg_SymbolGenerated.exit, label %for.body.i

dfg_SymbolGenerated.exit:                         ; preds = %for.inc.us.i, %for.inc.i, %for.cond.preheader.i
  %153 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i.i = getelementptr inbounds %struct.signature** %153, i32 %shr.i.i54.i
  %154 = load %struct.signature** %arrayidx.i.i.i, align 4
  %generatedBy.i.i = getelementptr inbounds %struct.signature* %154, i32 0, i32 6
  store %struct.LIST_HELP* %115, %struct.LIST_HELP** %generatedBy.i.i, align 4
  br label %sw.epilog1200

sw.bb217:                                         ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb219:                                         ; preds = %yyreduce
  store i32 1, i32* %2, align 4
  br label %sw.epilog1200

sw.bb221:                                         ; preds = %yyreduce
  %string223 = bitcast %union.yystype* %yyvsp.2 to i8**
  %155 = load i8** %string223, align 4
  %call.i.i1810 = call i8* @memory_Malloc(i32 8) #1
  %car.i.i1811 = getelementptr inbounds i8* %call.i.i1810, i32 4
  %156 = bitcast i8* %car.i.i1811 to i8**
  store i8* %155, i8** %156, align 4
  %cdr.i.i1812 = bitcast i8* %call.i.i1810 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1812, align 4
  %call224.c = ptrtoint i8* %call.i.i1810 to i32
  store i32 %call224.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb226:                                         ; preds = %yyreduce
  %string228 = bitcast %union.yystype* %yyvsp.2 to i8**
  %157 = load i8** %string228, align 4
  %arrayidx229 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %list230 = bitcast %union.yystype* %arrayidx229 to %struct.LIST_HELP**
  %158 = load %struct.LIST_HELP** %list230, align 4
  %call.i1813 = call i8* @memory_Malloc(i32 8) #1
  %car.i = getelementptr inbounds i8* %call.i1813, i32 4
  %159 = bitcast i8* %car.i to i8**
  store i8* %157, i8** %159, align 4
  %cdr.i = bitcast i8* %call.i1813 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %158, %struct.LIST_HELP** %cdr.i, align 4
  %call231.c = ptrtoint i8* %call.i1813 to i32
  store i32 %call231.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb233:                                         ; preds = %yyreduce
  %string235 = bitcast %union.yystype* %yyvsp.2 to i8**
  %160 = load i8** %string235, align 4
  call void @string_StringFree(i8* %160) #1
  br label %sw.epilog1200

sw.bb236:                                         ; preds = %yyreduce
  %string238 = bitcast %union.yystype* %yyvsp.2 to i8**
  %161 = load i8** %string238, align 4
  call void @string_StringFree(i8* %161) #1
  br label %sw.epilog1200

sw.bb239:                                         ; preds = %yyreduce
  %arrayidx240 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %list241 = bitcast %union.yystype* %arrayidx240 to %struct.LIST_HELP**
  %162 = load %struct.LIST_HELP** %list241, align 4
  %call242 = call %struct.LIST_HELP* @list_NReverse(%struct.LIST_HELP* %162) #1
  %bool244 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -5, i32 0
  %163 = load i32* %bool244, align 4
  %tobool245 = icmp eq i32 %163, 0
  br i1 %tobool245, label %if.else250, label %if.then246

if.then246:                                       ; preds = %sw.bb239
  %164 = load %struct.LIST_HELP** @dfg_AXIOMLIST, align 4
  %165 = load %struct.LIST_HELP** %list241, align 4
  %cmp.i.i1814 = icmp eq %struct.LIST_HELP* %164, null
  br i1 %cmp.i.i1814, label %list_Nconc.exit1824, label %if.end.i1816

if.end.i1816:                                     ; preds = %if.then246
  %cmp.i18.i1815 = icmp eq %struct.LIST_HELP* %165, null
  br i1 %cmp.i18.i1815, label %list_Nconc.exit1824, label %for.cond.i1821

for.cond.i1821:                                   ; preds = %if.end.i1816, %for.cond.i1821
  %List1.addr.0.i1817 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i1819, %for.cond.i1821 ], [ %164, %if.end.i1816 ]
  %List1.addr.0.idx15.i1818 = getelementptr %struct.LIST_HELP* %List1.addr.0.i1817, i32 0, i32 0
  %List1.addr.0.idx15.val.i1819 = load %struct.LIST_HELP** %List1.addr.0.idx15.i1818, align 4
  %cmp.i16.i1820 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i1819, null
  br i1 %cmp.i16.i1820, label %for.end.i1822, label %for.cond.i1821

for.end.i1822:                                    ; preds = %for.cond.i1821
  store %struct.LIST_HELP* %165, %struct.LIST_HELP** %List1.addr.0.idx15.i1818, align 4
  br label %list_Nconc.exit1824

list_Nconc.exit1824:                              ; preds = %if.then246, %if.end.i1816, %for.end.i1822
  %retval.0.i1823 = phi %struct.LIST_HELP* [ %164, %for.end.i1822 ], [ %165, %if.then246 ], [ %164, %if.end.i1816 ]
  store %struct.LIST_HELP* %retval.0.i1823, %struct.LIST_HELP** @dfg_AXIOMLIST, align 4
  br label %sw.epilog1200

if.else250:                                       ; preds = %sw.bb239
  %166 = load %struct.LIST_HELP** @dfg_CONJECLIST, align 4
  %167 = load %struct.LIST_HELP** %list241, align 4
  %cmp.i.i1825 = icmp eq %struct.LIST_HELP* %166, null
  br i1 %cmp.i.i1825, label %list_Nconc.exit1835, label %if.end.i1827

if.end.i1827:                                     ; preds = %if.else250
  %cmp.i18.i1826 = icmp eq %struct.LIST_HELP* %167, null
  br i1 %cmp.i18.i1826, label %list_Nconc.exit1835, label %for.cond.i1832

for.cond.i1832:                                   ; preds = %if.end.i1827, %for.cond.i1832
  %List1.addr.0.i1828 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i1830, %for.cond.i1832 ], [ %166, %if.end.i1827 ]
  %List1.addr.0.idx15.i1829 = getelementptr %struct.LIST_HELP* %List1.addr.0.i1828, i32 0, i32 0
  %List1.addr.0.idx15.val.i1830 = load %struct.LIST_HELP** %List1.addr.0.idx15.i1829, align 4
  %cmp.i16.i1831 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i1830, null
  br i1 %cmp.i16.i1831, label %for.end.i1833, label %for.cond.i1832

for.end.i1833:                                    ; preds = %for.cond.i1832
  store %struct.LIST_HELP* %167, %struct.LIST_HELP** %List1.addr.0.idx15.i1829, align 4
  br label %list_Nconc.exit1835

list_Nconc.exit1835:                              ; preds = %if.else250, %if.end.i1827, %for.end.i1833
  %retval.0.i1834 = phi %struct.LIST_HELP* [ %166, %for.end.i1833 ], [ %167, %if.else250 ], [ %166, %if.end.i1827 ]
  store %struct.LIST_HELP* %retval.0.i1834, %struct.LIST_HELP** @dfg_CONJECLIST, align 4
  br label %sw.epilog1200

sw.bb255:                                         ; preds = %yyreduce
  store i32 1, i32* %2, align 4
  br label %sw.epilog1200

sw.bb257:                                         ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb259:                                         ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb262:                                         ; preds = %yyreduce
  %arrayidx263 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -3
  %term264 = bitcast %union.yystype* %arrayidx263 to %struct.term**
  %168 = load %struct.term** %term264, align 4
  %cmp265 = icmp eq %struct.term* %168, null
  %arrayidx268 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %string269 = bitcast %union.yystype* %arrayidx268 to i8**
  %169 = load i8** %string269, align 4
  br i1 %cmp265, label %if.then267, label %if.else279

if.then267:                                       ; preds = %sw.bb262
  %cmp270 = icmp eq i8* %169, null
  br i1 %cmp270, label %if.end275, label %if.then272

if.then272:                                       ; preds = %if.then267
  call void @string_StringFree(i8* %169) #1
  br label %if.end275

if.end275:                                        ; preds = %if.then267, %if.then272
  %arrayidx276 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -6
  %list277 = bitcast %union.yystype* %arrayidx276 to %struct.LIST_HELP**
  %170 = load %struct.LIST_HELP** %list277, align 4
  %.c1755 = ptrtoint %struct.LIST_HELP* %170 to i32
  br label %if.end289

if.else279:                                       ; preds = %sw.bb262
  %171 = bitcast %struct.term* %168 to %struct.LIST_HELP*
  %call.i.i1836 = call i8* @memory_Malloc(i32 8) #1
  %car.i.i1837 = getelementptr inbounds i8* %call.i.i1836, i32 4
  %172 = bitcast i8* %car.i.i1837 to i8**
  store i8* %169, i8** %172, align 4
  %cdr.i.i1838 = bitcast i8* %call.i.i1836 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %171, %struct.LIST_HELP** %cdr.i.i1838, align 4
  %arrayidx285 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -6
  %list286 = bitcast %union.yystype* %arrayidx285 to %struct.LIST_HELP**
  %173 = load %struct.LIST_HELP** %list286, align 4
  %call.i1839 = call i8* @memory_Malloc(i32 8) #1
  %car.i1840 = getelementptr inbounds i8* %call.i1839, i32 4
  %174 = bitcast i8* %car.i1840 to i8**
  store i8* %call.i.i1836, i8** %174, align 4
  %cdr.i1841 = bitcast i8* %call.i1839 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %173, %struct.LIST_HELP** %cdr.i1841, align 4
  %call287.c = ptrtoint i8* %call.i1839 to i32
  br label %if.end289

if.end289:                                        ; preds = %if.else279, %if.end275
  %storemerge2213 = phi i32 [ %call287.c, %if.else279 ], [ %.c1755, %if.end275 ]
  store i32 %storemerge2213, i32* %2, align 4
  %175 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %cmp.i.i1842 = icmp eq %struct.LIST_HELP* %175, null
  br i1 %cmp.i.i1842, label %dfg_VarCheck.exit1847, label %if.then.i1845

if.then.i1845:                                    ; preds = %if.end289
  %176 = load %struct._IO_FILE** @stdout, align 4
  %call1.i1843 = call i32 @fflush(%struct._IO_FILE* %176) #1
  %177 = load %struct._IO_FILE** @stderr, align 4
  %call2.i1844 = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %177, i8* getelementptr inbounds ([31 x i8]* @.str27, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str28, i32 0, i32 0), i32 1881) #1
  call void (i8*, ...)* @misc_ErrorReport(i8* getelementptr inbounds ([55 x i8]* @.str41, i32 0, i32 0)) #1
  %178 = load %struct._IO_FILE** @stderr, align 4
  %179 = call i32 @fwrite(i8* getelementptr inbounds ([133 x i8]* @.str30, i32 0, i32 0), i32 132, i32 1, %struct._IO_FILE* %178) #1
  call fastcc void @misc_DumpCore() #1
  unreachable

dfg_VarCheck.exit1847:                            ; preds = %if.end289
  store i32 0, i32* @symbol_STANDARDVARCOUNTER, align 4
  br label %sw.epilog1200

sw.bb290:                                         ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb292:                                         ; preds = %yyreduce
  %string294 = bitcast %union.yystype* %yyvsp.2 to i8**
  %180 = load i8** %string294, align 4
  %.c1754 = ptrtoint i8* %180 to i32
  store i32 %.c1754, i32* %2, align 4
  br label %sw.epilog1200

sw.bb296:                                         ; preds = %yyreduce
  %term298 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %181 = load %struct.term** %term298, align 4
  %.c1753 = ptrtoint %struct.term* %181 to i32
  store i32 %.c1753, i32* %2, align 4
  br label %sw.epilog1200

sw.bb300:                                         ; preds = %yyreduce
  %182 = load i32* @dfg_IGNORE, align 4
  %tobool301 = icmp eq i32 %182, 0
  br i1 %tobool301, label %cond.false303, label %cond.end309

cond.false303:                                    ; preds = %sw.bb300
  %183 = load i32* @fol_NOT, align 4
  %arrayidx305 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %term306 = bitcast %union.yystype* %arrayidx305 to %struct.term**
  %184 = load %struct.term** %term306, align 4
  %185 = bitcast %struct.term* %184 to i8*
  %call.i.i1848 = call i8* @memory_Malloc(i32 8) #1
  %186 = bitcast i8* %call.i.i1848 to %struct.LIST_HELP*
  %car.i.i1849 = getelementptr inbounds i8* %call.i.i1848, i32 4
  %187 = bitcast i8* %car.i.i1849 to i8**
  store i8* %185, i8** %187, align 4
  %cdr.i.i1850 = bitcast i8* %call.i.i1848 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1850, align 4
  %call308 = call %struct.term* @term_Create(i32 %183, %struct.LIST_HELP* %186) #1
  %phitmp1752 = ptrtoint %struct.term* %call308 to i32
  br label %cond.end309

cond.end309:                                      ; preds = %sw.bb300, %cond.false303
  %cond310 = phi i32 [ %phitmp1752, %cond.false303 ], [ 0, %sw.bb300 ]
  store i32 %cond310, i32* %2, align 4
  br label %sw.epilog1200

sw.bb312:                                         ; preds = %yyreduce
  %188 = load i32* @dfg_IGNORE, align 4
  %tobool313 = icmp eq i32 %188, 0
  br i1 %tobool313, label %cond.false315, label %cond.end324

cond.false315:                                    ; preds = %sw.bb312
  %symbol = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -5, i32 0
  %189 = load i32* %symbol, align 4
  %arrayidx317 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -3
  %term318 = bitcast %union.yystype* %arrayidx317 to %struct.term**
  %190 = load %struct.term** %term318, align 4
  %191 = bitcast %struct.term* %190 to i8*
  %arrayidx319 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %term320 = bitcast %union.yystype* %arrayidx319 to %struct.term**
  %192 = load %struct.term** %term320, align 4
  %193 = bitcast %struct.term* %192 to i8*
  %call.i.i1851 = call i8* @memory_Malloc(i32 8) #1
  %194 = bitcast i8* %call.i.i1851 to %struct.LIST_HELP*
  %car.i.i1852 = getelementptr inbounds i8* %call.i.i1851, i32 4
  %195 = bitcast i8* %car.i.i1852 to i8**
  store i8* %193, i8** %195, align 4
  %cdr.i.i1853 = bitcast i8* %call.i.i1851 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1853, align 4
  %call.i1854 = call i8* @memory_Malloc(i32 8) #1
  %196 = bitcast i8* %call.i1854 to %struct.LIST_HELP*
  %car.i1855 = getelementptr inbounds i8* %call.i1854, i32 4
  %197 = bitcast i8* %car.i1855 to i8**
  store i8* %191, i8** %197, align 4
  %cdr.i1856 = bitcast i8* %call.i1854 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %194, %struct.LIST_HELP** %cdr.i1856, align 4
  %call323 = call %struct.term* @term_Create(i32 %189, %struct.LIST_HELP* %196) #1
  %phitmp1751 = ptrtoint %struct.term* %call323 to i32
  br label %cond.end324

cond.end324:                                      ; preds = %sw.bb312, %cond.false315
  %cond325 = phi i32 [ %phitmp1751, %cond.false315 ], [ 0, %sw.bb312 ]
  store i32 %cond325, i32* %2, align 4
  br label %sw.epilog1200

sw.bb327:                                         ; preds = %yyreduce
  %198 = load i32* @dfg_IGNORE, align 4
  %tobool328 = icmp eq i32 %198, 0
  br i1 %tobool328, label %cond.false330, label %cond.end336

cond.false330:                                    ; preds = %sw.bb327
  %symbol332 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -3, i32 0
  %199 = load i32* %symbol332, align 4
  %arrayidx333 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %list334 = bitcast %union.yystype* %arrayidx333 to %struct.LIST_HELP**
  %200 = load %struct.LIST_HELP** %list334, align 4
  %call335 = call %struct.term* @term_Create(i32 %199, %struct.LIST_HELP* %200) #1
  %phitmp1750 = ptrtoint %struct.term* %call335 to i32
  br label %cond.end336

cond.end336:                                      ; preds = %sw.bb327, %cond.false330
  %cond337 = phi i32 [ %phitmp1750, %cond.false330 ], [ 0, %sw.bb327 ]
  store i32 %cond337, i32* %2, align 4
  br label %sw.epilog1200

sw.bb339:                                         ; preds = %yyreduce
  %201 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %call.i.i.i1857 = call i8* @memory_Malloc(i32 8) #1
  %202 = bitcast i8* %call.i.i.i1857 to %struct.LIST_HELP*
  %car.i.i.i1858 = getelementptr inbounds i8* %call.i.i.i1857, i32 4
  %203 = bitcast i8* %car.i.i.i1858 to i8**
  store i8* null, i8** %203, align 4
  %cdr.i.i.i1859 = bitcast i8* %call.i.i.i1857 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %201, %struct.LIST_HELP** %cdr.i.i.i1859, align 4
  store %struct.LIST_HELP* %202, %struct.LIST_HELP** @dfg_VARLIST, align 4
  store i1 true, i1* @dfg_VARDECL, align 1
  br label %sw.epilog1200

sw.bb340:                                         ; preds = %yyreduce
  store i1 false, i1* @dfg_VARDECL, align 1
  br label %sw.epilog1200

sw.bb341:                                         ; preds = %yyreduce
  %204 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %.idx.i1860 = getelementptr %struct.LIST_HELP* %204, i32 0, i32 1
  %.idx.val.i1861 = load i8** %.idx.i1860, align 4
  %205 = bitcast i8* %.idx.val.i1861 to %struct.LIST_HELP*
  call void @list_DeleteWithElement(%struct.LIST_HELP* %205, void (i8*)* bitcast (void (%struct.DFG_VARENTRY*)* @dfg_VarFree to void (i8*)*)) #1
  %206 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %L.idx.i.i1862 = getelementptr %struct.LIST_HELP* %206, i32 0, i32 0
  %L.idx.val.i.i1863 = load %struct.LIST_HELP** %L.idx.i.i1862, align 4
  %207 = bitcast %struct.LIST_HELP* %206 to i8*
  %208 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i1864 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %208, i32 0, i32 4
  %209 = load i32* %total_size.i.i.i.i1864, align 4
  %210 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i1865 = add i32 %210, %209
  store i32 %add24.i.i.i.i1865, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i1866 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %208, i32 0, i32 0
  %211 = load i8** %free.i.i.i.i1866, align 4
  %.c.i.i.i1867 = bitcast i8* %211 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i1867, %struct.LIST_HELP** %L.idx.i.i1862, align 4
  %212 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i1868 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %212, i32 0, i32 0
  store i8* %207, i8** %free27.i.i.i.i1868, align 4
  store %struct.LIST_HELP* %L.idx.val.i.i1863, %struct.LIST_HELP** @dfg_VARLIST, align 4
  %213 = load i32* @dfg_IGNORE, align 4
  %tobool342 = icmp eq i32 %213, 0
  br i1 %tobool342, label %cond.false344, label %cond.end352

cond.false344:                                    ; preds = %sw.bb341
  %symbol346 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -9, i32 0
  %214 = load i32* %symbol346, align 4
  %arrayidx347 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -5
  %list348 = bitcast %union.yystype* %arrayidx347 to %struct.LIST_HELP**
  %215 = load %struct.LIST_HELP** %list348, align 4
  %arrayidx349 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %term350 = bitcast %union.yystype* %arrayidx349 to %struct.term**
  %216 = load %struct.term** %term350, align 4
  %call351 = call %struct.term* @dfg_CreateQuantifier(i32 %214, %struct.LIST_HELP* %215, %struct.term* %216)
  %phitmp1749 = ptrtoint %struct.term* %call351 to i32
  br label %cond.end352

cond.end352:                                      ; preds = %sw.bb341, %cond.false344
  %cond353 = phi i32 [ %phitmp1749, %cond.false344 ], [ 0, %sw.bb341 ]
  store i32 %cond353, i32* %2, align 4
  br label %sw.epilog1200

sw.bb355:                                         ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb357:                                         ; preds = %yyreduce
  %term359 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %217 = load %struct.term** %term359, align 4
  %.c1748 = ptrtoint %struct.term* %217 to i32
  store i32 %.c1748, i32* %2, align 4
  br label %sw.epilog1200

sw.bb361:                                         ; preds = %yyreduce
  %218 = load i32* @dfg_IGNORE, align 4
  %tobool362 = icmp eq i32 %218, 0
  br i1 %tobool362, label %cond.false365, label %cond.end369

cond.false365:                                    ; preds = %sw.bb361
  %term367 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %219 = load %struct.term** %term367, align 4
  %220 = bitcast %struct.term* %219 to i8*
  %call.i.i1869 = call i8* @memory_Malloc(i32 8) #1
  %car.i.i1870 = getelementptr inbounds i8* %call.i.i1869, i32 4
  %221 = bitcast i8* %car.i.i1870 to i8**
  store i8* %220, i8** %221, align 4
  %cdr.i.i1871 = bitcast i8* %call.i.i1869 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1871, align 4
  %phitmp1747 = ptrtoint i8* %call.i.i1869 to i32
  br label %cond.end369

cond.end369:                                      ; preds = %sw.bb361, %cond.false365
  %cond370 = phi i32 [ %phitmp1747, %cond.false365 ], [ 0, %sw.bb361 ]
  store i32 %cond370, i32* %2, align 4
  br label %sw.epilog1200

sw.bb372:                                         ; preds = %yyreduce
  %222 = load i32* @dfg_IGNORE, align 4
  %tobool373 = icmp eq i32 %222, 0
  %arrayidx375 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %list376 = bitcast %union.yystype* %arrayidx375 to %struct.LIST_HELP**
  %223 = load %struct.LIST_HELP** %list376, align 4
  br i1 %tobool373, label %cond.false377, label %cond.end384

cond.false377:                                    ; preds = %sw.bb372
  %term381 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %224 = load %struct.term** %term381, align 4
  %225 = bitcast %struct.term* %224 to i8*
  %call.i.i1872 = call i8* @memory_Malloc(i32 8) #1
  %226 = bitcast i8* %call.i.i1872 to %struct.LIST_HELP*
  %car.i.i1873 = getelementptr inbounds i8* %call.i.i1872, i32 4
  %227 = bitcast i8* %car.i.i1873 to i8**
  store i8* %225, i8** %227, align 4
  %cdr.i.i1874 = bitcast i8* %call.i.i1872 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1874, align 4
  %cmp.i.i1875 = icmp eq %struct.LIST_HELP* %223, null
  br i1 %cmp.i.i1875, label %cond.end384, label %if.end.i1877

if.end.i1877:                                     ; preds = %cond.false377
  %cmp.i18.i1876 = icmp eq i8* %call.i.i1872, null
  br i1 %cmp.i18.i1876, label %cond.end384, label %for.cond.i1882

for.cond.i1882:                                   ; preds = %if.end.i1877, %for.cond.i1882
  %List1.addr.0.i1878 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i1880, %for.cond.i1882 ], [ %223, %if.end.i1877 ]
  %List1.addr.0.idx15.i1879 = getelementptr %struct.LIST_HELP* %List1.addr.0.i1878, i32 0, i32 0
  %List1.addr.0.idx15.val.i1880 = load %struct.LIST_HELP** %List1.addr.0.idx15.i1879, align 4
  %cmp.i16.i1881 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i1880, null
  br i1 %cmp.i16.i1881, label %for.end.i1883, label %for.cond.i1882

for.end.i1883:                                    ; preds = %for.cond.i1882
  store %struct.LIST_HELP* %226, %struct.LIST_HELP** %List1.addr.0.idx15.i1879, align 4
  br label %cond.end384

cond.end384:                                      ; preds = %for.end.i1883, %if.end.i1877, %cond.false377, %sw.bb372
  %cond385 = phi %struct.LIST_HELP* [ %223, %sw.bb372 ], [ %223, %for.end.i1883 ], [ %226, %cond.false377 ], [ %223, %if.end.i1877 ]
  %cond385.c = ptrtoint %struct.LIST_HELP* %cond385 to i32
  store i32 %cond385.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb387:                                         ; preds = %yyreduce
  %228 = load i32* @fol_EQUIV, align 4
  store i32 %228, i32* %2, align 4
  br label %sw.epilog1200

sw.bb390:                                         ; preds = %yyreduce
  %229 = load i32* @fol_IMPLIED, align 4
  store i32 %229, i32* %2, align 4
  br label %sw.epilog1200

sw.bb393:                                         ; preds = %yyreduce
  %230 = load i32* @fol_IMPLIES, align 4
  store i32 %230, i32* %2, align 4
  br label %sw.epilog1200

sw.bb396:                                         ; preds = %yyreduce
  %231 = load i32* @fol_AND, align 4
  store i32 %231, i32* %2, align 4
  br label %sw.epilog1200

sw.bb399:                                         ; preds = %yyreduce
  %232 = load i32* @fol_OR, align 4
  store i32 %232, i32* %2, align 4
  br label %sw.epilog1200

sw.bb402:                                         ; preds = %yyreduce
  %233 = load i32* @fol_EXIST, align 4
  store i32 %233, i32* %2, align 4
  br label %sw.epilog1200

sw.bb405:                                         ; preds = %yyreduce
  %234 = load i32* @fol_ALL, align 4
  store i32 %234, i32* %2, align 4
  br label %sw.epilog1200

sw.bb408:                                         ; preds = %yyreduce
  %235 = load i32* @dfg_IGNORE, align 4
  %tobool409 = icmp eq i32 %235, 0
  %string412 = bitcast %union.yystype* %yyvsp.2 to i8**
  %236 = load i8** %string412, align 4
  br i1 %tobool409, label %if.else414, label %if.then410

if.then410:                                       ; preds = %sw.bb408
  call void @string_StringFree(i8* %236) #1
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

if.else414:                                       ; preds = %sw.bb408
  %.c1746 = ptrtoint i8* %236 to i32
  store i32 %.c1746, i32* %2, align 4
  br label %sw.epilog1200

sw.bb419:                                         ; preds = %yyreduce
  %237 = load i32* @dfg_IGNORE, align 4
  %tobool420 = icmp eq i32 %237, 0
  br i1 %tobool420, label %cond.false422, label %cond.end426

cond.false422:                                    ; preds = %sw.bb419
  %number424 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 0, i32 0
  %238 = load i32* %number424, align 4
  %call425 = call i8* @string_IntToString(i32 %238) #1
  %phitmp1745 = ptrtoint i8* %call425 to i32
  br label %cond.end426

cond.end426:                                      ; preds = %sw.bb419, %cond.false422
  %cond427 = phi i32 [ %phitmp1745, %cond.false422 ], [ 0, %sw.bb419 ]
  store i32 %cond427, i32* %2, align 4
  br label %sw.epilog1200

sw.bb429:                                         ; preds = %yyreduce
  %239 = load i32* @dfg_IGNORE, align 4
  %tobool430 = icmp eq i32 %239, 0
  br i1 %tobool430, label %cond.false432, label %cond.end434

cond.false432:                                    ; preds = %sw.bb429
  %call433 = call i8* @string_StringCopy(i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0)) #1
  %phitmp1744 = ptrtoint i8* %call433 to i32
  br label %cond.end434

cond.end434:                                      ; preds = %sw.bb429, %cond.false432
  %cond435 = phi i32 [ %phitmp1744, %cond.false432 ], [ 0, %sw.bb429 ]
  store i32 %cond435, i32* %2, align 4
  br label %sw.epilog1200

sw.bb437:                                         ; preds = %yyreduce
  %240 = load i32* @dfg_IGNORE, align 4
  %tobool438 = icmp eq i32 %240, 0
  br i1 %tobool438, label %cond.false440, label %cond.end442

cond.false440:                                    ; preds = %sw.bb437
  %call441 = call i8* @string_StringCopy(i8* getelementptr inbounds ([12 x i8]* @.str1, i32 0, i32 0)) #1
  %phitmp1743 = ptrtoint i8* %call441 to i32
  br label %cond.end442

cond.end442:                                      ; preds = %sw.bb437, %cond.false440
  %cond443 = phi i32 [ %phitmp1743, %cond.false440 ], [ 0, %sw.bb437 ]
  store i32 %cond443, i32* %2, align 4
  br label %sw.epilog1200

sw.bb445:                                         ; preds = %yyreduce
  %241 = load i32* @dfg_IGNORE, align 4
  %tobool446 = icmp eq i32 %241, 0
  br i1 %tobool446, label %cond.false448, label %cond.end450

cond.false448:                                    ; preds = %sw.bb445
  %call449 = call i8* @string_StringCopy(i8* getelementptr inbounds ([15 x i8]* @.str2, i32 0, i32 0)) #1
  %phitmp1742 = ptrtoint i8* %call449 to i32
  br label %cond.end450

cond.end450:                                      ; preds = %sw.bb445, %cond.false448
  %cond451 = phi i32 [ %phitmp1742, %cond.false448 ], [ 0, %sw.bb445 ]
  store i32 %cond451, i32* %2, align 4
  br label %sw.epilog1200

sw.bb453:                                         ; preds = %yyreduce
  %242 = load i32* @dfg_IGNORE, align 4
  %tobool454 = icmp eq i32 %242, 0
  br i1 %tobool454, label %cond.false457, label %cond.end461

cond.false457:                                    ; preds = %sw.bb453
  %term459 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %243 = load %struct.term** %term459, align 4
  %244 = bitcast %struct.term* %243 to i8*
  %call.i.i1886 = call i8* @memory_Malloc(i32 8) #1
  %car.i.i1887 = getelementptr inbounds i8* %call.i.i1886, i32 4
  %245 = bitcast i8* %car.i.i1887 to i8**
  store i8* %244, i8** %245, align 4
  %cdr.i.i1888 = bitcast i8* %call.i.i1886 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1888, align 4
  %phitmp1741 = ptrtoint i8* %call.i.i1886 to i32
  br label %cond.end461

cond.end461:                                      ; preds = %sw.bb453, %cond.false457
  %cond462 = phi i32 [ %phitmp1741, %cond.false457 ], [ 0, %sw.bb453 ]
  store i32 %cond462, i32* %2, align 4
  br label %sw.epilog1200

sw.bb464:                                         ; preds = %yyreduce
  %246 = load i32* @dfg_IGNORE, align 4
  %tobool465 = icmp eq i32 %246, 0
  %arrayidx467 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %list468 = bitcast %union.yystype* %arrayidx467 to %struct.LIST_HELP**
  %247 = load %struct.LIST_HELP** %list468, align 4
  br i1 %tobool465, label %cond.false469, label %cond.end476

cond.false469:                                    ; preds = %sw.bb464
  %term473 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %248 = load %struct.term** %term473, align 4
  %249 = bitcast %struct.term* %248 to i8*
  %call.i.i1889 = call i8* @memory_Malloc(i32 8) #1
  %250 = bitcast i8* %call.i.i1889 to %struct.LIST_HELP*
  %car.i.i1890 = getelementptr inbounds i8* %call.i.i1889, i32 4
  %251 = bitcast i8* %car.i.i1890 to i8**
  store i8* %249, i8** %251, align 4
  %cdr.i.i1891 = bitcast i8* %call.i.i1889 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1891, align 4
  %cmp.i.i1892 = icmp eq %struct.LIST_HELP* %247, null
  br i1 %cmp.i.i1892, label %cond.end476, label %if.end.i1894

if.end.i1894:                                     ; preds = %cond.false469
  %cmp.i18.i1893 = icmp eq i8* %call.i.i1889, null
  br i1 %cmp.i18.i1893, label %cond.end476, label %for.cond.i1899

for.cond.i1899:                                   ; preds = %if.end.i1894, %for.cond.i1899
  %List1.addr.0.i1895 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i1897, %for.cond.i1899 ], [ %247, %if.end.i1894 ]
  %List1.addr.0.idx15.i1896 = getelementptr %struct.LIST_HELP* %List1.addr.0.i1895, i32 0, i32 0
  %List1.addr.0.idx15.val.i1897 = load %struct.LIST_HELP** %List1.addr.0.idx15.i1896, align 4
  %cmp.i16.i1898 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i1897, null
  br i1 %cmp.i16.i1898, label %for.end.i1900, label %for.cond.i1899

for.end.i1900:                                    ; preds = %for.cond.i1899
  store %struct.LIST_HELP* %250, %struct.LIST_HELP** %List1.addr.0.idx15.i1896, align 4
  br label %cond.end476

cond.end476:                                      ; preds = %for.end.i1900, %if.end.i1894, %cond.false469, %sw.bb464
  %cond477 = phi %struct.LIST_HELP* [ %247, %sw.bb464 ], [ %247, %for.end.i1900 ], [ %250, %cond.false469 ], [ %247, %if.end.i1894 ]
  %cond477.c = ptrtoint %struct.LIST_HELP* %cond477 to i32
  store i32 %cond477.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb479:                                         ; preds = %yyreduce
  %252 = load i32* @dfg_IGNORE, align 4
  %tobool480 = icmp eq i32 %252, 0
  br i1 %tobool480, label %if.then481, label %sw.epilog1200

if.then481:                                       ; preds = %sw.bb479
  %string483 = bitcast %union.yystype* %yyvsp.2 to i8**
  %253 = load i8** %string483, align 4
  %call484 = call fastcc i32 @dfg_Symbol(i8* %253, i32 0)
  %cmp.i1903 = icmp sgt i32 %call484, 0
  br i1 %cmp.i1903, label %if.end489, label %if.then487

if.then487:                                       ; preds = %if.then481
  %254 = load %struct._IO_FILE** @stdout, align 4
  %call488 = call i32 @fflush(%struct._IO_FILE* %254) #1
  %255 = load i32* @dfg_LINENUMBER, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([38 x i8]* @.str3, i32 0, i32 0), i32 %255) #1
  call fastcc void @misc_Error()
  unreachable

if.end489:                                        ; preds = %if.then481
  %call491 = call %struct.term* @term_Create(i32 %call484, %struct.LIST_HELP* null) #1
  %call491.c = ptrtoint %struct.term* %call491 to i32
  store i32 %call491.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb494:                                         ; preds = %yyreduce
  %256 = load i32* @dfg_IGNORE, align 4
  %tobool495 = icmp eq i32 %256, 0
  br i1 %tobool495, label %if.then496, label %sw.epilog1200

if.then496:                                       ; preds = %sw.bb494
  %arrayidx497 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -3
  %string498 = bitcast %union.yystype* %arrayidx497 to i8**
  %257 = load i8** %string498, align 4
  %call499 = call fastcc i32 @dfg_Symbol(i8* %257, i32 1)
  %tobool.i = icmp sgt i32 %call499, -1
  br i1 %tobool.i, label %if.then502, label %land.rhs.i

land.rhs.i:                                       ; preds = %if.then496
  %sub.i.i1904 = sub nsw i32 0, %call499
  %and.i.i1905 = and i32 %3, %sub.i.i1904
  %cmp.i1906 = icmp eq i32 %and.i.i1905, 2
  br i1 %cmp.i1906, label %if.end504, label %if.then502

if.then502:                                       ; preds = %if.then496, %land.rhs.i
  %258 = load %struct._IO_FILE** @stdout, align 4
  %call503 = call i32 @fflush(%struct._IO_FILE* %258) #1
  %259 = load i32* @dfg_LINENUMBER, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([39 x i8]* @.str4, i32 0, i32 0), i32 %259) #1
  call fastcc void @misc_Error()
  unreachable

if.end504:                                        ; preds = %land.rhs.i
  %arrayidx505 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %string506 = bitcast %union.yystype* %arrayidx505 to i8**
  %260 = load i8** %string506, align 4
  %call507 = call fastcc i32 @dfg_Symbol(i8* %260, i32 0)
  %cmp.i1907 = icmp sgt i32 %call507, 0
  br i1 %cmp.i1907, label %if.end512, label %if.then510

if.then510:                                       ; preds = %if.end504
  %261 = load %struct._IO_FILE** @stdout, align 4
  %call511 = call i32 @fflush(%struct._IO_FILE* %261) #1
  %262 = load i32* @dfg_LINENUMBER, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([38 x i8]* @.str3, i32 0, i32 0), i32 %262) #1
  call fastcc void @misc_Error()
  unreachable

if.end512:                                        ; preds = %if.end504
  %call514 = call %struct.term* @term_Create(i32 %call507, %struct.LIST_HELP* null) #1
  %263 = bitcast %struct.term* %call514 to i8*
  %call.i.i1909 = call i8* @memory_Malloc(i32 8) #1
  %264 = bitcast i8* %call.i.i1909 to %struct.LIST_HELP*
  %car.i.i1910 = getelementptr inbounds i8* %call.i.i1909, i32 4
  %265 = bitcast i8* %car.i.i1910 to i8**
  store i8* %263, i8** %265, align 4
  %cdr.i.i1911 = bitcast i8* %call.i.i1909 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1911, align 4
  %call516 = call %struct.term* @term_Create(i32 %call499, %struct.LIST_HELP* %264) #1
  %call516.c = ptrtoint %struct.term* %call516 to i32
  store i32 %call516.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb519:                                         ; preds = %yyreduce
  %arrayidx520 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %list521 = bitcast %union.yystype* %arrayidx520 to %struct.LIST_HELP**
  %266 = load %struct.LIST_HELP** %list521, align 4
  %call522 = call %struct.LIST_HELP* @list_NReverse(%struct.LIST_HELP* %266) #1
  %bool524 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -7, i32 0
  %267 = load i32* %bool524, align 4
  %tobool525 = icmp eq i32 %267, 0
  br i1 %tobool525, label %if.else530, label %if.then526

if.then526:                                       ; preds = %sw.bb519
  %268 = load %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  %269 = load %struct.LIST_HELP** %list521, align 4
  %cmp.i.i1912 = icmp eq %struct.LIST_HELP* %268, null
  br i1 %cmp.i.i1912, label %list_Nconc.exit1922, label %if.end.i1914

if.end.i1914:                                     ; preds = %if.then526
  %cmp.i18.i1913 = icmp eq %struct.LIST_HELP* %269, null
  br i1 %cmp.i18.i1913, label %list_Nconc.exit1922, label %for.cond.i1919

for.cond.i1919:                                   ; preds = %if.end.i1914, %for.cond.i1919
  %List1.addr.0.i1915 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i1917, %for.cond.i1919 ], [ %268, %if.end.i1914 ]
  %List1.addr.0.idx15.i1916 = getelementptr %struct.LIST_HELP* %List1.addr.0.i1915, i32 0, i32 0
  %List1.addr.0.idx15.val.i1917 = load %struct.LIST_HELP** %List1.addr.0.idx15.i1916, align 4
  %cmp.i16.i1918 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i1917, null
  br i1 %cmp.i16.i1918, label %for.end.i1920, label %for.cond.i1919

for.end.i1920:                                    ; preds = %for.cond.i1919
  store %struct.LIST_HELP* %269, %struct.LIST_HELP** %List1.addr.0.idx15.i1916, align 4
  br label %list_Nconc.exit1922

list_Nconc.exit1922:                              ; preds = %if.then526, %if.end.i1914, %for.end.i1920
  %retval.0.i1921 = phi %struct.LIST_HELP* [ %268, %for.end.i1920 ], [ %269, %if.then526 ], [ %268, %if.end.i1914 ]
  store %struct.LIST_HELP* %retval.0.i1921, %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  br label %sw.epilog1200

if.else530:                                       ; preds = %sw.bb519
  %270 = load %struct.LIST_HELP** @dfg_CONCLAUSES, align 4
  %271 = load %struct.LIST_HELP** %list521, align 4
  %cmp.i.i1923 = icmp eq %struct.LIST_HELP* %270, null
  br i1 %cmp.i.i1923, label %list_Nconc.exit1933, label %if.end.i1925

if.end.i1925:                                     ; preds = %if.else530
  %cmp.i18.i1924 = icmp eq %struct.LIST_HELP* %271, null
  br i1 %cmp.i18.i1924, label %list_Nconc.exit1933, label %for.cond.i1930

for.cond.i1930:                                   ; preds = %if.end.i1925, %for.cond.i1930
  %List1.addr.0.i1926 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i1928, %for.cond.i1930 ], [ %270, %if.end.i1925 ]
  %List1.addr.0.idx15.i1927 = getelementptr %struct.LIST_HELP* %List1.addr.0.i1926, i32 0, i32 0
  %List1.addr.0.idx15.val.i1928 = load %struct.LIST_HELP** %List1.addr.0.idx15.i1927, align 4
  %cmp.i16.i1929 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i1928, null
  br i1 %cmp.i16.i1929, label %for.end.i1931, label %for.cond.i1930

for.end.i1931:                                    ; preds = %for.cond.i1930
  store %struct.LIST_HELP* %271, %struct.LIST_HELP** %List1.addr.0.idx15.i1927, align 4
  br label %list_Nconc.exit1933

list_Nconc.exit1933:                              ; preds = %if.else530, %if.end.i1925, %for.end.i1931
  %retval.0.i1932 = phi %struct.LIST_HELP* [ %270, %for.end.i1931 ], [ %271, %if.else530 ], [ %270, %if.end.i1925 ]
  store %struct.LIST_HELP* %retval.0.i1932, %struct.LIST_HELP** @dfg_CONCLAUSES, align 4
  br label %sw.epilog1200

sw.bb535:                                         ; preds = %yyreduce
  %272 = load i32* @dfg_IGNORE, align 4
  %273 = inttoptr i32 %272 to i8*
  %274 = load i32* @stack_POINTER, align 4
  %inc.i = add i32 %274, 1
  store i32 %inc.i, i32* @stack_POINTER, align 4
  %arrayidx.i = getelementptr inbounds [10000 x i8*]* @stack_STACK, i32 0, i32 %274
  store i8* %273, i8** %arrayidx.i, align 4
  store i32 1, i32* @dfg_IGNORE, align 4
  br label %sw.epilog1200

sw.bb536:                                         ; preds = %yyreduce
  %275 = load i32* @stack_POINTER, align 4
  %dec.i = add i32 %275, -1
  store i32 %dec.i, i32* @stack_POINTER, align 4
  %arrayidx.i1934 = getelementptr inbounds [10000 x i8*]* @stack_STACK, i32 0, i32 %dec.i
  %276 = load i8** %arrayidx.i1934, align 4
  %277 = ptrtoint i8* %276 to i32
  store i32 %277, i32* @dfg_IGNORE, align 4
  br label %sw.epilog1200

sw.bb538:                                         ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb541:                                         ; preds = %yyreduce
  %arrayidx543 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -3
  %term544 = bitcast %union.yystype* %arrayidx543 to %struct.term**
  %278 = load %struct.term** %term544, align 4
  %cmp545 = icmp eq %struct.term* %278, null
  %arrayidx548 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %string549 = bitcast %union.yystype* %arrayidx548 to i8**
  %279 = load i8** %string549, align 4
  br i1 %cmp545, label %if.then547, label %if.else559

if.then547:                                       ; preds = %sw.bb541
  %cmp550 = icmp eq i8* %279, null
  br i1 %cmp550, label %if.end555, label %if.then552

if.then552:                                       ; preds = %if.then547
  call void @string_StringFree(i8* %279) #1
  br label %if.end555

if.end555:                                        ; preds = %if.then547, %if.then552
  %arrayidx556 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -6
  %list557 = bitcast %union.yystype* %arrayidx556 to %struct.LIST_HELP**
  %280 = load %struct.LIST_HELP** %list557, align 4
  %.c1740 = ptrtoint %struct.LIST_HELP* %280 to i32
  br label %if.end569

if.else559:                                       ; preds = %sw.bb541
  %281 = bitcast %struct.term* %278 to %struct.LIST_HELP*
  %call.i.i1935 = call i8* @memory_Malloc(i32 8) #1
  %car.i.i1936 = getelementptr inbounds i8* %call.i.i1935, i32 4
  %282 = bitcast i8* %car.i.i1936 to i8**
  store i8* %279, i8** %282, align 4
  %cdr.i.i1937 = bitcast i8* %call.i.i1935 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %281, %struct.LIST_HELP** %cdr.i.i1937, align 4
  %arrayidx565 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -6
  %list566 = bitcast %union.yystype* %arrayidx565 to %struct.LIST_HELP**
  %283 = load %struct.LIST_HELP** %list566, align 4
  %call.i1938 = call i8* @memory_Malloc(i32 8) #1
  %car.i1939 = getelementptr inbounds i8* %call.i1938, i32 4
  %284 = bitcast i8* %car.i1939 to i8**
  store i8* %call.i.i1935, i8** %284, align 4
  %cdr.i1940 = bitcast i8* %call.i1938 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %283, %struct.LIST_HELP** %cdr.i1940, align 4
  %call567.c = ptrtoint i8* %call.i1938 to i32
  br label %if.end569

if.end569:                                        ; preds = %if.else559, %if.end555
  %storemerge = phi i32 [ %call567.c, %if.else559 ], [ %.c1740, %if.end555 ]
  store i32 %storemerge, i32* %2, align 4
  %285 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %cmp.i.i1941 = icmp eq %struct.LIST_HELP* %285, null
  br i1 %cmp.i.i1941, label %dfg_VarCheck.exit1946, label %if.then.i1944

if.then.i1944:                                    ; preds = %if.end569
  %286 = load %struct._IO_FILE** @stdout, align 4
  %call1.i1942 = call i32 @fflush(%struct._IO_FILE* %286) #1
  %287 = load %struct._IO_FILE** @stderr, align 4
  %call2.i1943 = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %287, i8* getelementptr inbounds ([31 x i8]* @.str27, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str28, i32 0, i32 0), i32 1881) #1
  call void (i8*, ...)* @misc_ErrorReport(i8* getelementptr inbounds ([55 x i8]* @.str41, i32 0, i32 0)) #1
  %288 = load %struct._IO_FILE** @stderr, align 4
  %289 = call i32 @fwrite(i8* getelementptr inbounds ([133 x i8]* @.str30, i32 0, i32 0), i32 132, i32 1, %struct._IO_FILE* %288) #1
  call fastcc void @misc_DumpCore() #1
  unreachable

dfg_VarCheck.exit1946:                            ; preds = %if.end569
  store i32 0, i32* @symbol_STANDARDVARCOUNTER, align 4
  br label %sw.epilog1200

sw.bb570:                                         ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb572:                                         ; preds = %yyreduce
  %term574 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %290 = load %struct.term** %term574, align 4
  %.c1739 = ptrtoint %struct.term* %290 to i32
  store i32 %.c1739, i32* %2, align 4
  br label %sw.epilog1200

sw.bb576:                                         ; preds = %yyreduce
  %term578 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %291 = load %struct.term** %term578, align 4
  %.c1738 = ptrtoint %struct.term* %291 to i32
  store i32 %.c1738, i32* %2, align 4
  br label %sw.epilog1200

sw.bb580:                                         ; preds = %yyreduce
  %292 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %call.i.i.i1947 = call i8* @memory_Malloc(i32 8) #1
  %293 = bitcast i8* %call.i.i.i1947 to %struct.LIST_HELP*
  %car.i.i.i1948 = getelementptr inbounds i8* %call.i.i.i1947, i32 4
  %294 = bitcast i8* %car.i.i.i1948 to i8**
  store i8* null, i8** %294, align 4
  %cdr.i.i.i1949 = bitcast i8* %call.i.i.i1947 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %292, %struct.LIST_HELP** %cdr.i.i.i1949, align 4
  store %struct.LIST_HELP* %293, %struct.LIST_HELP** @dfg_VARLIST, align 4
  store i1 true, i1* @dfg_VARDECL, align 1
  br label %sw.epilog1200

sw.bb581:                                         ; preds = %yyreduce
  store i1 false, i1* @dfg_VARDECL, align 1
  br label %sw.epilog1200

sw.bb582:                                         ; preds = %yyreduce
  %295 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %.idx.i1950 = getelementptr %struct.LIST_HELP* %295, i32 0, i32 1
  %.idx.val.i1951 = load i8** %.idx.i1950, align 4
  %296 = bitcast i8* %.idx.val.i1951 to %struct.LIST_HELP*
  call void @list_DeleteWithElement(%struct.LIST_HELP* %296, void (i8*)* bitcast (void (%struct.DFG_VARENTRY*)* @dfg_VarFree to void (i8*)*)) #1
  %297 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %L.idx.i.i1952 = getelementptr %struct.LIST_HELP* %297, i32 0, i32 0
  %L.idx.val.i.i1953 = load %struct.LIST_HELP** %L.idx.i.i1952, align 4
  %298 = bitcast %struct.LIST_HELP* %297 to i8*
  %299 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i1954 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %299, i32 0, i32 4
  %300 = load i32* %total_size.i.i.i.i1954, align 4
  %301 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i1955 = add i32 %301, %300
  store i32 %add24.i.i.i.i1955, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i1956 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %299, i32 0, i32 0
  %302 = load i8** %free.i.i.i.i1956, align 4
  %.c.i.i.i1957 = bitcast i8* %302 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i1957, %struct.LIST_HELP** %L.idx.i.i1952, align 4
  %303 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i1958 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %303, i32 0, i32 0
  store i8* %298, i8** %free27.i.i.i.i1958, align 4
  store %struct.LIST_HELP* %L.idx.val.i.i1953, %struct.LIST_HELP** @dfg_VARLIST, align 4
  %304 = load i32* @dfg_IGNORE, align 4
  %tobool583 = icmp eq i32 %304, 0
  br i1 %tobool583, label %cond.false585, label %cond.end592

cond.false585:                                    ; preds = %sw.bb582
  %305 = load i32* @fol_ALL, align 4
  %arrayidx587 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -5
  %list588 = bitcast %union.yystype* %arrayidx587 to %struct.LIST_HELP**
  %306 = load %struct.LIST_HELP** %list588, align 4
  %arrayidx589 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %term590 = bitcast %union.yystype* %arrayidx589 to %struct.term**
  %307 = load %struct.term** %term590, align 4
  %call591 = call %struct.term* @dfg_CreateQuantifier(i32 %305, %struct.LIST_HELP* %306, %struct.term* %307)
  %phitmp1737 = ptrtoint %struct.term* %call591 to i32
  br label %cond.end592

cond.end592:                                      ; preds = %sw.bb582, %cond.false585
  %cond593 = phi i32 [ %phitmp1737, %cond.false585 ], [ 0, %sw.bb582 ]
  store i32 %cond593, i32* %2, align 4
  br label %sw.epilog1200

sw.bb595:                                         ; preds = %yyreduce
  %308 = load i32* @dfg_IGNORE, align 4
  %tobool596 = icmp eq i32 %308, 0
  br i1 %tobool596, label %cond.false598, label %cond.end603

cond.false598:                                    ; preds = %sw.bb595
  %309 = load i32* @fol_OR, align 4
  %arrayidx600 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %list601 = bitcast %union.yystype* %arrayidx600 to %struct.LIST_HELP**
  %310 = load %struct.LIST_HELP** %list601, align 4
  %call602 = call %struct.term* @term_Create(i32 %309, %struct.LIST_HELP* %310) #1
  %phitmp1736 = ptrtoint %struct.term* %call602 to i32
  br label %cond.end603

cond.end603:                                      ; preds = %sw.bb595, %cond.false598
  %cond604 = phi i32 [ %phitmp1736, %cond.false598 ], [ 0, %sw.bb595 ]
  store i32 %cond604, i32* %2, align 4
  br label %sw.epilog1200

sw.bb606:                                         ; preds = %yyreduce
  %311 = load i32* @dfg_IGNORE, align 4
  %tobool607 = icmp eq i32 %311, 0
  br i1 %tobool607, label %cond.false610, label %cond.end614

cond.false610:                                    ; preds = %sw.bb606
  %term612 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %312 = load %struct.term** %term612, align 4
  %313 = bitcast %struct.term* %312 to i8*
  %call.i.i1959 = call i8* @memory_Malloc(i32 8) #1
  %car.i.i1960 = getelementptr inbounds i8* %call.i.i1959, i32 4
  %314 = bitcast i8* %car.i.i1960 to i8**
  store i8* %313, i8** %314, align 4
  %cdr.i.i1961 = bitcast i8* %call.i.i1959 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1961, align 4
  %phitmp1735 = ptrtoint i8* %call.i.i1959 to i32
  br label %cond.end614

cond.end614:                                      ; preds = %sw.bb606, %cond.false610
  %cond615 = phi i32 [ %phitmp1735, %cond.false610 ], [ 0, %sw.bb606 ]
  store i32 %cond615, i32* %2, align 4
  br label %sw.epilog1200

sw.bb617:                                         ; preds = %yyreduce
  %315 = load i32* @dfg_IGNORE, align 4
  %tobool618 = icmp eq i32 %315, 0
  %arrayidx620 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %list621 = bitcast %union.yystype* %arrayidx620 to %struct.LIST_HELP**
  %316 = load %struct.LIST_HELP** %list621, align 4
  br i1 %tobool618, label %cond.false622, label %cond.end629

cond.false622:                                    ; preds = %sw.bb617
  %term626 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %317 = load %struct.term** %term626, align 4
  %318 = bitcast %struct.term* %317 to i8*
  %call.i.i1962 = call i8* @memory_Malloc(i32 8) #1
  %319 = bitcast i8* %call.i.i1962 to %struct.LIST_HELP*
  %car.i.i1963 = getelementptr inbounds i8* %call.i.i1962, i32 4
  %320 = bitcast i8* %car.i.i1963 to i8**
  store i8* %318, i8** %320, align 4
  %cdr.i.i1964 = bitcast i8* %call.i.i1962 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1964, align 4
  %cmp.i.i1965 = icmp eq %struct.LIST_HELP* %316, null
  br i1 %cmp.i.i1965, label %cond.end629, label %if.end.i1967

if.end.i1967:                                     ; preds = %cond.false622
  %cmp.i18.i1966 = icmp eq i8* %call.i.i1962, null
  br i1 %cmp.i18.i1966, label %cond.end629, label %for.cond.i1972

for.cond.i1972:                                   ; preds = %if.end.i1967, %for.cond.i1972
  %List1.addr.0.i1968 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i1970, %for.cond.i1972 ], [ %316, %if.end.i1967 ]
  %List1.addr.0.idx15.i1969 = getelementptr %struct.LIST_HELP* %List1.addr.0.i1968, i32 0, i32 0
  %List1.addr.0.idx15.val.i1970 = load %struct.LIST_HELP** %List1.addr.0.idx15.i1969, align 4
  %cmp.i16.i1971 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i1970, null
  br i1 %cmp.i16.i1971, label %for.end.i1973, label %for.cond.i1972

for.end.i1973:                                    ; preds = %for.cond.i1972
  store %struct.LIST_HELP* %319, %struct.LIST_HELP** %List1.addr.0.idx15.i1969, align 4
  br label %cond.end629

cond.end629:                                      ; preds = %for.end.i1973, %if.end.i1967, %cond.false622, %sw.bb617
  %cond630 = phi %struct.LIST_HELP* [ %316, %sw.bb617 ], [ %316, %for.end.i1973 ], [ %319, %cond.false622 ], [ %316, %if.end.i1967 ]
  %cond630.c = ptrtoint %struct.LIST_HELP* %cond630 to i32
  store i32 %cond630.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb632:                                         ; preds = %yyreduce
  %term634 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %321 = load %struct.term** %term634, align 4
  %.c1734 = ptrtoint %struct.term* %321 to i32
  store i32 %.c1734, i32* %2, align 4
  br label %sw.epilog1200

sw.bb636:                                         ; preds = %yyreduce
  %322 = load i32* @dfg_IGNORE, align 4
  %tobool637 = icmp eq i32 %322, 0
  br i1 %tobool637, label %cond.false641, label %cond.true638

cond.true638:                                     ; preds = %sw.bb636
  %arrayidx639 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %term640 = bitcast %union.yystype* %arrayidx639 to %struct.term**
  %323 = load %struct.term** %term640, align 4
  br label %cond.end647

cond.false641:                                    ; preds = %sw.bb636
  %324 = load i32* @fol_NOT, align 4
  %arrayidx643 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %term644 = bitcast %union.yystype* %arrayidx643 to %struct.term**
  %325 = load %struct.term** %term644, align 4
  %326 = bitcast %struct.term* %325 to i8*
  %call.i.i1976 = call i8* @memory_Malloc(i32 8) #1
  %327 = bitcast i8* %call.i.i1976 to %struct.LIST_HELP*
  %car.i.i1977 = getelementptr inbounds i8* %call.i.i1976, i32 4
  %328 = bitcast i8* %car.i.i1977 to i8**
  store i8* %326, i8** %328, align 4
  %cdr.i.i1978 = bitcast i8* %call.i.i1976 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1978, align 4
  %call646 = call %struct.term* @term_Create(i32 %324, %struct.LIST_HELP* %327) #1
  br label %cond.end647

cond.end647:                                      ; preds = %cond.false641, %cond.true638
  %cond648 = phi %struct.term* [ %323, %cond.true638 ], [ %call646, %cond.false641 ]
  %cond648.c = ptrtoint %struct.term* %cond648 to i32
  store i32 %cond648.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb650:                                         ; preds = %yyreduce
  %term652 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %329 = load %struct.term** %term652, align 4
  %330 = bitcast %struct.term* %329 to i8*
  %call.i.i1979 = call i8* @memory_Malloc(i32 8) #1
  %car.i.i1980 = getelementptr inbounds i8* %call.i.i1979, i32 4
  %331 = bitcast i8* %car.i.i1980 to i8**
  store i8* %330, i8** %331, align 4
  %cdr.i.i1981 = bitcast i8* %call.i.i1979 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1981, align 4
  %call653.c = ptrtoint i8* %call.i.i1979 to i32
  store i32 %call653.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb655:                                         ; preds = %yyreduce
  %arrayidx656 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %list657 = bitcast %union.yystype* %arrayidx656 to %struct.LIST_HELP**
  %332 = load %struct.LIST_HELP** %list657, align 4
  %term659 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %333 = load %struct.term** %term659, align 4
  %334 = bitcast %struct.term* %333 to i8*
  %call.i.i1982 = call i8* @memory_Malloc(i32 8) #1
  %335 = bitcast i8* %call.i.i1982 to %struct.LIST_HELP*
  %car.i.i1983 = getelementptr inbounds i8* %call.i.i1982, i32 4
  %336 = bitcast i8* %car.i.i1983 to i8**
  store i8* %334, i8** %336, align 4
  %cdr.i.i1984 = bitcast i8* %call.i.i1982 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i1984, align 4
  %cmp.i.i1985 = icmp eq %struct.LIST_HELP* %332, null
  br i1 %cmp.i.i1985, label %list_Nconc.exit1995, label %if.end.i1987

if.end.i1987:                                     ; preds = %sw.bb655
  %cmp.i18.i1986 = icmp eq i8* %call.i.i1982, null
  br i1 %cmp.i18.i1986, label %list_Nconc.exit1995, label %for.cond.i1992

for.cond.i1992:                                   ; preds = %if.end.i1987, %for.cond.i1992
  %List1.addr.0.i1988 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i1990, %for.cond.i1992 ], [ %332, %if.end.i1987 ]
  %List1.addr.0.idx15.i1989 = getelementptr %struct.LIST_HELP* %List1.addr.0.i1988, i32 0, i32 0
  %List1.addr.0.idx15.val.i1990 = load %struct.LIST_HELP** %List1.addr.0.idx15.i1989, align 4
  %cmp.i16.i1991 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i1990, null
  br i1 %cmp.i16.i1991, label %for.end.i1993, label %for.cond.i1992

for.end.i1993:                                    ; preds = %for.cond.i1992
  store %struct.LIST_HELP* %335, %struct.LIST_HELP** %List1.addr.0.idx15.i1989, align 4
  br label %list_Nconc.exit1995

list_Nconc.exit1995:                              ; preds = %sw.bb655, %if.end.i1987, %for.end.i1993
  %retval.0.i1994 = phi %struct.LIST_HELP* [ %332, %for.end.i1993 ], [ %335, %sw.bb655 ], [ %332, %if.end.i1987 ]
  %call661.c = ptrtoint %struct.LIST_HELP* %retval.0.i1994 to i32
  store i32 %call661.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb663:                                         ; preds = %yyreduce
  %337 = load i32* @dfg_IGNORE, align 4
  %tobool664 = icmp eq i32 %337, 0
  br i1 %tobool664, label %cond.false666, label %cond.end671

cond.false666:                                    ; preds = %sw.bb663
  %string668 = bitcast %union.yystype* %yyvsp.2 to i8**
  %338 = load i8** %string668, align 4
  %call.i1996 = call i32 @list_Length(%struct.LIST_HELP* null) #1
  %call1.i1997 = call fastcc i32 @dfg_Symbol(i8* %338, i32 %call.i1996) #1
  %tobool.i.i1998 = icmp sgt i32 %call1.i1997, -1
  br i1 %tobool.i.i1998, label %if.then.i2002, label %land.rhs.i.i2001

land.rhs.i.i2001:                                 ; preds = %cond.false666
  %sub.i.i.i1999 = sub nsw i32 0, %call1.i1997
  %and.i.i.i2000 = and i32 %3, %sub.i.i.i1999
  %cmp.i10.i = icmp eq i32 %and.i.i.i2000, 2
  br i1 %cmp.i10.i, label %dfg_AtomCreate.exit, label %if.then.i2002

if.then.i2002:                                    ; preds = %land.rhs.i.i2001, %cond.false666
  %339 = load %struct._IO_FILE** @stdout, align 4
  %call5.i = call i32 @fflush(%struct._IO_FILE* %339) #1
  %340 = load i32* @dfg_LINENUMBER, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([39 x i8]* @.str4, i32 0, i32 0), i32 %340) #1
  call fastcc void @misc_Error() #1
  unreachable

dfg_AtomCreate.exit:                              ; preds = %land.rhs.i.i2001
  %call6.i = call %struct.term* @term_Create(i32 %call1.i1997, %struct.LIST_HELP* null) #1
  %phitmp1733 = ptrtoint %struct.term* %call6.i to i32
  br label %cond.end671

cond.end671:                                      ; preds = %sw.bb663, %dfg_AtomCreate.exit
  %cond672 = phi i32 [ %phitmp1733, %dfg_AtomCreate.exit ], [ 0, %sw.bb663 ]
  store i32 %cond672, i32* %2, align 4
  br label %sw.epilog1200

sw.bb674:                                         ; preds = %yyreduce
  %341 = load i32* @dfg_IGNORE, align 4
  %tobool675 = icmp eq i32 %341, 0
  br i1 %tobool675, label %cond.false677, label %cond.end681

cond.false677:                                    ; preds = %sw.bb674
  %342 = load i32* @fol_TRUE, align 4
  %call680 = call %struct.term* @term_Create(i32 %342, %struct.LIST_HELP* null) #1
  %phitmp1732 = ptrtoint %struct.term* %call680 to i32
  br label %cond.end681

cond.end681:                                      ; preds = %sw.bb674, %cond.false677
  %cond682 = phi i32 [ %phitmp1732, %cond.false677 ], [ 0, %sw.bb674 ]
  store i32 %cond682, i32* %2, align 4
  br label %sw.epilog1200

sw.bb684:                                         ; preds = %yyreduce
  %343 = load i32* @dfg_IGNORE, align 4
  %tobool685 = icmp eq i32 %343, 0
  br i1 %tobool685, label %cond.false687, label %cond.end691

cond.false687:                                    ; preds = %sw.bb684
  %344 = load i32* @fol_FALSE, align 4
  %call690 = call %struct.term* @term_Create(i32 %344, %struct.LIST_HELP* null) #1
  %phitmp1731 = ptrtoint %struct.term* %call690 to i32
  br label %cond.end691

cond.end691:                                      ; preds = %sw.bb684, %cond.false687
  %cond692 = phi i32 [ %phitmp1731, %cond.false687 ], [ 0, %sw.bb684 ]
  store i32 %cond692, i32* %2, align 4
  br label %sw.epilog1200

sw.bb694:                                         ; preds = %yyreduce
  %345 = load i32* @dfg_IGNORE, align 4
  %tobool695 = icmp eq i32 %345, 0
  br i1 %tobool695, label %cond.false697, label %cond.end706

cond.false697:                                    ; preds = %sw.bb694
  %346 = load i32* @fol_EQUALITY, align 4
  %arrayidx699 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -3
  %term700 = bitcast %union.yystype* %arrayidx699 to %struct.term**
  %347 = load %struct.term** %term700, align 4
  %348 = bitcast %struct.term* %347 to i8*
  %arrayidx701 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %term702 = bitcast %union.yystype* %arrayidx701 to %struct.term**
  %349 = load %struct.term** %term702, align 4
  %350 = bitcast %struct.term* %349 to i8*
  %call.i.i2004 = call i8* @memory_Malloc(i32 8) #1
  %351 = bitcast i8* %call.i.i2004 to %struct.LIST_HELP*
  %car.i.i2005 = getelementptr inbounds i8* %call.i.i2004, i32 4
  %352 = bitcast i8* %car.i.i2005 to i8**
  store i8* %350, i8** %352, align 4
  %cdr.i.i2006 = bitcast i8* %call.i.i2004 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i2006, align 4
  %call.i2007 = call i8* @memory_Malloc(i32 8) #1
  %353 = bitcast i8* %call.i2007 to %struct.LIST_HELP*
  %car.i2008 = getelementptr inbounds i8* %call.i2007, i32 4
  %354 = bitcast i8* %car.i2008 to i8**
  store i8* %348, i8** %354, align 4
  %cdr.i2009 = bitcast i8* %call.i2007 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %351, %struct.LIST_HELP** %cdr.i2009, align 4
  %call705 = call %struct.term* @term_Create(i32 %346, %struct.LIST_HELP* %353) #1
  %phitmp1730 = ptrtoint %struct.term* %call705 to i32
  br label %cond.end706

cond.end706:                                      ; preds = %sw.bb694, %cond.false697
  %cond707 = phi i32 [ %phitmp1730, %cond.false697 ], [ 0, %sw.bb694 ]
  store i32 %cond707, i32* %2, align 4
  br label %sw.epilog1200

sw.bb709:                                         ; preds = %yyreduce
  %355 = load i32* @dfg_IGNORE, align 4
  %tobool710 = icmp eq i32 %355, 0
  br i1 %tobool710, label %cond.false712, label %cond.end718

cond.false712:                                    ; preds = %sw.bb709
  %arrayidx713 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -3
  %string714 = bitcast %union.yystype* %arrayidx713 to i8**
  %356 = load i8** %string714, align 4
  %arrayidx715 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %list716 = bitcast %union.yystype* %arrayidx715 to %struct.LIST_HELP**
  %357 = load %struct.LIST_HELP** %list716, align 4
  %call.i2010 = call i32 @list_Length(%struct.LIST_HELP* %357) #1
  %call1.i2011 = call fastcc i32 @dfg_Symbol(i8* %356, i32 %call.i2010) #1
  %tobool.i.i2012 = icmp sgt i32 %call1.i2011, -1
  br i1 %tobool.i.i2012, label %if.then.i2018, label %land.rhs.i.i2016

land.rhs.i.i2016:                                 ; preds = %cond.false712
  %sub.i.i.i2013 = sub nsw i32 0, %call1.i2011
  %and.i.i.i2014 = and i32 %3, %sub.i.i.i2013
  %cmp.i10.i2015 = icmp eq i32 %and.i.i.i2014, 2
  br i1 %cmp.i10.i2015, label %dfg_AtomCreate.exit2021, label %if.then.i2018

if.then.i2018:                                    ; preds = %land.rhs.i.i2016, %cond.false712
  %358 = load %struct._IO_FILE** @stdout, align 4
  %call5.i2017 = call i32 @fflush(%struct._IO_FILE* %358) #1
  %359 = load i32* @dfg_LINENUMBER, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([39 x i8]* @.str4, i32 0, i32 0), i32 %359) #1
  call fastcc void @misc_Error() #1
  unreachable

dfg_AtomCreate.exit2021:                          ; preds = %land.rhs.i.i2016
  %call6.i2019 = call %struct.term* @term_Create(i32 %call1.i2011, %struct.LIST_HELP* %357) #1
  %phitmp1729 = ptrtoint %struct.term* %call6.i2019 to i32
  br label %cond.end718

cond.end718:                                      ; preds = %sw.bb709, %dfg_AtomCreate.exit2021
  %cond719 = phi i32 [ %phitmp1729, %dfg_AtomCreate.exit2021 ], [ 0, %sw.bb709 ]
  store i32 %cond719, i32* %2, align 4
  br label %sw.epilog1200

sw.bb721:                                         ; preds = %yyreduce
  %360 = load i32* @dfg_IGNORE, align 4
  %tobool722 = icmp eq i32 %360, 0
  br i1 %tobool722, label %cond.false724, label %cond.end729

cond.false724:                                    ; preds = %sw.bb721
  %string726 = bitcast %union.yystype* %yyvsp.2 to i8**
  %361 = load i8** %string726, align 4
  %call.i2022 = call i32 @list_Length(%struct.LIST_HELP* null) #1
  %call1.i2023 = call fastcc i32 @dfg_Symbol(i8* %361, i32 %call.i2022) #1
  %cmp.i.i2024 = icmp sgt i32 %call1.i2023, 0
  br i1 %cmp.i.i2024, label %dfg_TermCreate.exit, label %land.lhs.true.i

land.lhs.true.i:                                  ; preds = %cond.false724
  %tobool.i.i2025 = icmp sgt i32 %call1.i2023, -1
  br i1 %tobool.i.i2025, label %if.then.i2030, label %land.rhs.i.i2028

land.rhs.i.i2028:                                 ; preds = %land.lhs.true.i
  %sub.i6.i.i2026 = sub nsw i32 0, %call1.i2023
  %and.i7.i.i2027 = and i32 %3, %sub.i6.i.i2026
  %362 = icmp ult i32 %and.i7.i.i2027, 2
  br i1 %362, label %dfg_TermCreate.exit, label %if.then.i2030

if.then.i2030:                                    ; preds = %land.rhs.i.i2028, %land.lhs.true.i
  %363 = load %struct._IO_FILE** @stdout, align 4
  %call5.i2029 = call i32 @fflush(%struct._IO_FILE* %363) #1
  %364 = load i32* @dfg_LINENUMBER, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([31 x i8]* @.str233, i32 0, i32 0), i32 %364) #1
  call fastcc void @misc_Error() #1
  unreachable

dfg_TermCreate.exit:                              ; preds = %cond.false724, %land.rhs.i.i2028
  %call6.i2031 = call %struct.term* @term_Create(i32 %call1.i2023, %struct.LIST_HELP* null) #1
  %phitmp1728 = ptrtoint %struct.term* %call6.i2031 to i32
  br label %cond.end729

cond.end729:                                      ; preds = %sw.bb721, %dfg_TermCreate.exit
  %cond730 = phi i32 [ %phitmp1728, %dfg_TermCreate.exit ], [ 0, %sw.bb721 ]
  store i32 %cond730, i32* %2, align 4
  br label %sw.epilog1200

sw.bb732:                                         ; preds = %yyreduce
  %365 = load i32* @dfg_IGNORE, align 4
  %tobool733 = icmp eq i32 %365, 0
  br i1 %tobool733, label %cond.false735, label %cond.end741

cond.false735:                                    ; preds = %sw.bb732
  %arrayidx736 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -3
  %string737 = bitcast %union.yystype* %arrayidx736 to i8**
  %366 = load i8** %string737, align 4
  %arrayidx738 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %list739 = bitcast %union.yystype* %arrayidx738 to %struct.LIST_HELP**
  %367 = load %struct.LIST_HELP** %list739, align 4
  %call.i2033 = call i32 @list_Length(%struct.LIST_HELP* %367) #1
  %call1.i2034 = call fastcc i32 @dfg_Symbol(i8* %366, i32 %call.i2033) #1
  %cmp.i.i2035 = icmp sgt i32 %call1.i2034, 0
  br i1 %cmp.i.i2035, label %dfg_TermCreate.exit2045, label %land.lhs.true.i2037

land.lhs.true.i2037:                              ; preds = %cond.false735
  %tobool.i.i2036 = icmp sgt i32 %call1.i2034, -1
  br i1 %tobool.i.i2036, label %if.then.i2042, label %land.rhs.i.i2040

land.rhs.i.i2040:                                 ; preds = %land.lhs.true.i2037
  %sub.i6.i.i2038 = sub nsw i32 0, %call1.i2034
  %and.i7.i.i2039 = and i32 %3, %sub.i6.i.i2038
  %368 = icmp ult i32 %and.i7.i.i2039, 2
  br i1 %368, label %dfg_TermCreate.exit2045, label %if.then.i2042

if.then.i2042:                                    ; preds = %land.rhs.i.i2040, %land.lhs.true.i2037
  %369 = load %struct._IO_FILE** @stdout, align 4
  %call5.i2041 = call i32 @fflush(%struct._IO_FILE* %369) #1
  %370 = load i32* @dfg_LINENUMBER, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([31 x i8]* @.str233, i32 0, i32 0), i32 %370) #1
  call fastcc void @misc_Error() #1
  unreachable

dfg_TermCreate.exit2045:                          ; preds = %cond.false735, %land.rhs.i.i2040
  %call6.i2043 = call %struct.term* @term_Create(i32 %call1.i2034, %struct.LIST_HELP* %367) #1
  %phitmp1727 = ptrtoint %struct.term* %call6.i2043 to i32
  br label %cond.end741

cond.end741:                                      ; preds = %sw.bb732, %dfg_TermCreate.exit2045
  %cond742 = phi i32 [ %phitmp1727, %dfg_TermCreate.exit2045 ], [ 0, %sw.bb732 ]
  store i32 %cond742, i32* %2, align 4
  br label %sw.epilog1200

sw.bb744:                                         ; preds = %yyreduce
  %371 = load i32* @dfg_IGNORE, align 4
  %tobool745 = icmp eq i32 %371, 0
  br i1 %tobool745, label %cond.false748, label %cond.end752

cond.false748:                                    ; preds = %sw.bb744
  %term750 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %372 = load %struct.term** %term750, align 4
  %373 = bitcast %struct.term* %372 to i8*
  %call.i.i2046 = call i8* @memory_Malloc(i32 8) #1
  %car.i.i2047 = getelementptr inbounds i8* %call.i.i2046, i32 4
  %374 = bitcast i8* %car.i.i2047 to i8**
  store i8* %373, i8** %374, align 4
  %cdr.i.i2048 = bitcast i8* %call.i.i2046 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i2048, align 4
  %phitmp1726 = ptrtoint i8* %call.i.i2046 to i32
  br label %cond.end752

cond.end752:                                      ; preds = %sw.bb744, %cond.false748
  %cond753 = phi i32 [ %phitmp1726, %cond.false748 ], [ 0, %sw.bb744 ]
  store i32 %cond753, i32* %2, align 4
  br label %sw.epilog1200

sw.bb755:                                         ; preds = %yyreduce
  %375 = load i32* @dfg_IGNORE, align 4
  %tobool756 = icmp eq i32 %375, 0
  %arrayidx758 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %list759 = bitcast %union.yystype* %arrayidx758 to %struct.LIST_HELP**
  %376 = load %struct.LIST_HELP** %list759, align 4
  br i1 %tobool756, label %cond.false760, label %cond.end767

cond.false760:                                    ; preds = %sw.bb755
  %term764 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %377 = load %struct.term** %term764, align 4
  %378 = bitcast %struct.term* %377 to i8*
  %call.i.i2049 = call i8* @memory_Malloc(i32 8) #1
  %379 = bitcast i8* %call.i.i2049 to %struct.LIST_HELP*
  %car.i.i2050 = getelementptr inbounds i8* %call.i.i2049, i32 4
  %380 = bitcast i8* %car.i.i2050 to i8**
  store i8* %378, i8** %380, align 4
  %cdr.i.i2051 = bitcast i8* %call.i.i2049 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i2051, align 4
  %cmp.i.i2052 = icmp eq %struct.LIST_HELP* %376, null
  br i1 %cmp.i.i2052, label %cond.end767, label %if.end.i2054

if.end.i2054:                                     ; preds = %cond.false760
  %cmp.i18.i2053 = icmp eq i8* %call.i.i2049, null
  br i1 %cmp.i18.i2053, label %cond.end767, label %for.cond.i2059

for.cond.i2059:                                   ; preds = %if.end.i2054, %for.cond.i2059
  %List1.addr.0.i2055 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i2057, %for.cond.i2059 ], [ %376, %if.end.i2054 ]
  %List1.addr.0.idx15.i2056 = getelementptr %struct.LIST_HELP* %List1.addr.0.i2055, i32 0, i32 0
  %List1.addr.0.idx15.val.i2057 = load %struct.LIST_HELP** %List1.addr.0.idx15.i2056, align 4
  %cmp.i16.i2058 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i2057, null
  br i1 %cmp.i16.i2058, label %for.end.i2060, label %for.cond.i2059

for.end.i2060:                                    ; preds = %for.cond.i2059
  store %struct.LIST_HELP* %379, %struct.LIST_HELP** %List1.addr.0.idx15.i2056, align 4
  br label %cond.end767

cond.end767:                                      ; preds = %for.end.i2060, %if.end.i2054, %cond.false760, %sw.bb755
  %cond768 = phi %struct.LIST_HELP* [ %376, %sw.bb755 ], [ %376, %for.end.i2060 ], [ %379, %cond.false760 ], [ %376, %if.end.i2054 ]
  %cond768.c = ptrtoint %struct.LIST_HELP* %cond768 to i32
  store i32 %cond768.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb770:                                         ; preds = %yyreduce
  %arrayidx771 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %string772 = bitcast %union.yystype* %arrayidx771 to i8**
  %381 = load i8** %string772, align 4
  %call.i2063 = call i32 @strcmp(i8* %381, i8* getelementptr inbounds ([6 x i8]* @.str5, i32 0, i32 0)) #1
  %cmp.i2064 = icmp eq i32 %call.i2063, 0
  br i1 %cmp.i2064, label %sw.epilog1200, label %if.then775

if.then775:                                       ; preds = %sw.bb770
  %382 = load i32* @dfg_IGNORE, align 4
  %383 = inttoptr i32 %382 to i8*
  %384 = load i32* @stack_POINTER, align 4
  %inc.i2066 = add i32 %384, 1
  store i32 %inc.i2066, i32* @stack_POINTER, align 4
  %arrayidx.i2067 = getelementptr inbounds [10000 x i8*]* @stack_STACK, i32 0, i32 %384
  store i8* %383, i8** %arrayidx.i2067, align 4
  store i32 1, i32* @dfg_IGNORE, align 4
  br label %sw.epilog1200

sw.bb777:                                         ; preds = %yyreduce
  %arrayidx778 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -6
  %string779 = bitcast %union.yystype* %arrayidx778 to i8**
  %385 = load i8** %string779, align 4
  %call.i2068 = call i32 @strcmp(i8* %385, i8* getelementptr inbounds ([6 x i8]* @.str5, i32 0, i32 0)) #1
  %cmp.i2069 = icmp eq i32 %call.i2068, 0
  br i1 %cmp.i2069, label %if.end784, label %if.then782

if.then782:                                       ; preds = %sw.bb777
  %386 = load i32* @stack_POINTER, align 4
  %dec.i2071 = add i32 %386, -1
  store i32 %dec.i2071, i32* @stack_POINTER, align 4
  %arrayidx.i2072 = getelementptr inbounds [10000 x i8*]* @stack_STACK, i32 0, i32 %dec.i2071
  %387 = load i8** %arrayidx.i2072, align 4
  %388 = ptrtoint i8* %387 to i32
  store i32 %388, i32* @dfg_IGNORE, align 4
  %.pre = load i8** %string779, align 4
  br label %if.end784

if.end784:                                        ; preds = %sw.bb777, %if.then782
  %389 = phi i8* [ %385, %sw.bb777 ], [ %.pre, %if.then782 ]
  call void @string_StringFree(i8* %389) #1
  br label %sw.epilog1200

sw.bb787:                                         ; preds = %yyreduce
  %390 = load i32* @dfg_IGNORE, align 4
  %tobool788 = icmp eq i32 %390, 0
  %arrayidx789 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -11
  %string790 = bitcast %union.yystype* %arrayidx789 to i8**
  %391 = load i8** %string790, align 4
  br i1 %tobool788, label %land.lhs.true, label %if.else823

land.lhs.true:                                    ; preds = %sw.bb787
  %cmp791 = icmp eq i8* %391, null
  br i1 %cmp791, label %if.end831, label %land.lhs.true793

land.lhs.true793:                                 ; preds = %land.lhs.true
  %arrayidx794 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -9
  %term795 = bitcast %union.yystype* %arrayidx794 to %struct.term**
  %392 = load %struct.term** %term795, align 4
  %cmp796 = icmp eq %struct.term* %392, null
  br i1 %cmp796, label %if.then828, label %land.lhs.true798

land.lhs.true798:                                 ; preds = %land.lhs.true793
  %arrayidx799 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -4
  %list800 = bitcast %union.yystype* %arrayidx799 to %struct.LIST_HELP**
  %393 = load %struct.LIST_HELP** %list800, align 4
  %cmp.i2073 = icmp eq %struct.LIST_HELP* %393, null
  br i1 %cmp.i2073, label %if.else823, label %if.then803

if.then803:                                       ; preds = %land.lhs.true798
  %arrayidx804 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -7
  %string805 = bitcast %union.yystype* %arrayidx804 to i8**
  %394 = load i8** %string805, align 4
  %call806 = call i32 @clause_GetOriginFromString(i8* %394) #1
  %395 = load i8** %string805, align 4
  call void @string_StringFree(i8* %395) #1
  %number810 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2, i32 0
  %396 = load i32* %number810, align 4
  %397 = inttoptr i32 %396 to i8*
  %398 = inttoptr i32 %call806 to i8*
  %call.i.i2075 = call i8* @memory_Malloc(i32 8) #1
  %399 = bitcast i8* %call.i.i2075 to %struct.LIST_HELP*
  %car.i.i2076 = getelementptr inbounds i8* %call.i.i2075, i32 4
  %400 = bitcast i8* %car.i.i2076 to i8**
  store i8* %398, i8** %400, align 4
  %cdr.i.i2077 = bitcast i8* %call.i.i2075 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i2077, align 4
  %call.i2078 = call i8* @memory_Malloc(i32 8) #1
  %401 = bitcast i8* %call.i2078 to %struct.LIST_HELP*
  %car.i2079 = getelementptr inbounds i8* %call.i2078, i32 4
  %402 = bitcast i8* %car.i2079 to i8**
  store i8* %397, i8** %402, align 4
  %cdr.i2080 = bitcast i8* %call.i2078 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %399, %struct.LIST_HELP** %cdr.i2080, align 4
  %403 = load i8** %string790, align 4
  %404 = load %struct.term** %term795, align 4
  %405 = bitcast %struct.term* %404 to i8*
  %406 = load %struct.LIST_HELP** %list800, align 4
  %407 = bitcast %struct.LIST_HELP* %406 to i8*
  %call.i2081 = call i8* @memory_Malloc(i32 8) #1
  %408 = bitcast i8* %call.i2081 to %struct.LIST_HELP*
  %car.i2082 = getelementptr inbounds i8* %call.i2081, i32 4
  %409 = bitcast i8* %car.i2082 to i8**
  store i8* %407, i8** %409, align 4
  %cdr.i2083 = bitcast i8* %call.i2081 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %401, %struct.LIST_HELP** %cdr.i2083, align 4
  %call.i2084 = call i8* @memory_Malloc(i32 8) #1
  %410 = bitcast i8* %call.i2084 to %struct.LIST_HELP*
  %car.i2085 = getelementptr inbounds i8* %call.i2084, i32 4
  %411 = bitcast i8* %car.i2085 to i8**
  store i8* %405, i8** %411, align 4
  %cdr.i2086 = bitcast i8* %call.i2084 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %408, %struct.LIST_HELP** %cdr.i2086, align 4
  %call.i2087 = call i8* @memory_Malloc(i32 8) #1
  %car.i2088 = getelementptr inbounds i8* %call.i2087, i32 4
  %412 = bitcast i8* %car.i2088 to i8**
  store i8* %403, i8** %412, align 4
  %cdr.i2089 = bitcast i8* %call.i2087 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %410, %struct.LIST_HELP** %cdr.i2089, align 4
  %413 = load %struct.LIST_HELP** @dfg_PROOFLIST, align 4
  %call.i2090 = call i8* @memory_Malloc(i32 8) #1
  %414 = bitcast i8* %call.i2090 to %struct.LIST_HELP*
  %car.i2091 = getelementptr inbounds i8* %call.i2090, i32 4
  %415 = bitcast i8* %car.i2091 to i8**
  store i8* %call.i2087, i8** %415, align 4
  %cdr.i2092 = bitcast i8* %call.i2090 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %413, %struct.LIST_HELP** %cdr.i2092, align 4
  store %struct.LIST_HELP* %414, %struct.LIST_HELP** @dfg_PROOFLIST, align 4
  br label %if.end850

if.else823:                                       ; preds = %sw.bb787, %land.lhs.true798
  %cmp826 = icmp eq i8* %391, null
  br i1 %cmp826, label %if.end831, label %if.then828

if.then828:                                       ; preds = %land.lhs.true793, %if.else823
  call void @string_StringFree(i8* %391) #1
  br label %if.end831

if.end831:                                        ; preds = %land.lhs.true, %if.else823, %if.then828
  %arrayidx832 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -9
  %term833 = bitcast %union.yystype* %arrayidx832 to %struct.term**
  %416 = load %struct.term** %term833, align 4
  %cmp834 = icmp eq %struct.term* %416, null
  br i1 %cmp834, label %if.end839, label %if.then836

if.then836:                                       ; preds = %if.end831
  call void @term_Delete(%struct.term* %416) #1
  br label %if.end839

if.end839:                                        ; preds = %if.end831, %if.then836
  %arrayidx840 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -7
  %string841 = bitcast %union.yystype* %arrayidx840 to i8**
  %417 = load i8** %string841, align 4
  %cmp842 = icmp eq i8* %417, null
  br i1 %cmp842, label %if.end847, label %if.then844

if.then844:                                       ; preds = %if.end839
  call void @string_StringFree(i8* %417) #1
  br label %if.end847

if.end847:                                        ; preds = %if.end839, %if.then844
  %arrayidx848 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -4
  %list849 = bitcast %union.yystype* %arrayidx848 to %struct.LIST_HELP**
  %418 = load %struct.LIST_HELP** %list849, align 4
  call void @list_DeleteWithElement(%struct.LIST_HELP* %418, void (i8*)* @string_StringFree) #1
  br label %if.end850

if.end850:                                        ; preds = %if.end847, %if.then803
  %419 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %cmp.i.i2093 = icmp eq %struct.LIST_HELP* %419, null
  br i1 %cmp.i.i2093, label %dfg_VarCheck.exit2098, label %if.then.i2096

if.then.i2096:                                    ; preds = %if.end850
  %420 = load %struct._IO_FILE** @stdout, align 4
  %call1.i2094 = call i32 @fflush(%struct._IO_FILE* %420) #1
  %421 = load %struct._IO_FILE** @stderr, align 4
  %call2.i2095 = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %421, i8* getelementptr inbounds ([31 x i8]* @.str27, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str28, i32 0, i32 0), i32 1881) #1
  call void (i8*, ...)* @misc_ErrorReport(i8* getelementptr inbounds ([55 x i8]* @.str41, i32 0, i32 0)) #1
  %422 = load %struct._IO_FILE** @stderr, align 4
  %423 = call i32 @fwrite(i8* getelementptr inbounds ([133 x i8]* @.str30, i32 0, i32 0), i32 132, i32 1, %struct._IO_FILE* %422) #1
  call fastcc void @misc_DumpCore() #1
  unreachable

dfg_VarCheck.exit2098:                            ; preds = %if.end850
  store i32 0, i32* @symbol_STANDARDVARCOUNTER, align 4
  br label %sw.epilog1200

sw.bb851:                                         ; preds = %yyreduce
  %424 = load i32* @dfg_IGNORE, align 4
  %tobool852 = icmp eq i32 %424, 0
  br i1 %tobool852, label %lor.lhs.false853, label %cond.end864

lor.lhs.false853:                                 ; preds = %sw.bb851
  %string855 = bitcast %union.yystype* %yyvsp.2 to i8**
  %425 = load i8** %string855, align 4
  %cmp856 = icmp eq i8* %425, null
  br i1 %cmp856, label %cond.end864, label %cond.false860

cond.false860:                                    ; preds = %lor.lhs.false853
  %call.i.i2099 = call i8* @memory_Malloc(i32 8) #1
  %car.i.i2100 = getelementptr inbounds i8* %call.i.i2099, i32 4
  %426 = bitcast i8* %car.i.i2100 to i8**
  store i8* %425, i8** %426, align 4
  %cdr.i.i2101 = bitcast i8* %call.i.i2099 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i2101, align 4
  %phitmp = ptrtoint i8* %call.i.i2099 to i32
  br label %cond.end864

cond.end864:                                      ; preds = %lor.lhs.false853, %sw.bb851, %cond.false860
  %cond865 = phi i32 [ %phitmp, %cond.false860 ], [ 0, %sw.bb851 ], [ 0, %lor.lhs.false853 ]
  store i32 %cond865, i32* %2, align 4
  br label %sw.epilog1200

sw.bb867:                                         ; preds = %yyreduce
  %427 = load i32* @dfg_IGNORE, align 4
  %tobool868 = icmp eq i32 %427, 0
  br i1 %tobool868, label %lor.lhs.false869, label %cond.true874

lor.lhs.false869:                                 ; preds = %sw.bb867
  %string871 = bitcast %union.yystype* %yyvsp.2 to i8**
  %428 = load i8** %string871, align 4
  %cmp872 = icmp eq i8* %428, null
  br i1 %cmp872, label %cond.true874, label %cond.false877

cond.true874:                                     ; preds = %sw.bb867, %lor.lhs.false869
  %arrayidx875 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %list876 = bitcast %union.yystype* %arrayidx875 to %struct.LIST_HELP**
  %429 = load %struct.LIST_HELP** %list876, align 4
  br label %cond.end884

cond.false877:                                    ; preds = %lor.lhs.false869
  %arrayidx878 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %list879 = bitcast %union.yystype* %arrayidx878 to %struct.LIST_HELP**
  %430 = load %struct.LIST_HELP** %list879, align 4
  %call.i.i2102 = call i8* @memory_Malloc(i32 8) #1
  %431 = bitcast i8* %call.i.i2102 to %struct.LIST_HELP*
  %car.i.i2103 = getelementptr inbounds i8* %call.i.i2102, i32 4
  %432 = bitcast i8* %car.i.i2103 to i8**
  store i8* %428, i8** %432, align 4
  %cdr.i.i2104 = bitcast i8* %call.i.i2102 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i2104, align 4
  %cmp.i.i2105 = icmp eq %struct.LIST_HELP* %430, null
  br i1 %cmp.i.i2105, label %cond.end884, label %if.end.i2107

if.end.i2107:                                     ; preds = %cond.false877
  %cmp.i18.i2106 = icmp eq i8* %call.i.i2102, null
  br i1 %cmp.i18.i2106, label %cond.end884, label %for.cond.i2112

for.cond.i2112:                                   ; preds = %if.end.i2107, %for.cond.i2112
  %List1.addr.0.i2108 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i2110, %for.cond.i2112 ], [ %430, %if.end.i2107 ]
  %List1.addr.0.idx15.i2109 = getelementptr %struct.LIST_HELP* %List1.addr.0.i2108, i32 0, i32 0
  %List1.addr.0.idx15.val.i2110 = load %struct.LIST_HELP** %List1.addr.0.idx15.i2109, align 4
  %cmp.i16.i2111 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i2110, null
  br i1 %cmp.i16.i2111, label %for.end.i2113, label %for.cond.i2112

for.end.i2113:                                    ; preds = %for.cond.i2112
  store %struct.LIST_HELP* %431, %struct.LIST_HELP** %List1.addr.0.idx15.i2109, align 4
  br label %cond.end884

cond.end884:                                      ; preds = %for.end.i2113, %if.end.i2107, %cond.false877, %cond.true874
  %cond885 = phi %struct.LIST_HELP* [ %429, %cond.true874 ], [ %430, %for.end.i2113 ], [ %431, %cond.false877 ], [ %430, %if.end.i2107 ]
  %cond885.c = ptrtoint %struct.LIST_HELP* %cond885 to i32
  store i32 %cond885.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb887:                                         ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb889:                                         ; preds = %yyreduce
  %number891 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1, i32 0
  %433 = load i32* %number891, align 4
  store i32 %433, i32* %2, align 4
  br label %sw.epilog1200

sw.bb893:                                         ; preds = %yyreduce
  %434 = load i32* @dfg_IGNORE, align 4
  %tobool894 = icmp eq i32 %434, 0
  %arrayidx896 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %string897 = bitcast %union.yystype* %arrayidx896 to i8**
  br i1 %tobool894, label %land.lhs.true895, label %if.else915

land.lhs.true895:                                 ; preds = %sw.bb893
  %435 = load i8** %string897, align 4
  %cmp898 = icmp eq i8* %435, null
  br i1 %cmp898, label %if.else915, label %land.lhs.true900

land.lhs.true900:                                 ; preds = %land.lhs.true895
  %string902 = bitcast %union.yystype* %yyvsp.2 to i8**
  %436 = load i8** %string902, align 4
  %cmp903 = icmp eq i8* %436, null
  br i1 %cmp903, label %if.else915, label %land.lhs.true905

land.lhs.true905:                                 ; preds = %land.lhs.true900
  %call.i2116 = call i32 @strcmp(i8* %435, i8* getelementptr inbounds ([11 x i8]* @.str6, i32 0, i32 0)) #1
  %cmp.i2117 = icmp eq i32 %call.i2116, 0
  br i1 %cmp.i2117, label %if.then910, label %if.else915

if.then910:                                       ; preds = %land.lhs.true905
  %call914 = call i32 @string_StringToInt(i8* %436, i32 1, i32* %2) #1
  br label %if.end917

if.else915:                                       ; preds = %sw.bb893, %land.lhs.true905, %land.lhs.true900, %land.lhs.true895
  store i32 0, i32* %2, align 4
  br label %if.end917

if.end917:                                        ; preds = %if.else915, %if.then910
  %437 = load i8** %string897, align 4
  %cmp920 = icmp eq i8* %437, null
  br i1 %cmp920, label %if.end925, label %if.then922

if.then922:                                       ; preds = %if.end917
  call void @string_StringFree(i8* %437) #1
  br label %if.end925

if.end925:                                        ; preds = %if.end917, %if.then922
  %string927 = bitcast %union.yystype* %yyvsp.2 to i8**
  %438 = load i8** %string927, align 4
  %cmp928 = icmp eq i8* %438, null
  br i1 %cmp928, label %sw.epilog1200, label %if.then930

if.then930:                                       ; preds = %if.end925
  call void @string_StringFree(i8* %438) #1
  br label %sw.epilog1200

sw.bb934:                                         ; preds = %yyreduce
  %439 = load i32* @dfg_IGNORE, align 4
  %tobool935 = icmp eq i32 %439, 0
  %arrayidx937 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %string938 = bitcast %union.yystype* %arrayidx937 to i8**
  br i1 %tobool935, label %land.lhs.true936, label %if.else956

land.lhs.true936:                                 ; preds = %sw.bb934
  %440 = load i8** %string938, align 4
  %cmp939 = icmp eq i8* %440, null
  br i1 %cmp939, label %if.else956, label %land.lhs.true941

land.lhs.true941:                                 ; preds = %land.lhs.true936
  %string943 = bitcast %union.yystype* %yyvsp.2 to i8**
  %441 = load i8** %string943, align 4
  %cmp944 = icmp eq i8* %441, null
  br i1 %cmp944, label %if.else956, label %land.lhs.true946

land.lhs.true946:                                 ; preds = %land.lhs.true941
  %call.i2119 = call i32 @strcmp(i8* %440, i8* getelementptr inbounds ([11 x i8]* @.str6, i32 0, i32 0)) #1
  %cmp.i2120 = icmp eq i32 %call.i2119, 0
  br i1 %cmp.i2120, label %if.then951, label %if.else956

if.then951:                                       ; preds = %land.lhs.true946
  %call955 = call i32 @string_StringToInt(i8* %441, i32 1, i32* %2) #1
  br label %if.end960

if.else956:                                       ; preds = %sw.bb934, %land.lhs.true946, %land.lhs.true941, %land.lhs.true936
  %number958 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -4, i32 0
  %442 = load i32* %number958, align 4
  store i32 %442, i32* %2, align 4
  br label %if.end960

if.end960:                                        ; preds = %if.else956, %if.then951
  %443 = load i8** %string938, align 4
  %cmp963 = icmp eq i8* %443, null
  br i1 %cmp963, label %if.end968, label %if.then965

if.then965:                                       ; preds = %if.end960
  call void @string_StringFree(i8* %443) #1
  br label %if.end968

if.end968:                                        ; preds = %if.end960, %if.then965
  %string970 = bitcast %union.yystype* %yyvsp.2 to i8**
  %444 = load i8** %string970, align 4
  %cmp971 = icmp eq i8* %444, null
  br i1 %cmp971, label %sw.epilog1200, label %if.then973

if.then973:                                       ; preds = %if.end968
  call void @string_StringFree(i8* %444) #1
  br label %sw.epilog1200

sw.bb977:                                         ; preds = %yyreduce
  %445 = load i32* @dfg_IGNORE, align 4
  %446 = inttoptr i32 %445 to i8*
  %447 = load i32* @stack_POINTER, align 4
  %inc.i2122 = add i32 %447, 1
  store i32 %inc.i2122, i32* @stack_POINTER, align 4
  %arrayidx.i2123 = getelementptr inbounds [10000 x i8*]* @stack_STACK, i32 0, i32 %447
  store i8* %446, i8** %arrayidx.i2123, align 4
  store i32 1, i32* @dfg_IGNORE, align 4
  br label %sw.epilog1200

sw.bb978:                                         ; preds = %yyreduce
  %448 = load i32* @stack_POINTER, align 4
  %dec.i2124 = add i32 %448, -1
  store i32 %dec.i2124, i32* @stack_POINTER, align 4
  %arrayidx.i2125 = getelementptr inbounds [10000 x i8*]* @stack_STACK, i32 0, i32 %dec.i2124
  %449 = load i8** %arrayidx.i2125, align 4
  %450 = ptrtoint i8* %449 to i32
  store i32 %450, i32* @dfg_IGNORE, align 4
  %bool981 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 0, i32 0
  %451 = load i32* %bool981, align 4
  %tobool982 = icmp eq i32 %451, 0
  %arrayidx984 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %string985 = bitcast %union.yystype* %arrayidx984 to i8**
  %452 = load i8** %string985, align 4
  br i1 %tobool982, label %if.else993, label %if.then983

if.then983:                                       ; preds = %sw.bb978
  %cmp986 = icmp eq i8* %452, null
  br i1 %cmp986, label %if.end991, label %if.then988

if.then988:                                       ; preds = %if.then983
  call void @string_StringFree(i8* %452) #1
  br label %if.end991

if.end991:                                        ; preds = %if.then983, %if.then988
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

if.else993:                                       ; preds = %sw.bb978
  %.c1725 = ptrtoint i8* %452 to i32
  store i32 %.c1725, i32* %2, align 4
  br label %sw.epilog1200

sw.bb998:                                         ; preds = %yyreduce
  %string1000 = bitcast %union.yystype* %yyvsp.2 to i8**
  %453 = load i8** %string1000, align 4
  %.c1724 = ptrtoint i8* %453 to i32
  store i32 %.c1724, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1002:                                        ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1004:                                        ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1006:                                        ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1008:                                        ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1010:                                        ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1012:                                        ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1014:                                        ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1016:                                        ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1018:                                        ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1020:                                        ; preds = %yyreduce
  store i32 1, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1022:                                        ; preds = %yyreduce
  store i32 1, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1024:                                        ; preds = %yyreduce
  %term1026 = bitcast %union.yystype* %yyvsp.2 to %struct.term**
  %454 = load %struct.term** %term1026, align 4
  %.c = ptrtoint %struct.term* %454 to i32
  store i32 %.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1028:                                        ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1030:                                        ; preds = %yyreduce
  %455 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %call.i.i.i2126 = call i8* @memory_Malloc(i32 8) #1
  %456 = bitcast i8* %call.i.i.i2126 to %struct.LIST_HELP*
  %car.i.i.i2127 = getelementptr inbounds i8* %call.i.i.i2126, i32 4
  %457 = bitcast i8* %car.i.i.i2127 to i8**
  store i8* null, i8** %457, align 4
  %cdr.i.i.i2128 = bitcast i8* %call.i.i.i2126 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %455, %struct.LIST_HELP** %cdr.i.i.i2128, align 4
  store %struct.LIST_HELP* %456, %struct.LIST_HELP** @dfg_VARLIST, align 4
  store i1 true, i1* @dfg_VARDECL, align 1
  br label %sw.epilog1200

sw.bb1031:                                        ; preds = %yyreduce
  store i1 false, i1* @dfg_VARDECL, align 1
  %458 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %.idx.i2129 = getelementptr %struct.LIST_HELP* %458, i32 0, i32 1
  %.idx.val.i2130 = load i8** %.idx.i2129, align 4
  %459 = bitcast i8* %.idx.val.i2130 to %struct.LIST_HELP*
  call void @list_DeleteWithElement(%struct.LIST_HELP* %459, void (i8*)* bitcast (void (%struct.DFG_VARENTRY*)* @dfg_VarFree to void (i8*)*)) #1
  %460 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %L.idx.i.i2131 = getelementptr %struct.LIST_HELP* %460, i32 0, i32 0
  %L.idx.val.i.i2132 = load %struct.LIST_HELP** %L.idx.i.i2131, align 4
  %461 = bitcast %struct.LIST_HELP* %460 to i8*
  %462 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i2133 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %462, i32 0, i32 4
  %463 = load i32* %total_size.i.i.i.i2133, align 4
  %464 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i2134 = add i32 %464, %463
  store i32 %add24.i.i.i.i2134, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i2135 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %462, i32 0, i32 0
  %465 = load i8** %free.i.i.i.i2135, align 4
  %.c.i.i.i2136 = bitcast i8* %465 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i2136, %struct.LIST_HELP** %L.idx.i.i2131, align 4
  %466 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i2137 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %466, i32 0, i32 0
  store i8* %461, i8** %free27.i.i.i.i2137, align 4
  store %struct.LIST_HELP* %L.idx.val.i.i2132, %struct.LIST_HELP** @dfg_VARLIST, align 4
  %cmp.i.i2138 = icmp eq %struct.LIST_HELP* %L.idx.val.i.i2132, null
  br i1 %cmp.i.i2138, label %dfg_VarCheck.exit2143, label %if.then.i2141

if.then.i2141:                                    ; preds = %sw.bb1031
  %467 = load %struct._IO_FILE** @stdout, align 4
  %call1.i2139 = call i32 @fflush(%struct._IO_FILE* %467) #1
  %468 = load %struct._IO_FILE** @stderr, align 4
  %call2.i2140 = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %468, i8* getelementptr inbounds ([31 x i8]* @.str27, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str28, i32 0, i32 0), i32 1881) #1
  call void (i8*, ...)* @misc_ErrorReport(i8* getelementptr inbounds ([55 x i8]* @.str41, i32 0, i32 0)) #1
  %469 = load %struct._IO_FILE** @stderr, align 4
  %470 = call i32 @fwrite(i8* getelementptr inbounds ([133 x i8]* @.str30, i32 0, i32 0), i32 132, i32 1, %struct._IO_FILE* %469) #1
  call fastcc void @misc_DumpCore() #1
  unreachable

dfg_VarCheck.exit2143:                            ; preds = %sw.bb1031
  store i32 0, i32* @symbol_STANDARDVARCOUNTER, align 4
  br label %sw.epilog1200

sw.bb1032:                                        ; preds = %yyreduce
  %471 = load %struct.LIST_HELP** @dfg_TERMLIST, align 4
  %arrayidx1033 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1
  %term1034 = bitcast %union.yystype* %arrayidx1033 to %struct.term**
  %472 = load %struct.term** %term1034, align 4
  %473 = bitcast %struct.term* %472 to i8*
  %call.i.i2144 = call i8* @memory_Malloc(i32 8) #1
  %474 = bitcast i8* %call.i.i2144 to %struct.LIST_HELP*
  %car.i.i2145 = getelementptr inbounds i8* %call.i.i2144, i32 4
  %475 = bitcast i8* %car.i.i2145 to i8**
  store i8* %473, i8** %475, align 4
  %cdr.i.i2146 = bitcast i8* %call.i.i2144 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i2146, align 4
  %cmp.i.i2147 = icmp eq %struct.LIST_HELP* %471, null
  br i1 %cmp.i.i2147, label %list_Nconc.exit2157, label %if.end.i2149

if.end.i2149:                                     ; preds = %sw.bb1032
  %cmp.i18.i2148 = icmp eq i8* %call.i.i2144, null
  br i1 %cmp.i18.i2148, label %list_Nconc.exit2157, label %for.cond.i2154

for.cond.i2154:                                   ; preds = %if.end.i2149, %for.cond.i2154
  %List1.addr.0.i2150 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i2152, %for.cond.i2154 ], [ %471, %if.end.i2149 ]
  %List1.addr.0.idx15.i2151 = getelementptr %struct.LIST_HELP* %List1.addr.0.i2150, i32 0, i32 0
  %List1.addr.0.idx15.val.i2152 = load %struct.LIST_HELP** %List1.addr.0.idx15.i2151, align 4
  %cmp.i16.i2153 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i2152, null
  br i1 %cmp.i16.i2153, label %for.end.i2155, label %for.cond.i2154

for.end.i2155:                                    ; preds = %for.cond.i2154
  store %struct.LIST_HELP* %474, %struct.LIST_HELP** %List1.addr.0.idx15.i2151, align 4
  br label %list_Nconc.exit2157

list_Nconc.exit2157:                              ; preds = %sw.bb1032, %if.end.i2149, %for.end.i2155
  %retval.0.i2156 = phi %struct.LIST_HELP* [ %471, %for.end.i2155 ], [ %474, %sw.bb1032 ], [ %471, %if.end.i2149 ]
  store %struct.LIST_HELP* %retval.0.i2156, %struct.LIST_HELP** @dfg_TERMLIST, align 4
  br label %sw.epilog1200

sw.bb1037:                                        ; preds = %yyreduce
  %string1039 = bitcast %union.yystype* %yyvsp.2 to i8**
  %476 = load i8** %string1039, align 4
  %call.i2158 = call i32 @strcmp(i8* %476, i8* getelementptr inbounds ([6 x i8]* @.str5, i32 0, i32 0)) #1
  %cmp.i2159 = icmp eq i32 %call.i2158, 0
  br i1 %cmp.i2159, label %if.then1042, label %if.end1043

if.then1042:                                      ; preds = %sw.bb1037
  store i32 0, i32* @dfg_IGNORETEXT, align 4
  %.pre2536 = load i8** %string1039, align 4
  br label %if.end1043

if.end1043:                                       ; preds = %sw.bb1037, %if.then1042
  %477 = phi i8* [ %476, %sw.bb1037 ], [ %.pre2536, %if.then1042 ]
  call void @string_StringFree(i8* %477) #1
  br label %sw.epilog1200

sw.bb1046:                                        ; preds = %yyreduce
  store i32 1, i32* @dfg_IGNORETEXT, align 4
  br label %sw.epilog1200

sw.bb1047:                                        ; preds = %yyreduce
  %string1049 = bitcast %union.yystype* %yyvsp.2 to i8**
  %478 = load i8** %string1049, align 4
  call void @string_StringFree(i8* %478) #1
  br label %sw.epilog1200

for.body:                                         ; preds = %for.body.lr.ph, %if.end1075
  %479 = phi %struct.LIST_HELP* [ %25, %for.body.lr.ph ], [ %L.idx.val.i, %if.end1075 ]
  %.idx1760 = getelementptr %struct.LIST_HELP* %479, i32 0, i32 1
  %.idx1760.val = load i8** %.idx1760, align 4
  %call1059 = call i32 @symbol_Lookup(i8* %.idx1760.val) #1
  %cmp1060 = icmp eq i32 %call1059, 0
  br i1 %cmp1060, label %if.then1062, label %if.end1067

if.then1062:                                      ; preds = %for.body
  %480 = load %struct._IO_FILE** @stdout, align 4
  %call1063 = call i32 @fflush(%struct._IO_FILE* %480) #1
  %481 = load %struct.LIST_HELP** %list1053, align 4
  %.idx1759 = getelementptr %struct.LIST_HELP* %481, i32 0, i32 1
  %.idx1759.val = load i8** %.idx1759, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([22 x i8]* @.str7, i32 0, i32 0), i8* %.idx1759.val) #1
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([19 x i8]* @.str8, i32 0, i32 0)) #1
  call fastcc void @misc_Error()
  unreachable

if.end1067:                                       ; preds = %for.body
  %tobool.i2163 = icmp sgt i32 %call1059, -1
  br i1 %tobool.i2163, label %if.then1070, label %land.rhs.i2167

land.rhs.i2167:                                   ; preds = %if.end1067
  %sub.i.i2164 = sub nsw i32 0, %call1059
  %and.i.i2165 = and i32 %3, %sub.i.i2164
  %cmp.i2166 = icmp eq i32 %and.i.i2165, 2
  br i1 %cmp.i2166, label %if.end1075, label %if.then1070

if.then1070:                                      ; preds = %if.end1067, %land.rhs.i2167
  %482 = load %struct._IO_FILE** @stdout, align 4
  %call1071 = call i32 @fflush(%struct._IO_FILE* %482) #1
  %483 = load %struct.LIST_HELP** %list1053, align 4
  %.idx1758 = getelementptr %struct.LIST_HELP* %483, i32 0, i32 1
  %.idx1758.val = load i8** %.idx1758, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([30 x i8]* @.str9, i32 0, i32 0), i8* %.idx1758.val) #1
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([19 x i8]* @.str8, i32 0, i32 0)) #1
  call fastcc void @misc_Error()
  unreachable

if.end1075:                                       ; preds = %land.rhs.i2167
  %484 = load %struct.LIST_HELP** %list1053, align 4
  %.idx = getelementptr %struct.LIST_HELP* %484, i32 0, i32 1
  %.idx.val = load i8** %.idx, align 4
  call void @string_StringFree(i8* %.idx.val) #1
  %shr.i.i = ashr i32 %sub.i.i2164, %4
  %485 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i = getelementptr inbounds %struct.signature** %485, i32 %shr.i.i
  %486 = load %struct.signature** %arrayidx.i.i, align 4
  %props.i = getelementptr inbounds %struct.signature* %486, i32 0, i32 4
  %487 = load i32* %props.i, align 4
  %or.i = or i32 %487, 64
  store i32 %or.i, i32* %props.i, align 4
  %488 = load %struct.LIST_HELP** %list1053, align 4
  %L.idx.i = getelementptr %struct.LIST_HELP* %488, i32 0, i32 0
  %L.idx.val.i = load %struct.LIST_HELP** %L.idx.i, align 4
  %489 = bitcast %struct.LIST_HELP* %488 to i8*
  %490 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %490, i32 0, i32 4
  %491 = load i32* %total_size.i.i.i, align 4
  %492 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i = add i32 %492, %491
  store i32 %add24.i.i.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %490, i32 0, i32 0
  %493 = load i8** %free.i.i.i, align 4
  %.c.i.i = bitcast i8* %493 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i, %struct.LIST_HELP** %L.idx.i, align 4
  %494 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %494, i32 0, i32 0
  store i8* %489, i8** %free27.i.i.i, align 4
  %call1081.c = ptrtoint %struct.LIST_HELP* %L.idx.val.i to i32
  store i32 %call1081.c, i32* %26, align 4
  %cmp.i2161 = icmp eq %struct.LIST_HELP* %L.idx.val.i, null
  br i1 %cmp.i2161, label %sw.epilog1200, label %for.body

sw.bb1084:                                        ; preds = %yyreduce
  %arrayidx1085 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -3
  %string1086 = bitcast %union.yystype* %arrayidx1085 to i8**
  %495 = load i8** %string1086, align 4
  %call1087 = call i32 @flag_Id(i8* %495) #1
  %cmp1088 = icmp eq i32 %call1087, -1
  br i1 %cmp1088, label %if.then1090, label %if.end1094

if.then1090:                                      ; preds = %sw.bb1084
  %496 = load %struct._IO_FILE** @stdout, align 4
  %call1091 = call i32 @fflush(%struct._IO_FILE* %496) #1
  %497 = load i8** %string1086, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([24 x i8]* @.str10, i32 0, i32 0), i8* %497) #1
  call fastcc void @misc_Error()
  unreachable

if.end1094:                                       ; preds = %sw.bb1084
  %498 = load i8** %string1086, align 4
  call void @string_StringFree(i8* %498) #1
  %499 = load i32** @dfg_FLAGS, align 4
  %number1098 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1, i32 0
  %500 = load i32* %number1098, align 4
  %call.i.i2171 = call i32 @flag_Minimum(i32 %call1087) #1
  %cmp.i.i2172 = icmp slt i32 %call.i.i2171, %500
  br i1 %cmp.i.i2172, label %if.else.i.i, label %if.then.i.i2173

if.then.i.i2173:                                  ; preds = %if.end1094
  %501 = load %struct._IO_FILE** @stdout, align 4
  %call1.i.i = call i32 @fflush(%struct._IO_FILE* %501) #1
  %call2.i.i = call i8* @flag_Name(i32 %call1087) #1
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([50 x i8]* @.str231, i32 0, i32 0), i32 %500, i8* %call2.i.i) #1
  call fastcc void @misc_Error() #1
  unreachable

if.else.i.i:                                      ; preds = %if.end1094
  %call3.i.i = call i32 @flag_Maximum(i32 %call1087) #1
  %cmp4.i.i = icmp sgt i32 %call3.i.i, %500
  br i1 %cmp4.i.i, label %flag_SetFlagValue.exit, label %if.then5.i.i

if.then5.i.i:                                     ; preds = %if.else.i.i
  %502 = load %struct._IO_FILE** @stdout, align 4
  %call6.i.i = call i32 @fflush(%struct._IO_FILE* %502) #1
  %call7.i.i = call i8* @flag_Name(i32 %call1087) #1
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([50 x i8]* @.str232, i32 0, i32 0), i32 %500, i8* %call7.i.i) #1
  call fastcc void @misc_Error() #1
  unreachable

flag_SetFlagValue.exit:                           ; preds = %if.else.i.i
  %arrayidx.i2174 = getelementptr inbounds i32* %499, i32 %call1087
  store i32 %500, i32* %arrayidx.i2174, align 4
  br label %sw.epilog1200

sw.bb1099:                                        ; preds = %yyreduce
  %string1102 = bitcast %union.yystype* %yyvsp.2 to i8**
  %503 = load i8** %string1102, align 4
  %call1103 = call i32 @symbol_Lookup(i8* %503) #1
  %cmp1104 = icmp eq i32 %call1103, 0
  br i1 %cmp1104, label %if.then1106, label %if.end1110

if.then1106:                                      ; preds = %sw.bb1099
  %504 = load %struct._IO_FILE** @stdout, align 4
  %call1107 = call i32 @fflush(%struct._IO_FILE* %504) #1
  %505 = load i8** %string1102, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([23 x i8]* @.str11, i32 0, i32 0), i8* %505) #1
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([22 x i8]* @.str12, i32 0, i32 0)) #1
  call fastcc void @misc_Error()
  unreachable

if.end1110:                                       ; preds = %sw.bb1099
  %506 = load i8** %string1102, align 4
  call void @string_StringFree(i8* %506) #1
  %507 = load i32** @dfg_PRECEDENCE, align 4
  %call.i2175 = call i32 @symbol_GetIncreasedOrderingCounter() #1
  %sub.i.i.i2176 = sub nsw i32 0, %call1103
  %shr.i.i.i = ashr i32 %sub.i.i.i2176, %4
  %arrayidx.i.i2177 = getelementptr inbounds i32* %507, i32 %shr.i.i.i
  store i32 %call.i2175, i32* %arrayidx.i.i2177, align 4
  %508 = inttoptr i32 %call1103 to i8*
  %509 = load %struct.LIST_HELP** @dfg_USERPRECEDENCE, align 4
  %call.i2178 = call i8* @memory_Malloc(i32 8) #1
  %510 = bitcast i8* %call.i2178 to %struct.LIST_HELP*
  %car.i2179 = getelementptr inbounds i8* %call.i2178, i32 4
  %511 = bitcast i8* %car.i2179 to i8**
  store i8* %508, i8** %511, align 4
  %cdr.i2180 = bitcast i8* %call.i2178 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %509, %struct.LIST_HELP** %cdr.i2180, align 4
  store %struct.LIST_HELP* %510, %struct.LIST_HELP** @dfg_USERPRECEDENCE, align 4
  br label %sw.epilog1200

sw.bb1114:                                        ; preds = %yyreduce
  %arrayidx1116 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -4
  %string1117 = bitcast %union.yystype* %arrayidx1116 to i8**
  %512 = load i8** %string1117, align 4
  %call1118 = call i32 @symbol_Lookup(i8* %512) #1
  %cmp1119 = icmp eq i32 %call1118, 0
  br i1 %cmp1119, label %if.then1121, label %if.end1125

if.then1121:                                      ; preds = %sw.bb1114
  %513 = load %struct._IO_FILE** @stdout, align 4
  %call1122 = call i32 @fflush(%struct._IO_FILE* %513) #1
  %514 = load i8** %string1117, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([22 x i8]* @.str7, i32 0, i32 0), i8* %514) #1
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([21 x i8]* @.str13, i32 0, i32 0)) #1
  call fastcc void @misc_Error()
  unreachable

if.end1125:                                       ; preds = %sw.bb1114
  %515 = load i8** %string1117, align 4
  call void @string_StringFree(i8* %515) #1
  %516 = load i32** @dfg_PRECEDENCE, align 4
  %call.i2181 = call i32 @symbol_GetIncreasedOrderingCounter() #1
  %sub.i.i.i2182 = sub nsw i32 0, %call1118
  %shr.i.i.i2183 = ashr i32 %sub.i.i.i2182, %4
  %arrayidx.i.i2184 = getelementptr inbounds i32* %516, i32 %shr.i.i.i2183
  store i32 %call.i2181, i32* %arrayidx.i.i2184, align 4
  %517 = inttoptr i32 %call1118 to i8*
  %518 = load %struct.LIST_HELP** @dfg_USERPRECEDENCE, align 4
  %call.i2185 = call i8* @memory_Malloc(i32 8) #1
  %519 = bitcast i8* %call.i2185 to %struct.LIST_HELP*
  %car.i2186 = getelementptr inbounds i8* %call.i2185, i32 4
  %520 = bitcast i8* %car.i2186 to i8**
  store i8* %517, i8** %520, align 4
  %cdr.i2187 = bitcast i8* %call.i2185 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %518, %struct.LIST_HELP** %cdr.i2187, align 4
  store %struct.LIST_HELP* %519, %struct.LIST_HELP** @dfg_USERPRECEDENCE, align 4
  %number1130 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2, i32 0
  %521 = load i32* %number1130, align 4
  %522 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i2190 = getelementptr inbounds %struct.signature** %522, i32 %shr.i.i.i2183
  %523 = load %struct.signature** %arrayidx.i.i2190, align 4
  %weight.i = getelementptr inbounds %struct.signature* %523, i32 0, i32 2
  store i32 %521, i32* %weight.i, align 4
  %property = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -1, i32 0
  %524 = load i32* %property, align 4
  %cmp1132 = icmp eq i32 %524, 0
  br i1 %cmp1132, label %sw.epilog1200, label %if.then1134

if.then1134:                                      ; preds = %if.end1125
  %525 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i2193 = getelementptr inbounds %struct.signature** %525, i32 %shr.i.i.i2183
  %526 = load %struct.signature** %arrayidx.i.i2193, align 4
  %props.i2194 = getelementptr inbounds %struct.signature* %526, i32 0, i32 4
  %527 = load i32* %props.i2194, align 4
  %or.i2195 = or i32 %527, %524
  store i32 %or.i2195, i32* %props.i2194, align 4
  br label %sw.epilog1200

sw.bb1138:                                        ; preds = %yyreduce
  store i32 0, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1140:                                        ; preds = %yyreduce
  %string1142 = bitcast %union.yystype* %yyvsp.2 to i8**
  %528 = load i8** %string1142, align 4
  %arrayidx1143 = getelementptr inbounds i8* %528, i32 1
  %529 = load i8* %arrayidx1143, align 1
  %cmp1145 = icmp eq i8 %529, 0
  br i1 %cmp1145, label %lor.lhs.false1147, label %if.then1168

lor.lhs.false1147:                                ; preds = %sw.bb1140
  %530 = load i8* %528, align 1
  switch i8 %530, label %if.then1168 [
    i8 108, label %if.end1172
    i8 109, label %if.end1172
    i8 114, label %if.end1172
  ]

if.then1168:                                      ; preds = %lor.lhs.false1147, %sw.bb1140
  %531 = load %struct._IO_FILE** @stdout, align 4
  %call1169 = call i32 @fflush(%struct._IO_FILE* %531) #1
  %532 = load i8** %string1142, align 4
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([27 x i8]* @.str14, i32 0, i32 0), i8* %532) #1
  call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([21 x i8]* @.str15, i32 0, i32 0)) #1
  call fastcc void @misc_Error()
  unreachable

if.end1172:                                       ; preds = %lor.lhs.false1147, %lor.lhs.false1147, %lor.lhs.false1147
  %conv1176 = sext i8 %530 to i32
  switch i32 %conv1176, label %sw.default [
    i32 109, label %sw.bb1177
    i32 114, label %sw.bb1179
  ]

sw.bb1177:                                        ; preds = %if.end1172
  store i32 16, i32* %2, align 4
  br label %sw.epilog

sw.bb1179:                                        ; preds = %if.end1172
  store i32 8, i32* %2, align 4
  br label %sw.epilog

sw.default:                                       ; preds = %if.end1172
  store i32 0, i32* %2, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb1179, %sw.bb1177
  %533 = load i8** %string1142, align 4
  call void @string_StringFree(i8* %533) #1
  br label %sw.epilog1200

sw.bb1184:                                        ; preds = %yyreduce
  %arrayidx1185 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %list1186 = bitcast %union.yystype* %arrayidx1185 to %struct.LIST_HELP**
  %534 = load %struct.LIST_HELP** %list1186, align 4
  call void @list_DeleteWithElement(%struct.LIST_HELP* %534, void (i8*)* @string_StringFree) #1
  br label %sw.epilog1200

sw.bb1187:                                        ; preds = %yyreduce
  %string1189 = bitcast %union.yystype* %yyvsp.2 to i8**
  %535 = load i8** %string1189, align 4
  %call.i.i2196 = call i8* @memory_Malloc(i32 8) #1
  %car.i.i2197 = getelementptr inbounds i8* %call.i.i2196, i32 4
  %536 = bitcast i8* %car.i.i2197 to i8**
  store i8* %535, i8** %536, align 4
  %cdr.i.i2198 = bitcast i8* %call.i.i2196 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i2198, align 4
  %call1190.c = ptrtoint i8* %call.i.i2196 to i32
  store i32 %call1190.c, i32* %2, align 4
  br label %sw.epilog1200

sw.bb1192:                                        ; preds = %yyreduce
  %arrayidx1193 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 -2
  %list1194 = bitcast %union.yystype* %arrayidx1193 to %struct.LIST_HELP**
  %537 = load %struct.LIST_HELP** %list1194, align 4
  %string1196 = bitcast %union.yystype* %yyvsp.2 to i8**
  %538 = load i8** %string1196, align 4
  %call.i.i2199 = call i8* @memory_Malloc(i32 8) #1
  %539 = bitcast i8* %call.i.i2199 to %struct.LIST_HELP*
  %car.i.i2200 = getelementptr inbounds i8* %call.i.i2199, i32 4
  %540 = bitcast i8* %car.i.i2200 to i8**
  store i8* %538, i8** %540, align 4
  %cdr.i.i2201 = bitcast i8* %call.i.i2199 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i2201, align 4
  %cmp.i.i2202 = icmp eq %struct.LIST_HELP* %537, null
  br i1 %cmp.i.i2202, label %list_Nconc.exit2212, label %if.end.i2204

if.end.i2204:                                     ; preds = %sw.bb1192
  %cmp.i18.i2203 = icmp eq i8* %call.i.i2199, null
  br i1 %cmp.i18.i2203, label %list_Nconc.exit2212, label %for.cond.i2209

for.cond.i2209:                                   ; preds = %if.end.i2204, %for.cond.i2209
  %List1.addr.0.i2205 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i2207, %for.cond.i2209 ], [ %537, %if.end.i2204 ]
  %List1.addr.0.idx15.i2206 = getelementptr %struct.LIST_HELP* %List1.addr.0.i2205, i32 0, i32 0
  %List1.addr.0.idx15.val.i2207 = load %struct.LIST_HELP** %List1.addr.0.idx15.i2206, align 4
  %cmp.i16.i2208 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i2207, null
  br i1 %cmp.i16.i2208, label %for.end.i2210, label %for.cond.i2209

for.end.i2210:                                    ; preds = %for.cond.i2209
  store %struct.LIST_HELP* %539, %struct.LIST_HELP** %List1.addr.0.idx15.i2206, align 4
  br label %list_Nconc.exit2212

list_Nconc.exit2212:                              ; preds = %sw.bb1192, %if.end.i2204, %for.end.i2210
  %retval.0.i2211 = phi %struct.LIST_HELP* [ %537, %for.end.i2210 ], [ %539, %sw.bb1192 ], [ %537, %if.end.i2204 ]
  %call1198.c = ptrtoint %struct.LIST_HELP* %retval.0.i2211 to i32
  store i32 %call1198.c, i32* %2, align 4
  br label %sw.epilog1200

sw.epilog1200:                                    ; preds = %for.cond.preheader, %if.end1075, %sw.bb770, %sw.bb479, %sw.bb494, %if.end925, %if.end968, %if.end1125, %if.then1134, %if.end991, %if.else993, %if.then973, %if.then930, %if.then775, %list_Nconc.exit1922, %list_Nconc.exit1933, %if.end512, %if.end489, %if.then410, %if.else414, %list_Nconc.exit1824, %list_Nconc.exit1835, %yyreduce, %list_Nconc.exit2212, %sw.bb1187, %sw.bb1184, %sw.epilog, %sw.bb1138, %if.end1110, %flag_SetFlagValue.exit, %sw.bb1047, %sw.bb1046, %if.end1043, %list_Nconc.exit2157, %dfg_VarCheck.exit2143, %sw.bb1030, %sw.bb1028, %sw.bb1024, %sw.bb1022, %sw.bb1020, %sw.bb1018, %sw.bb1016, %sw.bb1014, %sw.bb1012, %sw.bb1010, %sw.bb1008, %sw.bb1006, %sw.bb1004, %sw.bb1002, %sw.bb998, %sw.bb977, %sw.bb889, %sw.bb887, %cond.end884, %cond.end864, %dfg_VarCheck.exit2098, %if.end784, %cond.end767, %cond.end752, %cond.end741, %cond.end729, %cond.end718, %cond.end706, %cond.end691, %cond.end681, %cond.end671, %list_Nconc.exit1995, %sw.bb650, %cond.end647, %sw.bb632, %cond.end629, %cond.end614, %cond.end603, %cond.end592, %sw.bb581, %sw.bb580, %sw.bb576, %sw.bb572, %sw.bb570, %dfg_VarCheck.exit1946, %sw.bb538, %sw.bb536, %sw.bb535, %cond.end476, %cond.end461, %cond.end450, %cond.end442, %cond.end434, %cond.end426, %sw.bb405, %sw.bb402, %sw.bb399, %sw.bb396, %sw.bb393, %sw.bb390, %sw.bb387, %cond.end384, %cond.end369, %sw.bb357, %sw.bb355, %cond.end352, %sw.bb340, %sw.bb339, %cond.end336, %cond.end324, %cond.end309, %sw.bb296, %sw.bb292, %sw.bb290, %dfg_VarCheck.exit1847, %sw.bb259, %sw.bb257, %sw.bb255, %sw.bb236, %sw.bb233, %sw.bb226, %sw.bb221, %sw.bb219, %sw.bb217, %dfg_SymbolGenerated.exit, %list_Nconc.exit1792, %sw.bb199, %sw.bb198, %sw.bb195, %list_Nconc.exit, %dfg_SubSort.exit, %sw.bb181, %sw.bb179, %sw.bb174, %sw.bb171, %sw.bb166, %sw.bb163, %sw.bb160, %sw.bb157, %sw.bb152, %sw.bb149, %sw.bb145, %sw.bb142, %sw.bb140, %sw.bb138, %sw.bb136, %sw.bb133, %sw.bb130, %sw.bb127, %sw.bb124, %sw.bb122, %sw.bb119, %sw.bb116
  %idx.neg = sub i32 0, %conv112
  %add.ptr1203 = getelementptr inbounds i16* %yyssp.2, i32 %idx.neg
  %incdec.ptr1204 = getelementptr inbounds %union.yystype* %yyvsp.2, i32 %sub113
  %541 = load i32* %2, align 4
  %542 = getelementptr inbounds %union.yystype* %incdec.ptr1204, i32 0, i32 0
  store i32 %541, i32* %542, align 4
  %arrayidx1205 = getelementptr inbounds [197 x i8]* @yyr1, i32 0, i32 %conv106
  %543 = load i8* %arrayidx1205, align 1
  %conv1206 = zext i8 %543 to i32
  %sub1207 = add nsw i32 %conv1206, -71
  %arrayidx1208 = getelementptr inbounds [100 x i16]* @yypgoto, i32 0, i32 %sub1207
  %544 = load i16* %arrayidx1208, align 2
  %conv1209 = sext i16 %544 to i32
  %545 = load i16* %add.ptr1203, align 2
  %conv1210 = sext i16 %545 to i32
  %add1211 = add nsw i32 %conv1210, %conv1209
  %546 = icmp ult i32 %add1211, 507
  br i1 %546, label %land.lhs.true1217, label %if.else1226

land.lhs.true1217:                                ; preds = %sw.epilog1200
  %arrayidx1218 = getelementptr inbounds [507 x i16]* @yycheck, i32 0, i32 %add1211
  %547 = load i16* %arrayidx1218, align 2
  %cmp1221 = icmp eq i16 %547, %545
  br i1 %cmp1221, label %if.then1223, label %if.else1226

if.then1223:                                      ; preds = %land.lhs.true1217
  %arrayidx1224 = getelementptr inbounds [507 x i16]* @yytable, i32 0, i32 %add1211
  %548 = load i16* %arrayidx1224, align 2
  %conv1225 = zext i16 %548 to i32
  br label %yynewstate

if.else1226:                                      ; preds = %land.lhs.true1217, %sw.epilog1200
  %arrayidx1228 = getelementptr inbounds [100 x i16]* @yydefgoto, i32 0, i32 %sub1207
  %549 = load i16* %arrayidx1228, align 2
  %conv1229 = sext i16 %549 to i32
  br label %yynewstate

if.then1232:                                      ; preds = %yydefault, %if.end79
  %550 = load i32* @dfg_nerrs, align 4
  %inc = add nsw i32 %550, 1
  store i32 %inc, i32* @dfg_nerrs, align 4
  %cmp1235 = icmp sgt i16 %11, -356
  br i1 %cmp1235, label %if.then1240, label %if.else1329

if.then1240:                                      ; preds = %if.then1232
  %551 = load i32* @dfg_char, align 4
  %cmp1242 = icmp ult i32 %551, 319
  br i1 %cmp1242, label %cond.true1244, label %cond.end1248

cond.true1244:                                    ; preds = %if.then1240
  %arrayidx1245 = getelementptr inbounds [319 x i8]* @yytranslate, i32 0, i32 %551
  %552 = load i8* %arrayidx1245, align 1
  %conv1246 = zext i8 %552 to i32
  br label %cond.end1248

cond.end1248:                                     ; preds = %if.then1240, %cond.true1244
  %cond1249 = phi i32 [ %conv1246, %cond.true1244 ], [ 2, %if.then1240 ]
  %cmp1250 = icmp slt i16 %11, 0
  %sub1253 = sub nsw i32 0, %conv51
  %sub1253. = select i1 %cmp1250, i32 %sub1253, i32 0
  %cmp12582218 = icmp slt i32 %sub1253., 172
  br i1 %cmp12582218, label %for.body1260, label %for.end1278

for.body1260:                                     ; preds = %cond.end1248, %for.inc1276
  %yycount.02221 = phi i32 [ %yycount.1, %for.inc1276 ], [ 0, %cond.end1248 ]
  %yyx.02220 = phi i32 [ %inc1277, %for.inc1276 ], [ %sub1253., %cond.end1248 ]
  %yysize1241.02219 = phi i32 [ %yysize1241.1, %for.inc1276 ], [ 0, %cond.end1248 ]
  %add1261 = add nsw i32 %yyx.02220, %conv51
  %arrayidx1262 = getelementptr inbounds [507 x i16]* @yycheck, i32 0, i32 %add1261
  %553 = load i16* %arrayidx1262, align 2
  %conv1263 = sext i16 %553 to i32
  %cmp1264 = icmp eq i32 %conv1263, %yyx.02220
  %cmp1267 = icmp ne i32 %yyx.02220, 1
  %or.cond1400 = and i1 %cmp1264, %cmp1267
  br i1 %or.cond1400, label %if.then1269, label %for.inc1276

if.then1269:                                      ; preds = %for.body1260
  %arrayidx1270 = getelementptr inbounds [172 x i8*]* @yytname, i32 0, i32 %yyx.02220
  %554 = load i8** %arrayidx1270, align 4
  %call1271 = call i32 @strlen(i8* %554) #6
  %add1272 = add i32 %yysize1241.02219, 15
  %add1273 = add i32 %add1272, %call1271
  %inc1274 = add nsw i32 %yycount.02221, 1
  br label %for.inc1276

for.inc1276:                                      ; preds = %for.body1260, %if.then1269
  %yysize1241.1 = phi i32 [ %add1273, %if.then1269 ], [ %yysize1241.02219, %for.body1260 ]
  %yycount.1 = phi i32 [ %inc1274, %if.then1269 ], [ %yycount.02221, %for.body1260 ]
  %inc1277 = add nsw i32 %yyx.02220, 1
  %exitcond2382 = icmp eq i32 %inc1277, 172
  br i1 %exitcond2382, label %for.end1278, label %for.body1260

for.end1278:                                      ; preds = %for.inc1276, %cond.end1248
  %yycount.0.lcssa = phi i32 [ 0, %cond.end1248 ], [ %yycount.1, %for.inc1276 ]
  %yysize1241.0.lcssa = phi i32 [ 0, %cond.end1248 ], [ %yysize1241.1, %for.inc1276 ]
  %add1279 = add i32 %yysize1241.0.lcssa, 25
  %arrayidx1280 = getelementptr inbounds [172 x i8*]* @yytname, i32 0, i32 %cond1249
  %555 = load i8** %arrayidx1280, align 4
  %call1281 = call i32 @strlen(i8* %555) #6
  %add1282 = add i32 %add1279, %call1281
  %556 = alloca i8, i32 %add1282, align 4
  %557 = getelementptr i8* %556, i32 24
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %556, i8* getelementptr inbounds ([25 x i8]* @.str16, i32 0, i32 0), i32 25, i32 1, i1 false)
  %call1288 = call i8* @stpcpy(i8* %557, i8* %555) #1
  %cmp1289.not = icmp sgt i32 %yycount.0.lcssa, 4
  %cmp12582218.not = xor i1 %cmp12582218, true
  %brmerge = or i1 %cmp1289.not, %cmp12582218.not
  br i1 %brmerge, label %if.end1323, label %for.body1302

for.body1302:                                     ; preds = %for.end1278, %for.inc1320
  %yyp.02217 = phi i8* [ %yyp.1, %for.inc1320 ], [ %call1288, %for.end1278 ]
  %yycount.22216 = phi i32 [ %yycount.3, %for.inc1320 ], [ 0, %for.end1278 ]
  %yyx.12215 = phi i32 [ %inc1321, %for.inc1320 ], [ %sub1253., %for.end1278 ]
  %add1303 = add nsw i32 %yyx.12215, %conv51
  %arrayidx1304 = getelementptr inbounds [507 x i16]* @yycheck, i32 0, i32 %add1303
  %558 = load i16* %arrayidx1304, align 2
  %conv1305 = sext i16 %558 to i32
  %cmp1306 = icmp eq i32 %conv1305, %yyx.12215
  %cmp1309 = icmp ne i32 %yyx.12215, 1
  %or.cond1401 = and i1 %cmp1306, %cmp1309
  br i1 %or.cond1401, label %if.then1311, label %for.inc1320

if.then1311:                                      ; preds = %for.body1302
  %lnot1313 = icmp eq i32 %yycount.22216, 0
  %cond1314 = select i1 %lnot1313, i8* getelementptr inbounds ([13 x i8]* @.str17, i32 0, i32 0), i8* getelementptr inbounds ([5 x i8]* @.str18, i32 0, i32 0)
  %call1315 = call i8* @stpcpy(i8* %yyp.02217, i8* %cond1314) #1
  %arrayidx1316 = getelementptr inbounds [172 x i8*]* @yytname, i32 0, i32 %yyx.12215
  %559 = load i8** %arrayidx1316, align 4
  %call1317 = call i8* @stpcpy(i8* %call1315, i8* %559) #1
  %inc1318 = add nsw i32 %yycount.22216, 1
  br label %for.inc1320

for.inc1320:                                      ; preds = %for.body1302, %if.then1311
  %yycount.3 = phi i32 [ %inc1318, %if.then1311 ], [ %yycount.22216, %for.body1302 ]
  %yyp.1 = phi i8* [ %call1317, %if.then1311 ], [ %yyp.02217, %for.body1302 ]
  %inc1321 = add nsw i32 %yyx.12215, 1
  %exitcond = icmp eq i32 %inc1321, 172
  br i1 %exitcond, label %if.end1323, label %for.body1302

if.end1323:                                       ; preds = %for.end1278, %for.inc1320
  call void @dfg_error(i8* %556)
  unreachable

if.else1329:                                      ; preds = %if.then1232
  call void @dfg_error(i8* getelementptr inbounds ([12 x i8]* @.str20, i32 0, i32 0))
  unreachable

yyoverflowlab:                                    ; preds = %if.then
  call void @dfg_error(i8* getelementptr inbounds ([22 x i8]* @.str21, i32 0, i32 0))
  unreachable

yyreturn:                                         ; preds = %if.end, %if.end92, %sw.bb
  %yyresult.0 = phi i32 [ 0, %sw.bb ], [ 1, %if.end ], [ 0, %if.end92 ]
  call void @llvm.lifetime.end(i64 800, i8* %1) #1
  call void @llvm.lifetime.end(i64 400, i8* %0) #1
  ret i32 %yyresult.0
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) #1

declare i32 @dfg_lex() #2

declare void @string_StringFree(i8*) #2

; cmp16: 	.ent	dfg_parse
; cmp16:	.end	dfg_parse 

; Function Attrs: nounwind
define internal fastcc void @dfg_SymbolDecl(i32 %SymbolType, i8* %Name, i32 %Arity) #0 {
entry:
  switch i32 %Arity, label %sw.default [
    i32 -2, label %sw.epilog
    i32 -1, label %sw.bb1
  ]

sw.bb1:                                           ; preds = %entry
  %0 = load %struct._IO_FILE** @stdout, align 4
  %call = tail call i32 @fflush(%struct._IO_FILE* %0) #1
  %1 = load i32* @dfg_LINENUMBER, align 4
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([58 x i8]* @.str52, i32 0, i32 0), i32 %1) #1
  tail call fastcc void @misc_Error()
  unreachable

sw.default:                                       ; preds = %entry
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.default
  %arity.0 = phi i32 [ %Arity, %sw.default ], [ 0, %entry ]
  %call2 = tail call i32 @strlen(i8* %Name) #6
  %cmp = icmp ugt i32 %call2, 63
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %sw.epilog
  %arrayidx = getelementptr inbounds i8* %Name, i32 63
  store i8 0, i8* %arrayidx, align 1
  br label %if.end

if.end:                                           ; preds = %if.then, %sw.epilog
  %call3 = tail call i32 @symbol_Lookup(i8* %Name) #1
  %cmp4 = icmp eq i32 %call3, 0
  br i1 %cmp4, label %if.else, label %if.then5

if.then5:                                         ; preds = %if.end
  switch i32 %SymbolType, label %if.end27 [
    i32 284, label %land.lhs.true
    i32 298, label %land.lhs.true9
    i32 300, label %land.lhs.true16
    i32 294, label %land.lhs.true16
  ]

land.lhs.true:                                    ; preds = %if.then5
  %tobool.i = icmp sgt i32 %call3, -1
  br i1 %tobool.i, label %if.then19, label %land.rhs.i

land.rhs.i:                                       ; preds = %land.lhs.true
  %sub.i6.i = sub nsw i32 0, %call3
  %2 = load i32* @symbol_TYPEMASK, align 4
  %and.i7.i = and i32 %2, %sub.i6.i
  %3 = icmp ult i32 %and.i7.i, 2
  br i1 %3, label %if.end27, label %if.then19

land.lhs.true9:                                   ; preds = %if.then5
  %tobool.i77 = icmp sgt i32 %call3, -1
  br i1 %tobool.i77, label %if.then19, label %land.rhs.i78

land.rhs.i78:                                     ; preds = %land.lhs.true9
  %sub.i.i = sub nsw i32 0, %call3
  %4 = load i32* @symbol_TYPEMASK, align 4
  %and.i.i = and i32 %4, %sub.i.i
  %cmp.i = icmp eq i32 %and.i.i, 2
  br i1 %cmp.i, label %lor.lhs.false12, label %if.then19

lor.lhs.false12:                                  ; preds = %land.rhs.i78
  switch i32 %SymbolType, label %if.end27 [
    i32 300, label %land.lhs.true16
    i32 294, label %land.lhs.true16
  ]

land.lhs.true16:                                  ; preds = %if.then5, %if.then5, %lor.lhs.false12, %lor.lhs.false12
  %tobool.i80 = icmp sgt i32 %call3, -1
  br i1 %tobool.i80, label %if.then19, label %land.rhs.i84

land.rhs.i84:                                     ; preds = %land.lhs.true16
  %sub.i.i81 = sub nsw i32 0, %call3
  %5 = load i32* @symbol_TYPEMASK, align 4
  %and.i.i82 = and i32 %5, %sub.i.i81
  %cmp.i83 = icmp eq i32 %and.i.i82, 3
  br i1 %cmp.i83, label %if.end27, label %if.then19

if.then19:                                        ; preds = %land.lhs.true16, %land.lhs.true9, %land.lhs.true, %land.rhs.i, %land.rhs.i78, %land.rhs.i84
  %6 = load %struct._IO_FILE** @stdout, align 4
  %call20 = tail call i32 @fflush(%struct._IO_FILE* %6) #1
  %7 = load i32* @dfg_LINENUMBER, align 4
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([46 x i8]* @.str53, i32 0, i32 0), i32 %7, i8* %Name) #1
  %sub.i = sub nsw i32 0, %call3
  %8 = load i32* @symbol_TYPEMASK, align 4
  %and.i = and i32 %8, %sub.i
  switch i32 %and.i, label %sw.default25 [
    i32 0, label %sw.bb22
    i32 1, label %sw.bb22
    i32 2, label %sw.bb23
    i32 3, label %sw.bb24
  ]

sw.bb22:                                          ; preds = %if.then19, %if.then19
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([11 x i8]* @.str54, i32 0, i32 0)) #1
  br label %sw.epilog26

sw.bb23:                                          ; preds = %if.then19
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([12 x i8]* @.str55, i32 0, i32 0)) #1
  br label %sw.epilog26

sw.bb24:                                          ; preds = %if.then19
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([10 x i8]* @.str56, i32 0, i32 0)) #1
  br label %sw.epilog26

sw.default25:                                     ; preds = %if.then19
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([15 x i8]* @.str57, i32 0, i32 0)) #1
  br label %sw.epilog26

sw.epilog26:                                      ; preds = %sw.default25, %sw.bb24, %sw.bb23, %sw.bb22
  tail call fastcc void @misc_Error()
  unreachable

if.end27:                                         ; preds = %land.rhs.i, %land.rhs.i84, %if.then5, %lor.lhs.false12
  %cmp28 = icmp eq i32 %Arity, -2
  br i1 %cmp28, label %if.end46, label %land.lhs.true29

land.lhs.true29:                                  ; preds = %if.end27
  %sub.i.i88 = sub nsw i32 0, %call3
  %9 = load i32* @symbol_TYPESTATBITS, align 4
  %shr.i.i89 = ashr i32 %sub.i.i88, %9
  %10 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i90 = getelementptr inbounds %struct.signature** %10, i32 %shr.i.i89
  %11 = load %struct.signature** %arrayidx.i.i90, align 4
  %arity.i91 = getelementptr inbounds %struct.signature* %11, i32 0, i32 3
  %12 = load i32* %arity.i91, align 4
  %cmp31 = icmp eq i32 %12, %Arity
  br i1 %cmp31, label %if.end46, label %if.then32

if.then32:                                        ; preds = %land.lhs.true29
  %13 = load %struct._IO_FILE** @stdout, align 4
  %call33 = tail call i32 @fflush(%struct._IO_FILE* %13) #1
  %14 = load i32* @dfg_LINENUMBER, align 4
  %15 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i = getelementptr inbounds %struct.signature** %15, i32 %shr.i.i89
  %16 = load %struct.signature** %arrayidx.i.i, align 4
  %arity.i87 = getelementptr inbounds %struct.signature* %16, i32 0, i32 3
  %17 = load i32* %arity.i87, align 4
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([57 x i8]* @.str58, i32 0, i32 0), i32 %14, i8* %Name, i32 %17) #1
  tail call fastcc void @misc_Error()
  unreachable

if.else:                                          ; preds = %if.end
  switch i32 %SymbolType, label %sw.default40 [
    i32 284, label %sw.bb36
    i32 298, label %sw.bb38
  ]

sw.bb36:                                          ; preds = %if.else
  %18 = load i32** @dfg_PRECEDENCE, align 4
  %call37 = tail call i32 @symbol_CreateFunction(i8* %Name, i32 %arity.0, i32 0, i32* %18) #1
  br label %sw.epilog42

sw.bb38:                                          ; preds = %if.else
  %19 = load i32** @dfg_PRECEDENCE, align 4
  %call39 = tail call i32 @symbol_CreatePredicate(i8* %Name, i32 %arity.0, i32 0, i32* %19) #1
  br label %sw.epilog42

sw.default40:                                     ; preds = %if.else
  %20 = load i32** @dfg_PRECEDENCE, align 4
  %call41 = tail call i32 @symbol_CreateJunctor(i8* %Name, i32 %arity.0, i32 0, i32* %20) #1
  br label %sw.epilog42

sw.epilog42:                                      ; preds = %sw.default40, %sw.bb38, %sw.bb36
  %symbol.0 = phi i32 [ %call41, %sw.default40 ], [ %call39, %sw.bb38 ], [ %call37, %sw.bb36 ]
  %cmp43 = icmp eq i32 %Arity, -2
  br i1 %cmp43, label %if.then44, label %if.end46

if.then44:                                        ; preds = %sw.epilog42
  %call.i.i = tail call i8* @memory_Malloc(i32 12) #1
  %symbol.i = bitcast i8* %call.i.i to i32*
  store i32 %symbol.0, i32* %symbol.i, align 4
  %valid.i = getelementptr inbounds i8* %call.i.i, i32 4
  %21 = bitcast i8* %valid.i to i32*
  store i32 0, i32* %21, align 4
  %arity.i = getelementptr inbounds i8* %call.i.i, i32 8
  %22 = bitcast i8* %arity.i to i32*
  store i32 0, i32* %22, align 4
  %23 = load %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  %call.i5.i = tail call i8* @memory_Malloc(i32 8) #1
  %24 = bitcast i8* %call.i5.i to %struct.LIST_HELP*
  %car.i.i = getelementptr inbounds i8* %call.i5.i, i32 4
  %25 = bitcast i8* %car.i.i to i8**
  store i8* %call.i.i, i8** %25, align 4
  %cdr.i.i = bitcast i8* %call.i5.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* %23, %struct.LIST_HELP** %cdr.i.i, align 4
  store %struct.LIST_HELP* %24, %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  br label %if.end46

if.end46:                                         ; preds = %land.lhs.true29, %if.end27, %sw.epilog42, %if.then44
  br i1 %cmp, label %if.then48, label %if.end50

if.then48:                                        ; preds = %if.end46
  %arrayidx49 = getelementptr inbounds i8* %Name, i32 63
  store i8 32, i8* %arrayidx49, align 1
  br label %if.end50

if.end50:                                         ; preds = %if.then48, %if.end46
  tail call void @string_StringFree(i8* %Name) #1
  ret void
}

; Function Attrs: nounwind
define %struct.term* @dfg_CreateQuantifier(i32 %Symbol, %struct.LIST_HELP* %VarTermList, %struct.term* %Term) #0 {
entry:
  %cmp.i240 = icmp eq %struct.LIST_HELP* %VarTermList, null
  br i1 %cmp.i240, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %VarTermList.addr.0243 = phi %struct.LIST_HELP* [ %L.idx.val.i, %for.inc ], [ %VarTermList, %entry ]
  %sortlist.0242 = phi %struct.LIST_HELP* [ %sortlist.1, %for.inc ], [ null, %entry ]
  %varlist.0241 = phi %struct.LIST_HELP* [ %varlist.1, %for.inc ], [ null, %entry ]
  %VarTermList.addr.0.idx = getelementptr %struct.LIST_HELP* %VarTermList.addr.0243, i32 0, i32 1
  %VarTermList.addr.0.idx.val = load i8** %VarTermList.addr.0.idx, align 4
  %0 = bitcast i8* %VarTermList.addr.0.idx.val to %struct.term*
  %.idx135 = bitcast i8* %VarTermList.addr.0.idx.val to i32*
  %.idx135.val = load i32* %.idx135, align 4
  %cmp.i.i157 = icmp sgt i32 %.idx135.val, 0
  br i1 %cmp.i.i157, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %1 = inttoptr i32 %.idx135.val to i8*
  %call.i.i165 = tail call i8* @memory_Malloc(i32 8) #1
  %2 = bitcast i8* %call.i.i165 to %struct.LIST_HELP*
  %car.i.i166 = getelementptr inbounds i8* %call.i.i165, i32 4
  %3 = bitcast i8* %car.i.i166 to i8**
  store i8* %1, i8** %3, align 4
  %cdr.i.i167 = bitcast i8* %call.i.i165 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i167, align 4
  %cmp.i.i168 = icmp eq %struct.LIST_HELP* %varlist.0241, null
  br i1 %cmp.i.i168, label %list_Nconc.exit178, label %if.end.i170

if.end.i170:                                      ; preds = %if.then
  %cmp.i18.i169 = icmp eq i8* %call.i.i165, null
  br i1 %cmp.i18.i169, label %list_Nconc.exit178, label %for.cond.i175

for.cond.i175:                                    ; preds = %if.end.i170, %for.cond.i175
  %List1.addr.0.i171 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i173, %for.cond.i175 ], [ %varlist.0241, %if.end.i170 ]
  %List1.addr.0.idx15.i172 = getelementptr %struct.LIST_HELP* %List1.addr.0.i171, i32 0, i32 0
  %List1.addr.0.idx15.val.i173 = load %struct.LIST_HELP** %List1.addr.0.idx15.i172, align 4
  %cmp.i16.i174 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i173, null
  br i1 %cmp.i16.i174, label %for.end.i176, label %for.cond.i175

for.end.i176:                                     ; preds = %for.cond.i175
  store %struct.LIST_HELP* %2, %struct.LIST_HELP** %List1.addr.0.idx15.i172, align 4
  br label %list_Nconc.exit178

list_Nconc.exit178:                               ; preds = %if.then, %if.end.i170, %for.end.i176
  %retval.0.i177 = phi %struct.LIST_HELP* [ %varlist.0241, %for.end.i176 ], [ %2, %if.then ], [ %varlist.0241, %if.end.i170 ]
  tail call void @term_Delete(%struct.term* %0) #1
  br label %for.inc

if.else:                                          ; preds = %for.body
  %.idx136 = getelementptr i8* %VarTermList.addr.0.idx.val, i32 8
  %4 = bitcast i8* %.idx136 to %struct.LIST_HELP**
  %.idx136.val = load %struct.LIST_HELP** %4, align 4
  %.idx136.val.idx = getelementptr %struct.LIST_HELP* %.idx136.val, i32 0, i32 1
  %.idx136.val.idx.val = load i8** %.idx136.val.idx, align 4
  %call8.idx = bitcast i8* %.idx136.val.idx.val to i32*
  %call8.idx.val = load i32* %call8.idx, align 4
  %5 = inttoptr i32 %call8.idx.val to i8*
  %call.i.i233 = tail call i8* @memory_Malloc(i32 8) #1
  %6 = bitcast i8* %call.i.i233 to %struct.LIST_HELP*
  %car.i.i234 = getelementptr inbounds i8* %call.i.i233, i32 4
  %7 = bitcast i8* %car.i.i234 to i8**
  store i8* %5, i8** %7, align 4
  %cdr.i.i235 = bitcast i8* %call.i.i233 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i235, align 4
  %cmp.i.i222 = icmp eq %struct.LIST_HELP* %varlist.0241, null
  br i1 %cmp.i.i222, label %list_Nconc.exit232, label %if.end.i224

if.end.i224:                                      ; preds = %if.else
  %cmp.i18.i223 = icmp eq i8* %call.i.i233, null
  br i1 %cmp.i18.i223, label %list_Nconc.exit232, label %for.cond.i229

for.cond.i229:                                    ; preds = %if.end.i224, %for.cond.i229
  %List1.addr.0.i225 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i227, %for.cond.i229 ], [ %varlist.0241, %if.end.i224 ]
  %List1.addr.0.idx15.i226 = getelementptr %struct.LIST_HELP* %List1.addr.0.i225, i32 0, i32 0
  %List1.addr.0.idx15.val.i227 = load %struct.LIST_HELP** %List1.addr.0.idx15.i226, align 4
  %cmp.i16.i228 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i227, null
  br i1 %cmp.i16.i228, label %for.end.i230, label %for.cond.i229

for.end.i230:                                     ; preds = %for.cond.i229
  store %struct.LIST_HELP* %6, %struct.LIST_HELP** %List1.addr.0.idx15.i226, align 4
  br label %list_Nconc.exit232

list_Nconc.exit232:                               ; preds = %if.else, %if.end.i224, %for.end.i230
  %retval.0.i231 = phi %struct.LIST_HELP* [ %varlist.0241, %for.end.i230 ], [ %6, %if.else ], [ %varlist.0241, %if.end.i224 ]
  %call.i.i219 = tail call i8* @memory_Malloc(i32 8) #1
  %8 = bitcast i8* %call.i.i219 to %struct.LIST_HELP*
  %car.i.i220 = getelementptr inbounds i8* %call.i.i219, i32 4
  %9 = bitcast i8* %car.i.i220 to i8**
  store i8* %VarTermList.addr.0.idx.val, i8** %9, align 4
  %cdr.i.i221 = bitcast i8* %call.i.i219 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i221, align 4
  %cmp.i.i208 = icmp eq %struct.LIST_HELP* %sortlist.0242, null
  br i1 %cmp.i.i208, label %for.inc, label %if.end.i210

if.end.i210:                                      ; preds = %list_Nconc.exit232
  %cmp.i18.i209 = icmp eq i8* %call.i.i219, null
  br i1 %cmp.i18.i209, label %for.inc, label %for.cond.i215

for.cond.i215:                                    ; preds = %if.end.i210, %for.cond.i215
  %List1.addr.0.i211 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i213, %for.cond.i215 ], [ %sortlist.0242, %if.end.i210 ]
  %List1.addr.0.idx15.i212 = getelementptr %struct.LIST_HELP* %List1.addr.0.i211, i32 0, i32 0
  %List1.addr.0.idx15.val.i213 = load %struct.LIST_HELP** %List1.addr.0.idx15.i212, align 4
  %cmp.i16.i214 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i213, null
  br i1 %cmp.i16.i214, label %for.end.i216, label %for.cond.i215

for.end.i216:                                     ; preds = %for.cond.i215
  store %struct.LIST_HELP* %8, %struct.LIST_HELP** %List1.addr.0.idx15.i212, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.end.i216, %if.end.i210, %list_Nconc.exit232, %list_Nconc.exit178
  %varlist.1 = phi %struct.LIST_HELP* [ %retval.0.i177, %list_Nconc.exit178 ], [ %retval.0.i231, %list_Nconc.exit232 ], [ %retval.0.i231, %if.end.i210 ], [ %retval.0.i231, %for.end.i216 ]
  %sortlist.1 = phi %struct.LIST_HELP* [ %sortlist.0242, %list_Nconc.exit178 ], [ %8, %list_Nconc.exit232 ], [ %sortlist.0242, %if.end.i210 ], [ %sortlist.0242, %for.end.i216 ]
  %L.idx.i = getelementptr %struct.LIST_HELP* %VarTermList.addr.0243, i32 0, i32 0
  %L.idx.val.i = load %struct.LIST_HELP** %L.idx.i, align 4
  %10 = bitcast %struct.LIST_HELP* %VarTermList.addr.0243 to i8*
  %11 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %11, i32 0, i32 4
  %12 = load i32* %total_size.i.i.i, align 4
  %13 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i = add i32 %13, %12
  store i32 %add24.i.i.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %11, i32 0, i32 0
  %14 = load i8** %free.i.i.i, align 4
  %.c.i.i = bitcast i8* %14 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i, %struct.LIST_HELP** %L.idx.i, align 4
  %15 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %15, i32 0, i32 0
  store i8* %10, i8** %free27.i.i.i, align 4
  %cmp.i = icmp eq %struct.LIST_HELP* %L.idx.val.i, null
  br i1 %cmp.i, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  %sortlist.0.lcssa = phi %struct.LIST_HELP* [ null, %entry ], [ %sortlist.1, %for.inc ]
  %varlist.0.lcssa = phi %struct.LIST_HELP* [ null, %entry ], [ %varlist.1, %for.inc ]
  %call15 = tail call %struct.LIST_HELP* @list_PointerDeleteDuplicates(%struct.LIST_HELP* %varlist.0.lcssa) #1
  %cmp.i206238 = icmp eq %struct.LIST_HELP* %call15, null
  br i1 %cmp.i206238, label %for.end26, label %for.body20

for.body20:                                       ; preds = %for.end, %for.body20
  %scan.0239 = phi %struct.LIST_HELP* [ %scan.0.idx133.val, %for.body20 ], [ %call15, %for.end ]
  %scan.0.idx = getelementptr %struct.LIST_HELP* %scan.0239, i32 0, i32 1
  %scan.0.idx.val = load i8** %scan.0.idx, align 4
  %16 = ptrtoint i8* %scan.0.idx.val to i32
  %call23 = tail call %struct.term* @term_Create(i32 %16, %struct.LIST_HELP* null) #1
  %17 = bitcast %struct.term* %call23 to i8*
  store i8* %17, i8** %scan.0.idx, align 4
  %scan.0.idx133 = getelementptr %struct.LIST_HELP* %scan.0239, i32 0, i32 0
  %scan.0.idx133.val = load %struct.LIST_HELP** %scan.0.idx133, align 4
  %cmp.i206 = icmp eq %struct.LIST_HELP* %scan.0.idx133.val, null
  br i1 %cmp.i206, label %for.end26, label %for.body20

for.end26:                                        ; preds = %for.body20, %for.end
  %cmp.i203 = icmp eq %struct.LIST_HELP* %sortlist.0.lcssa, null
  br i1 %cmp.i203, label %if.end90, label %if.then29

if.then29:                                        ; preds = %for.end26
  %18 = load i32* @fol_ALL, align 4
  %cmp.i201 = icmp eq i32 %18, %Symbol
  br i1 %cmp.i201, label %if.then33, label %if.else70

if.then33:                                        ; preds = %if.then29
  %19 = load i32* @fol_OR, align 4
  %Term.idx134 = getelementptr %struct.term* %Term, i32 0, i32 0
  %Term.idx134.val = load i32* %Term.idx134, align 4
  %cmp.i199 = icmp eq i32 %19, %Term.idx134.val
  br i1 %cmp.i199, label %for.body43, label %if.else53

for.body43:                                       ; preds = %if.then33, %for.body43
  %scan.1237 = phi %struct.LIST_HELP* [ %scan.1.idx132.val, %for.body43 ], [ %sortlist.0.lcssa, %if.then33 ]
  %20 = load i32* @fol_NOT, align 4
  %scan.1.idx = getelementptr %struct.LIST_HELP* %scan.1237, i32 0, i32 1
  %scan.1.idx.val = load i8** %scan.1.idx, align 4
  %call.i.i194 = tail call i8* @memory_Malloc(i32 8) #1
  %21 = bitcast i8* %call.i.i194 to %struct.LIST_HELP*
  %car.i.i195 = getelementptr inbounds i8* %call.i.i194, i32 4
  %22 = bitcast i8* %car.i.i195 to i8**
  store i8* %scan.1.idx.val, i8** %22, align 4
  %cdr.i.i196 = bitcast i8* %call.i.i194 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i196, align 4
  %call47 = tail call %struct.term* @term_Create(i32 %20, %struct.LIST_HELP* %21) #1
  %23 = bitcast %struct.term* %call47 to i8*
  store i8* %23, i8** %scan.1.idx, align 4
  %scan.1.idx132 = getelementptr %struct.LIST_HELP* %scan.1237, i32 0, i32 0
  %scan.1.idx132.val = load %struct.LIST_HELP** %scan.1.idx132, align 4
  %cmp.i197 = icmp eq %struct.LIST_HELP* %scan.1.idx132.val, null
  br i1 %cmp.i197, label %if.end.i184, label %for.body43

if.end.i184:                                      ; preds = %for.body43
  %Term.idx138 = getelementptr %struct.term* %Term, i32 0, i32 2
  %Term.idx138.val = load %struct.LIST_HELP** %Term.idx138, align 4
  %cmp.i18.i183 = icmp eq %struct.LIST_HELP* %Term.idx138.val, null
  br i1 %cmp.i18.i183, label %list_Nconc.exit192, label %for.cond.i189

for.cond.i189:                                    ; preds = %if.end.i184, %for.cond.i189
  %List1.addr.0.i185 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i187, %for.cond.i189 ], [ %sortlist.0.lcssa, %if.end.i184 ]
  %List1.addr.0.idx15.i186 = getelementptr %struct.LIST_HELP* %List1.addr.0.i185, i32 0, i32 0
  %List1.addr.0.idx15.val.i187 = load %struct.LIST_HELP** %List1.addr.0.idx15.i186, align 4
  %cmp.i16.i188 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i187, null
  br i1 %cmp.i16.i188, label %for.end.i190, label %for.cond.i189

for.end.i190:                                     ; preds = %for.cond.i189
  store %struct.LIST_HELP* %Term.idx138.val, %struct.LIST_HELP** %List1.addr.0.idx15.i186, align 4
  br label %list_Nconc.exit192

list_Nconc.exit192:                               ; preds = %if.end.i184, %for.end.i190
  store %struct.LIST_HELP* %sortlist.0.lcssa, %struct.LIST_HELP** %Term.idx138, align 4
  br label %if.end90

if.else53:                                        ; preds = %if.then33
  %sortlist.0.idx = getelementptr %struct.LIST_HELP* %sortlist.0.lcssa, i32 0, i32 0
  %sortlist.0.idx.val = load %struct.LIST_HELP** %sortlist.0.idx, align 4
  %cmp.i179 = icmp eq %struct.LIST_HELP* %sortlist.0.idx.val, null
  br i1 %cmp.i179, label %if.then57, label %if.else61

if.then57:                                        ; preds = %if.else53
  %24 = bitcast %struct.term* %Term to i8*
  %call.i.i162 = tail call i8* @memory_Malloc(i32 8) #1
  %25 = bitcast i8* %call.i.i162 to %struct.LIST_HELP*
  %car.i.i163 = getelementptr inbounds i8* %call.i.i162, i32 4
  %26 = bitcast i8* %car.i.i163 to i8**
  store i8* %24, i8** %26, align 4
  %cdr.i.i164 = bitcast i8* %call.i.i162 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i164, align 4
  store %struct.LIST_HELP* %25, %struct.LIST_HELP** %sortlist.0.idx, align 4
  %27 = load i32* @fol_IMPLIES, align 4
  %call60 = tail call %struct.term* @term_Create(i32 %27, %struct.LIST_HELP* %sortlist.0.lcssa) #1
  br label %if.end90

if.else61:                                        ; preds = %if.else53
  %28 = load i32* @fol_AND, align 4
  %call63 = tail call %struct.term* @term_Create(i32 %28, %struct.LIST_HELP* %sortlist.0.lcssa) #1
  %29 = load i32* @fol_IMPLIES, align 4
  %30 = bitcast %struct.term* %call63 to i8*
  %31 = bitcast %struct.term* %Term to i8*
  %call.i.i158 = tail call i8* @memory_Malloc(i32 8) #1
  %32 = bitcast i8* %call.i.i158 to %struct.LIST_HELP*
  %car.i.i159 = getelementptr inbounds i8* %call.i.i158, i32 4
  %33 = bitcast i8* %car.i.i159 to i8**
  store i8* %31, i8** %33, align 4
  %cdr.i.i160 = bitcast i8* %call.i.i158 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i160, align 4
  %call.i = tail call i8* @memory_Malloc(i32 8) #1
  %34 = bitcast i8* %call.i to %struct.LIST_HELP*
  %car.i = getelementptr inbounds i8* %call.i, i32 4
  %35 = bitcast i8* %car.i to i8**
  store i8* %30, i8** %35, align 4
  %cdr.i = bitcast i8* %call.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* %32, %struct.LIST_HELP** %cdr.i, align 4
  %call67 = tail call %struct.term* @term_Create(i32 %29, %struct.LIST_HELP* %34) #1
  br label %if.end90

if.else70:                                        ; preds = %if.then29
  %36 = load i32* @fol_EXIST, align 4
  %cmp.i155 = icmp eq i32 %36, %Symbol
  br i1 %cmp.i155, label %if.then74, label %if.end90

if.then74:                                        ; preds = %if.else70
  %37 = load i32* @fol_AND, align 4
  %Term.idx = getelementptr %struct.term* %Term, i32 0, i32 0
  %Term.idx.val = load i32* %Term.idx, align 4
  %cmp.i153 = icmp eq i32 %37, %Term.idx.val
  br i1 %cmp.i153, label %if.end.i144, label %if.end.i

if.end.i144:                                      ; preds = %if.then74
  %Term.idx137 = getelementptr %struct.term* %Term, i32 0, i32 2
  %Term.idx137.val = load %struct.LIST_HELP** %Term.idx137, align 4
  %cmp.i18.i143 = icmp eq %struct.LIST_HELP* %Term.idx137.val, null
  br i1 %cmp.i18.i143, label %list_Nconc.exit152, label %for.cond.i149

for.cond.i149:                                    ; preds = %if.end.i144, %for.cond.i149
  %List1.addr.0.i145 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i147, %for.cond.i149 ], [ %sortlist.0.lcssa, %if.end.i144 ]
  %List1.addr.0.idx15.i146 = getelementptr %struct.LIST_HELP* %List1.addr.0.i145, i32 0, i32 0
  %List1.addr.0.idx15.val.i147 = load %struct.LIST_HELP** %List1.addr.0.idx15.i146, align 4
  %cmp.i16.i148 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i147, null
  br i1 %cmp.i16.i148, label %for.end.i150, label %for.cond.i149

for.end.i150:                                     ; preds = %for.cond.i149
  store %struct.LIST_HELP* %Term.idx137.val, %struct.LIST_HELP** %List1.addr.0.idx15.i146, align 4
  br label %list_Nconc.exit152

list_Nconc.exit152:                               ; preds = %if.end.i144, %for.end.i150
  store %struct.LIST_HELP* %sortlist.0.lcssa, %struct.LIST_HELP** %Term.idx137, align 4
  br label %if.end90

if.end.i:                                         ; preds = %if.then74
  %38 = bitcast %struct.term* %Term to i8*
  %call.i.i139 = tail call i8* @memory_Malloc(i32 8) #1
  %39 = bitcast i8* %call.i.i139 to %struct.LIST_HELP*
  %car.i.i140 = getelementptr inbounds i8* %call.i.i139, i32 4
  %40 = bitcast i8* %car.i.i140 to i8**
  store i8* %38, i8** %40, align 4
  %cdr.i.i141 = bitcast i8* %call.i.i139 to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i141, align 4
  %cmp.i18.i = icmp eq i8* %call.i.i139, null
  br i1 %cmp.i18.i, label %list_Nconc.exit, label %for.cond.i

for.cond.i:                                       ; preds = %if.end.i, %for.cond.i
  %List1.addr.0.i = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i, %for.cond.i ], [ %sortlist.0.lcssa, %if.end.i ]
  %List1.addr.0.idx15.i = getelementptr %struct.LIST_HELP* %List1.addr.0.i, i32 0, i32 0
  %List1.addr.0.idx15.val.i = load %struct.LIST_HELP** %List1.addr.0.idx15.i, align 4
  %cmp.i16.i = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i, null
  br i1 %cmp.i16.i, label %for.end.i, label %for.cond.i

for.end.i:                                        ; preds = %for.cond.i
  store %struct.LIST_HELP* %39, %struct.LIST_HELP** %List1.addr.0.idx15.i, align 4
  br label %list_Nconc.exit

list_Nconc.exit:                                  ; preds = %if.end.i, %for.end.i
  %41 = load i32* @fol_AND, align 4
  %call86 = tail call %struct.term* @term_Create(i32 %41, %struct.LIST_HELP* %sortlist.0.lcssa) #1
  br label %if.end90

if.end90:                                         ; preds = %if.else70, %for.end26, %if.then57, %if.else61, %list_Nconc.exit192, %list_Nconc.exit152, %list_Nconc.exit
  %Term.addr.0 = phi %struct.term* [ %Term, %for.end26 ], [ %Term, %list_Nconc.exit192 ], [ %call60, %if.then57 ], [ %call67, %if.else61 ], [ %Term, %list_Nconc.exit152 ], [ %call86, %list_Nconc.exit ], [ %Term, %if.else70 ]
  %42 = bitcast %struct.term* %Term.addr.0 to i8*
  %call.i.i = tail call i8* @memory_Malloc(i32 8) #1
  %43 = bitcast i8* %call.i.i to %struct.LIST_HELP*
  %car.i.i = getelementptr inbounds i8* %call.i.i, i32 4
  %44 = bitcast i8* %car.i.i to i8**
  store i8* %42, i8** %44, align 4
  %cdr.i.i = bitcast i8* %call.i.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i, align 4
  %call92 = tail call %struct.term* @fol_CreateQuantifier(i32 %Symbol, %struct.LIST_HELP* %call15, %struct.LIST_HELP* %43) #1
  ret %struct.term* %call92
}

; Function Attrs: nounwind
define internal fastcc i32 @dfg_Symbol(i8* %Name, i32 %Arity) #0 {
entry:
  %call = tail call i32 @strlen(i8* %Name) #6
  %cmp = icmp ugt i32 %call, 63
  br i1 %cmp, label %if.then4, label %if.end

if.end:                                           ; preds = %entry
  %call2 = tail call i32 @symbol_Lookup(i8* %Name) #1
  br label %if.end6

if.then4:                                         ; preds = %entry
  %arrayidx = getelementptr inbounds i8* %Name, i32 63
  %0 = load i8* %arrayidx, align 1
  store i8 0, i8* %arrayidx, align 1
  %call234 = tail call i32 @symbol_Lookup(i8* %Name) #1
  store i8 %0, i8* %arrayidx, align 1
  br label %if.end6

if.end6:                                          ; preds = %if.end, %if.then4
  %call236 = phi i32 [ %call234, %if.then4 ], [ %call2, %if.end ]
  %cmp7 = icmp eq i32 %call236, 0
  br i1 %cmp7, label %if.else, label %if.then8

if.then8:                                         ; preds = %if.end6
  tail call void @string_StringFree(i8* %Name) #1
  %scan.047.i = load %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  %cmp.i48.i = icmp eq %struct.LIST_HELP* %scan.047.i, null
  br i1 %cmp.i48.i, label %while.end.i, label %while.body.i

while.cond.i:                                     ; preds = %while.body.i
  %scan.0.idx35.i = getelementptr %struct.LIST_HELP* %scan.049.i, i32 0, i32 0
  %scan.0.i = load %struct.LIST_HELP** %scan.0.idx35.i, align 4
  %cmp.i.i = icmp eq %struct.LIST_HELP* %scan.0.i, null
  br i1 %cmp.i.i, label %while.end.i, label %while.body.i

while.body.i:                                     ; preds = %if.then8, %while.cond.i
  %scan.049.i = phi %struct.LIST_HELP* [ %scan.0.i, %while.cond.i ], [ %scan.047.i, %if.then8 ]
  %scan.0.idx.i = getelementptr %struct.LIST_HELP* %scan.049.i, i32 0, i32 1
  %scan.0.idx.val.i = load i8** %scan.0.idx.i, align 4
  %symbol.i = bitcast i8* %scan.0.idx.val.i to i32*
  %1 = load i32* %symbol.i, align 4
  %cmp.i = icmp eq i32 %1, %call236
  br i1 %cmp.i, label %if.then.i, label %while.cond.i

if.then.i:                                        ; preds = %while.body.i
  %valid.i = getelementptr inbounds i8* %scan.0.idx.val.i, i32 4
  %2 = bitcast i8* %valid.i to i32*
  %3 = load i32* %2, align 4
  %tobool2.i = icmp eq i32 %3, 0
  %arity9.i = getelementptr inbounds i8* %scan.0.idx.val.i, i32 8
  %4 = bitcast i8* %arity9.i to i32*
  br i1 %tobool2.i, label %if.else.i, label %if.then3.i

if.then3.i:                                       ; preds = %if.then.i
  %5 = load i32* %4, align 4
  %cmp4.i = icmp eq i32 %5, %Arity
  br i1 %cmp4.i, label %if.end14, label %if.then5.i

if.then5.i:                                       ; preds = %if.then3.i
  %6 = load %struct._IO_FILE** @stdout, align 4
  %call6.i = tail call i32 @fflush(%struct._IO_FILE* %6) #1
  %7 = load i32* @dfg_LINENUMBER, align 4
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([11 x i8]* @.str47, i32 0, i32 0), i32 %7) #1
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([21 x i8]* @.str48, i32 0, i32 0), i32 %Arity) #1
  %sub.i.i43.i = sub nsw i32 0, %call236
  %8 = load i32* @symbol_TYPESTATBITS, align 4
  %shr.i.i44.i = ashr i32 %sub.i.i43.i, %8
  %9 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i45.i = getelementptr inbounds %struct.signature** %9, i32 %shr.i.i44.i
  %10 = load %struct.signature** %arrayidx.i.i45.i, align 4
  %name.i46.i = getelementptr inbounds %struct.signature* %10, i32 0, i32 0
  %11 = load i8** %name.i46.i, align 4
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([22 x i8]* @.str49, i32 0, i32 0), i8* %11) #1
  %12 = load i32* %4, align 4
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([30 x i8]* @.str50, i32 0, i32 0), i32 %12) #1
  tail call fastcc void @misc_Error() #1
  unreachable

if.else.i:                                        ; preds = %if.then.i
  store i32 %Arity, i32* %4, align 4
  store i32 1, i32* %2, align 4
  br label %if.end14

while.end.i:                                      ; preds = %while.cond.i, %if.then8
  %sub.i.i39.i = sub nsw i32 0, %call236
  %13 = load i32* @symbol_TYPESTATBITS, align 4
  %shr.i.i40.i = ashr i32 %sub.i.i39.i, %13
  %14 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i41.i = getelementptr inbounds %struct.signature** %14, i32 %shr.i.i40.i
  %15 = load %struct.signature** %arrayidx.i.i41.i, align 4
  %arity.i42.i = getelementptr inbounds %struct.signature* %15, i32 0, i32 3
  %16 = load i32* %arity.i42.i, align 4
  %cmp15.i = icmp eq i32 %16, %Arity
  br i1 %cmp15.i, label %if.end14, label %if.then16.i

if.then16.i:                                      ; preds = %while.end.i
  %17 = load %struct._IO_FILE** @stdout, align 4
  %call17.i = tail call i32 @fflush(%struct._IO_FILE* %17) #1
  %18 = load i32* @dfg_LINENUMBER, align 4
  %19 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i38.i = getelementptr inbounds %struct.signature** %19, i32 %shr.i.i40.i
  %20 = load %struct.signature** %arrayidx.i.i38.i, align 4
  %name.i.i = getelementptr inbounds %struct.signature* %20, i32 0, i32 0
  %21 = load i8** %name.i.i, align 4
  %arity.i.i = getelementptr inbounds %struct.signature* %20, i32 0, i32 3
  %22 = load i32* %arity.i.i, align 4
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([50 x i8]* @.str51, i32 0, i32 0), i32 %18, i8* %21, i32 %22) #1
  tail call fastcc void @misc_Error() #1
  unreachable

if.else:                                          ; preds = %if.end6
  %cmp9 = icmp eq i32 %Arity, 0
  br i1 %cmp9, label %if.end12, label %if.then10

if.then10:                                        ; preds = %if.else
  %23 = load %struct._IO_FILE** @stdout, align 4
  %call11 = tail call i32 @fflush(%struct._IO_FILE* %23) #1
  %24 = load i32* @dfg_LINENUMBER, align 4
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([33 x i8]* @.str45, i32 0, i32 0), i32 %24, i8* %Name) #1
  tail call fastcc void @misc_Error()
  unreachable

if.end12:                                         ; preds = %if.else
  %scan.064.i = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %cmp.i65.i = icmp eq %struct.LIST_HELP* %scan.064.i, null
  br i1 %cmp.i65.i, label %if.else.i33, label %while.body.i28

while.body.i28:                                   ; preds = %if.end12, %while.end.i31
  %scan.066.i = phi %struct.LIST_HELP* [ %scan.0.i29, %while.end.i31 ], [ %scan.064.i, %if.end12 ]
  %scan.0.idx.i26 = getelementptr %struct.LIST_HELP* %scan.066.i, i32 0, i32 1
  %scan.0.idx.val.i27 = load i8** %scan.0.idx.i26, align 4
  %25 = bitcast i8* %scan.0.idx.val.i27 to %struct.LIST_HELP*
  %cmp.i6062.i = icmp eq i8* %scan.0.idx.val.i27, null
  br i1 %cmp.i6062.i, label %while.end.i31, label %land.rhs9.i

land.rhs9.i:                                      ; preds = %while.body.i28, %while.body15.i
  %scan2.163.i = phi %struct.LIST_HELP* [ %scan2.1.idx48.val.i, %while.body15.i ], [ %25, %while.body.i28 ]
  %scan2.1.idx.i = getelementptr %struct.LIST_HELP* %scan2.163.i, i32 0, i32 1
  %scan2.1.idx.val.i = load i8** %scan2.1.idx.i, align 4
  %.idx49.i = bitcast i8* %scan2.1.idx.val.i to i8**
  %.idx49.val.i = load i8** %.idx49.i, align 4
  %call.i57.i = tail call i32 @strcmp(i8* %.idx49.val.i, i8* %Name) #1
  %cmp.i58.i = icmp eq i32 %call.i57.i, 0
  br i1 %cmp.i58.i, label %while.end.i31, label %while.body15.i

while.body15.i:                                   ; preds = %land.rhs9.i
  %scan2.1.idx48.i = getelementptr %struct.LIST_HELP* %scan2.163.i, i32 0, i32 0
  %scan2.1.idx48.val.i = load %struct.LIST_HELP** %scan2.1.idx48.i, align 4
  %cmp.i60.i = icmp eq %struct.LIST_HELP* %scan2.1.idx48.val.i, null
  br i1 %cmp.i60.i, label %while.end.i31, label %land.rhs9.i

while.end.i31:                                    ; preds = %while.body15.i, %land.rhs9.i, %while.body.i28
  %scan2.1.lcssa.i = phi %struct.LIST_HELP* [ %25, %while.body.i28 ], [ null, %while.body15.i ], [ %scan2.163.i, %land.rhs9.i ]
  %scan.0.idx47.i = getelementptr %struct.LIST_HELP* %scan.066.i, i32 0, i32 0
  %scan.0.i29 = load %struct.LIST_HELP** %scan.0.idx47.i, align 4
  %cmp.i.i30 = icmp ne %struct.LIST_HELP* %scan.0.i29, null
  %cmp.i53.i = icmp eq %struct.LIST_HELP* %scan2.1.lcssa.i, null
  %or.cond.i = and i1 %cmp.i.i30, %cmp.i53.i
  br i1 %or.cond.i, label %while.body.i28, label %while.end18.i

while.end18.i:                                    ; preds = %while.end.i31
  br i1 %cmp.i53.i, label %if.else.i33, label %if.then.i32

if.then.i32:                                      ; preds = %while.end18.i
  tail call void @string_StringFree(i8* %Name) #1
  %scan2.0.idx.i = getelementptr %struct.LIST_HELP* %scan2.1.lcssa.i, i32 0, i32 1
  %scan2.0.idx.val.i = load i8** %scan2.0.idx.i, align 4
  br label %dfg_VarLookup.exit

if.else.i33:                                      ; preds = %while.end18.i, %if.end12
  %.b.i = load i1* @dfg_VARDECL, align 1
  br i1 %.b.i, label %if.then24.i, label %if.else31.i

if.then24.i:                                      ; preds = %if.else.i33
  %call.i52.i = tail call i8* @memory_Malloc(i32 8) #1
  %name.i = bitcast i8* %call.i52.i to i8**
  store i8* %Name, i8** %name.i, align 4
  %26 = load i32* @symbol_STANDARDVARCOUNTER, align 4
  %inc.i.i = add nsw i32 %26, 1
  store i32 %inc.i.i, i32* @symbol_STANDARDVARCOUNTER, align 4
  %symbol27.i = getelementptr inbounds i8* %call.i52.i, i32 4
  %27 = bitcast i8* %symbol27.i to i32*
  store i32 %inc.i.i, i32* %27, align 4
  %28 = load %struct.LIST_HELP** @dfg_VARLIST, align 4
  %.idx.i = getelementptr %struct.LIST_HELP* %28, i32 0, i32 1
  %.idx.val.i = load i8** %.idx.i, align 4
  %29 = bitcast i8* %.idx.val.i to %struct.LIST_HELP*
  %call.i.i = tail call i8* @memory_Malloc(i32 8) #1
  %car.i51.i = getelementptr inbounds i8* %call.i.i, i32 4
  %30 = bitcast i8* %car.i51.i to i8**
  store i8* %call.i52.i, i8** %30, align 4
  %cdr.i.i = bitcast i8* %call.i.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* %29, %struct.LIST_HELP** %cdr.i.i, align 4
  store i8* %call.i.i, i8** %.idx.i, align 4
  br label %dfg_VarLookup.exit

if.else31.i:                                      ; preds = %if.else.i33
  %31 = load %struct._IO_FILE** @stdout, align 4
  %call32.i = tail call i32 @fflush(%struct._IO_FILE* %31) #1
  %32 = load i32* @dfg_LINENUMBER, align 4
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([30 x i8]* @.str46, i32 0, i32 0), i32 %32, i8* %Name) #1
  tail call fastcc void @misc_Error() #1
  unreachable

dfg_VarLookup.exit:                               ; preds = %if.then.i32, %if.then24.i
  %call.i52.pn.i = phi i8* [ %call.i52.i, %if.then24.i ], [ %scan2.0.idx.val.i, %if.then.i32 ]
  %symbol.0.in.in.i = getelementptr i8* %call.i52.pn.i, i32 4
  %symbol.0.in.i = bitcast i8* %symbol.0.in.in.i to i32*
  %symbol.0.i = load i32* %symbol.0.in.i, align 4
  br label %if.end14

if.end14:                                         ; preds = %while.end.i, %if.else.i, %if.then3.i, %dfg_VarLookup.exit
  %symbol.0 = phi i32 [ %symbol.0.i, %dfg_VarLookup.exit ], [ %call236, %if.then3.i ], [ %call236, %if.else.i ], [ %call236, %while.end.i ]
  ret i32 %symbol.0
}

declare %struct.LIST_HELP* @list_NReverse(%struct.LIST_HELP*) #2

declare %struct.term* @term_Create(i32, %struct.LIST_HELP*) #2

declare i8* @string_IntToString(i32) #2

declare i8* @string_StringCopy(i8*) #2

; Function Attrs: nounwind
declare i32 @fflush(%struct._IO_FILE* nocapture) #0

declare void @misc_UserErrorReport(i8*, ...) #2

; Function Attrs: inlinehint noreturn nounwind
define internal fastcc void @misc_Error() #3 {
entry:
  %0 = load %struct._IO_FILE** @stderr, align 4
  %call = tail call i32 @fflush(%struct._IO_FILE* %0) #1
  %1 = load %struct._IO_FILE** @stdout, align 4
  %call1 = tail call i32 @fflush(%struct._IO_FILE* %1) #1
  %2 = load %struct._IO_FILE** @stderr, align 4
  %call2 = tail call i32 @fflush(%struct._IO_FILE* %2) #1
  tail call void @exit(i32 1) #7
  unreachable
}

declare i32 @clause_GetOriginFromString(i8*) #2

declare void @term_Delete(%struct.term*) #2

declare i32 @string_StringToInt(i8*, i32, i32*) #2

declare i32 @symbol_Lookup(i8*) #2

declare i32 @flag_Id(i8*) #2

; Function Attrs: nounwind readonly
declare i32 @strlen(i8* nocapture) #4

; Function Attrs: nounwind
declare i8* @stpcpy(i8*, i8* nocapture) #0

; Function Attrs: noreturn nounwind
define void @dfg_error(i8* %s) #5 {
entry:
  %0 = load %struct._IO_FILE** @stdout, align 4
  %call = tail call i32 @fflush(%struct._IO_FILE* %0) #1
  %1 = load i32* @dfg_LINENUMBER, align 4
  tail call void (i8*, ...)* @misc_UserErrorReport(i8* getelementptr inbounds ([15 x i8]* @.str22, i32 0, i32 0), i32 %1, i8* %s) #1
  tail call fastcc void @misc_Error()
  unreachable
}

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

; Function Attrs: nounwind
define void @dfg_Free() #0 {
entry:
  %0 = load i8** @dfg_DESC.0, align 4
  %cmp = icmp eq i8* %0, null
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void @string_StringFree(i8* %0) #1
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %1 = load i8** @dfg_DESC.1, align 4
  %cmp1 = icmp eq i8* %1, null
  br i1 %cmp1, label %if.end3, label %if.then2

if.then2:                                         ; preds = %if.end
  tail call void @string_StringFree(i8* %1) #1
  br label %if.end3

if.end3:                                          ; preds = %if.end, %if.then2
  %2 = load i8** @dfg_DESC.2, align 4
  %cmp4 = icmp eq i8* %2, null
  br i1 %cmp4, label %if.end6, label %if.then5

if.then5:                                         ; preds = %if.end3
  tail call void @string_StringFree(i8* %2) #1
  br label %if.end6

if.end6:                                          ; preds = %if.end3, %if.then5
  %3 = load i8** @dfg_DESC.3, align 4
  %cmp7 = icmp eq i8* %3, null
  br i1 %cmp7, label %if.end9, label %if.then8

if.then8:                                         ; preds = %if.end6
  tail call void @string_StringFree(i8* %3) #1
  br label %if.end9

if.end9:                                          ; preds = %if.end6, %if.then8
  %4 = load i8** @dfg_DESC.5, align 4
  %cmp10 = icmp eq i8* %4, null
  br i1 %cmp10, label %if.end12, label %if.then11

if.then11:                                        ; preds = %if.end9
  tail call void @string_StringFree(i8* %4) #1
  br label %if.end12

if.end12:                                         ; preds = %if.end9, %if.then11
  %5 = load i8** @dfg_DESC.6, align 4
  %cmp13 = icmp eq i8* %5, null
  br i1 %cmp13, label %if.end15, label %if.then14

if.then14:                                        ; preds = %if.end12
  tail call void @string_StringFree(i8* %5) #1
  br label %if.end15

if.end15:                                         ; preds = %if.end12, %if.then14
  ret void
}

; Function Attrs: nounwind readonly
define i8* @dfg_ProblemName() #4 {
entry:
  %0 = load i8** @dfg_DESC.0, align 4
  ret i8* %0
}

; Function Attrs: nounwind readonly
define i8* @dfg_ProblemAuthor() #4 {
entry:
  %0 = load i8** @dfg_DESC.1, align 4
  ret i8* %0
}

; Function Attrs: nounwind readonly
define i8* @dfg_ProblemVersion() #4 {
entry:
  %0 = load i8** @dfg_DESC.2, align 4
  ret i8* %0
}

; Function Attrs: nounwind readonly
define i8* @dfg_ProblemLogic() #4 {
entry:
  %0 = load i8** @dfg_DESC.3, align 4
  ret i8* %0
}

; Function Attrs: nounwind readonly
define i32 @dfg_ProblemStatus() #4 {
entry:
  %0 = load i32* @dfg_DESC.4, align 4
  ret i32 %0
}

; Function Attrs: nounwind
define i8* @dfg_ProblemStatusString() #0 {
entry:
  %0 = load i32* @dfg_DESC.4, align 4
  switch i32 %0, label %sw.default [
    i32 0, label %sw.epilog
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
  ]

sw.bb1:                                           ; preds = %entry
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  %1 = load %struct._IO_FILE** @stdout, align 4
  %call = tail call i32 @fflush(%struct._IO_FILE* %1) #1
  %2 = load %struct._IO_FILE** @stderr, align 4
  %call3 = tail call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %2, i8* getelementptr inbounds ([31 x i8]* @.str27, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str28, i32 0, i32 0), i32 1025) #1
  tail call void (i8*, ...)* @misc_ErrorReport(i8* getelementptr inbounds ([47 x i8]* @.str29, i32 0, i32 0)) #1
  %3 = load %struct._IO_FILE** @stderr, align 4
  %4 = tail call i32 @fwrite(i8* getelementptr inbounds ([133 x i8]* @.str30, i32 0, i32 0), i32 132, i32 1, %struct._IO_FILE* %3)
  tail call fastcc void @misc_DumpCore()
  unreachable

sw.epilog:                                        ; preds = %entry, %sw.bb2, %sw.bb1
  %result.0 = phi i8* [ getelementptr inbounds ([8 x i8]* @.str26, i32 0, i32 0), %sw.bb2 ], [ getelementptr inbounds ([14 x i8]* @.str25, i32 0, i32 0), %sw.bb1 ], [ getelementptr inbounds ([12 x i8]* @.str24, i32 0, i32 0), %entry ]
  ret i8* %result.0
}

; Function Attrs: nounwind
declare i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture, ...) #0

declare void @misc_ErrorReport(i8*, ...) #2

; Function Attrs: nounwind
declare i32 @fputs(i8* nocapture, %struct._IO_FILE* nocapture) #0

; Function Attrs: inlinehint noreturn nounwind
define internal fastcc void @misc_DumpCore() #3 {
entry:
  %0 = load %struct._IO_FILE** @stderr, align 4
  %1 = tail call i32 @fwrite(i8* getelementptr inbounds ([3 x i8]* @.str59, i32 0, i32 0), i32 2, i32 1, %struct._IO_FILE* %0)
  %2 = load %struct._IO_FILE** @stderr, align 4
  %call1 = tail call i32 @fflush(%struct._IO_FILE* %2) #1
  %3 = load %struct._IO_FILE** @stdout, align 4
  %call2 = tail call i32 @fflush(%struct._IO_FILE* %3) #1
  %4 = load %struct._IO_FILE** @stderr, align 4
  %call3 = tail call i32 @fflush(%struct._IO_FILE* %4) #1
  tail call void @abort() #7
  unreachable
}

; Function Attrs: nounwind readonly
define i8* @dfg_ProblemDescription() #4 {
entry:
  %0 = load i8** @dfg_DESC.5, align 4
  ret i8* %0
}

; Function Attrs: nounwind readonly
define i8* @dfg_ProblemDate() #4 {
entry:
  %0 = load i8** @dfg_DESC.6, align 4
  ret i8* %0
}

; Function Attrs: nounwind
define void @dfg_FPrintDescription(%struct._IO_FILE* %File) #0 {
entry:
  %0 = tail call i32 @fwrite(i8* getelementptr inbounds ([30 x i8]* @.str31, i32 0, i32 0), i32 29, i32 1, %struct._IO_FILE* %File)
  %1 = load i8** @dfg_DESC.0, align 4
  %cmp = icmp eq i8* %1, null
  br i1 %cmp, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %call1 = tail call i32 @fputs(i8* %1, %struct._IO_FILE* %File) #1
  br label %if.end

if.else:                                          ; preds = %entry
  %2 = tail call i32 @fwrite(i8* getelementptr inbounds ([6 x i8]* @.str32, i32 0, i32 0), i32 5, i32 1, %struct._IO_FILE* %File)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %3 = tail call i32 @fwrite(i8* getelementptr inbounds ([13 x i8]* @.str33, i32 0, i32 0), i32 12, i32 1, %struct._IO_FILE* %File)
  %4 = load i8** @dfg_DESC.1, align 4
  %cmp4 = icmp eq i8* %4, null
  br i1 %cmp4, label %if.else7, label %if.then5

if.then5:                                         ; preds = %if.end
  %call6 = tail call i32 @fputs(i8* %4, %struct._IO_FILE* %File) #1
  br label %if.end9

if.else7:                                         ; preds = %if.end
  %5 = tail call i32 @fwrite(i8* getelementptr inbounds ([6 x i8]* @.str32, i32 0, i32 0), i32 5, i32 1, %struct._IO_FILE* %File)
  br label %if.end9

if.end9:                                          ; preds = %if.else7, %if.then5
  %6 = tail call i32 @fwrite(i8* getelementptr inbounds ([4 x i8]* @.str34, i32 0, i32 0), i32 3, i32 1, %struct._IO_FILE* %File)
  %7 = load i8** @dfg_DESC.2, align 4
  %cmp11 = icmp eq i8* %7, null
  br i1 %cmp11, label %if.end16, label %if.then12

if.then12:                                        ; preds = %if.end9
  %8 = tail call i32 @fwrite(i8* getelementptr inbounds ([11 x i8]* @.str35, i32 0, i32 0), i32 10, i32 1, %struct._IO_FILE* %File)
  %9 = load i8** @dfg_DESC.2, align 4
  %call14 = tail call i32 @fputs(i8* %9, %struct._IO_FILE* %File) #1
  %10 = tail call i32 @fwrite(i8* getelementptr inbounds ([4 x i8]* @.str34, i32 0, i32 0), i32 3, i32 1, %struct._IO_FILE* %File)
  br label %if.end16

if.end16:                                         ; preds = %if.end9, %if.then12
  %11 = load i8** @dfg_DESC.3, align 4
  %cmp17 = icmp eq i8* %11, null
  br i1 %cmp17, label %if.end22, label %if.then18

if.then18:                                        ; preds = %if.end16
  %12 = tail call i32 @fwrite(i8* getelementptr inbounds ([9 x i8]* @.str36, i32 0, i32 0), i32 8, i32 1, %struct._IO_FILE* %File)
  %13 = load i8** @dfg_DESC.3, align 4
  %call20 = tail call i32 @fputs(i8* %13, %struct._IO_FILE* %File) #1
  %14 = tail call i32 @fwrite(i8* getelementptr inbounds ([4 x i8]* @.str34, i32 0, i32 0), i32 3, i32 1, %struct._IO_FILE* %File)
  br label %if.end22

if.end22:                                         ; preds = %if.end16, %if.then18
  %15 = tail call i32 @fwrite(i8* getelementptr inbounds ([10 x i8]* @.str37, i32 0, i32 0), i32 9, i32 1, %struct._IO_FILE* %File)
  %16 = load i32* @dfg_DESC.4, align 4
  switch i32 %16, label %sw.default.i [
    i32 0, label %dfg_ProblemStatusString.exit
    i32 1, label %sw.bb1.i
    i32 2, label %sw.bb2.i
  ]

sw.bb1.i:                                         ; preds = %if.end22
  br label %dfg_ProblemStatusString.exit

sw.bb2.i:                                         ; preds = %if.end22
  br label %dfg_ProblemStatusString.exit

sw.default.i:                                     ; preds = %if.end22
  %17 = load %struct._IO_FILE** @stdout, align 4
  %call.i = tail call i32 @fflush(%struct._IO_FILE* %17) #1
  %18 = load %struct._IO_FILE** @stderr, align 4
  %call3.i = tail call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %18, i8* getelementptr inbounds ([31 x i8]* @.str27, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str28, i32 0, i32 0), i32 1025) #1
  tail call void (i8*, ...)* @misc_ErrorReport(i8* getelementptr inbounds ([47 x i8]* @.str29, i32 0, i32 0)) #1
  %19 = load %struct._IO_FILE** @stderr, align 4
  %20 = tail call i32 @fwrite(i8* getelementptr inbounds ([133 x i8]* @.str30, i32 0, i32 0), i32 132, i32 1, %struct._IO_FILE* %19) #1
  tail call fastcc void @misc_DumpCore() #1
  unreachable

dfg_ProblemStatusString.exit:                     ; preds = %if.end22, %sw.bb1.i, %sw.bb2.i
  %result.0.i = phi i8* [ getelementptr inbounds ([8 x i8]* @.str26, i32 0, i32 0), %sw.bb2.i ], [ getelementptr inbounds ([14 x i8]* @.str25, i32 0, i32 0), %sw.bb1.i ], [ getelementptr inbounds ([12 x i8]* @.str24, i32 0, i32 0), %if.end22 ]
  %call25 = tail call i32 @fputs(i8* %result.0.i, %struct._IO_FILE* %File) #1
  %21 = tail call i32 @fwrite(i8* getelementptr inbounds ([18 x i8]* @.str38, i32 0, i32 0), i32 17, i32 1, %struct._IO_FILE* %File)
  %22 = load i8** @dfg_DESC.5, align 4
  %cmp27 = icmp eq i8* %22, null
  br i1 %cmp27, label %if.else30, label %if.then28

if.then28:                                        ; preds = %dfg_ProblemStatusString.exit
  %call29 = tail call i32 @fputs(i8* %22, %struct._IO_FILE* %File) #1
  br label %if.end32

if.else30:                                        ; preds = %dfg_ProblemStatusString.exit
  %23 = tail call i32 @fwrite(i8* getelementptr inbounds ([6 x i8]* @.str32, i32 0, i32 0), i32 5, i32 1, %struct._IO_FILE* %File)
  br label %if.end32

if.end32:                                         ; preds = %if.else30, %if.then28
  %24 = tail call i32 @fwrite(i8* getelementptr inbounds ([4 x i8]* @.str34, i32 0, i32 0), i32 3, i32 1, %struct._IO_FILE* %File)
  %25 = load i8** @dfg_DESC.6, align 4
  %cmp34 = icmp eq i8* %25, null
  br i1 %cmp34, label %if.end39, label %if.then35

if.then35:                                        ; preds = %if.end32
  %26 = tail call i32 @fwrite(i8* getelementptr inbounds ([8 x i8]* @.str39, i32 0, i32 0), i32 7, i32 1, %struct._IO_FILE* %File)
  %27 = load i8** @dfg_DESC.6, align 4
  %call37 = tail call i32 @fputs(i8* %27, %struct._IO_FILE* %File) #1
  %28 = tail call i32 @fwrite(i8* getelementptr inbounds ([4 x i8]* @.str34, i32 0, i32 0), i32 3, i32 1, %struct._IO_FILE* %File)
  br label %if.end39

if.end39:                                         ; preds = %if.end32, %if.then35
  %29 = tail call i32 @fwrite(i8* getelementptr inbounds ([13 x i8]* @.str40, i32 0, i32 0), i32 12, i32 1, %struct._IO_FILE* %File)
  ret void
}

; Function Attrs: nounwind
define %struct.LIST_HELP* @dfg_DFGParser(%struct._IO_FILE* %File, i32* %Flags, i32* %Precedence, %struct.LIST_HELP** nocapture %Axioms, %struct.LIST_HELP** nocapture %Conjectures, %struct.LIST_HELP** nocapture %SortDecl, %struct.LIST_HELP** nocapture %UserDefinedPrecedence) #0 {
entry:
  store %struct._IO_FILE* %File, %struct._IO_FILE** @dfg_in, align 4
  store i32 1, i32* @dfg_LINENUMBER, align 4
  store i32 1, i32* @dfg_IGNORETEXT, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_AXIOMLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_CONJECLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_SORTDECLLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_USERPRECEDENCE, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_CONCLAUSES, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_PROOFLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_TERMLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_VARLIST, align 4
  store i1 false, i1* @dfg_VARDECL, align 1
  store i32 0, i32* @dfg_IGNORE, align 4
  store i32* %Flags, i32** @dfg_FLAGS, align 4
  store i32* %Precedence, i32** @dfg_PRECEDENCE, align 4
  store i8* null, i8** @dfg_DESC.0, align 4
  store i8* null, i8** @dfg_DESC.1, align 4
  store i8* null, i8** @dfg_DESC.2, align 4
  store i8* null, i8** @dfg_DESC.3, align 4
  store i32 2, i32* @dfg_DESC.4, align 4
  store i8* null, i8** @dfg_DESC.5, align 4
  store i8* null, i8** @dfg_DESC.6, align 4
  %call1 = tail call i32 @dfg_parse()
  %0 = load %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  %cmp.i13.i = icmp eq %struct.LIST_HELP* %0, null
  br i1 %cmp.i13.i, label %for.cond.preheader, label %while.body.lr.ph.i

while.body.lr.ph.i:                               ; preds = %entry
  %1 = load i32* @symbol_TYPESTATBITS, align 4
  br label %while.body.i

while.body.i:                                     ; preds = %if.end.i109, %while.body.lr.ph.i
  %2 = phi %struct.LIST_HELP* [ %0, %while.body.lr.ph.i ], [ %L.idx.val.i.i, %if.end.i109 ]
  %.idx.i = getelementptr %struct.LIST_HELP* %2, i32 0, i32 1
  %.idx.val.i = load i8** %.idx.i, align 4
  %symbol.i = bitcast i8* %.idx.val.i to i32*
  %3 = load i32* %symbol.i, align 4
  %arity.i = getelementptr inbounds i8* %.idx.val.i, i32 8
  %4 = bitcast i8* %arity.i to i32*
  %5 = load i32* %4, align 4
  %sub.i.i9.i = sub nsw i32 0, %3
  %shr.i.i10.i = ashr i32 %sub.i.i9.i, %1
  %6 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i11.i = getelementptr inbounds %struct.signature** %6, i32 %shr.i.i10.i
  %7 = load %struct.signature** %arrayidx.i.i11.i, align 4
  %arity.i12.i = getelementptr inbounds %struct.signature* %7, i32 0, i32 3
  %8 = load i32* %arity.i12.i, align 4
  %cmp.i = icmp eq i32 %5, %8
  br i1 %cmp.i, label %if.end.i109, label %if.then.i

if.then.i:                                        ; preds = %while.body.i
  store i32 %5, i32* %arity.i12.i, align 4
  br label %if.end.i109

if.end.i109:                                      ; preds = %if.then.i, %while.body.i
  %9 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 12), align 4
  %total_size.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %9, i32 0, i32 4
  %10 = load i32* %total_size.i.i.i, align 4
  %11 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i = add i32 %11, %10
  store i32 %add24.i.i.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %9, i32 0, i32 0
  %12 = load i8** %free.i.i.i, align 4
  %.c.i.i = ptrtoint i8* %12 to i32
  store i32 %.c.i.i, i32* %symbol.i, align 4
  %13 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 12), align 4
  %free27.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %13, i32 0, i32 0
  store i8* %.idx.val.i, i8** %free27.i.i.i, align 4
  %14 = load %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  %L.idx.i.i = getelementptr %struct.LIST_HELP* %14, i32 0, i32 0
  %L.idx.val.i.i = load %struct.LIST_HELP** %L.idx.i.i, align 4
  %15 = bitcast %struct.LIST_HELP* %14 to i8*
  %16 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %16, i32 0, i32 4
  %17 = load i32* %total_size.i.i.i.i, align 4
  %18 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i = add i32 %18, %17
  store i32 %add24.i.i.i.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %16, i32 0, i32 0
  %19 = load i8** %free.i.i.i.i, align 4
  %.c.i.i.i = bitcast i8* %19 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i, %struct.LIST_HELP** %L.idx.i.i, align 4
  %20 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %20, i32 0, i32 0
  store i8* %15, i8** %free27.i.i.i.i, align 4
  store %struct.LIST_HELP* %L.idx.val.i.i, %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  %cmp.i.i108 = icmp eq %struct.LIST_HELP* %L.idx.val.i.i, null
  br i1 %cmp.i.i108, label %for.cond.preheader, label %while.body.i

for.cond.preheader:                               ; preds = %if.end.i109, %entry
  %scan.0127 = load %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  %cmp.i115128 = icmp eq %struct.LIST_HELP* %scan.0127, null
  br i1 %cmp.i115128, label %for.end, label %for.body

for.body:                                         ; preds = %for.cond.preheader, %if.end
  %scan.0129 = phi %struct.LIST_HELP* [ %scan.0, %if.end ], [ %scan.0127, %for.cond.preheader ]
  %scan.0.idx = getelementptr %struct.LIST_HELP* %scan.0129, i32 0, i32 1
  %scan.0.idx.val = load i8** %scan.0.idx, align 4
  %.idx59 = bitcast i8* %scan.0.idx.val to %struct.LIST_HELP**
  %.idx59.val = load %struct.LIST_HELP** %.idx59, align 4
  %21 = bitcast %struct.LIST_HELP* %.idx59.val to %struct.term*
  %call5 = tail call %struct.CLAUSE_HELP* @dfg_CreateClauseFromTerm(%struct.term* %21, i32 1, i32* %Flags, i32* %Precedence)
  %22 = bitcast %struct.CLAUSE_HELP* %call5 to i8*
  store i8* %22, i8** %scan.0.idx, align 4
  %.idx63 = getelementptr i8* %scan.0.idx.val, i32 4
  %23 = bitcast i8* %.idx63 to i8**
  %.idx63.val = load i8** %23, align 4
  %cmp = icmp eq i8* %.idx63.val, null
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  tail call void @string_StringFree(i8* %.idx63.val) #1
  br label %if.end

if.end:                                           ; preds = %for.body, %if.then
  %24 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i118 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %24, i32 0, i32 4
  %25 = load i32* %total_size.i.i.i118, align 4
  %26 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i119 = add i32 %26, %25
  store i32 %add24.i.i.i119, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i120 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %24, i32 0, i32 0
  %27 = load i8** %free.i.i.i120, align 4
  %.c.i.i121 = bitcast i8* %27 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i121, %struct.LIST_HELP** %.idx59, align 4
  %28 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i122 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %28, i32 0, i32 0
  store i8* %scan.0.idx.val, i8** %free27.i.i.i122, align 4
  %scan.0.idx58 = getelementptr %struct.LIST_HELP* %scan.0129, i32 0, i32 0
  %scan.0 = load %struct.LIST_HELP** %scan.0.idx58, align 4
  %cmp.i115 = icmp eq %struct.LIST_HELP* %scan.0, null
  br i1 %cmp.i115, label %for.cond.for.end_crit_edge, label %for.body

for.cond.for.end_crit_edge:                       ; preds = %if.end
  %.pre = load %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %for.cond.preheader
  %29 = phi %struct.LIST_HELP* [ %.pre, %for.cond.for.end_crit_edge ], [ null, %for.cond.preheader ]
  %call9 = tail call %struct.LIST_HELP* @list_PointerDeleteElement(%struct.LIST_HELP* %29, i8* null) #1
  store %struct.LIST_HELP* %call9, %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  %scan.1124 = load %struct.LIST_HELP** @dfg_CONCLAUSES, align 4
  %cmp.i116125 = icmp eq %struct.LIST_HELP* %scan.1124, null
  br i1 %cmp.i116125, label %for.end25, label %for.body14

for.body14:                                       ; preds = %for.end, %if.end22
  %scan.1126 = phi %struct.LIST_HELP* [ %scan.1, %if.end22 ], [ %scan.1124, %for.end ]
  %scan.1.idx = getelementptr %struct.LIST_HELP* %scan.1126, i32 0, i32 1
  %scan.1.idx.val = load i8** %scan.1.idx, align 4
  %.idx = bitcast i8* %scan.1.idx.val to %struct.LIST_HELP**
  %.idx.val = load %struct.LIST_HELP** %.idx, align 4
  %30 = bitcast %struct.LIST_HELP* %.idx.val to %struct.term*
  %call17 = tail call %struct.CLAUSE_HELP* @dfg_CreateClauseFromTerm(%struct.term* %30, i32 0, i32* %Flags, i32* %Precedence)
  %31 = bitcast %struct.CLAUSE_HELP* %call17 to i8*
  store i8* %31, i8** %scan.1.idx, align 4
  %.idx61 = getelementptr i8* %scan.1.idx.val, i32 4
  %32 = bitcast i8* %.idx61 to i8**
  %.idx61.val = load i8** %32, align 4
  %cmp19 = icmp eq i8* %.idx61.val, null
  br i1 %cmp19, label %if.end22, label %if.then20

if.then20:                                        ; preds = %for.body14
  tail call void @string_StringFree(i8* %.idx61.val) #1
  br label %if.end22

if.end22:                                         ; preds = %for.body14, %if.then20
  %33 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i110 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %33, i32 0, i32 4
  %34 = load i32* %total_size.i.i.i110, align 4
  %35 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i111 = add i32 %35, %34
  store i32 %add24.i.i.i111, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i112 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %33, i32 0, i32 0
  %36 = load i8** %free.i.i.i112, align 4
  %.c.i.i113 = bitcast i8* %36 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i113, %struct.LIST_HELP** %.idx, align 4
  %37 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i114 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %37, i32 0, i32 0
  store i8* %scan.1.idx.val, i8** %free27.i.i.i114, align 4
  %scan.1.idx57 = getelementptr %struct.LIST_HELP* %scan.1126, i32 0, i32 0
  %scan.1 = load %struct.LIST_HELP** %scan.1.idx57, align 4
  %cmp.i116 = icmp eq %struct.LIST_HELP* %scan.1, null
  br i1 %cmp.i116, label %for.cond10.for.end25_crit_edge, label %for.body14

for.cond10.for.end25_crit_edge:                   ; preds = %if.end22
  %.pre130 = load %struct.LIST_HELP** @dfg_CONCLAUSES, align 4
  br label %for.end25

for.end25:                                        ; preds = %for.cond10.for.end25_crit_edge, %for.end
  %38 = phi %struct.LIST_HELP* [ %.pre130, %for.cond10.for.end25_crit_edge ], [ null, %for.end ]
  %call26 = tail call %struct.LIST_HELP* @list_PointerDeleteElement(%struct.LIST_HELP* %38, i8* null) #1
  store %struct.LIST_HELP* %call26, %struct.LIST_HELP** @dfg_CONCLAUSES, align 4
  %39 = load %struct.LIST_HELP** @dfg_PROOFLIST, align 4
  tail call void @dfg_DeleteProofList(%struct.LIST_HELP* %39)
  %40 = load %struct.LIST_HELP** @dfg_TERMLIST, align 4
  tail call void @list_DeleteWithElement(%struct.LIST_HELP* %40, void (i8*)* bitcast (void (%struct.term*)* @term_Delete to void (i8*)*)) #1
  %41 = load %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  %42 = load %struct.LIST_HELP** @dfg_CONCLAUSES, align 4
  %cmp.i.i97 = icmp eq %struct.LIST_HELP* %41, null
  br i1 %cmp.i.i97, label %list_Nconc.exit107, label %if.end.i99

if.end.i99:                                       ; preds = %for.end25
  %cmp.i18.i98 = icmp eq %struct.LIST_HELP* %42, null
  br i1 %cmp.i18.i98, label %list_Nconc.exit107, label %for.cond.i104

for.cond.i104:                                    ; preds = %if.end.i99, %for.cond.i104
  %List1.addr.0.i100 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i102, %for.cond.i104 ], [ %41, %if.end.i99 ]
  %List1.addr.0.idx15.i101 = getelementptr %struct.LIST_HELP* %List1.addr.0.i100, i32 0, i32 0
  %List1.addr.0.idx15.val.i102 = load %struct.LIST_HELP** %List1.addr.0.idx15.i101, align 4
  %cmp.i16.i103 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i102, null
  br i1 %cmp.i16.i103, label %for.end.i105, label %for.cond.i104

for.end.i105:                                     ; preds = %for.cond.i104
  store %struct.LIST_HELP* %42, %struct.LIST_HELP** %List1.addr.0.idx15.i101, align 4
  br label %list_Nconc.exit107

list_Nconc.exit107:                               ; preds = %for.end25, %if.end.i99, %for.end.i105
  %retval.0.i106 = phi %struct.LIST_HELP* [ %41, %for.end.i105 ], [ %42, %for.end25 ], [ %41, %if.end.i99 ]
  %43 = load %struct.LIST_HELP** %Axioms, align 4
  %44 = load %struct.LIST_HELP** @dfg_AXIOMLIST, align 4
  %cmp.i.i86 = icmp eq %struct.LIST_HELP* %43, null
  br i1 %cmp.i.i86, label %list_Nconc.exit96, label %if.end.i88

if.end.i88:                                       ; preds = %list_Nconc.exit107
  %cmp.i18.i87 = icmp eq %struct.LIST_HELP* %44, null
  br i1 %cmp.i18.i87, label %list_Nconc.exit96, label %for.cond.i93

for.cond.i93:                                     ; preds = %if.end.i88, %for.cond.i93
  %List1.addr.0.i89 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i91, %for.cond.i93 ], [ %43, %if.end.i88 ]
  %List1.addr.0.idx15.i90 = getelementptr %struct.LIST_HELP* %List1.addr.0.i89, i32 0, i32 0
  %List1.addr.0.idx15.val.i91 = load %struct.LIST_HELP** %List1.addr.0.idx15.i90, align 4
  %cmp.i16.i92 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i91, null
  br i1 %cmp.i16.i92, label %for.end.i94, label %for.cond.i93

for.end.i94:                                      ; preds = %for.cond.i93
  store %struct.LIST_HELP* %44, %struct.LIST_HELP** %List1.addr.0.idx15.i90, align 4
  br label %list_Nconc.exit96

list_Nconc.exit96:                                ; preds = %list_Nconc.exit107, %if.end.i88, %for.end.i94
  %retval.0.i95 = phi %struct.LIST_HELP* [ %43, %for.end.i94 ], [ %44, %list_Nconc.exit107 ], [ %43, %if.end.i88 ]
  store %struct.LIST_HELP* %retval.0.i95, %struct.LIST_HELP** %Axioms, align 4
  %45 = load %struct.LIST_HELP** %Conjectures, align 4
  %46 = load %struct.LIST_HELP** @dfg_CONJECLIST, align 4
  %cmp.i.i75 = icmp eq %struct.LIST_HELP* %45, null
  br i1 %cmp.i.i75, label %list_Nconc.exit85, label %if.end.i77

if.end.i77:                                       ; preds = %list_Nconc.exit96
  %cmp.i18.i76 = icmp eq %struct.LIST_HELP* %46, null
  br i1 %cmp.i18.i76, label %list_Nconc.exit85, label %for.cond.i82

for.cond.i82:                                     ; preds = %if.end.i77, %for.cond.i82
  %List1.addr.0.i78 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i80, %for.cond.i82 ], [ %45, %if.end.i77 ]
  %List1.addr.0.idx15.i79 = getelementptr %struct.LIST_HELP* %List1.addr.0.i78, i32 0, i32 0
  %List1.addr.0.idx15.val.i80 = load %struct.LIST_HELP** %List1.addr.0.idx15.i79, align 4
  %cmp.i16.i81 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i80, null
  br i1 %cmp.i16.i81, label %for.end.i83, label %for.cond.i82

for.end.i83:                                      ; preds = %for.cond.i82
  store %struct.LIST_HELP* %46, %struct.LIST_HELP** %List1.addr.0.idx15.i79, align 4
  br label %list_Nconc.exit85

list_Nconc.exit85:                                ; preds = %list_Nconc.exit96, %if.end.i77, %for.end.i83
  %retval.0.i84 = phi %struct.LIST_HELP* [ %45, %for.end.i83 ], [ %46, %list_Nconc.exit96 ], [ %45, %if.end.i77 ]
  store %struct.LIST_HELP* %retval.0.i84, %struct.LIST_HELP** %Conjectures, align 4
  %47 = load %struct.LIST_HELP** %SortDecl, align 4
  %48 = load %struct.LIST_HELP** @dfg_SORTDECLLIST, align 4
  %cmp.i.i64 = icmp eq %struct.LIST_HELP* %47, null
  br i1 %cmp.i.i64, label %list_Nconc.exit74, label %if.end.i66

if.end.i66:                                       ; preds = %list_Nconc.exit85
  %cmp.i18.i65 = icmp eq %struct.LIST_HELP* %48, null
  br i1 %cmp.i18.i65, label %list_Nconc.exit74, label %for.cond.i71

for.cond.i71:                                     ; preds = %if.end.i66, %for.cond.i71
  %List1.addr.0.i67 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i69, %for.cond.i71 ], [ %47, %if.end.i66 ]
  %List1.addr.0.idx15.i68 = getelementptr %struct.LIST_HELP* %List1.addr.0.i67, i32 0, i32 0
  %List1.addr.0.idx15.val.i69 = load %struct.LIST_HELP** %List1.addr.0.idx15.i68, align 4
  %cmp.i16.i70 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i69, null
  br i1 %cmp.i16.i70, label %for.end.i72, label %for.cond.i71

for.end.i72:                                      ; preds = %for.cond.i71
  store %struct.LIST_HELP* %48, %struct.LIST_HELP** %List1.addr.0.idx15.i68, align 4
  br label %list_Nconc.exit74

list_Nconc.exit74:                                ; preds = %list_Nconc.exit85, %if.end.i66, %for.end.i72
  %retval.0.i73 = phi %struct.LIST_HELP* [ %47, %for.end.i72 ], [ %48, %list_Nconc.exit85 ], [ %47, %if.end.i66 ]
  store %struct.LIST_HELP* %retval.0.i73, %struct.LIST_HELP** %SortDecl, align 4
  %49 = load %struct.LIST_HELP** @dfg_USERPRECEDENCE, align 4
  %call31 = tail call %struct.LIST_HELP* @list_NReverse(%struct.LIST_HELP* %49) #1
  %50 = load %struct.LIST_HELP** %UserDefinedPrecedence, align 4
  %51 = load %struct.LIST_HELP** @dfg_USERPRECEDENCE, align 4
  %cmp.i.i = icmp eq %struct.LIST_HELP* %50, null
  br i1 %cmp.i.i, label %list_Nconc.exit, label %if.end.i

if.end.i:                                         ; preds = %list_Nconc.exit74
  %cmp.i18.i = icmp eq %struct.LIST_HELP* %51, null
  br i1 %cmp.i18.i, label %list_Nconc.exit, label %for.cond.i

for.cond.i:                                       ; preds = %if.end.i, %for.cond.i
  %List1.addr.0.i = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i, %for.cond.i ], [ %50, %if.end.i ]
  %List1.addr.0.idx15.i = getelementptr %struct.LIST_HELP* %List1.addr.0.i, i32 0, i32 0
  %List1.addr.0.idx15.val.i = load %struct.LIST_HELP** %List1.addr.0.idx15.i, align 4
  %cmp.i16.i = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i, null
  br i1 %cmp.i16.i, label %for.end.i, label %for.cond.i

for.end.i:                                        ; preds = %for.cond.i
  store %struct.LIST_HELP* %51, %struct.LIST_HELP** %List1.addr.0.idx15.i, align 4
  br label %list_Nconc.exit

list_Nconc.exit:                                  ; preds = %list_Nconc.exit74, %if.end.i, %for.end.i
  %retval.0.i = phi %struct.LIST_HELP* [ %50, %for.end.i ], [ %51, %list_Nconc.exit74 ], [ %50, %if.end.i ]
  store %struct.LIST_HELP* %retval.0.i, %struct.LIST_HELP** %UserDefinedPrecedence, align 4
  ret %struct.LIST_HELP* %retval.0.i106
}

; Function Attrs: nounwind
define %struct.CLAUSE_HELP* @dfg_CreateClauseFromTerm(%struct.term* %Clause, i32 %IsAxiom, i32* %Flags, i32* %Precedence) #0 {
entry:
  %Clause.idx = getelementptr %struct.term* %Clause, i32 0, i32 0
  %Clause.idx.val = load i32* %Clause.idx, align 4
  %0 = load i32* @fol_ALL, align 4
  %cmp = icmp eq i32 %Clause.idx.val, %0
  %Clause.idx66 = getelementptr %struct.term* %Clause, i32 0, i32 2
  %Clause.idx66.val = load %struct.LIST_HELP** %Clause.idx66, align 4
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %Clause.idx66.val.idx = getelementptr %struct.LIST_HELP* %Clause.idx66.val, i32 0, i32 0
  %Clause.idx66.val.idx.val = load %struct.LIST_HELP** %Clause.idx66.val.idx, align 4
  %Clause.idx66.val.idx.val.idx = getelementptr %struct.LIST_HELP* %Clause.idx66.val.idx.val, i32 0, i32 1
  %Clause.idx66.val.idx.val.idx.val = load i8** %Clause.idx66.val.idx.val.idx, align 4
  %call2.idx = getelementptr i8* %Clause.idx66.val.idx.val.idx.val, i32 8
  %1 = bitcast i8* %call2.idx to %struct.LIST_HELP**
  %call2.idx.val = load %struct.LIST_HELP** %1, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %1, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %Clause.idx66, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %literals.0 = phi %struct.LIST_HELP* [ %call2.idx.val, %if.then ], [ %Clause.idx66.val, %if.else ]
  tail call void @term_Delete(%struct.term* %Clause) #1
  %cmp.i7880 = icmp eq %struct.LIST_HELP* %literals.0, null
  br i1 %cmp.i7880, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %if.end
  %car.i = getelementptr inbounds %struct.LIST_HELP* %literals.0, i32 0, i32 1
  %2 = load i32* @symbol_TYPEMASK, align 4
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.inc
  %scan.081 = phi %struct.LIST_HELP* [ %literals.0, %for.body.lr.ph ], [ %scan.0.idx62.val, %for.inc ]
  %scan.0.idx = getelementptr %struct.LIST_HELP* %scan.081, i32 0, i32 1
  %scan.0.idx.val = load i8** %scan.0.idx, align 4
  %3 = bitcast i8* %scan.0.idx.val to %struct.term*
  %.idx = bitcast i8* %scan.0.idx.val to i32*
  %.idx.val = load i32* %.idx, align 4
  %tobool.i = icmp sgt i32 %.idx.val, -1
  br i1 %tobool.i, label %if.else24, label %land.rhs.i

land.rhs.i:                                       ; preds = %for.body
  %sub.i.i = sub nsw i32 0, %.idx.val
  %and.i.i = and i32 %2, %sub.i.i
  %cmp.i = icmp eq i32 %and.i.i, 2
  br i1 %cmp.i, label %if.then13, label %if.else24

if.then13:                                        ; preds = %land.rhs.i
  %4 = load i32* @fol_TRUE, align 4
  %cmp.i.i76 = icmp eq i32 %4, %.idx.val
  br i1 %cmp.i.i76, label %if.then16, label %if.else18

if.then16:                                        ; preds = %if.then13
  %call17 = tail call %struct.LIST_HELP* @list_PointerDeleteElement(%struct.LIST_HELP* %literals.0, i8* null) #1
  tail call void @list_DeleteWithElement(%struct.LIST_HELP* %literals.0, void (i8*)* bitcast (void (%struct.term*)* @term_Delete to void (i8*)*)) #1
  br label %return

if.else18:                                        ; preds = %if.then13
  %5 = load i32* @fol_FALSE, align 4
  %cmp.i.i74 = icmp eq i32 %5, %.idx.val
  br i1 %cmp.i.i74, label %if.then21, label %for.inc

if.then21:                                        ; preds = %if.else18
  tail call void @term_Delete(%struct.term* %3) #1
  store i8* null, i8** %scan.0.idx, align 4
  br label %for.inc

if.else24:                                        ; preds = %for.body, %land.rhs.i
  %.idx63 = getelementptr i8* %scan.0.idx.val, i32 8
  %6 = bitcast i8* %.idx63 to %struct.LIST_HELP**
  %.idx63.val = load %struct.LIST_HELP** %6, align 4
  %.idx63.val.idx = getelementptr %struct.LIST_HELP* %.idx63.val, i32 0, i32 1
  %.idx63.val.idx.val = load i8** %.idx63.val.idx, align 4
  %call25.idx68 = bitcast i8* %.idx63.val.idx.val to i32*
  %call25.idx68.val = load i32* %call25.idx68, align 4
  %7 = load i32* @fol_FALSE, align 4
  %cmp.i.i71 = icmp eq i32 %7, %call25.idx68.val
  br i1 %cmp.i.i71, label %if.then28, label %if.else30

if.then28:                                        ; preds = %if.else24
  %call29 = tail call %struct.LIST_HELP* @list_PointerDeleteElement(%struct.LIST_HELP* %literals.0, i8* null) #1
  tail call void @list_DeleteWithElement(%struct.LIST_HELP* %literals.0, void (i8*)* bitcast (void (%struct.term*)* @term_Delete to void (i8*)*)) #1
  br label %return

if.else30:                                        ; preds = %if.else24
  %8 = load i32* @fol_TRUE, align 4
  %cmp.i.i70 = icmp eq i32 %8, %call25.idx68.val
  br i1 %cmp.i.i70, label %if.then33, label %for.inc

if.then33:                                        ; preds = %if.else30
  tail call void @term_Delete(%struct.term* %3) #1
  store i8* null, i8** %car.i, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.else30, %if.else18, %if.then21, %if.then33
  %scan.0.idx62 = getelementptr %struct.LIST_HELP* %scan.081, i32 0, i32 0
  %scan.0.idx62.val = load %struct.LIST_HELP** %scan.0.idx62, align 4
  %cmp.i78 = icmp eq %struct.LIST_HELP* %scan.0.idx62.val, null
  br i1 %cmp.i78, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %if.end
  %call38 = tail call %struct.LIST_HELP* @list_PointerDeleteElement(%struct.LIST_HELP* %literals.0, i8* null) #1
  %lnot40 = icmp eq i32 %IsAxiom, 0
  %lnot.ext = zext i1 %lnot40 to i32
  %call41 = tail call %struct.CLAUSE_HELP* @clause_CreateFromLiterals(%struct.LIST_HELP* %call38, i32 0, i32 %lnot.ext, i32 0, i32* %Flags, i32* %Precedence) #1
  %cmp.i5.i = icmp eq %struct.LIST_HELP* %call38, null
  br i1 %cmp.i5.i, label %return, label %while.body.i

while.body.i:                                     ; preds = %for.end, %while.body.i
  %L.addr.06.i = phi %struct.LIST_HELP* [ %L.addr.0.idx.val.i, %while.body.i ], [ %call38, %for.end ]
  %L.addr.0.idx.i = getelementptr %struct.LIST_HELP* %L.addr.06.i, i32 0, i32 0
  %L.addr.0.idx.val.i = load %struct.LIST_HELP** %L.addr.0.idx.i, align 4
  %9 = bitcast %struct.LIST_HELP* %L.addr.06.i to i8*
  %10 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %10, i32 0, i32 4
  %11 = load i32* %total_size.i.i.i, align 4
  %12 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i = add i32 %12, %11
  store i32 %add24.i.i.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %10, i32 0, i32 0
  %13 = load i8** %free.i.i.i, align 4
  %.c.i.i = bitcast i8* %13 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i, %struct.LIST_HELP** %L.addr.0.idx.i, align 4
  %14 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %14, i32 0, i32 0
  store i8* %9, i8** %free27.i.i.i, align 4
  %cmp.i.i = icmp eq %struct.LIST_HELP* %L.addr.0.idx.val.i, null
  br i1 %cmp.i.i, label %return, label %while.body.i

return:                                           ; preds = %while.body.i, %for.end, %if.then28, %if.then16
  %retval.0 = phi %struct.CLAUSE_HELP* [ null, %if.then16 ], [ null, %if.then28 ], [ %call41, %for.end ], [ %call41, %while.body.i ]
  ret %struct.CLAUSE_HELP* %retval.0
}

declare %struct.LIST_HELP* @list_PointerDeleteElement(%struct.LIST_HELP*, i8*) #2

; Function Attrs: nounwind
define void @dfg_DeleteProofList(%struct.LIST_HELP* %Proof) #0 {
entry:
  %cmp.i18 = icmp eq %struct.LIST_HELP* %Proof, null
  br i1 %cmp.i18, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %list_Delete.exit
  %Proof.addr.019 = phi %struct.LIST_HELP* [ %L.idx.val.i, %list_Delete.exit ], [ %Proof, %entry ]
  %Proof.addr.0.idx = getelementptr %struct.LIST_HELP* %Proof.addr.019, i32 0, i32 1
  %Proof.addr.0.idx.val = load i8** %Proof.addr.0.idx, align 4
  %.idx = getelementptr i8* %Proof.addr.0.idx.val, i32 4
  %0 = bitcast i8* %.idx to i8**
  %.idx.val = load i8** %0, align 4
  tail call void @string_StringFree(i8* %.idx.val) #1
  %.idx11 = bitcast i8* %Proof.addr.0.idx.val to %struct.LIST_HELP**
  %.idx11.val = load %struct.LIST_HELP** %.idx11, align 4
  %.idx11.val.idx = getelementptr %struct.LIST_HELP* %.idx11.val, i32 0, i32 1
  %.idx11.val.idx.val = load i8** %.idx11.val.idx, align 4
  %1 = bitcast i8* %.idx11.val.idx.val to %struct.term*
  tail call void @term_Delete(%struct.term* %1) #1
  %.idx12.val = load %struct.LIST_HELP** %.idx11, align 4
  %.idx12.val.idx = getelementptr %struct.LIST_HELP* %.idx12.val, i32 0, i32 0
  %.idx12.val.idx.val = load %struct.LIST_HELP** %.idx12.val.idx, align 4
  %.idx12.val.idx.val.idx = getelementptr %struct.LIST_HELP* %.idx12.val.idx.val, i32 0, i32 1
  %.idx12.val.idx.val.idx.val = load i8** %.idx12.val.idx.val.idx, align 4
  %2 = bitcast i8* %.idx12.val.idx.val.idx.val to %struct.LIST_HELP*
  tail call void @list_DeleteWithElement(%struct.LIST_HELP* %2, void (i8*)* @string_StringFree) #1
  %cmp.i5.i = icmp eq i8* %Proof.addr.0.idx.val, null
  br i1 %cmp.i5.i, label %list_Delete.exit, label %while.body.i.preheader

while.body.i.preheader:                           ; preds = %for.body
  %3 = bitcast i8* %Proof.addr.0.idx.val to %struct.LIST_HELP*
  br label %while.body.i

while.body.i:                                     ; preds = %while.body.i.preheader, %while.body.i
  %L.addr.06.i = phi %struct.LIST_HELP* [ %L.addr.0.idx.val.i, %while.body.i ], [ %3, %while.body.i.preheader ]
  %L.addr.0.idx.i = getelementptr %struct.LIST_HELP* %L.addr.06.i, i32 0, i32 0
  %L.addr.0.idx.val.i = load %struct.LIST_HELP** %L.addr.0.idx.i, align 4
  %4 = bitcast %struct.LIST_HELP* %L.addr.06.i to i8*
  %5 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i13 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %5, i32 0, i32 4
  %6 = load i32* %total_size.i.i.i13, align 4
  %7 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i14 = add i32 %7, %6
  store i32 %add24.i.i.i14, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i15 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %5, i32 0, i32 0
  %8 = load i8** %free.i.i.i15, align 4
  %.c.i.i16 = bitcast i8* %8 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i16, %struct.LIST_HELP** %L.addr.0.idx.i, align 4
  %9 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i17 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %9, i32 0, i32 0
  store i8* %4, i8** %free27.i.i.i17, align 4
  %cmp.i.i = icmp eq %struct.LIST_HELP* %L.addr.0.idx.val.i, null
  br i1 %cmp.i.i, label %list_Delete.exit, label %while.body.i

list_Delete.exit:                                 ; preds = %while.body.i, %for.body
  %L.idx.i = getelementptr %struct.LIST_HELP* %Proof.addr.019, i32 0, i32 0
  %L.idx.val.i = load %struct.LIST_HELP** %L.idx.i, align 4
  %10 = bitcast %struct.LIST_HELP* %Proof.addr.019 to i8*
  %11 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %11, i32 0, i32 4
  %12 = load i32* %total_size.i.i.i, align 4
  %13 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i = add i32 %13, %12
  store i32 %add24.i.i.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %11, i32 0, i32 0
  %14 = load i8** %free.i.i.i, align 4
  %.c.i.i = bitcast i8* %14 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i, %struct.LIST_HELP** %L.idx.i, align 4
  %15 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %15, i32 0, i32 0
  store i8* %10, i8** %free27.i.i.i, align 4
  %cmp.i = icmp eq %struct.LIST_HELP* %L.idx.val.i, null
  br i1 %cmp.i, label %for.end, label %for.body

for.end:                                          ; preds = %list_Delete.exit, %entry
  ret void
}

; Function Attrs: nounwind
define %struct.LIST_HELP* @dfg_ProofParser(%struct._IO_FILE* %File, i32* %Flags, i32* %Precedence) #0 {
entry:
  store %struct._IO_FILE* %File, %struct._IO_FILE** @dfg_in, align 4
  store i32 1, i32* @dfg_LINENUMBER, align 4
  store i32 1, i32* @dfg_IGNORETEXT, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_AXIOMLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_CONJECLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_SORTDECLLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_USERPRECEDENCE, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_CONCLAUSES, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_PROOFLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_TERMLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_VARLIST, align 4
  store i1 false, i1* @dfg_VARDECL, align 1
  store i32 0, i32* @dfg_IGNORE, align 4
  store i32* %Flags, i32** @dfg_FLAGS, align 4
  store i32* %Precedence, i32** @dfg_PRECEDENCE, align 4
  store i8* null, i8** @dfg_DESC.0, align 4
  store i8* null, i8** @dfg_DESC.1, align 4
  store i8* null, i8** @dfg_DESC.2, align 4
  store i8* null, i8** @dfg_DESC.3, align 4
  store i32 2, i32* @dfg_DESC.4, align 4
  store i8* null, i8** @dfg_DESC.5, align 4
  store i8* null, i8** @dfg_DESC.6, align 4
  %call1 = tail call i32 @dfg_parse()
  %0 = load %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  %cmp.i13.i = icmp eq %struct.LIST_HELP* %0, null
  br i1 %cmp.i13.i, label %dfg_SymCleanUp.exit, label %while.body.lr.ph.i

while.body.lr.ph.i:                               ; preds = %entry
  %1 = load i32* @symbol_TYPESTATBITS, align 4
  br label %while.body.i

while.body.i:                                     ; preds = %if.end.i42, %while.body.lr.ph.i
  %2 = phi %struct.LIST_HELP* [ %0, %while.body.lr.ph.i ], [ %L.idx.val.i.i35, %if.end.i42 ]
  %.idx.i30 = getelementptr %struct.LIST_HELP* %2, i32 0, i32 1
  %.idx.val.i31 = load i8** %.idx.i30, align 4
  %symbol.i = bitcast i8* %.idx.val.i31 to i32*
  %3 = load i32* %symbol.i, align 4
  %arity.i = getelementptr inbounds i8* %.idx.val.i31, i32 8
  %4 = bitcast i8* %arity.i to i32*
  %5 = load i32* %4, align 4
  %sub.i.i9.i = sub nsw i32 0, %3
  %shr.i.i10.i = ashr i32 %sub.i.i9.i, %1
  %6 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i11.i = getelementptr inbounds %struct.signature** %6, i32 %shr.i.i10.i
  %7 = load %struct.signature** %arrayidx.i.i11.i, align 4
  %arity.i12.i = getelementptr inbounds %struct.signature* %7, i32 0, i32 3
  %8 = load i32* %arity.i12.i, align 4
  %cmp.i32 = icmp eq i32 %5, %8
  br i1 %cmp.i32, label %if.end.i42, label %if.then.i33

if.then.i33:                                      ; preds = %while.body.i
  store i32 %5, i32* %arity.i12.i, align 4
  br label %if.end.i42

if.end.i42:                                       ; preds = %if.then.i33, %while.body.i
  %9 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 12), align 4
  %total_size.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %9, i32 0, i32 4
  %10 = load i32* %total_size.i.i.i, align 4
  %11 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i = add i32 %11, %10
  store i32 %add24.i.i.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %9, i32 0, i32 0
  %12 = load i8** %free.i.i.i, align 4
  %.c.i.i = ptrtoint i8* %12 to i32
  store i32 %.c.i.i, i32* %symbol.i, align 4
  %13 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 12), align 4
  %free27.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %13, i32 0, i32 0
  store i8* %.idx.val.i31, i8** %free27.i.i.i, align 4
  %14 = load %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  %L.idx.i.i34 = getelementptr %struct.LIST_HELP* %14, i32 0, i32 0
  %L.idx.val.i.i35 = load %struct.LIST_HELP** %L.idx.i.i34, align 4
  %15 = bitcast %struct.LIST_HELP* %14 to i8*
  %16 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i36 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %16, i32 0, i32 4
  %17 = load i32* %total_size.i.i.i.i36, align 4
  %18 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i37 = add i32 %18, %17
  store i32 %add24.i.i.i.i37, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i38 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %16, i32 0, i32 0
  %19 = load i8** %free.i.i.i.i38, align 4
  %.c.i.i.i39 = bitcast i8* %19 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i39, %struct.LIST_HELP** %L.idx.i.i34, align 4
  %20 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i40 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %20, i32 0, i32 0
  store i8* %15, i8** %free27.i.i.i.i40, align 4
  store %struct.LIST_HELP* %L.idx.val.i.i35, %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  %cmp.i.i41 = icmp eq %struct.LIST_HELP* %L.idx.val.i.i35, null
  br i1 %cmp.i.i41, label %dfg_SymCleanUp.exit, label %while.body.i

dfg_SymCleanUp.exit:                              ; preds = %if.end.i42, %entry
  %21 = load %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  %22 = load %struct.LIST_HELP** @dfg_CONCLAUSES, align 4
  %cmp.i.i43 = icmp eq %struct.LIST_HELP* %21, null
  br i1 %cmp.i.i43, label %list_Nconc.exit53, label %if.end.i45

if.end.i45:                                       ; preds = %dfg_SymCleanUp.exit
  %cmp.i18.i44 = icmp eq %struct.LIST_HELP* %22, null
  br i1 %cmp.i18.i44, label %list_Nconc.exit53.thread, label %for.cond.i50

list_Nconc.exit53.thread:                         ; preds = %if.end.i45
  store %struct.LIST_HELP* %21, %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_CONCLAUSES, align 4
  br label %for.body

for.cond.i50:                                     ; preds = %if.end.i45, %for.cond.i50
  %List1.addr.0.i46 = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i48, %for.cond.i50 ], [ %21, %if.end.i45 ]
  %List1.addr.0.idx15.i47 = getelementptr %struct.LIST_HELP* %List1.addr.0.i46, i32 0, i32 0
  %List1.addr.0.idx15.val.i48 = load %struct.LIST_HELP** %List1.addr.0.idx15.i47, align 4
  %cmp.i16.i49 = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i48, null
  br i1 %cmp.i16.i49, label %for.end.i51, label %for.cond.i50

for.end.i51:                                      ; preds = %for.cond.i50
  store %struct.LIST_HELP* %22, %struct.LIST_HELP** %List1.addr.0.idx15.i47, align 4
  br label %list_Nconc.exit53

list_Nconc.exit53:                                ; preds = %dfg_SymCleanUp.exit, %for.end.i51
  %retval.0.i52 = phi %struct.LIST_HELP* [ %21, %for.end.i51 ], [ %22, %dfg_SymCleanUp.exit ]
  store %struct.LIST_HELP* %retval.0.i52, %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_CONCLAUSES, align 4
  %cmp.i81122 = icmp eq %struct.LIST_HELP* %retval.0.i52, null
  br i1 %cmp.i81122, label %for.end, label %for.body

for.body:                                         ; preds = %list_Nconc.exit53, %list_Nconc.exit53.thread, %for.inc
  %scan.0123 = phi %struct.LIST_HELP* [ %scan.0.idx24.val, %for.inc ], [ %21, %list_Nconc.exit53.thread ], [ %retval.0.i52, %list_Nconc.exit53 ]
  %scan.0.idx = getelementptr %struct.LIST_HELP* %scan.0123, i32 0, i32 1
  %scan.0.idx.val = load i8** %scan.0.idx, align 4
  %.idx = bitcast i8* %scan.0.idx.val to %struct.LIST_HELP**
  %.idx.val = load %struct.LIST_HELP** %.idx, align 4
  %.idx25 = getelementptr i8* %scan.0.idx.val, i32 4
  %23 = bitcast i8* %.idx25 to i8**
  %.idx25.val = load i8** %23, align 4
  %cmp = icmp eq i8* %.idx25.val, null
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %24 = bitcast %struct.LIST_HELP* %.idx.val to %struct.term*
  tail call void @term_Delete(%struct.term* %24) #1
  %25 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i113 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %25, i32 0, i32 4
  %26 = load i32* %total_size.i.i.i113, align 4
  %27 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i114 = add i32 %27, %26
  store i32 %add24.i.i.i114, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i115 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %25, i32 0, i32 0
  %28 = load i8** %free.i.i.i115, align 4
  %.c.i.i116 = bitcast i8* %28 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i116, %struct.LIST_HELP** %.idx, align 4
  %29 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i117 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %29, i32 0, i32 0
  store i8* %scan.0.idx.val, i8** %free27.i.i.i117, align 4
  store i8* null, i8** %scan.0.idx, align 4
  br label %for.inc

if.else:                                          ; preds = %for.body
  %30 = bitcast %struct.LIST_HELP* %.idx.val to i8*
  %call.i.i = tail call i8* @memory_Malloc(i32 8) #1
  %31 = bitcast i8* %call.i.i to %struct.LIST_HELP*
  %car.i.i = getelementptr inbounds i8* %call.i.i, i32 4
  %32 = bitcast i8* %car.i.i to i8**
  store i8* inttoptr (i32 16 to i8*), i8** %32, align 4
  %cdr.i.i = bitcast i8* %call.i.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* null, %struct.LIST_HELP** %cdr.i.i, align 4
  %call.i118 = tail call i8* @memory_Malloc(i32 8) #1
  %33 = bitcast i8* %call.i118 to %struct.LIST_HELP*
  %car.i119 = getelementptr inbounds i8* %call.i118, i32 4
  %34 = bitcast i8* %car.i119 to i8**
  store i8* null, i8** %34, align 4
  %cdr.i120 = bitcast i8* %call.i118 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %31, %struct.LIST_HELP** %cdr.i120, align 4
  %call.i110 = tail call i8* @memory_Malloc(i32 8) #1
  %35 = bitcast i8* %call.i110 to %struct.LIST_HELP*
  %car.i111 = getelementptr inbounds i8* %call.i110, i32 4
  %36 = bitcast i8* %car.i111 to i8**
  store i8* null, i8** %36, align 4
  %cdr.i112 = bitcast i8* %call.i110 to %struct.LIST_HELP**
  store %struct.LIST_HELP* %33, %struct.LIST_HELP** %cdr.i112, align 4
  %call.i = tail call i8* @memory_Malloc(i32 8) #1
  %37 = bitcast i8* %call.i to %struct.LIST_HELP*
  %car.i = getelementptr inbounds i8* %call.i, i32 4
  %38 = bitcast i8* %car.i to i8**
  store i8* %30, i8** %38, align 4
  %cdr.i109 = bitcast i8* %call.i to %struct.LIST_HELP**
  store %struct.LIST_HELP* %35, %struct.LIST_HELP** %cdr.i109, align 4
  store %struct.LIST_HELP* %37, %struct.LIST_HELP** %.idx, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.else
  %scan.0.idx24 = getelementptr %struct.LIST_HELP* %scan.0123, i32 0, i32 0
  %scan.0.idx24.val = load %struct.LIST_HELP** %scan.0.idx24, align 4
  %cmp.i81 = icmp eq %struct.LIST_HELP* %scan.0.idx24.val, null
  br i1 %cmp.i81, label %for.cond.for.end_crit_edge, label %for.body

for.cond.for.end_crit_edge:                       ; preds = %for.inc
  %.pre = load %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %list_Nconc.exit53
  %39 = phi %struct.LIST_HELP* [ %.pre, %for.cond.for.end_crit_edge ], [ null, %list_Nconc.exit53 ]
  %call14 = tail call %struct.LIST_HELP* @list_PointerDeleteElement(%struct.LIST_HELP* %39, i8* null) #1
  store %struct.LIST_HELP* %call14, %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  %40 = load %struct.LIST_HELP** @dfg_AXIOMLIST, align 4
  %cmp.i18.i82 = icmp eq %struct.LIST_HELP* %40, null
  br i1 %cmp.i18.i82, label %dfg_DeleteFormulaPairList.exit108, label %for.body.i91

for.body.i91:                                     ; preds = %for.end, %if.end.i106
  %FormulaPairs.addr.019.i83 = phi %struct.LIST_HELP* [ %L.idx.val.i.i99, %if.end.i106 ], [ %40, %for.end ]
  %FormulaPairs.addr.0.idx.i84 = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i83, i32 0, i32 1
  %FormulaPairs.addr.0.idx.val.i85 = load i8** %FormulaPairs.addr.0.idx.i84, align 4
  %.idx.i86 = bitcast i8* %FormulaPairs.addr.0.idx.val.i85 to %struct.LIST_HELP**
  %.idx.val.i87 = load %struct.LIST_HELP** %.idx.i86, align 4
  %41 = bitcast %struct.LIST_HELP* %.idx.val.i87 to %struct.term*
  tail call void @term_Delete(%struct.term* %41) #1
  %.idx12.i88 = getelementptr i8* %FormulaPairs.addr.0.idx.val.i85, i32 4
  %42 = bitcast i8* %.idx12.i88 to i8**
  %.idx12.val.i89 = load i8** %42, align 4
  %cmp.i90 = icmp eq i8* %.idx12.val.i89, null
  br i1 %cmp.i90, label %if.end.i106, label %if.then.i92

if.then.i92:                                      ; preds = %for.body.i91
  tail call void @string_StringFree(i8* %.idx12.val.i89) #1
  br label %if.end.i106

if.end.i106:                                      ; preds = %if.then.i92, %for.body.i91
  %43 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i13.i93 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %43, i32 0, i32 4
  %44 = load i32* %total_size.i.i.i13.i93, align 4
  %45 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i14.i94 = add i32 %45, %44
  store i32 %add24.i.i.i14.i94, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i15.i95 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %43, i32 0, i32 0
  %46 = load i8** %free.i.i.i15.i95, align 4
  %.c.i.i16.i96 = bitcast i8* %46 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i16.i96, %struct.LIST_HELP** %.idx.i86, align 4
  %47 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i17.i97 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %47, i32 0, i32 0
  store i8* %FormulaPairs.addr.0.idx.val.i85, i8** %free27.i.i.i17.i97, align 4
  %L.idx.i.i98 = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i83, i32 0, i32 0
  %L.idx.val.i.i99 = load %struct.LIST_HELP** %L.idx.i.i98, align 4
  %48 = bitcast %struct.LIST_HELP* %FormulaPairs.addr.019.i83 to i8*
  %49 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i100 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %49, i32 0, i32 4
  %50 = load i32* %total_size.i.i.i.i100, align 4
  %51 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i101 = add i32 %51, %50
  store i32 %add24.i.i.i.i101, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i102 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %49, i32 0, i32 0
  %52 = load i8** %free.i.i.i.i102, align 4
  %.c.i.i.i103 = bitcast i8* %52 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i103, %struct.LIST_HELP** %L.idx.i.i98, align 4
  %53 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i104 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %53, i32 0, i32 0
  store i8* %48, i8** %free27.i.i.i.i104, align 4
  %cmp.i.i105 = icmp eq %struct.LIST_HELP* %L.idx.val.i.i99, null
  br i1 %cmp.i.i105, label %dfg_DeleteFormulaPairList.exit108, label %for.body.i91

dfg_DeleteFormulaPairList.exit108:                ; preds = %if.end.i106, %for.end
  %54 = load %struct.LIST_HELP** @dfg_CONJECLIST, align 4
  %cmp.i18.i54 = icmp eq %struct.LIST_HELP* %54, null
  br i1 %cmp.i18.i54, label %dfg_DeleteFormulaPairList.exit80, label %for.body.i63

for.body.i63:                                     ; preds = %dfg_DeleteFormulaPairList.exit108, %if.end.i78
  %FormulaPairs.addr.019.i55 = phi %struct.LIST_HELP* [ %L.idx.val.i.i71, %if.end.i78 ], [ %54, %dfg_DeleteFormulaPairList.exit108 ]
  %FormulaPairs.addr.0.idx.i56 = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i55, i32 0, i32 1
  %FormulaPairs.addr.0.idx.val.i57 = load i8** %FormulaPairs.addr.0.idx.i56, align 4
  %.idx.i58 = bitcast i8* %FormulaPairs.addr.0.idx.val.i57 to %struct.LIST_HELP**
  %.idx.val.i59 = load %struct.LIST_HELP** %.idx.i58, align 4
  %55 = bitcast %struct.LIST_HELP* %.idx.val.i59 to %struct.term*
  tail call void @term_Delete(%struct.term* %55) #1
  %.idx12.i60 = getelementptr i8* %FormulaPairs.addr.0.idx.val.i57, i32 4
  %56 = bitcast i8* %.idx12.i60 to i8**
  %.idx12.val.i61 = load i8** %56, align 4
  %cmp.i62 = icmp eq i8* %.idx12.val.i61, null
  br i1 %cmp.i62, label %if.end.i78, label %if.then.i64

if.then.i64:                                      ; preds = %for.body.i63
  tail call void @string_StringFree(i8* %.idx12.val.i61) #1
  br label %if.end.i78

if.end.i78:                                       ; preds = %if.then.i64, %for.body.i63
  %57 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i13.i65 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %57, i32 0, i32 4
  %58 = load i32* %total_size.i.i.i13.i65, align 4
  %59 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i14.i66 = add i32 %59, %58
  store i32 %add24.i.i.i14.i66, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i15.i67 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %57, i32 0, i32 0
  %60 = load i8** %free.i.i.i15.i67, align 4
  %.c.i.i16.i68 = bitcast i8* %60 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i16.i68, %struct.LIST_HELP** %.idx.i58, align 4
  %61 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i17.i69 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %61, i32 0, i32 0
  store i8* %FormulaPairs.addr.0.idx.val.i57, i8** %free27.i.i.i17.i69, align 4
  %L.idx.i.i70 = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i55, i32 0, i32 0
  %L.idx.val.i.i71 = load %struct.LIST_HELP** %L.idx.i.i70, align 4
  %62 = bitcast %struct.LIST_HELP* %FormulaPairs.addr.019.i55 to i8*
  %63 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i72 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %63, i32 0, i32 4
  %64 = load i32* %total_size.i.i.i.i72, align 4
  %65 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i73 = add i32 %65, %64
  store i32 %add24.i.i.i.i73, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i74 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %63, i32 0, i32 0
  %66 = load i8** %free.i.i.i.i74, align 4
  %.c.i.i.i75 = bitcast i8* %66 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i75, %struct.LIST_HELP** %L.idx.i.i70, align 4
  %67 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i76 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %67, i32 0, i32 0
  store i8* %62, i8** %free27.i.i.i.i76, align 4
  %cmp.i.i77 = icmp eq %struct.LIST_HELP* %L.idx.val.i.i71, null
  br i1 %cmp.i.i77, label %dfg_DeleteFormulaPairList.exit80, label %for.body.i63

dfg_DeleteFormulaPairList.exit80:                 ; preds = %if.end.i78, %dfg_DeleteFormulaPairList.exit108
  %68 = load %struct.LIST_HELP** @dfg_SORTDECLLIST, align 4
  %cmp.i18.i26 = icmp eq %struct.LIST_HELP* %68, null
  br i1 %cmp.i18.i26, label %dfg_DeleteFormulaPairList.exit, label %for.body.i

for.body.i:                                       ; preds = %dfg_DeleteFormulaPairList.exit80, %if.end.i28
  %FormulaPairs.addr.019.i = phi %struct.LIST_HELP* [ %L.idx.val.i.i, %if.end.i28 ], [ %68, %dfg_DeleteFormulaPairList.exit80 ]
  %FormulaPairs.addr.0.idx.i = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i, i32 0, i32 1
  %FormulaPairs.addr.0.idx.val.i = load i8** %FormulaPairs.addr.0.idx.i, align 4
  %.idx.i = bitcast i8* %FormulaPairs.addr.0.idx.val.i to %struct.LIST_HELP**
  %.idx.val.i = load %struct.LIST_HELP** %.idx.i, align 4
  %69 = bitcast %struct.LIST_HELP* %.idx.val.i to %struct.term*
  tail call void @term_Delete(%struct.term* %69) #1
  %.idx12.i = getelementptr i8* %FormulaPairs.addr.0.idx.val.i, i32 4
  %70 = bitcast i8* %.idx12.i to i8**
  %.idx12.val.i = load i8** %70, align 4
  %cmp.i = icmp eq i8* %.idx12.val.i, null
  br i1 %cmp.i, label %if.end.i28, label %if.then.i

if.then.i:                                        ; preds = %for.body.i
  tail call void @string_StringFree(i8* %.idx12.val.i) #1
  br label %if.end.i28

if.end.i28:                                       ; preds = %if.then.i, %for.body.i
  %71 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i13.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %71, i32 0, i32 4
  %72 = load i32* %total_size.i.i.i13.i, align 4
  %73 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i14.i = add i32 %73, %72
  store i32 %add24.i.i.i14.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i15.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %71, i32 0, i32 0
  %74 = load i8** %free.i.i.i15.i, align 4
  %.c.i.i16.i = bitcast i8* %74 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i16.i, %struct.LIST_HELP** %.idx.i, align 4
  %75 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i17.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %75, i32 0, i32 0
  store i8* %FormulaPairs.addr.0.idx.val.i, i8** %free27.i.i.i17.i, align 4
  %L.idx.i.i = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i, i32 0, i32 0
  %L.idx.val.i.i = load %struct.LIST_HELP** %L.idx.i.i, align 4
  %76 = bitcast %struct.LIST_HELP* %FormulaPairs.addr.019.i to i8*
  %77 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %77, i32 0, i32 4
  %78 = load i32* %total_size.i.i.i.i, align 4
  %79 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i = add i32 %79, %78
  store i32 %add24.i.i.i.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %77, i32 0, i32 0
  %80 = load i8** %free.i.i.i.i, align 4
  %.c.i.i.i = bitcast i8* %80 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i, %struct.LIST_HELP** %L.idx.i.i, align 4
  %81 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %81, i32 0, i32 0
  store i8* %76, i8** %free27.i.i.i.i, align 4
  %cmp.i.i27 = icmp eq %struct.LIST_HELP* %L.idx.val.i.i, null
  br i1 %cmp.i.i27, label %dfg_DeleteFormulaPairList.exit, label %for.body.i

dfg_DeleteFormulaPairList.exit:                   ; preds = %if.end.i28, %dfg_DeleteFormulaPairList.exit80
  %82 = load %struct.LIST_HELP** @dfg_TERMLIST, align 4
  tail call void @list_DeleteWithElement(%struct.LIST_HELP* %82, void (i8*)* bitcast (void (%struct.term*)* @term_Delete to void (i8*)*)) #1
  %83 = load %struct.LIST_HELP** @dfg_PROOFLIST, align 4
  %call15 = tail call %struct.LIST_HELP* @list_NReverse(%struct.LIST_HELP* %83) #1
  store %struct.LIST_HELP* %call15, %struct.LIST_HELP** @dfg_PROOFLIST, align 4
  %84 = load %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  %cmp.i.i = icmp eq %struct.LIST_HELP* %84, null
  br i1 %cmp.i.i, label %list_Nconc.exit, label %if.end.i

if.end.i:                                         ; preds = %dfg_DeleteFormulaPairList.exit
  %cmp.i18.i = icmp eq %struct.LIST_HELP* %call15, null
  br i1 %cmp.i18.i, label %list_Nconc.exit, label %for.cond.i

for.cond.i:                                       ; preds = %if.end.i, %for.cond.i
  %List1.addr.0.i = phi %struct.LIST_HELP* [ %List1.addr.0.idx15.val.i, %for.cond.i ], [ %84, %if.end.i ]
  %List1.addr.0.idx15.i = getelementptr %struct.LIST_HELP* %List1.addr.0.i, i32 0, i32 0
  %List1.addr.0.idx15.val.i = load %struct.LIST_HELP** %List1.addr.0.idx15.i, align 4
  %cmp.i16.i = icmp eq %struct.LIST_HELP* %List1.addr.0.idx15.val.i, null
  br i1 %cmp.i16.i, label %for.end.i, label %for.cond.i

for.end.i:                                        ; preds = %for.cond.i
  store %struct.LIST_HELP* %call15, %struct.LIST_HELP** %List1.addr.0.idx15.i, align 4
  br label %list_Nconc.exit

list_Nconc.exit:                                  ; preds = %dfg_DeleteFormulaPairList.exit, %if.end.i, %for.end.i
  %retval.0.i = phi %struct.LIST_HELP* [ %84, %for.end.i ], [ %call15, %dfg_DeleteFormulaPairList.exit ], [ %84, %if.end.i ]
  store %struct.LIST_HELP* %retval.0.i, %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  ret %struct.LIST_HELP* %retval.0.i
}

; Function Attrs: nounwind
define void @dfg_DeleteFormulaPairList(%struct.LIST_HELP* %FormulaPairs) #0 {
entry:
  %cmp.i18 = icmp eq %struct.LIST_HELP* %FormulaPairs, null
  br i1 %cmp.i18, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %if.end
  %FormulaPairs.addr.019 = phi %struct.LIST_HELP* [ %L.idx.val.i, %if.end ], [ %FormulaPairs, %entry ]
  %FormulaPairs.addr.0.idx = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019, i32 0, i32 1
  %FormulaPairs.addr.0.idx.val = load i8** %FormulaPairs.addr.0.idx, align 4
  %.idx = bitcast i8* %FormulaPairs.addr.0.idx.val to %struct.LIST_HELP**
  %.idx.val = load %struct.LIST_HELP** %.idx, align 4
  %0 = bitcast %struct.LIST_HELP* %.idx.val to %struct.term*
  tail call void @term_Delete(%struct.term* %0) #1
  %.idx12 = getelementptr i8* %FormulaPairs.addr.0.idx.val, i32 4
  %1 = bitcast i8* %.idx12 to i8**
  %.idx12.val = load i8** %1, align 4
  %cmp = icmp eq i8* %.idx12.val, null
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  tail call void @string_StringFree(i8* %.idx12.val) #1
  br label %if.end

if.end:                                           ; preds = %for.body, %if.then
  %2 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i13 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %2, i32 0, i32 4
  %3 = load i32* %total_size.i.i.i13, align 4
  %4 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i14 = add i32 %4, %3
  store i32 %add24.i.i.i14, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i15 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %2, i32 0, i32 0
  %5 = load i8** %free.i.i.i15, align 4
  %.c.i.i16 = bitcast i8* %5 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i16, %struct.LIST_HELP** %.idx, align 4
  %6 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i17 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %6, i32 0, i32 0
  store i8* %FormulaPairs.addr.0.idx.val, i8** %free27.i.i.i17, align 4
  %L.idx.i = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019, i32 0, i32 0
  %L.idx.val.i = load %struct.LIST_HELP** %L.idx.i, align 4
  %7 = bitcast %struct.LIST_HELP* %FormulaPairs.addr.019 to i8*
  %8 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %8, i32 0, i32 4
  %9 = load i32* %total_size.i.i.i, align 4
  %10 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i = add i32 %10, %9
  store i32 %add24.i.i.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %8, i32 0, i32 0
  %11 = load i8** %free.i.i.i, align 4
  %.c.i.i = bitcast i8* %11 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i, %struct.LIST_HELP** %L.idx.i, align 4
  %12 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %12, i32 0, i32 0
  store i8* %7, i8** %free27.i.i.i, align 4
  %cmp.i = icmp eq %struct.LIST_HELP* %L.idx.val.i, null
  br i1 %cmp.i, label %for.end, label %for.body

for.end:                                          ; preds = %if.end, %entry
  ret void
}

; Function Attrs: nounwind
define %struct.LIST_HELP* @dfg_TermParser(%struct._IO_FILE* %File, i32* %Flags, i32* %Precedence) #0 {
entry:
  store %struct._IO_FILE* %File, %struct._IO_FILE** @dfg_in, align 4
  store i32 1, i32* @dfg_LINENUMBER, align 4
  store i32 1, i32* @dfg_IGNORETEXT, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_AXIOMLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_CONJECLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_SORTDECLLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_USERPRECEDENCE, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_CONCLAUSES, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_PROOFLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_TERMLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  store %struct.LIST_HELP* null, %struct.LIST_HELP** @dfg_VARLIST, align 4
  store i1 false, i1* @dfg_VARDECL, align 1
  store i32 0, i32* @dfg_IGNORE, align 4
  store i32* %Flags, i32** @dfg_FLAGS, align 4
  store i32* %Precedence, i32** @dfg_PRECEDENCE, align 4
  store i8* null, i8** @dfg_DESC.0, align 4
  store i8* null, i8** @dfg_DESC.1, align 4
  store i8* null, i8** @dfg_DESC.2, align 4
  store i8* null, i8** @dfg_DESC.3, align 4
  store i32 2, i32* @dfg_DESC.4, align 4
  store i8* null, i8** @dfg_DESC.5, align 4
  store i8* null, i8** @dfg_DESC.6, align 4
  %call1 = tail call i32 @dfg_parse()
  %0 = load %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  %cmp.i13.i = icmp eq %struct.LIST_HELP* %0, null
  br i1 %cmp.i13.i, label %dfg_SymCleanUp.exit, label %while.body.lr.ph.i

while.body.lr.ph.i:                               ; preds = %entry
  %1 = load i32* @symbol_TYPESTATBITS, align 4
  br label %while.body.i

while.body.i:                                     ; preds = %if.end.i14, %while.body.lr.ph.i
  %2 = phi %struct.LIST_HELP* [ %0, %while.body.lr.ph.i ], [ %L.idx.val.i.i7, %if.end.i14 ]
  %.idx.i2 = getelementptr %struct.LIST_HELP* %2, i32 0, i32 1
  %.idx.val.i3 = load i8** %.idx.i2, align 4
  %symbol.i = bitcast i8* %.idx.val.i3 to i32*
  %3 = load i32* %symbol.i, align 4
  %arity.i = getelementptr inbounds i8* %.idx.val.i3, i32 8
  %4 = bitcast i8* %arity.i to i32*
  %5 = load i32* %4, align 4
  %sub.i.i9.i = sub nsw i32 0, %3
  %shr.i.i10.i = ashr i32 %sub.i.i9.i, %1
  %6 = load %struct.signature*** @symbol_SIGNATURE, align 4
  %arrayidx.i.i11.i = getelementptr inbounds %struct.signature** %6, i32 %shr.i.i10.i
  %7 = load %struct.signature** %arrayidx.i.i11.i, align 4
  %arity.i12.i = getelementptr inbounds %struct.signature* %7, i32 0, i32 3
  %8 = load i32* %arity.i12.i, align 4
  %cmp.i4 = icmp eq i32 %5, %8
  br i1 %cmp.i4, label %if.end.i14, label %if.then.i5

if.then.i5:                                       ; preds = %while.body.i
  store i32 %5, i32* %arity.i12.i, align 4
  br label %if.end.i14

if.end.i14:                                       ; preds = %if.then.i5, %while.body.i
  %9 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 12), align 4
  %total_size.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %9, i32 0, i32 4
  %10 = load i32* %total_size.i.i.i, align 4
  %11 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i = add i32 %11, %10
  store i32 %add24.i.i.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %9, i32 0, i32 0
  %12 = load i8** %free.i.i.i, align 4
  %.c.i.i = ptrtoint i8* %12 to i32
  store i32 %.c.i.i, i32* %symbol.i, align 4
  %13 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 12), align 4
  %free27.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %13, i32 0, i32 0
  store i8* %.idx.val.i3, i8** %free27.i.i.i, align 4
  %14 = load %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  %L.idx.i.i6 = getelementptr %struct.LIST_HELP* %14, i32 0, i32 0
  %L.idx.val.i.i7 = load %struct.LIST_HELP** %L.idx.i.i6, align 4
  %15 = bitcast %struct.LIST_HELP* %14 to i8*
  %16 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i8 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %16, i32 0, i32 4
  %17 = load i32* %total_size.i.i.i.i8, align 4
  %18 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i9 = add i32 %18, %17
  store i32 %add24.i.i.i.i9, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i10 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %16, i32 0, i32 0
  %19 = load i8** %free.i.i.i.i10, align 4
  %.c.i.i.i11 = bitcast i8* %19 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i11, %struct.LIST_HELP** %L.idx.i.i6, align 4
  %20 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i12 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %20, i32 0, i32 0
  store i8* %15, i8** %free27.i.i.i.i12, align 4
  store %struct.LIST_HELP* %L.idx.val.i.i7, %struct.LIST_HELP** @dfg_SYMBOLLIST, align 4
  %cmp.i.i13 = icmp eq %struct.LIST_HELP* %L.idx.val.i.i7, null
  br i1 %cmp.i.i13, label %dfg_SymCleanUp.exit, label %while.body.i

dfg_SymCleanUp.exit:                              ; preds = %if.end.i14, %entry
  %21 = load %struct.LIST_HELP** @dfg_AXCLAUSES, align 4
  %cmp.i18.i15 = icmp eq %struct.LIST_HELP* %21, null
  br i1 %cmp.i18.i15, label %dfg_DeleteFormulaPairList.exit40, label %for.body.i24

for.body.i24:                                     ; preds = %dfg_SymCleanUp.exit, %if.end.i39
  %FormulaPairs.addr.019.i16 = phi %struct.LIST_HELP* [ %L.idx.val.i.i32, %if.end.i39 ], [ %21, %dfg_SymCleanUp.exit ]
  %FormulaPairs.addr.0.idx.i17 = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i16, i32 0, i32 1
  %FormulaPairs.addr.0.idx.val.i18 = load i8** %FormulaPairs.addr.0.idx.i17, align 4
  %.idx.i19 = bitcast i8* %FormulaPairs.addr.0.idx.val.i18 to %struct.LIST_HELP**
  %.idx.val.i20 = load %struct.LIST_HELP** %.idx.i19, align 4
  %22 = bitcast %struct.LIST_HELP* %.idx.val.i20 to %struct.term*
  tail call void @term_Delete(%struct.term* %22) #1
  %.idx12.i21 = getelementptr i8* %FormulaPairs.addr.0.idx.val.i18, i32 4
  %23 = bitcast i8* %.idx12.i21 to i8**
  %.idx12.val.i22 = load i8** %23, align 4
  %cmp.i23 = icmp eq i8* %.idx12.val.i22, null
  br i1 %cmp.i23, label %if.end.i39, label %if.then.i25

if.then.i25:                                      ; preds = %for.body.i24
  tail call void @string_StringFree(i8* %.idx12.val.i22) #1
  br label %if.end.i39

if.end.i39:                                       ; preds = %if.then.i25, %for.body.i24
  %24 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i13.i26 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %24, i32 0, i32 4
  %25 = load i32* %total_size.i.i.i13.i26, align 4
  %26 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i14.i27 = add i32 %26, %25
  store i32 %add24.i.i.i14.i27, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i15.i28 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %24, i32 0, i32 0
  %27 = load i8** %free.i.i.i15.i28, align 4
  %.c.i.i16.i29 = bitcast i8* %27 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i16.i29, %struct.LIST_HELP** %.idx.i19, align 4
  %28 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i17.i30 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %28, i32 0, i32 0
  store i8* %FormulaPairs.addr.0.idx.val.i18, i8** %free27.i.i.i17.i30, align 4
  %L.idx.i.i31 = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i16, i32 0, i32 0
  %L.idx.val.i.i32 = load %struct.LIST_HELP** %L.idx.i.i31, align 4
  %29 = bitcast %struct.LIST_HELP* %FormulaPairs.addr.019.i16 to i8*
  %30 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i33 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %30, i32 0, i32 4
  %31 = load i32* %total_size.i.i.i.i33, align 4
  %32 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i34 = add i32 %32, %31
  store i32 %add24.i.i.i.i34, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i35 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %30, i32 0, i32 0
  %33 = load i8** %free.i.i.i.i35, align 4
  %.c.i.i.i36 = bitcast i8* %33 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i36, %struct.LIST_HELP** %L.idx.i.i31, align 4
  %34 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i37 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %34, i32 0, i32 0
  store i8* %29, i8** %free27.i.i.i.i37, align 4
  %cmp.i.i38 = icmp eq %struct.LIST_HELP* %L.idx.val.i.i32, null
  br i1 %cmp.i.i38, label %dfg_DeleteFormulaPairList.exit40, label %for.body.i24

dfg_DeleteFormulaPairList.exit40:                 ; preds = %if.end.i39, %dfg_SymCleanUp.exit
  %35 = load %struct.LIST_HELP** @dfg_CONCLAUSES, align 4
  %cmp.i18.i41 = icmp eq %struct.LIST_HELP* %35, null
  br i1 %cmp.i18.i41, label %dfg_DeleteFormulaPairList.exit66, label %for.body.i50

for.body.i50:                                     ; preds = %dfg_DeleteFormulaPairList.exit40, %if.end.i65
  %FormulaPairs.addr.019.i42 = phi %struct.LIST_HELP* [ %L.idx.val.i.i58, %if.end.i65 ], [ %35, %dfg_DeleteFormulaPairList.exit40 ]
  %FormulaPairs.addr.0.idx.i43 = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i42, i32 0, i32 1
  %FormulaPairs.addr.0.idx.val.i44 = load i8** %FormulaPairs.addr.0.idx.i43, align 4
  %.idx.i45 = bitcast i8* %FormulaPairs.addr.0.idx.val.i44 to %struct.LIST_HELP**
  %.idx.val.i46 = load %struct.LIST_HELP** %.idx.i45, align 4
  %36 = bitcast %struct.LIST_HELP* %.idx.val.i46 to %struct.term*
  tail call void @term_Delete(%struct.term* %36) #1
  %.idx12.i47 = getelementptr i8* %FormulaPairs.addr.0.idx.val.i44, i32 4
  %37 = bitcast i8* %.idx12.i47 to i8**
  %.idx12.val.i48 = load i8** %37, align 4
  %cmp.i49 = icmp eq i8* %.idx12.val.i48, null
  br i1 %cmp.i49, label %if.end.i65, label %if.then.i51

if.then.i51:                                      ; preds = %for.body.i50
  tail call void @string_StringFree(i8* %.idx12.val.i48) #1
  br label %if.end.i65

if.end.i65:                                       ; preds = %if.then.i51, %for.body.i50
  %38 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i13.i52 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %38, i32 0, i32 4
  %39 = load i32* %total_size.i.i.i13.i52, align 4
  %40 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i14.i53 = add i32 %40, %39
  store i32 %add24.i.i.i14.i53, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i15.i54 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %38, i32 0, i32 0
  %41 = load i8** %free.i.i.i15.i54, align 4
  %.c.i.i16.i55 = bitcast i8* %41 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i16.i55, %struct.LIST_HELP** %.idx.i45, align 4
  %42 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i17.i56 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %42, i32 0, i32 0
  store i8* %FormulaPairs.addr.0.idx.val.i44, i8** %free27.i.i.i17.i56, align 4
  %L.idx.i.i57 = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i42, i32 0, i32 0
  %L.idx.val.i.i58 = load %struct.LIST_HELP** %L.idx.i.i57, align 4
  %43 = bitcast %struct.LIST_HELP* %FormulaPairs.addr.019.i42 to i8*
  %44 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i59 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %44, i32 0, i32 4
  %45 = load i32* %total_size.i.i.i.i59, align 4
  %46 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i60 = add i32 %46, %45
  store i32 %add24.i.i.i.i60, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i61 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %44, i32 0, i32 0
  %47 = load i8** %free.i.i.i.i61, align 4
  %.c.i.i.i62 = bitcast i8* %47 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i62, %struct.LIST_HELP** %L.idx.i.i57, align 4
  %48 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i63 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %48, i32 0, i32 0
  store i8* %43, i8** %free27.i.i.i.i63, align 4
  %cmp.i.i64 = icmp eq %struct.LIST_HELP* %L.idx.val.i.i58, null
  br i1 %cmp.i.i64, label %dfg_DeleteFormulaPairList.exit66, label %for.body.i50

dfg_DeleteFormulaPairList.exit66:                 ; preds = %if.end.i65, %dfg_DeleteFormulaPairList.exit40
  %49 = load %struct.LIST_HELP** @dfg_AXIOMLIST, align 4
  %cmp.i18.i67 = icmp eq %struct.LIST_HELP* %49, null
  br i1 %cmp.i18.i67, label %dfg_DeleteFormulaPairList.exit92, label %for.body.i76

for.body.i76:                                     ; preds = %dfg_DeleteFormulaPairList.exit66, %if.end.i91
  %FormulaPairs.addr.019.i68 = phi %struct.LIST_HELP* [ %L.idx.val.i.i84, %if.end.i91 ], [ %49, %dfg_DeleteFormulaPairList.exit66 ]
  %FormulaPairs.addr.0.idx.i69 = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i68, i32 0, i32 1
  %FormulaPairs.addr.0.idx.val.i70 = load i8** %FormulaPairs.addr.0.idx.i69, align 4
  %.idx.i71 = bitcast i8* %FormulaPairs.addr.0.idx.val.i70 to %struct.LIST_HELP**
  %.idx.val.i72 = load %struct.LIST_HELP** %.idx.i71, align 4
  %50 = bitcast %struct.LIST_HELP* %.idx.val.i72 to %struct.term*
  tail call void @term_Delete(%struct.term* %50) #1
  %.idx12.i73 = getelementptr i8* %FormulaPairs.addr.0.idx.val.i70, i32 4
  %51 = bitcast i8* %.idx12.i73 to i8**
  %.idx12.val.i74 = load i8** %51, align 4
  %cmp.i75 = icmp eq i8* %.idx12.val.i74, null
  br i1 %cmp.i75, label %if.end.i91, label %if.then.i77

if.then.i77:                                      ; preds = %for.body.i76
  tail call void @string_StringFree(i8* %.idx12.val.i74) #1
  br label %if.end.i91

if.end.i91:                                       ; preds = %if.then.i77, %for.body.i76
  %52 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i13.i78 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %52, i32 0, i32 4
  %53 = load i32* %total_size.i.i.i13.i78, align 4
  %54 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i14.i79 = add i32 %54, %53
  store i32 %add24.i.i.i14.i79, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i15.i80 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %52, i32 0, i32 0
  %55 = load i8** %free.i.i.i15.i80, align 4
  %.c.i.i16.i81 = bitcast i8* %55 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i16.i81, %struct.LIST_HELP** %.idx.i71, align 4
  %56 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i17.i82 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %56, i32 0, i32 0
  store i8* %FormulaPairs.addr.0.idx.val.i70, i8** %free27.i.i.i17.i82, align 4
  %L.idx.i.i83 = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i68, i32 0, i32 0
  %L.idx.val.i.i84 = load %struct.LIST_HELP** %L.idx.i.i83, align 4
  %57 = bitcast %struct.LIST_HELP* %FormulaPairs.addr.019.i68 to i8*
  %58 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i85 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %58, i32 0, i32 4
  %59 = load i32* %total_size.i.i.i.i85, align 4
  %60 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i86 = add i32 %60, %59
  store i32 %add24.i.i.i.i86, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i87 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %58, i32 0, i32 0
  %61 = load i8** %free.i.i.i.i87, align 4
  %.c.i.i.i88 = bitcast i8* %61 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i88, %struct.LIST_HELP** %L.idx.i.i83, align 4
  %62 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i89 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %62, i32 0, i32 0
  store i8* %57, i8** %free27.i.i.i.i89, align 4
  %cmp.i.i90 = icmp eq %struct.LIST_HELP* %L.idx.val.i.i84, null
  br i1 %cmp.i.i90, label %dfg_DeleteFormulaPairList.exit92, label %for.body.i76

dfg_DeleteFormulaPairList.exit92:                 ; preds = %if.end.i91, %dfg_DeleteFormulaPairList.exit66
  %63 = load %struct.LIST_HELP** @dfg_CONJECLIST, align 4
  %cmp.i18.i93 = icmp eq %struct.LIST_HELP* %63, null
  br i1 %cmp.i18.i93, label %dfg_DeleteFormulaPairList.exit118, label %for.body.i102

for.body.i102:                                    ; preds = %dfg_DeleteFormulaPairList.exit92, %if.end.i117
  %FormulaPairs.addr.019.i94 = phi %struct.LIST_HELP* [ %L.idx.val.i.i110, %if.end.i117 ], [ %63, %dfg_DeleteFormulaPairList.exit92 ]
  %FormulaPairs.addr.0.idx.i95 = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i94, i32 0, i32 1
  %FormulaPairs.addr.0.idx.val.i96 = load i8** %FormulaPairs.addr.0.idx.i95, align 4
  %.idx.i97 = bitcast i8* %FormulaPairs.addr.0.idx.val.i96 to %struct.LIST_HELP**
  %.idx.val.i98 = load %struct.LIST_HELP** %.idx.i97, align 4
  %64 = bitcast %struct.LIST_HELP* %.idx.val.i98 to %struct.term*
  tail call void @term_Delete(%struct.term* %64) #1
  %.idx12.i99 = getelementptr i8* %FormulaPairs.addr.0.idx.val.i96, i32 4
  %65 = bitcast i8* %.idx12.i99 to i8**
  %.idx12.val.i100 = load i8** %65, align 4
  %cmp.i101 = icmp eq i8* %.idx12.val.i100, null
  br i1 %cmp.i101, label %if.end.i117, label %if.then.i103

if.then.i103:                                     ; preds = %for.body.i102
  tail call void @string_StringFree(i8* %.idx12.val.i100) #1
  br label %if.end.i117

if.end.i117:                                      ; preds = %if.then.i103, %for.body.i102
  %66 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i13.i104 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %66, i32 0, i32 4
  %67 = load i32* %total_size.i.i.i13.i104, align 4
  %68 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i14.i105 = add i32 %68, %67
  store i32 %add24.i.i.i14.i105, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i15.i106 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %66, i32 0, i32 0
  %69 = load i8** %free.i.i.i15.i106, align 4
  %.c.i.i16.i107 = bitcast i8* %69 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i16.i107, %struct.LIST_HELP** %.idx.i97, align 4
  %70 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i17.i108 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %70, i32 0, i32 0
  store i8* %FormulaPairs.addr.0.idx.val.i96, i8** %free27.i.i.i17.i108, align 4
  %L.idx.i.i109 = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i94, i32 0, i32 0
  %L.idx.val.i.i110 = load %struct.LIST_HELP** %L.idx.i.i109, align 4
  %71 = bitcast %struct.LIST_HELP* %FormulaPairs.addr.019.i94 to i8*
  %72 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i111 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %72, i32 0, i32 4
  %73 = load i32* %total_size.i.i.i.i111, align 4
  %74 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i112 = add i32 %74, %73
  store i32 %add24.i.i.i.i112, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i113 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %72, i32 0, i32 0
  %75 = load i8** %free.i.i.i.i113, align 4
  %.c.i.i.i114 = bitcast i8* %75 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i114, %struct.LIST_HELP** %L.idx.i.i109, align 4
  %76 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i115 = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %76, i32 0, i32 0
  store i8* %71, i8** %free27.i.i.i.i115, align 4
  %cmp.i.i116 = icmp eq %struct.LIST_HELP* %L.idx.val.i.i110, null
  br i1 %cmp.i.i116, label %dfg_DeleteFormulaPairList.exit118, label %for.body.i102

dfg_DeleteFormulaPairList.exit118:                ; preds = %if.end.i117, %dfg_DeleteFormulaPairList.exit92
  %77 = load %struct.LIST_HELP** @dfg_PROOFLIST, align 4
  tail call void @dfg_DeleteProofList(%struct.LIST_HELP* %77)
  %78 = load %struct.LIST_HELP** @dfg_SORTDECLLIST, align 4
  %cmp.i18.i = icmp eq %struct.LIST_HELP* %78, null
  br i1 %cmp.i18.i, label %dfg_DeleteFormulaPairList.exit, label %for.body.i

for.body.i:                                       ; preds = %dfg_DeleteFormulaPairList.exit118, %if.end.i
  %FormulaPairs.addr.019.i = phi %struct.LIST_HELP* [ %L.idx.val.i.i, %if.end.i ], [ %78, %dfg_DeleteFormulaPairList.exit118 ]
  %FormulaPairs.addr.0.idx.i = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i, i32 0, i32 1
  %FormulaPairs.addr.0.idx.val.i = load i8** %FormulaPairs.addr.0.idx.i, align 4
  %.idx.i = bitcast i8* %FormulaPairs.addr.0.idx.val.i to %struct.LIST_HELP**
  %.idx.val.i = load %struct.LIST_HELP** %.idx.i, align 4
  %79 = bitcast %struct.LIST_HELP* %.idx.val.i to %struct.term*
  tail call void @term_Delete(%struct.term* %79) #1
  %.idx12.i = getelementptr i8* %FormulaPairs.addr.0.idx.val.i, i32 4
  %80 = bitcast i8* %.idx12.i to i8**
  %.idx12.val.i = load i8** %80, align 4
  %cmp.i = icmp eq i8* %.idx12.val.i, null
  br i1 %cmp.i, label %if.end.i, label %if.then.i

if.then.i:                                        ; preds = %for.body.i
  tail call void @string_StringFree(i8* %.idx12.val.i) #1
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %for.body.i
  %81 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i13.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %81, i32 0, i32 4
  %82 = load i32* %total_size.i.i.i13.i, align 4
  %83 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i14.i = add i32 %83, %82
  store i32 %add24.i.i.i14.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i15.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %81, i32 0, i32 0
  %84 = load i8** %free.i.i.i15.i, align 4
  %.c.i.i16.i = bitcast i8* %84 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i16.i, %struct.LIST_HELP** %.idx.i, align 4
  %85 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i17.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %85, i32 0, i32 0
  store i8* %FormulaPairs.addr.0.idx.val.i, i8** %free27.i.i.i17.i, align 4
  %L.idx.i.i = getelementptr %struct.LIST_HELP* %FormulaPairs.addr.019.i, i32 0, i32 0
  %L.idx.val.i.i = load %struct.LIST_HELP** %L.idx.i.i, align 4
  %86 = bitcast %struct.LIST_HELP* %FormulaPairs.addr.019.i to i8*
  %87 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %87, i32 0, i32 4
  %88 = load i32* %total_size.i.i.i.i, align 4
  %89 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i.i = add i32 %89, %88
  store i32 %add24.i.i.i.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %87, i32 0, i32 0
  %90 = load i8** %free.i.i.i.i, align 4
  %.c.i.i.i = bitcast i8* %90 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i.i, %struct.LIST_HELP** %L.idx.i.i, align 4
  %91 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %91, i32 0, i32 0
  store i8* %86, i8** %free27.i.i.i.i, align 4
  %cmp.i.i = icmp eq %struct.LIST_HELP* %L.idx.val.i.i, null
  br i1 %cmp.i.i, label %dfg_DeleteFormulaPairList.exit, label %for.body.i

dfg_DeleteFormulaPairList.exit:                   ; preds = %if.end.i, %dfg_DeleteFormulaPairList.exit118
  %92 = load %struct.LIST_HELP** @dfg_TERMLIST, align 4
  ret %struct.LIST_HELP* %92
}

; Function Attrs: nounwind
define void @dfg_StripLabelsFromList(%struct.LIST_HELP* %FormulaPairs) #0 {
entry:
  %cmp.i15 = icmp eq %struct.LIST_HELP* %FormulaPairs, null
  br i1 %cmp.i15, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %if.end
  %scan.016 = phi %struct.LIST_HELP* [ %scan.0.idx12.val, %if.end ], [ %FormulaPairs, %entry ]
  %scan.0.idx = getelementptr %struct.LIST_HELP* %scan.016, i32 0, i32 1
  %scan.0.idx.val = load i8** %scan.0.idx, align 4
  %.idx = bitcast i8* %scan.0.idx.val to %struct.LIST_HELP**
  %.idx.val = load %struct.LIST_HELP** %.idx, align 4
  %0 = bitcast %struct.LIST_HELP* %.idx.val to i8*
  store i8* %0, i8** %scan.0.idx, align 4
  %.idx14 = getelementptr i8* %scan.0.idx.val, i32 4
  %1 = bitcast i8* %.idx14 to i8**
  %.idx14.val = load i8** %1, align 4
  %cmp = icmp eq i8* %.idx14.val, null
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  tail call void @string_StringFree(i8* %.idx14.val) #1
  br label %if.end

if.end:                                           ; preds = %for.body, %if.then
  %2 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %2, i32 0, i32 4
  %3 = load i32* %total_size.i.i.i, align 4
  %4 = load i32* @memory_FREEDBYTES, align 4
  %add24.i.i.i = add i32 %4, %3
  store i32 %add24.i.i.i, i32* @memory_FREEDBYTES, align 4
  %free.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %2, i32 0, i32 0
  %5 = load i8** %free.i.i.i, align 4
  %.c.i.i = bitcast i8* %5 to %struct.LIST_HELP*
  store %struct.LIST_HELP* %.c.i.i, %struct.LIST_HELP** %.idx, align 4
  %6 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i.i.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %6, i32 0, i32 0
  store i8* %scan.0.idx.val, i8** %free27.i.i.i, align 4
  %scan.0.idx12 = getelementptr %struct.LIST_HELP* %scan.016, i32 0, i32 0
  %scan.0.idx12.val = load %struct.LIST_HELP** %scan.0.idx12, align 4
  %cmp.i = icmp eq %struct.LIST_HELP* %scan.0.idx12.val, null
  br i1 %cmp.i, label %for.end, label %for.body

for.end:                                          ; preds = %if.end, %entry
  ret void
}

declare %struct.LIST_HELP* @list_PointerDeleteDuplicates(%struct.LIST_HELP*) #2

declare %struct.term* @fol_CreateQuantifier(i32, %struct.LIST_HELP*, %struct.LIST_HELP*) #2

declare %struct.CLAUSE_HELP* @clause_CreateFromLiterals(%struct.LIST_HELP*, i32, i32, i32, i32*, i32*) #2

declare void @list_DeleteWithElement(%struct.LIST_HELP*, void (i8*)*) #2

; Function Attrs: nounwind
define internal void @dfg_VarFree(%struct.DFG_VARENTRY* %Entry) #0 {
entry:
  %name = getelementptr inbounds %struct.DFG_VARENTRY* %Entry, i32 0, i32 0
  %0 = load i8** %name, align 4
  tail call void @string_StringFree(i8* %0) #1
  %1 = bitcast %struct.DFG_VARENTRY* %Entry to i8*
  %2 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %total_size.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %2, i32 0, i32 4
  %3 = load i32* %total_size.i, align 4
  %4 = load i32* @memory_FREEDBYTES, align 4
  %add24.i = add i32 %4, %3
  store i32 %add24.i, i32* @memory_FREEDBYTES, align 4
  %free.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %2, i32 0, i32 0
  %5 = load i8** %free.i, align 4
  store i8* %5, i8** %name, align 4
  %6 = load %struct.MEMORY_RESOURCEHELP** getelementptr inbounds ([0 x %struct.MEMORY_RESOURCEHELP*]* @memory_ARRAY, i32 0, i32 8), align 4
  %free27.i = getelementptr inbounds %struct.MEMORY_RESOURCEHELP* %6, i32 0, i32 0
  store i8* %1, i8** %free27.i, align 4
  ret void
}

declare %struct.term* @term_Copy(%struct.term*) #2

declare i8* @memory_Malloc(i32) #2

declare i32 @symbol_CreateFunction(i8*, i32, i32, i32*) #2

declare i32 @symbol_CreatePredicate(i8*, i32, i32, i32*) #2

declare i32 @symbol_CreateJunctor(i8*, i32, i32, i32*) #2

; Function Attrs: noreturn nounwind
declare void @abort() #5

declare i32 @symbol_GetIncreasedOrderingCounter() #2

declare i32 @flag_Minimum(i32) #2

declare i8* @flag_Name(i32) #2

declare i32 @flag_Maximum(i32) #2

; Function Attrs: nounwind readonly
declare i32 @strcmp(i8* nocapture, i8* nocapture) #4

declare i32 @list_Length(%struct.LIST_HELP*) #2

; Function Attrs: noreturn nounwind
declare void @exit(i32) #5

; Function Attrs: nounwind
declare i32 @fwrite(i8* nocapture, i32, i32, %struct._IO_FILE* nocapture) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { nounwind }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #3 = { inlinehint noreturn nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #4 = { nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #5 = { noreturn nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #6 = { nounwind readonly }
attributes #7 = { noreturn nounwind }

