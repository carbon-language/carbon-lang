; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin10.0 -relocation-model=pic -disable-fp-elim | not grep dil

	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
	%struct.spec_fd_t = type { i32, i32, i32, i8* }
@globalCrc = internal global i32 0		; <i32*> [#uses=6]
@bsStream.b = internal global i1 false		; <i1*> [#uses=41]
@bsLive = internal global i32 0		; <i32*> [#uses=91]
@bsBuff = internal global i32 0		; <i32*> [#uses=91]
@bytesIn = internal global i32 0		; <i32*> [#uses=4]
@nInUse = internal global i32 0		; <i32*> [#uses=3]
@inUse = internal global [256 x i8] zeroinitializer, align 32		; <[256 x i8]*> [#uses=7]
@seqToUnseq = internal global [256 x i8] zeroinitializer, align 32		; <[256 x i8]*> [#uses=3]
@unseqToSeq = internal global [256 x i8] zeroinitializer, align 32		; <[256 x i8]*> [#uses=1]
@mtfFreq = internal global [258 x i32] zeroinitializer, align 32		; <[258 x i32]*> [#uses=2]
@block = internal global i8* null		; <i8**> [#uses=3]
@zptr = internal global i32* null		; <i32**> [#uses=2]
@szptr = internal global i16* null		; <i16**> [#uses=3]
@last = internal global i32 0		; <i32*> [#uses=13]
@nMTF = internal global i32 0		; <i32*> [#uses=2]
@quadrant = internal global i16* null		; <i16**> [#uses=1]
@workDone = internal global i32 0		; <i32*> [#uses=3]
@rNums = internal constant [512 x i32] [i32 619, i32 720, i32 127, i32 481, i32 931, i32 816, i32 813, i32 233, i32 566, i32 247, i32 985, i32 724, i32 205, i32 454, i32 863, i32 491, i32 741, i32 242, i32 949, i32 214, i32 733, i32 859, i32 335, i32 708, i32 621, i32 574, i32 73, i32 654, i32 730, i32 472, i32 419, i32 436, i32 278, i32 496, i32 867, i32 210, i32 399, i32 680, i32 480, i32 51, i32 878, i32 465, i32 811, i32 169, i32 869, i32 675, i32 611, i32 697, i32 867, i32 561, i32 862, i32 687, i32 507, i32 283, i32 482, i32 129, i32 807, i32 591, i32 733, i32 623, i32 150, i32 238, i32 59, i32 379, i32 684, i32 877, i32 625, i32 169, i32 643, i32 105, i32 170, i32 607, i32 520, i32 932, i32 727, i32 476, i32 693, i32 425, i32 174, i32 647, i32 73, i32 122, i32 335, i32 530, i32 442, i32 853, i32 695, i32 249, i32 445, i32 515, i32 909, i32 545, i32 703, i32 919, i32 874, i32 474, i32 882, i32 500, i32 594, i32 612, i32 641, i32 801, i32 220, i32 162, i32 819, i32 984, i32 589, i32 513, i32 495, i32 799, i32 161, i32 604, i32 958, i32 533, i32 221, i32 400, i32 386, i32 867, i32 600, i32 782, i32 382, i32 596, i32 414, i32 171, i32 516, i32 375, i32 682, i32 485, i32 911, i32 276, i32 98, i32 553, i32 163, i32 354, i32 666, i32 933, i32 424, i32 341, i32 533, i32 870, i32 227, i32 730, i32 475, i32 186, i32 263, i32 647, i32 537, i32 686, i32 600, i32 224, i32 469, i32 68, i32 770, i32 919, i32 190, i32 373, i32 294, i32 822, i32 808, i32 206, i32 184, i32 943, i32 795, i32 384, i32 383, i32 461, i32 404, i32 758, i32 839, i32 887, i32 715, i32 67, i32 618, i32 276, i32 204, i32 918, i32 873, i32 777, i32 604, i32 560, i32 951, i32 160, i32 578, i32 722, i32 79, i32 804, i32 96, i32 409, i32 713, i32 940, i32 652, i32 934, i32 970, i32 447, i32 318, i32 353, i32 859, i32 672, i32 112, i32 785, i32 645, i32 863, i32 803, i32 350, i32 139, i32 93, i32 354, i32 99, i32 820, i32 908, i32 609, i32 772, i32 154, i32 274, i32 580, i32 184, i32 79, i32 626, i32 630, i32 742, i32 653, i32 282, i32 762, i32 623, i32 680, i32 81, i32 927, i32 626, i32 789, i32 125, i32 411, i32 521, i32 938, i32 300, i32 821, i32 78, i32 343, i32 175, i32 128, i32 250, i32 170, i32 774, i32 972, i32 275, i32 999, i32 639, i32 495, i32 78, i32 352, i32 126, i32 857, i32 956, i32 358, i32 619, i32 580, i32 124, i32 737, i32 594, i32 701, i32 612, i32 669, i32 112, i32 134, i32 694, i32 363, i32 992, i32 809, i32 743, i32 168, i32 974, i32 944, i32 375, i32 748, i32 52, i32 600, i32 747, i32 642, i32 182, i32 862, i32 81, i32 344, i32 805, i32 988, i32 739, i32 511, i32 655, i32 814, i32 334, i32 249, i32 515, i32 897, i32 955, i32 664, i32 981, i32 649, i32 113, i32 974, i32 459, i32 893, i32 228, i32 433, i32 837, i32 553, i32 268, i32 926, i32 240, i32 102, i32 654, i32 459, i32 51, i32 686, i32 754, i32 806, i32 760, i32 493, i32 403, i32 415, i32 394, i32 687, i32 700, i32 946, i32 670, i32 656, i32 610, i32 738, i32 392, i32 760, i32 799, i32 887, i32 653, i32 978, i32 321, i32 576, i32 617, i32 626, i32 502, i32 894, i32 679, i32 243, i32 440, i32 680, i32 879, i32 194, i32 572, i32 640, i32 724, i32 926, i32 56, i32 204, i32 700, i32 707, i32 151, i32 457, i32 449, i32 797, i32 195, i32 791, i32 558, i32 945, i32 679, i32 297, i32 59, i32 87, i32 824, i32 713, i32 663, i32 412, i32 693, i32 342, i32 606, i32 134, i32 108, i32 571, i32 364, i32 631, i32 212, i32 174, i32 643, i32 304, i32 329, i32 343, i32 97, i32 430, i32 751, i32 497, i32 314, i32 983, i32 374, i32 822, i32 928, i32 140, i32 206, i32 73, i32 263, i32 980, i32 736, i32 876, i32 478, i32 430, i32 305, i32 170, i32 514, i32 364, i32 692, i32 829, i32 82, i32 855, i32 953, i32 676, i32 246, i32 369, i32 970, i32 294, i32 750, i32 807, i32 827, i32 150, i32 790, i32 288, i32 923, i32 804, i32 378, i32 215, i32 828, i32 592, i32 281, i32 565, i32 555, i32 710, i32 82, i32 896, i32 831, i32 547, i32 261, i32 524, i32 462, i32 293, i32 465, i32 502, i32 56, i32 661, i32 821, i32 976, i32 991, i32 658, i32 869, i32 905, i32 758, i32 745, i32 193, i32 768, i32 550, i32 608, i32 933, i32 378, i32 286, i32 215, i32 979, i32 792, i32 961, i32 61, i32 688, i32 793, i32 644, i32 986, i32 403, i32 106, i32 366, i32 905, i32 644, i32 372, i32 567, i32 466, i32 434, i32 645, i32 210, i32 389, i32 550, i32 919, i32 135, i32 780, i32 773, i32 635, i32 389, i32 707, i32 100, i32 626, i32 958, i32 165, i32 504, i32 920, i32 176, i32 193, i32 713, i32 857, i32 265, i32 203, i32 50, i32 668, i32 108, i32 645, i32 990, i32 626, i32 197, i32 510, i32 357, i32 358, i32 850, i32 858, i32 364, i32 936, i32 638], align 32		; <[512 x i32]*> [#uses=3]
@__stderrp = external global %struct.FILE*		; <%struct.FILE**> [#uses=9]
@"\01LC" = internal constant [36 x i8] c"\09Input file = %s, output file = %s\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[36 x i8]*> [#uses=1]
@inName = internal global [1024 x i8] zeroinitializer, align 32		; <[1024 x i8]*> [#uses=1]
@outName = internal global [1024 x i8] zeroinitializer, align 32		; <[1024 x i8]*> [#uses=1]
@workLimit = internal global i32 0		; <i32*> [#uses=3]
@firstAttempt.b = internal global i1 false		; <i1*> [#uses=3]
@"\01LC5" = internal constant [146 x i8] c"\0A%s: Can't allocate enough memory for compression.\0A\09Requested %d bytes for a block size of %d.\0A\09Try selecting a small block size (with flag -s).\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[146 x i8]*> [#uses=1]
@"\01LC6" = internal constant [206 x i8] c"\0A%s: Can't allocate enough memory for decompression.\0A\09Requested %d bytes for a block size of %d.\0A\09Try selecting space-economic decompress (with flag -s)\0A\09and failing that, find a machine with more memory.\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[206 x i8]*> [#uses=1]
@"\01LC11" = internal constant [21 x i8] c"hbMakeCodeLengths(1)\00", section "__TEXT,__cstring,cstring_literals"		; <[21 x i8]*> [#uses=1]
@"\01LC12" = internal constant [21 x i8] c"hbMakeCodeLengths(2)\00", section "__TEXT,__cstring,cstring_literals"		; <[21 x i8]*> [#uses=1]
@"\01LC13" = internal constant [243 x i8] c"\0AIt is possible that the compressed file(s) have become corrupted.\0AYou can use the -tvv option to test integrity of such files.\0A\0AYou can use the `bzip2recover' program to *attempt* to recover\0Adata from undamaged sections of corrupted files.\0A\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[243 x i8]*> [#uses=1]
@"\01LC17" = internal constant [86 x i8] c"\0A%s: bad block header in the compressed file,\0A\09which probably means it is corrupted.\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[86 x i8]*> [#uses=1]
@ftab = internal global i32* null		; <i32**> [#uses=1]
@"\01LC32" = internal constant [27 x i8] c"doReversibleTransformation\00", section "__TEXT,__cstring,cstring_literals"		; <[27 x i8]*> [#uses=1]
@selectorMtf = internal global [18002 x i8] zeroinitializer, align 32		; <[18002 x i8]*> [#uses=4]
@selector = internal global [18002 x i8] zeroinitializer, align 32		; <[18002 x i8]*> [#uses=7]
@len = internal global [6 x [258 x i8]] zeroinitializer, align 32		; <[6 x [258 x i8]]*> [#uses=24]
@perm = internal global [6 x [258 x i32]] zeroinitializer, align 32		; <[6 x [258 x i32]]*> [#uses=4]
@base = internal global [6 x [258 x i32]] zeroinitializer, align 32		; <[6 x [258 x i32]]*> [#uses=11]
@limit = internal global [6 x [258 x i32]] zeroinitializer, align 32		; <[6 x [258 x i32]]*> [#uses=7]
@minLens = internal global [6 x i32] zeroinitializer		; <[6 x i32]*> [#uses=4]
@blockSize100k = internal global i32 0		; <i32*> [#uses=6]
@unzftab = internal global [256 x i32] zeroinitializer, align 32		; <[256 x i32]*> [#uses=4]
@crc32Table = internal constant [256 x i32] [i32 0, i32 79764919, i32 159529838, i32 222504665, i32 319059676, i32 398814059, i32 445009330, i32 507990021, i32 638119352, i32 583659535, i32 797628118, i32 726387553, i32 890018660, i32 835552979, i32 1015980042, i32 944750013, i32 1276238704, i32 1221641927, i32 1167319070, i32 1095957929, i32 1595256236, i32 1540665371, i32 1452775106, i32 1381403509, i32 1780037320, i32 1859660671, i32 1671105958, i32 1733955601, i32 2031960084, i32 2111593891, i32 1889500026, i32 1952343757, i32 -1742489888, i32 -1662866601, i32 -1851683442, i32 -1788833735, i32 -1960329156, i32 -1880695413, i32 -2103051438, i32 -2040207643, i32 -1104454824, i32 -1159051537, i32 -1213636554, i32 -1284997759, i32 -1389417084, i32 -1444007885, i32 -1532160278, i32 -1603531939, i32 -734892656, i32 -789352409, i32 -575645954, i32 -646886583, i32 -952755380, i32 -1007220997, i32 -827056094, i32 -898286187, i32 -231047128, i32 -151282273, i32 -71779514, i32 -8804623, i32 -515967244, i32 -436212925, i32 -390279782, i32 -327299027, i32 881225847, i32 809987520, i32 1023691545, i32 969234094, i32 662832811, i32 591600412, i32 771767749, i32 717299826, i32 311336399, i32 374308984, i32 453813921, i32 533576470, i32 25881363, i32 88864420, i32 134795389, i32 214552010, i32 2023205639, i32 2086057648, i32 1897238633, i32 1976864222, i32 1804852699, i32 1867694188, i32 1645340341, i32 1724971778, i32 1587496639, i32 1516133128, i32 1461550545, i32 1406951526, i32 1302016099, i32 1230646740, i32 1142491917, i32 1087903418, i32 -1398421865, i32 -1469785312, i32 -1524105735, i32 -1578704818, i32 -1079922613, i32 -1151291908, i32 -1239184603, i32 -1293773166, i32 -1968362705, i32 -1905510760, i32 -2094067647, i32 -2014441994, i32 -1716953613, i32 -1654112188, i32 -1876203875, i32 -1796572374, i32 -525066777, i32 -462094256, i32 -382327159, i32 -302564546, i32 -206542021, i32 -143559028, i32 -97365931, i32 -17609246, i32 -960696225, i32 -1031934488, i32 -817968335, i32 -872425850, i32 -709327229, i32 -780559564, i32 -600130067, i32 -654598054, i32 1762451694, i32 1842216281, i32 1619975040, i32 1682949687, i32 2047383090, i32 2127137669, i32 1938468188, i32 2001449195, i32 1325665622, i32 1271206113, i32 1183200824, i32 1111960463, i32 1543535498, i32 1489069629, i32 1434599652, i32 1363369299, i32 622672798, i32 568075817, i32 748617968, i32 677256519, i32 907627842, i32 853037301, i32 1067152940, i32 995781531, i32 51762726, i32 131386257, i32 177728840, i32 240578815, i32 269590778, i32 349224269, i32 429104020, i32 491947555, i32 -248556018, i32 -168932423, i32 -122852000, i32 -60002089, i32 -500490030, i32 -420856475, i32 -341238852, i32 -278395381, i32 -685261898, i32 -739858943, i32 -559578920, i32 -630940305, i32 -1004286614, i32 -1058877219, i32 -845023740, i32 -916395085, i32 -1119974018, i32 -1174433591, i32 -1262701040, i32 -1333941337, i32 -1371866206, i32 -1426332139, i32 -1481064244, i32 -1552294533, i32 -1690935098, i32 -1611170447, i32 -1833673816, i32 -1770699233, i32 -2009983462, i32 -1930228819, i32 -2119160460, i32 -2056179517, i32 1569362073, i32 1498123566, i32 1409854455, i32 1355396672, i32 1317987909, i32 1246755826, i32 1192025387, i32 1137557660, i32 2072149281, i32 2135122070, i32 1912620623, i32 1992383480, i32 1753615357, i32 1816598090, i32 1627664531, i32 1707420964, i32 295390185, i32 358241886, i32 404320391, i32 483945776, i32 43990325, i32 106832002, i32 186451547, i32 266083308, i32 932423249, i32 861060070, i32 1041341759, i32 986742920, i32 613929101, i32 542559546, i32 756411363, i32 701822548, i32 -978770311, i32 -1050133554, i32 -869589737, i32 -924188512, i32 -693284699, i32 -764654318, i32 -550540341, i32 -605129092, i32 -475935807, i32 -413084042, i32 -366743377, i32 -287118056, i32 -257573603, i32 -194731862, i32 -114850189, i32 -35218492, i32 -1984365303, i32 -1921392450, i32 -2143631769, i32 -2063868976, i32 -1698919467, i32 -1635936670, i32 -1824608069, i32 -1744851700, i32 -1347415887, i32 -1418654458, i32 -1506661409, i32 -1561119128, i32 -1129027987, i32 -1200260134, i32 -1254728445, i32 -1309196108], align 32		; <[256 x i32]*> [#uses=4]
@"\01LC35" = internal constant [17 x i8] c"sendMTFValues(0)\00", section "__TEXT,__cstring,cstring_literals"		; <[17 x i8]*> [#uses=1]
@rfreq = internal global [6 x [258 x i32]] zeroinitializer, align 32		; <[6 x [258 x i32]]*> [#uses=3]
@"\01LC40" = internal constant [17 x i8] c"sendMTFValues(2)\00", section "__TEXT,__cstring,cstring_literals"		; <[17 x i8]*> [#uses=1]
@"\01LC41" = internal constant [17 x i8] c"sendMTFValues(3)\00", section "__TEXT,__cstring,cstring_literals"		; <[17 x i8]*> [#uses=1]
@"\01LC42" = internal constant [17 x i8] c"sendMTFValues(4)\00", section "__TEXT,__cstring,cstring_literals"		; <[17 x i8]*> [#uses=1]
@code = internal global [6 x [258 x i32]] zeroinitializer, align 32		; <[6 x [258 x i32]]*> [#uses=2]
@"\01LC46" = internal constant [17 x i8] c"sendMTFValues(5)\00", section "__TEXT,__cstring,cstring_literals"		; <[17 x i8]*> [#uses=1]
@"\01LC54" = internal constant [28 x i8] c"setDecompressStructureSizes\00", section "__TEXT,__cstring,cstring_literals"		; <[28 x i8]*> [#uses=1]
@spec_fd = internal global [3 x %struct.spec_fd_t] zeroinitializer, align 32		; <[3 x %struct.spec_fd_t]*> [#uses=128]
@"\01LC1179" = internal constant [10 x i8] c"spec_init\00", section "__TEXT,__cstring,cstring_literals"		; <[10 x i8]*> [#uses=1]
@"\01LC1280" = internal constant [35 x i8] c"spec_init: Error mallocing memory!\00", section "__TEXT,__cstring,cstring_literals"		; <[35 x i8]*> [#uses=1]
@"\01LC1684" = internal constant [24 x i8] c"Can't open file %s: %s\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[24 x i8]*> [#uses=1]
@"\01LC1785" = internal constant [27 x i8] c"Error reading from %s: %s\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[27 x i8]*> [#uses=1]
@"\01LC1886" = internal constant [22 x i8] c"Duplicating %d bytes\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[22 x i8]*> [#uses=1]
@"\01LC1987" = internal constant [15 x i8] c"input.combined\00", section "__TEXT,__cstring,cstring_literals"		; <[15 x i8]*> [#uses=1]
@"\01LC2088" = internal constant [19 x i8] c"Loading Input Data\00", section "__TEXT,__cstring,cstring_literals"		; <[19 x i8]*> [#uses=1]
@"\01LC2189" = internal constant [31 x i8] c"Input data %d bytes in length\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[31 x i8]*> [#uses=1]
@"\01LC2290" = internal constant [30 x i8] c"main: Error mallocing memory!\00", section "__TEXT,__cstring,cstring_literals"		; <[30 x i8]*> [#uses=1]
@"\01LC2391" = internal constant [34 x i8] c"Compressing Input Data, level %d\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[34 x i8]*> [#uses=1]
@"\01LC2492" = internal constant [36 x i8] c"Compressed data %d bytes in length\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[36 x i8]*> [#uses=1]
@"\01LC2593" = internal constant [19 x i8] c"Uncompressing Data\00", section "__TEXT,__cstring,cstring_literals"		; <[19 x i8]*> [#uses=1]
@"\01LC2694" = internal constant [38 x i8] c"Uncompressed data %d bytes in length\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[38 x i8]*> [#uses=1]
@"\01LC2795" = internal constant [35 x i8] c"Tested %dMB buffer: Miscompared!!\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[35 x i8]*> [#uses=1]
@"\01LC2896" = internal constant [37 x i8] c"Uncompressed data compared correctly\00", section "__TEXT,__cstring,cstring_literals"		; <[37 x i8]*> [#uses=1]
@"\01LC2997" = internal constant [25 x i8] c"Tested %dMB buffer: OK!\0A\00", section "__TEXT,__cstring,cstring_literals"		; <[25 x i8]*> [#uses=1]
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (i32, i8**)* @main to i8*)], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind ssp {
entry:
	%parent.i.i.i = alloca [516 x i32], align 4		; <[516 x i32]*> [#uses=8]
	%weight.i.i.i = alloca [516 x i32], align 4		; <[516 x i32]*> [#uses=19]
	%heap.i.i.i = alloca [260 x i32], align 4		; <[260 x i32]*> [#uses=25]
	%inUse16.i.i = alloca [16 x i8], align 1		; <[16 x i8]*> [#uses=3]
	%pos.i.i = alloca [6 x i8], align 1		; <[6 x i8]*> [#uses=3]
	%fave.i.i = alloca [6 x i32], align 4		; <[6 x i32]*> [#uses=2]
	%cost.i.i = alloca [6 x i16], align 2		; <[6 x i16]*> [#uses=9]
	%pos.i.i.i = alloca [6 x i8], align 1		; <[6 x i8]*> [#uses=5]
	%inUse16.i.i.i = alloca [16 x i8], align 1		; <[16 x i8]*> [#uses=2]
	%yy.i.i = alloca [256 x i8], align 1		; <[256 x i8]*> [#uses=10]
	%cftab.i.i = alloca [257 x i32], align 4		; <[257 x i32]*> [#uses=5]
	%0 = icmp sgt i32 %argc, 1		; <i1> [#uses=1]
	br i1 %0, label %bb, label %bb1

bb:		; preds = %entry
	%1 = getelementptr i8** %argv, i32 1		; <i8**> [#uses=1]
	%2 = load i8** %1, align 4		; <i8*> [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%input_name.0 = phi i8* [ %2, %bb ], [ getelementptr ([15 x i8]* @"\01LC1987", i32 0, i32 0), %entry ]		; <i8*> [#uses=3]
	%3 = icmp sgt i32 %argc, 2		; <i1> [#uses=1]
	br i1 %3, label %bb2, label %bb3

bb2:		; preds = %bb1
	%4 = getelementptr i8** %argv, i32 2		; <i8**> [#uses=1]
	%5 = load i8** %4, align 4		; <i8*> [#uses=1]
	%6 = tail call i32 @atoi(i8* %5) nounwind		; <i32> [#uses=1]
	br label %bb3

bb3:		; preds = %bb2, %bb1
	%input_size.0 = phi i32 [ %6, %bb2 ], [ 64, %bb1 ]		; <i32> [#uses=5]
	%7 = icmp sgt i32 %argc, 3		; <i1> [#uses=1]
	br i1 %7, label %bb4, label %bb6

bb4:		; preds = %bb3
	%8 = getelementptr i8** %argv, i32 3		; <i8**> [#uses=1]
	%9 = load i8** %8, align 4		; <i8*> [#uses=1]
	%10 = tail call i32 @atoi(i8* %9) nounwind		; <i32> [#uses=1]
	br label %bb6

bb6:		; preds = %bb4, %bb3
	%compressed_size.0 = phi i32 [ %10, %bb4 ], [ %input_size.0, %bb3 ]		; <i32> [#uses=1]
	%11 = shl i32 %input_size.0, 20		; <i32> [#uses=9]
	store i32 %11, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 0), align 32
	%12 = shl i32 %compressed_size.0, 20		; <i32> [#uses=1]
	store i32 %12, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 0), align 16
	store i32 %11, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 2, i32 0), align 32
	%13 = tail call i32 @puts(i8* getelementptr ([10 x i8]* @"\01LC1179", i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	br label %bb11.i

bb2.i:		; preds = %bb11.i
	%scevgep35.i = bitcast i32* %scevgep3.i to i8*		; <i8*> [#uses=1]
	%14 = load i32* %scevgep3.i, align 16		; <i32> [#uses=4]
	tail call void @llvm.memset.i32(i8* %scevgep35.i, i8 0, i32 16, i32 16) nounwind
	store i32 %14, i32* %scevgep3.i, align 16
	%15 = add i32 %14, 102400		; <i32> [#uses=1]
	%16 = malloc i8, i32 %15		; <i8*> [#uses=2]
	store i8* %16, i8** %scevgep4.i, align 4
	%17 = icmp eq i8* %16, null		; <i1> [#uses=1]
	br i1 %17, label %bb6.i, label %bb9.preheader.i

bb9.preheader.i:		; preds = %bb2.i
	%18 = icmp sgt i32 %14, 0		; <i1> [#uses=1]
	br i1 %18, label %bb8.i, label %bb10.i

bb6.i:		; preds = %bb2.i
	%19 = tail call i32 @puts(i8* getelementptr ([35 x i8]* @"\01LC1280", i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	tail call void @exit(i32 1) noreturn nounwind
	unreachable

bb8.i:		; preds = %bb8.i, %bb9.preheader.i
	%indvar.i = phi i32 [ %indvar.next.i, %bb8.i ], [ 0, %bb9.preheader.i ]		; <i32> [#uses=2]
	%tmp.i = shl i32 %indvar.i, 10		; <i32> [#uses=2]
	%20 = load i8** %scevgep4.i, align 4		; <i8*> [#uses=1]
	%scevgep.i = getelementptr i8* %20, i32 %tmp.i		; <i8*> [#uses=1]
	store i8 0, i8* %scevgep.i, align 1
	%tmp2.i = add i32 %tmp.i, 1024		; <i32> [#uses=1]
	%21 = icmp slt i32 %tmp2.i, %14		; <i1> [#uses=1]
	%indvar.next.i = add i32 %indvar.i, 1		; <i32> [#uses=1]
	br i1 %21, label %bb8.i, label %bb10.i

bb10.i:		; preds = %bb8.i, %bb9.preheader.i
	%22 = add i32 %23, 1		; <i32> [#uses=1]
	br label %bb11.i

bb11.i:		; preds = %bb10.i, %bb6
	%23 = phi i32 [ %22, %bb10.i ], [ 0, %bb6 ]		; <i32> [#uses=4]
	%scevgep3.i = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %23, i32 0		; <i32*> [#uses=3]
	%scevgep4.i = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %23, i32 3		; <i8**> [#uses=2]
	%24 = icmp sgt i32 %23, 2		; <i1> [#uses=1]
	br i1 %24, label %bb8, label %bb2.i

bb8:		; preds = %bb11.i
	%25 = tail call i32 @puts(i8* getelementptr ([19 x i8]* @"\01LC2088", i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	%26 = tail call i32 (i8*, i32, ...)* @"\01_open$UNIX2003"(i8* %input_name.0, i32 0) nounwind		; <i32> [#uses=3]
	%27 = icmp slt i32 %26, 0		; <i1> [#uses=1]
	br i1 %27, label %bb.i77, label %bb1.i

bb.i77:		; preds = %bb8
	%28 = tail call i32* @__error() nounwind		; <i32*> [#uses=1]
	%29 = load i32* %28, align 4		; <i32> [#uses=1]
	%30 = tail call i8* @"\01_strerror$UNIX2003"(i32 %29) nounwind		; <i8*> [#uses=1]
	%31 = load %struct.FILE** @__stderrp, align 4		; <%struct.FILE*> [#uses=1]
	%32 = tail call i32 (%struct.FILE*, i8*, ...)* @fprintf(%struct.FILE* %31, i8* getelementptr ([24 x i8]* @"\01LC1684", i32 0, i32 0), i8* %input_name.0, i8* %30) nounwind		; <i32> [#uses=0]
	tail call void @exit(i32 1) noreturn nounwind
	unreachable

bb1.i:		; preds = %bb8
	store i32 0, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4
	store i32 0, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 2), align 8
	br label %bb6.i80

bb2.i78:		; preds = %bb6.i80
	%33 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 3), align 4		; <i8*> [#uses=1]
	%34 = getelementptr i8* %33, i32 %i.0.i		; <i8*> [#uses=1]
	%35 = tail call i32 (...)* @read(i32 %26, i8* %34, i32 131072) nounwind		; <i32> [#uses=4]
	%36 = icmp eq i32 %35, 0		; <i1> [#uses=1]
	br i1 %36, label %bb7.i81, label %bb3.i

bb3.i:		; preds = %bb2.i78
	%37 = icmp slt i32 %35, 0		; <i1> [#uses=1]
	br i1 %37, label %bb4.i, label %bb5.i79

bb4.i:		; preds = %bb3.i
	%38 = tail call i32* @__error() nounwind		; <i32*> [#uses=1]
	%39 = load i32* %38, align 4		; <i32> [#uses=1]
	%40 = tail call i8* @"\01_strerror$UNIX2003"(i32 %39) nounwind		; <i8*> [#uses=1]
	%41 = load %struct.FILE** @__stderrp, align 4		; <%struct.FILE*> [#uses=1]
	%42 = tail call i32 (%struct.FILE*, i8*, ...)* @fprintf(%struct.FILE* %41, i8* getelementptr ([27 x i8]* @"\01LC1785", i32 0, i32 0), i8* %input_name.0, i8* %40) nounwind		; <i32> [#uses=0]
	tail call void @exit(i32 1) noreturn nounwind
	unreachable

bb5.i79:		; preds = %bb3.i
	%43 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4		; <i32> [#uses=1]
	%44 = add i32 %43, %35		; <i32> [#uses=1]
	store i32 %44, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4
	%45 = add i32 %35, %i.0.i		; <i32> [#uses=1]
	br label %bb6.i80

bb6.i80:		; preds = %bb5.i79, %bb1.i
	%i.0.i = phi i32 [ 0, %bb1.i ], [ %45, %bb5.i79 ]		; <i32> [#uses=3]
	%46 = icmp slt i32 %i.0.i, %11		; <i1> [#uses=1]
	br i1 %46, label %bb2.i78, label %bb7.i81

bb7.i81:		; preds = %bb6.i80, %bb2.i78
	%47 = tail call i32 (...)* @close(i32 %26) nounwind		; <i32> [#uses=0]
	%48 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4		; <i32> [#uses=3]
	%49 = icmp slt i32 %48, %11		; <i1> [#uses=1]
	br i1 %49, label %bb14.i, label %bb10

bb14.i:		; preds = %bb14.i, %bb7.i81
	%50 = phi i32 [ %57, %bb14.i ], [ %48, %bb7.i81 ]		; <i32> [#uses=3]
	%51 = sub i32 %11, %50		; <i32> [#uses=2]
	%52 = icmp slt i32 %50, %51		; <i1> [#uses=1]
	%tmp.0.i = select i1 %52, i32 %50, i32 %51		; <i32> [#uses=3]
	%53 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([22 x i8]* @"\01LC1886", i32 0, i32 0), i32 %tmp.0.i) nounwind		; <i32> [#uses=0]
	%.pre.i = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4		; <i32> [#uses=1]
	%54 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 3), align 4		; <i8*> [#uses=2]
	%55 = getelementptr i8* %54, i32 %.pre.i		; <i8*> [#uses=1]
	tail call void @llvm.memcpy.i32(i8* %55, i8* %54, i32 %tmp.0.i, i32 1) nounwind
	%56 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4		; <i32> [#uses=1]
	%57 = add i32 %56, %tmp.0.i		; <i32> [#uses=4]
	store i32 %57, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4
	%58 = icmp slt i32 %57, %11		; <i1> [#uses=1]
	br i1 %58, label %bb14.i, label %bb10

bb10:		; preds = %bb14.i, %bb7.i81
	%59 = phi i32 [ %48, %bb7.i81 ], [ %57, %bb14.i ]		; <i32> [#uses=1]
	%60 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([31 x i8]* @"\01LC2189", i32 0, i32 0), i32 %59) nounwind		; <i32> [#uses=0]
	%61 = shl i32 %input_size.0, 10		; <i32> [#uses=1]
	%62 = malloc i8, i32 %61		; <i8*> [#uses=3]
	%63 = icmp eq i8* %62, null		; <i1> [#uses=1]
	br i1 %63, label %bb11, label %bb14.preheader

bb14.preheader:		; preds = %bb10
	%64 = icmp sgt i32 %11, 0		; <i1> [#uses=1]
	br i1 %64, label %bb13, label %bb15

bb11:		; preds = %bb10
	%65 = tail call i32 @puts(i8* getelementptr ([30 x i8]* @"\01LC2290", i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	ret i32 1

bb13:		; preds = %bb13, %bb14.preheader
	%i.01 = phi i32 [ %68, %bb13 ], [ 0, %bb14.preheader ]		; <i32> [#uses=3]
	%scevgep5 = getelementptr i8* %62, i32 %i.01		; <i8*> [#uses=1]
	%tmp6 = mul i32 %i.01, 1027		; <i32> [#uses=2]
	%66 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 3), align 4		; <i8*> [#uses=1]
	%scevgep7 = getelementptr i8* %66, i32 %tmp6		; <i8*> [#uses=1]
	%67 = load i8* %scevgep7, align 1		; <i8> [#uses=1]
	store i8 %67, i8* %scevgep5, align 1
	%68 = add i32 %i.01, 1		; <i32> [#uses=1]
	%phitmp = add i32 %tmp6, 1027		; <i32> [#uses=1]
	%69 = icmp slt i32 %phitmp, %11		; <i1> [#uses=1]
	br i1 %69, label %bb13, label %bb15

bb15:		; preds = %bb13, %bb14.preheader
	store i32 9, i32* @blockSize100k, align 4
	store i1 false, i1* @bsStream.b
	%70 = malloc [900021 x i8]		; <[900021 x i8]*> [#uses=3]
	%.sub.i = getelementptr [900021 x i8]* %70, i32 0, i32 0		; <i8*> [#uses=1]
	store i8* %.sub.i, i8** @block, align 4
	%71 = malloc [900020 x i16]		; <[900020 x i16]*> [#uses=2]
	%.sub1.i = getelementptr [900020 x i16]* %71, i32 0, i32 0		; <i16*> [#uses=1]
	store i16* %.sub1.i, i16** @quadrant, align 4
	%72 = malloc [900000 x i32]		; <[900000 x i32]*> [#uses=3]
	%.sub2.i = getelementptr [900000 x i32]* %72, i32 0, i32 0		; <i32*> [#uses=1]
	store i32* %.sub2.i, i32** @zptr, align 4
	%73 = malloc [65537 x i32]		; <[65537 x i32]*> [#uses=2]
	%.sub.i.i = getelementptr [65537 x i32]* %73, i32 0, i32 0		; <i32*> [#uses=1]
	store i32* %.sub.i.i, i32** @ftab, align 4
	%74 = icmp eq [900021 x i8]* %70, null		; <i1> [#uses=1]
	%75 = icmp eq [900020 x i16]* %71, null		; <i1> [#uses=1]
	%76 = icmp eq [900000 x i32]* %72, null		; <i1> [#uses=1]
	%77 = icmp eq [65537 x i32]* %73, null		; <i1> [#uses=1]
	%or.cond.i.i = or i1 %75, %74		; <i1> [#uses=1]
	%or.cond2.i.i = or i1 %or.cond.i.i, %76		; <i1> [#uses=1]
	%or.cond3.i.i = or i1 %or.cond2.i.i, %77		; <i1> [#uses=1]
	br i1 %or.cond3.i.i, label %bb3.i.i, label %spec_initbufs.exit

bb3.i.i:		; preds = %bb15
	%78 = load %struct.FILE** @__stderrp, align 4		; <%struct.FILE*> [#uses=1]
	%79 = tail call i32 (%struct.FILE*, i8*, ...)* @fprintf(%struct.FILE* %78, i8* getelementptr ([146 x i8]* @"\01LC5", i32 0, i32 0), i8* null, i32 6562209, i32 900000) nounwind		; <i32> [#uses=0]
	%80 = load %struct.FILE** @__stderrp, align 4		; <%struct.FILE*> [#uses=1]
	%81 = tail call i32 (%struct.FILE*, i8*, ...)* @fprintf(%struct.FILE* %80, i8* getelementptr ([36 x i8]* @"\01LC", i32 0, i32 0), i8* getelementptr ([1024 x i8]* @inName, i32 0, i32 0), i8* getelementptr ([1024 x i8]* @outName, i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	tail call fastcc void @cleanUpAndFail(i32 1) nounwind ssp
	unreachable

spec_initbufs.exit:		; preds = %bb15
	%82 = getelementptr [900021 x i8]* %70, i32 0, i32 1		; <i8*> [#uses=1]
	store i8* %82, i8** @block, align 4
	%83 = bitcast [900000 x i32]* %72 to i16*		; <i16*> [#uses=1]
	store i16* %83, i16** @szptr, align 4
	%84 = getelementptr [6 x i8]* %pos.i.i.i, i32 0, i32 0		; <i8*> [#uses=1]
	%85 = getelementptr [256 x i8]* %yy.i.i, i32 0, i32 0		; <i8*> [#uses=2]
	%86 = getelementptr [257 x i32]* %cftab.i.i, i32 0, i32 0		; <i32*> [#uses=1]
	%87 = getelementptr [6 x i16]* %cost.i.i, i32 0, i32 0		; <i16*> [#uses=1]
	%88 = getelementptr [6 x i16]* %cost.i.i, i32 0, i32 1		; <i16*> [#uses=1]
	%89 = getelementptr [6 x i16]* %cost.i.i, i32 0, i32 2		; <i16*> [#uses=1]
	%90 = getelementptr [6 x i16]* %cost.i.i, i32 0, i32 3		; <i16*> [#uses=1]
	%91 = getelementptr [6 x i16]* %cost.i.i, i32 0, i32 4		; <i16*> [#uses=1]
	%92 = getelementptr [6 x i16]* %cost.i.i, i32 0, i32 5		; <i16*> [#uses=1]
	%93 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 0		; <i32*> [#uses=1]
	%94 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 0		; <i32*> [#uses=1]
	%95 = getelementptr [516 x i32]* %parent.i.i.i, i32 0, i32 0		; <i32*> [#uses=1]
	%96 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 1		; <i32*> [#uses=4]
	%97 = getelementptr [6 x i8]* %pos.i.i, i32 0, i32 0		; <i8*> [#uses=2]
	br label %bb32

bb18:		; preds = %bb32
	%98 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([34 x i8]* @"\01LC2391", i32 0, i32 0), i32 %level.0) nounwind		; <i32> [#uses=0]
	store i32 %level.0, i32* @blockSize100k, align 4
	store i1 true, i1* @bsStream.b
	store i32 0, i32* @bsLive, align 4
	store i32 0, i32* @bsBuff, align 4
	store i32 0, i32* @bytesIn, align 4
	br label %bb1.i.i9

bb.i.i8:		; preds = %bb1.i.i9
	%99 = lshr i32 %110, 24		; <i32> [#uses=1]
	%100 = trunc i32 %99 to i8		; <i8> [#uses=1]
	%101 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 3), align 4		; <i8*> [#uses=1]
	%102 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 2), align 8		; <i32> [#uses=2]
	%103 = getelementptr i8* %101, i32 %102		; <i8*> [#uses=1]
	store i8 %100, i8* %103, align 1
	%104 = add i32 %102, 1		; <i32> [#uses=1]
	store i32 %104, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 2), align 8
	%105 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 1), align 4		; <i32> [#uses=1]
	%106 = add i32 %105, 1		; <i32> [#uses=1]
	store i32 %106, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 1), align 4
	%107 = shl i32 %bsBuff.tmp.0496, 8		; <i32> [#uses=2]
	%108 = add i32 %bsLive.tmp.0497, -8		; <i32> [#uses=3]
	%phitmp86 = icmp sgt i32 %108, 7		; <i1> [#uses=1]
	br label %bb1.i.i9

bb1.i.i9:		; preds = %bb.i.i8, %bb18
	%bsLive.tmp.0497 = phi i32 [ 0, %bb18 ], [ %108, %bb.i.i8 ]		; <i32> [#uses=1]
	%bsBuff.tmp.0496 = phi i32 [ 0, %bb18 ], [ %107, %bb.i.i8 ]		; <i32> [#uses=2]
	%109 = phi i32 [ %108, %bb.i.i8 ], [ 0, %bb18 ]		; <i32> [#uses=2]
	%110 = phi i32 [ %107, %bb.i.i8 ], [ 0, %bb18 ]		; <i32> [#uses=1]
	%111 = phi i1 [ %phitmp86, %bb.i.i8 ], [ false, %bb18 ]		; <i1> [#uses=1]
	br i1 %111, label %bb.i.i8, label %bsW.exit.i

bsW.exit.i:		; preds = %bb1.i.i9
	%112 = sub i32 24, %109		; <i32> [#uses=1]
	%113 = shl i32 66, %112		; <i32> [#uses=1]
	%114 = or i32 %113, %bsBuff.tmp.0496		; <i32> [#uses=3]
	store i32 %114, i32* @bsBuff, align 4
	%115 = add i32 %109, 8		; <i32> [#uses=4]
	store i32 %115, i32* @bsLive, align 4
	%.b.i68.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%116 = zext i1 %.b.i68.i to i32		; <i32> [#uses=3]
	%117 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %116, i32 3		; <i8**> [#uses=1]
	%118 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %116, i32 2		; <i32*> [#uses=2]
	%119 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %116, i32 1		; <i32*> [#uses=2]
	br label %bb1.i70.i

bb.i69.i:		; preds = %bb1.i70.i
	%120 = lshr i32 %131, 24		; <i32> [#uses=1]
	%121 = trunc i32 %120 to i8		; <i8> [#uses=1]
	%122 = load i8** %117, align 4		; <i8*> [#uses=1]
	%123 = load i32* %118, align 8		; <i32> [#uses=2]
	%124 = getelementptr i8* %122, i32 %123		; <i8*> [#uses=1]
	store i8 %121, i8* %124, align 1
	%125 = add i32 %123, 1		; <i32> [#uses=1]
	store i32 %125, i32* %118, align 8
	%126 = load i32* %119, align 4		; <i32> [#uses=1]
	%127 = add i32 %126, 1		; <i32> [#uses=1]
	store i32 %127, i32* %119, align 4
	%128 = shl i32 %bsBuff.tmp.0500, 8		; <i32> [#uses=2]
	%129 = add i32 %bsLive.tmp.0501, -8		; <i32> [#uses=3]
	br label %bb1.i70.i

bb1.i70.i:		; preds = %bb.i69.i, %bsW.exit.i
	%bsLive.tmp.0501 = phi i32 [ %115, %bsW.exit.i ], [ %129, %bb.i69.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0500 = phi i32 [ %114, %bsW.exit.i ], [ %128, %bb.i69.i ]		; <i32> [#uses=2]
	%130 = phi i32 [ %129, %bb.i69.i ], [ %115, %bsW.exit.i ]		; <i32> [#uses=2]
	%131 = phi i32 [ %128, %bb.i69.i ], [ %114, %bsW.exit.i ]		; <i32> [#uses=1]
	%132 = phi i32 [ %129, %bb.i69.i ], [ %115, %bsW.exit.i ]		; <i32> [#uses=1]
	%133 = icmp sgt i32 %132, 7		; <i1> [#uses=1]
	br i1 %133, label %bb.i69.i, label %bsW.exit71.i

bsW.exit71.i:		; preds = %bb1.i70.i
	%134 = sub i32 24, %130		; <i32> [#uses=1]
	%135 = shl i32 90, %134		; <i32> [#uses=1]
	%136 = or i32 %135, %bsBuff.tmp.0500		; <i32> [#uses=3]
	store i32 %136, i32* @bsBuff, align 4
	%137 = add i32 %130, 8		; <i32> [#uses=4]
	store i32 %137, i32* @bsLive, align 4
	%.b.i75.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%138 = zext i1 %.b.i75.i to i32		; <i32> [#uses=3]
	%139 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %138, i32 3		; <i8**> [#uses=1]
	%140 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %138, i32 2		; <i32*> [#uses=2]
	%141 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %138, i32 1		; <i32*> [#uses=2]
	br label %bb1.i78.i

bb.i76.i:		; preds = %bb1.i78.i
	%142 = lshr i32 %153, 24		; <i32> [#uses=1]
	%143 = trunc i32 %142 to i8		; <i8> [#uses=1]
	%144 = load i8** %139, align 4		; <i8*> [#uses=1]
	%145 = load i32* %140, align 8		; <i32> [#uses=2]
	%146 = getelementptr i8* %144, i32 %145		; <i8*> [#uses=1]
	store i8 %143, i8* %146, align 1
	%147 = add i32 %145, 1		; <i32> [#uses=1]
	store i32 %147, i32* %140, align 8
	%148 = load i32* %141, align 4		; <i32> [#uses=1]
	%149 = add i32 %148, 1		; <i32> [#uses=1]
	store i32 %149, i32* %141, align 4
	%150 = shl i32 %bsBuff.tmp.0504, 8		; <i32> [#uses=2]
	%151 = add i32 %bsLive.tmp.0505, -8		; <i32> [#uses=3]
	br label %bb1.i78.i

bb1.i78.i:		; preds = %bb.i76.i, %bsW.exit71.i
	%bsLive.tmp.0505 = phi i32 [ %137, %bsW.exit71.i ], [ %151, %bb.i76.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0504 = phi i32 [ %136, %bsW.exit71.i ], [ %150, %bb.i76.i ]		; <i32> [#uses=2]
	%152 = phi i32 [ %151, %bb.i76.i ], [ %137, %bsW.exit71.i ]		; <i32> [#uses=2]
	%153 = phi i32 [ %150, %bb.i76.i ], [ %136, %bsW.exit71.i ]		; <i32> [#uses=1]
	%154 = phi i32 [ %151, %bb.i76.i ], [ %137, %bsW.exit71.i ]		; <i32> [#uses=1]
	%155 = icmp sgt i32 %154, 7		; <i1> [#uses=1]
	br i1 %155, label %bb.i76.i, label %bsW.exit79.i

bsW.exit79.i:		; preds = %bb1.i78.i
	%156 = sub i32 24, %152		; <i32> [#uses=1]
	%157 = shl i32 104, %156		; <i32> [#uses=1]
	%158 = or i32 %157, %bsBuff.tmp.0504		; <i32> [#uses=3]
	store i32 %158, i32* @bsBuff, align 4
	%159 = add i32 %152, 8		; <i32> [#uses=4]
	store i32 %159, i32* @bsLive, align 4
	%160 = load i32* @blockSize100k, align 4		; <i32> [#uses=1]
	%161 = add i32 %160, 48		; <i32> [#uses=1]
	%162 = and i32 %161, 255		; <i32> [#uses=1]
	%.b.i189.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%163 = zext i1 %.b.i189.i to i32		; <i32> [#uses=3]
	%164 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %163, i32 3		; <i8**> [#uses=1]
	%165 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %163, i32 2		; <i32*> [#uses=2]
	%166 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %163, i32 1		; <i32*> [#uses=2]
	br label %bb1.i192.i

bb.i190.i:		; preds = %bb1.i192.i
	%167 = lshr i32 %178, 24		; <i32> [#uses=1]
	%168 = trunc i32 %167 to i8		; <i8> [#uses=1]
	%169 = load i8** %164, align 4		; <i8*> [#uses=1]
	%170 = load i32* %165, align 8		; <i32> [#uses=2]
	%171 = getelementptr i8* %169, i32 %170		; <i8*> [#uses=1]
	store i8 %168, i8* %171, align 1
	%172 = add i32 %170, 1		; <i32> [#uses=1]
	store i32 %172, i32* %165, align 8
	%173 = load i32* %166, align 4		; <i32> [#uses=1]
	%174 = add i32 %173, 1		; <i32> [#uses=1]
	store i32 %174, i32* %166, align 4
	%175 = shl i32 %bsBuff.tmp.0508, 8		; <i32> [#uses=2]
	%176 = add i32 %bsLive.tmp.0509, -8		; <i32> [#uses=3]
	br label %bb1.i192.i

bb1.i192.i:		; preds = %bb.i190.i, %bsW.exit79.i
	%bsLive.tmp.0509 = phi i32 [ %159, %bsW.exit79.i ], [ %176, %bb.i190.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0508 = phi i32 [ %158, %bsW.exit79.i ], [ %175, %bb.i190.i ]		; <i32> [#uses=2]
	%177 = phi i32 [ %176, %bb.i190.i ], [ %159, %bsW.exit79.i ]		; <i32> [#uses=2]
	%178 = phi i32 [ %175, %bb.i190.i ], [ %158, %bsW.exit79.i ]		; <i32> [#uses=1]
	%179 = phi i32 [ %176, %bb.i190.i ], [ %159, %bsW.exit79.i ]		; <i32> [#uses=1]
	%180 = icmp sgt i32 %179, 7		; <i1> [#uses=1]
	br i1 %180, label %bb.i190.i, label %bsW.exit193.i

bsW.exit193.i:		; preds = %bb1.i192.i
	%181 = sub i32 24, %177		; <i32> [#uses=1]
	%182 = shl i32 %162, %181		; <i32> [#uses=1]
	%183 = or i32 %182, %bsBuff.tmp.0508		; <i32> [#uses=1]
	store i32 %183, i32* @bsBuff, align 4
	%184 = add i32 %177, 8		; <i32> [#uses=1]
	store i32 %184, i32* @bsLive, align 4
	store i32 -1, i32* @globalCrc, align 4
	tail call fastcc void @loadAndRLEsource() nounwind ssp
	%185 = load i32* @last, align 4		; <i32> [#uses=2]
	%186 = icmp eq i32 %185, -1		; <i1> [#uses=1]
	br i1 %186, label %bb8.i70, label %bb2.i11

bb2.i11:		; preds = %sendMTFValues.exit.i, %bsW.exit193.i
	%187 = phi i32 [ %1058, %sendMTFValues.exit.i ], [ %185, %bsW.exit193.i ]		; <i32> [#uses=1]
	%combinedCRC.01.i = phi i32 [ %192, %sendMTFValues.exit.i ], [ 0, %bsW.exit193.i ]		; <i32> [#uses=2]
	%188 = load i32* @globalCrc, align 4		; <i32> [#uses=1]
	%not.i.i10 = xor i32 %188, -1		; <i32> [#uses=5]
	%189 = lshr i32 %combinedCRC.01.i, 31		; <i32> [#uses=1]
	%190 = shl i32 %combinedCRC.01.i, 1		; <i32> [#uses=1]
	%191 = or i32 %189, %190		; <i32> [#uses=1]
	%192 = xor i32 %191, %not.i.i10		; <i32> [#uses=2]
	%193 = mul i32 %187, 30		; <i32> [#uses=1]
	store i32 %193, i32* @workLimit, align 4
	store i32 0, i32* @workDone, align 4
	store i1 true, i1* @firstAttempt.b
	tail call fastcc void @sortIt() nounwind ssp
	%194 = load i32* @workDone, align 4		; <i32> [#uses=1]
	%195 = load i32* @workLimit, align 4		; <i32> [#uses=1]
	%196 = icmp sgt i32 %194, %195		; <i1> [#uses=1]
	br i1 %196, label %bb4.i174.i, label %bb9.i182.i

bb4.i174.i:		; preds = %bb2.i11
	%.b.i173.i = load i1* @firstAttempt.b		; <i1> [#uses=1]
	br i1 %.b.i173.i, label %bb.i.i176.i, label %bb9.i182.i

bb.i.i176.i:		; preds = %bb.i.i176.i, %bb4.i174.i
	%i.04.i.i.i = phi i32 [ %197, %bb.i.i176.i ], [ 0, %bb4.i174.i ]		; <i32> [#uses=2]
	%scevgep8.i.i.i = getelementptr [256 x i8]* @inUse, i32 0, i32 %i.04.i.i.i		; <i8*> [#uses=1]
	store i8 0, i8* %scevgep8.i.i.i, align 1
	%197 = add i32 %i.04.i.i.i, 1		; <i32> [#uses=2]
	%exitcond.i175.i = icmp eq i32 %197, 256		; <i1> [#uses=1]
	br i1 %exitcond.i175.i, label %bb7.loopexit.i.i.i, label %bb.i.i176.i

bb3.i.i178.i:		; preds = %bb3.i.i178.i.preheader, %bb6.i.i.i14
	%rTPos.13.i.i.i = phi i32 [ %rTPos.0.i.i.i, %bb6.i.i.i14 ], [ 0, %bb3.i.i178.i.preheader ]		; <i32> [#uses=3]
	%rNToGo.12.i.i.i = phi i32 [ %203, %bb6.i.i.i14 ], [ 0, %bb3.i.i178.i.preheader ]		; <i32> [#uses=2]
	%i.11.i.i.i = phi i32 [ %tmp.i.i177.i, %bb6.i.i.i14 ], [ 0, %bb3.i.i178.i.preheader ]		; <i32> [#uses=2]
	%tmp.i.i177.i = add i32 %i.11.i.i.i, 1		; <i32> [#uses=2]
	%198 = icmp eq i32 %rNToGo.12.i.i.i, 0		; <i1> [#uses=1]
	br i1 %198, label %bb4.i.i.i13, label %bb6.i.i.i14

bb4.i.i.i13:		; preds = %bb3.i.i178.i
	%199 = getelementptr [512 x i32]* @rNums, i32 0, i32 %rTPos.13.i.i.i		; <i32*> [#uses=1]
	%200 = load i32* %199, align 4		; <i32> [#uses=2]
	%201 = add i32 %rTPos.13.i.i.i, 1		; <i32> [#uses=2]
	%202 = icmp eq i32 %201, 512		; <i1> [#uses=1]
	br i1 %202, label %bb5.i.i179.i, label %bb6.i.i.i14

bb5.i.i179.i:		; preds = %bb4.i.i.i13
	br label %bb6.i.i.i14

bb6.i.i.i14:		; preds = %bb5.i.i179.i, %bb4.i.i.i13, %bb3.i.i178.i
	%rNToGo.0.i.i.i = phi i32 [ %200, %bb5.i.i179.i ], [ %rNToGo.12.i.i.i, %bb3.i.i178.i ], [ %200, %bb4.i.i.i13 ]		; <i32> [#uses=1]
	%rTPos.0.i.i.i = phi i32 [ 0, %bb5.i.i179.i ], [ %rTPos.13.i.i.i, %bb3.i.i178.i ], [ %201, %bb4.i.i.i13 ]		; <i32> [#uses=1]
	%203 = add i32 %rNToGo.0.i.i.i, -1		; <i32> [#uses=2]
	%scevgep.i180.i = getelementptr i8* %213, i32 %i.11.i.i.i		; <i8*> [#uses=2]
	%204 = load i8* %scevgep.i180.i, align 1		; <i8> [#uses=1]
	%205 = icmp eq i32 %203, 1		; <i1> [#uses=1]
	%206 = zext i1 %205 to i8		; <i8> [#uses=1]
	%207 = xor i8 %204, %206		; <i8> [#uses=2]
	store i8 %207, i8* %scevgep.i180.i, align 1
	%208 = zext i8 %207 to i32		; <i32> [#uses=1]
	%209 = getelementptr [256 x i8]* @inUse, i32 0, i32 %208		; <i8*> [#uses=1]
	store i8 1, i8* %209, align 1
	%210 = icmp sgt i32 %tmp.i.i177.i, %211		; <i1> [#uses=1]
	br i1 %210, label %randomiseBlock.exit.i.i, label %bb3.i.i178.i

bb7.loopexit.i.i.i:		; preds = %bb.i.i176.i
	%211 = load i32* @last, align 4		; <i32> [#uses=2]
	%212 = icmp slt i32 %211, 0		; <i1> [#uses=1]
	br i1 %212, label %randomiseBlock.exit.i.i, label %bb3.i.i178.i.preheader

bb3.i.i178.i.preheader:		; preds = %bb7.loopexit.i.i.i
	%213 = load i8** @block, align 4		; <i8*> [#uses=1]
	br label %bb3.i.i178.i

randomiseBlock.exit.i.i:		; preds = %bb7.loopexit.i.i.i, %bb6.i.i.i14
	store i32 0, i32* @workDone, align 4
	store i32 0, i32* @workLimit, align 4
	store i1 false, i1* @firstAttempt.b
	tail call fastcc void @sortIt() nounwind ssp
	br label %bb9.i182.i

bb9.i182.i:		; preds = %randomiseBlock.exit.i.i, %bb4.i174.i, %bb2.i11
	%blockRandomised.0 = phi i8 [ 1, %randomiseBlock.exit.i.i ], [ 0, %bb4.i174.i ], [ 0, %bb2.i11 ]		; <i8> [#uses=1]
	%214 = load i32* @last, align 4		; <i32> [#uses=1]
	%215 = load i32** @zptr, align 4		; <i32*> [#uses=1]
	br label %bb13.i.i16

bb10.i183.i:		; preds = %bb13.i.i16
	%scevgep4.i.i = getelementptr i32* %215, i32 %219		; <i32*> [#uses=1]
	%216 = load i32* %scevgep4.i.i, align 4		; <i32> [#uses=1]
	%217 = icmp eq i32 %216, 0		; <i1> [#uses=1]
	br i1 %217, label %bb14.i185.i, label %bb12.i184.i

bb12.i184.i:		; preds = %bb10.i183.i
	%218 = add i32 %219, 1		; <i32> [#uses=1]
	br label %bb13.i.i16

bb13.i.i16:		; preds = %bb12.i184.i, %bb9.i182.i
	%219 = phi i32 [ 0, %bb9.i182.i ], [ %218, %bb12.i184.i ]		; <i32> [#uses=5]
	%220 = icmp sgt i32 %219, %214		; <i1> [#uses=1]
	br i1 %220, label %bb15.i186.i, label %bb10.i183.i

bb14.i185.i:		; preds = %bb10.i183.i
	%221 = icmp eq i32 %219, -1		; <i1> [#uses=1]
	br i1 %221, label %bb15.i186.i, label %doReversibleTransformation.exit.i

bb15.i186.i:		; preds = %bb14.i185.i, %bb13.i.i16
	tail call fastcc void @panic(i8* getelementptr ([27 x i8]* @"\01LC32", i32 0, i32 0)) nounwind ssp
	unreachable

doReversibleTransformation.exit.i:		; preds = %bb14.i185.i
	%.pr.i164.i = load i32* @bsLive		; <i32> [#uses=3]
	%.pre.i165.i = load i32* @bsBuff, align 4		; <i32> [#uses=2]
	%.b.i166.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%222 = zext i1 %.b.i166.i to i32		; <i32> [#uses=3]
	%223 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %222, i32 3		; <i8**> [#uses=1]
	%224 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %222, i32 2		; <i32*> [#uses=2]
	%225 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %222, i32 1		; <i32*> [#uses=2]
	br label %bb1.i169.i

bb.i167.i:		; preds = %bb1.i169.i
	%226 = lshr i32 %237, 24		; <i32> [#uses=1]
	%227 = trunc i32 %226 to i8		; <i8> [#uses=1]
	%228 = load i8** %223, align 4		; <i8*> [#uses=1]
	%229 = load i32* %224, align 8		; <i32> [#uses=2]
	%230 = getelementptr i8* %228, i32 %229		; <i8*> [#uses=1]
	store i8 %227, i8* %230, align 1
	%231 = add i32 %229, 1		; <i32> [#uses=1]
	store i32 %231, i32* %224, align 8
	%232 = load i32* %225, align 4		; <i32> [#uses=1]
	%233 = add i32 %232, 1		; <i32> [#uses=1]
	store i32 %233, i32* %225, align 4
	%234 = shl i32 %bsBuff.tmp.0408, 8		; <i32> [#uses=2]
	%235 = add i32 %bsLive.tmp.0409, -8		; <i32> [#uses=3]
	br label %bb1.i169.i

bb1.i169.i:		; preds = %bb.i167.i, %doReversibleTransformation.exit.i
	%bsLive.tmp.0409 = phi i32 [ %.pr.i164.i, %doReversibleTransformation.exit.i ], [ %235, %bb.i167.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0408 = phi i32 [ %.pre.i165.i, %doReversibleTransformation.exit.i ], [ %234, %bb.i167.i ]		; <i32> [#uses=2]
	%236 = phi i32 [ %235, %bb.i167.i ], [ %.pr.i164.i, %doReversibleTransformation.exit.i ]		; <i32> [#uses=2]
	%237 = phi i32 [ %234, %bb.i167.i ], [ %.pre.i165.i, %doReversibleTransformation.exit.i ]		; <i32> [#uses=1]
	%238 = phi i32 [ %235, %bb.i167.i ], [ %.pr.i164.i, %doReversibleTransformation.exit.i ]		; <i32> [#uses=1]
	%239 = icmp sgt i32 %238, 7		; <i1> [#uses=1]
	br i1 %239, label %bb.i167.i, label %bsW.exit170.i

bsW.exit170.i:		; preds = %bb1.i169.i
	%240 = sub i32 24, %236		; <i32> [#uses=1]
	%241 = shl i32 49, %240		; <i32> [#uses=1]
	%242 = or i32 %241, %bsBuff.tmp.0408		; <i32> [#uses=3]
	store i32 %242, i32* @bsBuff, align 4
	%243 = add i32 %236, 8		; <i32> [#uses=4]
	store i32 %243, i32* @bsLive, align 4
	%.b.i159.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%244 = zext i1 %.b.i159.i to i32		; <i32> [#uses=3]
	%245 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %244, i32 3		; <i8**> [#uses=1]
	%246 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %244, i32 2		; <i32*> [#uses=2]
	%247 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %244, i32 1		; <i32*> [#uses=2]
	br label %bb1.i162.i

bb.i160.i:		; preds = %bb1.i162.i
	%248 = lshr i32 %259, 24		; <i32> [#uses=1]
	%249 = trunc i32 %248 to i8		; <i8> [#uses=1]
	%250 = load i8** %245, align 4		; <i8*> [#uses=1]
	%251 = load i32* %246, align 8		; <i32> [#uses=2]
	%252 = getelementptr i8* %250, i32 %251		; <i8*> [#uses=1]
	store i8 %249, i8* %252, align 1
	%253 = add i32 %251, 1		; <i32> [#uses=1]
	store i32 %253, i32* %246, align 8
	%254 = load i32* %247, align 4		; <i32> [#uses=1]
	%255 = add i32 %254, 1		; <i32> [#uses=1]
	store i32 %255, i32* %247, align 4
	%256 = shl i32 %bsBuff.tmp.0412, 8		; <i32> [#uses=2]
	%257 = add i32 %bsLive.tmp.0413, -8		; <i32> [#uses=3]
	br label %bb1.i162.i

bb1.i162.i:		; preds = %bb.i160.i, %bsW.exit170.i
	%bsLive.tmp.0413 = phi i32 [ %243, %bsW.exit170.i ], [ %257, %bb.i160.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0412 = phi i32 [ %242, %bsW.exit170.i ], [ %256, %bb.i160.i ]		; <i32> [#uses=2]
	%258 = phi i32 [ %257, %bb.i160.i ], [ %243, %bsW.exit170.i ]		; <i32> [#uses=2]
	%259 = phi i32 [ %256, %bb.i160.i ], [ %242, %bsW.exit170.i ]		; <i32> [#uses=1]
	%260 = phi i32 [ %257, %bb.i160.i ], [ %243, %bsW.exit170.i ]		; <i32> [#uses=1]
	%261 = icmp sgt i32 %260, 7		; <i1> [#uses=1]
	br i1 %261, label %bb.i160.i, label %bsW.exit163.i

bsW.exit163.i:		; preds = %bb1.i162.i
	%262 = sub i32 24, %258		; <i32> [#uses=1]
	%263 = shl i32 65, %262		; <i32> [#uses=1]
	%264 = or i32 %263, %bsBuff.tmp.0412		; <i32> [#uses=3]
	store i32 %264, i32* @bsBuff, align 4
	%265 = add i32 %258, 8		; <i32> [#uses=4]
	store i32 %265, i32* @bsLive, align 4
	%.b.i152.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%266 = zext i1 %.b.i152.i to i32		; <i32> [#uses=3]
	%267 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %266, i32 3		; <i8**> [#uses=1]
	%268 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %266, i32 2		; <i32*> [#uses=2]
	%269 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %266, i32 1		; <i32*> [#uses=2]
	br label %bb1.i155.i

bb.i153.i:		; preds = %bb1.i155.i
	%270 = lshr i32 %281, 24		; <i32> [#uses=1]
	%271 = trunc i32 %270 to i8		; <i8> [#uses=1]
	%272 = load i8** %267, align 4		; <i8*> [#uses=1]
	%273 = load i32* %268, align 8		; <i32> [#uses=2]
	%274 = getelementptr i8* %272, i32 %273		; <i8*> [#uses=1]
	store i8 %271, i8* %274, align 1
	%275 = add i32 %273, 1		; <i32> [#uses=1]
	store i32 %275, i32* %268, align 8
	%276 = load i32* %269, align 4		; <i32> [#uses=1]
	%277 = add i32 %276, 1		; <i32> [#uses=1]
	store i32 %277, i32* %269, align 4
	%278 = shl i32 %bsBuff.tmp.0416, 8		; <i32> [#uses=2]
	%279 = add i32 %bsLive.tmp.0417, -8		; <i32> [#uses=3]
	br label %bb1.i155.i

bb1.i155.i:		; preds = %bb.i153.i, %bsW.exit163.i
	%bsLive.tmp.0417 = phi i32 [ %265, %bsW.exit163.i ], [ %279, %bb.i153.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0416 = phi i32 [ %264, %bsW.exit163.i ], [ %278, %bb.i153.i ]		; <i32> [#uses=2]
	%280 = phi i32 [ %279, %bb.i153.i ], [ %265, %bsW.exit163.i ]		; <i32> [#uses=2]
	%281 = phi i32 [ %278, %bb.i153.i ], [ %264, %bsW.exit163.i ]		; <i32> [#uses=1]
	%282 = phi i32 [ %279, %bb.i153.i ], [ %265, %bsW.exit163.i ]		; <i32> [#uses=1]
	%283 = icmp sgt i32 %282, 7		; <i1> [#uses=1]
	br i1 %283, label %bb.i153.i, label %bsW.exit156.i

bsW.exit156.i:		; preds = %bb1.i155.i
	%284 = sub i32 24, %280		; <i32> [#uses=1]
	%285 = shl i32 89, %284		; <i32> [#uses=1]
	%286 = or i32 %285, %bsBuff.tmp.0416		; <i32> [#uses=3]
	store i32 %286, i32* @bsBuff, align 4
	%287 = add i32 %280, 8		; <i32> [#uses=4]
	store i32 %287, i32* @bsLive, align 4
	%.b.i145.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%288 = zext i1 %.b.i145.i to i32		; <i32> [#uses=3]
	%289 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %288, i32 3		; <i8**> [#uses=1]
	%290 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %288, i32 2		; <i32*> [#uses=2]
	%291 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %288, i32 1		; <i32*> [#uses=2]
	br label %bb1.i148.i

bb.i146.i:		; preds = %bb1.i148.i
	%292 = lshr i32 %303, 24		; <i32> [#uses=1]
	%293 = trunc i32 %292 to i8		; <i8> [#uses=1]
	%294 = load i8** %289, align 4		; <i8*> [#uses=1]
	%295 = load i32* %290, align 8		; <i32> [#uses=2]
	%296 = getelementptr i8* %294, i32 %295		; <i8*> [#uses=1]
	store i8 %293, i8* %296, align 1
	%297 = add i32 %295, 1		; <i32> [#uses=1]
	store i32 %297, i32* %290, align 8
	%298 = load i32* %291, align 4		; <i32> [#uses=1]
	%299 = add i32 %298, 1		; <i32> [#uses=1]
	store i32 %299, i32* %291, align 4
	%300 = shl i32 %bsBuff.tmp.0420, 8		; <i32> [#uses=2]
	%301 = add i32 %bsLive.tmp.0421, -8		; <i32> [#uses=3]
	br label %bb1.i148.i

bb1.i148.i:		; preds = %bb.i146.i, %bsW.exit156.i
	%bsLive.tmp.0421 = phi i32 [ %287, %bsW.exit156.i ], [ %301, %bb.i146.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0420 = phi i32 [ %286, %bsW.exit156.i ], [ %300, %bb.i146.i ]		; <i32> [#uses=2]
	%302 = phi i32 [ %301, %bb.i146.i ], [ %287, %bsW.exit156.i ]		; <i32> [#uses=2]
	%303 = phi i32 [ %300, %bb.i146.i ], [ %286, %bsW.exit156.i ]		; <i32> [#uses=1]
	%304 = phi i32 [ %301, %bb.i146.i ], [ %287, %bsW.exit156.i ]		; <i32> [#uses=1]
	%305 = icmp sgt i32 %304, 7		; <i1> [#uses=1]
	br i1 %305, label %bb.i146.i, label %bsW.exit149.i

bsW.exit149.i:		; preds = %bb1.i148.i
	%306 = sub i32 24, %302		; <i32> [#uses=1]
	%307 = shl i32 38, %306		; <i32> [#uses=1]
	%308 = or i32 %307, %bsBuff.tmp.0420		; <i32> [#uses=3]
	store i32 %308, i32* @bsBuff, align 4
	%309 = add i32 %302, 8		; <i32> [#uses=4]
	store i32 %309, i32* @bsLive, align 4
	%.b.i138.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%310 = zext i1 %.b.i138.i to i32		; <i32> [#uses=3]
	%311 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %310, i32 3		; <i8**> [#uses=1]
	%312 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %310, i32 2		; <i32*> [#uses=2]
	%313 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %310, i32 1		; <i32*> [#uses=2]
	br label %bb1.i141.i

bb.i139.i:		; preds = %bb1.i141.i
	%314 = lshr i32 %325, 24		; <i32> [#uses=1]
	%315 = trunc i32 %314 to i8		; <i8> [#uses=1]
	%316 = load i8** %311, align 4		; <i8*> [#uses=1]
	%317 = load i32* %312, align 8		; <i32> [#uses=2]
	%318 = getelementptr i8* %316, i32 %317		; <i8*> [#uses=1]
	store i8 %315, i8* %318, align 1
	%319 = add i32 %317, 1		; <i32> [#uses=1]
	store i32 %319, i32* %312, align 8
	%320 = load i32* %313, align 4		; <i32> [#uses=1]
	%321 = add i32 %320, 1		; <i32> [#uses=1]
	store i32 %321, i32* %313, align 4
	%322 = shl i32 %bsBuff.tmp.0424, 8		; <i32> [#uses=2]
	%323 = add i32 %bsLive.tmp.0425, -8		; <i32> [#uses=3]
	br label %bb1.i141.i

bb1.i141.i:		; preds = %bb.i139.i, %bsW.exit149.i
	%bsLive.tmp.0425 = phi i32 [ %309, %bsW.exit149.i ], [ %323, %bb.i139.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0424 = phi i32 [ %308, %bsW.exit149.i ], [ %322, %bb.i139.i ]		; <i32> [#uses=2]
	%324 = phi i32 [ %323, %bb.i139.i ], [ %309, %bsW.exit149.i ]		; <i32> [#uses=2]
	%325 = phi i32 [ %322, %bb.i139.i ], [ %308, %bsW.exit149.i ]		; <i32> [#uses=1]
	%326 = phi i32 [ %323, %bb.i139.i ], [ %309, %bsW.exit149.i ]		; <i32> [#uses=1]
	%327 = icmp sgt i32 %326, 7		; <i1> [#uses=1]
	br i1 %327, label %bb.i139.i, label %bsW.exit142.i

bsW.exit142.i:		; preds = %bb1.i141.i
	%328 = sub i32 24, %324		; <i32> [#uses=1]
	%329 = shl i32 83, %328		; <i32> [#uses=1]
	%330 = or i32 %329, %bsBuff.tmp.0424		; <i32> [#uses=3]
	store i32 %330, i32* @bsBuff, align 4
	%331 = add i32 %324, 8		; <i32> [#uses=4]
	store i32 %331, i32* @bsLive, align 4
	%.b.i131.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%332 = zext i1 %.b.i131.i to i32		; <i32> [#uses=3]
	%333 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %332, i32 3		; <i8**> [#uses=1]
	%334 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %332, i32 2		; <i32*> [#uses=2]
	%335 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %332, i32 1		; <i32*> [#uses=2]
	br label %bb1.i134.i

bb.i132.i:		; preds = %bb1.i134.i
	%336 = lshr i32 %347, 24		; <i32> [#uses=1]
	%337 = trunc i32 %336 to i8		; <i8> [#uses=1]
	%338 = load i8** %333, align 4		; <i8*> [#uses=1]
	%339 = load i32* %334, align 8		; <i32> [#uses=2]
	%340 = getelementptr i8* %338, i32 %339		; <i8*> [#uses=1]
	store i8 %337, i8* %340, align 1
	%341 = add i32 %339, 1		; <i32> [#uses=1]
	store i32 %341, i32* %334, align 8
	%342 = load i32* %335, align 4		; <i32> [#uses=1]
	%343 = add i32 %342, 1		; <i32> [#uses=1]
	store i32 %343, i32* %335, align 4
	%344 = shl i32 %bsBuff.tmp.0428, 8		; <i32> [#uses=2]
	%345 = add i32 %bsLive.tmp.0429, -8		; <i32> [#uses=3]
	br label %bb1.i134.i

bb1.i134.i:		; preds = %bb.i132.i, %bsW.exit142.i
	%bsLive.tmp.0429 = phi i32 [ %331, %bsW.exit142.i ], [ %345, %bb.i132.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0428 = phi i32 [ %330, %bsW.exit142.i ], [ %344, %bb.i132.i ]		; <i32> [#uses=2]
	%346 = phi i32 [ %345, %bb.i132.i ], [ %331, %bsW.exit142.i ]		; <i32> [#uses=2]
	%347 = phi i32 [ %344, %bb.i132.i ], [ %330, %bsW.exit142.i ]		; <i32> [#uses=1]
	%348 = phi i32 [ %345, %bb.i132.i ], [ %331, %bsW.exit142.i ]		; <i32> [#uses=1]
	%349 = icmp sgt i32 %348, 7		; <i1> [#uses=1]
	br i1 %349, label %bb.i132.i, label %bsW.exit135.i

bsW.exit135.i:		; preds = %bb1.i134.i
	%350 = sub i32 24, %346		; <i32> [#uses=1]
	%351 = shl i32 89, %350		; <i32> [#uses=1]
	%352 = or i32 %351, %bsBuff.tmp.0428		; <i32> [#uses=3]
	store i32 %352, i32* @bsBuff, align 4
	%353 = add i32 %346, 8		; <i32> [#uses=4]
	store i32 %353, i32* @bsLive, align 4
	%354 = lshr i32 %not.i.i10, 24		; <i32> [#uses=1]
	%.b.i124.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%355 = zext i1 %.b.i124.i to i32		; <i32> [#uses=3]
	%356 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %355, i32 3		; <i8**> [#uses=1]
	%357 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %355, i32 2		; <i32*> [#uses=2]
	%358 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %355, i32 1		; <i32*> [#uses=2]
	br label %bb1.i127.i

bb.i125.i:		; preds = %bb1.i127.i
	%359 = lshr i32 %370, 24		; <i32> [#uses=1]
	%360 = trunc i32 %359 to i8		; <i8> [#uses=1]
	%361 = load i8** %356, align 4		; <i8*> [#uses=1]
	%362 = load i32* %357, align 8		; <i32> [#uses=2]
	%363 = getelementptr i8* %361, i32 %362		; <i8*> [#uses=1]
	store i8 %360, i8* %363, align 1
	%364 = add i32 %362, 1		; <i32> [#uses=1]
	store i32 %364, i32* %357, align 8
	%365 = load i32* %358, align 4		; <i32> [#uses=1]
	%366 = add i32 %365, 1		; <i32> [#uses=1]
	store i32 %366, i32* %358, align 4
	%367 = shl i32 %bsBuff.tmp.0432, 8		; <i32> [#uses=2]
	%368 = add i32 %bsLive.tmp.0433, -8		; <i32> [#uses=3]
	br label %bb1.i127.i

bb1.i127.i:		; preds = %bb.i125.i, %bsW.exit135.i
	%bsLive.tmp.0433 = phi i32 [ %353, %bsW.exit135.i ], [ %368, %bb.i125.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0432 = phi i32 [ %352, %bsW.exit135.i ], [ %367, %bb.i125.i ]		; <i32> [#uses=2]
	%369 = phi i32 [ %368, %bb.i125.i ], [ %353, %bsW.exit135.i ]		; <i32> [#uses=2]
	%370 = phi i32 [ %367, %bb.i125.i ], [ %352, %bsW.exit135.i ]		; <i32> [#uses=1]
	%371 = phi i32 [ %368, %bb.i125.i ], [ %353, %bsW.exit135.i ]		; <i32> [#uses=1]
	%372 = icmp sgt i32 %371, 7		; <i1> [#uses=1]
	br i1 %372, label %bb.i125.i, label %bsW.exit128.i

bsW.exit128.i:		; preds = %bb1.i127.i
	%373 = sub i32 24, %369		; <i32> [#uses=1]
	%374 = shl i32 %354, %373		; <i32> [#uses=1]
	%375 = or i32 %374, %bsBuff.tmp.0432		; <i32> [#uses=3]
	store i32 %375, i32* @bsBuff, align 4
	%376 = add i32 %369, 8		; <i32> [#uses=4]
	store i32 %376, i32* @bsLive, align 4
	%377 = lshr i32 %not.i.i10, 16		; <i32> [#uses=1]
	%378 = and i32 %377, 255		; <i32> [#uses=1]
	%.b.i117.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%379 = zext i1 %.b.i117.i to i32		; <i32> [#uses=3]
	%380 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %379, i32 3		; <i8**> [#uses=1]
	%381 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %379, i32 2		; <i32*> [#uses=2]
	%382 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %379, i32 1		; <i32*> [#uses=2]
	br label %bb1.i120.i

bb.i118.i:		; preds = %bb1.i120.i
	%383 = lshr i32 %394, 24		; <i32> [#uses=1]
	%384 = trunc i32 %383 to i8		; <i8> [#uses=1]
	%385 = load i8** %380, align 4		; <i8*> [#uses=1]
	%386 = load i32* %381, align 8		; <i32> [#uses=2]
	%387 = getelementptr i8* %385, i32 %386		; <i8*> [#uses=1]
	store i8 %384, i8* %387, align 1
	%388 = add i32 %386, 1		; <i32> [#uses=1]
	store i32 %388, i32* %381, align 8
	%389 = load i32* %382, align 4		; <i32> [#uses=1]
	%390 = add i32 %389, 1		; <i32> [#uses=1]
	store i32 %390, i32* %382, align 4
	%391 = shl i32 %bsBuff.tmp.0436, 8		; <i32> [#uses=2]
	%392 = add i32 %bsLive.tmp.0437, -8		; <i32> [#uses=3]
	br label %bb1.i120.i

bb1.i120.i:		; preds = %bb.i118.i, %bsW.exit128.i
	%bsLive.tmp.0437 = phi i32 [ %376, %bsW.exit128.i ], [ %392, %bb.i118.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0436 = phi i32 [ %375, %bsW.exit128.i ], [ %391, %bb.i118.i ]		; <i32> [#uses=2]
	%393 = phi i32 [ %392, %bb.i118.i ], [ %376, %bsW.exit128.i ]		; <i32> [#uses=2]
	%394 = phi i32 [ %391, %bb.i118.i ], [ %375, %bsW.exit128.i ]		; <i32> [#uses=1]
	%395 = phi i32 [ %392, %bb.i118.i ], [ %376, %bsW.exit128.i ]		; <i32> [#uses=1]
	%396 = icmp sgt i32 %395, 7		; <i1> [#uses=1]
	br i1 %396, label %bb.i118.i, label %bsW.exit121.i

bsW.exit121.i:		; preds = %bb1.i120.i
	%397 = sub i32 24, %393		; <i32> [#uses=1]
	%398 = shl i32 %378, %397		; <i32> [#uses=1]
	%399 = or i32 %398, %bsBuff.tmp.0436		; <i32> [#uses=3]
	store i32 %399, i32* @bsBuff, align 4
	%400 = add i32 %393, 8		; <i32> [#uses=4]
	store i32 %400, i32* @bsLive, align 4
	%401 = lshr i32 %not.i.i10, 8		; <i32> [#uses=1]
	%402 = and i32 %401, 255		; <i32> [#uses=1]
	%.b.i110.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%403 = zext i1 %.b.i110.i to i32		; <i32> [#uses=3]
	%404 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %403, i32 3		; <i8**> [#uses=1]
	%405 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %403, i32 2		; <i32*> [#uses=2]
	%406 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %403, i32 1		; <i32*> [#uses=2]
	br label %bb1.i113.i

bb.i111.i:		; preds = %bb1.i113.i
	%407 = lshr i32 %418, 24		; <i32> [#uses=1]
	%408 = trunc i32 %407 to i8		; <i8> [#uses=1]
	%409 = load i8** %404, align 4		; <i8*> [#uses=1]
	%410 = load i32* %405, align 8		; <i32> [#uses=2]
	%411 = getelementptr i8* %409, i32 %410		; <i8*> [#uses=1]
	store i8 %408, i8* %411, align 1
	%412 = add i32 %410, 1		; <i32> [#uses=1]
	store i32 %412, i32* %405, align 8
	%413 = load i32* %406, align 4		; <i32> [#uses=1]
	%414 = add i32 %413, 1		; <i32> [#uses=1]
	store i32 %414, i32* %406, align 4
	%415 = shl i32 %bsBuff.tmp.0440, 8		; <i32> [#uses=2]
	%416 = add i32 %bsLive.tmp.0441, -8		; <i32> [#uses=3]
	br label %bb1.i113.i

bb1.i113.i:		; preds = %bb.i111.i, %bsW.exit121.i
	%bsLive.tmp.0441 = phi i32 [ %400, %bsW.exit121.i ], [ %416, %bb.i111.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0440 = phi i32 [ %399, %bsW.exit121.i ], [ %415, %bb.i111.i ]		; <i32> [#uses=2]
	%417 = phi i32 [ %416, %bb.i111.i ], [ %400, %bsW.exit121.i ]		; <i32> [#uses=2]
	%418 = phi i32 [ %415, %bb.i111.i ], [ %399, %bsW.exit121.i ]		; <i32> [#uses=1]
	%419 = phi i32 [ %416, %bb.i111.i ], [ %400, %bsW.exit121.i ]		; <i32> [#uses=1]
	%420 = icmp sgt i32 %419, 7		; <i1> [#uses=1]
	br i1 %420, label %bb.i111.i, label %bsW.exit114.i

bsW.exit114.i:		; preds = %bb1.i113.i
	%421 = sub i32 24, %417		; <i32> [#uses=1]
	%422 = shl i32 %402, %421		; <i32> [#uses=1]
	%423 = or i32 %422, %bsBuff.tmp.0440		; <i32> [#uses=3]
	store i32 %423, i32* @bsBuff, align 4
	%424 = add i32 %417, 8		; <i32> [#uses=4]
	store i32 %424, i32* @bsLive, align 4
	%425 = and i32 %not.i.i10, 255		; <i32> [#uses=1]
	%.b.i103.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%426 = zext i1 %.b.i103.i to i32		; <i32> [#uses=3]
	%427 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %426, i32 3		; <i8**> [#uses=1]
	%428 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %426, i32 2		; <i32*> [#uses=2]
	%429 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %426, i32 1		; <i32*> [#uses=2]
	br label %bb1.i106.i

bb.i104.i:		; preds = %bb1.i106.i
	%430 = lshr i32 %441, 24		; <i32> [#uses=1]
	%431 = trunc i32 %430 to i8		; <i8> [#uses=1]
	%432 = load i8** %427, align 4		; <i8*> [#uses=1]
	%433 = load i32* %428, align 8		; <i32> [#uses=2]
	%434 = getelementptr i8* %432, i32 %433		; <i8*> [#uses=1]
	store i8 %431, i8* %434, align 1
	%435 = add i32 %433, 1		; <i32> [#uses=1]
	store i32 %435, i32* %428, align 8
	%436 = load i32* %429, align 4		; <i32> [#uses=1]
	%437 = add i32 %436, 1		; <i32> [#uses=1]
	store i32 %437, i32* %429, align 4
	%438 = shl i32 %bsBuff.tmp.0444, 8		; <i32> [#uses=2]
	%439 = add i32 %bsLive.tmp.0445, -8		; <i32> [#uses=3]
	br label %bb1.i106.i

bb1.i106.i:		; preds = %bb.i104.i, %bsW.exit114.i
	%bsLive.tmp.0445 = phi i32 [ %424, %bsW.exit114.i ], [ %439, %bb.i104.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0444 = phi i32 [ %423, %bsW.exit114.i ], [ %438, %bb.i104.i ]		; <i32> [#uses=2]
	%440 = phi i32 [ %439, %bb.i104.i ], [ %424, %bsW.exit114.i ]		; <i32> [#uses=2]
	%441 = phi i32 [ %438, %bb.i104.i ], [ %423, %bsW.exit114.i ]		; <i32> [#uses=1]
	%442 = phi i32 [ %439, %bb.i104.i ], [ %424, %bsW.exit114.i ]		; <i32> [#uses=1]
	%443 = icmp sgt i32 %442, 7		; <i1> [#uses=1]
	br i1 %443, label %bb.i104.i, label %bsW.exit107.i

bsW.exit107.i:		; preds = %bb1.i106.i
	%444 = sub i32 24, %440		; <i32> [#uses=1]
	%445 = shl i32 %425, %444		; <i32> [#uses=1]
	%446 = or i32 %445, %bsBuff.tmp.0444		; <i32> [#uses=5]
	store i32 %446, i32* @bsBuff, align 4
	%447 = add i32 %440, 8		; <i32> [#uses=7]
	store i32 %447, i32* @bsLive, align 4
	%448 = icmp eq i8 %blockRandomised.0, 0		; <i1> [#uses=1]
	%.b.i89.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%449 = zext i1 %.b.i89.i to i32		; <i32> [#uses=3]
	%450 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %449, i32 3		; <i8**> [#uses=2]
	%451 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %449, i32 2		; <i32*> [#uses=4]
	%452 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %449, i32 1		; <i32*> [#uses=4]
	br i1 %448, label %bb1.i92.i, label %bb1.i99.i

bb.i97.i:		; preds = %bb1.i99.i
	%453 = lshr i32 %464, 24		; <i32> [#uses=1]
	%454 = trunc i32 %453 to i8		; <i8> [#uses=1]
	%455 = load i8** %450, align 4		; <i8*> [#uses=1]
	%456 = load i32* %451, align 8		; <i32> [#uses=2]
	%457 = getelementptr i8* %455, i32 %456		; <i8*> [#uses=1]
	store i8 %454, i8* %457, align 1
	%458 = add i32 %456, 1		; <i32> [#uses=1]
	store i32 %458, i32* %451, align 8
	%459 = load i32* %452, align 4		; <i32> [#uses=1]
	%460 = add i32 %459, 1		; <i32> [#uses=1]
	store i32 %460, i32* %452, align 4
	%461 = shl i32 %bsBuff.tmp.0448, 8		; <i32> [#uses=2]
	%462 = add i32 %bsLive.tmp.0449, -8		; <i32> [#uses=3]
	br label %bb1.i99.i

bb1.i99.i:		; preds = %bb.i97.i, %bsW.exit107.i
	%bsLive.tmp.0449 = phi i32 [ %462, %bb.i97.i ], [ %447, %bsW.exit107.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0448 = phi i32 [ %461, %bb.i97.i ], [ %446, %bsW.exit107.i ]		; <i32> [#uses=2]
	%463 = phi i32 [ %462, %bb.i97.i ], [ %447, %bsW.exit107.i ]		; <i32> [#uses=2]
	%464 = phi i32 [ %461, %bb.i97.i ], [ %446, %bsW.exit107.i ]		; <i32> [#uses=1]
	%465 = phi i32 [ %462, %bb.i97.i ], [ %447, %bsW.exit107.i ]		; <i32> [#uses=1]
	%466 = icmp sgt i32 %465, 7		; <i1> [#uses=1]
	br i1 %466, label %bb.i97.i, label %bsW.exit100.i

bsW.exit100.i:		; preds = %bb1.i99.i
	%467 = sub i32 31, %463		; <i32> [#uses=1]
	%468 = shl i32 1, %467		; <i32> [#uses=1]
	%469 = or i32 %468, %bsBuff.tmp.0448		; <i32> [#uses=2]
	store i32 %469, i32* @bsBuff, align 4
	%470 = add i32 %463, 1		; <i32> [#uses=2]
	store i32 %470, i32* @bsLive, align 4
	br label %bb7.i18

bb.i90.i:		; preds = %bb1.i92.i
	%471 = lshr i32 %482, 24		; <i32> [#uses=1]
	%472 = trunc i32 %471 to i8		; <i8> [#uses=1]
	%473 = load i8** %450, align 4		; <i8*> [#uses=1]
	%474 = load i32* %451, align 8		; <i32> [#uses=2]
	%475 = getelementptr i8* %473, i32 %474		; <i8*> [#uses=1]
	store i8 %472, i8* %475, align 1
	%476 = add i32 %474, 1		; <i32> [#uses=1]
	store i32 %476, i32* %451, align 8
	%477 = load i32* %452, align 4		; <i32> [#uses=1]
	%478 = add i32 %477, 1		; <i32> [#uses=1]
	store i32 %478, i32* %452, align 4
	%479 = shl i32 %bsBuff.tmp.0404, 8		; <i32> [#uses=2]
	%480 = add i32 %bsLive.tmp.0405, -8		; <i32> [#uses=3]
	br label %bb1.i92.i

bb1.i92.i:		; preds = %bb.i90.i, %bsW.exit107.i
	%bsLive.tmp.0405 = phi i32 [ %480, %bb.i90.i ], [ %447, %bsW.exit107.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0404 = phi i32 [ %479, %bb.i90.i ], [ %446, %bsW.exit107.i ]		; <i32> [#uses=3]
	%481 = phi i32 [ %480, %bb.i90.i ], [ %447, %bsW.exit107.i ]		; <i32> [#uses=1]
	%482 = phi i32 [ %479, %bb.i90.i ], [ %446, %bsW.exit107.i ]		; <i32> [#uses=1]
	%483 = phi i32 [ %480, %bb.i90.i ], [ %447, %bsW.exit107.i ]		; <i32> [#uses=1]
	%484 = icmp sgt i32 %483, 7		; <i1> [#uses=1]
	br i1 %484, label %bb.i90.i, label %bsW.exit93.i

bsW.exit93.i:		; preds = %bb1.i92.i
	store i32 %bsBuff.tmp.0404, i32* @bsBuff
	%485 = add i32 %481, 1		; <i32> [#uses=2]
	store i32 %485, i32* @bsLive, align 4
	br label %bb7.i18

bb7.i18:		; preds = %bsW.exit93.i, %bsW.exit100.i
	%.pre.i81.i = phi i32 [ %bsBuff.tmp.0404, %bsW.exit93.i ], [ %469, %bsW.exit100.i ]		; <i32> [#uses=2]
	%.pr.i80.i = phi i32 [ %485, %bsW.exit93.i ], [ %470, %bsW.exit100.i ]		; <i32> [#uses=3]
	%.b.i82.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%486 = zext i1 %.b.i82.i to i32		; <i32> [#uses=3]
	%487 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %486, i32 3		; <i8**> [#uses=1]
	%488 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %486, i32 2		; <i32*> [#uses=2]
	%489 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %486, i32 1		; <i32*> [#uses=2]
	br label %bb1.i85.i

bb.i83.i:		; preds = %bb1.i85.i
	%490 = lshr i32 %501, 24		; <i32> [#uses=1]
	%491 = trunc i32 %490 to i8		; <i8> [#uses=1]
	%492 = load i8** %487, align 4		; <i8*> [#uses=1]
	%493 = load i32* %488, align 8		; <i32> [#uses=2]
	%494 = getelementptr i8* %492, i32 %493		; <i8*> [#uses=1]
	store i8 %491, i8* %494, align 1
	%495 = add i32 %493, 1		; <i32> [#uses=1]
	store i32 %495, i32* %488, align 8
	%496 = load i32* %489, align 4		; <i32> [#uses=1]
	%497 = add i32 %496, 1		; <i32> [#uses=1]
	store i32 %497, i32* %489, align 4
	%498 = shl i32 %bsBuff.tmp.0452, 8		; <i32> [#uses=2]
	%499 = add i32 %bsLive.tmp.0453, -8		; <i32> [#uses=3]
	br label %bb1.i85.i

bb1.i85.i:		; preds = %bb.i83.i, %bb7.i18
	%bsLive.tmp.0453 = phi i32 [ %.pr.i80.i, %bb7.i18 ], [ %499, %bb.i83.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0452 = phi i32 [ %.pre.i81.i, %bb7.i18 ], [ %498, %bb.i83.i ]		; <i32> [#uses=2]
	%500 = phi i32 [ %499, %bb.i83.i ], [ %.pr.i80.i, %bb7.i18 ]		; <i32> [#uses=2]
	%501 = phi i32 [ %498, %bb.i83.i ], [ %.pre.i81.i, %bb7.i18 ]		; <i32> [#uses=1]
	%502 = phi i32 [ %499, %bb.i83.i ], [ %.pr.i80.i, %bb7.i18 ]		; <i32> [#uses=1]
	%503 = icmp sgt i32 %502, 7		; <i1> [#uses=1]
	br i1 %503, label %bb.i83.i, label %bsW.exit86.i

bsW.exit86.i:		; preds = %bb1.i85.i
	%504 = sub i32 8, %500		; <i32> [#uses=1]
	%505 = shl i32 %219, %504		; <i32> [#uses=1]
	%506 = or i32 %505, %bsBuff.tmp.0452		; <i32> [#uses=1]
	store i32 %506, i32* @bsBuff, align 4
	%507 = add i32 %500, 24		; <i32> [#uses=1]
	store i32 %507, i32* @bsLive, align 4
	tail call fastcc void @generateMTFValues() nounwind ssp
	%508 = load i32* @nInUse, align 4		; <i32> [#uses=4]
	%509 = add i32 %508, 2		; <i32> [#uses=20]
	%510 = icmp sgt i32 %509, 0		; <i1> [#uses=6]
	br i1 %510, label %bb4.i.i19, label %bb8.i.i20

bb4.i.i19:		; preds = %bb4.i.i19, %bsW.exit86.i
	%v.0107.i.i = phi i32 [ %511, %bb4.i.i19 ], [ 0, %bsW.exit86.i ]		; <i32> [#uses=2]
	%scevgep268.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 0, i32 %v.0107.i.i		; <i8*> [#uses=1]
	store i8 15, i8* %scevgep268.i.i, align 1
	%511 = add i32 %v.0107.i.i, 1		; <i32> [#uses=2]
	%exitcond267.i.i = icmp eq i32 %511, %509		; <i1> [#uses=1]
	br i1 %exitcond267.i.i, label %bb4.1.i.i, label %bb4.i.i19

bb8.i.i20:		; preds = %bb4.5.i.i, %bsW.exit86.i
	%512 = load i32* @nMTF, align 4		; <i32> [#uses=8]
	%513 = icmp sgt i32 %512, 0		; <i1> [#uses=1]
	br i1 %513, label %bb10.i.i21, label %bb9.i.i

bb9.i.i:		; preds = %bb8.i.i20
	call fastcc void @panic(i8* getelementptr ([17 x i8]* @"\01LC35", i32 0, i32 0)) nounwind ssp
	unreachable

bb10.i.i21:		; preds = %bb8.i.i20
	%514 = icmp sgt i32 %512, 199		; <i1> [#uses=1]
	br i1 %514, label %bb12.i.i22, label %bb.nph106.i.i

bb12.i.i22:		; preds = %bb10.i.i21
	%515 = icmp sgt i32 %512, 799		; <i1> [#uses=1]
	br i1 %515, label %bb.nph106.i.i, label %bb15.i.i24

bb15.i.i24:		; preds = %bb12.i.i22
	br label %bb.nph106.i.i

bb.nph106.i.i:		; preds = %bb15.i.i24, %bb12.i.i22, %bb10.i.i21
	%nGroups.0.i.i.reg2mem.0 = phi i32 [ 4, %bb15.i.i24 ], [ 2, %bb10.i.i21 ], [ 6, %bb12.i.i22 ]		; <i32> [#uses=9]
	%516 = add i32 %508, 1		; <i32> [#uses=2]
	%tmp254.i.i = mul i32 %nGroups.0.i.i.reg2mem.0, 258		; <i32> [#uses=1]
	%tmp255.i.i = add i32 %tmp254.i.i, -258		; <i32> [#uses=1]
	%tmp258.i.i = sub i32 0, %nGroups.0.i.i.reg2mem.0		; <i32> [#uses=2]
	%tmp259.i.i = icmp sgt i32 %tmp258.i.i, -1		; <i1> [#uses=1]
	%smax260.i.i = select i1 %tmp259.i.i, i32 %tmp258.i.i, i32 -1		; <i32> [#uses=1]
	%tmp261.i.i = add i32 %nGroups.0.i.i.reg2mem.0, %smax260.i.i		; <i32> [#uses=1]
	%tmp262.i.i = add i32 %tmp261.i.i, 1		; <i32> [#uses=1]
	br label %bb16.i.i

bb16.i.i:		; preds = %bb37.i.i32, %bb.nph106.i.i
	%indvar251.i.i = phi i32 [ 0, %bb.nph106.i.i ], [ %indvar.next252.i.i, %bb37.i.i32 ]		; <i32> [#uses=5]
	%gs.0105.i.i = phi i32 [ 0, %bb.nph106.i.i ], [ %540, %bb37.i.i32 ]		; <i32> [#uses=4]
	%remF.0104.i.i = phi i32 [ %512, %bb.nph106.i.i ], [ %541, %bb37.i.i32 ]		; <i32> [#uses=2]
	%tmp253.i.i = sub i32 0, %indvar251.i.i		; <i32> [#uses=1]
	%nPart.0103.i.i = sub i32 %nGroups.0.i.i.reg2mem.0, %indvar251.i.i		; <i32> [#uses=2]
	%517 = sdiv i32 %remF.0104.i.i, %nPart.0103.i.i		; <i32> [#uses=2]
	%518 = add i32 %gs.0105.i.i, -1		; <i32> [#uses=2]
	%519 = icmp sgt i32 %517, 0		; <i1> [#uses=1]
	%520 = icmp sgt i32 %516, %518		; <i1> [#uses=1]
	%or.cond197.i.i = and i1 %519, %520		; <i1> [#uses=1]
	br i1 %or.cond197.i.i, label %bb17.i.i25, label %bb20.i.i27

bb17.i.i25:		; preds = %bb17.i.i25, %bb16.i.i
	%indvar245.i.i = phi i32 [ %indvar.next246.i.i, %bb17.i.i25 ], [ 0, %bb16.i.i ]		; <i32> [#uses=2]
	%aFreq.096.i.i = phi i32 [ %522, %bb17.i.i25 ], [ 0, %bb16.i.i ]		; <i32> [#uses=1]
	%tmp247.i.i = add i32 %indvar245.i.i, %gs.0105.i.i		; <i32> [#uses=3]
	%scevgep248.i.i = getelementptr [258 x i32]* @mtfFreq, i32 0, i32 %tmp247.i.i		; <i32*> [#uses=1]
	%521 = load i32* %scevgep248.i.i, align 4		; <i32> [#uses=1]
	%522 = add i32 %521, %aFreq.096.i.i		; <i32> [#uses=3]
	%523 = icmp slt i32 %522, %517		; <i1> [#uses=1]
	%524 = icmp sgt i32 %516, %tmp247.i.i		; <i1> [#uses=1]
	%or.cond1.i.i = and i1 %523, %524		; <i1> [#uses=1]
	%indvar.next246.i.i = add i32 %indvar245.i.i, 1		; <i32> [#uses=1]
	br i1 %or.cond1.i.i, label %bb17.i.i25, label %bb20.i.i27

bb20.i.i27:		; preds = %bb17.i.i25, %bb16.i.i
	%aFreq.0.lcssa.i.i = phi i32 [ 0, %bb16.i.i ], [ %522, %bb17.i.i25 ]		; <i32> [#uses=3]
	%ge.0.lcssa.i.i = phi i32 [ %518, %bb16.i.i ], [ %tmp247.i.i, %bb17.i.i25 ]		; <i32> [#uses=5]
	%525 = icmp sgt i32 %ge.0.lcssa.i.i, %gs.0105.i.i		; <i1> [#uses=1]
	%526 = icmp ne i32 %indvar251.i.i, 0		; <i1> [#uses=1]
	%527 = and i1 %525, %526		; <i1> [#uses=1]
	%.not.i.i = xor i1 %527, true		; <i1> [#uses=1]
	%528 = icmp eq i32 %nPart.0103.i.i, 1		; <i1> [#uses=1]
	%or.cond.i.i26 = or i1 %528, %.not.i.i		; <i1> [#uses=1]
	br i1 %or.cond.i.i26, label %bb26.i.i30, label %bb24.i.i28

bb24.i.i28:		; preds = %bb20.i.i27
	%529 = srem i32 %indvar251.i.i, 2		; <i32> [#uses=1]
	%530 = icmp eq i32 %529, 1		; <i1> [#uses=1]
	br i1 %530, label %bb25.i.i29, label %bb26.i.i30

bb25.i.i29:		; preds = %bb24.i.i28
	%531 = getelementptr [258 x i32]* @mtfFreq, i32 0, i32 %ge.0.lcssa.i.i		; <i32*> [#uses=1]
	%532 = load i32* %531, align 4		; <i32> [#uses=1]
	%533 = sub i32 %aFreq.0.lcssa.i.i, %532		; <i32> [#uses=1]
	%534 = add i32 %ge.0.lcssa.i.i, -1		; <i32> [#uses=1]
	br label %bb26.i.i30

bb26.i.i30:		; preds = %bb25.i.i29, %bb24.i.i28, %bb20.i.i27
	%ge.1.i.i = phi i32 [ %534, %bb25.i.i29 ], [ %ge.0.lcssa.i.i, %bb20.i.i27 ], [ %ge.0.lcssa.i.i, %bb24.i.i28 ]		; <i32> [#uses=2]
	%aFreq.1.i.i = phi i32 [ %533, %bb25.i.i29 ], [ %aFreq.0.lcssa.i.i, %bb20.i.i27 ], [ %aFreq.0.lcssa.i.i, %bb24.i.i28 ]		; <i32> [#uses=1]
	br i1 %510, label %bb29.i.i31, label %bb37.i.i32

bb29.i.i31:		; preds = %bb29.i.i31, %bb26.i.i30
	%535 = phi i32 [ %539, %bb29.i.i31 ], [ 0, %bb26.i.i30 ]		; <i32> [#uses=4]
	%tmp256.i.i = add i32 %535, %tmp255.i.i		; <i32> [#uses=1]
	%scevgep257.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 %tmp253.i.i, i32 %tmp256.i.i		; <i8*> [#uses=1]
	%536 = icmp sge i32 %535, %gs.0105.i.i		; <i1> [#uses=1]
	%537 = icmp sle i32 %535, %ge.1.i.i		; <i1> [#uses=1]
	%538 = and i1 %536, %537		; <i1> [#uses=1]
	%storemerge.i.i = select i1 %538, i8 0, i8 15		; <i8> [#uses=1]
	store i8 %storemerge.i.i, i8* %scevgep257.i.i
	%539 = add i32 %535, 1		; <i32> [#uses=2]
	%exitcond250.i.i = icmp eq i32 %539, %509		; <i1> [#uses=1]
	br i1 %exitcond250.i.i, label %bb37.i.i32, label %bb29.i.i31

bb37.i.i32:		; preds = %bb29.i.i31, %bb26.i.i30
	%540 = add i32 %ge.1.i.i, 1		; <i32> [#uses=1]
	%541 = sub i32 %remF.0104.i.i, %aFreq.1.i.i		; <i32> [#uses=1]
	%indvar.next252.i.i = add i32 %indvar251.i.i, 1		; <i32> [#uses=2]
	%exitcond263.i.i = icmp eq i32 %indvar.next252.i.i, %tmp262.i.i		; <i1> [#uses=1]
	br i1 %exitcond263.i.i, label %bb85.loopexit.i.i, label %bb16.i.i

bb41.i.i33:		; preds = %bb41.i.i33.preheader, %bb41.i.i33
	%t.147.i.i = phi i32 [ %542, %bb41.i.i33 ], [ 0, %bb41.i.i33.preheader ]		; <i32> [#uses=2]
	%scevgep183.i.i = getelementptr [6 x i32]* %fave.i.i, i32 0, i32 %t.147.i.i		; <i32*> [#uses=1]
	store i32 0, i32* %scevgep183.i.i, align 4
	%542 = add i32 %t.147.i.i, 1		; <i32> [#uses=2]
	%exitcond182.i.i = icmp eq i32 %542, %smax181.i.i		; <i1> [#uses=1]
	br i1 %exitcond182.i.i, label %bb48.preheader.i.i, label %bb41.i.i33

bb48.preheader.i.i:		; preds = %bb41.i.i33
	br i1 %724, label %bb50.preheader.i.i, label %bb45.i.i

bb45.i.i:		; preds = %bb47.i.i34, %bb45.i.i, %bb48.preheader.i.i
	%.ph = phi i32 [ %544, %bb47.i.i34 ], [ 0, %bb48.preheader.i.i ], [ %.ph, %bb45.i.i ]		; <i32> [#uses=3]
	%v.249.i.i = phi i32 [ %543, %bb45.i.i ], [ 0, %bb48.preheader.i.i ], [ 0, %bb47.i.i34 ]		; <i32> [#uses=2]
	%scevgep186.i.i = getelementptr [6 x [258 x i32]]* @rfreq, i32 0, i32 %.ph, i32 %v.249.i.i		; <i32*> [#uses=1]
	store i32 0, i32* %scevgep186.i.i, align 4
	%543 = add i32 %v.249.i.i, 1		; <i32> [#uses=2]
	%exitcond185.i.i = icmp eq i32 %543, %509		; <i1> [#uses=1]
	br i1 %exitcond185.i.i, label %bb47.i.i34, label %bb45.i.i

bb47.i.i34:		; preds = %bb45.i.i
	%544 = add i32 %.ph, 1		; <i32> [#uses=2]
	%exitcond189.i.i = icmp eq i32 %544, %smax181.i.i		; <i1> [#uses=1]
	br i1 %exitcond189.i.i, label %bb50.preheader.i.i, label %bb45.i.i

bb50.preheader.i.i:		; preds = %bb47.i.i34, %bb48.preheader.i.i
	br i1 %723, label %bb54.i.i.preheader, label %bb82.i.i

bb54.i.i.preheader:		; preds = %bb75.i.i, %bb50.preheader.i.i
	%gs.183.i.i = phi i32 [ %598, %bb75.i.i ], [ 0, %bb50.preheader.i.i ]		; <i32> [#uses=8]
	%nSelectors.182.i.i = phi i32 [ %tmp231.i.i, %bb75.i.i ], [ 0, %bb50.preheader.i.i ]		; <i32> [#uses=2]
	%scevgep230.i.i = getelementptr [18002 x i8]* @selector, i32 0, i32 %nSelectors.182.i.i		; <i8*> [#uses=1]
	%tmp231.i.i = add i32 %nSelectors.182.i.i, 1		; <i32> [#uses=2]
	%545 = add i32 %gs.183.i.i, 49		; <i32> [#uses=2]
	%546 = icmp slt i32 %545, %512		; <i1> [#uses=1]
	%..i.i = select i1 %546, i32 %545, i32 %726		; <i32> [#uses=5]
	br label %bb54.i.i

bb54.i.i:		; preds = %bb54.i.i, %bb54.i.i.preheader
	%t.357.i.i = phi i32 [ %547, %bb54.i.i ], [ 0, %bb54.i.i.preheader ]		; <i32> [#uses=2]
	%scevgep205.i.i = getelementptr [6 x i16]* %cost.i.i, i32 0, i32 %t.357.i.i		; <i16*> [#uses=1]
	store i16 0, i16* %scevgep205.i.i, align 2
	%547 = add i32 %t.357.i.i, 1		; <i32> [#uses=2]
	%exitcond204.i.i = icmp eq i32 %547, %smax181.i.i		; <i1> [#uses=1]
	br i1 %exitcond204.i.i, label %bb56.i.i, label %bb54.i.i

bb56.i.i:		; preds = %bb54.i.i
	%548 = icmp sgt i32 %gs.183.i.i, %..i.i		; <i1> [#uses=3]
	br i1 %722, label %bb59.preheader.i.i, label %bb66.preheader.i.i

bb66.preheader.i.i:		; preds = %bb56.i.i
	br i1 %548, label %bb68.i.i37, label %bb.nph56.split.i.i

bb59.preheader.i.i:		; preds = %bb56.i.i
	br i1 %548, label %bb60.i.i, label %bb.nph66.i.i

bb.nph66.i.i:		; preds = %bb59.preheader.i.i
	%tmp216.i.i = add i32 %gs.183.i.i, 1		; <i32> [#uses=1]
	br label %bb58.i.i

bb58.i.i:		; preds = %bb58.i.i, %bb.nph66.i.i
	%indvar212.i.i = phi i32 [ 0, %bb.nph66.i.i ], [ %indvar.next213.i.i, %bb58.i.i ]		; <i32> [#uses=3]
	%cost5.064.i.i = phi i16 [ 0, %bb.nph66.i.i ], [ %574, %bb58.i.i ]		; <i16> [#uses=1]
	%cost4.063.i.i = phi i16 [ 0, %bb.nph66.i.i ], [ %570, %bb58.i.i ]		; <i16> [#uses=1]
	%cost3.062.i.i = phi i16 [ 0, %bb.nph66.i.i ], [ %566, %bb58.i.i ]		; <i16> [#uses=1]
	%cost2.061.i.i = phi i16 [ 0, %bb.nph66.i.i ], [ %562, %bb58.i.i ]		; <i16> [#uses=1]
	%cost1.060.i.i = phi i16 [ 0, %bb.nph66.i.i ], [ %558, %bb58.i.i ]		; <i16> [#uses=1]
	%cost0.059.i.i = phi i16 [ 0, %bb.nph66.i.i ], [ %554, %bb58.i.i ]		; <i16> [#uses=1]
	%tmp214.i.i = add i32 %indvar212.i.i, %gs.183.i.i		; <i32> [#uses=1]
	%scevgep215.i.i = getelementptr i16* %727, i32 %tmp214.i.i		; <i16*> [#uses=1]
	%549 = load i16* %scevgep215.i.i, align 2		; <i16> [#uses=1]
	%550 = zext i16 %549 to i32		; <i32> [#uses=6]
	%551 = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 0, i32 %550		; <i8*> [#uses=1]
	%552 = load i8* %551, align 1		; <i8> [#uses=1]
	%553 = zext i8 %552 to i16		; <i16> [#uses=1]
	%554 = add i16 %553, %cost0.059.i.i		; <i16> [#uses=2]
	%555 = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 1, i32 %550		; <i8*> [#uses=1]
	%556 = load i8* %555, align 1		; <i8> [#uses=1]
	%557 = zext i8 %556 to i16		; <i16> [#uses=1]
	%558 = add i16 %557, %cost1.060.i.i		; <i16> [#uses=2]
	%559 = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 2, i32 %550		; <i8*> [#uses=1]
	%560 = load i8* %559, align 1		; <i8> [#uses=1]
	%561 = zext i8 %560 to i16		; <i16> [#uses=1]
	%562 = add i16 %561, %cost2.061.i.i		; <i16> [#uses=2]
	%563 = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 3, i32 %550		; <i8*> [#uses=1]
	%564 = load i8* %563, align 1		; <i8> [#uses=1]
	%565 = zext i8 %564 to i16		; <i16> [#uses=1]
	%566 = add i16 %565, %cost3.062.i.i		; <i16> [#uses=2]
	%567 = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 4, i32 %550		; <i8*> [#uses=1]
	%568 = load i8* %567, align 1		; <i8> [#uses=1]
	%569 = zext i8 %568 to i16		; <i16> [#uses=1]
	%570 = add i16 %569, %cost4.063.i.i		; <i16> [#uses=2]
	%571 = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 5, i32 %550		; <i8*> [#uses=1]
	%572 = load i8* %571, align 1		; <i8> [#uses=1]
	%573 = zext i8 %572 to i16		; <i16> [#uses=1]
	%574 = add i16 %573, %cost5.064.i.i		; <i16> [#uses=2]
	%tmp217.i.i = add i32 %indvar212.i.i, %tmp216.i.i		; <i32> [#uses=1]
	%575 = icmp sgt i32 %tmp217.i.i, %..i.i		; <i1> [#uses=1]
	%indvar.next213.i.i = add i32 %indvar212.i.i, 1		; <i32> [#uses=1]
	br i1 %575, label %bb60.i.i, label %bb58.i.i

bb60.i.i:		; preds = %bb58.i.i, %bb59.preheader.i.i
	%cost5.0.lcssa.i.i = phi i16 [ 0, %bb59.preheader.i.i ], [ %574, %bb58.i.i ]		; <i16> [#uses=1]
	%cost4.0.lcssa.i.i = phi i16 [ 0, %bb59.preheader.i.i ], [ %570, %bb58.i.i ]		; <i16> [#uses=1]
	%cost3.0.lcssa.i.i = phi i16 [ 0, %bb59.preheader.i.i ], [ %566, %bb58.i.i ]		; <i16> [#uses=1]
	%cost2.0.lcssa.i.i = phi i16 [ 0, %bb59.preheader.i.i ], [ %562, %bb58.i.i ]		; <i16> [#uses=1]
	%cost1.0.lcssa.i.i = phi i16 [ 0, %bb59.preheader.i.i ], [ %558, %bb58.i.i ]		; <i16> [#uses=1]
	%cost0.0.lcssa.i.i = phi i16 [ 0, %bb59.preheader.i.i ], [ %554, %bb58.i.i ]		; <i16> [#uses=1]
	store i16 %cost0.0.lcssa.i.i, i16* %87, align 2
	store i16 %cost1.0.lcssa.i.i, i16* %88, align 2
	store i16 %cost2.0.lcssa.i.i, i16* %89, align 2
	store i16 %cost3.0.lcssa.i.i, i16* %90, align 2
	store i16 %cost4.0.lcssa.i.i, i16* %91, align 2
	store i16 %cost5.0.lcssa.i.i, i16* %92, align 2
	br label %bb68.i.i37

bb.nph56.split.i.i:		; preds = %bb66.preheader.i.i
	%tmp200.i.i = add i32 %gs.183.i.i, 1		; <i32> [#uses=1]
	br label %bb62.i.i

bb62.i.i:		; preds = %bb65.i.i36, %bb.nph56.split.i.i
	%indvar196.i.i = phi i32 [ 0, %bb.nph56.split.i.i ], [ %indvar.next197.i.i, %bb65.i.i36 ]		; <i32> [#uses=3]
	%tmp201.i.i = add i32 %indvar196.i.i, %tmp200.i.i		; <i32> [#uses=1]
	%tmp198.i.i = add i32 %indvar196.i.i, %gs.183.i.i		; <i32> [#uses=1]
	%scevgep199.i.i = getelementptr i16* %727, i32 %tmp198.i.i		; <i16*> [#uses=1]
	%576 = load i16* %scevgep199.i.i, align 2		; <i16> [#uses=1]
	%tmp194.i.i = zext i16 %576 to i32		; <i32> [#uses=1]
	br label %bb63.i.i

bb63.i.i:		; preds = %bb63.i.i, %bb62.i.i
	%t.453.i.i = phi i32 [ 0, %bb62.i.i ], [ %581, %bb63.i.i ]		; <i32> [#uses=3]
	%scevgep193.i.i = getelementptr [6 x i16]* %cost.i.i, i32 0, i32 %t.453.i.i		; <i16*> [#uses=2]
	%scevgep195.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 %t.453.i.i, i32 %tmp194.i.i		; <i8*> [#uses=1]
	%577 = load i16* %scevgep193.i.i, align 2		; <i16> [#uses=1]
	%578 = load i8* %scevgep195.i.i, align 1		; <i8> [#uses=1]
	%579 = zext i8 %578 to i16		; <i16> [#uses=1]
	%580 = add i16 %579, %577		; <i16> [#uses=1]
	store i16 %580, i16* %scevgep193.i.i, align 2
	%581 = add i32 %t.453.i.i, 1		; <i32> [#uses=2]
	%exitcond192.i.i = icmp eq i32 %581, %smax181.i.i		; <i1> [#uses=1]
	br i1 %exitcond192.i.i, label %bb65.i.i36, label %bb63.i.i

bb65.i.i36:		; preds = %bb63.i.i
	%582 = icmp sgt i32 %tmp201.i.i, %..i.i		; <i1> [#uses=1]
	%indvar.next197.i.i = add i32 %indvar196.i.i, 1		; <i32> [#uses=1]
	br i1 %582, label %bb68.i.i37, label %bb62.i.i

bb68.i.i37:		; preds = %bb68.i.i37, %bb65.i.i36, %bb60.i.i, %bb66.preheader.i.i
	%583 = phi i32 [ %587, %bb68.i.i37 ], [ 0, %bb65.i.i36 ], [ 0, %bb66.preheader.i.i ], [ 0, %bb60.i.i ]		; <i32> [#uses=3]
	%bc.174.i.i = phi i32 [ %bc.0.i.i, %bb68.i.i37 ], [ 999999999, %bb65.i.i36 ], [ 999999999, %bb66.preheader.i.i ], [ 999999999, %bb60.i.i ]		; <i32> [#uses=2]
	%bt.173.i.i = phi i32 [ %bt.0.i.i, %bb68.i.i37 ], [ -1, %bb65.i.i36 ], [ -1, %bb66.preheader.i.i ], [ -1, %bb60.i.i ]		; <i32> [#uses=1]
	%scevgep221.i.i = getelementptr [6 x i16]* %cost.i.i, i32 0, i32 %583		; <i16*> [#uses=1]
	%584 = load i16* %scevgep221.i.i, align 2		; <i16> [#uses=1]
	%585 = zext i16 %584 to i32		; <i32> [#uses=2]
	%586 = icmp slt i32 %585, %bc.174.i.i		; <i1> [#uses=2]
	%bt.0.i.i = select i1 %586, i32 %583, i32 %bt.173.i.i		; <i32> [#uses=4]
	%bc.0.i.i = select i1 %586, i32 %585, i32 %bc.174.i.i		; <i32> [#uses=1]
	%587 = add i32 %583, 1		; <i32> [#uses=2]
	%exitcond220.i.i = icmp eq i32 %587, %smax181.i.i		; <i1> [#uses=1]
	br i1 %exitcond220.i.i, label %bb72.i.i, label %bb68.i.i37

bb72.i.i:		; preds = %bb68.i.i37
	%588 = getelementptr [6 x i32]* %fave.i.i, i32 0, i32 %bt.0.i.i		; <i32*> [#uses=2]
	%589 = load i32* %588, align 4		; <i32> [#uses=1]
	%590 = add i32 %589, 1		; <i32> [#uses=1]
	store i32 %590, i32* %588, align 4
	%591 = trunc i32 %bt.0.i.i to i8		; <i8> [#uses=1]
	store i8 %591, i8* %scevgep230.i.i, align 1
	br i1 %548, label %bb75.i.i, label %bb.nph80.i.i

bb.nph80.i.i:		; preds = %bb72.i.i
	%tmp226.i.i = add i32 %gs.183.i.i, 1		; <i32> [#uses=1]
	br label %bb73.i.i

bb73.i.i:		; preds = %bb73.i.i, %bb.nph80.i.i
	%indvar222.i.i = phi i32 [ 0, %bb.nph80.i.i ], [ %indvar.next223.i.i, %bb73.i.i ]		; <i32> [#uses=3]
	%tmp224.i.i = add i32 %indvar222.i.i, %gs.183.i.i		; <i32> [#uses=1]
	%scevgep225.i.i = getelementptr i16* %727, i32 %tmp224.i.i		; <i16*> [#uses=1]
	%592 = load i16* %scevgep225.i.i, align 2		; <i16> [#uses=1]
	%593 = zext i16 %592 to i32		; <i32> [#uses=1]
	%594 = getelementptr [6 x [258 x i32]]* @rfreq, i32 0, i32 %bt.0.i.i, i32 %593		; <i32*> [#uses=2]
	%595 = load i32* %594, align 4		; <i32> [#uses=1]
	%596 = add i32 %595, 1		; <i32> [#uses=1]
	store i32 %596, i32* %594, align 4
	%tmp227.i.i = add i32 %indvar222.i.i, %tmp226.i.i		; <i32> [#uses=1]
	%597 = icmp sgt i32 %tmp227.i.i, %..i.i		; <i1> [#uses=1]
	%indvar.next223.i.i = add i32 %indvar222.i.i, 1		; <i32> [#uses=1]
	br i1 %597, label %bb75.i.i, label %bb73.i.i

bb75.i.i:		; preds = %bb73.i.i, %bb72.i.i
	%598 = add i32 %..i.i, 1		; <i32> [#uses=2]
	%599 = icmp slt i32 %598, %512		; <i1> [#uses=1]
	br i1 %599, label %bb54.i.i.preheader, label %bb82.i.i

bb82.i.i:		; preds = %hbMakeCodeLengths.exit.i.i, %bb75.i.i, %bb50.preheader.i.i
	%nSelectors.1.lcssa.i.i = phi i32 [ 0, %bb50.preheader.i.i ], [ %tmp231.i.i, %bb75.i.i ], [ %nSelectors.1.lcssa.i.i, %hbMakeCodeLengths.exit.i.i ]		; <i32> [#uses=7]
	%t.789.i.i = phi i32 [ %721, %hbMakeCodeLengths.exit.i.i ], [ 0, %bb75.i.i ], [ 0, %bb50.preheader.i.i ]		; <i32> [#uses=3]
	br i1 %510, label %bb.i.i.i38, label %bb12.i.i.i39

bb.i.i.i38:		; preds = %bb.i.i.i38, %bb82.i.i
	%i.026.i.i.i = phi i32 [ %tmp67.i.i.i, %bb.i.i.i38 ], [ 0, %bb82.i.i ]		; <i32> [#uses=2]
	%scevgep66.i.i.i = getelementptr [6 x [258 x i32]]* @rfreq, i32 0, i32 %t.789.i.i, i32 %i.026.i.i.i		; <i32*> [#uses=1]
	%tmp67.i.i.i = add i32 %i.026.i.i.i, 1		; <i32> [#uses=3]
	%scevgep68.i.i.i = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %tmp67.i.i.i		; <i32*> [#uses=1]
	%600 = load i32* %scevgep66.i.i.i, align 4		; <i32> [#uses=2]
	%601 = shl i32 %600, 8		; <i32> [#uses=1]
	%602 = icmp eq i32 %600, 0		; <i1> [#uses=1]
	%.69.i.i.i = select i1 %602, i32 256, i32 %601		; <i32> [#uses=1]
	store i32 %.69.i.i.i, i32* %scevgep68.i.i.i, align 4
	%exitcond65.i.i.i = icmp eq i32 %tmp67.i.i.i, %509		; <i1> [#uses=1]
	br i1 %exitcond65.i.i.i, label %bb12.i.i.i39, label %bb.i.i.i38

bb12.i.i.i39:		; preds = %bb52.i.i.i, %bb53.preheader.i.i.i, %bb.i.i.i38, %bb82.i.i
	store i32 0, i32* %93, align 4
	store i32 0, i32* %94, align 4
	store i32 -2, i32* %95, align 4
	br i1 %724, label %bb40.i.i.i52, label %bb13.i.i.i40

bb13.i.i.i40:		; preds = %bb16.i.i.i42, %bb12.i.i.i39
	%nHeap.06.i.i.i = phi i32 [ %i.15.i.i.i, %bb16.i.i.i42 ], [ 0, %bb12.i.i.i39 ]		; <i32> [#uses=5]
	%i.15.i.i.i = add i32 %nHeap.06.i.i.i, 1		; <i32> [#uses=13]
	%tmp.i1.i.i = add i32 %nHeap.06.i.i.i, 2		; <i32> [#uses=1]
	%scevgep.i2.i.i = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %i.15.i.i.i		; <i32*> [#uses=1]
	%scevgep30.i.i.i = getelementptr [516 x i32]* %parent.i.i.i, i32 0, i32 %i.15.i.i.i		; <i32*> [#uses=1]
	%scevgep31.i.i.i = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %i.15.i.i.i		; <i32*> [#uses=1]
	store i32 -1, i32* %scevgep30.i.i.i, align 4
	store i32 %i.15.i.i.i, i32* %scevgep.i2.i.i, align 4
	%603 = load i32* %scevgep31.i.i.i, align 4		; <i32> [#uses=2]
	%604 = ashr i32 %i.15.i.i.i, 1		; <i32> [#uses=1]
	%605 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %604		; <i32*> [#uses=1]
	%606 = load i32* %605, align 4		; <i32> [#uses=1]
	%607 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %606		; <i32*> [#uses=1]
	%608 = load i32* %607, align 4		; <i32> [#uses=1]
	%609 = icmp slt i32 %603, %608		; <i1> [#uses=1]
	br i1 %609, label %bb14.i.i.i41, label %bb16.i.i.i42

bb14.i.i.i41:		; preds = %bb14.i.i.i41, %bb13.i.i.i40
	%zz7.03.i.i.i = phi i32 [ %610, %bb14.i.i.i41 ], [ %i.15.i.i.i, %bb13.i.i.i40 ]		; <i32> [#uses=3]
	%610 = ashr i32 %zz7.03.i.i.i, 1		; <i32> [#uses=3]
	%611 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %610		; <i32*> [#uses=1]
	%612 = load i32* %611, align 4		; <i32> [#uses=1]
	%613 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %zz7.03.i.i.i		; <i32*> [#uses=1]
	store i32 %612, i32* %613, align 4
	%614 = ashr i32 %zz7.03.i.i.i, 2		; <i32> [#uses=1]
	%615 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %614		; <i32*> [#uses=1]
	%616 = load i32* %615, align 4		; <i32> [#uses=1]
	%617 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %616		; <i32*> [#uses=1]
	%618 = load i32* %617, align 4		; <i32> [#uses=1]
	%619 = icmp slt i32 %603, %618		; <i1> [#uses=1]
	br i1 %619, label %bb14.i.i.i41, label %bb16.i.i.i42

bb16.i.i.i42:		; preds = %bb14.i.i.i41, %bb13.i.i.i40
	%zz7.0.lcssa.i.i.i = phi i32 [ %i.15.i.i.i, %bb13.i.i.i40 ], [ %610, %bb14.i.i.i41 ]		; <i32> [#uses=1]
	%620 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %zz7.0.lcssa.i.i.i		; <i32*> [#uses=1]
	store i32 %i.15.i.i.i, i32* %620, align 4
	%621 = icmp sgt i32 %tmp.i1.i.i, %509		; <i1> [#uses=1]
	br i1 %621, label %bb18.i.i.i, label %bb13.i.i.i40

bb18.i.i.i:		; preds = %bb16.i.i.i42
	%622 = icmp sgt i32 %i.15.i.i.i, 259		; <i1> [#uses=1]
	br i1 %622, label %bb19.i.i.i43, label %bb39.preheader.i.i.i

bb39.preheader.i.i.i:		; preds = %bb18.i.i.i
	%623 = icmp sgt i32 %i.15.i.i.i, 1		; <i1> [#uses=1]
	br i1 %623, label %bb.nph14.i.i.i, label %bb40.i.i.i52

bb19.i.i.i43:		; preds = %bb18.i.i.i
	call fastcc void @panic(i8* getelementptr ([21 x i8]* @"\01LC11", i32 0, i32 0)) nounwind ssp
	unreachable

bb.nph14.i.i.i:		; preds = %bb39.preheader.i.i.i
	%tmp34.i.i.i = add i32 %i.15.i.i.i, %509		; <i32> [#uses=1]
	%tmp49.i.i.i44 = add i32 %nHeap.06.i.i.i, -1		; <i32> [#uses=1]
	br label %bb20.i.i.i

bb20.i.i.i:		; preds = %bb38.i.i.i51, %bb.nph14.i.i.i
	%indvar.i3.i.i = phi i32 [ 0, %bb.nph14.i.i.i ], [ %indvar.next.i4.i.i, %bb38.i.i.i51 ]		; <i32> [#uses=5]
	%tmp38.i.i.i = add i32 %indvar.i3.i.i, %tmp37.i.i.i		; <i32> [#uses=6]
	%scevgep40.i.i.i = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %tmp38.i.i.i		; <i32*> [#uses=1]
	%scevgep42.i.i.i = getelementptr [516 x i32]* %parent.i.i.i, i32 0, i32 %tmp38.i.i.i		; <i32*> [#uses=1]
	%tmp44.i.i.i = sub i32 %i.15.i.i.i, %indvar.i3.i.i		; <i32> [#uses=1]
	%scevgep45.i.i.i = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %tmp44.i.i.i		; <i32*> [#uses=1]
	%scevgep46.sum.i.i.i = sub i32 %nHeap.06.i.i.i, %indvar.i3.i.i		; <i32> [#uses=6]
	%scevgep47.i.i.i = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %scevgep46.sum.i.i.i		; <i32*> [#uses=2]
	%tmp50.i.i.i = sub i32 %tmp49.i.i.i44, %indvar.i3.i.i		; <i32> [#uses=2]
	%624 = load i32* %96, align 4		; <i32> [#uses=2]
	%625 = load i32* %scevgep45.i.i.i, align 4		; <i32> [#uses=3]
	store i32 %625, i32* %96, align 4
	%626 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %625		; <i32*> [#uses=1]
	br label %bb21.i.i.i45

bb21.i.i.i45:		; preds = %bb26.i.i.i, %bb20.i.i.i
	%zz5.0.i.i.i = phi i32 [ 1, %bb20.i.i.i ], [ %yy4.0.i.i.i, %bb26.i.i.i ]		; <i32> [#uses=3]
	%627 = shl i32 %zz5.0.i.i.i, 1		; <i32> [#uses=6]
	%628 = icmp sgt i32 %627, %scevgep46.sum.i.i.i		; <i1> [#uses=1]
	br i1 %628, label %bb27.i.i.i47, label %bb22.i.i.i46

bb22.i.i.i46:		; preds = %bb21.i.i.i45
	%629 = icmp slt i32 %627, %scevgep46.sum.i.i.i		; <i1> [#uses=1]
	br i1 %629, label %bb23.i.i.i, label %bb25.i.i.i

bb23.i.i.i:		; preds = %bb22.i.i.i46
	%630 = or i32 %627, 1		; <i32> [#uses=1]
	%631 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %630		; <i32*> [#uses=1]
	%632 = load i32* %631, align 4		; <i32> [#uses=1]
	%633 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %632		; <i32*> [#uses=1]
	%634 = load i32* %633, align 4		; <i32> [#uses=1]
	%635 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %627		; <i32*> [#uses=1]
	%636 = load i32* %635, align 4		; <i32> [#uses=1]
	%637 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %636		; <i32*> [#uses=1]
	%638 = load i32* %637, align 4		; <i32> [#uses=1]
	%639 = icmp slt i32 %634, %638		; <i1> [#uses=1]
	%640 = zext i1 %639 to i32		; <i32> [#uses=1]
	%..i.i.i = or i32 %640, %627		; <i32> [#uses=1]
	br label %bb25.i.i.i

bb25.i.i.i:		; preds = %bb23.i.i.i, %bb22.i.i.i46
	%yy4.0.i.i.i = phi i32 [ %..i.i.i, %bb23.i.i.i ], [ %627, %bb22.i.i.i46 ]		; <i32> [#uses=2]
	%641 = load i32* %626, align 4		; <i32> [#uses=1]
	%642 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %yy4.0.i.i.i		; <i32*> [#uses=1]
	%643 = load i32* %642, align 4		; <i32> [#uses=2]
	%644 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %643		; <i32*> [#uses=1]
	%645 = load i32* %644, align 4		; <i32> [#uses=1]
	%646 = icmp slt i32 %641, %645		; <i1> [#uses=1]
	br i1 %646, label %bb27.i.i.i47, label %bb26.i.i.i

bb26.i.i.i:		; preds = %bb25.i.i.i
	%647 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %zz5.0.i.i.i		; <i32*> [#uses=1]
	store i32 %643, i32* %647, align 4
	br label %bb21.i.i.i45

bb27.i.i.i47:		; preds = %bb25.i.i.i, %bb21.i.i.i45
	%648 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %zz5.0.i.i.i		; <i32*> [#uses=1]
	store i32 %625, i32* %648, align 4
	%649 = load i32* %96, align 4		; <i32> [#uses=2]
	%650 = load i32* %scevgep47.i.i.i, align 4		; <i32> [#uses=3]
	store i32 %650, i32* %96, align 4
	%651 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %650		; <i32*> [#uses=1]
	br label %bb28.i.i.i48

bb28.i.i.i48:		; preds = %bb33.i.i.i, %bb27.i.i.i47
	%zz2.0.i.i.i = phi i32 [ 1, %bb27.i.i.i47 ], [ %yy.0.i.i.i, %bb33.i.i.i ]		; <i32> [#uses=3]
	%652 = shl i32 %zz2.0.i.i.i, 1		; <i32> [#uses=6]
	%653 = icmp sgt i32 %652, %tmp50.i.i.i		; <i1> [#uses=1]
	br i1 %653, label %bb34.i.i.i, label %bb29.i.i.i

bb29.i.i.i:		; preds = %bb28.i.i.i48
	%654 = icmp slt i32 %652, %tmp50.i.i.i		; <i1> [#uses=1]
	br i1 %654, label %bb30.i.i.i49, label %bb32.i.i.i

bb30.i.i.i49:		; preds = %bb29.i.i.i
	%655 = or i32 %652, 1		; <i32> [#uses=1]
	%656 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %655		; <i32*> [#uses=1]
	%657 = load i32* %656, align 4		; <i32> [#uses=1]
	%658 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %657		; <i32*> [#uses=1]
	%659 = load i32* %658, align 4		; <i32> [#uses=1]
	%660 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %652		; <i32*> [#uses=1]
	%661 = load i32* %660, align 4		; <i32> [#uses=1]
	%662 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %661		; <i32*> [#uses=1]
	%663 = load i32* %662, align 4		; <i32> [#uses=1]
	%664 = icmp slt i32 %659, %663		; <i1> [#uses=1]
	%665 = zext i1 %664 to i32		; <i32> [#uses=1]
	%.1.i.i.i = or i32 %665, %652		; <i32> [#uses=1]
	br label %bb32.i.i.i

bb32.i.i.i:		; preds = %bb30.i.i.i49, %bb29.i.i.i
	%yy.0.i.i.i = phi i32 [ %.1.i.i.i, %bb30.i.i.i49 ], [ %652, %bb29.i.i.i ]		; <i32> [#uses=2]
	%666 = load i32* %651, align 4		; <i32> [#uses=1]
	%667 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %yy.0.i.i.i		; <i32*> [#uses=1]
	%668 = load i32* %667, align 4		; <i32> [#uses=2]
	%669 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %668		; <i32*> [#uses=1]
	%670 = load i32* %669, align 4		; <i32> [#uses=1]
	%671 = icmp slt i32 %666, %670		; <i1> [#uses=1]
	br i1 %671, label %bb34.i.i.i, label %bb33.i.i.i

bb33.i.i.i:		; preds = %bb32.i.i.i
	%672 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %zz2.0.i.i.i		; <i32*> [#uses=1]
	store i32 %668, i32* %672, align 4
	br label %bb28.i.i.i48

bb34.i.i.i:		; preds = %bb32.i.i.i, %bb28.i.i.i48
	%673 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %zz2.0.i.i.i		; <i32*> [#uses=1]
	store i32 %650, i32* %673, align 4
	%674 = getelementptr [516 x i32]* %parent.i.i.i, i32 0, i32 %649		; <i32*> [#uses=1]
	store i32 %tmp38.i.i.i, i32* %674, align 4
	%675 = getelementptr [516 x i32]* %parent.i.i.i, i32 0, i32 %624		; <i32*> [#uses=1]
	store i32 %tmp38.i.i.i, i32* %675, align 4
	%676 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %624		; <i32*> [#uses=1]
	%677 = load i32* %676, align 4		; <i32> [#uses=2]
	%678 = and i32 %677, -256		; <i32> [#uses=1]
	%679 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %649		; <i32*> [#uses=1]
	%680 = load i32* %679, align 4		; <i32> [#uses=2]
	%681 = and i32 %680, -256		; <i32> [#uses=1]
	%682 = add i32 %681, %678		; <i32> [#uses=1]
	%683 = and i32 %680, 255		; <i32> [#uses=2]
	%684 = and i32 %677, 255		; <i32> [#uses=2]
	%685 = icmp uge i32 %683, %684		; <i1> [#uses=1]
	%max.i.i.i = select i1 %685, i32 %683, i32 %684		; <i32> [#uses=1]
	%686 = add i32 %max.i.i.i, 1		; <i32> [#uses=1]
	%687 = or i32 %686, %682		; <i32> [#uses=3]
	store i32 %687, i32* %scevgep40.i.i.i, align 4
	store i32 -1, i32* %scevgep42.i.i.i, align 4
	store i32 %tmp38.i.i.i, i32* %scevgep47.i.i.i, align 4
	%688 = ashr i32 %scevgep46.sum.i.i.i, 1		; <i32> [#uses=1]
	%689 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %688		; <i32*> [#uses=1]
	%690 = load i32* %689, align 4		; <i32> [#uses=1]
	%691 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %690		; <i32*> [#uses=1]
	%692 = load i32* %691, align 4		; <i32> [#uses=1]
	%693 = icmp slt i32 %687, %692		; <i1> [#uses=1]
	br i1 %693, label %bb36.i.i.i, label %bb38.i.i.i51

bb36.i.i.i:		; preds = %bb36.i.i.i, %bb34.i.i.i
	%zz.09.i.i.i = phi i32 [ %694, %bb36.i.i.i ], [ %scevgep46.sum.i.i.i, %bb34.i.i.i ]		; <i32> [#uses=3]
	%694 = ashr i32 %zz.09.i.i.i, 1		; <i32> [#uses=3]
	%695 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %694		; <i32*> [#uses=1]
	%696 = load i32* %695, align 4		; <i32> [#uses=1]
	%697 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %zz.09.i.i.i		; <i32*> [#uses=1]
	store i32 %696, i32* %697, align 4
	%698 = ashr i32 %zz.09.i.i.i, 2		; <i32> [#uses=1]
	%699 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %698		; <i32*> [#uses=1]
	%700 = load i32* %699, align 4		; <i32> [#uses=1]
	%701 = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %700		; <i32*> [#uses=1]
	%702 = load i32* %701, align 4		; <i32> [#uses=1]
	%703 = icmp slt i32 %687, %702		; <i1> [#uses=1]
	br i1 %703, label %bb36.i.i.i, label %bb38.i.i.i51

bb38.i.i.i51:		; preds = %bb36.i.i.i, %bb34.i.i.i
	%zz.0.lcssa.i.i.i = phi i32 [ %scevgep46.sum.i.i.i, %bb34.i.i.i ], [ %694, %bb36.i.i.i ]		; <i32> [#uses=1]
	%704 = getelementptr [260 x i32]* %heap.i.i.i, i32 0, i32 %zz.0.lcssa.i.i.i		; <i32*> [#uses=1]
	store i32 %tmp38.i.i.i, i32* %704, align 4
	%indvar.next.i4.i.i = add i32 %indvar.i3.i.i, 1		; <i32> [#uses=2]
	%exitcond.i.i.i50 = icmp eq i32 %indvar.next.i4.i.i, %nHeap.06.i.i.i		; <i1> [#uses=1]
	br i1 %exitcond.i.i.i50, label %bb39.bb40_crit_edge.i.i.i, label %bb20.i.i.i

bb39.bb40_crit_edge.i.i.i:		; preds = %bb38.i.i.i51
	%tmp35.i.i.i = add i32 %tmp34.i.i.i, -1		; <i32> [#uses=1]
	br label %bb40.i.i.i52

bb40.i.i.i52:		; preds = %bb39.bb40_crit_edge.i.i.i, %bb39.preheader.i.i.i, %bb12.i.i.i39
	%nNodes.0.lcssa.i.i.i = phi i32 [ %tmp35.i.i.i, %bb39.bb40_crit_edge.i.i.i ], [ %509, %bb39.preheader.i.i.i ], [ %509, %bb12.i.i.i39 ]		; <i32> [#uses=1]
	%705 = icmp sgt i32 %nNodes.0.lcssa.i.i.i, 515		; <i1> [#uses=1]
	br i1 %705, label %bb41.i.i.i54, label %bb49.preheader.i.i.i53

bb49.preheader.i.i.i53:		; preds = %bb40.i.i.i52
	br i1 %724, label %hbMakeCodeLengths.exit.i.i, label %bb45.preheader.i.i.i

bb41.i.i.i54:		; preds = %bb40.i.i.i52
	call fastcc void @panic(i8* getelementptr ([21 x i8]* @"\01LC12", i32 0, i32 0)) nounwind ssp
	unreachable

bb44.i.i.i55:		; preds = %bb45.preheader.i.i.i, %bb44.i.i.i55
	%j.017.i.i.i = phi i32 [ %tmp52.i.i.i, %bb44.i.i.i55 ], [ 0, %bb45.preheader.i.i.i ]		; <i32> [#uses=1]
	%k.016.i.i.i = phi i32 [ %707, %bb44.i.i.i55 ], [ %i.221.i.i.i, %bb45.preheader.i.i.i ]		; <i32> [#uses=1]
	%tmp52.i.i.i = add i32 %j.017.i.i.i, 1		; <i32> [#uses=2]
	%706 = getelementptr [516 x i32]* %parent.i.i.i, i32 0, i32 %k.016.i.i.i		; <i32*> [#uses=1]
	%707 = load i32* %706, align 4		; <i32> [#uses=2]
	%708 = getelementptr [516 x i32]* %parent.i.i.i, i32 0, i32 %707		; <i32*> [#uses=1]
	%709 = load i32* %708, align 4		; <i32> [#uses=1]
	%710 = icmp slt i32 %709, 0		; <i1> [#uses=1]
	br i1 %710, label %bb46.i.i.i, label %bb44.i.i.i55

bb46.i.i.i:		; preds = %bb45.preheader.i.i.i, %bb44.i.i.i55
	%j.0.lcssa.i.i.i = phi i32 [ 0, %bb45.preheader.i.i.i ], [ %tmp52.i.i.i, %bb44.i.i.i55 ]		; <i32> [#uses=2]
	%711 = trunc i32 %j.0.lcssa.i.i.i to i8		; <i8> [#uses=1]
	store i8 %711, i8* %scevgep55.i.i.i, align 1
	%712 = icmp sgt i32 %j.0.lcssa.i.i.i, 20		; <i1> [#uses=1]
	%tooLong.0.i.i.i = select i1 %712, i8 1, i8 %tooLong.120.i.i.i		; <i8> [#uses=2]
	%713 = icmp sgt i32 %tmp56.i.i.i, %509		; <i1> [#uses=1]
	br i1 %713, label %bb49.bb50_crit_edge.i.i.i, label %bb45.preheader.i.i.i

bb49.bb50_crit_edge.i.i.i:		; preds = %bb46.i.i.i
	%phitmp28.i.i.i = icmp eq i8 %tooLong.0.i.i.i, 0		; <i1> [#uses=1]
	br i1 %phitmp28.i.i.i, label %hbMakeCodeLengths.exit.i.i, label %bb53.preheader.i.i.i

bb45.preheader.i.i.i:		; preds = %bb46.i.i.i, %bb49.preheader.i.i.i53
	%indvar53.i.i.i = phi i32 [ %i.221.i.i.i, %bb46.i.i.i ], [ 0, %bb49.preheader.i.i.i53 ]		; <i32> [#uses=3]
	%tooLong.120.i.i.i = phi i8 [ %tooLong.0.i.i.i, %bb46.i.i.i ], [ 0, %bb49.preheader.i.i.i53 ]		; <i8> [#uses=1]
	%scevgep55.i.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 %t.789.i.i, i32 %indvar53.i.i.i		; <i8*> [#uses=1]
	%tmp56.i.i.i = add i32 %indvar53.i.i.i, 2		; <i32> [#uses=1]
	%i.221.i.i.i = add i32 %indvar53.i.i.i, 1		; <i32> [#uses=3]
	%scevgep58.i.i.i = getelementptr [516 x i32]* %parent.i.i.i, i32 0, i32 %i.221.i.i.i		; <i32*> [#uses=1]
	%714 = load i32* %scevgep58.i.i.i, align 4		; <i32> [#uses=1]
	%715 = icmp slt i32 %714, 0		; <i1> [#uses=1]
	br i1 %715, label %bb46.i.i.i, label %bb44.i.i.i55

bb53.preheader.i.i.i:		; preds = %bb49.bb50_crit_edge.i.i.i
	br i1 %725, label %bb52.i.i.i, label %bb12.i.i.i39

bb52.i.i.i:		; preds = %bb52.i.i.i, %bb53.preheader.i.i.i
	%indvar59.i.i.i = phi i32 [ %tmp63.i.i.i, %bb52.i.i.i ], [ 0, %bb53.preheader.i.i.i ]		; <i32> [#uses=1]
	%tmp63.i.i.i = add i32 %indvar59.i.i.i, 1		; <i32> [#uses=3]
	%scevgep64.i.i.i = getelementptr [516 x i32]* %weight.i.i.i, i32 0, i32 %tmp63.i.i.i		; <i32*> [#uses=2]
	%716 = load i32* %scevgep64.i.i.i, align 4		; <i32> [#uses=1]
	%717 = ashr i32 %716, 8		; <i32> [#uses=1]
	%718 = sdiv i32 %717, 2		; <i32> [#uses=1]
	%719 = shl i32 %718, 8		; <i32> [#uses=1]
	%720 = add i32 %719, 256		; <i32> [#uses=1]
	store i32 %720, i32* %scevgep64.i.i.i, align 4
	%exitcond62.i.i.i = icmp eq i32 %tmp63.i.i.i, %tmp61.i.i.i		; <i1> [#uses=1]
	br i1 %exitcond62.i.i.i, label %bb12.i.i.i39, label %bb52.i.i.i

hbMakeCodeLengths.exit.i.i:		; preds = %bb49.bb50_crit_edge.i.i.i, %bb49.preheader.i.i.i53
	%721 = add i32 %t.789.i.i, 1		; <i32> [#uses=2]
	%exitcond238.i.i = icmp eq i32 %721, %smax181.i.i		; <i1> [#uses=1]
	br i1 %exitcond238.i.i, label %bb84.i.i, label %bb82.i.i

bb84.i.i:		; preds = %hbMakeCodeLengths.exit.i.i
	%exitcond241.i.i = icmp eq i32 %tmp242.i.i, 4		; <i1> [#uses=1]
	br i1 %exitcond241.i.i, label %bb88.i.i, label %bb41.i.i33.preheader

bb85.loopexit.i.i:		; preds = %bb37.i.i32
	%722 = icmp eq i32 %nGroups.0.i.i.reg2mem.0, 6		; <i1> [#uses=1]
	%tmp180.i.i = icmp ugt i32 %nGroups.0.i.i.reg2mem.0, 1		; <i1> [#uses=1]
	%smax181.i.i = select i1 %tmp180.i.i, i32 %nGroups.0.i.i.reg2mem.0, i32 1		; <i32> [#uses=8]
	%723 = icmp sgt i32 %512, 0		; <i1> [#uses=1]
	%724 = icmp slt i32 %509, 1		; <i1> [#uses=3]
	%725 = icmp sgt i32 %509, 1		; <i1> [#uses=1]
	%tmp37.i.i.i = add i32 %508, 3		; <i32> [#uses=1]
	%tmp61.i.i.i = add i32 %508, 1		; <i32> [#uses=1]
	%726 = add i32 %512, -1		; <i32> [#uses=1]
	%727 = load i16** @szptr, align 4		; <i16*> [#uses=3]
	br label %bb41.i.i33.preheader

bb41.i.i33.preheader:		; preds = %bb85.loopexit.i.i, %bb84.i.i
	%iter.091.i.i = phi i32 [ 0, %bb85.loopexit.i.i ], [ %tmp242.i.i, %bb84.i.i ]		; <i32> [#uses=1]
	%tmp242.i.i = add i32 %iter.091.i.i, 1		; <i32> [#uses=2]
	br label %bb41.i.i33

bb88.i.i:		; preds = %bb84.i.i
	%728 = icmp sgt i32 %nSelectors.1.lcssa.i.i, 18002		; <i1> [#uses=1]
	br i1 %728, label %bb89.i.i, label %bb91.i.i

bb89.i.i:		; preds = %bb88.i.i
	call fastcc void @panic(i8* getelementptr ([17 x i8]* @"\01LC40", i32 0, i32 0)) nounwind ssp
	unreachable

bb91.i.i:		; preds = %bb91.i.i, %bb88.i.i
	%i.345.i.i = phi i32 [ %729, %bb91.i.i ], [ 0, %bb88.i.i ]		; <i32> [#uses=3]
	%scevgep178.i.i = getelementptr [6 x i8]* %pos.i.i, i32 0, i32 %i.345.i.i		; <i8*> [#uses=1]
	%tmp179.i.i = trunc i32 %i.345.i.i to i8		; <i8> [#uses=1]
	store i8 %tmp179.i.i, i8* %scevgep178.i.i, align 1
	%729 = add i32 %i.345.i.i, 1		; <i32> [#uses=2]
	%exitcond177.i.i = icmp eq i32 %729, %smax181.i.i		; <i1> [#uses=1]
	br i1 %exitcond177.i.i, label %bb98.loopexit.i.i, label %bb91.i.i

bb.nph44.i.i:		; preds = %bb98.loopexit.i.i
	%.pre.i72.i = load i8* %97, align 1		; <i8> [#uses=1]
	br label %bb94.i.i

bb94.i.i:		; preds = %bb97.i.i, %bb.nph44.i.i
	%730 = phi i8 [ %.pre.i72.i, %bb.nph44.i.i ], [ %tmp.0.lcssa.i.i, %bb97.i.i ]		; <i8> [#uses=3]
	%i.443.i.i = phi i32 [ 0, %bb.nph44.i.i ], [ %735, %bb97.i.i ]		; <i32> [#uses=3]
	%scevgep173.i.i = getelementptr [18002 x i8]* @selector, i32 0, i32 %i.443.i.i		; <i8*> [#uses=1]
	%scevgep174.i.i = getelementptr [18002 x i8]* @selectorMtf, i32 0, i32 %i.443.i.i		; <i8*> [#uses=1]
	%731 = load i8* %scevgep173.i.i, align 1		; <i8> [#uses=2]
	%732 = icmp eq i8 %731, %730		; <i1> [#uses=1]
	br i1 %732, label %bb97.i.i, label %bb95.i.i

bb95.i.i:		; preds = %bb95.i.i, %bb94.i.i
	%tmp.039.i.i = phi i8 [ %733, %bb95.i.i ], [ %730, %bb94.i.i ]		; <i8> [#uses=1]
	%j.038.i.i = phi i32 [ %tmp170.i.i, %bb95.i.i ], [ 0, %bb94.i.i ]		; <i32> [#uses=1]
	%tmp170.i.i = add i32 %j.038.i.i, 1		; <i32> [#uses=3]
	%scevgep171.i.i = getelementptr [6 x i8]* %pos.i.i, i32 0, i32 %tmp170.i.i		; <i8*> [#uses=2]
	%733 = load i8* %scevgep171.i.i, align 1		; <i8> [#uses=3]
	store i8 %tmp.039.i.i, i8* %scevgep171.i.i, align 1
	%734 = icmp eq i8 %731, %733		; <i1> [#uses=1]
	br i1 %734, label %bb96.bb97_crit_edge.i.i, label %bb95.i.i

bb96.bb97_crit_edge.i.i:		; preds = %bb95.i.i
	%phitmp119.i.i = trunc i32 %tmp170.i.i to i8		; <i8> [#uses=1]
	br label %bb97.i.i

bb97.i.i:		; preds = %bb96.bb97_crit_edge.i.i, %bb94.i.i
	%tmp.0.lcssa.i.i = phi i8 [ %733, %bb96.bb97_crit_edge.i.i ], [ %730, %bb94.i.i ]		; <i8> [#uses=2]
	%j.0.lcssa.i.i56 = phi i8 [ %phitmp119.i.i, %bb96.bb97_crit_edge.i.i ], [ 0, %bb94.i.i ]		; <i8> [#uses=1]
	store i8 %tmp.0.lcssa.i.i, i8* %97, align 1
	store i8 %j.0.lcssa.i.i56, i8* %scevgep174.i.i, align 1
	%735 = add i32 %i.443.i.i, 1		; <i32> [#uses=2]
	%exitcond172.i.i = icmp eq i32 %735, %nSelectors.1.lcssa.i.i		; <i1> [#uses=1]
	br i1 %exitcond172.i.i, label %bb112.i.i, label %bb94.i.i

bb98.loopexit.i.i:		; preds = %bb91.i.i
	%736 = icmp sgt i32 %nSelectors.1.lcssa.i.i, 0		; <i1> [#uses=2]
	br i1 %736, label %bb.nph44.i.i, label %bb112.i.i

bb101.i.i:		; preds = %bb106.preheader.i.i, %bb101.i.i
	%i.534.i.i = phi i32 [ %741, %bb101.i.i ], [ 0, %bb106.preheader.i.i ]		; <i32> [#uses=2]
	%maxLen.133.i.i = phi i32 [ %maxLen.0.i.i, %bb101.i.i ], [ 0, %bb106.preheader.i.i ]		; <i32> [#uses=2]
	%minLen.132.i.i = phi i32 [ %minLen.0.i.i, %bb101.i.i ], [ 32, %bb106.preheader.i.i ]		; <i32> [#uses=2]
	%scevgep167.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 %752, i32 %i.534.i.i		; <i8*> [#uses=1]
	%737 = load i8* %scevgep167.i.i, align 1		; <i8> [#uses=1]
	%738 = zext i8 %737 to i32		; <i32> [#uses=4]
	%739 = icmp sgt i32 %738, %maxLen.133.i.i		; <i1> [#uses=1]
	%maxLen.0.i.i = select i1 %739, i32 %738, i32 %maxLen.133.i.i		; <i32> [#uses=3]
	%740 = icmp slt i32 %738, %minLen.132.i.i		; <i1> [#uses=1]
	%minLen.0.i.i = select i1 %740, i32 %738, i32 %minLen.132.i.i		; <i32> [#uses=3]
	%741 = add i32 %i.534.i.i, 1		; <i32> [#uses=2]
	%exitcond166.i.i = icmp eq i32 %741, %509		; <i1> [#uses=1]
	br i1 %exitcond166.i.i, label %bb107.i.i, label %bb101.i.i

bb107.i.i:		; preds = %bb101.i.i
	%742 = icmp sgt i32 %maxLen.0.i.i, 20		; <i1> [#uses=1]
	br i1 %742, label %bb108.i.i, label %bb109.i.i

bb108.i.i:		; preds = %bb107.i.i
	call fastcc void @panic(i8* getelementptr ([17 x i8]* @"\01LC41", i32 0, i32 0)) nounwind ssp
	unreachable

bb109.i.i:		; preds = %bb107.i.i
	%743 = icmp sgt i32 %minLen.0.i.i, 0		; <i1> [#uses=1]
	br i1 %743, label %bb111.i.i, label %bb110.i.i

bb110.i.i:		; preds = %bb109.i.i
	call fastcc void @panic(i8* getelementptr ([17 x i8]* @"\01LC42", i32 0, i32 0)) nounwind ssp
	unreachable

bb111.i.i:		; preds = %bb106.preheader.i.i, %bb109.i.i
	%maxLen.1.lcssa.i.i.reg2mem.0 = phi i32 [ %maxLen.0.i.i, %bb109.i.i ], [ 0, %bb106.preheader.i.i ]		; <i32> [#uses=2]
	%minLen.1.lcssa.i.i.reg2mem.0 = phi i32 [ %minLen.0.i.i, %bb109.i.i ], [ 32, %bb106.preheader.i.i ]		; <i32> [#uses=3]
	%.not.i.i.i = icmp sle i32 %minLen.1.lcssa.i.i.reg2mem.0, %maxLen.1.lcssa.i.i.reg2mem.0		; <i1> [#uses=1]
	%or.cond.i.i.i = and i1 %.not.i.i.i, %510		; <i1> [#uses=1]
	br i1 %or.cond.i.i.i, label %bb.nph6.split.i.i.i, label %hbAssignCodes.exit.i.i

bb1.i.i.i58:		; preds = %bb4.preheader.i.i.i, %bb3.i.i.i60
	%vec.12.i.i.i = phi i32 [ %vec.24.i.i.i, %bb4.preheader.i.i.i ], [ %vec.0.i.i.i, %bb3.i.i.i60 ]		; <i32> [#uses=3]
	%i.01.i.i.i = phi i32 [ 0, %bb4.preheader.i.i.i ], [ %748, %bb3.i.i.i60 ]		; <i32> [#uses=3]
	%scevgep.i.i.i57 = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 %752, i32 %i.01.i.i.i		; <i8*> [#uses=1]
	%744 = load i8* %scevgep.i.i.i57, align 1		; <i8> [#uses=1]
	%745 = zext i8 %744 to i32		; <i32> [#uses=1]
	%746 = icmp eq i32 %745, %n.05.i.i.i		; <i1> [#uses=1]
	br i1 %746, label %bb2.i.i.i59, label %bb3.i.i.i60

bb2.i.i.i59:		; preds = %bb1.i.i.i58
	%scevgep7.i.i.i = getelementptr [6 x [258 x i32]]* @code, i32 0, i32 %752, i32 %i.01.i.i.i		; <i32*> [#uses=1]
	store i32 %vec.12.i.i.i, i32* %scevgep7.i.i.i, align 4
	%747 = add i32 %vec.12.i.i.i, 1		; <i32> [#uses=1]
	br label %bb3.i.i.i60

bb3.i.i.i60:		; preds = %bb2.i.i.i59, %bb1.i.i.i58
	%vec.0.i.i.i = phi i32 [ %747, %bb2.i.i.i59 ], [ %vec.12.i.i.i, %bb1.i.i.i58 ]		; <i32> [#uses=2]
	%748 = add i32 %i.01.i.i.i, 1		; <i32> [#uses=2]
	%exitcond161.i.i = icmp eq i32 %748, %509		; <i1> [#uses=1]
	br i1 %exitcond161.i.i, label %bb5.i.i.i, label %bb1.i.i.i58

bb5.i.i.i:		; preds = %bb3.i.i.i60
	%749 = shl i32 %vec.0.i.i.i, 1		; <i32> [#uses=1]
	%750 = icmp sgt i32 %tmp8.i.i.i, %maxLen.1.lcssa.i.i.reg2mem.0		; <i1> [#uses=1]
	%indvar.next.i.i.i61 = add i32 %indvar.i.i.i63, 1		; <i32> [#uses=1]
	br i1 %750, label %hbAssignCodes.exit.i.i, label %bb4.preheader.i.i.i

bb.nph6.split.i.i.i:		; preds = %bb111.i.i
	%tmp.i.i.i62 = add i32 %minLen.1.lcssa.i.i.reg2mem.0, 1		; <i32> [#uses=1]
	br label %bb4.preheader.i.i.i

bb4.preheader.i.i.i:		; preds = %bb.nph6.split.i.i.i, %bb5.i.i.i
	%indvar.i.i.i63 = phi i32 [ 0, %bb.nph6.split.i.i.i ], [ %indvar.next.i.i.i61, %bb5.i.i.i ]		; <i32> [#uses=3]
	%vec.24.i.i.i = phi i32 [ 0, %bb.nph6.split.i.i.i ], [ %749, %bb5.i.i.i ]		; <i32> [#uses=1]
	%tmp8.i.i.i = add i32 %indvar.i.i.i63, %tmp.i.i.i62		; <i32> [#uses=1]
	%n.05.i.i.i = add i32 %indvar.i.i.i63, %minLen.1.lcssa.i.i.reg2mem.0		; <i32> [#uses=1]
	br label %bb1.i.i.i58

hbAssignCodes.exit.i.i:		; preds = %bb5.i.i.i, %bb111.i.i
	%751 = add i32 %752, 1		; <i32> [#uses=1]
	br label %bb112.i.i

bb112.i.i:		; preds = %hbAssignCodes.exit.i.i, %bb98.loopexit.i.i, %bb97.i.i
	%752 = phi i32 [ %751, %hbAssignCodes.exit.i.i ], [ 0, %bb97.i.i ], [ 0, %bb98.loopexit.i.i ]		; <i32> [#uses=5]
	%753 = icmp slt i32 %752, %nGroups.0.i.i.reg2mem.0		; <i1> [#uses=1]
	br i1 %753, label %bb106.preheader.i.i, label %bb114.i.i

bb106.preheader.i.i:		; preds = %bb112.i.i
	br i1 %510, label %bb101.i.i, label %bb111.i.i

bb114.i.i:		; preds = %bb119.i.i, %bb112.i.i
	%i.630.i.i = phi i32 [ %757, %bb119.i.i ], [ 0, %bb112.i.i ]		; <i32> [#uses=3]
	%tmp154.i.i = shl i32 %i.630.i.i, 4		; <i32> [#uses=1]
	%scevgep159.i.i = getelementptr [16 x i8]* %inUse16.i.i, i32 0, i32 %i.630.i.i		; <i8*> [#uses=2]
	store i8 0, i8* %scevgep159.i.i, align 1
	br label %bb115.i.i

bb115.i.i:		; preds = %bb117.i.i, %bb114.i.i
	%j.128.i.i = phi i32 [ 0, %bb114.i.i ], [ %756, %bb117.i.i ]		; <i32> [#uses=2]
	%tmp155.i.i = add i32 %j.128.i.i, %tmp154.i.i		; <i32> [#uses=1]
	%scevgep156.i.i = getelementptr [256 x i8]* @inUse, i32 0, i32 %tmp155.i.i		; <i8*> [#uses=1]
	%754 = load i8* %scevgep156.i.i, align 1		; <i8> [#uses=1]
	%755 = icmp eq i8 %754, 0		; <i1> [#uses=1]
	br i1 %755, label %bb117.i.i, label %bb116.i.i

bb116.i.i:		; preds = %bb115.i.i
	store i8 1, i8* %scevgep159.i.i, align 1
	br label %bb117.i.i

bb117.i.i:		; preds = %bb116.i.i, %bb115.i.i
	%756 = add i32 %j.128.i.i, 1		; <i32> [#uses=2]
	%exitcond153.i.i = icmp eq i32 %756, 16		; <i1> [#uses=1]
	br i1 %exitcond153.i.i, label %bb119.i.i, label %bb115.i.i

bb119.i.i:		; preds = %bb117.i.i
	%757 = add i32 %i.630.i.i, 1		; <i32> [#uses=2]
	%exitcond157.i.i = icmp eq i32 %757, 16		; <i1> [#uses=1]
	br i1 %exitcond157.i.i, label %bb121.i.i, label %bb114.i.i

bb121.i.i:		; preds = %bb119.i.i
	%.b.i87.i.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%758 = zext i1 %.b.i87.i.i to i32		; <i32> [#uses=3]
	%759 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %758, i32 3		; <i8**> [#uses=2]
	%760 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %758, i32 2		; <i32*> [#uses=4]
	%761 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %758, i32 1		; <i32*> [#uses=4]
	%bsBuff.promoted462 = load i32* @bsBuff		; <i32> [#uses=1]
	%bsLive.promoted463 = load i32* @bsLive		; <i32> [#uses=1]
	br label %bb122.i.i

bb122.i.i:		; preds = %bb125.i.i, %bb121.i.i
	%bsLive.tmp.0466 = phi i32 [ %bsLive.promoted463, %bb121.i.i ], [ %storemerge, %bb125.i.i ]		; <i32> [#uses=6]
	%bsBuff.tmp.1465 = phi i32 [ %bsBuff.promoted462, %bb121.i.i ], [ %bsBuff.tmp.0464, %bb125.i.i ]		; <i32> [#uses=4]
	%i.726.i.i = phi i32 [ 0, %bb121.i.i ], [ %795, %bb125.i.i ]		; <i32> [#uses=2]
	%scevgep152.i.i = getelementptr [16 x i8]* %inUse16.i.i, i32 0, i32 %i.726.i.i		; <i8*> [#uses=1]
	%762 = load i8* %scevgep152.i.i, align 1		; <i8> [#uses=1]
	%763 = icmp eq i8 %762, 0		; <i1> [#uses=1]
	br i1 %763, label %bb1.i89.i.i, label %bb1.i6.i.i

bb.i5.i.i:		; preds = %bb1.i6.i.i
	%764 = lshr i32 %775, 24		; <i32> [#uses=1]
	%765 = trunc i32 %764 to i8		; <i8> [#uses=1]
	%766 = load i8** %759, align 4		; <i8*> [#uses=1]
	%767 = load i32* %760, align 8		; <i32> [#uses=2]
	%768 = getelementptr i8* %766, i32 %767		; <i8*> [#uses=1]
	store i8 %765, i8* %768, align 1
	%769 = add i32 %767, 1		; <i32> [#uses=1]
	store i32 %769, i32* %760, align 8
	%770 = load i32* %761, align 4		; <i32> [#uses=1]
	%771 = add i32 %770, 1		; <i32> [#uses=1]
	store i32 %771, i32* %761, align 4
	%772 = shl i32 %bsBuff.tmp.0456, 8		; <i32> [#uses=2]
	%773 = add i32 %bsLive.tmp.0457, -8		; <i32> [#uses=3]
	br label %bb1.i6.i.i

bb1.i6.i.i:		; preds = %bb.i5.i.i, %bb122.i.i
	%bsLive.tmp.0457 = phi i32 [ %773, %bb.i5.i.i ], [ %bsLive.tmp.0466, %bb122.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0456 = phi i32 [ %772, %bb.i5.i.i ], [ %bsBuff.tmp.1465, %bb122.i.i ]		; <i32> [#uses=2]
	%774 = phi i32 [ %773, %bb.i5.i.i ], [ %bsLive.tmp.0466, %bb122.i.i ]		; <i32> [#uses=2]
	%775 = phi i32 [ %772, %bb.i5.i.i ], [ %bsBuff.tmp.1465, %bb122.i.i ]		; <i32> [#uses=1]
	%776 = phi i32 [ %773, %bb.i5.i.i ], [ %bsLive.tmp.0466, %bb122.i.i ]		; <i32> [#uses=1]
	%777 = icmp sgt i32 %776, 7		; <i1> [#uses=1]
	br i1 %777, label %bb.i5.i.i, label %bsW.exit.i.i

bsW.exit.i.i:		; preds = %bb1.i6.i.i
	%778 = sub i32 31, %774		; <i32> [#uses=1]
	%779 = shl i32 1, %778		; <i32> [#uses=1]
	%780 = or i32 %779, %bsBuff.tmp.0456		; <i32> [#uses=1]
	br label %bb125.i.i

bb.i88.i.i:		; preds = %bb1.i89.i.i
	%781 = lshr i32 %792, 24		; <i32> [#uses=1]
	%782 = trunc i32 %781 to i8		; <i8> [#uses=1]
	%783 = load i8** %759, align 4		; <i8*> [#uses=1]
	%784 = load i32* %760, align 8		; <i32> [#uses=2]
	%785 = getelementptr i8* %783, i32 %784		; <i8*> [#uses=1]
	store i8 %782, i8* %785, align 1
	%786 = add i32 %784, 1		; <i32> [#uses=1]
	store i32 %786, i32* %760, align 8
	%787 = load i32* %761, align 4		; <i32> [#uses=1]
	%788 = add i32 %787, 1		; <i32> [#uses=1]
	store i32 %788, i32* %761, align 4
	%789 = shl i32 %bsBuff.tmp.0460, 8		; <i32> [#uses=2]
	%790 = add i32 %bsLive.tmp.0461, -8		; <i32> [#uses=3]
	br label %bb1.i89.i.i

bb1.i89.i.i:		; preds = %bb.i88.i.i, %bb122.i.i
	%bsLive.tmp.0461 = phi i32 [ %790, %bb.i88.i.i ], [ %bsLive.tmp.0466, %bb122.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0460 = phi i32 [ %789, %bb.i88.i.i ], [ %bsBuff.tmp.1465, %bb122.i.i ]		; <i32> [#uses=2]
	%791 = phi i32 [ %790, %bb.i88.i.i ], [ %bsLive.tmp.0466, %bb122.i.i ]		; <i32> [#uses=1]
	%792 = phi i32 [ %789, %bb.i88.i.i ], [ %bsBuff.tmp.1465, %bb122.i.i ]		; <i32> [#uses=1]
	%793 = phi i32 [ %790, %bb.i88.i.i ], [ %bsLive.tmp.0466, %bb122.i.i ]		; <i32> [#uses=1]
	%794 = icmp sgt i32 %793, 7		; <i1> [#uses=1]
	br i1 %794, label %bb.i88.i.i, label %bb125.i.i

bb125.i.i:		; preds = %bb1.i89.i.i, %bsW.exit.i.i
	%bsBuff.tmp.0464 = phi i32 [ %780, %bsW.exit.i.i ], [ %bsBuff.tmp.0460, %bb1.i89.i.i ]		; <i32> [#uses=3]
	%storemerge.in = phi i32 [ %774, %bsW.exit.i.i ], [ %791, %bb1.i89.i.i ]		; <i32> [#uses=1]
	%storemerge = add i32 %storemerge.in, 1		; <i32> [#uses=3]
	%795 = add i32 %i.726.i.i, 1		; <i32> [#uses=2]
	%exitcond151.i.i = icmp eq i32 %795, 16		; <i1> [#uses=1]
	br i1 %exitcond151.i.i, label %bb128.i.i.preheader, label %bb122.i.i

bb128.i.i.preheader:		; preds = %bb125.i.i
	store i32 %bsBuff.tmp.0464, i32* @bsBuff
	store i32 %storemerge, i32* @bsLive
	%.b.i73.i.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%796 = zext i1 %.b.i73.i.i to i32		; <i32> [#uses=3]
	%797 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %796, i32 3		; <i8**> [#uses=2]
	%798 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %796, i32 2		; <i32*> [#uses=4]
	%799 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %796, i32 1		; <i32*> [#uses=4]
	br label %bb128.i.i

bb128.i.i:		; preds = %bb135.i.i, %bb128.i.i.preheader
	%bsLive.tmp.1485 = phi i32 [ %storemerge, %bb128.i.i.preheader ], [ %bsLive.tmp.0484, %bb135.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.1483 = phi i32 [ %bsBuff.tmp.0464, %bb128.i.i.preheader ], [ %bsBuff.tmp.0482, %bb135.i.i ]		; <i32> [#uses=2]
	%i.824.i.i = phi i32 [ %836, %bb135.i.i ], [ 0, %bb128.i.i.preheader ]		; <i32> [#uses=3]
	%tmp145.i.i = shl i32 %i.824.i.i, 4		; <i32> [#uses=1]
	%scevgep150.i.i = getelementptr [16 x i8]* %inUse16.i.i, i32 0, i32 %i.824.i.i		; <i8*> [#uses=1]
	%800 = load i8* %scevgep150.i.i, align 1		; <i8> [#uses=1]
	%801 = icmp eq i8 %800, 0		; <i1> [#uses=1]
	br i1 %801, label %bb135.i.i, label %bb130.i.i

bb130.i.i:		; preds = %bb133.i.i, %bb128.i.i
	%bsLive.tmp.0479 = phi i32 [ %storemerge85, %bb133.i.i ], [ %bsLive.tmp.1485, %bb128.i.i ]		; <i32> [#uses=6]
	%bsBuff.tmp.1478 = phi i32 [ %bsBuff.tmp.0477, %bb133.i.i ], [ %bsBuff.tmp.1483, %bb128.i.i ]		; <i32> [#uses=4]
	%j.222.i.i = phi i32 [ %835, %bb133.i.i ], [ 0, %bb128.i.i ]		; <i32> [#uses=2]
	%tmp146.i.i = add i32 %j.222.i.i, %tmp145.i.i		; <i32> [#uses=1]
	%scevgep147.i.i = getelementptr [256 x i8]* @inUse, i32 0, i32 %tmp146.i.i		; <i8*> [#uses=1]
	%802 = load i8* %scevgep147.i.i, align 1		; <i8> [#uses=1]
	%803 = icmp eq i8 %802, 0		; <i1> [#uses=1]
	br i1 %803, label %bb1.i75.i.i, label %bb1.i82.i.i

bb.i81.i.i:		; preds = %bb1.i82.i.i
	%804 = lshr i32 %815, 24		; <i32> [#uses=1]
	%805 = trunc i32 %804 to i8		; <i8> [#uses=1]
	%806 = load i8** %797, align 4		; <i8*> [#uses=1]
	%807 = load i32* %798, align 8		; <i32> [#uses=2]
	%808 = getelementptr i8* %806, i32 %807		; <i8*> [#uses=1]
	store i8 %805, i8* %808, align 1
	%809 = add i32 %807, 1		; <i32> [#uses=1]
	store i32 %809, i32* %798, align 8
	%810 = load i32* %799, align 4		; <i32> [#uses=1]
	%811 = add i32 %810, 1		; <i32> [#uses=1]
	store i32 %811, i32* %799, align 4
	%812 = shl i32 %bsBuff.tmp.0473, 8		; <i32> [#uses=2]
	%813 = add i32 %bsLive.tmp.0474, -8		; <i32> [#uses=3]
	br label %bb1.i82.i.i

bb1.i82.i.i:		; preds = %bb.i81.i.i, %bb130.i.i
	%bsLive.tmp.0474 = phi i32 [ %813, %bb.i81.i.i ], [ %bsLive.tmp.0479, %bb130.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0473 = phi i32 [ %812, %bb.i81.i.i ], [ %bsBuff.tmp.1478, %bb130.i.i ]		; <i32> [#uses=2]
	%814 = phi i32 [ %813, %bb.i81.i.i ], [ %bsLive.tmp.0479, %bb130.i.i ]		; <i32> [#uses=2]
	%815 = phi i32 [ %812, %bb.i81.i.i ], [ %bsBuff.tmp.1478, %bb130.i.i ]		; <i32> [#uses=1]
	%816 = phi i32 [ %813, %bb.i81.i.i ], [ %bsLive.tmp.0479, %bb130.i.i ]		; <i32> [#uses=1]
	%817 = icmp sgt i32 %816, 7		; <i1> [#uses=1]
	br i1 %817, label %bb.i81.i.i, label %bsW.exit84.i.i

bsW.exit84.i.i:		; preds = %bb1.i82.i.i
	%818 = sub i32 31, %814		; <i32> [#uses=1]
	%819 = shl i32 1, %818		; <i32> [#uses=1]
	%820 = or i32 %819, %bsBuff.tmp.0473		; <i32> [#uses=1]
	br label %bb133.i.i

bb.i74.i.i:		; preds = %bb1.i75.i.i
	%821 = lshr i32 %832, 24		; <i32> [#uses=1]
	%822 = trunc i32 %821 to i8		; <i8> [#uses=1]
	%823 = load i8** %797, align 4		; <i8*> [#uses=1]
	%824 = load i32* %798, align 8		; <i32> [#uses=2]
	%825 = getelementptr i8* %823, i32 %824		; <i8*> [#uses=1]
	store i8 %822, i8* %825, align 1
	%826 = add i32 %824, 1		; <i32> [#uses=1]
	store i32 %826, i32* %798, align 8
	%827 = load i32* %799, align 4		; <i32> [#uses=1]
	%828 = add i32 %827, 1		; <i32> [#uses=1]
	store i32 %828, i32* %799, align 4
	%829 = shl i32 %bsBuff.tmp.0469, 8		; <i32> [#uses=2]
	%830 = add i32 %bsLive.tmp.0470, -8		; <i32> [#uses=3]
	br label %bb1.i75.i.i

bb1.i75.i.i:		; preds = %bb.i74.i.i, %bb130.i.i
	%bsLive.tmp.0470 = phi i32 [ %830, %bb.i74.i.i ], [ %bsLive.tmp.0479, %bb130.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0469 = phi i32 [ %829, %bb.i74.i.i ], [ %bsBuff.tmp.1478, %bb130.i.i ]		; <i32> [#uses=2]
	%831 = phi i32 [ %830, %bb.i74.i.i ], [ %bsLive.tmp.0479, %bb130.i.i ]		; <i32> [#uses=1]
	%832 = phi i32 [ %829, %bb.i74.i.i ], [ %bsBuff.tmp.1478, %bb130.i.i ]		; <i32> [#uses=1]
	%833 = phi i32 [ %830, %bb.i74.i.i ], [ %bsLive.tmp.0479, %bb130.i.i ]		; <i32> [#uses=1]
	%834 = icmp sgt i32 %833, 7		; <i1> [#uses=1]
	br i1 %834, label %bb.i74.i.i, label %bb133.i.i

bb133.i.i:		; preds = %bb1.i75.i.i, %bsW.exit84.i.i
	%bsBuff.tmp.0477 = phi i32 [ %820, %bsW.exit84.i.i ], [ %bsBuff.tmp.0469, %bb1.i75.i.i ]		; <i32> [#uses=2]
	%storemerge85.in = phi i32 [ %814, %bsW.exit84.i.i ], [ %831, %bb1.i75.i.i ]		; <i32> [#uses=1]
	%storemerge85 = add i32 %storemerge85.in, 1		; <i32> [#uses=2]
	%835 = add i32 %j.222.i.i, 1		; <i32> [#uses=2]
	%exitcond144.i.i = icmp eq i32 %835, 16		; <i1> [#uses=1]
	br i1 %exitcond144.i.i, label %bb135.i.i, label %bb130.i.i

bb135.i.i:		; preds = %bb133.i.i, %bb128.i.i
	%bsLive.tmp.0484 = phi i32 [ %bsLive.tmp.1485, %bb128.i.i ], [ %storemerge85, %bb133.i.i ]		; <i32> [#uses=5]
	%bsBuff.tmp.0482 = phi i32 [ %bsBuff.tmp.1483, %bb128.i.i ], [ %bsBuff.tmp.0477, %bb133.i.i ]		; <i32> [#uses=4]
	%836 = add i32 %i.824.i.i, 1		; <i32> [#uses=2]
	%exitcond148.i.i = icmp eq i32 %836, 16		; <i1> [#uses=1]
	br i1 %exitcond148.i.i, label %bb137.i.i, label %bb128.i.i

bb137.i.i:		; preds = %bb135.i.i
	store i32 %bsBuff.tmp.0482, i32* @bsBuff
	store i32 %bsLive.tmp.0484, i32* @bsLive
	%.b.i66.i.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%837 = zext i1 %.b.i66.i.i to i32		; <i32> [#uses=3]
	%838 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %837, i32 3		; <i8**> [#uses=1]
	%839 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %837, i32 2		; <i32*> [#uses=2]
	%840 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %837, i32 1		; <i32*> [#uses=2]
	br label %bb1.i68.i.i

bb.i67.i.i:		; preds = %bb1.i68.i.i
	%841 = lshr i32 %852, 24		; <i32> [#uses=1]
	%842 = trunc i32 %841 to i8		; <i8> [#uses=1]
	%843 = load i8** %838, align 4		; <i8*> [#uses=1]
	%844 = load i32* %839, align 8		; <i32> [#uses=2]
	%845 = getelementptr i8* %843, i32 %844		; <i8*> [#uses=1]
	store i8 %842, i8* %845, align 1
	%846 = add i32 %844, 1		; <i32> [#uses=1]
	store i32 %846, i32* %839, align 8
	%847 = load i32* %840, align 4		; <i32> [#uses=1]
	%848 = add i32 %847, 1		; <i32> [#uses=1]
	store i32 %848, i32* %840, align 4
	%849 = shl i32 %bsBuff.tmp.0488, 8		; <i32> [#uses=2]
	%850 = add i32 %bsLive.tmp.0489, -8		; <i32> [#uses=3]
	br label %bb1.i68.i.i

bb1.i68.i.i:		; preds = %bb.i67.i.i, %bb137.i.i
	%bsLive.tmp.0489 = phi i32 [ %bsLive.tmp.0484, %bb137.i.i ], [ %850, %bb.i67.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0488 = phi i32 [ %bsBuff.tmp.0482, %bb137.i.i ], [ %849, %bb.i67.i.i ]		; <i32> [#uses=2]
	%851 = phi i32 [ %850, %bb.i67.i.i ], [ %bsLive.tmp.0484, %bb137.i.i ]		; <i32> [#uses=2]
	%852 = phi i32 [ %849, %bb.i67.i.i ], [ %bsBuff.tmp.0482, %bb137.i.i ]		; <i32> [#uses=1]
	%853 = phi i32 [ %850, %bb.i67.i.i ], [ %bsLive.tmp.0484, %bb137.i.i ]		; <i32> [#uses=1]
	%854 = icmp sgt i32 %853, 7		; <i1> [#uses=1]
	br i1 %854, label %bb.i67.i.i, label %bsW.exit70.i.i

bsW.exit70.i.i:		; preds = %bb1.i68.i.i
	%855 = sub i32 29, %851		; <i32> [#uses=1]
	%856 = shl i32 %nGroups.0.i.i.reg2mem.0, %855		; <i32> [#uses=1]
	%857 = or i32 %856, %bsBuff.tmp.0488		; <i32> [#uses=3]
	store i32 %857, i32* @bsBuff, align 4
	%858 = add i32 %851, 3		; <i32> [#uses=4]
	store i32 %858, i32* @bsLive, align 4
	%.b.i59.i.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%859 = zext i1 %.b.i59.i.i to i32		; <i32> [#uses=3]
	%860 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %859, i32 3		; <i8**> [#uses=1]
	%861 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %859, i32 2		; <i32*> [#uses=2]
	%862 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %859, i32 1		; <i32*> [#uses=2]
	br label %bb1.i61.i.i

bb.i60.i.i:		; preds = %bb1.i61.i.i
	%863 = lshr i32 %874, 24		; <i32> [#uses=1]
	%864 = trunc i32 %863 to i8		; <i8> [#uses=1]
	%865 = load i8** %860, align 4		; <i8*> [#uses=1]
	%866 = load i32* %861, align 8		; <i32> [#uses=2]
	%867 = getelementptr i8* %865, i32 %866		; <i8*> [#uses=1]
	store i8 %864, i8* %867, align 1
	%868 = add i32 %866, 1		; <i32> [#uses=1]
	store i32 %868, i32* %861, align 8
	%869 = load i32* %862, align 4		; <i32> [#uses=1]
	%870 = add i32 %869, 1		; <i32> [#uses=1]
	store i32 %870, i32* %862, align 4
	%871 = shl i32 %bsBuff.tmp.0492, 8		; <i32> [#uses=2]
	%872 = add i32 %bsLive.tmp.0493, -8		; <i32> [#uses=3]
	br label %bb1.i61.i.i

bb1.i61.i.i:		; preds = %bb.i60.i.i, %bsW.exit70.i.i
	%bsLive.tmp.0493 = phi i32 [ %858, %bsW.exit70.i.i ], [ %872, %bb.i60.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0492 = phi i32 [ %857, %bsW.exit70.i.i ], [ %871, %bb.i60.i.i ]		; <i32> [#uses=2]
	%873 = phi i32 [ %872, %bb.i60.i.i ], [ %858, %bsW.exit70.i.i ]		; <i32> [#uses=2]
	%874 = phi i32 [ %871, %bb.i60.i.i ], [ %857, %bsW.exit70.i.i ]		; <i32> [#uses=1]
	%875 = phi i32 [ %872, %bb.i60.i.i ], [ %858, %bsW.exit70.i.i ]		; <i32> [#uses=1]
	%876 = icmp sgt i32 %875, 7		; <i1> [#uses=1]
	br i1 %876, label %bb.i60.i.i, label %bsW.exit63.i.i

bsW.exit63.i.i:		; preds = %bb1.i61.i.i
	%877 = sub i32 17, %873		; <i32> [#uses=1]
	%878 = shl i32 %nSelectors.1.lcssa.i.i, %877		; <i32> [#uses=1]
	%879 = or i32 %878, %bsBuff.tmp.0492		; <i32> [#uses=3]
	store i32 %879, i32* @bsBuff, align 4
	%880 = add i32 %873, 15		; <i32> [#uses=3]
	store i32 %880, i32* @bsLive, align 4
	br i1 %736, label %bb142.preheader.i.i.preheader, label %bb.nph17.i.i

bb142.preheader.i.i.preheader:		; preds = %bsW.exit63.i.i
	%.b.i45.i.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%881 = zext i1 %.b.i45.i.i to i32		; <i32> [#uses=3]
	%882 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %881, i32 3		; <i8**> [#uses=2]
	%883 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %881, i32 2		; <i32*> [#uses=4]
	%884 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %881, i32 1		; <i32*> [#uses=4]
	br label %bb142.preheader.i.i

bb.i53.i.i:		; preds = %bb1.i54.i.i
	%885 = lshr i32 %896, 24		; <i32> [#uses=1]
	%886 = trunc i32 %885 to i8		; <i8> [#uses=1]
	%887 = load i8** %882, align 4		; <i8*> [#uses=1]
	%888 = load i32* %883, align 8		; <i32> [#uses=2]
	%889 = getelementptr i8* %887, i32 %888		; <i8*> [#uses=1]
	store i8 %886, i8* %889, align 1
	%890 = add i32 %888, 1		; <i32> [#uses=1]
	store i32 %890, i32* %883, align 8
	%891 = load i32* %884, align 4		; <i32> [#uses=1]
	%892 = add i32 %891, 1		; <i32> [#uses=1]
	store i32 %892, i32* %884, align 4
	%893 = shl i32 %bsBuff.tmp.0386, 8		; <i32> [#uses=2]
	%894 = add i32 %bsLive.tmp.0387, -8		; <i32> [#uses=3]
	br label %bb1.i54.i.i

bb1.i54.i.i:		; preds = %bb141.i.i.preheader, %bsW.exit56.i.i, %bb.i53.i.i
	%j.318.i.i = phi i32 [ %903, %bsW.exit56.i.i ], [ 0, %bb141.i.i.preheader ], [ %j.318.i.i, %bb.i53.i.i ]		; <i32> [#uses=2]
	%bsLive.tmp.0387 = phi i32 [ %894, %bb.i53.i.i ], [ %bsLive.tmp.0400, %bb141.i.i.preheader ], [ %902, %bsW.exit56.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0386 = phi i32 [ %893, %bb.i53.i.i ], [ %bsBuff.tmp.0398, %bb141.i.i.preheader ], [ %901, %bsW.exit56.i.i ]		; <i32> [#uses=2]
	%895 = phi i32 [ %894, %bb.i53.i.i ], [ %bsLive.tmp.0400, %bb141.i.i.preheader ], [ %902, %bsW.exit56.i.i ]		; <i32> [#uses=2]
	%896 = phi i32 [ %893, %bb.i53.i.i ], [ %bsBuff.tmp.0398, %bb141.i.i.preheader ], [ %901, %bsW.exit56.i.i ]		; <i32> [#uses=1]
	%897 = phi i32 [ %894, %bb.i53.i.i ], [ %bsLive.tmp.0400, %bb141.i.i.preheader ], [ %902, %bsW.exit56.i.i ]		; <i32> [#uses=1]
	%898 = icmp sgt i32 %897, 7		; <i1> [#uses=1]
	br i1 %898, label %bb.i53.i.i, label %bsW.exit56.i.i

bsW.exit56.i.i:		; preds = %bb1.i54.i.i
	%899 = sub i32 31, %895		; <i32> [#uses=1]
	%900 = shl i32 1, %899		; <i32> [#uses=1]
	%901 = or i32 %900, %bsBuff.tmp.0386		; <i32> [#uses=4]
	%902 = add i32 %895, 1		; <i32> [#uses=6]
	%903 = add i32 %j.318.i.i, 1		; <i32> [#uses=2]
	%904 = icmp sgt i32 %923, %903		; <i1> [#uses=1]
	br i1 %904, label %bb1.i54.i.i, label %bb1.i47.i.i

bb.i46.i.i:		; preds = %bb1.i47.i.i
	%905 = lshr i32 %916, 24		; <i32> [#uses=1]
	%906 = trunc i32 %905 to i8		; <i8> [#uses=1]
	%907 = load i8** %882, align 4		; <i8*> [#uses=1]
	%908 = load i32* %883, align 8		; <i32> [#uses=2]
	%909 = getelementptr i8* %907, i32 %908		; <i8*> [#uses=1]
	store i8 %906, i8* %909, align 1
	%910 = add i32 %908, 1		; <i32> [#uses=1]
	store i32 %910, i32* %883, align 8
	%911 = load i32* %884, align 4		; <i32> [#uses=1]
	%912 = add i32 %911, 1		; <i32> [#uses=1]
	store i32 %912, i32* %884, align 4
	%913 = shl i32 %bsBuff.tmp.0394, 8		; <i32> [#uses=2]
	%914 = add i32 %bsLive.tmp.0395, -8		; <i32> [#uses=3]
	br label %bb1.i47.i.i

bb1.i47.i.i:		; preds = %bb142.preheader.i.i, %bb.i46.i.i, %bsW.exit56.i.i
	%bsLive.tmp.0395 = phi i32 [ %914, %bb.i46.i.i ], [ %bsLive.tmp.0400, %bb142.preheader.i.i ], [ %902, %bsW.exit56.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0394 = phi i32 [ %913, %bb.i46.i.i ], [ %bsBuff.tmp.0398, %bb142.preheader.i.i ], [ %901, %bsW.exit56.i.i ]		; <i32> [#uses=4]
	%915 = phi i32 [ %914, %bb.i46.i.i ], [ %bsLive.tmp.0400, %bb142.preheader.i.i ], [ %902, %bsW.exit56.i.i ]		; <i32> [#uses=1]
	%916 = phi i32 [ %913, %bb.i46.i.i ], [ %bsBuff.tmp.0398, %bb142.preheader.i.i ], [ %901, %bsW.exit56.i.i ]		; <i32> [#uses=1]
	%917 = phi i32 [ %914, %bb.i46.i.i ], [ %bsLive.tmp.0400, %bb142.preheader.i.i ], [ %902, %bsW.exit56.i.i ]		; <i32> [#uses=1]
	%918 = icmp sgt i32 %917, 7		; <i1> [#uses=1]
	br i1 %918, label %bb.i46.i.i, label %bsW.exit49.i.i

bsW.exit49.i.i:		; preds = %bb1.i47.i.i
	%919 = add i32 %915, 1		; <i32> [#uses=3]
	%920 = add i32 %i.920.i.i, 1		; <i32> [#uses=2]
	%exitcond142.i.i = icmp eq i32 %920, %nSelectors.1.lcssa.i.i		; <i1> [#uses=1]
	br i1 %exitcond142.i.i, label %bb145.i.i.loopexit, label %bb142.preheader.i.i

bb142.preheader.i.i:		; preds = %bsW.exit49.i.i, %bb142.preheader.i.i.preheader
	%bsLive.tmp.0400 = phi i32 [ %880, %bb142.preheader.i.i.preheader ], [ %919, %bsW.exit49.i.i ]		; <i32> [#uses=6]
	%bsBuff.tmp.0398 = phi i32 [ %879, %bb142.preheader.i.i.preheader ], [ %bsBuff.tmp.0394, %bsW.exit49.i.i ]		; <i32> [#uses=4]
	%i.920.i.i = phi i32 [ %920, %bsW.exit49.i.i ], [ 0, %bb142.preheader.i.i.preheader ]		; <i32> [#uses=2]
	%scevgep143.i.i = getelementptr [18002 x i8]* @selectorMtf, i32 0, i32 %i.920.i.i		; <i8*> [#uses=1]
	%921 = load i8* %scevgep143.i.i, align 1		; <i8> [#uses=2]
	%922 = icmp eq i8 %921, 0		; <i1> [#uses=1]
	br i1 %922, label %bb1.i47.i.i, label %bb141.i.i.preheader

bb141.i.i.preheader:		; preds = %bb142.preheader.i.i
	%923 = zext i8 %921 to i32		; <i32> [#uses=1]
	br label %bb1.i54.i.i

bb145.i.i.loopexit:		; preds = %bsW.exit49.i.i
	store i32 %bsBuff.tmp.0394, i32* @bsBuff
	store i32 %919, i32* @bsLive
	br label %bb.nph17.i.i

bb.nph17.i.i:		; preds = %bb145.i.i.loopexit, %bsW.exit63.i.i
	%bsLive.promoted379 = phi i32 [ %919, %bb145.i.i.loopexit ], [ %880, %bsW.exit63.i.i ]		; <i32> [#uses=1]
	%bsBuff.promoted378 = phi i32 [ %bsBuff.tmp.0394, %bb145.i.i.loopexit ], [ %879, %bsW.exit63.i.i ]		; <i32> [#uses=1]
	%.b.i38.i.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%924 = zext i1 %.b.i38.i.i to i32		; <i32> [#uses=3]
	%925 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %924, i32 3		; <i8**> [#uses=4]
	%926 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %924, i32 2		; <i32*> [#uses=8]
	%927 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %924, i32 1		; <i32*> [#uses=8]
	br label %bb148.i.i

bb148.i.i:		; preds = %bb155.i.i, %bb.nph17.i.i
	%bsLive.tmp.0382 = phi i32 [ %bsLive.promoted379, %bb.nph17.i.i ], [ %bsLive.tmp.1383, %bb155.i.i ]		; <i32> [#uses=3]
	%bsBuff.tmp.0380 = phi i32 [ %bsBuff.promoted378, %bb.nph17.i.i ], [ %bsBuff.tmp.1381, %bb155.i.i ]		; <i32> [#uses=2]
	%928 = phi i32 [ 0, %bb.nph17.i.i ], [ %1009, %bb155.i.i ]		; <i32> [#uses=3]
	%scevgep140.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 %928, i32 0		; <i8*> [#uses=1]
	%929 = load i8* %scevgep140.i.i, align 2		; <i8> [#uses=1]
	%930 = zext i8 %929 to i32		; <i32> [#uses=2]
	br label %bb1.i40.i.i

bb.i39.i.i:		; preds = %bb1.i40.i.i
	%931 = lshr i32 %942, 24		; <i32> [#uses=1]
	%932 = trunc i32 %931 to i8		; <i8> [#uses=1]
	%933 = load i8** %925, align 4		; <i8*> [#uses=1]
	%934 = load i32* %926, align 8		; <i32> [#uses=2]
	%935 = getelementptr i8* %933, i32 %934		; <i8*> [#uses=1]
	store i8 %932, i8* %935, align 1
	%936 = add i32 %934, 1		; <i32> [#uses=1]
	store i32 %936, i32* %926, align 8
	%937 = load i32* %927, align 4		; <i32> [#uses=1]
	%938 = add i32 %937, 1		; <i32> [#uses=1]
	store i32 %938, i32* %927, align 4
	%939 = shl i32 %bsBuff.tmp.0376, 8		; <i32> [#uses=2]
	%940 = add i32 %bsLive.tmp.0377, -8		; <i32> [#uses=3]
	br label %bb1.i40.i.i

bb1.i40.i.i:		; preds = %bb.i39.i.i, %bb148.i.i
	%bsLive.tmp.0377 = phi i32 [ %bsLive.tmp.0382, %bb148.i.i ], [ %940, %bb.i39.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0376 = phi i32 [ %bsBuff.tmp.0380, %bb148.i.i ], [ %939, %bb.i39.i.i ]		; <i32> [#uses=2]
	%941 = phi i32 [ %940, %bb.i39.i.i ], [ %bsLive.tmp.0382, %bb148.i.i ]		; <i32> [#uses=2]
	%942 = phi i32 [ %939, %bb.i39.i.i ], [ %bsBuff.tmp.0380, %bb148.i.i ]		; <i32> [#uses=1]
	%943 = phi i32 [ %940, %bb.i39.i.i ], [ %bsLive.tmp.0382, %bb148.i.i ]		; <i32> [#uses=1]
	%944 = icmp sgt i32 %943, 7		; <i1> [#uses=1]
	br i1 %944, label %bb.i39.i.i, label %bsW.exit42.i.i

bsW.exit42.i.i:		; preds = %bb1.i40.i.i
	%945 = sub i32 27, %941		; <i32> [#uses=1]
	%946 = shl i32 %930, %945		; <i32> [#uses=1]
	%947 = or i32 %946, %bsBuff.tmp.0376		; <i32> [#uses=2]
	%948 = add i32 %941, 5		; <i32> [#uses=2]
	br i1 %510, label %bb150.preheader.i.i, label %bb155.i.i

bb.nph8.i.i:		; preds = %bb150.preheader.i.i
	%tmp128.i.i = add i32 %curr.114.i.i, 1		; <i32> [#uses=1]
	br label %bb149.i.i

bb149.i.i:		; preds = %bsW.exit35.i.i, %bb.nph8.i.i
	%bsLive.tmp.0363 = phi i32 [ %bsLive.tmp.0372, %bb.nph8.i.i ], [ %966, %bsW.exit35.i.i ]		; <i32> [#uses=3]
	%bsBuff.tmp.0362 = phi i32 [ %bsBuff.tmp.0370, %bb.nph8.i.i ], [ %965, %bsW.exit35.i.i ]		; <i32> [#uses=2]
	%indvar126.i.i = phi i32 [ 0, %bb.nph8.i.i ], [ %indvar.next127.i.i, %bsW.exit35.i.i ]		; <i32> [#uses=2]
	%tmp129.i.i = add i32 %indvar126.i.i, %tmp128.i.i		; <i32> [#uses=2]
	br label %bb1.i33.i.i

bb.i32.i.i:		; preds = %bb1.i33.i.i
	%949 = lshr i32 %960, 24		; <i32> [#uses=1]
	%950 = trunc i32 %949 to i8		; <i8> [#uses=1]
	%951 = load i8** %925, align 4		; <i8*> [#uses=1]
	%952 = load i32* %926, align 8		; <i32> [#uses=2]
	%953 = getelementptr i8* %951, i32 %952		; <i8*> [#uses=1]
	store i8 %950, i8* %953, align 1
	%954 = add i32 %952, 1		; <i32> [#uses=1]
	store i32 %954, i32* %926, align 8
	%955 = load i32* %927, align 4		; <i32> [#uses=1]
	%956 = add i32 %955, 1		; <i32> [#uses=1]
	store i32 %956, i32* %927, align 4
	%957 = shl i32 %bsBuff.tmp.0358, 8		; <i32> [#uses=2]
	%958 = add i32 %bsLive.tmp.0359, -8		; <i32> [#uses=3]
	br label %bb1.i33.i.i

bb1.i33.i.i:		; preds = %bb.i32.i.i, %bb149.i.i
	%bsLive.tmp.0359 = phi i32 [ %bsLive.tmp.0363, %bb149.i.i ], [ %958, %bb.i32.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0358 = phi i32 [ %bsBuff.tmp.0362, %bb149.i.i ], [ %957, %bb.i32.i.i ]		; <i32> [#uses=2]
	%959 = phi i32 [ %958, %bb.i32.i.i ], [ %bsLive.tmp.0363, %bb149.i.i ]		; <i32> [#uses=2]
	%960 = phi i32 [ %957, %bb.i32.i.i ], [ %bsBuff.tmp.0362, %bb149.i.i ]		; <i32> [#uses=1]
	%961 = phi i32 [ %958, %bb.i32.i.i ], [ %bsLive.tmp.0363, %bb149.i.i ]		; <i32> [#uses=1]
	%962 = icmp sgt i32 %961, 7		; <i1> [#uses=1]
	br i1 %962, label %bb.i32.i.i, label %bsW.exit35.i.i

bsW.exit35.i.i:		; preds = %bb1.i33.i.i
	%963 = sub i32 30, %959		; <i32> [#uses=1]
	%964 = shl i32 2, %963		; <i32> [#uses=1]
	%965 = or i32 %964, %bsBuff.tmp.0358		; <i32> [#uses=2]
	%966 = add i32 %959, 2		; <i32> [#uses=2]
	%967 = icmp sgt i32 %1007, %tmp129.i.i		; <i1> [#uses=1]
	%indvar.next127.i.i = add i32 %indvar126.i.i, 1		; <i32> [#uses=1]
	br i1 %967, label %bb149.i.i, label %bb152.preheader.i.i

bb152.preheader.i.i:		; preds = %bb150.preheader.i.i, %bsW.exit35.i.i
	%bsLive.tmp.1373 = phi i32 [ %bsLive.tmp.0372, %bb150.preheader.i.i ], [ %966, %bsW.exit35.i.i ]		; <i32> [#uses=4]
	%bsBuff.tmp.1371 = phi i32 [ %bsBuff.tmp.0370, %bb150.preheader.i.i ], [ %965, %bsW.exit35.i.i ]		; <i32> [#uses=3]
	%curr.0.lcssa.i.i = phi i32 [ %curr.114.i.i, %bb150.preheader.i.i ], [ %tmp129.i.i, %bsW.exit35.i.i ]		; <i32> [#uses=3]
	%968 = icmp slt i32 %1007, %curr.0.lcssa.i.i		; <i1> [#uses=1]
	br i1 %968, label %bb.nph11.i.i, label %bb1.i19.i.i

bb.nph11.i.i:		; preds = %bb152.preheader.i.i
	%tmp134.i.i = add i32 %curr.0.lcssa.i.i, -1		; <i32> [#uses=1]
	%969 = load i8* %scevgep137.i.i, align 1		; <i8> [#uses=1]
	%970 = zext i8 %969 to i32		; <i32> [#uses=1]
	br label %bb151.i.i

bb151.i.i:		; preds = %bsW.exit28.i.i, %bb.nph11.i.i
	%bsLive.tmp.0355 = phi i32 [ %bsLive.tmp.1373, %bb.nph11.i.i ], [ %988, %bsW.exit28.i.i ]		; <i32> [#uses=3]
	%bsBuff.tmp.0354 = phi i32 [ %bsBuff.tmp.1371, %bb.nph11.i.i ], [ %987, %bsW.exit28.i.i ]		; <i32> [#uses=2]
	%indvar131.i.i = phi i32 [ 0, %bb.nph11.i.i ], [ %indvar.next132.i.i, %bsW.exit28.i.i ]		; <i32> [#uses=2]
	%tmp135.i.i = sub i32 %tmp134.i.i, %indvar131.i.i		; <i32> [#uses=2]
	br label %bb1.i26.i.i

bb.i25.i.i:		; preds = %bb1.i26.i.i
	%971 = lshr i32 %982, 24		; <i32> [#uses=1]
	%972 = trunc i32 %971 to i8		; <i8> [#uses=1]
	%973 = load i8** %925, align 4		; <i8*> [#uses=1]
	%974 = load i32* %926, align 8		; <i32> [#uses=2]
	%975 = getelementptr i8* %973, i32 %974		; <i8*> [#uses=1]
	store i8 %972, i8* %975, align 1
	%976 = add i32 %974, 1		; <i32> [#uses=1]
	store i32 %976, i32* %926, align 8
	%977 = load i32* %927, align 4		; <i32> [#uses=1]
	%978 = add i32 %977, 1		; <i32> [#uses=1]
	store i32 %978, i32* %927, align 4
	%979 = shl i32 %bsBuff.tmp.0350, 8		; <i32> [#uses=2]
	%980 = add i32 %bsLive.tmp.0351, -8		; <i32> [#uses=3]
	br label %bb1.i26.i.i

bb1.i26.i.i:		; preds = %bb.i25.i.i, %bb151.i.i
	%bsLive.tmp.0351 = phi i32 [ %bsLive.tmp.0355, %bb151.i.i ], [ %980, %bb.i25.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0350 = phi i32 [ %bsBuff.tmp.0354, %bb151.i.i ], [ %979, %bb.i25.i.i ]		; <i32> [#uses=2]
	%981 = phi i32 [ %980, %bb.i25.i.i ], [ %bsLive.tmp.0355, %bb151.i.i ]		; <i32> [#uses=2]
	%982 = phi i32 [ %979, %bb.i25.i.i ], [ %bsBuff.tmp.0354, %bb151.i.i ]		; <i32> [#uses=1]
	%983 = phi i32 [ %980, %bb.i25.i.i ], [ %bsLive.tmp.0355, %bb151.i.i ]		; <i32> [#uses=1]
	%984 = icmp sgt i32 %983, 7		; <i1> [#uses=1]
	br i1 %984, label %bb.i25.i.i, label %bsW.exit28.i.i

bsW.exit28.i.i:		; preds = %bb1.i26.i.i
	%985 = sub i32 30, %981		; <i32> [#uses=1]
	%986 = shl i32 3, %985		; <i32> [#uses=1]
	%987 = or i32 %986, %bsBuff.tmp.0350		; <i32> [#uses=3]
	%988 = add i32 %981, 2		; <i32> [#uses=4]
	%989 = icmp slt i32 %970, %tmp135.i.i		; <i1> [#uses=1]
	%indvar.next132.i.i = add i32 %indvar131.i.i, 1		; <i32> [#uses=1]
	br i1 %989, label %bb151.i.i, label %bb1.i19.i.i

bb.i18.i.i:		; preds = %bb1.i19.i.i
	%990 = lshr i32 %1001, 24		; <i32> [#uses=1]
	%991 = trunc i32 %990 to i8		; <i8> [#uses=1]
	%992 = load i8** %925, align 4		; <i8*> [#uses=1]
	%993 = load i32* %926, align 8		; <i32> [#uses=2]
	%994 = getelementptr i8* %992, i32 %993		; <i8*> [#uses=1]
	store i8 %991, i8* %994, align 1
	%995 = add i32 %993, 1		; <i32> [#uses=1]
	store i32 %995, i32* %926, align 8
	%996 = load i32* %927, align 4		; <i32> [#uses=1]
	%997 = add i32 %996, 1		; <i32> [#uses=1]
	store i32 %997, i32* %927, align 4
	%998 = shl i32 %bsBuff.tmp.0366, 8		; <i32> [#uses=2]
	%999 = add i32 %bsLive.tmp.0367, -8		; <i32> [#uses=3]
	br label %bb1.i19.i.i

bb1.i19.i.i:		; preds = %bb.i18.i.i, %bsW.exit28.i.i, %bb152.preheader.i.i
	%curr.2.lcssa.i.i = phi i32 [ %curr.0.lcssa.i.i, %bb152.preheader.i.i ], [ %tmp135.i.i, %bsW.exit28.i.i ], [ %curr.2.lcssa.i.i, %bb.i18.i.i ]		; <i32> [#uses=2]
	%bsLive.tmp.0367 = phi i32 [ %999, %bb.i18.i.i ], [ %bsLive.tmp.1373, %bb152.preheader.i.i ], [ %988, %bsW.exit28.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0366 = phi i32 [ %998, %bb.i18.i.i ], [ %bsBuff.tmp.1371, %bb152.preheader.i.i ], [ %987, %bsW.exit28.i.i ]		; <i32> [#uses=3]
	%1000 = phi i32 [ %999, %bb.i18.i.i ], [ %bsLive.tmp.1373, %bb152.preheader.i.i ], [ %988, %bsW.exit28.i.i ]		; <i32> [#uses=1]
	%1001 = phi i32 [ %998, %bb.i18.i.i ], [ %bsBuff.tmp.1371, %bb152.preheader.i.i ], [ %987, %bsW.exit28.i.i ]		; <i32> [#uses=1]
	%1002 = phi i32 [ %999, %bb.i18.i.i ], [ %bsLive.tmp.1373, %bb152.preheader.i.i ], [ %988, %bsW.exit28.i.i ]		; <i32> [#uses=1]
	%1003 = icmp sgt i32 %1002, 7		; <i1> [#uses=1]
	br i1 %1003, label %bb.i18.i.i, label %bsW.exit21.i.i

bsW.exit21.i.i:		; preds = %bb1.i19.i.i
	%1004 = add i32 %1000, 1		; <i32> [#uses=2]
	%1005 = add i32 %i.1013.i.i, 1		; <i32> [#uses=2]
	%exitcond.i.i66 = icmp eq i32 %1005, %509		; <i1> [#uses=1]
	br i1 %exitcond.i.i66, label %bb155.i.i, label %bb150.preheader.i.i

bb150.preheader.i.i:		; preds = %bsW.exit21.i.i, %bsW.exit42.i.i
	%bsLive.tmp.0372 = phi i32 [ %1004, %bsW.exit21.i.i ], [ %948, %bsW.exit42.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0370 = phi i32 [ %bsBuff.tmp.0366, %bsW.exit21.i.i ], [ %947, %bsW.exit42.i.i ]		; <i32> [#uses=2]
	%curr.114.i.i = phi i32 [ %curr.2.lcssa.i.i, %bsW.exit21.i.i ], [ %930, %bsW.exit42.i.i ]		; <i32> [#uses=3]
	%i.1013.i.i = phi i32 [ %1005, %bsW.exit21.i.i ], [ 0, %bsW.exit42.i.i ]		; <i32> [#uses=2]
	%scevgep137.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 %928, i32 %i.1013.i.i		; <i8*> [#uses=2]
	%1006 = load i8* %scevgep137.i.i, align 1		; <i8> [#uses=1]
	%1007 = zext i8 %1006 to i32		; <i32> [#uses=3]
	%1008 = icmp sgt i32 %1007, %curr.114.i.i		; <i1> [#uses=1]
	br i1 %1008, label %bb.nph8.i.i, label %bb152.preheader.i.i

bb155.i.i:		; preds = %bsW.exit21.i.i, %bsW.exit42.i.i
	%bsLive.tmp.1383 = phi i32 [ %948, %bsW.exit42.i.i ], [ %1004, %bsW.exit21.i.i ]		; <i32> [#uses=3]
	%bsBuff.tmp.1381 = phi i32 [ %947, %bsW.exit42.i.i ], [ %bsBuff.tmp.0366, %bsW.exit21.i.i ]		; <i32> [#uses=3]
	%1009 = add i32 %928, 1		; <i32> [#uses=2]
	%exitcond139.i.i = icmp eq i32 %1009, %smax181.i.i		; <i1> [#uses=1]
	br i1 %exitcond139.i.i, label %bb157.i.i, label %bb148.i.i

bb157.i.i:		; preds = %bb155.i.i
	store i32 %bsBuff.tmp.1381, i32* @bsBuff
	store i32 %bsLive.tmp.1383, i32* @bsLive
	%1010 = load i32* @nMTF, align 4		; <i32> [#uses=4]
	%1011 = icmp sgt i32 %1010, 0		; <i1> [#uses=1]
	br i1 %1011, label %bb161.i.i.preheader, label %bb167.i.i

bb161.i.i.preheader:		; preds = %bb157.i.i
	%1012 = load i16** @szptr, align 4		; <i16*> [#uses=1]
	%.b.i10.i.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%1013 = zext i1 %.b.i10.i.i to i32		; <i32> [#uses=3]
	%1014 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1013, i32 3		; <i8**> [#uses=1]
	%1015 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1013, i32 2		; <i32*> [#uses=2]
	%1016 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1013, i32 1		; <i32*> [#uses=2]
	br label %bb161.i.i

bb161.i.i:		; preds = %bb166.i.i, %bb161.i.i.preheader
	%bsLive.tmp.1347 = phi i32 [ %bsLive.tmp.1383, %bb161.i.i.preheader ], [ %bsLive.tmp.0346, %bb166.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.1345 = phi i32 [ %bsBuff.tmp.1381, %bb161.i.i.preheader ], [ %bsBuff.tmp.0344, %bb166.i.i ]		; <i32> [#uses=2]
	%gs.24.i.i = phi i32 [ %1050, %bb166.i.i ], [ 0, %bb161.i.i.preheader ]		; <i32> [#uses=4]
	%selCtr.03.i.i = phi i32 [ %tmp124.i.i, %bb166.i.i ], [ 0, %bb161.i.i.preheader ]		; <i32> [#uses=2]
	%tmp124.i.i = add i32 %selCtr.03.i.i, 1		; <i32> [#uses=2]
	%1017 = add i32 %gs.24.i.i, 49		; <i32> [#uses=2]
	%1018 = add i32 %1010, -1		; <i32> [#uses=1]
	%1019 = icmp slt i32 %1017, %1010		; <i1> [#uses=1]
	%.285.i.i = select i1 %1019, i32 %1017, i32 %1018		; <i32> [#uses=3]
	%1020 = icmp sgt i32 %gs.24.i.i, %.285.i.i		; <i1> [#uses=1]
	br i1 %1020, label %bb166.i.i, label %bb.nph.i.i

bb.nph.i.i:		; preds = %bb161.i.i
	%scevgep123.i.i = getelementptr [18002 x i8]* @selector, i32 0, i32 %selCtr.03.i.i		; <i8*> [#uses=1]
	%tmp121.i.i = add i32 %gs.24.i.i, 1		; <i32> [#uses=1]
	%1021 = load i8* %scevgep123.i.i, align 1		; <i8> [#uses=1]
	%1022 = zext i8 %1021 to i32		; <i32> [#uses=2]
	br label %bb164.i.i

bb164.i.i:		; preds = %bsW.exit14.i.i, %bb.nph.i.i
	%bsLive.tmp.0341 = phi i32 [ %bsLive.tmp.1347, %bb.nph.i.i ], [ %1048, %bsW.exit14.i.i ]		; <i32> [#uses=3]
	%bsBuff.tmp.0340 = phi i32 [ %bsBuff.tmp.1345, %bb.nph.i.i ], [ %1047, %bsW.exit14.i.i ]		; <i32> [#uses=2]
	%indvar.i.i67 = phi i32 [ 0, %bb.nph.i.i ], [ %indvar.next.i.i69, %bsW.exit14.i.i ]		; <i32> [#uses=3]
	%tmp.i.i = add i32 %indvar.i.i67, %gs.24.i.i		; <i32> [#uses=1]
	%scevgep.i.i68 = getelementptr i16* %1012, i32 %tmp.i.i		; <i16*> [#uses=1]
	%1023 = load i16* %scevgep.i.i68, align 2		; <i16> [#uses=1]
	%1024 = zext i16 %1023 to i32		; <i32> [#uses=2]
	%1025 = getelementptr [6 x [258 x i32]]* @code, i32 0, i32 %1022, i32 %1024		; <i32*> [#uses=1]
	%1026 = load i32* %1025, align 4		; <i32> [#uses=1]
	%1027 = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 %1022, i32 %1024		; <i8*> [#uses=1]
	%1028 = load i8* %1027, align 1		; <i8> [#uses=1]
	%1029 = zext i8 %1028 to i32		; <i32> [#uses=2]
	br label %bb1.i12.i.i

bb.i11.i.i:		; preds = %bb1.i12.i.i
	%1030 = lshr i32 %1041, 24		; <i32> [#uses=1]
	%1031 = trunc i32 %1030 to i8		; <i8> [#uses=1]
	%1032 = load i8** %1014, align 4		; <i8*> [#uses=1]
	%1033 = load i32* %1015, align 8		; <i32> [#uses=2]
	%1034 = getelementptr i8* %1032, i32 %1033		; <i8*> [#uses=1]
	store i8 %1031, i8* %1034, align 1
	%1035 = add i32 %1033, 1		; <i32> [#uses=1]
	store i32 %1035, i32* %1015, align 8
	%1036 = load i32* %1016, align 4		; <i32> [#uses=1]
	%1037 = add i32 %1036, 1		; <i32> [#uses=1]
	store i32 %1037, i32* %1016, align 4
	%1038 = shl i32 %bsBuff.tmp.0336, 8		; <i32> [#uses=2]
	%1039 = add i32 %bsLive.tmp.0337, -8		; <i32> [#uses=3]
	br label %bb1.i12.i.i

bb1.i12.i.i:		; preds = %bb.i11.i.i, %bb164.i.i
	%bsLive.tmp.0337 = phi i32 [ %bsLive.tmp.0341, %bb164.i.i ], [ %1039, %bb.i11.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0336 = phi i32 [ %bsBuff.tmp.0340, %bb164.i.i ], [ %1038, %bb.i11.i.i ]		; <i32> [#uses=2]
	%1040 = phi i32 [ %1039, %bb.i11.i.i ], [ %bsLive.tmp.0341, %bb164.i.i ]		; <i32> [#uses=2]
	%1041 = phi i32 [ %1038, %bb.i11.i.i ], [ %bsBuff.tmp.0340, %bb164.i.i ]		; <i32> [#uses=1]
	%1042 = phi i32 [ %1039, %bb.i11.i.i ], [ %bsLive.tmp.0341, %bb164.i.i ]		; <i32> [#uses=1]
	%1043 = icmp sgt i32 %1042, 7		; <i1> [#uses=1]
	br i1 %1043, label %bb.i11.i.i, label %bsW.exit14.i.i

bsW.exit14.i.i:		; preds = %bb1.i12.i.i
	%1044 = sub i32 32, %1029		; <i32> [#uses=1]
	%1045 = sub i32 %1044, %1040		; <i32> [#uses=1]
	%1046 = shl i32 %1026, %1045		; <i32> [#uses=1]
	%1047 = or i32 %1046, %bsBuff.tmp.0336		; <i32> [#uses=2]
	%1048 = add i32 %1040, %1029		; <i32> [#uses=2]
	%tmp122.i.i = add i32 %indvar.i.i67, %tmp121.i.i		; <i32> [#uses=1]
	%1049 = icmp sgt i32 %tmp122.i.i, %.285.i.i		; <i1> [#uses=1]
	%indvar.next.i.i69 = add i32 %indvar.i.i67, 1		; <i32> [#uses=1]
	br i1 %1049, label %bb166.i.i, label %bb164.i.i

bb166.i.i:		; preds = %bsW.exit14.i.i, %bb161.i.i
	%bsLive.tmp.0346 = phi i32 [ %bsLive.tmp.1347, %bb161.i.i ], [ %1048, %bsW.exit14.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0344 = phi i32 [ %bsBuff.tmp.1345, %bb161.i.i ], [ %1047, %bsW.exit14.i.i ]		; <i32> [#uses=2]
	%1050 = add i32 %.285.i.i, 1		; <i32> [#uses=2]
	%1051 = icmp slt i32 %1050, %1010		; <i1> [#uses=1]
	br i1 %1051, label %bb161.i.i, label %bb167.i.i.loopexit

bb167.i.i.loopexit:		; preds = %bb166.i.i
	store i32 %bsBuff.tmp.0344, i32* @bsBuff
	store i32 %bsLive.tmp.0346, i32* @bsLive
	br label %bb167.i.i

bb167.i.i:		; preds = %bb167.i.i.loopexit, %bb157.i.i
	%selCtr.0.lcssa.i.i = phi i32 [ 0, %bb157.i.i ], [ %tmp124.i.i, %bb167.i.i.loopexit ]		; <i32> [#uses=1]
	%1052 = icmp eq i32 %selCtr.0.lcssa.i.i, %nSelectors.1.lcssa.i.i		; <i1> [#uses=1]
	br i1 %1052, label %sendMTFValues.exit.i, label %bb168.i.i

bb168.i.i:		; preds = %bb167.i.i
	call fastcc void @panic(i8* getelementptr ([17 x i8]* @"\01LC46", i32 0, i32 0)) nounwind ssp
	unreachable

bb4.1.i.i:		; preds = %bb4.1.i.i, %bb4.i.i19
	%v.0107.1.i.i = phi i32 [ %1053, %bb4.1.i.i ], [ 0, %bb4.i.i19 ]		; <i32> [#uses=2]
	%scevgep268.1.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 1, i32 %v.0107.1.i.i		; <i8*> [#uses=1]
	store i8 15, i8* %scevgep268.1.i.i, align 1
	%1053 = add i32 %v.0107.1.i.i, 1		; <i32> [#uses=2]
	%exitcond267.1.i.i = icmp eq i32 %1053, %509		; <i1> [#uses=1]
	br i1 %exitcond267.1.i.i, label %bb4.2.i.i, label %bb4.1.i.i

bb4.2.i.i:		; preds = %bb4.2.i.i, %bb4.1.i.i
	%v.0107.2.i.i = phi i32 [ %1054, %bb4.2.i.i ], [ 0, %bb4.1.i.i ]		; <i32> [#uses=2]
	%scevgep268.2.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 2, i32 %v.0107.2.i.i		; <i8*> [#uses=1]
	store i8 15, i8* %scevgep268.2.i.i, align 1
	%1054 = add i32 %v.0107.2.i.i, 1		; <i32> [#uses=2]
	%exitcond267.2.i.i = icmp eq i32 %1054, %509		; <i1> [#uses=1]
	br i1 %exitcond267.2.i.i, label %bb4.3.i.i, label %bb4.2.i.i

bb4.3.i.i:		; preds = %bb4.3.i.i, %bb4.2.i.i
	%v.0107.3.i.i = phi i32 [ %1055, %bb4.3.i.i ], [ 0, %bb4.2.i.i ]		; <i32> [#uses=2]
	%scevgep268.3.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 3, i32 %v.0107.3.i.i		; <i8*> [#uses=1]
	store i8 15, i8* %scevgep268.3.i.i, align 1
	%1055 = add i32 %v.0107.3.i.i, 1		; <i32> [#uses=2]
	%exitcond267.3.i.i = icmp eq i32 %1055, %509		; <i1> [#uses=1]
	br i1 %exitcond267.3.i.i, label %bb4.4.i.i, label %bb4.3.i.i

bb4.4.i.i:		; preds = %bb4.4.i.i, %bb4.3.i.i
	%v.0107.4.i.i = phi i32 [ %1056, %bb4.4.i.i ], [ 0, %bb4.3.i.i ]		; <i32> [#uses=2]
	%scevgep268.4.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 4, i32 %v.0107.4.i.i		; <i8*> [#uses=1]
	store i8 15, i8* %scevgep268.4.i.i, align 1
	%1056 = add i32 %v.0107.4.i.i, 1		; <i32> [#uses=2]
	%exitcond267.4.i.i = icmp eq i32 %1056, %509		; <i1> [#uses=1]
	br i1 %exitcond267.4.i.i, label %bb4.5.i.i, label %bb4.4.i.i

bb4.5.i.i:		; preds = %bb4.5.i.i, %bb4.4.i.i
	%v.0107.5.i.i = phi i32 [ %1057, %bb4.5.i.i ], [ 0, %bb4.4.i.i ]		; <i32> [#uses=2]
	%scevgep268.5.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 5, i32 %v.0107.5.i.i		; <i8*> [#uses=1]
	store i8 15, i8* %scevgep268.5.i.i, align 1
	%1057 = add i32 %v.0107.5.i.i, 1		; <i32> [#uses=2]
	%exitcond267.5.i.i = icmp eq i32 %1057, %509		; <i1> [#uses=1]
	br i1 %exitcond267.5.i.i, label %bb8.i.i20, label %bb4.5.i.i

sendMTFValues.exit.i:		; preds = %bb167.i.i
	store i32 -1, i32* @globalCrc, align 4
	tail call fastcc void @loadAndRLEsource() nounwind ssp
	%1058 = load i32* @last, align 4		; <i32> [#uses=2]
	%1059 = icmp eq i32 %1058, -1		; <i1> [#uses=1]
	br i1 %1059, label %bb8.i70, label %bb2.i11

bb8.i70:		; preds = %sendMTFValues.exit.i, %bsW.exit193.i
	%combinedCRC.0.lcssa.i = phi i32 [ 0, %bsW.exit193.i ], [ %192, %sendMTFValues.exit.i ]		; <i32> [#uses=4]
	%.pr.i60.i = load i32* @bsLive		; <i32> [#uses=3]
	%.pre.i61.i = load i32* @bsBuff, align 4		; <i32> [#uses=2]
	%.b.i62.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%1060 = zext i1 %.b.i62.i to i32		; <i32> [#uses=3]
	%1061 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1060, i32 3		; <i8**> [#uses=1]
	%1062 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1060, i32 2		; <i32*> [#uses=2]
	%1063 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1060, i32 1		; <i32*> [#uses=2]
	br label %bb1.i64.i71

bb.i63.i:		; preds = %bb1.i64.i71
	%1064 = lshr i32 %1075, 24		; <i32> [#uses=1]
	%1065 = trunc i32 %1064 to i8		; <i8> [#uses=1]
	%1066 = load i8** %1061, align 4		; <i8*> [#uses=1]
	%1067 = load i32* %1062, align 8		; <i32> [#uses=2]
	%1068 = getelementptr i8* %1066, i32 %1067		; <i8*> [#uses=1]
	store i8 %1065, i8* %1068, align 1
	%1069 = add i32 %1067, 1		; <i32> [#uses=1]
	store i32 %1069, i32* %1062, align 8
	%1070 = load i32* %1063, align 4		; <i32> [#uses=1]
	%1071 = add i32 %1070, 1		; <i32> [#uses=1]
	store i32 %1071, i32* %1063, align 4
	%1072 = shl i32 %bsBuff.tmp.0512, 8		; <i32> [#uses=2]
	%1073 = add i32 %bsLive.tmp.0513, -8		; <i32> [#uses=3]
	br label %bb1.i64.i71

bb1.i64.i71:		; preds = %bb.i63.i, %bb8.i70
	%bsLive.tmp.0513 = phi i32 [ %.pr.i60.i, %bb8.i70 ], [ %1073, %bb.i63.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0512 = phi i32 [ %.pre.i61.i, %bb8.i70 ], [ %1072, %bb.i63.i ]		; <i32> [#uses=2]
	%1074 = phi i32 [ %1073, %bb.i63.i ], [ %.pr.i60.i, %bb8.i70 ]		; <i32> [#uses=2]
	%1075 = phi i32 [ %1072, %bb.i63.i ], [ %.pre.i61.i, %bb8.i70 ]		; <i32> [#uses=1]
	%1076 = phi i32 [ %1073, %bb.i63.i ], [ %.pr.i60.i, %bb8.i70 ]		; <i32> [#uses=1]
	%1077 = icmp sgt i32 %1076, 7		; <i1> [#uses=1]
	br i1 %1077, label %bb.i63.i, label %bsW.exit65.i

bsW.exit65.i:		; preds = %bb1.i64.i71
	%1078 = sub i32 24, %1074		; <i32> [#uses=1]
	%1079 = shl i32 23, %1078		; <i32> [#uses=1]
	%1080 = or i32 %1079, %bsBuff.tmp.0512		; <i32> [#uses=3]
	store i32 %1080, i32* @bsBuff, align 4
	%1081 = add i32 %1074, 8		; <i32> [#uses=4]
	store i32 %1081, i32* @bsLive, align 4
	%.b.i56.i72 = load i1* @bsStream.b		; <i1> [#uses=1]
	%1082 = zext i1 %.b.i56.i72 to i32		; <i32> [#uses=3]
	%1083 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1082, i32 3		; <i8**> [#uses=1]
	%1084 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1082, i32 2		; <i32*> [#uses=2]
	%1085 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1082, i32 1		; <i32*> [#uses=2]
	br label %bb1.i58.i

bb.i57.i:		; preds = %bb1.i58.i
	%1086 = lshr i32 %1097, 24		; <i32> [#uses=1]
	%1087 = trunc i32 %1086 to i8		; <i8> [#uses=1]
	%1088 = load i8** %1083, align 4		; <i8*> [#uses=1]
	%1089 = load i32* %1084, align 8		; <i32> [#uses=2]
	%1090 = getelementptr i8* %1088, i32 %1089		; <i8*> [#uses=1]
	store i8 %1087, i8* %1090, align 1
	%1091 = add i32 %1089, 1		; <i32> [#uses=1]
	store i32 %1091, i32* %1084, align 8
	%1092 = load i32* %1085, align 4		; <i32> [#uses=1]
	%1093 = add i32 %1092, 1		; <i32> [#uses=1]
	store i32 %1093, i32* %1085, align 4
	%1094 = shl i32 %bsBuff.tmp.0516, 8		; <i32> [#uses=2]
	%1095 = add i32 %bsLive.tmp.0517, -8		; <i32> [#uses=3]
	br label %bb1.i58.i

bb1.i58.i:		; preds = %bb.i57.i, %bsW.exit65.i
	%bsLive.tmp.0517 = phi i32 [ %1081, %bsW.exit65.i ], [ %1095, %bb.i57.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0516 = phi i32 [ %1080, %bsW.exit65.i ], [ %1094, %bb.i57.i ]		; <i32> [#uses=2]
	%1096 = phi i32 [ %1095, %bb.i57.i ], [ %1081, %bsW.exit65.i ]		; <i32> [#uses=2]
	%1097 = phi i32 [ %1094, %bb.i57.i ], [ %1080, %bsW.exit65.i ]		; <i32> [#uses=1]
	%1098 = phi i32 [ %1095, %bb.i57.i ], [ %1081, %bsW.exit65.i ]		; <i32> [#uses=1]
	%1099 = icmp sgt i32 %1098, 7		; <i1> [#uses=1]
	br i1 %1099, label %bb.i57.i, label %bsW.exit59.i

bsW.exit59.i:		; preds = %bb1.i58.i
	%1100 = sub i32 24, %1096		; <i32> [#uses=1]
	%1101 = shl i32 114, %1100		; <i32> [#uses=1]
	%1102 = or i32 %1101, %bsBuff.tmp.0516		; <i32> [#uses=3]
	store i32 %1102, i32* @bsBuff, align 4
	%1103 = add i32 %1096, 8		; <i32> [#uses=4]
	store i32 %1103, i32* @bsLive, align 4
	%.b.i50.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%1104 = zext i1 %.b.i50.i to i32		; <i32> [#uses=3]
	%1105 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1104, i32 3		; <i8**> [#uses=1]
	%1106 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1104, i32 2		; <i32*> [#uses=2]
	%1107 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1104, i32 1		; <i32*> [#uses=2]
	br label %bb1.i52.i

bb.i51.i:		; preds = %bb1.i52.i
	%1108 = lshr i32 %1119, 24		; <i32> [#uses=1]
	%1109 = trunc i32 %1108 to i8		; <i8> [#uses=1]
	%1110 = load i8** %1105, align 4		; <i8*> [#uses=1]
	%1111 = load i32* %1106, align 8		; <i32> [#uses=2]
	%1112 = getelementptr i8* %1110, i32 %1111		; <i8*> [#uses=1]
	store i8 %1109, i8* %1112, align 1
	%1113 = add i32 %1111, 1		; <i32> [#uses=1]
	store i32 %1113, i32* %1106, align 8
	%1114 = load i32* %1107, align 4		; <i32> [#uses=1]
	%1115 = add i32 %1114, 1		; <i32> [#uses=1]
	store i32 %1115, i32* %1107, align 4
	%1116 = shl i32 %bsBuff.tmp.0520, 8		; <i32> [#uses=2]
	%1117 = add i32 %bsLive.tmp.0521, -8		; <i32> [#uses=3]
	br label %bb1.i52.i

bb1.i52.i:		; preds = %bb.i51.i, %bsW.exit59.i
	%bsLive.tmp.0521 = phi i32 [ %1103, %bsW.exit59.i ], [ %1117, %bb.i51.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0520 = phi i32 [ %1102, %bsW.exit59.i ], [ %1116, %bb.i51.i ]		; <i32> [#uses=2]
	%1118 = phi i32 [ %1117, %bb.i51.i ], [ %1103, %bsW.exit59.i ]		; <i32> [#uses=2]
	%1119 = phi i32 [ %1116, %bb.i51.i ], [ %1102, %bsW.exit59.i ]		; <i32> [#uses=1]
	%1120 = phi i32 [ %1117, %bb.i51.i ], [ %1103, %bsW.exit59.i ]		; <i32> [#uses=1]
	%1121 = icmp sgt i32 %1120, 7		; <i1> [#uses=1]
	br i1 %1121, label %bb.i51.i, label %bsW.exit53.i

bsW.exit53.i:		; preds = %bb1.i52.i
	%1122 = sub i32 24, %1118		; <i32> [#uses=1]
	%1123 = shl i32 69, %1122		; <i32> [#uses=1]
	%1124 = or i32 %1123, %bsBuff.tmp.0520		; <i32> [#uses=3]
	store i32 %1124, i32* @bsBuff, align 4
	%1125 = add i32 %1118, 8		; <i32> [#uses=4]
	store i32 %1125, i32* @bsLive, align 4
	%.b.i44.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%1126 = zext i1 %.b.i44.i to i32		; <i32> [#uses=3]
	%1127 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1126, i32 3		; <i8**> [#uses=1]
	%1128 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1126, i32 2		; <i32*> [#uses=2]
	%1129 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1126, i32 1		; <i32*> [#uses=2]
	br label %bb1.i46.i

bb.i45.i:		; preds = %bb1.i46.i
	%1130 = lshr i32 %1141, 24		; <i32> [#uses=1]
	%1131 = trunc i32 %1130 to i8		; <i8> [#uses=1]
	%1132 = load i8** %1127, align 4		; <i8*> [#uses=1]
	%1133 = load i32* %1128, align 8		; <i32> [#uses=2]
	%1134 = getelementptr i8* %1132, i32 %1133		; <i8*> [#uses=1]
	store i8 %1131, i8* %1134, align 1
	%1135 = add i32 %1133, 1		; <i32> [#uses=1]
	store i32 %1135, i32* %1128, align 8
	%1136 = load i32* %1129, align 4		; <i32> [#uses=1]
	%1137 = add i32 %1136, 1		; <i32> [#uses=1]
	store i32 %1137, i32* %1129, align 4
	%1138 = shl i32 %bsBuff.tmp.0524, 8		; <i32> [#uses=2]
	%1139 = add i32 %bsLive.tmp.0525, -8		; <i32> [#uses=3]
	br label %bb1.i46.i

bb1.i46.i:		; preds = %bb.i45.i, %bsW.exit53.i
	%bsLive.tmp.0525 = phi i32 [ %1125, %bsW.exit53.i ], [ %1139, %bb.i45.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0524 = phi i32 [ %1124, %bsW.exit53.i ], [ %1138, %bb.i45.i ]		; <i32> [#uses=2]
	%1140 = phi i32 [ %1139, %bb.i45.i ], [ %1125, %bsW.exit53.i ]		; <i32> [#uses=2]
	%1141 = phi i32 [ %1138, %bb.i45.i ], [ %1124, %bsW.exit53.i ]		; <i32> [#uses=1]
	%1142 = phi i32 [ %1139, %bb.i45.i ], [ %1125, %bsW.exit53.i ]		; <i32> [#uses=1]
	%1143 = icmp sgt i32 %1142, 7		; <i1> [#uses=1]
	br i1 %1143, label %bb.i45.i, label %bsW.exit47.i

bsW.exit47.i:		; preds = %bb1.i46.i
	%1144 = sub i32 24, %1140		; <i32> [#uses=1]
	%1145 = shl i32 56, %1144		; <i32> [#uses=1]
	%1146 = or i32 %1145, %bsBuff.tmp.0524		; <i32> [#uses=3]
	store i32 %1146, i32* @bsBuff, align 4
	%1147 = add i32 %1140, 8		; <i32> [#uses=4]
	store i32 %1147, i32* @bsLive, align 4
	%.b.i38.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%1148 = zext i1 %.b.i38.i to i32		; <i32> [#uses=3]
	%1149 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1148, i32 3		; <i8**> [#uses=1]
	%1150 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1148, i32 2		; <i32*> [#uses=2]
	%1151 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1148, i32 1		; <i32*> [#uses=2]
	br label %bb1.i40.i

bb.i39.i:		; preds = %bb1.i40.i
	%1152 = lshr i32 %1163, 24		; <i32> [#uses=1]
	%1153 = trunc i32 %1152 to i8		; <i8> [#uses=1]
	%1154 = load i8** %1149, align 4		; <i8*> [#uses=1]
	%1155 = load i32* %1150, align 8		; <i32> [#uses=2]
	%1156 = getelementptr i8* %1154, i32 %1155		; <i8*> [#uses=1]
	store i8 %1153, i8* %1156, align 1
	%1157 = add i32 %1155, 1		; <i32> [#uses=1]
	store i32 %1157, i32* %1150, align 8
	%1158 = load i32* %1151, align 4		; <i32> [#uses=1]
	%1159 = add i32 %1158, 1		; <i32> [#uses=1]
	store i32 %1159, i32* %1151, align 4
	%1160 = shl i32 %bsBuff.tmp.0528, 8		; <i32> [#uses=2]
	%1161 = add i32 %bsLive.tmp.0529, -8		; <i32> [#uses=3]
	br label %bb1.i40.i

bb1.i40.i:		; preds = %bb.i39.i, %bsW.exit47.i
	%bsLive.tmp.0529 = phi i32 [ %1147, %bsW.exit47.i ], [ %1161, %bb.i39.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0528 = phi i32 [ %1146, %bsW.exit47.i ], [ %1160, %bb.i39.i ]		; <i32> [#uses=2]
	%1162 = phi i32 [ %1161, %bb.i39.i ], [ %1147, %bsW.exit47.i ]		; <i32> [#uses=2]
	%1163 = phi i32 [ %1160, %bb.i39.i ], [ %1146, %bsW.exit47.i ]		; <i32> [#uses=1]
	%1164 = phi i32 [ %1161, %bb.i39.i ], [ %1147, %bsW.exit47.i ]		; <i32> [#uses=1]
	%1165 = icmp sgt i32 %1164, 7		; <i1> [#uses=1]
	br i1 %1165, label %bb.i39.i, label %bsW.exit41.i

bsW.exit41.i:		; preds = %bb1.i40.i
	%1166 = sub i32 24, %1162		; <i32> [#uses=1]
	%1167 = shl i32 80, %1166		; <i32> [#uses=1]
	%1168 = or i32 %1167, %bsBuff.tmp.0528		; <i32> [#uses=3]
	store i32 %1168, i32* @bsBuff, align 4
	%1169 = add i32 %1162, 8		; <i32> [#uses=4]
	store i32 %1169, i32* @bsLive, align 4
	%.b.i32.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%1170 = zext i1 %.b.i32.i to i32		; <i32> [#uses=3]
	%1171 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1170, i32 3		; <i8**> [#uses=1]
	%1172 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1170, i32 2		; <i32*> [#uses=2]
	%1173 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1170, i32 1		; <i32*> [#uses=2]
	br label %bb1.i34.i

bb.i33.i:		; preds = %bb1.i34.i
	%1174 = lshr i32 %1185, 24		; <i32> [#uses=1]
	%1175 = trunc i32 %1174 to i8		; <i8> [#uses=1]
	%1176 = load i8** %1171, align 4		; <i8*> [#uses=1]
	%1177 = load i32* %1172, align 8		; <i32> [#uses=2]
	%1178 = getelementptr i8* %1176, i32 %1177		; <i8*> [#uses=1]
	store i8 %1175, i8* %1178, align 1
	%1179 = add i32 %1177, 1		; <i32> [#uses=1]
	store i32 %1179, i32* %1172, align 8
	%1180 = load i32* %1173, align 4		; <i32> [#uses=1]
	%1181 = add i32 %1180, 1		; <i32> [#uses=1]
	store i32 %1181, i32* %1173, align 4
	%1182 = shl i32 %bsBuff.tmp.0532, 8		; <i32> [#uses=2]
	%1183 = add i32 %bsLive.tmp.0533, -8		; <i32> [#uses=3]
	br label %bb1.i34.i

bb1.i34.i:		; preds = %bb.i33.i, %bsW.exit41.i
	%bsLive.tmp.0533 = phi i32 [ %1169, %bsW.exit41.i ], [ %1183, %bb.i33.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0532 = phi i32 [ %1168, %bsW.exit41.i ], [ %1182, %bb.i33.i ]		; <i32> [#uses=2]
	%1184 = phi i32 [ %1183, %bb.i33.i ], [ %1169, %bsW.exit41.i ]		; <i32> [#uses=2]
	%1185 = phi i32 [ %1182, %bb.i33.i ], [ %1168, %bsW.exit41.i ]		; <i32> [#uses=1]
	%1186 = phi i32 [ %1183, %bb.i33.i ], [ %1169, %bsW.exit41.i ]		; <i32> [#uses=1]
	%1187 = icmp sgt i32 %1186, 7		; <i1> [#uses=1]
	br i1 %1187, label %bb.i33.i, label %bsW.exit35.i

bsW.exit35.i:		; preds = %bb1.i34.i
	%1188 = sub i32 24, %1184		; <i32> [#uses=1]
	%1189 = shl i32 144, %1188		; <i32> [#uses=1]
	%1190 = or i32 %1189, %bsBuff.tmp.0532		; <i32> [#uses=3]
	store i32 %1190, i32* @bsBuff, align 4
	%1191 = add i32 %1184, 8		; <i32> [#uses=4]
	store i32 %1191, i32* @bsLive, align 4
	%1192 = lshr i32 %combinedCRC.0.lcssa.i, 24		; <i32> [#uses=1]
	%.b.i26.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%1193 = zext i1 %.b.i26.i to i32		; <i32> [#uses=3]
	%1194 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1193, i32 3		; <i8**> [#uses=1]
	%1195 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1193, i32 2		; <i32*> [#uses=2]
	%1196 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1193, i32 1		; <i32*> [#uses=2]
	br label %bb1.i28.i

bb.i27.i:		; preds = %bb1.i28.i
	%1197 = lshr i32 %1208, 24		; <i32> [#uses=1]
	%1198 = trunc i32 %1197 to i8		; <i8> [#uses=1]
	%1199 = load i8** %1194, align 4		; <i8*> [#uses=1]
	%1200 = load i32* %1195, align 8		; <i32> [#uses=2]
	%1201 = getelementptr i8* %1199, i32 %1200		; <i8*> [#uses=1]
	store i8 %1198, i8* %1201, align 1
	%1202 = add i32 %1200, 1		; <i32> [#uses=1]
	store i32 %1202, i32* %1195, align 8
	%1203 = load i32* %1196, align 4		; <i32> [#uses=1]
	%1204 = add i32 %1203, 1		; <i32> [#uses=1]
	store i32 %1204, i32* %1196, align 4
	%1205 = shl i32 %bsBuff.tmp.0536, 8		; <i32> [#uses=2]
	%1206 = add i32 %bsLive.tmp.0537, -8		; <i32> [#uses=3]
	br label %bb1.i28.i

bb1.i28.i:		; preds = %bb.i27.i, %bsW.exit35.i
	%bsLive.tmp.0537 = phi i32 [ %1191, %bsW.exit35.i ], [ %1206, %bb.i27.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0536 = phi i32 [ %1190, %bsW.exit35.i ], [ %1205, %bb.i27.i ]		; <i32> [#uses=2]
	%1207 = phi i32 [ %1206, %bb.i27.i ], [ %1191, %bsW.exit35.i ]		; <i32> [#uses=2]
	%1208 = phi i32 [ %1205, %bb.i27.i ], [ %1190, %bsW.exit35.i ]		; <i32> [#uses=1]
	%1209 = phi i32 [ %1206, %bb.i27.i ], [ %1191, %bsW.exit35.i ]		; <i32> [#uses=1]
	%1210 = icmp sgt i32 %1209, 7		; <i1> [#uses=1]
	br i1 %1210, label %bb.i27.i, label %bsW.exit29.i

bsW.exit29.i:		; preds = %bb1.i28.i
	%1211 = sub i32 24, %1207		; <i32> [#uses=1]
	%1212 = shl i32 %1192, %1211		; <i32> [#uses=1]
	%1213 = or i32 %1212, %bsBuff.tmp.0536		; <i32> [#uses=3]
	store i32 %1213, i32* @bsBuff, align 4
	%1214 = add i32 %1207, 8		; <i32> [#uses=4]
	store i32 %1214, i32* @bsLive, align 4
	%1215 = lshr i32 %combinedCRC.0.lcssa.i, 16		; <i32> [#uses=1]
	%1216 = and i32 %1215, 255		; <i32> [#uses=1]
	%.b.i20.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%1217 = zext i1 %.b.i20.i to i32		; <i32> [#uses=3]
	%1218 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1217, i32 3		; <i8**> [#uses=1]
	%1219 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1217, i32 2		; <i32*> [#uses=2]
	%1220 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1217, i32 1		; <i32*> [#uses=2]
	br label %bb1.i22.i

bb.i21.i:		; preds = %bb1.i22.i
	%1221 = lshr i32 %1232, 24		; <i32> [#uses=1]
	%1222 = trunc i32 %1221 to i8		; <i8> [#uses=1]
	%1223 = load i8** %1218, align 4		; <i8*> [#uses=1]
	%1224 = load i32* %1219, align 8		; <i32> [#uses=2]
	%1225 = getelementptr i8* %1223, i32 %1224		; <i8*> [#uses=1]
	store i8 %1222, i8* %1225, align 1
	%1226 = add i32 %1224, 1		; <i32> [#uses=1]
	store i32 %1226, i32* %1219, align 8
	%1227 = load i32* %1220, align 4		; <i32> [#uses=1]
	%1228 = add i32 %1227, 1		; <i32> [#uses=1]
	store i32 %1228, i32* %1220, align 4
	%1229 = shl i32 %bsBuff.tmp.0540, 8		; <i32> [#uses=2]
	%1230 = add i32 %bsLive.tmp.0541, -8		; <i32> [#uses=3]
	br label %bb1.i22.i

bb1.i22.i:		; preds = %bb.i21.i, %bsW.exit29.i
	%bsLive.tmp.0541 = phi i32 [ %1214, %bsW.exit29.i ], [ %1230, %bb.i21.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0540 = phi i32 [ %1213, %bsW.exit29.i ], [ %1229, %bb.i21.i ]		; <i32> [#uses=2]
	%1231 = phi i32 [ %1230, %bb.i21.i ], [ %1214, %bsW.exit29.i ]		; <i32> [#uses=2]
	%1232 = phi i32 [ %1229, %bb.i21.i ], [ %1213, %bsW.exit29.i ]		; <i32> [#uses=1]
	%1233 = phi i32 [ %1230, %bb.i21.i ], [ %1214, %bsW.exit29.i ]		; <i32> [#uses=1]
	%1234 = icmp sgt i32 %1233, 7		; <i1> [#uses=1]
	br i1 %1234, label %bb.i21.i, label %bsW.exit23.i

bsW.exit23.i:		; preds = %bb1.i22.i
	%1235 = sub i32 24, %1231		; <i32> [#uses=1]
	%1236 = shl i32 %1216, %1235		; <i32> [#uses=1]
	%1237 = or i32 %1236, %bsBuff.tmp.0540		; <i32> [#uses=3]
	store i32 %1237, i32* @bsBuff, align 4
	%1238 = add i32 %1231, 8		; <i32> [#uses=4]
	store i32 %1238, i32* @bsLive, align 4
	%1239 = lshr i32 %combinedCRC.0.lcssa.i, 8		; <i32> [#uses=1]
	%1240 = and i32 %1239, 255		; <i32> [#uses=1]
	%.b.i14.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%1241 = zext i1 %.b.i14.i to i32		; <i32> [#uses=3]
	%1242 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1241, i32 3		; <i8**> [#uses=1]
	%1243 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1241, i32 2		; <i32*> [#uses=2]
	%1244 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1241, i32 1		; <i32*> [#uses=2]
	br label %bb1.i16.i

bb.i15.i:		; preds = %bb1.i16.i
	%1245 = lshr i32 %1256, 24		; <i32> [#uses=1]
	%1246 = trunc i32 %1245 to i8		; <i8> [#uses=1]
	%1247 = load i8** %1242, align 4		; <i8*> [#uses=1]
	%1248 = load i32* %1243, align 8		; <i32> [#uses=2]
	%1249 = getelementptr i8* %1247, i32 %1248		; <i8*> [#uses=1]
	store i8 %1246, i8* %1249, align 1
	%1250 = add i32 %1248, 1		; <i32> [#uses=1]
	store i32 %1250, i32* %1243, align 8
	%1251 = load i32* %1244, align 4		; <i32> [#uses=1]
	%1252 = add i32 %1251, 1		; <i32> [#uses=1]
	store i32 %1252, i32* %1244, align 4
	%1253 = shl i32 %bsBuff.tmp.0544, 8		; <i32> [#uses=2]
	%1254 = add i32 %bsLive.tmp.0545, -8		; <i32> [#uses=3]
	br label %bb1.i16.i

bb1.i16.i:		; preds = %bb.i15.i, %bsW.exit23.i
	%bsLive.tmp.0545 = phi i32 [ %1238, %bsW.exit23.i ], [ %1254, %bb.i15.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0544 = phi i32 [ %1237, %bsW.exit23.i ], [ %1253, %bb.i15.i ]		; <i32> [#uses=2]
	%1255 = phi i32 [ %1254, %bb.i15.i ], [ %1238, %bsW.exit23.i ]		; <i32> [#uses=2]
	%1256 = phi i32 [ %1253, %bb.i15.i ], [ %1237, %bsW.exit23.i ]		; <i32> [#uses=1]
	%1257 = phi i32 [ %1254, %bb.i15.i ], [ %1238, %bsW.exit23.i ]		; <i32> [#uses=1]
	%1258 = icmp sgt i32 %1257, 7		; <i1> [#uses=1]
	br i1 %1258, label %bb.i15.i, label %bsW.exit17.i

bsW.exit17.i:		; preds = %bb1.i16.i
	%1259 = sub i32 24, %1255		; <i32> [#uses=1]
	%1260 = shl i32 %1240, %1259		; <i32> [#uses=1]
	%1261 = or i32 %1260, %bsBuff.tmp.0544		; <i32> [#uses=3]
	store i32 %1261, i32* @bsBuff, align 4
	%1262 = add i32 %1255, 8		; <i32> [#uses=4]
	store i32 %1262, i32* @bsLive, align 4
	%1263 = and i32 %combinedCRC.0.lcssa.i, 255		; <i32> [#uses=1]
	%.b.i8.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%1264 = zext i1 %.b.i8.i to i32		; <i32> [#uses=3]
	%1265 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1264, i32 3		; <i8**> [#uses=1]
	%1266 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1264, i32 2		; <i32*> [#uses=2]
	%1267 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1264, i32 1		; <i32*> [#uses=2]
	br label %bb1.i10.i

bb.i9.i:		; preds = %bb1.i10.i
	%1268 = lshr i32 %1279, 24		; <i32> [#uses=1]
	%1269 = trunc i32 %1268 to i8		; <i8> [#uses=1]
	%1270 = load i8** %1265, align 4		; <i8*> [#uses=1]
	%1271 = load i32* %1266, align 8		; <i32> [#uses=2]
	%1272 = getelementptr i8* %1270, i32 %1271		; <i8*> [#uses=1]
	store i8 %1269, i8* %1272, align 1
	%1273 = add i32 %1271, 1		; <i32> [#uses=1]
	store i32 %1273, i32* %1266, align 8
	%1274 = load i32* %1267, align 4		; <i32> [#uses=1]
	%1275 = add i32 %1274, 1		; <i32> [#uses=1]
	store i32 %1275, i32* %1267, align 4
	%1276 = shl i32 %bsBuff.tmp.0548, 8		; <i32> [#uses=2]
	%1277 = add i32 %bsLive.tmp.0549, -8		; <i32> [#uses=3]
	br label %bb1.i10.i

bb1.i10.i:		; preds = %bb.i9.i, %bsW.exit17.i
	%bsLive.tmp.0549 = phi i32 [ %1262, %bsW.exit17.i ], [ %1277, %bb.i9.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0548 = phi i32 [ %1261, %bsW.exit17.i ], [ %1276, %bb.i9.i ]		; <i32> [#uses=2]
	%1278 = phi i32 [ %1277, %bb.i9.i ], [ %1262, %bsW.exit17.i ]		; <i32> [#uses=2]
	%1279 = phi i32 [ %1276, %bb.i9.i ], [ %1261, %bsW.exit17.i ]		; <i32> [#uses=1]
	%1280 = phi i32 [ %1277, %bb.i9.i ], [ %1262, %bsW.exit17.i ]		; <i32> [#uses=1]
	%1281 = icmp sgt i32 %1280, 7		; <i1> [#uses=1]
	br i1 %1281, label %bb.i9.i, label %bb1thread-split.i.i74

bb.i2.i:		; preds = %bb1.i5.i
	%1282 = lshr i32 %1300, 24		; <i32> [#uses=1]
	%1283 = trunc i32 %1282 to i8		; <i8> [#uses=1]
	%1284 = load i8** %1297, align 4		; <i8*> [#uses=1]
	%1285 = load i32* %1298, align 8		; <i32> [#uses=2]
	%1286 = getelementptr i8* %1284, i32 %1285		; <i8*> [#uses=1]
	store i8 %1283, i8* %1286, align 1
	%1287 = add i32 %1285, 1		; <i32> [#uses=1]
	store i32 %1287, i32* %1298, align 8
	%1288 = load i32* %1299, align 4		; <i32> [#uses=1]
	%1289 = add i32 %1288, 1		; <i32> [#uses=1]
	store i32 %1289, i32* %1299, align 4
	%1290 = shl i32 %bsBuff.tmp.0332, 8		; <i32> [#uses=2]
	%1291 = add i32 %bsLive.tmp.0333, -8		; <i32> [#uses=2]
	br label %bb1.i5.i

bb1thread-split.i.i74:		; preds = %bb1.i10.i
	%1292 = sub i32 24, %1278		; <i32> [#uses=1]
	%1293 = shl i32 %1263, %1292		; <i32> [#uses=1]
	%1294 = or i32 %1293, %bsBuff.tmp.0548		; <i32> [#uses=3]
	store i32 %1294, i32* @bsBuff, align 4
	%1295 = add i32 %1278, 8		; <i32> [#uses=3]
	store i32 %1295, i32* @bsLive, align 4
	%.b.i1.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%1296 = zext i1 %.b.i1.i to i32		; <i32> [#uses=3]
	%1297 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1296, i32 3		; <i8**> [#uses=1]
	%1298 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1296, i32 2		; <i32*> [#uses=2]
	%1299 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1296, i32 1		; <i32*> [#uses=2]
	br label %bb1.i5.i

bb1.i5.i:		; preds = %bb1thread-split.i.i74, %bb.i2.i
	%bsLive.tmp.0333 = phi i32 [ %1295, %bb1thread-split.i.i74 ], [ %1291, %bb.i2.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0332 = phi i32 [ %1294, %bb1thread-split.i.i74 ], [ %1290, %bb.i2.i ]		; <i32> [#uses=2]
	%1300 = phi i32 [ %1294, %bb1thread-split.i.i74 ], [ %1290, %bb.i2.i ]		; <i32> [#uses=1]
	%1301 = phi i32 [ %1295, %bb1thread-split.i.i74 ], [ %1291, %bb.i2.i ]		; <i32> [#uses=1]
	%1302 = icmp sgt i32 %1301, 0		; <i1> [#uses=1]
	br i1 %1302, label %bb.i2.i, label %bsFinishedWithStream.exit.i75

bsFinishedWithStream.exit.i75:		; preds = %bb1.i5.i
	store i32 %bsBuff.tmp.0332, i32* @bsBuff
	store i32 %bsLive.tmp.0333, i32* @bsLive
	store i1 false, i1* @bsStream.b
	%1303 = load i32* @bytesIn, align 4		; <i32> [#uses=1]
	%1304 = icmp eq i32 %1303, 0		; <i1> [#uses=1]
	br i1 %1304, label %bb23.i, label %bb24.i

bb23.i:		; preds = %bsFinishedWithStream.exit.i75
	store i32 1, i32* @bytesIn, align 4
	br label %bb24.i

bb24.i:		; preds = %bb23.i, %bsFinishedWithStream.exit.i75
	%1305 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 1), align 4		; <i32> [#uses=1]
	%1306 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([36 x i8]* @"\01LC2492", i32 0, i32 0), i32 %1305) nounwind		; <i32> [#uses=0]
	%1307 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4		; <i32> [#uses=1]
	%1308 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 3), align 4		; <i8*> [#uses=1]
	tail call void @llvm.memset.i32(i8* %1308, i8 0, i32 %1307, i32 1) nounwind
	store i32 0, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4
	store i32 0, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 2), align 8
	store i32 0, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 2), align 8
	%1309 = tail call i32 @puts(i8* getelementptr ([19 x i8]* @"\01LC2593", i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	store i32 0, i32* @blockSize100k, align 4
	store i1 true, i1* @bsStream.b
	store i32 0, i32* @bsLive, align 4
	store i32 0, i32* @bsBuff, align 4
	store i32 0, i32* @bytesIn, align 4
	br label %bb3.i.i.i

bb3.i3.i:		; preds = %bb3.i.i.i
	%1310 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 2), align 8		; <i32> [#uses=3]
	%1311 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 1), align 4		; <i32> [#uses=1]
	%1312 = icmp slt i32 %1310, %1311		; <i1> [#uses=1]
	br i1 %1312, label %bb2.i.i.i, label %bb1.i.i.i

bb1.i.i.i:		; preds = %bb3.i3.i
	store i32 %bsBuff.tmp.0552, i32* @bsBuff
	store i32 %bsLive.tmp.0553, i32* @bsLive
	tail call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i.i:		; preds = %bb3.i3.i
	%1313 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 3), align 4		; <i8*> [#uses=1]
	%1314 = getelementptr i8* %1313, i32 %1310		; <i8*> [#uses=1]
	%1315 = load i8* %1314, align 1		; <i8> [#uses=1]
	%1316 = add i32 %1310, 1		; <i32> [#uses=1]
	store i32 %1316, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 2), align 8
	%1317 = zext i8 %1315 to i32		; <i32> [#uses=1]
	%1318 = shl i32 %bsBuff.tmp.0552, 8		; <i32> [#uses=1]
	%1319 = or i32 %1318, %1317		; <i32> [#uses=2]
	%1320 = add i32 %bsLive.tmp.0553, 8		; <i32> [#uses=2]
	%phitmp65.i = icmp slt i32 %1320, 8		; <i1> [#uses=1]
	br label %bb3.i.i.i

bb3.i.i.i:		; preds = %bb2.i.i.i, %bb24.i
	%bsLive.tmp.0553 = phi i32 [ 0, %bb24.i ], [ %1320, %bb2.i.i.i ]		; <i32> [#uses=3]
	%bsBuff.tmp.0552 = phi i32 [ 0, %bb24.i ], [ %1319, %bb2.i.i.i ]		; <i32> [#uses=4]
	%1321 = phi i32 [ %1319, %bb2.i.i.i ], [ 0, %bb24.i ]		; <i32> [#uses=2]
	%1322 = phi i32 [ %bsLive.tmp.0553, %bb2.i.i.i ], [ -8, %bb24.i ]		; <i32> [#uses=5]
	%1323 = phi i1 [ %phitmp65.i, %bb2.i.i.i ], [ true, %bb24.i ]		; <i1> [#uses=1]
	br i1 %1323, label %bb3.i3.i, label %bsGetUChar.exit.i

bsGetUChar.exit.i:		; preds = %bb3.i.i.i
	store i32 %bsBuff.tmp.0552, i32* @bsBuff
	%1324 = lshr i32 %1321, %1322		; <i32> [#uses=1]
	store i32 %1322, i32* @bsLive, align 4
	%retval12.i.i = trunc i32 %1324 to i8		; <i8> [#uses=1]
	br label %bb3.i.i5.i

bb3.i11.i:		; preds = %bb3.i.i5.i
	%1325 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 2), align 8		; <i32> [#uses=3]
	%1326 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 1), align 4		; <i32> [#uses=1]
	%1327 = icmp slt i32 %1325, %1326		; <i1> [#uses=1]
	br i1 %1327, label %bb2.i.i4.i, label %bb1.i.i3.i

bb1.i.i3.i:		; preds = %bb3.i11.i
	store i32 %bsBuff.tmp.0556, i32* @bsBuff
	store i32 %bsLive.tmp.0557, i32* @bsLive
	tail call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i4.i:		; preds = %bb3.i11.i
	%1328 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 3), align 4		; <i8*> [#uses=1]
	%1329 = getelementptr i8* %1328, i32 %1325		; <i8*> [#uses=1]
	%1330 = load i8* %1329, align 1		; <i8> [#uses=1]
	%1331 = add i32 %1325, 1		; <i32> [#uses=1]
	store i32 %1331, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 2), align 8
	%1332 = zext i8 %1330 to i32		; <i32> [#uses=1]
	%1333 = shl i32 %bsBuff.tmp.0556, 8		; <i32> [#uses=1]
	%1334 = or i32 %1333, %1332		; <i32> [#uses=2]
	%1335 = add i32 %bsLive.tmp.0557, 8		; <i32> [#uses=3]
	br label %bb3.i.i5.i

bb3.i.i5.i:		; preds = %bb2.i.i4.i, %bsGetUChar.exit.i
	%bsLive.tmp.0557 = phi i32 [ %1322, %bsGetUChar.exit.i ], [ %1335, %bb2.i.i4.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0556 = phi i32 [ %bsBuff.tmp.0552, %bsGetUChar.exit.i ], [ %1334, %bb2.i.i4.i ]		; <i32> [#uses=4]
	%1336 = phi i32 [ %1334, %bb2.i.i4.i ], [ %1321, %bsGetUChar.exit.i ]		; <i32> [#uses=2]
	%1337 = phi i32 [ %1335, %bb2.i.i4.i ], [ %1322, %bsGetUChar.exit.i ]		; <i32> [#uses=1]
	%1338 = phi i32 [ %1335, %bb2.i.i4.i ], [ %1322, %bsGetUChar.exit.i ]		; <i32> [#uses=1]
	%1339 = icmp slt i32 %1338, 8		; <i1> [#uses=1]
	br i1 %1339, label %bb3.i11.i, label %bsGetUChar.exit7.i

bsGetUChar.exit7.i:		; preds = %bb3.i.i5.i
	store i32 %bsBuff.tmp.0556, i32* @bsBuff
	%1340 = add i32 %1337, -8		; <i32> [#uses=5]
	%1341 = lshr i32 %1336, %1340		; <i32> [#uses=1]
	store i32 %1340, i32* @bsLive, align 4
	%retval12.i6.i = trunc i32 %1341 to i8		; <i8> [#uses=1]
	br label %bb3.i.i12.i

bb3.i18.i:		; preds = %bb3.i.i12.i
	%1342 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 2), align 8		; <i32> [#uses=3]
	%1343 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 1), align 4		; <i32> [#uses=1]
	%1344 = icmp slt i32 %1342, %1343		; <i1> [#uses=1]
	br i1 %1344, label %bb2.i.i11.i, label %bb1.i.i10.i

bb1.i.i10.i:		; preds = %bb3.i18.i
	store i32 %bsBuff.tmp.0560, i32* @bsBuff
	store i32 %bsLive.tmp.0561, i32* @bsLive
	tail call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i11.i:		; preds = %bb3.i18.i
	%1345 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 3), align 4		; <i8*> [#uses=1]
	%1346 = getelementptr i8* %1345, i32 %1342		; <i8*> [#uses=1]
	%1347 = load i8* %1346, align 1		; <i8> [#uses=1]
	%1348 = add i32 %1342, 1		; <i32> [#uses=1]
	store i32 %1348, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 2), align 8
	%1349 = zext i8 %1347 to i32		; <i32> [#uses=1]
	%1350 = shl i32 %bsBuff.tmp.0560, 8		; <i32> [#uses=1]
	%1351 = or i32 %1350, %1349		; <i32> [#uses=2]
	%1352 = add i32 %bsLive.tmp.0561, 8		; <i32> [#uses=3]
	br label %bb3.i.i12.i

bb3.i.i12.i:		; preds = %bb2.i.i11.i, %bsGetUChar.exit7.i
	%bsLive.tmp.0561 = phi i32 [ %1340, %bsGetUChar.exit7.i ], [ %1352, %bb2.i.i11.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0560 = phi i32 [ %bsBuff.tmp.0556, %bsGetUChar.exit7.i ], [ %1351, %bb2.i.i11.i ]		; <i32> [#uses=4]
	%1353 = phi i32 [ %1351, %bb2.i.i11.i ], [ %1336, %bsGetUChar.exit7.i ]		; <i32> [#uses=2]
	%1354 = phi i32 [ %1352, %bb2.i.i11.i ], [ %1340, %bsGetUChar.exit7.i ]		; <i32> [#uses=1]
	%1355 = phi i32 [ %1352, %bb2.i.i11.i ], [ %1340, %bsGetUChar.exit7.i ]		; <i32> [#uses=1]
	%1356 = icmp slt i32 %1355, 8		; <i1> [#uses=1]
	br i1 %1356, label %bb3.i18.i, label %bsGetUChar.exit14.i

bsGetUChar.exit14.i:		; preds = %bb3.i.i12.i
	store i32 %bsBuff.tmp.0560, i32* @bsBuff
	%1357 = add i32 %1354, -8		; <i32> [#uses=5]
	%1358 = lshr i32 %1353, %1357		; <i32> [#uses=1]
	store i32 %1357, i32* @bsLive, align 4
	%retval12.i13.i = trunc i32 %1358 to i8		; <i8> [#uses=1]
	br label %bb3.i.i19.i

bb3.i25.i:		; preds = %bb3.i.i19.i
	%1359 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 2), align 8		; <i32> [#uses=3]
	%1360 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 1), align 4		; <i32> [#uses=1]
	%1361 = icmp slt i32 %1359, %1360		; <i1> [#uses=1]
	br i1 %1361, label %bb2.i.i18.i, label %bb1.i.i17.i

bb1.i.i17.i:		; preds = %bb3.i25.i
	store i32 %bsBuff.promoted326, i32* @bsBuff
	store i32 %bsLive.tmp.0565, i32* @bsLive
	tail call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i18.i:		; preds = %bb3.i25.i
	%1362 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 3), align 4		; <i8*> [#uses=1]
	%1363 = getelementptr i8* %1362, i32 %1359		; <i8*> [#uses=1]
	%1364 = load i8* %1363, align 1		; <i8> [#uses=1]
	%1365 = add i32 %1359, 1		; <i32> [#uses=1]
	store i32 %1365, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 2), align 8
	%1366 = zext i8 %1364 to i32		; <i32> [#uses=1]
	%1367 = shl i32 %bsBuff.promoted326, 8		; <i32> [#uses=1]
	%1368 = or i32 %1367, %1366		; <i32> [#uses=2]
	%1369 = add i32 %bsLive.tmp.0565, 8		; <i32> [#uses=3]
	br label %bb3.i.i19.i

bb3.i.i19.i:		; preds = %bb2.i.i18.i, %bsGetUChar.exit14.i
	%bsLive.tmp.0565 = phi i32 [ %1357, %bsGetUChar.exit14.i ], [ %1369, %bb2.i.i18.i ]		; <i32> [#uses=2]
	%bsBuff.promoted326 = phi i32 [ %bsBuff.tmp.0560, %bsGetUChar.exit14.i ], [ %1368, %bb2.i.i18.i ]		; <i32> [#uses=6]
	%1370 = phi i32 [ %1368, %bb2.i.i18.i ], [ %1353, %bsGetUChar.exit14.i ]		; <i32> [#uses=1]
	%1371 = phi i32 [ %1369, %bb2.i.i18.i ], [ %1357, %bsGetUChar.exit14.i ]		; <i32> [#uses=1]
	%1372 = phi i32 [ %1369, %bb2.i.i18.i ], [ %1357, %bsGetUChar.exit14.i ]		; <i32> [#uses=1]
	%1373 = icmp slt i32 %1372, 8		; <i1> [#uses=1]
	br i1 %1373, label %bb3.i25.i, label %bsGetUChar.exit21.i

bsGetUChar.exit21.i:		; preds = %bb3.i.i19.i
	store i32 %bsBuff.promoted326, i32* @bsBuff
	%1374 = add i32 %1371, -8		; <i32> [#uses=8]
	%1375 = lshr i32 %1370, %1374		; <i32> [#uses=2]
	store i32 %1374, i32* @bsLive, align 4
	%retval12.i20.i = trunc i32 %1375 to i8		; <i8> [#uses=2]
	%1376 = icmp ne i8 %retval12.i.i, 66		; <i1> [#uses=1]
	%1377 = icmp ne i8 %retval12.i6.i, 90		; <i1> [#uses=1]
	%1378 = or i1 %1377, %1376		; <i1> [#uses=1]
	br i1 %1378, label %bb23, label %bb.i

bb.i:		; preds = %bsGetUChar.exit21.i
	%1379 = icmp ne i8 %retval12.i13.i, 104		; <i1> [#uses=1]
	%1380 = icmp ult i8 %retval12.i20.i, 49		; <i1> [#uses=1]
	%1381 = icmp ugt i8 %retval12.i20.i, 57		; <i1> [#uses=1]
	%1382 = or i1 %1381, %1379		; <i1> [#uses=1]
	%or.cond.i = or i1 %1382, %1380		; <i1> [#uses=1]
	br i1 %or.cond.i, label %bb23, label %bb10.i1

bb10.i1:		; preds = %bb.i
	%1383 = and i32 %1375, 255		; <i32> [#uses=1]
	%1384 = add i32 %1383, -48		; <i32> [#uses=6]
	%1385 = icmp ugt i32 %1384, 9		; <i1> [#uses=1]
	br i1 %1385, label %bb4.i76.i, label %bb5.i77.i

bb4.i76.i:		; preds = %bb10.i1
	tail call fastcc void @panic(i8* getelementptr ([28 x i8]* @"\01LC54", i32 0, i32 0)) nounwind ssp
	unreachable

bb5.i77.i:		; preds = %bb10.i1
	%1386 = icmp eq i32 %1384, 0		; <i1> [#uses=1]
	br i1 %1386, label %bb13.i, label %bb6.i.i

bb6.i.i:		; preds = %bb5.i77.i
	store i32 %1384, i32* @blockSize100k, align 4
	%1387 = icmp eq i8* %ll8.2, null		; <i1> [#uses=1]
	br i1 %1387, label %bb12.i80.i, label %bb11.i.i

bb11.i.i:		; preds = %bb6.i.i
	free i8* %ll8.2
	br label %bb12.i80.i

bb12.i80.i:		; preds = %bb11.i.i, %bb6.i.i
	%1388 = icmp eq i32* %tt.2, null		; <i1> [#uses=1]
	br i1 %1388, label %bb14.i82.i, label %bb13.i81.i

bb13.i81.i:		; preds = %bb12.i80.i
	free i32* %tt.2
	br label %bb14.i82.i

bb14.i82.i:		; preds = %bb13.i81.i, %bb12.i80.i
	%1389 = icmp eq i32 %1384, 0		; <i1> [#uses=1]
	br i1 %1389, label %bb13.i, label %bb15.i83.i

bb15.i83.i:		; preds = %bb14.i82.i
	%1390 = mul i32 %1384, 100000		; <i32> [#uses=3]
	%1391 = malloc i8, i32 %1390		; <i8*> [#uses=2]
	%1392 = malloc i32, i32 %1390		; <i32*> [#uses=2]
	%1393 = icmp eq i8* %1391, null		; <i1> [#uses=1]
	%1394 = icmp eq i32* %1392, null		; <i1> [#uses=1]
	%or.cond2.i.i3 = or i1 %1393, %1394		; <i1> [#uses=1]
	br i1 %or.cond2.i.i3, label %bb21.i85.i, label %bb13.i

bb21.i85.i:		; preds = %bb15.i83.i
	%1395 = mul i32 %1384, 500000		; <i32> [#uses=1]
	%1396 = load %struct.FILE** @__stderrp, align 4		; <%struct.FILE*> [#uses=1]
	%1397 = tail call i32 (%struct.FILE*, i8*, ...)* @fprintf(%struct.FILE* %1396, i8* getelementptr ([206 x i8]* @"\01LC6", i32 0, i32 0), i8* null, i32 %1395, i32 %1390) nounwind		; <i32> [#uses=0]
	%1398 = load %struct.FILE** @__stderrp, align 4		; <%struct.FILE*> [#uses=1]
	%1399 = tail call i32 (%struct.FILE*, i8*, ...)* @fprintf(%struct.FILE* %1398, i8* getelementptr ([36 x i8]* @"\01LC", i32 0, i32 0), i8* getelementptr ([1024 x i8]* @inName, i32 0, i32 0), i8* getelementptr ([1024 x i8]* @outName, i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	tail call fastcc void @cleanUpAndFail(i32 1) nounwind ssp
	unreachable

bb13.i:		; preds = %bb54.i, %bb15.i83.i, %bb14.i82.i, %bb5.i77.i
	%tt.0 = phi i32* [ %tt.2, %bb5.i77.i ], [ %tt.2, %bb14.i82.i ], [ %1392, %bb15.i83.i ], [ %tt.0, %bb54.i ]		; <i32*> [#uses=8]
	%ll8.0 = phi i8* [ %ll8.2, %bb5.i77.i ], [ %ll8.2, %bb14.i82.i ], [ %1391, %bb15.i83.i ], [ %ll8.0, %bb54.i ]		; <i8*> [#uses=9]
	%bsLive.promoted303 = phi i32 [ %.pre.i.i22.i.pre, %bb54.i ], [ %1374, %bb15.i83.i ], [ %1374, %bb14.i82.i ], [ %1374, %bb5.i77.i ]		; <i32> [#uses=1]
	%bsBuff.promoted302 = phi i32 [ %bsBuff.promoted302.pre, %bb54.i ], [ %bsBuff.promoted326, %bb15.i83.i ], [ %bsBuff.promoted326, %bb14.i82.i ], [ %bsBuff.promoted326, %bb5.i77.i ]		; <i32> [#uses=1]
	%.b6.i = phi i32 [ %phitmp758, %bb54.i ], [ 1, %bb15.i83.i ], [ 1, %bb14.i82.i ], [ 1, %bb5.i77.i ]		; <i32> [#uses=3]
	%.pre.i.i22.i = phi i32 [ %.pre.i.i22.i.pre, %bb54.i ], [ %1374, %bb15.i83.i ], [ %1374, %bb14.i82.i ], [ %1374, %bb5.i77.i ]		; <i32> [#uses=2]
	%computedCombinedCRC.0.i = phi i32 [ %2147, %bb54.i ], [ 0, %bb15.i83.i ], [ 0, %bb14.i82.i ], [ 0, %bb5.i77.i ]		; <i32> [#uses=4]
	%1400 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %.b6.i, i32 2		; <i32*> [#uses=2]
	%1401 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %.b6.i, i32 1		; <i32*> [#uses=1]
	%1402 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %.b6.i, i32 3		; <i8**> [#uses=1]
	br label %bb3.i.i26.i

bb3.i88.i:		; preds = %bb3.i.i26.i
	%1403 = load i32* %1400, align 8		; <i32> [#uses=3]
	%1404 = load i32* %1401, align 4		; <i32> [#uses=1]
	%1405 = icmp slt i32 %1403, %1404		; <i1> [#uses=1]
	br i1 %1405, label %bb2.i.i25.i, label %bb1.i.i24.i

bb1.i.i24.i:		; preds = %bb3.i88.i
	store i32 %bsBuff.tmp.0304, i32* @bsBuff
	store i32 %bsLive.tmp.0305, i32* @bsLive
	tail call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i25.i:		; preds = %bb3.i88.i
	%1406 = load i8** %1402, align 4		; <i8*> [#uses=1]
	%1407 = getelementptr i8* %1406, i32 %1403		; <i8*> [#uses=1]
	%1408 = load i8* %1407, align 1		; <i8> [#uses=1]
	%1409 = add i32 %1403, 1		; <i32> [#uses=1]
	store i32 %1409, i32* %1400, align 8
	%1410 = zext i8 %1408 to i32		; <i32> [#uses=1]
	%1411 = shl i32 %bsBuff.tmp.0304, 8		; <i32> [#uses=1]
	%1412 = or i32 %1411, %1410		; <i32> [#uses=1]
	%1413 = add i32 %bsLive.tmp.0305, 8		; <i32> [#uses=3]
	br label %bb3.i.i26.i

bb3.i.i26.i:		; preds = %bb2.i.i25.i, %bb13.i
	%bsLive.tmp.0305 = phi i32 [ %bsLive.promoted303, %bb13.i ], [ %1413, %bb2.i.i25.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0304 = phi i32 [ %bsBuff.promoted302, %bb13.i ], [ %1412, %bb2.i.i25.i ]		; <i32> [#uses=6]
	%1414 = phi i32 [ %1413, %bb2.i.i25.i ], [ %.pre.i.i22.i, %bb13.i ]		; <i32> [#uses=1]
	%1415 = phi i32 [ %1413, %bb2.i.i25.i ], [ %.pre.i.i22.i, %bb13.i ]		; <i32> [#uses=1]
	%1416 = icmp slt i32 %1415, 8		; <i1> [#uses=1]
	br i1 %1416, label %bb3.i88.i, label %bsGetUChar.exit28.i

bsGetUChar.exit28.i:		; preds = %bb3.i.i26.i
	store i32 %bsBuff.tmp.0304, i32* @bsBuff
	%1417 = add i32 %1414, -8		; <i32> [#uses=5]
	%1418 = lshr i32 %bsBuff.tmp.0304, %1417		; <i32> [#uses=1]
	store i32 %1417, i32* @bsLive, align 4
	%retval12.i27.i = trunc i32 %1418 to i8		; <i8> [#uses=2]
	%.b5.i = load i1* @bsStream.b		; <i1> [#uses=1]
	%1419 = zext i1 %.b5.i to i32		; <i32> [#uses=3]
	%1420 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1419, i32 2		; <i32*> [#uses=10]
	%1421 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1419, i32 1		; <i32*> [#uses=5]
	%1422 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1419, i32 3		; <i8**> [#uses=5]
	br label %bb3.i.i33.i

bb3.i104.i:		; preds = %bb3.i.i33.i
	%1423 = load i32* %1420, align 8		; <i32> [#uses=3]
	%1424 = load i32* %1421, align 4		; <i32> [#uses=1]
	%1425 = icmp slt i32 %1423, %1424		; <i1> [#uses=1]
	br i1 %1425, label %bb2.i.i32.i, label %bb1.i.i31.i

bb1.i.i31.i:		; preds = %bb3.i104.i
	store i32 %bsBuff.tmp.0308, i32* @bsBuff
	store i32 %bsLive.tmp.0309, i32* @bsLive
	tail call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i32.i:		; preds = %bb3.i104.i
	%1426 = load i8** %1422, align 4		; <i8*> [#uses=1]
	%1427 = getelementptr i8* %1426, i32 %1423		; <i8*> [#uses=1]
	%1428 = load i8* %1427, align 1		; <i8> [#uses=1]
	%1429 = add i32 %1423, 1		; <i32> [#uses=1]
	store i32 %1429, i32* %1420, align 8
	%1430 = zext i8 %1428 to i32		; <i32> [#uses=1]
	%1431 = shl i32 %bsBuff.tmp.0308, 8		; <i32> [#uses=1]
	%1432 = or i32 %1431, %1430		; <i32> [#uses=2]
	%1433 = add i32 %bsLive.tmp.0309, 8		; <i32> [#uses=3]
	br label %bb3.i.i33.i

bb3.i.i33.i:		; preds = %bb2.i.i32.i, %bsGetUChar.exit28.i
	%bsLive.tmp.0309 = phi i32 [ %1417, %bsGetUChar.exit28.i ], [ %1433, %bb2.i.i32.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0308 = phi i32 [ %bsBuff.tmp.0304, %bsGetUChar.exit28.i ], [ %1432, %bb2.i.i32.i ]		; <i32> [#uses=4]
	%1434 = phi i32 [ %1432, %bb2.i.i32.i ], [ %bsBuff.tmp.0304, %bsGetUChar.exit28.i ]		; <i32> [#uses=2]
	%1435 = phi i32 [ %1433, %bb2.i.i32.i ], [ %1417, %bsGetUChar.exit28.i ]		; <i32> [#uses=1]
	%1436 = phi i32 [ %1433, %bb2.i.i32.i ], [ %1417, %bsGetUChar.exit28.i ]		; <i32> [#uses=1]
	%1437 = icmp slt i32 %1436, 8		; <i1> [#uses=1]
	br i1 %1437, label %bb3.i104.i, label %bsGetUChar.exit35.i

bsGetUChar.exit35.i:		; preds = %bb3.i.i33.i
	store i32 %bsBuff.tmp.0308, i32* @bsBuff
	%1438 = add i32 %1435, -8		; <i32> [#uses=5]
	%1439 = lshr i32 %1434, %1438		; <i32> [#uses=1]
	store i32 %1438, i32* @bsLive, align 4
	%retval12.i34.i = trunc i32 %1439 to i8		; <i8> [#uses=2]
	br label %bb3.i.i40.i

bb3.i120.i:		; preds = %bb3.i.i40.i
	%1440 = load i32* %1420, align 8		; <i32> [#uses=3]
	%1441 = load i32* %1421, align 4		; <i32> [#uses=1]
	%1442 = icmp slt i32 %1440, %1441		; <i1> [#uses=1]
	br i1 %1442, label %bb2.i.i39.i, label %bb1.i.i38.i

bb1.i.i38.i:		; preds = %bb3.i120.i
	store i32 %bsBuff.tmp.0312, i32* @bsBuff
	store i32 %bsLive.tmp.0313, i32* @bsLive
	tail call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i39.i:		; preds = %bb3.i120.i
	%1443 = load i8** %1422, align 4		; <i8*> [#uses=1]
	%1444 = getelementptr i8* %1443, i32 %1440		; <i8*> [#uses=1]
	%1445 = load i8* %1444, align 1		; <i8> [#uses=1]
	%1446 = add i32 %1440, 1		; <i32> [#uses=1]
	store i32 %1446, i32* %1420, align 8
	%1447 = zext i8 %1445 to i32		; <i32> [#uses=1]
	%1448 = shl i32 %bsBuff.tmp.0312, 8		; <i32> [#uses=1]
	%1449 = or i32 %1448, %1447		; <i32> [#uses=2]
	%1450 = add i32 %bsLive.tmp.0313, 8		; <i32> [#uses=3]
	br label %bb3.i.i40.i

bb3.i.i40.i:		; preds = %bb2.i.i39.i, %bsGetUChar.exit35.i
	%bsLive.tmp.0313 = phi i32 [ %1438, %bsGetUChar.exit35.i ], [ %1450, %bb2.i.i39.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0312 = phi i32 [ %bsBuff.tmp.0308, %bsGetUChar.exit35.i ], [ %1449, %bb2.i.i39.i ]		; <i32> [#uses=4]
	%1451 = phi i32 [ %1449, %bb2.i.i39.i ], [ %1434, %bsGetUChar.exit35.i ]		; <i32> [#uses=2]
	%1452 = phi i32 [ %1450, %bb2.i.i39.i ], [ %1438, %bsGetUChar.exit35.i ]		; <i32> [#uses=1]
	%1453 = phi i32 [ %1450, %bb2.i.i39.i ], [ %1438, %bsGetUChar.exit35.i ]		; <i32> [#uses=1]
	%1454 = icmp slt i32 %1453, 8		; <i1> [#uses=1]
	br i1 %1454, label %bb3.i120.i, label %bsGetUChar.exit42.i

bsGetUChar.exit42.i:		; preds = %bb3.i.i40.i
	store i32 %bsBuff.tmp.0312, i32* @bsBuff
	%1455 = add i32 %1452, -8		; <i32> [#uses=5]
	%1456 = lshr i32 %1451, %1455		; <i32> [#uses=1]
	store i32 %1455, i32* @bsLive, align 4
	%retval12.i41.i = trunc i32 %1456 to i8		; <i8> [#uses=2]
	br label %bb3.i.i47.i

bb3.i128.i:		; preds = %bb3.i.i47.i
	%1457 = load i32* %1420, align 8		; <i32> [#uses=3]
	%1458 = load i32* %1421, align 4		; <i32> [#uses=1]
	%1459 = icmp slt i32 %1457, %1458		; <i1> [#uses=1]
	br i1 %1459, label %bb2.i.i46.i, label %bb1.i.i45.i

bb1.i.i45.i:		; preds = %bb3.i128.i
	store i32 %bsBuff.tmp.0316, i32* @bsBuff
	store i32 %bsLive.tmp.0317, i32* @bsLive
	tail call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i46.i:		; preds = %bb3.i128.i
	%1460 = load i8** %1422, align 4		; <i8*> [#uses=1]
	%1461 = getelementptr i8* %1460, i32 %1457		; <i8*> [#uses=1]
	%1462 = load i8* %1461, align 1		; <i8> [#uses=1]
	%1463 = add i32 %1457, 1		; <i32> [#uses=1]
	store i32 %1463, i32* %1420, align 8
	%1464 = zext i8 %1462 to i32		; <i32> [#uses=1]
	%1465 = shl i32 %bsBuff.tmp.0316, 8		; <i32> [#uses=1]
	%1466 = or i32 %1465, %1464		; <i32> [#uses=2]
	%1467 = add i32 %bsLive.tmp.0317, 8		; <i32> [#uses=3]
	br label %bb3.i.i47.i

bb3.i.i47.i:		; preds = %bb2.i.i46.i, %bsGetUChar.exit42.i
	%bsLive.tmp.0317 = phi i32 [ %1455, %bsGetUChar.exit42.i ], [ %1467, %bb2.i.i46.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0316 = phi i32 [ %bsBuff.tmp.0312, %bsGetUChar.exit42.i ], [ %1466, %bb2.i.i46.i ]		; <i32> [#uses=4]
	%1468 = phi i32 [ %1466, %bb2.i.i46.i ], [ %1451, %bsGetUChar.exit42.i ]		; <i32> [#uses=2]
	%1469 = phi i32 [ %1467, %bb2.i.i46.i ], [ %1455, %bsGetUChar.exit42.i ]		; <i32> [#uses=1]
	%1470 = phi i32 [ %1467, %bb2.i.i46.i ], [ %1455, %bsGetUChar.exit42.i ]		; <i32> [#uses=1]
	%1471 = icmp slt i32 %1470, 8		; <i1> [#uses=1]
	br i1 %1471, label %bb3.i128.i, label %bsGetUChar.exit49.i

bsGetUChar.exit49.i:		; preds = %bb3.i.i47.i
	store i32 %bsBuff.tmp.0316, i32* @bsBuff
	%1472 = add i32 %1469, -8		; <i32> [#uses=5]
	%1473 = lshr i32 %1468, %1472		; <i32> [#uses=1]
	store i32 %1472, i32* @bsLive, align 4
	%retval12.i48.i = trunc i32 %1473 to i8		; <i8> [#uses=2]
	br label %bb3.i.i54.i

bb3.i112.i:		; preds = %bb3.i.i54.i
	%1474 = load i32* %1420, align 8		; <i32> [#uses=3]
	%1475 = load i32* %1421, align 4		; <i32> [#uses=1]
	%1476 = icmp slt i32 %1474, %1475		; <i1> [#uses=1]
	br i1 %1476, label %bb2.i.i53.i, label %bb1.i.i52.i

bb1.i.i52.i:		; preds = %bb3.i112.i
	store i32 %bsBuff.tmp.0320, i32* @bsBuff
	store i32 %bsLive.tmp.0321, i32* @bsLive
	tail call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i53.i:		; preds = %bb3.i112.i
	%1477 = load i8** %1422, align 4		; <i8*> [#uses=1]
	%1478 = getelementptr i8* %1477, i32 %1474		; <i8*> [#uses=1]
	%1479 = load i8* %1478, align 1		; <i8> [#uses=1]
	%1480 = add i32 %1474, 1		; <i32> [#uses=1]
	store i32 %1480, i32* %1420, align 8
	%1481 = zext i8 %1479 to i32		; <i32> [#uses=1]
	%1482 = shl i32 %bsBuff.tmp.0320, 8		; <i32> [#uses=1]
	%1483 = or i32 %1482, %1481		; <i32> [#uses=2]
	%1484 = add i32 %bsLive.tmp.0321, 8		; <i32> [#uses=3]
	br label %bb3.i.i54.i

bb3.i.i54.i:		; preds = %bb2.i.i53.i, %bsGetUChar.exit49.i
	%bsLive.tmp.0321 = phi i32 [ %1472, %bsGetUChar.exit49.i ], [ %1484, %bb2.i.i53.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0320 = phi i32 [ %bsBuff.tmp.0316, %bsGetUChar.exit49.i ], [ %1483, %bb2.i.i53.i ]		; <i32> [#uses=4]
	%1485 = phi i32 [ %1483, %bb2.i.i53.i ], [ %1468, %bsGetUChar.exit49.i ]		; <i32> [#uses=2]
	%1486 = phi i32 [ %1484, %bb2.i.i53.i ], [ %1472, %bsGetUChar.exit49.i ]		; <i32> [#uses=1]
	%1487 = phi i32 [ %1484, %bb2.i.i53.i ], [ %1472, %bsGetUChar.exit49.i ]		; <i32> [#uses=1]
	%1488 = icmp slt i32 %1487, 8		; <i1> [#uses=1]
	br i1 %1488, label %bb3.i112.i, label %bsGetUChar.exit56.i

bsGetUChar.exit56.i:		; preds = %bb3.i.i54.i
	store i32 %bsBuff.tmp.0320, i32* @bsBuff
	%1489 = add i32 %1486, -8		; <i32> [#uses=5]
	%1490 = lshr i32 %1485, %1489		; <i32> [#uses=1]
	store i32 %1489, i32* @bsLive, align 4
	%retval12.i55.i = trunc i32 %1490 to i8		; <i8> [#uses=2]
	br label %bb3.i.i61.i

bb3.i96.i:		; preds = %bb3.i.i61.i
	%1491 = load i32* %1420, align 8		; <i32> [#uses=3]
	%1492 = load i32* %1421, align 4		; <i32> [#uses=1]
	%1493 = icmp slt i32 %1491, %1492		; <i1> [#uses=1]
	br i1 %1493, label %bb2.i.i60.i, label %bb1.i.i59.i

bb1.i.i59.i:		; preds = %bb3.i96.i
	store i32 %bsBuff.tmp.0324, i32* @bsBuff
	store i32 %bsLive.tmp.0325, i32* @bsLive
	tail call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i60.i:		; preds = %bb3.i96.i
	%1494 = load i8** %1422, align 4		; <i8*> [#uses=1]
	%1495 = getelementptr i8* %1494, i32 %1491		; <i8*> [#uses=1]
	%1496 = load i8* %1495, align 1		; <i8> [#uses=1]
	%1497 = add i32 %1491, 1		; <i32> [#uses=1]
	store i32 %1497, i32* %1420, align 8
	%1498 = zext i8 %1496 to i32		; <i32> [#uses=1]
	%1499 = shl i32 %bsBuff.tmp.0324, 8		; <i32> [#uses=1]
	%1500 = or i32 %1499, %1498		; <i32> [#uses=2]
	%1501 = add i32 %bsLive.tmp.0325, 8		; <i32> [#uses=3]
	br label %bb3.i.i61.i

bb3.i.i61.i:		; preds = %bb2.i.i60.i, %bsGetUChar.exit56.i
	%bsLive.tmp.0325 = phi i32 [ %1489, %bsGetUChar.exit56.i ], [ %1501, %bb2.i.i60.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0324 = phi i32 [ %bsBuff.tmp.0320, %bsGetUChar.exit56.i ], [ %1500, %bb2.i.i60.i ]		; <i32> [#uses=3]
	%1502 = phi i32 [ %1500, %bb2.i.i60.i ], [ %1485, %bsGetUChar.exit56.i ]		; <i32> [#uses=1]
	%1503 = phi i32 [ %1501, %bb2.i.i60.i ], [ %1489, %bsGetUChar.exit56.i ]		; <i32> [#uses=1]
	%1504 = phi i32 [ %1501, %bb2.i.i60.i ], [ %1489, %bsGetUChar.exit56.i ]		; <i32> [#uses=1]
	%1505 = icmp slt i32 %1504, 8		; <i1> [#uses=1]
	br i1 %1505, label %bb3.i96.i, label %bsGetUChar.exit63.i

bsGetUChar.exit63.i:		; preds = %bb3.i.i61.i
	store i32 %bsBuff.tmp.0324, i32* @bsBuff
	%1506 = add i32 %1503, -8		; <i32> [#uses=2]
	%1507 = lshr i32 %1502, %1506		; <i32> [#uses=1]
	store i32 %1506, i32* @bsLive, align 4
	%retval12.i62.i = trunc i32 %1507 to i8		; <i8> [#uses=2]
	%1508 = icmp eq i8 %retval12.i27.i, 23		; <i1> [#uses=1]
	%1509 = icmp eq i8 %retval12.i34.i, 114		; <i1> [#uses=1]
	%1510 = and i1 %1509, %1508		; <i1> [#uses=1]
	br i1 %1510, label %bb17.i, label %bb25.i

bb17.i:		; preds = %bsGetUChar.exit63.i
	%1511 = icmp eq i8 %retval12.i41.i, 69		; <i1> [#uses=1]
	%1512 = icmp eq i8 %retval12.i48.i, 56		; <i1> [#uses=1]
	%1513 = and i1 %1512, %1511		; <i1> [#uses=1]
	br i1 %1513, label %bb21.i, label %bb25.i

bb21.i:		; preds = %bb17.i
	%1514 = icmp eq i8 %retval12.i55.i, 80		; <i1> [#uses=1]
	%1515 = icmp eq i8 %retval12.i62.i, -112		; <i1> [#uses=1]
	%1516 = and i1 %1515, %1514		; <i1> [#uses=1]
	br i1 %1516, label %bb55.i, label %bb25.i

bb25.i:		; preds = %bb21.i, %bb17.i, %bsGetUChar.exit63.i
	%1517 = icmp ne i8 %retval12.i27.i, 49		; <i1> [#uses=1]
	%1518 = icmp ne i8 %retval12.i34.i, 65		; <i1> [#uses=1]
	%1519 = or i1 %1518, %1517		; <i1> [#uses=1]
	br i1 %1519, label %bb37.i, label %bb29.i

bb29.i:		; preds = %bb25.i
	%1520 = icmp ne i8 %retval12.i41.i, 89		; <i1> [#uses=1]
	%1521 = icmp ne i8 %retval12.i48.i, 38		; <i1> [#uses=1]
	%1522 = or i1 %1521, %1520		; <i1> [#uses=1]
	br i1 %1522, label %bb37.i, label %bb33.i

bb33.i:		; preds = %bb29.i
	%1523 = icmp ne i8 %retval12.i55.i, 83		; <i1> [#uses=1]
	%1524 = icmp ne i8 %retval12.i62.i, 89		; <i1> [#uses=1]
	%1525 = or i1 %1524, %1523		; <i1> [#uses=1]
	br i1 %1525, label %bb37.i, label %bb38.i

bb37.i:		; preds = %bb33.i, %bb29.i, %bb25.i
	%1526 = load %struct.FILE** @__stderrp, align 4		; <%struct.FILE*> [#uses=1]
	%1527 = tail call i32 (%struct.FILE*, i8*, ...)* @fprintf(%struct.FILE* %1526, i8* getelementptr ([86 x i8]* @"\01LC17", i32 0, i32 0), i8* null) nounwind		; <i32> [#uses=0]
	%1528 = load %struct.FILE** @__stderrp, align 4		; <%struct.FILE*> [#uses=1]
	%1529 = tail call i32 (%struct.FILE*, i8*, ...)* @fprintf(%struct.FILE* %1528, i8* getelementptr ([36 x i8]* @"\01LC", i32 0, i32 0), i8* getelementptr ([1024 x i8]* @inName, i32 0, i32 0), i8* getelementptr ([1024 x i8]* @outName, i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	%1530 = load %struct.FILE** @__stderrp, align 4		; <%struct.FILE*> [#uses=1]
	%1531 = bitcast %struct.FILE* %1530 to i8*		; <i8*> [#uses=1]
	%1532 = tail call i32 @"\01_fwrite$UNIX2003"(i8* getelementptr ([243 x i8]* @"\01LC13", i32 0, i32 0), i32 1, i32 242, i8* %1531) nounwind		; <i32> [#uses=0]
	tail call fastcc void @cleanUpAndFail(i32 2) nounwind ssp
	unreachable

bb38.i:		; preds = %bb33.i
	%1533 = tail call fastcc i32 @bsGetUInt32() nounwind ssp		; <i32> [#uses=2]
	%.pre.i.i = load i32* @bsLive, align 4		; <i32> [#uses=3]
	%.b.i = load i1* @bsStream.b		; <i1> [#uses=4]
	%1534 = zext i1 %.b.i to i32		; <i32> [#uses=3]
	%1535 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1534, i32 2		; <i32*> [#uses=8]
	%1536 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1534, i32 1		; <i32*> [#uses=4]
	%1537 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1534, i32 3		; <i8**> [#uses=4]
	%bsBuff.promoted183 = load i32* @bsBuff		; <i32> [#uses=1]
	br label %bb3.i.i4

bb3.i69.i:		; preds = %bb3.i.i4
	%1538 = load i32* %1535, align 8		; <i32> [#uses=3]
	%1539 = load i32* %1536, align 4		; <i32> [#uses=1]
	%1540 = icmp slt i32 %1538, %1539		; <i1> [#uses=1]
	br i1 %1540, label %bb2.i.i, label %bb1.i.i

bb1.i.i:		; preds = %bb3.i69.i
	store i32 %bsBuff.tmp.0185, i32* @bsBuff
	store i32 %bsLive.tmp.0186, i32* @bsLive
	tail call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i:		; preds = %bb3.i69.i
	%1541 = load i8** %1537, align 4		; <i8*> [#uses=1]
	%1542 = getelementptr i8* %1541, i32 %1538		; <i8*> [#uses=1]
	%1543 = load i8* %1542, align 1		; <i8> [#uses=1]
	%1544 = add i32 %1538, 1		; <i32> [#uses=1]
	store i32 %1544, i32* %1535, align 8
	%1545 = zext i8 %1543 to i32		; <i32> [#uses=1]
	%1546 = shl i32 %bsBuff.tmp.0185, 8		; <i32> [#uses=1]
	%1547 = or i32 %1546, %1545		; <i32> [#uses=1]
	%1548 = add i32 %bsLive.tmp.0186, 8		; <i32> [#uses=3]
	br label %bb3.i.i4

bb3.i.i4:		; preds = %bb2.i.i, %bb38.i
	%bsLive.tmp.0186 = phi i32 [ %.pre.i.i, %bb38.i ], [ %1548, %bb2.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0185 = phi i32 [ %bsBuff.promoted183, %bb38.i ], [ %1547, %bb2.i.i ]		; <i32> [#uses=5]
	%1549 = phi i32 [ %1548, %bb2.i.i ], [ %.pre.i.i, %bb38.i ]		; <i32> [#uses=1]
	%1550 = phi i32 [ %1548, %bb2.i.i ], [ %.pre.i.i, %bb38.i ]		; <i32> [#uses=1]
	%1551 = icmp slt i32 %1550, 1		; <i1> [#uses=1]
	br i1 %1551, label %bb3.i69.i, label %bsR.exit.i

bsR.exit.i:		; preds = %bb3.i.i4
	store i32 %bsBuff.tmp.0185, i32* @bsBuff
	%1552 = add i32 %1549, -1		; <i32> [#uses=5]
	store i32 %1552, i32* @bsLive, align 4
	%1553 = load i32* @blockSize100k, align 4		; <i32> [#uses=1]
	%1554 = mul i32 %1553, 100000		; <i32> [#uses=2]
	br label %bb3.i.i.i.i

bb3.i9.i.i:		; preds = %bb3.i.i.i.i
	%1555 = load i32* %1535, align 8		; <i32> [#uses=3]
	%1556 = load i32* %1536, align 4		; <i32> [#uses=1]
	%1557 = icmp slt i32 %1555, %1556		; <i1> [#uses=1]
	br i1 %1557, label %bb2.i.i.i.i, label %bb1.i.i.i.i

bb1.i.i.i.i:		; preds = %bb3.i9.i.i
	store i32 %bsBuff.tmp.0189, i32* @bsBuff
	store i32 %bsLive.tmp.0190, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i.i.i:		; preds = %bb3.i9.i.i
	%1558 = load i8** %1537, align 4		; <i8*> [#uses=1]
	%1559 = getelementptr i8* %1558, i32 %1555		; <i8*> [#uses=1]
	%1560 = load i8* %1559, align 1		; <i8> [#uses=1]
	%1561 = add i32 %1555, 1		; <i32> [#uses=1]
	store i32 %1561, i32* %1535, align 8
	%1562 = zext i8 %1560 to i32		; <i32> [#uses=1]
	%1563 = shl i32 %bsBuff.tmp.0189, 8		; <i32> [#uses=1]
	%1564 = or i32 %1563, %1562		; <i32> [#uses=1]
	%1565 = add i32 %bsLive.tmp.0190, 8		; <i32> [#uses=3]
	br label %bb3.i.i.i.i

bb3.i.i.i.i:		; preds = %bb2.i.i.i.i, %bsR.exit.i
	%bsLive.tmp.0190 = phi i32 [ %1552, %bsR.exit.i ], [ %1565, %bb2.i.i.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0189 = phi i32 [ %bsBuff.tmp.0185, %bsR.exit.i ], [ %1564, %bb2.i.i.i.i ]		; <i32> [#uses=5]
	%1566 = phi i32 [ %1565, %bb2.i.i.i.i ], [ %1552, %bsR.exit.i ]		; <i32> [#uses=1]
	%1567 = phi i32 [ %1565, %bb2.i.i.i.i ], [ %1552, %bsR.exit.i ]		; <i32> [#uses=1]
	%1568 = icmp slt i32 %1567, 24		; <i1> [#uses=1]
	br i1 %1568, label %bb3.i9.i.i, label %bsGetIntVS.exit.i.i

bsGetIntVS.exit.i.i:		; preds = %bb3.i.i.i.i
	store i32 %bsBuff.tmp.0189, i32* @bsBuff
	%1569 = add i32 %1566, -24		; <i32> [#uses=5]
	%1570 = lshr i32 %bsBuff.tmp.0189, %1569		; <i32> [#uses=1]
	%1571 = and i32 %1570, 16777215		; <i32> [#uses=1]
	store i32 %1569, i32* @bsLive, align 4
	br label %bb4.i16.i.i

bb3.i10.i.i.i:		; preds = %bb3.i.i13.i.i
	%1572 = load i32* %1535, align 8		; <i32> [#uses=3]
	%1573 = load i32* %1536, align 4		; <i32> [#uses=1]
	%1574 = icmp slt i32 %1572, %1573		; <i1> [#uses=1]
	br i1 %1574, label %bb2.i.i12.i.i, label %bb1.i.i11.i.i

bb1.i.i11.i.i:		; preds = %bb3.i10.i.i.i
	store i32 %bsBuff.tmp.0193, i32* @bsBuff
	store i32 %bsLive.tmp.0194, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i12.i.i:		; preds = %bb3.i10.i.i.i
	%1575 = load i8** %1537, align 4		; <i8*> [#uses=1]
	%1576 = getelementptr i8* %1575, i32 %1572		; <i8*> [#uses=1]
	%1577 = load i8* %1576, align 1		; <i8> [#uses=1]
	%1578 = add i32 %1572, 1		; <i32> [#uses=1]
	store i32 %1578, i32* %1535, align 8
	%1579 = zext i8 %1577 to i32		; <i32> [#uses=1]
	%1580 = shl i32 %bsBuff.tmp.0193, 8		; <i32> [#uses=1]
	%1581 = or i32 %1580, %1579		; <i32> [#uses=1]
	%1582 = add i32 %bsLive.tmp.0194, 8		; <i32> [#uses=3]
	br label %bb3.i.i13.i.i

bb3.i.i13.i.i:		; preds = %bb4.i16.i.i, %bb2.i.i12.i.i
	%bsLive.tmp.0194 = phi i32 [ %1582, %bb2.i.i12.i.i ], [ %bsLive.promoted208, %bb4.i16.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0193 = phi i32 [ %1581, %bb2.i.i12.i.i ], [ %bsBuff.promoted207, %bb4.i16.i.i ]		; <i32> [#uses=4]
	%1583 = phi i32 [ %1582, %bb2.i.i12.i.i ], [ %.pre.i14.i.i, %bb4.i16.i.i ]		; <i32> [#uses=1]
	%1584 = phi i32 [ %1582, %bb2.i.i12.i.i ], [ %.pre.i.i15.i.i, %bb4.i16.i.i ]		; <i32> [#uses=1]
	%1585 = icmp slt i32 %1584, 1		; <i1> [#uses=1]
	br i1 %1585, label %bb3.i10.i.i.i, label %bsR.exit.i.i.i

bsR.exit.i.i.i:		; preds = %bb3.i.i13.i.i
	%1586 = add i32 %1583, -1		; <i32> [#uses=4]
	%1587 = lshr i32 %bsBuff.tmp.0193, %1586		; <i32> [#uses=1]
	%1588 = trunc i32 %1587 to i8		; <i8> [#uses=1]
	%storemerge.i.i.i = and i8 %1588, 1		; <i8> [#uses=1]
	store i8 %storemerge.i.i.i, i8* %scevgep107.i.i.i
	%1589 = add i32 %1590, 1		; <i32> [#uses=1]
	br label %bb4.i16.i.i

bb4.i16.i.i:		; preds = %bsR.exit.i.i.i, %bsGetIntVS.exit.i.i
	%bsLive.promoted208 = phi i32 [ %1569, %bsGetIntVS.exit.i.i ], [ %1586, %bsR.exit.i.i.i ]		; <i32> [#uses=3]
	%bsBuff.promoted207 = phi i32 [ %bsBuff.tmp.0189, %bsGetIntVS.exit.i.i ], [ %bsBuff.tmp.0193, %bsR.exit.i.i.i ]		; <i32> [#uses=3]
	%.pre.i14.i.i = phi i32 [ %1586, %bsR.exit.i.i.i ], [ %1569, %bsGetIntVS.exit.i.i ]		; <i32> [#uses=1]
	%.pre.i.i15.i.i = phi i32 [ %1586, %bsR.exit.i.i.i ], [ %1569, %bsGetIntVS.exit.i.i ]		; <i32> [#uses=1]
	%1590 = phi i32 [ 0, %bsGetIntVS.exit.i.i ], [ %1589, %bsR.exit.i.i.i ]		; <i32> [#uses=3]
	%scevgep107.i.i.i = getelementptr [16 x i8]* %inUse16.i.i.i, i32 0, i32 %1590		; <i8*> [#uses=1]
	%1591 = icmp sgt i32 %1590, 15		; <i1> [#uses=1]
	br i1 %1591, label %bb6.i.i.i.preheader, label %bb3.i.i13.i.i

bb6.i.i.i.preheader:		; preds = %bb4.i16.i.i
	store i32 %bsBuff.promoted207, i32* @bsBuff
	store i32 %bsLive.promoted208, i32* @bsLive
	br label %bb6.i.i.i

bb6.i.i.i:		; preds = %bb6.i.i.i, %bb6.i.i.i.preheader
	%i.166.i.i.i = phi i32 [ %1592, %bb6.i.i.i ], [ 0, %bb6.i.i.i.preheader ]		; <i32> [#uses=2]
	%scevgep106.i.i.i = getelementptr [256 x i8]* @inUse, i32 0, i32 %i.166.i.i.i		; <i8*> [#uses=1]
	store i8 0, i8* %scevgep106.i.i.i, align 1
	%1592 = add i32 %i.166.i.i.i, 1		; <i32> [#uses=2]
	%exitcond105.i.i.i = icmp eq i32 %1592, 256		; <i1> [#uses=1]
	br i1 %exitcond105.i.i.i, label %bb16.i.i.i, label %bb6.i.i.i

bb9.i.i.i:		; preds = %bb16.i.i.i
	%scevgep104.i.i.i = getelementptr [16 x i8]* %inUse16.i.i.i, i32 0, i32 %1616		; <i8*> [#uses=1]
	%1593 = load i8* %scevgep104.i.i.i, align 1		; <i8> [#uses=1]
	%1594 = icmp eq i8 %1593, 0		; <i1> [#uses=1]
	br i1 %1594, label %bb15.i.i.i, label %bb14.i.i.i

bb3.i18.i.i.i:		; preds = %bb3.i5.i.i.i
	%1595 = load i32* %1535, align 8		; <i32> [#uses=3]
	%1596 = load i32* %1536, align 4		; <i32> [#uses=1]
	%1597 = icmp slt i32 %1595, %1596		; <i1> [#uses=1]
	br i1 %1597, label %bb2.i4.i.i.i, label %bb1.i3.i.i.i

bb1.i3.i.i.i:		; preds = %bb3.i18.i.i.i
	store i32 %bsBuff.tmp.0201, i32* @bsBuff
	store i32 %bsLive.tmp.0202, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i4.i.i.i:		; preds = %bb3.i18.i.i.i
	%1598 = load i8** %1537, align 4		; <i8*> [#uses=1]
	%1599 = getelementptr i8* %1598, i32 %1595		; <i8*> [#uses=1]
	%1600 = load i8* %1599, align 1		; <i8> [#uses=1]
	%1601 = add i32 %1595, 1		; <i32> [#uses=1]
	store i32 %1601, i32* %1535, align 8
	%1602 = zext i8 %1600 to i32		; <i32> [#uses=1]
	%1603 = shl i32 %bsBuff.tmp.0201, 8		; <i32> [#uses=1]
	%1604 = or i32 %1603, %1602		; <i32> [#uses=1]
	%1605 = add i32 %bsLive.tmp.0202, 8		; <i32> [#uses=3]
	br label %bb3.i5.i.i.i

bb3.i5.i.i.i:		; preds = %bb14.i.i.i, %bb2.i4.i.i.i
	%bsLive.tmp.0202 = phi i32 [ %1605, %bb2.i4.i.i.i ], [ %bsLive.tmp.0206, %bb14.i.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0201 = phi i32 [ %1604, %bb2.i4.i.i.i ], [ %bsBuff.tmp.0205, %bb14.i.i.i ]		; <i32> [#uses=4]
	%1606 = phi i32 [ %1605, %bb2.i4.i.i.i ], [ %.pre130.i.i.i, %bb14.i.i.i ]		; <i32> [#uses=1]
	%1607 = phi i32 [ %1605, %bb2.i4.i.i.i ], [ %.pre.i1.i.i.i, %bb14.i.i.i ]		; <i32> [#uses=1]
	%1608 = icmp slt i32 %1607, 1		; <i1> [#uses=1]
	br i1 %1608, label %bb3.i18.i.i.i, label %bsR.exit6.i.i.i

bsR.exit6.i.i.i:		; preds = %bb3.i5.i.i.i
	%1609 = add i32 %1606, -1		; <i32> [#uses=4]
	%tmp49.i.i.i = shl i32 1, %1609		; <i32> [#uses=1]
	%1610 = and i32 %tmp49.i.i.i, %bsBuff.tmp.0201		; <i32> [#uses=1]
	%1611 = icmp eq i32 %1610, 0		; <i1> [#uses=1]
	br i1 %1611, label %bb13.i.i.i, label %bb12.i.i.i

bb12.i.i.i:		; preds = %bsR.exit6.i.i.i
	store i8 1, i8* %scevgep102.i.i.i, align 1
	br label %bb13.i.i.i

bb13.i.i.i:		; preds = %bb12.i.i.i, %bsR.exit6.i.i.i
	%1612 = add i32 %1613, 1		; <i32> [#uses=1]
	br label %bb14.i.i.i

bb14.i.i.i:		; preds = %bb13.i.i.i, %bb9.i.i.i
	%bsLive.tmp.0206 = phi i32 [ %1609, %bb13.i.i.i ], [ %bsLive.tmp.1, %bb9.i.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0205 = phi i32 [ %bsBuff.tmp.0201, %bb13.i.i.i ], [ %bsBuff.tmp.1, %bb9.i.i.i ]		; <i32> [#uses=2]
	%.pre130.i.i.i = phi i32 [ %1609, %bb13.i.i.i ], [ %bsLive.tmp.1, %bb9.i.i.i ]		; <i32> [#uses=1]
	%.pre.i1.i.i.i = phi i32 [ %1609, %bb13.i.i.i ], [ %bsLive.tmp.1, %bb9.i.i.i ]		; <i32> [#uses=1]
	%1613 = phi i32 [ %1612, %bb13.i.i.i ], [ 0, %bb9.i.i.i ]		; <i32> [#uses=3]
	%tmp101.i.i.i = add i32 %1613, %tmp100.i.i.i		; <i32> [#uses=1]
	%scevgep102.i.i.i = getelementptr [256 x i8]* @inUse, i32 0, i32 %tmp101.i.i.i		; <i8*> [#uses=1]
	%1614 = icmp sgt i32 %1613, 15		; <i1> [#uses=1]
	br i1 %1614, label %bb15.i.i.i, label %bb3.i5.i.i.i

bb15.i.i.i:		; preds = %bb14.i.i.i, %bb9.i.i.i
	%bsLive.tmp.0210 = phi i32 [ %bsLive.tmp.1, %bb9.i.i.i ], [ %bsLive.tmp.0206, %bb14.i.i.i ]		; <i32> [#uses=1]
	%bsBuff.tmp.0209 = phi i32 [ %bsBuff.tmp.1, %bb9.i.i.i ], [ %bsBuff.tmp.0205, %bb14.i.i.i ]		; <i32> [#uses=1]
	%1615 = add i32 %1616, 1		; <i32> [#uses=1]
	br label %bb16.i.i.i

bb16.i.i.i:		; preds = %bb15.i.i.i, %bb6.i.i.i
	%bsLive.tmp.1 = phi i32 [ %bsLive.tmp.0210, %bb15.i.i.i ], [ %bsLive.promoted208, %bb6.i.i.i ]		; <i32> [#uses=8]
	%bsBuff.tmp.1 = phi i32 [ %bsBuff.tmp.0209, %bb15.i.i.i ], [ %bsBuff.promoted207, %bb6.i.i.i ]		; <i32> [#uses=4]
	%1616 = phi i32 [ %1615, %bb15.i.i.i ], [ 0, %bb6.i.i.i ]		; <i32> [#uses=4]
	%tmp100.i.i.i = shl i32 %1616, 4		; <i32> [#uses=1]
	%1617 = icmp sgt i32 %1616, 15		; <i1> [#uses=1]
	br i1 %1617, label %bb17.i.i.i, label %bb9.i.i.i

bb17.i.i.i:		; preds = %bb16.i.i.i
	store i32 %bsBuff.tmp.1, i32* @bsBuff
	store i32 %bsLive.tmp.1, i32* @bsLive
	store i32 0, i32* @nInUse, align 4
	br label %bb.i43.i.i.i

bb.i43.i.i.i:		; preds = %bb2.i45.i.i.i, %bb17.i.i.i
	%nInUse.tmp.1.i.i.i.i = phi i32 [ 0, %bb17.i.i.i ], [ %nInUse.tmp.0.i.i.i.i, %bb2.i45.i.i.i ]		; <i32> [#uses=4]
	%i.01.i.i.i.i = phi i32 [ 0, %bb17.i.i.i ], [ %1623, %bb2.i45.i.i.i ]		; <i32> [#uses=4]
	%scevgep.i.i.i.i = getelementptr [256 x i8]* @inUse, i32 0, i32 %i.01.i.i.i.i		; <i8*> [#uses=1]
	%1618 = load i8* %scevgep.i.i.i.i, align 1		; <i8> [#uses=1]
	%1619 = icmp eq i8 %1618, 0		; <i1> [#uses=1]
	br i1 %1619, label %bb2.i45.i.i.i, label %bb1.i44.i.i.i

bb1.i44.i.i.i:		; preds = %bb.i43.i.i.i
	%scevgep2.i.i.i.i = getelementptr [256 x i8]* @unseqToSeq, i32 0, i32 %i.01.i.i.i.i		; <i8*> [#uses=1]
	%tmp.i.i.i.i = trunc i32 %i.01.i.i.i.i to i8		; <i8> [#uses=1]
	%1620 = getelementptr [256 x i8]* @seqToUnseq, i32 0, i32 %nInUse.tmp.1.i.i.i.i		; <i8*> [#uses=1]
	store i8 %tmp.i.i.i.i, i8* %1620, align 1
	%1621 = trunc i32 %nInUse.tmp.1.i.i.i.i to i8		; <i8> [#uses=1]
	store i8 %1621, i8* %scevgep2.i.i.i.i, align 1
	%1622 = add i32 %nInUse.tmp.1.i.i.i.i, 1		; <i32> [#uses=1]
	br label %bb2.i45.i.i.i

bb2.i45.i.i.i:		; preds = %bb1.i44.i.i.i, %bb.i43.i.i.i
	%nInUse.tmp.0.i.i.i.i = phi i32 [ %1622, %bb1.i44.i.i.i ], [ %nInUse.tmp.1.i.i.i.i, %bb.i43.i.i.i ]		; <i32> [#uses=4]
	%1623 = add i32 %i.01.i.i.i.i, 1		; <i32> [#uses=2]
	%exitcond96.i.i.i = icmp eq i32 %1623, 256		; <i1> [#uses=1]
	br i1 %exitcond96.i.i.i, label %makeMaps.exit.i.i.i, label %bb.i43.i.i.i

makeMaps.exit.i.i.i:		; preds = %bb2.i45.i.i.i
	store i32 %nInUse.tmp.0.i.i.i.i, i32* @nInUse
	%1624 = add i32 %nInUse.tmp.0.i.i.i.i, 2		; <i32> [#uses=5]
	%1625 = zext i1 %.b.i to i32		; <i32> [#uses=3]
	%1626 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1625, i32 2		; <i32*> [#uses=6]
	%1627 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1625, i32 1		; <i32*> [#uses=3]
	%1628 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1625, i32 3		; <i8**> [#uses=3]
	br label %bb3.i41.i.i.i

bb3.i32.i.i.i:		; preds = %bb3.i41.i.i.i
	%1629 = load i32* %1626, align 8		; <i32> [#uses=3]
	%1630 = load i32* %1627, align 4		; <i32> [#uses=1]
	%1631 = icmp slt i32 %1629, %1630		; <i1> [#uses=1]
	br i1 %1631, label %bb2.i40.i.i.i, label %bb1.i39.i.i.i

bb1.i39.i.i.i:		; preds = %bb3.i32.i.i.i
	store i32 %bsBuff.tmp.0213, i32* @bsBuff
	store i32 %bsLive.tmp.0214, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i40.i.i.i:		; preds = %bb3.i32.i.i.i
	%1632 = load i8** %1628, align 4		; <i8*> [#uses=1]
	%1633 = getelementptr i8* %1632, i32 %1629		; <i8*> [#uses=1]
	%1634 = load i8* %1633, align 1		; <i8> [#uses=1]
	%1635 = add i32 %1629, 1		; <i32> [#uses=1]
	store i32 %1635, i32* %1626, align 8
	%1636 = zext i8 %1634 to i32		; <i32> [#uses=1]
	%1637 = shl i32 %bsBuff.tmp.0213, 8		; <i32> [#uses=1]
	%1638 = or i32 %1637, %1636		; <i32> [#uses=1]
	%1639 = add i32 %bsLive.tmp.0214, 8		; <i32> [#uses=3]
	br label %bb3.i41.i.i.i

bb3.i41.i.i.i:		; preds = %bb2.i40.i.i.i, %makeMaps.exit.i.i.i
	%bsLive.tmp.0214 = phi i32 [ %bsLive.tmp.1, %makeMaps.exit.i.i.i ], [ %1639, %bb2.i40.i.i.i ]		; <i32> [#uses=3]
	%bsBuff.tmp.0213 = phi i32 [ %bsBuff.tmp.1, %makeMaps.exit.i.i.i ], [ %1638, %bb2.i40.i.i.i ]		; <i32> [#uses=6]
	%1640 = phi i32 [ %1639, %bb2.i40.i.i.i ], [ %bsLive.tmp.1, %makeMaps.exit.i.i.i ]		; <i32> [#uses=1]
	%1641 = phi i32 [ %1639, %bb2.i40.i.i.i ], [ %bsLive.tmp.1, %makeMaps.exit.i.i.i ]		; <i32> [#uses=1]
	%1642 = icmp slt i32 %1641, 3		; <i1> [#uses=1]
	br i1 %1642, label %bb3.i32.i.i.i, label %bsR.exit42.i.i.i

bsR.exit42.i.i.i:		; preds = %bb3.i41.i.i.i
	store i32 %bsBuff.tmp.0213, i32* @bsBuff
	store i32 %bsLive.tmp.0214, i32* @bsLive
	%1643 = add i32 %1640, -3		; <i32> [#uses=3]
	%1644 = lshr i32 %bsBuff.tmp.0213, %1643		; <i32> [#uses=1]
	%1645 = and i32 %1644, 7		; <i32> [#uses=5]
	br label %bb3.i35.i.i.i

bb3.i47.i.i.i:		; preds = %bb3.i35.i.i.i
	%1646 = load i32* %1626, align 8		; <i32> [#uses=3]
	%1647 = load i32* %1627, align 4		; <i32> [#uses=1]
	%1648 = icmp slt i32 %1646, %1647		; <i1> [#uses=1]
	br i1 %1648, label %bb2.i34.i.i.i, label %bb1.i33.i.i.i

bb1.i33.i.i.i:		; preds = %bb3.i47.i.i.i
	store i32 %1658, i32* @bsLive
	store i32 %bsBuff.tmp.0217, i32* @bsBuff
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i34.i.i.i:		; preds = %bb3.i47.i.i.i
	%1649 = load i8** %1628, align 4		; <i8*> [#uses=1]
	%1650 = getelementptr i8* %1649, i32 %1646		; <i8*> [#uses=1]
	%1651 = load i8* %1650, align 1		; <i8> [#uses=1]
	%1652 = add i32 %1646, 1		; <i32> [#uses=1]
	store i32 %1652, i32* %1626, align 8
	%1653 = zext i8 %1651 to i32		; <i32> [#uses=1]
	%1654 = shl i32 %bsBuff.tmp.0217, 8		; <i32> [#uses=1]
	%1655 = or i32 %1654, %1653		; <i32> [#uses=2]
	%1656 = add i32 %1658, 8		; <i32> [#uses=2]
	br label %bb3.i35.i.i.i

bb3.i35.i.i.i:		; preds = %bb2.i34.i.i.i, %bsR.exit42.i.i.i
	%bsBuff.tmp.0217 = phi i32 [ %bsBuff.tmp.0213, %bsR.exit42.i.i.i ], [ %1655, %bb2.i34.i.i.i ]		; <i32> [#uses=4]
	%1657 = phi i32 [ %1655, %bb2.i34.i.i.i ], [ %bsBuff.tmp.0213, %bsR.exit42.i.i.i ]		; <i32> [#uses=2]
	%1658 = phi i32 [ %1643, %bsR.exit42.i.i.i ], [ %1656, %bb2.i34.i.i.i ]		; <i32> [#uses=3]
	%1659 = phi i32 [ %1656, %bb2.i34.i.i.i ], [ %1643, %bsR.exit42.i.i.i ]		; <i32> [#uses=1]
	%1660 = icmp slt i32 %1659, 15		; <i1> [#uses=1]
	br i1 %1660, label %bb3.i47.i.i.i, label %bsR.exit36.i.i.i

bsR.exit36.i.i.i:		; preds = %bb3.i35.i.i.i
	store i32 %bsBuff.tmp.0217, i32* @bsBuff
	%1661 = add i32 %1658, -15		; <i32> [#uses=4]
	%1662 = lshr i32 %1657, %1661		; <i32> [#uses=1]
	%1663 = and i32 %1662, 32767		; <i32> [#uses=4]
	store i32 %1661, i32* @bsLive, align 4
	br label %bb22.i.i.i

bb19.i.i.i:		; preds = %bsR.exit30.i.i.i
	%1664 = add i32 %.ph178, 1		; <i32> [#uses=1]
	br label %bb3.i29.i.i.i

bb3.i61.i.i.i:		; preds = %bb3.i29.i.i.i
	%1665 = load i32* %1626, align 8		; <i32> [#uses=3]
	%1666 = load i32* %1627, align 4		; <i32> [#uses=1]
	%1667 = icmp slt i32 %1665, %1666		; <i1> [#uses=1]
	br i1 %1667, label %bb2.i28.i.i.i, label %bb1.i27.i.i.i

bb1.i27.i.i.i:		; preds = %bb3.i61.i.i.i
	store i32 %bsBuff.tmp.0220, i32* @bsBuff
	store i32 %bsLive.tmp.0221, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i28.i.i.i:		; preds = %bb3.i61.i.i.i
	%1668 = load i8** %1628, align 4		; <i8*> [#uses=1]
	%1669 = getelementptr i8* %1668, i32 %1665		; <i8*> [#uses=1]
	%1670 = load i8* %1669, align 1		; <i8> [#uses=1]
	%1671 = add i32 %1665, 1		; <i32> [#uses=1]
	store i32 %1671, i32* %1626, align 8
	%1672 = zext i8 %1670 to i32		; <i32> [#uses=1]
	%1673 = shl i32 %bsBuff.tmp.0220, 8		; <i32> [#uses=1]
	%1674 = or i32 %1673, %1672		; <i32> [#uses=2]
	%1675 = add i32 %bsLive.tmp.0221, 8		; <i32> [#uses=3]
	br label %bb3.i29.i.i.i

bb3.i29.i.i.i:		; preds = %bb22.i.i.i, %bb2.i28.i.i.i, %bb19.i.i.i
	%.ph178 = phi i32 [ %1664, %bb19.i.i.i ], [ %.ph178, %bb2.i28.i.i.i ], [ 0, %bb22.i.i.i ]		; <i32> [#uses=3]
	%bsLive.tmp.0221 = phi i32 [ %1675, %bb2.i28.i.i.i ], [ %1680, %bb19.i.i.i ], [ %bsLive.tmp.0229, %bb22.i.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0220 = phi i32 [ %1674, %bb2.i28.i.i.i ], [ %bsBuff.tmp.0220, %bb19.i.i.i ], [ %bsBuff.tmp.0228, %bb22.i.i.i ]		; <i32> [#uses=4]
	%1676 = phi i32 [ %1675, %bb2.i28.i.i.i ], [ %1680, %bb19.i.i.i ], [ %.pre.i25.rle108.i.i.i, %bb22.i.i.i ]		; <i32> [#uses=1]
	%1677 = phi i32 [ %1674, %bb2.i28.i.i.i ], [ %1677, %bb19.i.i.i ], [ %.rle110.i.i.i, %bb22.i.i.i ]		; <i32> [#uses=3]
	%1678 = phi i32 [ %1675, %bb2.i28.i.i.i ], [ %1680, %bb19.i.i.i ], [ %.pre.i25.rle108.i.i.i, %bb22.i.i.i ]		; <i32> [#uses=1]
	%1679 = icmp slt i32 %1678, 1		; <i1> [#uses=1]
	br i1 %1679, label %bb3.i61.i.i.i, label %bsR.exit30.i.i.i

bsR.exit30.i.i.i:		; preds = %bb3.i29.i.i.i
	%1680 = add i32 %1676, -1		; <i32> [#uses=6]
	%tmp.i.i.i = shl i32 1, %1680		; <i32> [#uses=1]
	%1681 = and i32 %tmp.i.i.i, %1677		; <i32> [#uses=1]
	%1682 = icmp eq i32 %1681, 0		; <i1> [#uses=1]
	br i1 %1682, label %bb21.i.i.i, label %bb19.i.i.i

bb21.i.i.i:		; preds = %bsR.exit30.i.i.i
	%1683 = trunc i32 %.ph178 to i8		; <i8> [#uses=1]
	store i8 %1683, i8* %scevgep95.i.i.i, align 1
	%1684 = add i32 %1685, 1		; <i32> [#uses=1]
	br label %bb22.i.i.i

bb22.i.i.i:		; preds = %bb21.i.i.i, %bsR.exit36.i.i.i
	%bsLive.tmp.0229 = phi i32 [ %1661, %bsR.exit36.i.i.i ], [ %1680, %bb21.i.i.i ]		; <i32> [#uses=3]
	%bsBuff.tmp.0228 = phi i32 [ %bsBuff.tmp.0217, %bsR.exit36.i.i.i ], [ %bsBuff.tmp.0220, %bb21.i.i.i ]		; <i32> [#uses=3]
	%.rle110.i.i.i = phi i32 [ %1677, %bb21.i.i.i ], [ %1657, %bsR.exit36.i.i.i ]		; <i32> [#uses=2]
	%.pre.i25.rle108.i.i.i = phi i32 [ %1680, %bb21.i.i.i ], [ %1661, %bsR.exit36.i.i.i ]		; <i32> [#uses=3]
	%1685 = phi i32 [ 0, %bsR.exit36.i.i.i ], [ %1684, %bb21.i.i.i ]		; <i32> [#uses=3]
	%scevgep95.i.i.i = getelementptr [18002 x i8]* @selectorMtf, i32 0, i32 %1685		; <i8*> [#uses=1]
	%1686 = icmp slt i32 %1685, %1663		; <i1> [#uses=1]
	br i1 %1686, label %bb3.i29.i.i.i, label %bb25.loopexit.i.i.i

bb24.i.i.i:		; preds = %bb25.loopexit.i.i.i, %bb24.i.i.i
	%1687 = phi i8 [ %1690, %bb24.i.i.i ], [ 0, %bb25.loopexit.i.i.i ]		; <i8> [#uses=3]
	%1688 = zext i8 %1687 to i32		; <i32> [#uses=1]
	%1689 = getelementptr [6 x i8]* %pos.i.i.i, i32 0, i32 %1688		; <i8*> [#uses=1]
	store i8 %1687, i8* %1689, align 1
	%1690 = add i8 %1687, 1		; <i8> [#uses=2]
	%phitmp.i.i.i = zext i8 %1690 to i32		; <i32> [#uses=1]
	%1691 = icmp ult i32 %phitmp.i.i.i, %1645		; <i1> [#uses=1]
	br i1 %1691, label %bb24.i.i.i, label %bb31.loopexit.i.i.i

bb25.loopexit.i.i.i:		; preds = %bb22.i.i.i
	store i32 %bsBuff.tmp.0228, i32* @bsBuff
	store i32 %bsLive.tmp.0229, i32* @bsLive
	%1692 = icmp eq i32 %1645, 0		; <i1> [#uses=2]
	br i1 %1692, label %bb31.loopexit.i.i.i, label %bb24.i.i.i

bb.nph62.i.i.i:		; preds = %bb31.loopexit.i.i.i
	%tmp88.i.i.i = icmp ugt i32 %1663, 1		; <i1> [#uses=1]
	%smax89.i.i.i = select i1 %tmp88.i.i.i, i32 %1663, i32 1		; <i32> [#uses=1]
	br label %bb27.i.i.i

bb27.i.i.i:		; preds = %bb30.i.i.i, %bb.nph62.i.i.i
	%i.461.i.i.i = phi i32 [ 0, %bb.nph62.i.i.i ], [ %1703, %bb30.i.i.i ]		; <i32> [#uses=3]
	%scevgep91.i.i.i = getelementptr [18002 x i8]* @selectorMtf, i32 0, i32 %i.461.i.i.i		; <i8*> [#uses=1]
	%scevgep92.i.i.i = getelementptr [18002 x i8]* @selector, i32 0, i32 %i.461.i.i.i		; <i8*> [#uses=1]
	%1693 = load i8* %scevgep91.i.i.i, align 1		; <i8> [#uses=4]
	%1694 = zext i8 %1693 to i32		; <i32> [#uses=1]
	%1695 = getelementptr [6 x i8]* %pos.i.i.i, i32 0, i32 %1694		; <i8*> [#uses=1]
	%1696 = load i8* %1695, align 1		; <i8> [#uses=2]
	%1697 = icmp eq i8 %1693, 0		; <i1> [#uses=1]
	br i1 %1697, label %bb30.i.i.i, label %bb28.i.i.i

bb28.i.i.i:		; preds = %bb28.i.i.i, %bb27.i.i.i
	%indvar.i.i.i = phi i8 [ %indvar.next.i.i.i, %bb28.i.i.i ], [ 0, %bb27.i.i.i ]		; <i8> [#uses=2]
	%v.159.i.i.i = sub i8 %1693, %indvar.i.i.i		; <i8> [#uses=1]
	%1698 = zext i8 %v.159.i.i.i to i32		; <i32> [#uses=2]
	%1699 = add i32 %1698, -1		; <i32> [#uses=1]
	%1700 = getelementptr [6 x i8]* %pos.i.i.i, i32 0, i32 %1699		; <i8*> [#uses=1]
	%1701 = load i8* %1700, align 1		; <i8> [#uses=1]
	%1702 = getelementptr [6 x i8]* %pos.i.i.i, i32 0, i32 %1698		; <i8*> [#uses=1]
	store i8 %1701, i8* %1702, align 1
	%indvar.next.i.i.i = add i8 %indvar.i.i.i, 1		; <i8> [#uses=2]
	%exitcond83.i.i.i = icmp eq i8 %indvar.next.i.i.i, %1693		; <i1> [#uses=1]
	br i1 %exitcond83.i.i.i, label %bb30.i.i.i, label %bb28.i.i.i

bb30.i.i.i:		; preds = %bb28.i.i.i, %bb27.i.i.i
	store i8 %1696, i8* %84, align 1
	store i8 %1696, i8* %scevgep92.i.i.i, align 1
	%1703 = add i32 %i.461.i.i.i, 1		; <i32> [#uses=2]
	%exitcond90.i.i.i = icmp eq i32 %1703, %smax89.i.i.i		; <i1> [#uses=1]
	br i1 %exitcond90.i.i.i, label %bb41.i.i.i.preheader, label %bb27.i.i.i

bb31.loopexit.i.i.i:		; preds = %bb25.loopexit.i.i.i, %bb24.i.i.i
	%1704 = icmp eq i32 %1663, 0		; <i1> [#uses=1]
	br i1 %1704, label %bb41.i.i.i.preheader, label %bb.nph62.i.i.i

bb41.i.i.i.preheader:		; preds = %bb31.loopexit.i.i.i, %bb30.i.i.i
	%1705 = zext i1 %.b.i to i32		; <i32> [#uses=3]
	%1706 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1705, i32 2		; <i32*> [#uses=6]
	%1707 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1705, i32 1		; <i32*> [#uses=3]
	%1708 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1705, i32 3		; <i8**> [#uses=3]
	br label %bb41.i.i.i

bb3.i54.i.i.i:		; preds = %bb3.i23.i.i.i
	%1709 = load i32* %1706, align 8		; <i32> [#uses=3]
	%1710 = load i32* %1707, align 4		; <i32> [#uses=1]
	%1711 = icmp slt i32 %1709, %1710		; <i1> [#uses=1]
	br i1 %1711, label %bb2.i22.i.i.i, label %bb1.i21.i.i.i

bb1.i21.i.i.i:		; preds = %bb3.i54.i.i.i
	store i32 %bsBuff.tmp.0232, i32* @bsBuff
	store i32 %bsLive.tmp.0233, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i22.i.i.i:		; preds = %bb3.i54.i.i.i
	%1712 = load i8** %1708, align 4		; <i8*> [#uses=1]
	%1713 = getelementptr i8* %1712, i32 %1709		; <i8*> [#uses=1]
	%1714 = load i8* %1713, align 1		; <i8> [#uses=1]
	%1715 = add i32 %1709, 1		; <i32> [#uses=1]
	store i32 %1715, i32* %1706, align 8
	%1716 = zext i8 %1714 to i32		; <i32> [#uses=1]
	%1717 = shl i32 %bsBuff.tmp.0232, 8		; <i32> [#uses=1]
	%1718 = or i32 %1717, %1716		; <i32> [#uses=2]
	%1719 = add i32 %bsLive.tmp.0233, 8		; <i32> [#uses=3]
	br label %bb3.i23.i.i.i

bb3.i23.i.i.i:		; preds = %bb41.i.i.i, %bb2.i22.i.i.i
	%bsLive.tmp.0233 = phi i32 [ %1719, %bb2.i22.i.i.i ], [ %bsLive.tmp.0253, %bb41.i.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0232 = phi i32 [ %1718, %bb2.i22.i.i.i ], [ %bsBuff.tmp.0252, %bb41.i.i.i ]		; <i32> [#uses=3]
	%1720 = phi i32 [ %1719, %bb2.i22.i.i.i ], [ %.pre.i19.i.i.i, %bb41.i.i.i ]		; <i32> [#uses=1]
	%1721 = phi i32 [ %1718, %bb2.i22.i.i.i ], [ %.rle118.i.i.i, %bb41.i.i.i ]		; <i32> [#uses=2]
	%1722 = phi i32 [ %1719, %bb2.i22.i.i.i ], [ %.pre.i19.i.i.i, %bb41.i.i.i ]		; <i32> [#uses=1]
	%1723 = icmp slt i32 %1722, 5		; <i1> [#uses=1]
	br i1 %1723, label %bb3.i54.i.i.i, label %bsR.exit24.i.i.i

bsR.exit24.i.i.i:		; preds = %bb3.i23.i.i.i
	%1724 = add i32 %1720, -5		; <i32> [#uses=3]
	%1725 = lshr i32 %1721, %1724		; <i32> [#uses=1]
	%1726 = and i32 %1725, 31		; <i32> [#uses=1]
	br label %bb39.i.i.i

bb3.i39.i.i.i:		; preds = %bb3.i17.i.i.i
	%1727 = load i32* %1706, align 8		; <i32> [#uses=3]
	%1728 = load i32* %1707, align 4		; <i32> [#uses=1]
	%1729 = icmp slt i32 %1727, %1728		; <i1> [#uses=1]
	br i1 %1729, label %bb2.i16.i.i.i, label %bb1.i15.i.i.i

bb1.i15.i.i.i:		; preds = %bb3.i39.i.i.i
	store i32 %bsBuff.tmp.0236, i32* @bsBuff
	store i32 %bsLive.tmp.0237, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i16.i.i.i:		; preds = %bb3.i39.i.i.i
	%1730 = load i8** %1708, align 4		; <i8*> [#uses=1]
	%1731 = getelementptr i8* %1730, i32 %1727		; <i8*> [#uses=1]
	%1732 = load i8* %1731, align 1		; <i8> [#uses=1]
	%1733 = add i32 %1727, 1		; <i32> [#uses=1]
	store i32 %1733, i32* %1706, align 8
	%1734 = zext i8 %1732 to i32		; <i32> [#uses=1]
	%1735 = shl i32 %bsBuff.tmp.0236, 8		; <i32> [#uses=1]
	%1736 = or i32 %1735, %1734		; <i32> [#uses=2]
	%1737 = add i32 %bsLive.tmp.0237, 8		; <i32> [#uses=3]
	br label %bb3.i17.i.i.i

bb3.i17.i.i.i:		; preds = %bsR.exit12.i.i.i, %bb2.i16.i.i.i
	%bsLive.tmp.0237 = phi i32 [ %1737, %bb2.i16.i.i.i ], [ %1760, %bsR.exit12.i.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0236 = phi i32 [ %1736, %bb2.i16.i.i.i ], [ %bsBuff.tmp.0240, %bsR.exit12.i.i.i ]		; <i32> [#uses=3]
	%1738 = phi i32 [ %1737, %bb2.i16.i.i.i ], [ %1760, %bsR.exit12.i.i.i ]		; <i32> [#uses=1]
	%1739 = phi i32 [ %1736, %bb2.i16.i.i.i ], [ %1757, %bsR.exit12.i.i.i ]		; <i32> [#uses=2]
	%1740 = phi i32 [ %1737, %bb2.i16.i.i.i ], [ %1760, %bsR.exit12.i.i.i ]		; <i32> [#uses=1]
	%1741 = icmp slt i32 %1740, 1		; <i1> [#uses=1]
	br i1 %1741, label %bb3.i39.i.i.i, label %bsR.exit18.i.i.i

bsR.exit18.i.i.i:		; preds = %bb3.i17.i.i.i
	%1742 = add i32 %1738, -1		; <i32> [#uses=4]
	%tmp46.i.i.i = shl i32 1, %1742		; <i32> [#uses=1]
	%1743 = and i32 %tmp46.i.i.i, %1739		; <i32> [#uses=1]
	%1744 = icmp eq i32 %1743, 0		; <i1> [#uses=1]
	%curr.0.be.v.i.i.i = select i1 %1744, i32 1, i32 -1		; <i32> [#uses=1]
	%curr.0.be.i.i.i = add i32 %curr.0.i.i.i.ph, %curr.0.be.v.i.i.i		; <i32> [#uses=1]
	br label %bb3.i11.i.i.i

bb3.i25.i.i.i:		; preds = %bb3.i11.i.i.i
	%1745 = load i32* %1706, align 8		; <i32> [#uses=3]
	%1746 = load i32* %1707, align 4		; <i32> [#uses=1]
	%1747 = icmp slt i32 %1745, %1746		; <i1> [#uses=1]
	br i1 %1747, label %bb2.i10.i.i.i, label %bb1.i9.i.i.i

bb1.i9.i.i.i:		; preds = %bb3.i25.i.i.i
	store i32 %bsBuff.tmp.0240, i32* @bsBuff
	store i32 %bsLive.tmp.0241, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i10.i.i.i:		; preds = %bb3.i25.i.i.i
	%1748 = load i8** %1708, align 4		; <i8*> [#uses=1]
	%1749 = getelementptr i8* %1748, i32 %1745		; <i8*> [#uses=1]
	%1750 = load i8* %1749, align 1		; <i8> [#uses=1]
	%1751 = add i32 %1745, 1		; <i32> [#uses=1]
	store i32 %1751, i32* %1706, align 8
	%1752 = zext i8 %1750 to i32		; <i32> [#uses=1]
	%1753 = shl i32 %bsBuff.tmp.0240, 8		; <i32> [#uses=1]
	%1754 = or i32 %1753, %1752		; <i32> [#uses=2]
	%1755 = add i32 %bsLive.tmp.0241, 8		; <i32> [#uses=3]
	br label %bb3.i11.i.i.i

bb3.i11.i.i.i:		; preds = %bb39.i.i.i, %bb2.i10.i.i.i, %bsR.exit18.i.i.i
	%curr.0.i.i.i.ph = phi i32 [ %curr.0.be.i.i.i, %bsR.exit18.i.i.i ], [ %curr.0.i.i.i.ph, %bb2.i10.i.i.i ], [ %curr.1.i.i.i, %bb39.i.i.i ]		; <i32> [#uses=4]
	%bsLive.tmp.0241 = phi i32 [ %1755, %bb2.i10.i.i.i ], [ %1742, %bsR.exit18.i.i.i ], [ %bsLive.tmp.0249, %bb39.i.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0240 = phi i32 [ %1754, %bb2.i10.i.i.i ], [ %bsBuff.tmp.0236, %bsR.exit18.i.i.i ], [ %bsBuff.tmp.0248, %bb39.i.i.i ]		; <i32> [#uses=4]
	%1756 = phi i32 [ %1755, %bb2.i10.i.i.i ], [ %1742, %bsR.exit18.i.i.i ], [ %.pre.i19.rle117.i.i.i, %bb39.i.i.i ]		; <i32> [#uses=1]
	%1757 = phi i32 [ %1754, %bb2.i10.i.i.i ], [ %1739, %bsR.exit18.i.i.i ], [ %.rle125.i.i.i, %bb39.i.i.i ]		; <i32> [#uses=3]
	%1758 = phi i32 [ %1755, %bb2.i10.i.i.i ], [ %1742, %bsR.exit18.i.i.i ], [ %.pre.i19.rle117.i.i.i, %bb39.i.i.i ]		; <i32> [#uses=1]
	%1759 = icmp slt i32 %1758, 1		; <i1> [#uses=1]
	br i1 %1759, label %bb3.i25.i.i.i, label %bsR.exit12.i.i.i

bsR.exit12.i.i.i:		; preds = %bb3.i11.i.i.i
	%1760 = add i32 %1756, -1		; <i32> [#uses=6]
	%tmp47.i.i.i = shl i32 1, %1760		; <i32> [#uses=1]
	%1761 = and i32 %tmp47.i.i.i, %1757		; <i32> [#uses=1]
	%1762 = icmp eq i32 %1761, 0		; <i1> [#uses=1]
	br i1 %1762, label %bb38.i.i.i, label %bb3.i17.i.i.i

bb38.i.i.i:		; preds = %bsR.exit12.i.i.i
	%1763 = trunc i32 %curr.0.i.i.i.ph to i8		; <i8> [#uses=1]
	store i8 %1763, i8* %scevgep82.i.i.i, align 1
	%1764 = add i32 %1765, 1		; <i32> [#uses=1]
	br label %bb39.i.i.i

bb39.i.i.i:		; preds = %bb38.i.i.i, %bsR.exit24.i.i.i
	%bsLive.tmp.0249 = phi i32 [ %1724, %bsR.exit24.i.i.i ], [ %1760, %bb38.i.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0248 = phi i32 [ %bsBuff.tmp.0232, %bsR.exit24.i.i.i ], [ %bsBuff.tmp.0240, %bb38.i.i.i ]		; <i32> [#uses=2]
	%.rle125.i.i.i = phi i32 [ %1757, %bb38.i.i.i ], [ %1721, %bsR.exit24.i.i.i ]		; <i32> [#uses=2]
	%.pre.i19.rle117.i.i.i = phi i32 [ %1760, %bb38.i.i.i ], [ %1724, %bsR.exit24.i.i.i ]		; <i32> [#uses=3]
	%1765 = phi i32 [ 0, %bsR.exit24.i.i.i ], [ %1764, %bb38.i.i.i ]		; <i32> [#uses=3]
	%curr.1.i.i.i = phi i32 [ %1726, %bsR.exit24.i.i.i ], [ %curr.0.i.i.i.ph, %bb38.i.i.i ]		; <i32> [#uses=1]
	%scevgep82.i.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 %1768, i32 %1765		; <i8*> [#uses=1]
	%1766 = icmp slt i32 %1765, %1624		; <i1> [#uses=1]
	br i1 %1766, label %bb3.i11.i.i.i, label %bb40.i.i.i

bb40.i.i.i:		; preds = %bb39.i.i.i
	%1767 = add i32 %1768, 1		; <i32> [#uses=1]
	br label %bb41.i.i.i

bb41.i.i.i:		; preds = %bb40.i.i.i, %bb41.i.i.i.preheader
	%bsLive.tmp.0253 = phi i32 [ %bsLive.tmp.0229, %bb41.i.i.i.preheader ], [ %bsLive.tmp.0249, %bb40.i.i.i ]		; <i32> [#uses=5]
	%bsBuff.tmp.0252 = phi i32 [ %bsBuff.tmp.0228, %bb41.i.i.i.preheader ], [ %bsBuff.tmp.0248, %bb40.i.i.i ]		; <i32> [#uses=3]
	%.rle118.i.i.i = phi i32 [ %.rle125.i.i.i, %bb40.i.i.i ], [ %.rle110.i.i.i, %bb41.i.i.i.preheader ]		; <i32> [#uses=1]
	%.pre.i19.i.i.i = phi i32 [ %.pre.i19.rle117.i.i.i, %bb40.i.i.i ], [ %.pre.i25.rle108.i.i.i, %bb41.i.i.i.preheader ]		; <i32> [#uses=2]
	%1768 = phi i32 [ %1767, %bb40.i.i.i ], [ 0, %bb41.i.i.i.preheader ]		; <i32> [#uses=3]
	%1769 = icmp slt i32 %1768, %1645		; <i1> [#uses=1]
	br i1 %1769, label %bb3.i23.i.i.i, label %bb51.loopexit.i.i.i

bb44.i.i.i:		; preds = %bb49.preheader.i.i.i, %bb44.i.i.i
	%i.652.i.i.i = phi i32 [ %1774, %bb44.i.i.i ], [ 0, %bb49.preheader.i.i.i ]		; <i32> [#uses=2]
	%maxLen.151.i.i.i = phi i32 [ %maxLen.0.i.i.i, %bb44.i.i.i ], [ 0, %bb49.preheader.i.i.i ]		; <i32> [#uses=2]
	%minLen.150.i.i.i = phi i32 [ %minLen.0.i.i.i, %bb44.i.i.i ], [ 32, %bb49.preheader.i.i.i ]		; <i32> [#uses=2]
	%scevgep.i.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 %1809, i32 %i.652.i.i.i		; <i8*> [#uses=1]
	%1770 = load i8* %scevgep.i.i.i, align 1		; <i8> [#uses=1]
	%1771 = zext i8 %1770 to i32		; <i32> [#uses=4]
	%1772 = icmp sgt i32 %1771, %maxLen.151.i.i.i		; <i1> [#uses=1]
	%maxLen.0.i.i.i = select i1 %1772, i32 %1771, i32 %maxLen.151.i.i.i		; <i32> [#uses=2]
	%1773 = icmp slt i32 %1771, %minLen.150.i.i.i		; <i1> [#uses=1]
	%minLen.0.i.i.i = select i1 %1773, i32 %1771, i32 %minLen.150.i.i.i		; <i32> [#uses=2]
	%1774 = add i32 %i.652.i.i.i, 1		; <i32> [#uses=2]
	%exitcond.i.i.i = icmp eq i32 %1774, %1624		; <i1> [#uses=1]
	br i1 %exitcond.i.i.i, label %bb50.i.i.i, label %bb44.i.i.i

bb50.i.i.i:		; preds = %bb49.preheader.i.i.i, %bb44.i.i.i
	%maxLen.1.lcssa.i.i.i = phi i32 [ 0, %bb49.preheader.i.i.i ], [ %maxLen.0.i.i.i, %bb44.i.i.i ]		; <i32> [#uses=5]
	%minLen.1.lcssa.i.i.i = phi i32 [ 32, %bb49.preheader.i.i.i ], [ %minLen.0.i.i.i, %bb44.i.i.i ]		; <i32> [#uses=9]
	%1775 = icmp sgt i32 %minLen.1.lcssa.i.i.i, %maxLen.1.lcssa.i.i.i		; <i1> [#uses=2]
	%.not.i.i.i.i = xor i1 %1775, true		; <i1> [#uses=1]
	%or.cond.i.i.i.i = and i1 %1808, %.not.i.i.i.i		; <i1> [#uses=1]
	br i1 %or.cond.i.i.i.i, label %bb.nph21.split.i.i.i.i, label %bb9.loopexit.i.i.i.i

bb1.i11.i.i.i:		; preds = %bb4.preheader.i.i.i.i, %bb3.i13.i.i.i
	%pp.116.i.i.i.i = phi i32 [ %pp.220.i.i.i.i, %bb4.preheader.i.i.i.i ], [ %pp.0.i.i.i.i, %bb3.i13.i.i.i ]		; <i32> [#uses=3]
	%1776 = phi i32 [ 0, %bb4.preheader.i.i.i.i ], [ %1782, %bb3.i13.i.i.i ]		; <i32> [#uses=3]
	%scevgep50.i.i.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 %1809, i32 %1776		; <i8*> [#uses=1]
	%1777 = load i8* %scevgep50.i.i.i.i, align 1		; <i8> [#uses=1]
	%1778 = zext i8 %1777 to i32		; <i32> [#uses=1]
	%1779 = icmp eq i32 %1778, %i.019.i.i.i.i		; <i1> [#uses=1]
	br i1 %1779, label %bb2.i12.i.i.i, label %bb3.i13.i.i.i

bb2.i12.i.i.i:		; preds = %bb1.i11.i.i.i
	%1780 = getelementptr [6 x [258 x i32]]* @perm, i32 0, i32 %1809, i32 %pp.116.i.i.i.i		; <i32*> [#uses=1]
	store i32 %1776, i32* %1780, align 4
	%1781 = add i32 %pp.116.i.i.i.i, 1		; <i32> [#uses=1]
	br label %bb3.i13.i.i.i

bb3.i13.i.i.i:		; preds = %bb2.i12.i.i.i, %bb1.i11.i.i.i
	%pp.0.i.i.i.i = phi i32 [ %1781, %bb2.i12.i.i.i ], [ %pp.116.i.i.i.i, %bb1.i11.i.i.i ]		; <i32> [#uses=2]
	%1782 = add i32 %1776, 1		; <i32> [#uses=2]
	%exitcond49.i.i.i.i = icmp eq i32 %1782, %1624		; <i1> [#uses=1]
	br i1 %exitcond49.i.i.i.i, label %bb5.i.i.i.i, label %bb1.i11.i.i.i

bb5.i.i.i.i:		; preds = %bb3.i13.i.i.i
	%1783 = icmp sgt i32 %tmp55.i.i.i.i, %maxLen.1.lcssa.i.i.i		; <i1> [#uses=1]
	%indvar.next52.i.i.i.i = add i32 %indvar51.i.i.i.i, 1		; <i32> [#uses=1]
	br i1 %1783, label %bb9.loopexit.i.i.i.i, label %bb4.preheader.i.i.i.i

bb.nph21.split.i.i.i.i:		; preds = %bb50.i.i.i
	%tmp54.i.i.i.i = add i32 %minLen.1.lcssa.i.i.i, 1		; <i32> [#uses=1]
	br label %bb4.preheader.i.i.i.i

bb4.preheader.i.i.i.i:		; preds = %bb.nph21.split.i.i.i.i, %bb5.i.i.i.i
	%indvar51.i.i.i.i = phi i32 [ 0, %bb.nph21.split.i.i.i.i ], [ %indvar.next52.i.i.i.i, %bb5.i.i.i.i ]		; <i32> [#uses=3]
	%pp.220.i.i.i.i = phi i32 [ 0, %bb.nph21.split.i.i.i.i ], [ %pp.0.i.i.i.i, %bb5.i.i.i.i ]		; <i32> [#uses=1]
	%i.019.i.i.i.i = add i32 %indvar51.i.i.i.i, %minLen.1.lcssa.i.i.i		; <i32> [#uses=1]
	%tmp55.i.i.i.i = add i32 %indvar51.i.i.i.i, %tmp54.i.i.i.i		; <i32> [#uses=1]
	br label %bb1.i11.i.i.i

bb9.loopexit.i.i.i.i:		; preds = %bb5.i.i.i.i, %bb50.i.i.i
	store i32 0, i32* %scevgep77.i.i.i, align 8
	%scevgep48.1.i.i.i.i = getelementptr [6 x [258 x i32]]* @base, i32 0, i32 %1809, i32 1		; <i32*> [#uses=1]
	%scevgep48.156.i.i.i.i = bitcast i32* %scevgep48.1.i.i.i.i to i8*		; <i8*> [#uses=1]
	call void @llvm.memset.i64(i8* %scevgep48.156.i.i.i.i, i8 0, i64 88, i32 4) nounwind
	br i1 %1808, label %bb11.i.i.i.i, label %bb14.i.i.i.i

bb11.i.i.i.i:		; preds = %bb11.i.i.i.i, %bb9.loopexit.i.i.i.i
	%i.211.i.i.i.i = phi i32 [ %1790, %bb11.i.i.i.i ], [ 0, %bb9.loopexit.i.i.i.i ]		; <i32> [#uses=2]
	%scevgep46.i.i.i.i = getelementptr [6 x [258 x i8]]* @len, i32 0, i32 %1809, i32 %i.211.i.i.i.i		; <i8*> [#uses=1]
	%1784 = load i8* %scevgep46.i.i.i.i, align 1		; <i8> [#uses=1]
	%1785 = zext i8 %1784 to i32		; <i32> [#uses=1]
	%1786 = add i32 %1785, 1		; <i32> [#uses=1]
	%1787 = getelementptr [6 x [258 x i32]]* @base, i32 0, i32 %1809, i32 %1786		; <i32*> [#uses=2]
	%1788 = load i32* %1787, align 4		; <i32> [#uses=1]
	%1789 = add i32 %1788, 1		; <i32> [#uses=1]
	store i32 %1789, i32* %1787, align 4
	%1790 = add i32 %i.211.i.i.i.i, 1		; <i32> [#uses=2]
	%exitcond45.i.i.i.i = icmp eq i32 %1790, %1624		; <i1> [#uses=1]
	br i1 %exitcond45.i.i.i.i, label %bb14.i.i.i.i, label %bb11.i.i.i.i

bb14.i.i.i.i:		; preds = %bb14.i.i.i.i, %bb11.i.i.i.i, %bb9.loopexit.i.i.i.i
	%indvar39.i.i.i.i = phi i32 [ %tmp42.i.i.i.i, %bb14.i.i.i.i ], [ 0, %bb11.i.i.i.i ], [ 0, %bb9.loopexit.i.i.i.i ]		; <i32> [#uses=2]
	%tmp42.i.i.i.i = add i32 %indvar39.i.i.i.i, 1		; <i32> [#uses=3]
	%scevgep43.i.i.i.i = getelementptr [6 x [258 x i32]]* @base, i32 0, i32 %1809, i32 %tmp42.i.i.i.i		; <i32*> [#uses=2]
	%scevgep44.i.i.i.i = getelementptr [6 x [258 x i32]]* @base, i32 0, i32 %1809, i32 %indvar39.i.i.i.i		; <i32*> [#uses=1]
	%1791 = load i32* %scevgep43.i.i.i.i, align 4		; <i32> [#uses=1]
	%1792 = load i32* %scevgep44.i.i.i.i, align 4		; <i32> [#uses=1]
	%1793 = add i32 %1792, %1791		; <i32> [#uses=1]
	store i32 %1793, i32* %scevgep43.i.i.i.i, align 4
	%exitcond41.i.i.i.i = icmp eq i32 %tmp42.i.i.i.i, 22		; <i1> [#uses=1]
	br i1 %exitcond41.i.i.i.i, label %bb18.loopexit.i.i.i.i, label %bb14.i.i.i.i

bb18.loopexit.i.i.i.i:		; preds = %bb14.i.i.i.i
	store i32 0, i32* %scevgep76.i.i.i, align 8
	%scevgep38.1.i.i.i.i = getelementptr [6 x [258 x i32]]* @limit, i32 0, i32 %1809, i32 1		; <i32*> [#uses=1]
	%scevgep38.157.i.i.i.i = bitcast i32* %scevgep38.1.i.i.i.i to i8*		; <i8*> [#uses=1]
	call void @llvm.memset.i64(i8* %scevgep38.157.i.i.i.i, i8 0, i64 88, i32 4) nounwind
	br i1 %1775, label %bb24.loopexit.i.i.i.i, label %bb.nph6.i.i.i.i

bb.nph6.i.i.i.i:		; preds = %bb18.loopexit.i.i.i.i
	%tmp35.i.i.i.i = add i32 %minLen.1.lcssa.i.i.i, 1		; <i32> [#uses=1]
	br label %bb20.i.i.i.i

bb20.i.i.i.i:		; preds = %bb20.i.i.i.i, %bb.nph6.i.i.i.i
	%indvar29.i.i.i.i = phi i32 [ 0, %bb.nph6.i.i.i.i ], [ %indvar.next30.i.i.i.i, %bb20.i.i.i.i ]		; <i32> [#uses=3]
	%vec.04.i.i.i.i = phi i32 [ 0, %bb.nph6.i.i.i.i ], [ %1799, %bb20.i.i.i.i ]		; <i32> [#uses=1]
	%tmp31.i.i.i.i = add i32 %indvar29.i.i.i.i, %minLen.1.lcssa.i.i.i		; <i32> [#uses=2]
	%scevgep32.i.i.i.i = getelementptr [6 x [258 x i32]]* @base, i32 0, i32 %1809, i32 %tmp31.i.i.i.i		; <i32*> [#uses=1]
	%scevgep33.i.i.i.i = getelementptr [6 x [258 x i32]]* @limit, i32 0, i32 %1809, i32 %tmp31.i.i.i.i		; <i32*> [#uses=1]
	%scevgep34.sum.i.i.i.i = add i32 %indvar29.i.i.i.i, %tmp35.i.i.i.i		; <i32> [#uses=2]
	%scevgep36.i.i.i.i = getelementptr [6 x [258 x i32]]* @base, i32 0, i32 %1809, i32 %scevgep34.sum.i.i.i.i		; <i32*> [#uses=1]
	%1794 = load i32* %scevgep36.i.i.i.i, align 4		; <i32> [#uses=1]
	%1795 = load i32* %scevgep32.i.i.i.i, align 4		; <i32> [#uses=1]
	%1796 = sub i32 %1794, %1795		; <i32> [#uses=1]
	%1797 = add i32 %1796, %vec.04.i.i.i.i		; <i32> [#uses=2]
	%1798 = add i32 %1797, -1		; <i32> [#uses=1]
	store i32 %1798, i32* %scevgep33.i.i.i.i, align 4
	%1799 = shl i32 %1797, 1		; <i32> [#uses=1]
	%1800 = icmp sgt i32 %scevgep34.sum.i.i.i.i, %maxLen.1.lcssa.i.i.i		; <i1> [#uses=1]
	%indvar.next30.i.i.i.i = add i32 %indvar29.i.i.i.i, 1		; <i32> [#uses=1]
	br i1 %1800, label %bb24.loopexit.i.i.i.i, label %bb20.i.i.i.i

bb.nph.i.i.i.i:		; preds = %bb24.loopexit.i.i.i.i
	%tmp.i14.i.i.i = add i32 %minLen.1.lcssa.i.i.i, 2		; <i32> [#uses=1]
	br label %bb23.i.i.i.i

bb23.i.i.i.i:		; preds = %bb23.i.i.i.i, %bb.nph.i.i.i.i
	%indvar.i.i.i.i = phi i32 [ 0, %bb.nph.i.i.i.i ], [ %indvar.next.i.i.i.i, %bb23.i.i.i.i ]		; <i32> [#uses=4]
	%scevgep.sum.i.i.i.i = add i32 %indvar.i.i.i.i, %i.62.i.i.i.i		; <i32> [#uses=1]
	%scevgep26.i.i.i.i = getelementptr [6 x [258 x i32]]* @base, i32 0, i32 %1809, i32 %scevgep.sum.i.i.i.i		; <i32*> [#uses=2]
	%tmp27.i.i.i.i = add i32 %indvar.i.i.i.i, %minLen.1.lcssa.i.i.i		; <i32> [#uses=1]
	%scevgep28.i.i.i.i = getelementptr [6 x [258 x i32]]* @limit, i32 0, i32 %1809, i32 %tmp27.i.i.i.i		; <i32*> [#uses=1]
	%1801 = load i32* %scevgep28.i.i.i.i, align 4		; <i32> [#uses=1]
	%1802 = shl i32 %1801, 1		; <i32> [#uses=1]
	%1803 = load i32* %scevgep26.i.i.i.i, align 4		; <i32> [#uses=1]
	%sum.i.i.i.i = add i32 %1803, -2		; <i32> [#uses=1]
	%1804 = sub i32 %1802, %sum.i.i.i.i		; <i32> [#uses=1]
	store i32 %1804, i32* %scevgep26.i.i.i.i, align 4
	%i.6.i.i.i.i = add i32 %indvar.i.i.i.i, %tmp.i14.i.i.i		; <i32> [#uses=1]
	%1805 = icmp sgt i32 %i.6.i.i.i.i, %maxLen.1.lcssa.i.i.i		; <i1> [#uses=1]
	%indvar.next.i.i.i.i = add i32 %indvar.i.i.i.i, 1		; <i32> [#uses=1]
	br i1 %1805, label %hbCreateDecodeTables.exit.i.i.i, label %bb23.i.i.i.i

bb24.loopexit.i.i.i.i:		; preds = %bb20.i.i.i.i, %bb18.loopexit.i.i.i.i
	%i.62.i.i.i.i = add i32 %minLen.1.lcssa.i.i.i, 1		; <i32> [#uses=2]
	%1806 = icmp sgt i32 %i.62.i.i.i.i, %maxLen.1.lcssa.i.i.i		; <i1> [#uses=1]
	br i1 %1806, label %hbCreateDecodeTables.exit.i.i.i, label %bb.nph.i.i.i.i

hbCreateDecodeTables.exit.i.i.i:		; preds = %bb24.loopexit.i.i.i.i, %bb23.i.i.i.i
	store i32 %minLen.1.lcssa.i.i.i, i32* %scevgep75.i.i.i, align 4
	%1807 = add i32 %1809, 1		; <i32> [#uses=2]
	%exitcond74.i.i.i = icmp eq i32 %1807, %smax.i.i.i		; <i1> [#uses=1]
	br i1 %exitcond74.i.i.i, label %recvDecodingTables.exit.i.i, label %bb49.preheader.i.i.i

bb51.loopexit.i.i.i:		; preds = %bb41.i.i.i
	store i32 %bsBuff.tmp.0252, i32* @bsBuff
	store i32 %bsLive.tmp.0253, i32* @bsLive
	br i1 %1692, label %recvDecodingTables.exit.i.i, label %bb.nph56.i.i.i

bb.nph56.i.i.i:		; preds = %bb51.loopexit.i.i.i
	%1808 = icmp sgt i32 %1624, 0		; <i1> [#uses=3]
	%tmp73.i.i.i = icmp ugt i32 %1645, 1		; <i1> [#uses=1]
	%smax.i.i.i = select i1 %tmp73.i.i.i, i32 %1645, i32 1		; <i32> [#uses=1]
	br label %bb49.preheader.i.i.i

bb49.preheader.i.i.i:		; preds = %bb.nph56.i.i.i, %hbCreateDecodeTables.exit.i.i.i
	%1809 = phi i32 [ 0, %bb.nph56.i.i.i ], [ %1807, %hbCreateDecodeTables.exit.i.i.i ]		; <i32> [#uses=18]
	%scevgep75.i.i.i = getelementptr [6 x i32]* @minLens, i32 0, i32 %1809		; <i32*> [#uses=1]
	%scevgep76.i.i.i = getelementptr [6 x [258 x i32]]* @limit, i32 0, i32 %1809, i32 0		; <i32*> [#uses=1]
	%scevgep77.i.i.i = getelementptr [6 x [258 x i32]]* @base, i32 0, i32 %1809, i32 0		; <i32*> [#uses=1]
	br i1 %1808, label %bb44.i.i.i, label %bb50.i.i.i

recvDecodingTables.exit.i.i:		; preds = %bb51.loopexit.i.i.i, %hbCreateDecodeTables.exit.i.i.i
	%1810 = add i32 %nInUse.tmp.0.i.i.i.i, 1		; <i32> [#uses=1]
	br label %bb.i34.i

bb.i34.i:		; preds = %bb.i34.i, %recvDecodingTables.exit.i.i
	%i.039.i.i = phi i32 [ 0, %recvDecodingTables.exit.i.i ], [ %1811, %bb.i34.i ]		; <i32> [#uses=2]
	%scevgep119.i.i = getelementptr [256 x i32]* @unzftab, i32 0, i32 %i.039.i.i		; <i32*> [#uses=1]
	store i32 0, i32* %scevgep119.i.i, align 4
	%1811 = add i32 %i.039.i.i, 1		; <i32> [#uses=2]
	%exitcond118.i.i = icmp eq i32 %1811, 256		; <i1> [#uses=1]
	br i1 %exitcond118.i.i, label %bb13.i35.i, label %bb.i34.i

bb13.i35.i:		; preds = %bb13.i35.i, %bb.i34.i
	%i.137.i.i = phi i32 [ %1812, %bb13.i35.i ], [ 0, %bb.i34.i ]		; <i32> [#uses=3]
	%scevgep116.i.i = getelementptr [256 x i8]* %yy.i.i, i32 0, i32 %i.137.i.i		; <i8*> [#uses=1]
	%tmp117.i.i = trunc i32 %i.137.i.i to i8		; <i8> [#uses=1]
	store i8 %tmp117.i.i, i8* %scevgep116.i.i, align 1
	%1812 = add i32 %i.137.i.i, 1		; <i32> [#uses=2]
	%exitcond115.i.i = icmp eq i32 %1812, 256		; <i1> [#uses=1]
	br i1 %exitcond115.i.i, label %bb15.i36.i, label %bb13.i35.i

bb15.i36.i:		; preds = %bb13.i35.i
	store i32 -1, i32* @last, align 4
	%1813 = load i8* getelementptr ([18002 x i8]* @selector, i32 0, i32 0), align 32		; <i8> [#uses=1]
	%1814 = zext i8 %1813 to i32		; <i32> [#uses=4]
	%1815 = getelementptr [6 x i32]* @minLens, i32 0, i32 %1814		; <i32*> [#uses=1]
	%1816 = load i32* %1815, align 4		; <i32> [#uses=5]
	%1817 = zext i1 %.b.i to i32		; <i32> [#uses=3]
	%1818 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1817, i32 2		; <i32*> [#uses=12]
	%1819 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1817, i32 1		; <i32*> [#uses=6]
	%1820 = getelementptr [3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 %1817, i32 3		; <i8**> [#uses=6]
	br label %bb3.i.i41.i

bb3.i26.i.i:		; preds = %bb3.i.i41.i
	%1821 = load i32* %1818, align 8		; <i32> [#uses=3]
	%1822 = load i32* %1819, align 4		; <i32> [#uses=1]
	%1823 = icmp slt i32 %1821, %1822		; <i1> [#uses=1]
	br i1 %1823, label %bb2.i.i40.i, label %bb1.i.i39.i

bb1.i.i39.i:		; preds = %bb3.i26.i.i
	store i32 %bsBuff.tmp.0256, i32* @bsBuff
	store i32 %bsLive.tmp.0257, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i.i40.i:		; preds = %bb3.i26.i.i
	%1824 = load i8** %1820, align 4		; <i8*> [#uses=1]
	%1825 = getelementptr i8* %1824, i32 %1821		; <i8*> [#uses=1]
	%1826 = load i8* %1825, align 1		; <i8> [#uses=1]
	%1827 = add i32 %1821, 1		; <i32> [#uses=1]
	store i32 %1827, i32* %1818, align 8
	%1828 = zext i8 %1826 to i32		; <i32> [#uses=1]
	%1829 = shl i32 %bsBuff.tmp.0256, 8		; <i32> [#uses=1]
	%1830 = or i32 %1829, %1828		; <i32> [#uses=1]
	%1831 = add i32 %bsLive.tmp.0257, 8		; <i32> [#uses=3]
	br label %bb3.i.i41.i

bb3.i.i41.i:		; preds = %bb2.i.i40.i, %bb15.i36.i
	%bsLive.tmp.0257 = phi i32 [ %bsLive.tmp.0253, %bb15.i36.i ], [ %1831, %bb2.i.i40.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0256 = phi i32 [ %bsBuff.tmp.0252, %bb15.i36.i ], [ %1830, %bb2.i.i40.i ]		; <i32> [#uses=6]
	%1832 = phi i32 [ %1831, %bb2.i.i40.i ], [ %bsLive.tmp.0253, %bb15.i36.i ]		; <i32> [#uses=1]
	%1833 = phi i32 [ %1831, %bb2.i.i40.i ], [ %bsLive.tmp.0253, %bb15.i36.i ]		; <i32> [#uses=1]
	%1834 = icmp slt i32 %1833, %1816		; <i1> [#uses=1]
	br i1 %1834, label %bb3.i26.i.i, label %bsR.exit.i.i

bsR.exit.i.i:		; preds = %bb3.i.i41.i
	store i32 %bsBuff.tmp.0256, i32* @bsBuff
	%1835 = sub i32 %1832, %1816		; <i32> [#uses=4]
	%1836 = lshr i32 %bsBuff.tmp.0256, %1835		; <i32> [#uses=1]
	%1837 = shl i32 1, %1816		; <i32> [#uses=1]
	%1838 = add i32 %1837, -1		; <i32> [#uses=1]
	%1839 = and i32 %1836, %1838		; <i32> [#uses=1]
	store i32 %1835, i32* @bsLive, align 4
	%tmp112.i.i = mul i32 %1814, 258		; <i32> [#uses=1]
	%tmp113.i.i = add i32 %1816, %tmp112.i.i		; <i32> [#uses=1]
	br label %bb24.i.i

bb3.i40.i.i:		; preds = %bb22.i45.i
	%1840 = load i32* %1818, align 8		; <i32> [#uses=3]
	%1841 = load i32* %1819, align 4		; <i32> [#uses=1]
	%1842 = icmp slt i32 %1840, %1841		; <i1> [#uses=1]
	br i1 %1842, label %bb21.i44.i, label %bb20.i43.i

bb20.i43.i:		; preds = %bb3.i40.i.i
	store i32 %bsBuff.tmp.0260, i32* @bsBuff
	store i32 %bsLive.tmp.0261, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb21.i44.i:		; preds = %bb3.i40.i.i
	%1843 = load i8** %1820, align 4		; <i8*> [#uses=1]
	%1844 = getelementptr i8* %1843, i32 %1840		; <i8*> [#uses=1]
	%1845 = load i8* %1844, align 1		; <i8> [#uses=1]
	%1846 = add i32 %1840, 1		; <i32> [#uses=1]
	store i32 %1846, i32* %1818, align 8
	%1847 = zext i8 %1845 to i32		; <i32> [#uses=1]
	%1848 = shl i32 %bsBuff.tmp.0260, 8		; <i32> [#uses=1]
	%1849 = or i32 %1848, %1847		; <i32> [#uses=2]
	%1850 = add i32 %bsLive.tmp.0261, 8		; <i32> [#uses=3]
	br label %bb22.i45.i

bb22.i45.i:		; preds = %bb24.i.i, %bb21.i44.i
	%bsLive.tmp.0261 = phi i32 [ %1850, %bb21.i44.i ], [ %bsLive.tmp.0265, %bb24.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0260 = phi i32 [ %1849, %bb21.i44.i ], [ %bsBuff.tmp.0264, %bb24.i.i ]		; <i32> [#uses=3]
	%1851 = phi i32 [ %1850, %bb21.i44.i ], [ %.pr.i47.i, %bb24.i.i ]		; <i32> [#uses=1]
	%1852 = phi i32 [ %1849, %bb21.i44.i ], [ %.rle125.i.i, %bb24.i.i ]		; <i32> [#uses=2]
	%1853 = phi i32 [ %1850, %bb21.i44.i ], [ %.pr.i47.i, %bb24.i.i ]		; <i32> [#uses=1]
	%1854 = icmp sgt i32 %1853, 0		; <i1> [#uses=1]
	br i1 %1854, label %bb23.i46.i, label %bb3.i40.i.i

bb23.i46.i:		; preds = %bb22.i45.i
	%1855 = add i32 %1851, -1		; <i32> [#uses=3]
	%1856 = lshr i32 %1852, %1855		; <i32> [#uses=1]
	%1857 = and i32 %1856, 1		; <i32> [#uses=1]
	%1858 = shl i32 %zvec8.0.i.i, 1		; <i32> [#uses=1]
	%1859 = or i32 %1857, %1858		; <i32> [#uses=1]
	%indvar.next108.i.i = add i32 %indvar107.i.i, 1		; <i32> [#uses=1]
	br label %bb24.i.i

bb24.i.i:		; preds = %bb23.i46.i, %bsR.exit.i.i
	%bsLive.tmp.0265 = phi i32 [ %1835, %bsR.exit.i.i ], [ %1855, %bb23.i46.i ]		; <i32> [#uses=3]
	%bsBuff.tmp.0264 = phi i32 [ %bsBuff.tmp.0256, %bsR.exit.i.i ], [ %bsBuff.tmp.0260, %bb23.i46.i ]		; <i32> [#uses=3]
	%.rle125.i.i = phi i32 [ %1852, %bb23.i46.i ], [ %bsBuff.tmp.0256, %bsR.exit.i.i ]		; <i32> [#uses=1]
	%.pr.i47.i = phi i32 [ %1855, %bb23.i46.i ], [ %1835, %bsR.exit.i.i ]		; <i32> [#uses=2]
	%indvar107.i.i = phi i32 [ 0, %bsR.exit.i.i ], [ %indvar.next108.i.i, %bb23.i46.i ]		; <i32> [#uses=3]
	%zvec8.0.i.i = phi i32 [ %1839, %bsR.exit.i.i ], [ %1859, %bb23.i46.i ]		; <i32> [#uses=3]
	%scevgep110.sum.i.i = add i32 %indvar107.i.i, %tmp113.i.i		; <i32> [#uses=1]
	%scevgep114.i.i = getelementptr [6 x [258 x i32]]* @limit, i32 0, i32 0, i32 %scevgep110.sum.i.i		; <i32*> [#uses=1]
	%1860 = load i32* %scevgep114.i.i, align 4		; <i32> [#uses=1]
	%1861 = icmp slt i32 %1860, %zvec8.0.i.i		; <i1> [#uses=1]
	br i1 %1861, label %bb22.i45.i, label %bb25.i48.i

bb25.i48.i:		; preds = %bb24.i.i
	store i32 %bsBuff.tmp.0264, i32* @bsBuff
	store i32 %bsLive.tmp.0265, i32* @bsLive
	%zn9.0.i.i = add i32 %indvar107.i.i, %1816		; <i32> [#uses=1]
	%1862 = getelementptr [6 x [258 x i32]]* @base, i32 0, i32 %1814, i32 %zn9.0.i.i		; <i32*> [#uses=1]
	%1863 = load i32* %1862, align 4		; <i32> [#uses=1]
	%1864 = sub i32 %zvec8.0.i.i, %1863		; <i32> [#uses=1]
	%1865 = getelementptr [6 x [258 x i32]]* @perm, i32 0, i32 %1814, i32 %1864		; <i32*> [#uses=1]
	%1866 = load i32* %1865, align 4		; <i32> [#uses=1]
	br label %bb26.i49.i

bb26.i49.i:		; preds = %bb71.i.i, %bb49.i.i, %bb25.i48.i
	%1867 = phi i32 [ -1, %bb25.i48.i ], [ %1942, %bb71.i.i ], [ %last.tmp.2, %bb49.i.i ]		; <i32> [#uses=10]
	%bsLive.tmp.1300 = phi i32 [ %bsLive.tmp.0265, %bb25.i48.i ], [ %bsLive.tmp.0277, %bb71.i.i ], [ %bsLive.tmp.0289, %bb49.i.i ]		; <i32> [#uses=6]
	%bsBuff.tmp.1298 = phi i32 [ %bsBuff.tmp.0264, %bb25.i48.i ], [ %bsBuff.tmp.0276, %bb71.i.i ], [ %bsBuff.tmp.0288, %bb49.i.i ]		; <i32> [#uses=4]
	%groupPos.2.i.i = phi i32 [ 49, %bb25.i48.i ], [ %1965, %bb71.i.i ], [ %1876, %bb49.i.i ]		; <i32> [#uses=3]
	%groupNo.2.i.i = phi i32 [ 0, %bb25.i48.i ], [ %.groupNo.2.i.i, %bb71.i.i ], [ %.groupNo.3.i.i, %bb49.i.i ]		; <i32> [#uses=2]
	%nextSym.0.i.i = phi i32 [ %1866, %bb25.i48.i ], [ %2016, %bb71.i.i ], [ %1927, %bb49.i.i ]		; <i32> [#uses=8]
	%1868 = icmp eq i32 %nextSym.0.i.i, %1810		; <i1> [#uses=1]
	br i1 %1868, label %getAndMoveToFrontDecode.exit.i, label %bb27.i.i

bb27.i.i:		; preds = %bb26.i49.i
	%1869 = icmp ugt i32 %nextSym.0.i.i, 1		; <i1> [#uses=1]
	br i1 %1869, label %bb51.i.i, label %bb29.i50.i

bb29.i50.i:		; preds = %bb43.i.i, %bb27.i.i
	%bsLive.tmp.0293 = phi i32 [ %bsLive.tmp.0289, %bb43.i.i ], [ %bsLive.tmp.1300, %bb27.i.i ]		; <i32> [#uses=3]
	%bsBuff.tmp.0292 = phi i32 [ %bsBuff.tmp.0288, %bb43.i.i ], [ %bsBuff.tmp.1298, %bb27.i.i ]		; <i32> [#uses=1]
	%s.1.i.i = phi i32 [ %s.0.i.i, %bb43.i.i ], [ -1, %bb27.i.i ]		; <i32> [#uses=3]
	%N.0.i.i = phi i32 [ %1873, %bb43.i.i ], [ 1, %bb27.i.i ]		; <i32> [#uses=3]
	%groupPos.3.i.i = phi i32 [ %1876, %bb43.i.i ], [ %groupPos.2.i.i, %bb27.i.i ]		; <i32> [#uses=2]
	%groupNo.3.i.i = phi i32 [ %.groupNo.3.i.i, %bb43.i.i ], [ %groupNo.2.i.i, %bb27.i.i ]		; <i32> [#uses=1]
	%nextSym.1.i.i = phi i32 [ %1927, %bb43.i.i ], [ %nextSym.0.i.i, %bb27.i.i ]		; <i32> [#uses=1]
	switch i32 %nextSym.1.i.i, label %bb33.i52.i [
		i32 0, label %bb30.i51.i
		i32 1, label %bb32.i.i
	]

bb30.i51.i:		; preds = %bb29.i50.i
	%1870 = add i32 %N.0.i.i, %s.1.i.i		; <i32> [#uses=1]
	br label %bb33.i52.i

bb32.i.i:		; preds = %bb29.i50.i
	%1871 = shl i32 %N.0.i.i, 1		; <i32> [#uses=1]
	%1872 = add i32 %1871, %s.1.i.i		; <i32> [#uses=1]
	br label %bb33.i52.i

bb33.i52.i:		; preds = %bb32.i.i, %bb30.i51.i, %bb29.i50.i
	%s.0.i.i = phi i32 [ %1870, %bb30.i51.i ], [ %1872, %bb32.i.i ], [ %s.1.i.i, %bb29.i50.i ]		; <i32> [#uses=2]
	%1873 = shl i32 %N.0.i.i, 1		; <i32> [#uses=1]
	%1874 = icmp eq i32 %groupPos.3.i.i, 0		; <i1> [#uses=2]
	%1875 = zext i1 %1874 to i32		; <i32> [#uses=1]
	%.groupNo.3.i.i = add i32 %1875, %groupNo.3.i.i		; <i32> [#uses=3]
	%groupPos.3.op.i.i = add i32 %groupPos.3.i.i, -1		; <i32> [#uses=1]
	%1876 = select i1 %1874, i32 49, i32 %groupPos.3.op.i.i		; <i32> [#uses=2]
	%1877 = getelementptr [18002 x i8]* @selector, i32 0, i32 %.groupNo.3.i.i		; <i8*> [#uses=1]
	%1878 = load i8* %1877, align 1		; <i8> [#uses=1]
	%1879 = zext i8 %1878 to i32		; <i32> [#uses=4]
	%1880 = getelementptr [6 x i32]* @minLens, i32 0, i32 %1879		; <i32*> [#uses=1]
	%1881 = load i32* %1880, align 4		; <i32> [#uses=5]
	br label %bb3.i5.i.i

bb3.i47.i.i:		; preds = %bb3.i5.i.i
	%1882 = load i32* %1818, align 8		; <i32> [#uses=3]
	%1883 = load i32* %1819, align 4		; <i32> [#uses=1]
	%1884 = icmp slt i32 %1882, %1883		; <i1> [#uses=1]
	br i1 %1884, label %bb2.i4.i.i, label %bb1.i3.i.i

bb1.i3.i.i:		; preds = %bb3.i47.i.i
	store i32 %1867, i32* @last
	store i32 %bsBuff.tmp.0280, i32* @bsBuff
	store i32 %bsLive.tmp.0281, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i4.i.i:		; preds = %bb3.i47.i.i
	%1885 = load i8** %1820, align 4		; <i8*> [#uses=1]
	%1886 = getelementptr i8* %1885, i32 %1882		; <i8*> [#uses=1]
	%1887 = load i8* %1886, align 1		; <i8> [#uses=1]
	%1888 = add i32 %1882, 1		; <i32> [#uses=1]
	store i32 %1888, i32* %1818, align 8
	%1889 = zext i8 %1887 to i32		; <i32> [#uses=1]
	%1890 = shl i32 %bsBuff.tmp.0280, 8		; <i32> [#uses=1]
	%1891 = or i32 %1890, %1889		; <i32> [#uses=1]
	%1892 = add i32 %bsLive.tmp.0281, 8		; <i32> [#uses=3]
	br label %bb3.i5.i.i

bb3.i5.i.i:		; preds = %bb2.i4.i.i, %bb33.i52.i
	%bsLive.tmp.0281 = phi i32 [ %bsLive.tmp.0293, %bb33.i52.i ], [ %1892, %bb2.i4.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0280 = phi i32 [ %bsBuff.tmp.0292, %bb33.i52.i ], [ %1891, %bb2.i4.i.i ]		; <i32> [#uses=5]
	%1893 = phi i32 [ %1892, %bb2.i4.i.i ], [ %bsLive.tmp.0293, %bb33.i52.i ]		; <i32> [#uses=1]
	%1894 = phi i32 [ %1892, %bb2.i4.i.i ], [ %bsLive.tmp.0293, %bb33.i52.i ]		; <i32> [#uses=1]
	%1895 = icmp slt i32 %1894, %1881		; <i1> [#uses=1]
	br i1 %1895, label %bb3.i47.i.i, label %bsR.exit6.i.i

bsR.exit6.i.i:		; preds = %bb3.i5.i.i
	%1896 = sub i32 %1893, %1881		; <i32> [#uses=3]
	%1897 = lshr i32 %bsBuff.tmp.0280, %1896		; <i32> [#uses=1]
	%1898 = shl i32 1, %1881		; <i32> [#uses=1]
	%1899 = add i32 %1898, -1		; <i32> [#uses=1]
	%1900 = and i32 %1897, %1899		; <i32> [#uses=1]
	%tmp53.i.i = mul i32 %1879, 258		; <i32> [#uses=1]
	%tmp54.i.i = add i32 %1881, %tmp53.i.i		; <i32> [#uses=1]
	br label %bb42.i.i

bb3.i54.i.i:		; preds = %bb40.i.i
	%1901 = load i32* %1818, align 8		; <i32> [#uses=3]
	%1902 = load i32* %1819, align 4		; <i32> [#uses=1]
	%1903 = icmp slt i32 %1901, %1902		; <i1> [#uses=1]
	br i1 %1903, label %bb39.i.i, label %bb38.i.i

bb38.i.i:		; preds = %bb3.i54.i.i
	store i32 %1867, i32* @last
	store i32 %bsBuff.tmp.0284, i32* @bsBuff
	store i32 %bsLive.tmp.0285, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb39.i.i:		; preds = %bb3.i54.i.i
	%1904 = load i8** %1820, align 4		; <i8*> [#uses=1]
	%1905 = getelementptr i8* %1904, i32 %1901		; <i8*> [#uses=1]
	%1906 = load i8* %1905, align 1		; <i8> [#uses=1]
	%1907 = add i32 %1901, 1		; <i32> [#uses=1]
	store i32 %1907, i32* %1818, align 8
	%1908 = zext i8 %1906 to i32		; <i32> [#uses=1]
	%1909 = shl i32 %bsBuff.tmp.0284, 8		; <i32> [#uses=1]
	%1910 = or i32 %1909, %1908		; <i32> [#uses=2]
	%1911 = add i32 %bsLive.tmp.0285, 8		; <i32> [#uses=3]
	br label %bb40.i.i

bb40.i.i:		; preds = %bb42.i.i, %bb39.i.i
	%bsLive.tmp.0285 = phi i32 [ %1911, %bb39.i.i ], [ %bsLive.tmp.0289, %bb42.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0284 = phi i32 [ %1910, %bb39.i.i ], [ %bsBuff.tmp.0288, %bb42.i.i ]		; <i32> [#uses=3]
	%1912 = phi i32 [ %1911, %bb39.i.i ], [ %.pr13.i.i, %bb42.i.i ]		; <i32> [#uses=1]
	%1913 = phi i32 [ %1910, %bb39.i.i ], [ %.rle127.i.i, %bb42.i.i ]		; <i32> [#uses=2]
	%1914 = phi i32 [ %1911, %bb39.i.i ], [ %.pr13.i.i, %bb42.i.i ]		; <i32> [#uses=1]
	%1915 = icmp sgt i32 %1914, 0		; <i1> [#uses=1]
	br i1 %1915, label %bb41.i.i, label %bb3.i54.i.i

bb41.i.i:		; preds = %bb40.i.i
	%1916 = add i32 %1912, -1		; <i32> [#uses=3]
	%1917 = lshr i32 %1913, %1916		; <i32> [#uses=1]
	%1918 = and i32 %1917, 1		; <i32> [#uses=1]
	%1919 = shl i32 %zvec3.0.i.i, 1		; <i32> [#uses=1]
	%1920 = or i32 %1918, %1919		; <i32> [#uses=1]
	%indvar.next50.i.i = add i32 %indvar49.i.i, 1		; <i32> [#uses=1]
	br label %bb42.i.i

bb42.i.i:		; preds = %bb41.i.i, %bsR.exit6.i.i
	%bsLive.tmp.0289 = phi i32 [ %1896, %bsR.exit6.i.i ], [ %1916, %bb41.i.i ]		; <i32> [#uses=4]
	%bsBuff.tmp.0288 = phi i32 [ %bsBuff.tmp.0280, %bsR.exit6.i.i ], [ %bsBuff.tmp.0284, %bb41.i.i ]		; <i32> [#uses=4]
	%.rle127.i.i = phi i32 [ %1913, %bb41.i.i ], [ %bsBuff.tmp.0280, %bsR.exit6.i.i ]		; <i32> [#uses=1]
	%.pr13.i.i = phi i32 [ %1916, %bb41.i.i ], [ %1896, %bsR.exit6.i.i ]		; <i32> [#uses=2]
	%indvar49.i.i = phi i32 [ 0, %bsR.exit6.i.i ], [ %indvar.next50.i.i, %bb41.i.i ]		; <i32> [#uses=3]
	%zvec3.0.i.i = phi i32 [ %1900, %bsR.exit6.i.i ], [ %1920, %bb41.i.i ]		; <i32> [#uses=3]
	%scevgep.sum.i.i = add i32 %indvar49.i.i, %tmp54.i.i		; <i32> [#uses=1]
	%scevgep55.i.i = getelementptr [6 x [258 x i32]]* @limit, i32 0, i32 0, i32 %scevgep.sum.i.i		; <i32*> [#uses=1]
	%1921 = load i32* %scevgep55.i.i, align 4		; <i32> [#uses=1]
	%1922 = icmp slt i32 %1921, %zvec3.0.i.i		; <i1> [#uses=1]
	br i1 %1922, label %bb40.i.i, label %bb43.i.i

bb43.i.i:		; preds = %bb42.i.i
	%zn4.0.i.i = add i32 %indvar49.i.i, %1881		; <i32> [#uses=1]
	%1923 = getelementptr [6 x [258 x i32]]* @base, i32 0, i32 %1879, i32 %zn4.0.i.i		; <i32*> [#uses=1]
	%1924 = load i32* %1923, align 4		; <i32> [#uses=1]
	%1925 = sub i32 %zvec3.0.i.i, %1924		; <i32> [#uses=1]
	%1926 = getelementptr [6 x [258 x i32]]* @perm, i32 0, i32 %1879, i32 %1925		; <i32*> [#uses=1]
	%1927 = load i32* %1926, align 4		; <i32> [#uses=3]
	%1928 = icmp ugt i32 %1927, 1		; <i1> [#uses=1]
	br i1 %1928, label %bb44.i.i, label %bb29.i50.i

bb44.i.i:		; preds = %bb43.i.i
	%1929 = add i32 %s.0.i.i, 1		; <i32> [#uses=3]
	%1930 = load i8* %85, align 1		; <i8> [#uses=1]
	%1931 = zext i8 %1930 to i32		; <i32> [#uses=1]
	%1932 = getelementptr [256 x i8]* @seqToUnseq, i32 0, i32 %1931		; <i8*> [#uses=1]
	%1933 = load i8* %1932, align 1		; <i8> [#uses=2]
	%1934 = zext i8 %1933 to i32		; <i32> [#uses=1]
	%1935 = getelementptr [256 x i32]* @unzftab, i32 0, i32 %1934		; <i32*> [#uses=2]
	%1936 = load i32* %1935, align 4		; <i32> [#uses=1]
	%1937 = add i32 %1936, %1929		; <i32> [#uses=1]
	store i32 %1937, i32* %1935, align 4
	%1938 = icmp sgt i32 %1929, 0		; <i1> [#uses=1]
	br i1 %1938, label %bb47.i.i, label %bb49.i.i

bb47.i.i:		; preds = %bb47.i.i, %bb44.i.i
	%last.tmp.0 = phi i32 [ %1939, %bb47.i.i ], [ %1867, %bb44.i.i ]		; <i32> [#uses=1]
	%indvar.i53.i = phi i32 [ %indvar.next.i.i, %bb47.i.i ], [ 0, %bb44.i.i ]		; <i32> [#uses=1]
	%1939 = add i32 %last.tmp.0, 1		; <i32> [#uses=3]
	%1940 = getelementptr i8* %ll8.0, i32 %1939		; <i8*> [#uses=1]
	store i8 %1933, i8* %1940, align 1
	%indvar.next.i.i = add i32 %indvar.i53.i, 1		; <i32> [#uses=2]
	%exitcond.i54.i = icmp eq i32 %indvar.next.i.i, %1929		; <i1> [#uses=1]
	br i1 %exitcond.i54.i, label %bb49.i.i, label %bb47.i.i

bb49.i.i:		; preds = %bb47.i.i, %bb44.i.i
	%last.tmp.2 = phi i32 [ %1867, %bb44.i.i ], [ %1939, %bb47.i.i ]		; <i32> [#uses=3]
	%1941 = icmp slt i32 %last.tmp.2, %1554		; <i1> [#uses=1]
	br i1 %1941, label %bb26.i49.i, label %bb50.i.i

bb50.i.i:		; preds = %bb49.i.i
	store i32 %bsBuff.tmp.0288, i32* @bsBuff
	store i32 %bsLive.tmp.0289, i32* @bsLive
	store i32 %last.tmp.2, i32* @last
	call fastcc void @blockOverrun() nounwind ssp
	unreachable

bb51.i.i:		; preds = %bb27.i.i
	%1942 = add i32 %1867, 1		; <i32> [#uses=6]
	%1943 = icmp slt i32 %1942, %1554		; <i1> [#uses=1]
	br i1 %1943, label %bb53.i.i, label %bb52.i.i

bb52.i.i:		; preds = %bb51.i.i
	store i32 %bsBuff.tmp.1298, i32* @bsBuff
	store i32 %bsLive.tmp.1300, i32* @bsLive
	store i32 %1942, i32* @last
	call fastcc void @blockOverrun() nounwind ssp
	unreachable

bb53.i.i:		; preds = %bb51.i.i
	%1944 = add i32 %nextSym.0.i.i, -1		; <i32> [#uses=4]
	%1945 = getelementptr [256 x i8]* %yy.i.i, i32 0, i32 %1944		; <i8*> [#uses=1]
	%1946 = load i8* %1945, align 1		; <i8> [#uses=2]
	%1947 = zext i8 %1946 to i32		; <i32> [#uses=1]
	%1948 = getelementptr [256 x i8]* @seqToUnseq, i32 0, i32 %1947		; <i8*> [#uses=1]
	%1949 = load i8* %1948, align 1		; <i8> [#uses=2]
	%1950 = zext i8 %1949 to i32		; <i32> [#uses=1]
	%1951 = getelementptr [256 x i32]* @unzftab, i32 0, i32 %1950		; <i32*> [#uses=2]
	%1952 = load i32* %1951, align 4		; <i32> [#uses=1]
	%1953 = add i32 %1952, 1		; <i32> [#uses=1]
	store i32 %1953, i32* %1951, align 4
	%1954 = getelementptr i8* %ll8.0, i32 %1942		; <i8*> [#uses=1]
	store i8 %1949, i8* %1954, align 1
	%1955 = icmp sgt i32 %1944, 3		; <i1> [#uses=1]
	br i1 %1955, label %bb.nph29.i.i, label %bb60.preheader.i.i

bb.nph29.i.i:		; preds = %bb53.i.i
	%tmp74.i.i = add i32 %nextSym.0.i.i, -2		; <i32> [#uses=1]
	%tmp77.i.i = add i32 %nextSym.0.i.i, -3		; <i32> [#uses=1]
	%tmp80.i.i = add i32 %nextSym.0.i.i, -4		; <i32> [#uses=1]
	%tmp83.i.i = add i32 %nextSym.0.i.i, -5		; <i32> [#uses=1]
	br label %bb57.i.i

bb57.i.i:		; preds = %bb57.i.i, %bb.nph29.i.i
	%indvar68.i.i = phi i32 [ 0, %bb.nph29.i.i ], [ %indvar.next69.i.i, %bb57.i.i ]		; <i32> [#uses=2]
	%tmp70.i.i = mul i32 %indvar68.i.i, -4		; <i32> [#uses=5]
	%tmp72.i.i = add i32 %tmp70.i.i, %1944		; <i32> [#uses=1]
	%scevgep73.i.i = getelementptr [256 x i8]* %yy.i.i, i32 0, i32 %tmp72.i.i		; <i8*> [#uses=1]
	%tmp75.i.i = add i32 %tmp70.i.i, %tmp74.i.i		; <i32> [#uses=1]
	%scevgep76.i.i = getelementptr [256 x i8]* %yy.i.i, i32 0, i32 %tmp75.i.i		; <i8*> [#uses=2]
	%tmp78.i.i = add i32 %tmp70.i.i, %tmp77.i.i		; <i32> [#uses=1]
	%scevgep79.i.i = getelementptr [256 x i8]* %yy.i.i, i32 0, i32 %tmp78.i.i		; <i8*> [#uses=2]
	%tmp81.i.i = add i32 %tmp70.i.i, %tmp80.i.i		; <i32> [#uses=1]
	%scevgep82.i.i = getelementptr [256 x i8]* %yy.i.i, i32 0, i32 %tmp81.i.i		; <i8*> [#uses=2]
	%tmp84.i.i = add i32 %tmp70.i.i, %tmp83.i.i		; <i32> [#uses=3]
	%scevgep85.i.i = getelementptr [256 x i8]* %yy.i.i, i32 0, i32 %tmp84.i.i		; <i8*> [#uses=1]
	%1956 = load i8* %scevgep76.i.i, align 1		; <i8> [#uses=1]
	store i8 %1956, i8* %scevgep73.i.i, align 1
	%1957 = load i8* %scevgep79.i.i, align 1		; <i8> [#uses=1]
	store i8 %1957, i8* %scevgep76.i.i, align 1
	%1958 = load i8* %scevgep82.i.i, align 1		; <i8> [#uses=1]
	store i8 %1958, i8* %scevgep79.i.i, align 1
	%1959 = load i8* %scevgep85.i.i, align 1		; <i8> [#uses=1]
	store i8 %1959, i8* %scevgep82.i.i, align 1
	%1960 = icmp sgt i32 %tmp84.i.i, 3		; <i1> [#uses=1]
	%indvar.next69.i.i = add i32 %indvar68.i.i, 1		; <i32> [#uses=1]
	br i1 %1960, label %bb57.i.i, label %bb60.preheader.i.i

bb60.preheader.i.i:		; preds = %bb57.i.i, %bb53.i.i
	%j.0.lcssa.i.i = phi i32 [ %1944, %bb53.i.i ], [ %tmp84.i.i, %bb57.i.i ]		; <i32> [#uses=4]
	%1961 = icmp sgt i32 %j.0.lcssa.i.i, 0		; <i1> [#uses=1]
	br i1 %1961, label %bb.nph32.i.i, label %bb61.i.i

bb.nph32.i.i:		; preds = %bb60.preheader.i.i
	%tmp92.i.i = add i32 %j.0.lcssa.i.i, -1		; <i32> [#uses=1]
	br label %bb59.i.i

bb59.i.i:		; preds = %bb59.i.i, %bb.nph32.i.i
	%indvar86.i.i = phi i32 [ 0, %bb.nph32.i.i ], [ %indvar.next87.i.i, %bb59.i.i ]		; <i32> [#uses=3]
	%tmp90.i.i = sub i32 %j.0.lcssa.i.i, %indvar86.i.i		; <i32> [#uses=1]
	%scevgep91.i.i = getelementptr [256 x i8]* %yy.i.i, i32 0, i32 %tmp90.i.i		; <i8*> [#uses=1]
	%tmp93.i.i = sub i32 %tmp92.i.i, %indvar86.i.i		; <i32> [#uses=1]
	%scevgep94.i.i = getelementptr [256 x i8]* %yy.i.i, i32 0, i32 %tmp93.i.i		; <i8*> [#uses=1]
	%1962 = load i8* %scevgep94.i.i, align 1		; <i8> [#uses=1]
	store i8 %1962, i8* %scevgep91.i.i, align 1
	%indvar.next87.i.i = add i32 %indvar86.i.i, 1		; <i32> [#uses=2]
	%exitcond88.i.i = icmp eq i32 %indvar.next87.i.i, %j.0.lcssa.i.i		; <i1> [#uses=1]
	br i1 %exitcond88.i.i, label %bb61.i.i, label %bb59.i.i

bb61.i.i:		; preds = %bb59.i.i, %bb60.preheader.i.i
	store i8 %1946, i8* %85, align 1
	%1963 = icmp eq i32 %groupPos.2.i.i, 0		; <i1> [#uses=2]
	%1964 = zext i1 %1963 to i32		; <i32> [#uses=1]
	%.groupNo.2.i.i = add i32 %1964, %groupNo.2.i.i		; <i32> [#uses=2]
	%groupPos.2.op.i.i = add i32 %groupPos.2.i.i, -1		; <i32> [#uses=1]
	%1965 = select i1 %1963, i32 49, i32 %groupPos.2.op.i.i		; <i32> [#uses=1]
	%1966 = getelementptr [18002 x i8]* @selector, i32 0, i32 %.groupNo.2.i.i		; <i8*> [#uses=1]
	%1967 = load i8* %1966, align 1		; <i8> [#uses=1]
	%1968 = zext i8 %1967 to i32		; <i32> [#uses=4]
	%1969 = getelementptr [6 x i32]* @minLens, i32 0, i32 %1968		; <i32*> [#uses=1]
	%1970 = load i32* %1969, align 4		; <i32> [#uses=5]
	br label %bb3.i11.i.i

bb3.i33.i.i:		; preds = %bb3.i11.i.i
	%1971 = load i32* %1818, align 8		; <i32> [#uses=3]
	%1972 = load i32* %1819, align 4		; <i32> [#uses=1]
	%1973 = icmp slt i32 %1971, %1972		; <i1> [#uses=1]
	br i1 %1973, label %bb2.i10.i.i, label %bb1.i9.i.i

bb1.i9.i.i:		; preds = %bb3.i33.i.i
	store i32 %1942, i32* @last
	store i32 %bsBuff.tmp.0268, i32* @bsBuff
	store i32 %bsLive.tmp.0269, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb2.i10.i.i:		; preds = %bb3.i33.i.i
	%1974 = load i8** %1820, align 4		; <i8*> [#uses=1]
	%1975 = getelementptr i8* %1974, i32 %1971		; <i8*> [#uses=1]
	%1976 = load i8* %1975, align 1		; <i8> [#uses=1]
	%1977 = add i32 %1971, 1		; <i32> [#uses=1]
	store i32 %1977, i32* %1818, align 8
	%1978 = zext i8 %1976 to i32		; <i32> [#uses=1]
	%1979 = shl i32 %bsBuff.tmp.0268, 8		; <i32> [#uses=1]
	%1980 = or i32 %1979, %1978		; <i32> [#uses=1]
	%1981 = add i32 %bsLive.tmp.0269, 8		; <i32> [#uses=3]
	br label %bb3.i11.i.i

bb3.i11.i.i:		; preds = %bb2.i10.i.i, %bb61.i.i
	%bsLive.tmp.0269 = phi i32 [ %bsLive.tmp.1300, %bb61.i.i ], [ %1981, %bb2.i10.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0268 = phi i32 [ %bsBuff.tmp.1298, %bb61.i.i ], [ %1980, %bb2.i10.i.i ]		; <i32> [#uses=5]
	%1982 = phi i32 [ %1981, %bb2.i10.i.i ], [ %bsLive.tmp.1300, %bb61.i.i ]		; <i32> [#uses=1]
	%1983 = phi i32 [ %1981, %bb2.i10.i.i ], [ %bsLive.tmp.1300, %bb61.i.i ]		; <i32> [#uses=1]
	%1984 = icmp slt i32 %1983, %1970		; <i1> [#uses=1]
	br i1 %1984, label %bb3.i33.i.i, label %bsR.exit12.i.i

bsR.exit12.i.i:		; preds = %bb3.i11.i.i
	%1985 = sub i32 %1982, %1970		; <i32> [#uses=3]
	%1986 = lshr i32 %bsBuff.tmp.0268, %1985		; <i32> [#uses=1]
	%1987 = shl i32 1, %1970		; <i32> [#uses=1]
	%1988 = add i32 %1987, -1		; <i32> [#uses=1]
	%1989 = and i32 %1986, %1988		; <i32> [#uses=1]
	%tmp102.i.i = mul i32 %1968, 258		; <i32> [#uses=1]
	%tmp103.i.i = add i32 %1970, %tmp102.i.i		; <i32> [#uses=1]
	br label %bb70.i.i

bb3.i19.i.i:		; preds = %bb68.i.i
	%1990 = load i32* %1818, align 8		; <i32> [#uses=3]
	%1991 = load i32* %1819, align 4		; <i32> [#uses=1]
	%1992 = icmp slt i32 %1990, %1991		; <i1> [#uses=1]
	br i1 %1992, label %bb67.i.i, label %bb66.i.i

bb66.i.i:		; preds = %bb3.i19.i.i
	store i32 %1942, i32* @last
	store i32 %bsBuff.tmp.0272, i32* @bsBuff
	store i32 %bsLive.tmp.0273, i32* @bsLive
	call fastcc void @compressedStreamEOF() nounwind ssp
	unreachable

bb67.i.i:		; preds = %bb3.i19.i.i
	%1993 = load i8** %1820, align 4		; <i8*> [#uses=1]
	%1994 = getelementptr i8* %1993, i32 %1990		; <i8*> [#uses=1]
	%1995 = load i8* %1994, align 1		; <i8> [#uses=1]
	%1996 = add i32 %1990, 1		; <i32> [#uses=1]
	store i32 %1996, i32* %1818, align 8
	%1997 = zext i8 %1995 to i32		; <i32> [#uses=1]
	%1998 = shl i32 %bsBuff.tmp.0272, 8		; <i32> [#uses=1]
	%1999 = or i32 %1998, %1997		; <i32> [#uses=2]
	%2000 = add i32 %bsLive.tmp.0273, 8		; <i32> [#uses=3]
	br label %bb68.i.i

bb68.i.i:		; preds = %bb70.i.i, %bb67.i.i
	%bsLive.tmp.0273 = phi i32 [ %2000, %bb67.i.i ], [ %bsLive.tmp.0277, %bb70.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0272 = phi i32 [ %1999, %bb67.i.i ], [ %bsBuff.tmp.0276, %bb70.i.i ]		; <i32> [#uses=3]
	%2001 = phi i32 [ %2000, %bb67.i.i ], [ %.pr14.i.i, %bb70.i.i ]		; <i32> [#uses=1]
	%2002 = phi i32 [ %1999, %bb67.i.i ], [ %.rle126.i.i, %bb70.i.i ]		; <i32> [#uses=2]
	%2003 = phi i32 [ %2000, %bb67.i.i ], [ %.pr14.i.i, %bb70.i.i ]		; <i32> [#uses=1]
	%2004 = icmp sgt i32 %2003, 0		; <i1> [#uses=1]
	br i1 %2004, label %bb69.i.i, label %bb3.i19.i.i

bb69.i.i:		; preds = %bb68.i.i
	%2005 = add i32 %2001, -1		; <i32> [#uses=3]
	%2006 = lshr i32 %2002, %2005		; <i32> [#uses=1]
	%2007 = and i32 %2006, 1		; <i32> [#uses=1]
	%2008 = shl i32 %zvec.0.i.i, 1		; <i32> [#uses=1]
	%2009 = or i32 %2007, %2008		; <i32> [#uses=1]
	%indvar.next98.i.i = add i32 %indvar97.i.i, 1		; <i32> [#uses=1]
	br label %bb70.i.i

bb70.i.i:		; preds = %bb69.i.i, %bsR.exit12.i.i
	%bsLive.tmp.0277 = phi i32 [ %1985, %bsR.exit12.i.i ], [ %2005, %bb69.i.i ]		; <i32> [#uses=2]
	%bsBuff.tmp.0276 = phi i32 [ %bsBuff.tmp.0268, %bsR.exit12.i.i ], [ %bsBuff.tmp.0272, %bb69.i.i ]		; <i32> [#uses=2]
	%.rle126.i.i = phi i32 [ %2002, %bb69.i.i ], [ %bsBuff.tmp.0268, %bsR.exit12.i.i ]		; <i32> [#uses=1]
	%.pr14.i.i = phi i32 [ %2005, %bb69.i.i ], [ %1985, %bsR.exit12.i.i ]		; <i32> [#uses=2]
	%indvar97.i.i = phi i32 [ 0, %bsR.exit12.i.i ], [ %indvar.next98.i.i, %bb69.i.i ]		; <i32> [#uses=3]
	%zvec.0.i.i = phi i32 [ %1989, %bsR.exit12.i.i ], [ %2009, %bb69.i.i ]		; <i32> [#uses=3]
	%scevgep100.sum.i.i = add i32 %indvar97.i.i, %tmp103.i.i		; <i32> [#uses=1]
	%scevgep104.i.i = getelementptr [6 x [258 x i32]]* @limit, i32 0, i32 0, i32 %scevgep100.sum.i.i		; <i32*> [#uses=1]
	%2010 = load i32* %scevgep104.i.i, align 4		; <i32> [#uses=1]
	%2011 = icmp slt i32 %2010, %zvec.0.i.i		; <i1> [#uses=1]
	br i1 %2011, label %bb68.i.i, label %bb71.i.i

bb71.i.i:		; preds = %bb70.i.i
	%zn.0.i.i = add i32 %indvar97.i.i, %1970		; <i32> [#uses=1]
	%2012 = getelementptr [6 x [258 x i32]]* @base, i32 0, i32 %1968, i32 %zn.0.i.i		; <i32*> [#uses=1]
	%2013 = load i32* %2012, align 4		; <i32> [#uses=1]
	%2014 = sub i32 %zvec.0.i.i, %2013		; <i32> [#uses=1]
	%2015 = getelementptr [6 x [258 x i32]]* @perm, i32 0, i32 %1968, i32 %2014		; <i32*> [#uses=1]
	%2016 = load i32* %2015, align 4		; <i32> [#uses=1]
	br label %bb26.i49.i

getAndMoveToFrontDecode.exit.i:		; preds = %bb26.i49.i
	store i32 %bsBuff.tmp.1298, i32* @bsBuff
	store i32 %bsLive.tmp.1300, i32* @bsLive
	store i32 %1867, i32* @last
	store i32 -1, i32* @globalCrc, align 4
	store i32 0, i32* %86, align 4
	br label %bb.i30.i

bb.i30.i:		; preds = %bb.i30.i, %getAndMoveToFrontDecode.exit.i
	%indvar41.i.i = phi i32 [ 0, %getAndMoveToFrontDecode.exit.i ], [ %tmp44.i.i, %bb.i30.i ]		; <i32> [#uses=2]
	%tmp44.i.i = add i32 %indvar41.i.i, 1		; <i32> [#uses=3]
	%scevgep45.i.i = getelementptr [257 x i32]* %cftab.i.i, i32 0, i32 %tmp44.i.i		; <i32*> [#uses=1]
	%scevgep46.i.i = getelementptr [256 x i32]* @unzftab, i32 0, i32 %indvar41.i.i		; <i32*> [#uses=1]
	%2017 = load i32* %scevgep46.i.i, align 4		; <i32> [#uses=1]
	store i32 %2017, i32* %scevgep45.i.i, align 4
	%exitcond43.i.i = icmp eq i32 %tmp44.i.i, 256		; <i1> [#uses=1]
	br i1 %exitcond43.i.i, label %bb5.i.i, label %bb.i30.i

bb5.i.i:		; preds = %bb5.i.i, %bb.i30.i
	%indvar.i.i = phi i32 [ %tmp38.i.i, %bb5.i.i ], [ 0, %bb.i30.i ]		; <i32> [#uses=2]
	%tmp38.i.i = add i32 %indvar.i.i, 1		; <i32> [#uses=3]
	%scevgep39.i.i = getelementptr [257 x i32]* %cftab.i.i, i32 0, i32 %tmp38.i.i		; <i32*> [#uses=2]
	%scevgep40.i.i = getelementptr [257 x i32]* %cftab.i.i, i32 0, i32 %indvar.i.i		; <i32*> [#uses=1]
	%2018 = load i32* %scevgep39.i.i, align 4		; <i32> [#uses=1]
	%2019 = load i32* %scevgep40.i.i, align 4		; <i32> [#uses=1]
	%2020 = add i32 %2019, %2018		; <i32> [#uses=1]
	store i32 %2020, i32* %scevgep39.i.i, align 4
	%exitcond.i.i = icmp eq i32 %tmp38.i.i, 256		; <i1> [#uses=1]
	br i1 %exitcond.i.i, label %bb9.loopexit.i.i, label %bb5.i.i

bb8.i.i:		; preds = %bb9.loopexit.i.i, %bb8.i.i
	%2021 = phi i32 [ %2029, %bb8.i.i ], [ 0, %bb9.loopexit.i.i ]		; <i32> [#uses=3]
	%scevgep.i.i = getelementptr i8* %ll8.0, i32 %2021		; <i8*> [#uses=1]
	%2022 = load i8* %scevgep.i.i, align 1		; <i8> [#uses=1]
	%2023 = zext i8 %2022 to i32		; <i32> [#uses=1]
	%2024 = getelementptr [257 x i32]* %cftab.i.i, i32 0, i32 %2023		; <i32*> [#uses=3]
	%2025 = load i32* %2024, align 4		; <i32> [#uses=1]
	%2026 = getelementptr i32* %tt.0, i32 %2025		; <i32*> [#uses=1]
	store i32 %2021, i32* %2026, align 4
	%2027 = load i32* %2024, align 4		; <i32> [#uses=1]
	%2028 = add i32 %2027, 1		; <i32> [#uses=1]
	store i32 %2028, i32* %2024, align 4
	%2029 = add i32 %2021, 1		; <i32> [#uses=2]
	%2030 = icmp sgt i32 %2029, %1867		; <i1> [#uses=1]
	br i1 %2030, label %bb10.i.i.loopexit, label %bb8.i.i

bb9.loopexit.i.i:		; preds = %bb5.i.i
	%2031 = icmp slt i32 %1867, 0		; <i1> [#uses=1]
	br i1 %2031, label %bb10.i.i, label %bb8.i.i

bb10.i.i.loopexit:		; preds = %bb8.i.i
	%.pre = load i32* @globalCrc, align 4		; <i32> [#uses=1]
	br label %bb10.i.i

bb10.i.i:		; preds = %bb10.i.i.loopexit, %bb9.loopexit.i.i
	%2032 = phi i32 [ %.pre, %bb10.i.i.loopexit ], [ -1, %bb9.loopexit.i.i ]		; <i32> [#uses=2]
	%2033 = getelementptr i32* %tt.0, i32 %1571		; <i32*> [#uses=1]
	%2034 = load i32* %2033, align 4		; <i32> [#uses=2]
	%tmp759 = shl i32 1, %1552		; <i32> [#uses=1]
	%storemerge.i757760 = and i32 %bsBuff.tmp.0185, %tmp759		; <i32> [#uses=1]
	%2035 = icmp eq i32 %storemerge.i757760, 0		; <i1> [#uses=1]
	%.pre.i33.i = load i32* @last, align 4		; <i32> [#uses=6]
	br i1 %2035, label %bb33.i.i, label %bb25.i.i

bb12.i.i:		; preds = %bb25.i.i
	%2036 = getelementptr i8* %ll8.0, i32 %tPos.0.i.i		; <i8*> [#uses=1]
	%2037 = load i8* %2036, align 1		; <i8> [#uses=1]
	%2038 = zext i8 %2037 to i32		; <i32> [#uses=1]
	%2039 = getelementptr i32* %tt.0, i32 %tPos.0.i.i		; <i32*> [#uses=1]
	%2040 = load i32* %2039, align 4		; <i32> [#uses=4]
	%2041 = icmp eq i32 %rNToGo.1.i.i, 0		; <i1> [#uses=1]
	br i1 %2041, label %bb13.i.i, label %bb15.i.i

bb13.i.i:		; preds = %bb12.i.i
	%2042 = getelementptr [512 x i32]* @rNums, i32 0, i32 %rTPos.1.i.i		; <i32*> [#uses=1]
	%2043 = load i32* %2042, align 4		; <i32> [#uses=2]
	%2044 = add i32 %rTPos.1.i.i, 1		; <i32> [#uses=2]
	%2045 = icmp eq i32 %2044, 512		; <i1> [#uses=1]
	br i1 %2045, label %bb14.i.i, label %bb15.i.i

bb14.i.i:		; preds = %bb13.i.i
	br label %bb15.i.i

bb15.i.i:		; preds = %bb14.i.i, %bb13.i.i, %bb12.i.i
	%rNToGo.0.i.i = phi i32 [ %2043, %bb14.i.i ], [ %rNToGo.1.i.i, %bb12.i.i ], [ %2043, %bb13.i.i ]		; <i32> [#uses=2]
	%rTPos.0.i.i = phi i32 [ 0, %bb14.i.i ], [ %rTPos.1.i.i, %bb12.i.i ], [ %2044, %bb13.i.i ]		; <i32> [#uses=5]
	%2046 = add i32 %rNToGo.0.i.i, -1		; <i32> [#uses=4]
	%2047 = icmp eq i32 %2046, 1		; <i1> [#uses=1]
	%2048 = zext i1 %2047 to i32		; <i32> [#uses=1]
	%2049 = xor i32 %2048, %2038		; <i32> [#uses=7]
	%2050 = add i32 %i2.0.i.i, 1		; <i32> [#uses=2]
	%2051 = trunc i32 %2049 to i8		; <i8> [#uses=2]
	%2052 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 3), align 4		; <i8*> [#uses=1]
	%2053 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 2), align 8		; <i32> [#uses=2]
	%2054 = getelementptr i8* %2052, i32 %2053		; <i8*> [#uses=1]
	store i8 %2051, i8* %2054, align 1
	%2055 = add i32 %2053, 1		; <i32> [#uses=2]
	store i32 %2055, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 2), align 8
	%2056 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4		; <i32> [#uses=1]
	%2057 = add i32 %2056, 1		; <i32> [#uses=1]
	store i32 %2057, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4
	%2058 = shl i32 %localCrc.0.i.i, 8		; <i32> [#uses=1]
	%2059 = lshr i32 %localCrc.0.i.i, 24		; <i32> [#uses=1]
	%2060 = xor i32 %2049, %2059		; <i32> [#uses=1]
	%2061 = getelementptr [256 x i32]* @crc32Table, i32 0, i32 %2060		; <i32*> [#uses=1]
	%2062 = load i32* %2061, align 4		; <i32> [#uses=1]
	%2063 = xor i32 %2062, %2058		; <i32> [#uses=3]
	%2064 = icmp eq i32 %2049, %ch2.0.i.i		; <i1> [#uses=1]
	br i1 %2064, label %bb17.i.i, label %bb25.i.i

bb17.i.i:		; preds = %bb15.i.i
	%2065 = add i32 %count.0.i.i, 1		; <i32> [#uses=2]
	%2066 = icmp sgt i32 %2065, 3		; <i1> [#uses=1]
	br i1 %2066, label %bb18.i.i, label %bb25.i.i

bb18.i.i:		; preds = %bb17.i.i
	%2067 = getelementptr i8* %ll8.0, i32 %2040		; <i8*> [#uses=1]
	%2068 = load i8* %2067, align 1		; <i8> [#uses=1]
	%2069 = getelementptr i32* %tt.0, i32 %2040		; <i32*> [#uses=1]
	%2070 = load i32* %2069, align 4		; <i32> [#uses=1]
	%2071 = icmp eq i32 %rNToGo.0.i.i, 1		; <i1> [#uses=1]
	br i1 %2071, label %bb19.i.i, label %bb21.i.i

bb19.i.i:		; preds = %bb18.i.i
	%2072 = getelementptr [512 x i32]* @rNums, i32 0, i32 %rTPos.0.i.i		; <i32*> [#uses=1]
	%2073 = load i32* %2072, align 4		; <i32> [#uses=2]
	%2074 = add i32 %rTPos.0.i.i, 1		; <i32> [#uses=2]
	%2075 = icmp eq i32 %2074, 512		; <i1> [#uses=1]
	br i1 %2075, label %bb20.i.i, label %bb21.i.i

bb20.i.i:		; preds = %bb19.i.i
	br label %bb21.i.i

bb21.i.i:		; preds = %bb20.i.i, %bb19.i.i, %bb18.i.i
	%rNToGo.2.i.i = phi i32 [ %2073, %bb20.i.i ], [ %2046, %bb18.i.i ], [ %2073, %bb19.i.i ]		; <i32> [#uses=1]
	%rTPos.2.i.i = phi i32 [ 0, %bb20.i.i ], [ %rTPos.0.i.i, %bb18.i.i ], [ %2074, %bb19.i.i ]		; <i32> [#uses=1]
	%2076 = add i32 %rNToGo.2.i.i, -1		; <i32> [#uses=2]
	%2077 = icmp eq i32 %2076, 1		; <i1> [#uses=1]
	%2078 = zext i1 %2077 to i8		; <i8> [#uses=1]
	%2079 = xor i8 %2078, %2068		; <i8> [#uses=1]
	%2080 = zext i8 %2079 to i32		; <i32> [#uses=1]
	br label %bb23.i.i

bb22.i.i:		; preds = %bb23.i.i
	%2081 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 3), align 4		; <i8*> [#uses=1]
	%2082 = getelementptr i8* %2081, i32 %2093		; <i8*> [#uses=1]
	store i8 %2051, i8* %2082, align 1
	%2083 = add i32 %2093, 1		; <i32> [#uses=2]
	store i32 %2083, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 2), align 8
	%2084 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4		; <i32> [#uses=1]
	%2085 = add i32 %2084, 1		; <i32> [#uses=1]
	store i32 %2085, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4
	%2086 = shl i32 %localCrc.2.i.i, 8		; <i32> [#uses=1]
	%2087 = lshr i32 %localCrc.2.i.i, 24		; <i32> [#uses=1]
	%2088 = xor i32 %2087, %2049		; <i32> [#uses=1]
	%2089 = getelementptr [256 x i32]* @crc32Table, i32 0, i32 %2088		; <i32*> [#uses=1]
	%2090 = load i32* %2089, align 4		; <i32> [#uses=1]
	%2091 = xor i32 %2090, %2086		; <i32> [#uses=1]
	%2092 = add i32 %2094, 1		; <i32> [#uses=1]
	br label %bb23.i.i

bb23.i.i:		; preds = %bb22.i.i, %bb21.i.i
	%2093 = phi i32 [ %2083, %bb22.i.i ], [ %2055, %bb21.i.i ]		; <i32> [#uses=2]
	%localCrc.2.i.i = phi i32 [ %2091, %bb22.i.i ], [ %2063, %bb21.i.i ]		; <i32> [#uses=3]
	%2094 = phi i32 [ %2092, %bb22.i.i ], [ 0, %bb21.i.i ]		; <i32> [#uses=2]
	%2095 = icmp sgt i32 %2080, %2094		; <i1> [#uses=1]
	br i1 %2095, label %bb22.i.i, label %bb24.split.i.i

bb24.split.i.i:		; preds = %bb23.i.i
	%2096 = add i32 %i2.0.i.i, 2		; <i32> [#uses=1]
	br label %bb25.i.i

bb25.i.i:		; preds = %bb24.split.i.i, %bb17.i.i, %bb15.i.i, %bb10.i.i
	%2097 = phi i32 [ %1867, %bb10.i.i ], [ %.pre.i33.i, %bb24.split.i.i ], [ %.pre.i33.i, %bb17.i.i ], [ %.pre.i33.i, %bb15.i.i ]		; <i32> [#uses=1]
	%i2.0.i.i = phi i32 [ 0, %bb10.i.i ], [ %2096, %bb24.split.i.i ], [ %2050, %bb15.i.i ], [ %2050, %bb17.i.i ]		; <i32> [#uses=3]
	%count.0.i.i = phi i32 [ 0, %bb10.i.i ], [ 0, %bb24.split.i.i ], [ 1, %bb15.i.i ], [ %2065, %bb17.i.i ]		; <i32> [#uses=1]
	%ch2.0.i.i = phi i32 [ 256, %bb10.i.i ], [ %2049, %bb24.split.i.i ], [ %2049, %bb17.i.i ], [ %2049, %bb15.i.i ]		; <i32> [#uses=1]
	%localCrc.0.i.i = phi i32 [ %2032, %bb10.i.i ], [ %localCrc.2.i.i, %bb24.split.i.i ], [ %2063, %bb15.i.i ], [ %2063, %bb17.i.i ]		; <i32> [#uses=3]
	%rNToGo.1.i.i = phi i32 [ 0, %bb10.i.i ], [ %2076, %bb24.split.i.i ], [ %2046, %bb15.i.i ], [ %2046, %bb17.i.i ]		; <i32> [#uses=2]
	%rTPos.1.i.i = phi i32 [ 0, %bb10.i.i ], [ %rTPos.2.i.i, %bb24.split.i.i ], [ %rTPos.0.i.i, %bb15.i.i ], [ %rTPos.0.i.i, %bb17.i.i ]		; <i32> [#uses=3]
	%tPos.0.i.i = phi i32 [ %2034, %bb10.i.i ], [ %2070, %bb24.split.i.i ], [ %2040, %bb15.i.i ], [ %2040, %bb17.i.i ]		; <i32> [#uses=2]
	%2098 = icmp sgt i32 %i2.0.i.i, %2097		; <i1> [#uses=1]
	br i1 %2098, label %undoReversibleTransformation_fast.exit.i, label %bb12.i.i

bb26.i.i:		; preds = %bb33.i.i
	%2099 = getelementptr i8* %ll8.0, i32 %tPos.1.i.i		; <i8*> [#uses=1]
	%2100 = load i8* %2099, align 1		; <i8> [#uses=3]
	%2101 = zext i8 %2100 to i32		; <i32> [#uses=6]
	%2102 = getelementptr i32* %tt.0, i32 %tPos.1.i.i		; <i32*> [#uses=1]
	%2103 = load i32* %2102, align 4		; <i32> [#uses=4]
	%2104 = add i32 %i2.1.i.i, 1		; <i32> [#uses=2]
	%2105 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 3), align 4		; <i8*> [#uses=1]
	%2106 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 2), align 8		; <i32> [#uses=2]
	%2107 = getelementptr i8* %2105, i32 %2106		; <i8*> [#uses=1]
	store i8 %2100, i8* %2107, align 1
	%2108 = add i32 %2106, 1		; <i32> [#uses=2]
	store i32 %2108, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 2), align 8
	%2109 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4		; <i32> [#uses=1]
	%2110 = add i32 %2109, 1		; <i32> [#uses=1]
	store i32 %2110, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4
	%2111 = shl i32 %localCrc.3.i.i, 8		; <i32> [#uses=1]
	%2112 = lshr i32 %localCrc.3.i.i, 24		; <i32> [#uses=1]
	%2113 = xor i32 %2101, %2112		; <i32> [#uses=1]
	%2114 = getelementptr [256 x i32]* @crc32Table, i32 0, i32 %2113		; <i32*> [#uses=1]
	%2115 = load i32* %2114, align 4		; <i32> [#uses=1]
	%2116 = xor i32 %2115, %2111		; <i32> [#uses=3]
	%2117 = icmp eq i32 %2101, %ch2.1.i.i		; <i1> [#uses=1]
	br i1 %2117, label %bb28.i.i, label %bb33.i.i

bb28.i.i:		; preds = %bb26.i.i
	%2118 = add i32 %count.1.i.i, 1		; <i32> [#uses=2]
	%2119 = icmp sgt i32 %2118, 3		; <i1> [#uses=1]
	br i1 %2119, label %bb29.i.i, label %bb33.i.i

bb29.i.i:		; preds = %bb28.i.i
	%2120 = getelementptr i8* %ll8.0, i32 %2103		; <i8*> [#uses=1]
	%2121 = load i8* %2120, align 1		; <i8> [#uses=1]
	%2122 = getelementptr i32* %tt.0, i32 %2103		; <i32*> [#uses=1]
	%2123 = load i32* %2122, align 4		; <i32> [#uses=1]
	%2124 = zext i8 %2121 to i32		; <i32> [#uses=1]
	br label %bb31.i.i

bb30.i.i:		; preds = %bb31.i.i
	%2125 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 3), align 4		; <i8*> [#uses=1]
	%2126 = getelementptr i8* %2125, i32 %2137		; <i8*> [#uses=1]
	store i8 %2100, i8* %2126, align 1
	%2127 = add i32 %2137, 1		; <i32> [#uses=2]
	store i32 %2127, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 2), align 8
	%2128 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4		; <i32> [#uses=1]
	%2129 = add i32 %2128, 1		; <i32> [#uses=1]
	store i32 %2129, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4
	%2130 = shl i32 %localCrc.4.i.i, 8		; <i32> [#uses=1]
	%2131 = lshr i32 %localCrc.4.i.i, 24		; <i32> [#uses=1]
	%2132 = xor i32 %2131, %2101		; <i32> [#uses=1]
	%2133 = getelementptr [256 x i32]* @crc32Table, i32 0, i32 %2132		; <i32*> [#uses=1]
	%2134 = load i32* %2133, align 4		; <i32> [#uses=1]
	%2135 = xor i32 %2134, %2130		; <i32> [#uses=1]
	%2136 = add i32 %2138, 1		; <i32> [#uses=1]
	br label %bb31.i.i

bb31.i.i:		; preds = %bb30.i.i, %bb29.i.i
	%2137 = phi i32 [ %2127, %bb30.i.i ], [ %2108, %bb29.i.i ]		; <i32> [#uses=2]
	%localCrc.4.i.i = phi i32 [ %2135, %bb30.i.i ], [ %2116, %bb29.i.i ]		; <i32> [#uses=3]
	%2138 = phi i32 [ %2136, %bb30.i.i ], [ 0, %bb29.i.i ]		; <i32> [#uses=2]
	%2139 = icmp sgt i32 %2124, %2138		; <i1> [#uses=1]
	br i1 %2139, label %bb30.i.i, label %bb32.split.i.i

bb32.split.i.i:		; preds = %bb31.i.i
	%2140 = add i32 %i2.1.i.i, 2		; <i32> [#uses=1]
	br label %bb33.i.i

bb33.i.i:		; preds = %bb32.split.i.i, %bb28.i.i, %bb26.i.i, %bb10.i.i
	%2141 = phi i32 [ %.pre.i33.i, %bb32.split.i.i ], [ %.pre.i33.i, %bb28.i.i ], [ %.pre.i33.i, %bb26.i.i ], [ %1867, %bb10.i.i ]		; <i32> [#uses=1]
	%i2.1.i.i = phi i32 [ %2140, %bb32.split.i.i ], [ %2104, %bb26.i.i ], [ %2104, %bb28.i.i ], [ 0, %bb10.i.i ]		; <i32> [#uses=3]
	%count.1.i.i = phi i32 [ 0, %bb32.split.i.i ], [ 1, %bb26.i.i ], [ %2118, %bb28.i.i ], [ 0, %bb10.i.i ]		; <i32> [#uses=1]
	%ch2.1.i.i = phi i32 [ %2101, %bb32.split.i.i ], [ %2101, %bb28.i.i ], [ %2101, %bb26.i.i ], [ 256, %bb10.i.i ]		; <i32> [#uses=1]
	%localCrc.3.i.i = phi i32 [ %localCrc.4.i.i, %bb32.split.i.i ], [ %2116, %bb26.i.i ], [ %2116, %bb28.i.i ], [ %2032, %bb10.i.i ]		; <i32> [#uses=3]
	%tPos.1.i.i = phi i32 [ %2123, %bb32.split.i.i ], [ %2103, %bb26.i.i ], [ %2103, %bb28.i.i ], [ %2034, %bb10.i.i ]		; <i32> [#uses=2]
	%2142 = icmp sgt i32 %i2.1.i.i, %2141		; <i1> [#uses=1]
	br i1 %2142, label %undoReversibleTransformation_fast.exit.i, label %bb26.i.i

undoReversibleTransformation_fast.exit.i:		; preds = %bb33.i.i, %bb25.i.i
	%localCrc.1.i.i = phi i32 [ %localCrc.0.i.i, %bb25.i.i ], [ %localCrc.3.i.i, %bb33.i.i ]		; <i32> [#uses=2]
	store i32 %localCrc.1.i.i, i32* @globalCrc, align 4
	%not.i.i = xor i32 %localCrc.1.i.i, -1		; <i32> [#uses=3]
	%2143 = icmp eq i32 %1533, %not.i.i		; <i1> [#uses=1]
	br i1 %2143, label %bb54.i, label %bb53.i

bb53.i:		; preds = %undoReversibleTransformation_fast.exit.i
	tail call fastcc void @crcError(i32 %1533, i32 %not.i.i) nounwind ssp
	unreachable

bb54.i:		; preds = %undoReversibleTransformation_fast.exit.i
	%2144 = lshr i32 %computedCombinedCRC.0.i, 31		; <i32> [#uses=1]
	%2145 = shl i32 %computedCombinedCRC.0.i, 1		; <i32> [#uses=1]
	%2146 = or i32 %2144, %2145		; <i32> [#uses=1]
	%2147 = xor i32 %2146, %not.i.i		; <i32> [#uses=1]
	%.pre.i.i22.i.pre = load i32* @bsLive, align 4		; <i32> [#uses=2]
	%.b6.i.pre = load i1* @bsStream.b		; <i1> [#uses=1]
	%bsBuff.promoted302.pre = load i32* @bsBuff		; <i32> [#uses=1]
	%phitmp758 = zext i1 %.b6.i.pre to i32		; <i32> [#uses=1]
	br label %bb13.i

bb55.i:		; preds = %bb21.i
	%2148 = tail call fastcc i32 @bsGetUInt32() nounwind ssp		; <i32> [#uses=2]
	%2149 = icmp eq i32 %2148, %computedCombinedCRC.0.i		; <i1> [#uses=1]
	br i1 %2149, label %bb23, label %bb60.i

bb60.i:		; preds = %bb55.i
	tail call fastcc void @crcError(i32 %2148, i32 %computedCombinedCRC.0.i) nounwind ssp
	unreachable

bb23:		; preds = %bb55.i, %bb.i, %bsGetUChar.exit21.i
	%ll8.1 = phi i8* [ %ll8.2, %bsGetUChar.exit21.i ], [ %ll8.2, %bb.i ], [ %ll8.0, %bb55.i ]		; <i8*> [#uses=1]
	%tt.1 = phi i32* [ %tt.2, %bsGetUChar.exit21.i ], [ %tt.2, %bb.i ], [ %tt.0, %bb55.i ]		; <i32*> [#uses=1]
	store i1 false, i1* @bsStream.b
	%2150 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 1), align 4		; <i32> [#uses=1]
	%2151 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([38 x i8]* @"\01LC2694", i32 0, i32 0), i32 %2150) nounwind		; <i32> [#uses=0]
	br label %bb28

bb25:		; preds = %bb28
	%scevgep2 = getelementptr i8* %62, i32 %i.1		; <i8*> [#uses=1]
	%2152 = load i8* %scevgep2, align 1		; <i8> [#uses=1]
	%2153 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 3), align 4		; <i8*> [#uses=1]
	%scevgep = getelementptr i8* %2153, i32 %tmp		; <i8*> [#uses=1]
	%2154 = load i8* %scevgep, align 1		; <i8> [#uses=1]
	%2155 = icmp eq i8 %2152, %2154		; <i1> [#uses=1]
	br i1 %2155, label %bb27, label %bb26

bb26:		; preds = %bb25
	%2156 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([35 x i8]* @"\01LC2795", i32 0, i32 0), i32 %input_size.0) nounwind		; <i32> [#uses=0]
	ret i32 1

bb27:		; preds = %bb25
	%2157 = add i32 %i.1, 1		; <i32> [#uses=1]
	br label %bb28

bb28:		; preds = %bb27, %bb23
	%i.1 = phi i32 [ %2157, %bb27 ], [ 0, %bb23 ]		; <i32> [#uses=3]
	%tmp = mul i32 %i.1, 1027		; <i32> [#uses=2]
	%2158 = icmp slt i32 %tmp, %11		; <i1> [#uses=1]
	br i1 %2158, label %bb25, label %bb31

bb31:		; preds = %bb28
	%2159 = tail call i32 @puts(i8* getelementptr ([37 x i8]* @"\01LC2896", i32 0, i32 0)) nounwind		; <i32> [#uses=0]
	%2160 = load i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 1), align 4		; <i32> [#uses=1]
	%2161 = load i8** getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 3), align 4		; <i8*> [#uses=1]
	tail call void @llvm.memset.i32(i8* %2161, i8 0, i32 %2160, i32 1) nounwind
	store i32 0, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 1), align 4
	store i32 0, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 1, i32 2), align 8
	store i32 0, i32* getelementptr ([3 x %struct.spec_fd_t]* @spec_fd, i32 0, i32 0, i32 2), align 8
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br label %bb32

bb32:		; preds = %bb31, %spec_initbufs.exit
	%ll8.2 = phi i8* [ null, %spec_initbufs.exit ], [ %ll8.1, %bb31 ]		; <i8*> [#uses=6]
	%tt.2 = phi i32* [ null, %spec_initbufs.exit ], [ %tt.1, %bb31 ]		; <i32*> [#uses=6]
	%indvar = phi i32 [ 0, %spec_initbufs.exit ], [ %indvar.next, %bb31 ]		; <i32> [#uses=2]
	%tmp3 = shl i32 %indvar, 1		; <i32> [#uses=1]
	%level.0 = add i32 %tmp3, 7		; <i32> [#uses=3]
	%2162 = icmp sgt i32 %level.0, 9		; <i1> [#uses=1]
	br i1 %2162, label %bb33, label %bb18

bb33:		; preds = %bb32
	%2163 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([25 x i8]* @"\01LC2997", i32 0, i32 0), i32 %input_size.0) nounwind		; <i32> [#uses=0]
	ret i32 0
}

declare fastcc void @generateMTFValues() nounwind ssp

declare i32 @fprintf(%struct.FILE* nocapture, i8* nocapture, ...) nounwind

declare fastcc void @cleanUpAndFail(i32) noreturn nounwind ssp

declare void @exit(i32) noreturn nounwind

declare fastcc void @panic(i8*) noreturn nounwind ssp

declare i32 @"\01_fwrite$UNIX2003"(i8*, i32, i32, i8*)

declare fastcc void @blockOverrun() noreturn nounwind ssp

declare fastcc void @compressedStreamEOF() noreturn nounwind ssp

declare fastcc void @crcError(i32, i32) noreturn nounwind ssp

declare fastcc void @sortIt() nounwind ssp

declare fastcc i32 @bsGetUInt32() nounwind ssp

declare fastcc void @loadAndRLEsource() nounwind ssp

declare i32 @printf(i8* nocapture, ...) nounwind

declare i32 @puts(i8* nocapture) nounwind

declare void @llvm.memcpy.i32(i8* nocapture, i8* nocapture, i32, i32) nounwind

declare void @llvm.memset.i32(i8* nocapture, i8, i32, i32) nounwind

declare i32 @"\01_open$UNIX2003"(i8*, i32, ...)

declare i32* @__error()

declare i8* @"\01_strerror$UNIX2003"(i32)

declare i32 @read(...)

declare i32 @close(...)

declare i32 @atoi(i8* nocapture) nounwind readonly

declare void @llvm.memset.i64(i8* nocapture, i8, i64, i32) nounwind
