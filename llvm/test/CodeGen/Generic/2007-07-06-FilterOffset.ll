; RUN: llvm-as < %s | llc -enable-eh -asm-verbose -o - | \
; RUN:   grep {\\-4.*TypeInfo index}

target triple = "i686-pc-linux-gnu"
	%struct.__class_type_info_pseudo = type { %struct.__type_info_pseudo }
	%struct.__type_info_pseudo = type { i8*, i8* }
@_ZTI4a000 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a000, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a001 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a001, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a002 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a002, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a003 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a003, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a004 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a004, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a005 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a005, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a006 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a006, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a007 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a007, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a008 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a008, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a009 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a009, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a010 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a010, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a011 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a011, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a012 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a012, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a013 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a013, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a014 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a014, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a015 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a015, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a016 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a016, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a017 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a017, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a018 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a018, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a019 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a019, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a020 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a020, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a021 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a021, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a022 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a022, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a023 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a023, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a024 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a024, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a025 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a025, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a026 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a026, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a027 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a027, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a028 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a028, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a029 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a029, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a030 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a030, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a031 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a031, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a032 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a032, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a033 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a033, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a034 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a034, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a035 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a035, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a036 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a036, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a037 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a037, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a038 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a038, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a039 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a039, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a040 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a040, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a041 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a041, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a042 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a042, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a043 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a043, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a044 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a044, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a045 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a045, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a046 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a046, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a047 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a047, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a048 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a048, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a049 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a049, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a050 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a050, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a051 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a051, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a052 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a052, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a053 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a053, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a054 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a054, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a055 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a055, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a056 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a056, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a057 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a057, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a058 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a058, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a059 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a059, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a060 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a060, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a061 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a061, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a062 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a062, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a063 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a063, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a064 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a064, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a065 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a065, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a066 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a066, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a067 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a067, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a068 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a068, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a069 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a069, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a070 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a070, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a071 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a071, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a072 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a072, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a073 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a073, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a074 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a074, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a075 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a075, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a076 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a076, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a077 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a077, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a078 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a078, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a079 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a079, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a080 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a080, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a081 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a081, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a082 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a082, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a083 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a083, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a084 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a084, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a085 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a085, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a086 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a086, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a087 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a087, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a088 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a088, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a089 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a089, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a090 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a090, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a091 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a091, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a092 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a092, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a093 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a093, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a094 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a094, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a095 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a095, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a096 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a096, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a097 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a097, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a098 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a098, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a099 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a099, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a100 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a100, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a101 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a101, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a102 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a102, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a103 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a103, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a104 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a104, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a105 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a105, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a106 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a106, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a107 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a107, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a108 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a108, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a109 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a109, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a110 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a110, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a111 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a111, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a112 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a112, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a113 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a113, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a114 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a114, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a115 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a115, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a116 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a116, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a117 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a117, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a118 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a118, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a119 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a119, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a120 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a120, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a121 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a121, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a122 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a122, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a123 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a123, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a124 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a124, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a125 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a125, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a126 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a126, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a127 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a127, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTI4a128 = weak constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { i8* inttoptr (i32 add (i32 ptrtoint ([0 x i32 (...)*]* @_ZTVN10__cxxabiv117__class_type_infoE to i32), i32 8) to i8*), i8* getelementptr ([6 x i8]* @_ZTS4a128, i32 0, i32 0) } }		; <%struct.__class_type_info_pseudo*> [#uses=1]
@_ZTVN10__cxxabiv117__class_type_infoE = external constant [0 x i32 (...)*]		; <[0 x i32 (...)*]*> [#uses=1]
@_ZTS4a128 = weak constant [6 x i8] c"4a128\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a127 = weak constant [6 x i8] c"4a127\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a126 = weak constant [6 x i8] c"4a126\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a125 = weak constant [6 x i8] c"4a125\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a124 = weak constant [6 x i8] c"4a124\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a123 = weak constant [6 x i8] c"4a123\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a122 = weak constant [6 x i8] c"4a122\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a121 = weak constant [6 x i8] c"4a121\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a120 = weak constant [6 x i8] c"4a120\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a119 = weak constant [6 x i8] c"4a119\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a118 = weak constant [6 x i8] c"4a118\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a117 = weak constant [6 x i8] c"4a117\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a116 = weak constant [6 x i8] c"4a116\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a115 = weak constant [6 x i8] c"4a115\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a114 = weak constant [6 x i8] c"4a114\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a113 = weak constant [6 x i8] c"4a113\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a112 = weak constant [6 x i8] c"4a112\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a111 = weak constant [6 x i8] c"4a111\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a110 = weak constant [6 x i8] c"4a110\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a109 = weak constant [6 x i8] c"4a109\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a108 = weak constant [6 x i8] c"4a108\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a107 = weak constant [6 x i8] c"4a107\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a106 = weak constant [6 x i8] c"4a106\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a105 = weak constant [6 x i8] c"4a105\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a104 = weak constant [6 x i8] c"4a104\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a103 = weak constant [6 x i8] c"4a103\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a102 = weak constant [6 x i8] c"4a102\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a101 = weak constant [6 x i8] c"4a101\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a100 = weak constant [6 x i8] c"4a100\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a099 = weak constant [6 x i8] c"4a099\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a098 = weak constant [6 x i8] c"4a098\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a097 = weak constant [6 x i8] c"4a097\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a096 = weak constant [6 x i8] c"4a096\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a095 = weak constant [6 x i8] c"4a095\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a094 = weak constant [6 x i8] c"4a094\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a093 = weak constant [6 x i8] c"4a093\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a092 = weak constant [6 x i8] c"4a092\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a091 = weak constant [6 x i8] c"4a091\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a090 = weak constant [6 x i8] c"4a090\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a089 = weak constant [6 x i8] c"4a089\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a088 = weak constant [6 x i8] c"4a088\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a087 = weak constant [6 x i8] c"4a087\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a086 = weak constant [6 x i8] c"4a086\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a085 = weak constant [6 x i8] c"4a085\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a084 = weak constant [6 x i8] c"4a084\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a083 = weak constant [6 x i8] c"4a083\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a082 = weak constant [6 x i8] c"4a082\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a081 = weak constant [6 x i8] c"4a081\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a080 = weak constant [6 x i8] c"4a080\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a079 = weak constant [6 x i8] c"4a079\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a078 = weak constant [6 x i8] c"4a078\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a077 = weak constant [6 x i8] c"4a077\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a076 = weak constant [6 x i8] c"4a076\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a075 = weak constant [6 x i8] c"4a075\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a074 = weak constant [6 x i8] c"4a074\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a073 = weak constant [6 x i8] c"4a073\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a072 = weak constant [6 x i8] c"4a072\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a071 = weak constant [6 x i8] c"4a071\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a070 = weak constant [6 x i8] c"4a070\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a069 = weak constant [6 x i8] c"4a069\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a068 = weak constant [6 x i8] c"4a068\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a067 = weak constant [6 x i8] c"4a067\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a066 = weak constant [6 x i8] c"4a066\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a065 = weak constant [6 x i8] c"4a065\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a064 = weak constant [6 x i8] c"4a064\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a063 = weak constant [6 x i8] c"4a063\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a062 = weak constant [6 x i8] c"4a062\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a061 = weak constant [6 x i8] c"4a061\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a060 = weak constant [6 x i8] c"4a060\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a059 = weak constant [6 x i8] c"4a059\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a058 = weak constant [6 x i8] c"4a058\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a057 = weak constant [6 x i8] c"4a057\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a056 = weak constant [6 x i8] c"4a056\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a055 = weak constant [6 x i8] c"4a055\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a054 = weak constant [6 x i8] c"4a054\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a053 = weak constant [6 x i8] c"4a053\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a052 = weak constant [6 x i8] c"4a052\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a051 = weak constant [6 x i8] c"4a051\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a050 = weak constant [6 x i8] c"4a050\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a049 = weak constant [6 x i8] c"4a049\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a048 = weak constant [6 x i8] c"4a048\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a047 = weak constant [6 x i8] c"4a047\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a046 = weak constant [6 x i8] c"4a046\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a045 = weak constant [6 x i8] c"4a045\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a044 = weak constant [6 x i8] c"4a044\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a043 = weak constant [6 x i8] c"4a043\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a042 = weak constant [6 x i8] c"4a042\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a041 = weak constant [6 x i8] c"4a041\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a040 = weak constant [6 x i8] c"4a040\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a039 = weak constant [6 x i8] c"4a039\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a038 = weak constant [6 x i8] c"4a038\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a037 = weak constant [6 x i8] c"4a037\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a036 = weak constant [6 x i8] c"4a036\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a035 = weak constant [6 x i8] c"4a035\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a034 = weak constant [6 x i8] c"4a034\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a033 = weak constant [6 x i8] c"4a033\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a032 = weak constant [6 x i8] c"4a032\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a031 = weak constant [6 x i8] c"4a031\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a030 = weak constant [6 x i8] c"4a030\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a029 = weak constant [6 x i8] c"4a029\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a028 = weak constant [6 x i8] c"4a028\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a027 = weak constant [6 x i8] c"4a027\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a026 = weak constant [6 x i8] c"4a026\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a025 = weak constant [6 x i8] c"4a025\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a024 = weak constant [6 x i8] c"4a024\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a023 = weak constant [6 x i8] c"4a023\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a022 = weak constant [6 x i8] c"4a022\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a021 = weak constant [6 x i8] c"4a021\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a020 = weak constant [6 x i8] c"4a020\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a019 = weak constant [6 x i8] c"4a019\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a018 = weak constant [6 x i8] c"4a018\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a017 = weak constant [6 x i8] c"4a017\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a016 = weak constant [6 x i8] c"4a016\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a015 = weak constant [6 x i8] c"4a015\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a014 = weak constant [6 x i8] c"4a014\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a013 = weak constant [6 x i8] c"4a013\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a012 = weak constant [6 x i8] c"4a012\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a011 = weak constant [6 x i8] c"4a011\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a010 = weak constant [6 x i8] c"4a010\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a009 = weak constant [6 x i8] c"4a009\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a008 = weak constant [6 x i8] c"4a008\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a007 = weak constant [6 x i8] c"4a007\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a006 = weak constant [6 x i8] c"4a006\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a005 = weak constant [6 x i8] c"4a005\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a004 = weak constant [6 x i8] c"4a004\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a003 = weak constant [6 x i8] c"4a003\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a002 = weak constant [6 x i8] c"4a002\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a001 = weak constant [6 x i8] c"4a001\00"		; <[6 x i8]*> [#uses=1]
@_ZTS4a000 = weak constant [6 x i8] c"4a000\00"		; <[6 x i8]*> [#uses=1]

declare void @_Z1Nv()

declare i8* @llvm.eh.exception()

declare i32 @llvm.eh.selector(i8*, i8*, ...)

declare i32 @llvm.eh.typeid.for(i8*)

declare i32 @__gxx_personality_v0(...)

declare i32 @_Unwind_Resume(...)

declare void @__cxa_call_unexpected(i8*)

define void @_Z1Qv() {
entry:
	invoke void @_Z1Nv( )
			to label %UnifiedReturnBlock2 unwind label %unwind

unwind:		; preds = %entry
	%eh_ptr = tail call i8* @llvm.eh.exception( )		; <i8*> [#uses=3]
	%eh_select = tail call i32 (i8*, i8*, ...)* @llvm.eh.selector( i8* %eh_ptr, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1, i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a000 to i8*), i32 1, i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a001 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a000 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a001 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a002 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a003 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a004 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a005 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a006 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a007 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a008 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a009 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a010 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a011 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a012 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a013 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a014 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a015 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a016 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a017 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a018 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a019 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a020 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a021 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a022 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a023 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a024 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a025 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a026 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a027 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a028 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a029 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a030 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a031 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a032 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a033 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a034 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a035 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a036 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a037 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a038 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a039 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a040 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a041 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a042 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a043 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a044 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a045 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a046 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a047 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a048 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a049 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a050 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a051 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a052 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a053 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a054 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a055 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a056 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a057 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a058 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a059 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a060 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a061 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a062 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a063 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a064 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a065 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a066 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a067 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a068 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a069 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a070 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a071 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a072 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a073 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a074 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a075 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a076 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a077 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a078 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a079 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a080 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a081 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a082 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a083 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a084 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a085 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a086 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a087 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a088 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a089 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a090 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a091 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a092 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a093 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a094 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a095 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a096 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a097 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a098 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a099 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a100 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a101 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a102 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a103 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a104 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a105 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a106 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a107 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a108 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a109 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a110 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a111 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a112 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a113 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a114 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a115 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a116 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a117 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a118 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a119 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a120 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a121 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a122 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a123 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a124 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a125 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a126 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a127 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a128 to i8*) )		; <i32> [#uses=2]
	%tmp260 = icmp slt i32 %eh_select, 0		; <i1> [#uses=1]
	br i1 %tmp260, label %filter, label %cleanup279

filter:		; preds = %unwind
	invoke void @__cxa_call_unexpected( i8* %eh_ptr )
			to label %UnifiedUnreachableBlock1 unwind label %unwind261

unwind261:		; preds = %filter
	%eh_ptr262 = tail call i8* @llvm.eh.exception( )		; <i8*> [#uses=3]
	%eh_select264 = tail call i32 (i8*, i8*, ...)* @llvm.eh.selector( i8* %eh_ptr262, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1, i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a001 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a000 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a001 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a002 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a003 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a004 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a005 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a006 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a007 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a008 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a009 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a010 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a011 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a012 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a013 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a014 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a015 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a016 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a017 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a018 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a019 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a020 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a021 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a022 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a023 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a024 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a025 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a026 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a027 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a028 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a029 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a030 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a031 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a032 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a033 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a034 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a035 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a036 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a037 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a038 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a039 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a040 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a041 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a042 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a043 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a044 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a045 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a046 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a047 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a048 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a049 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a050 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a051 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a052 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a053 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a054 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a055 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a056 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a057 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a058 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a059 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a060 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a061 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a062 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a063 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a064 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a065 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a066 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a067 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a068 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a069 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a070 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a071 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a072 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a073 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a074 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a075 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a076 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a077 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a078 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a079 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a080 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a081 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a082 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a083 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a084 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a085 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a086 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a087 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a088 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a089 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a090 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a091 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a092 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a093 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a094 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a095 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a096 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a097 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a098 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a099 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a100 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a101 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a102 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a103 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a104 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a105 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a106 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a107 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a108 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a109 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a110 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a111 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a112 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a113 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a114 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a115 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a116 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a117 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a118 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a119 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a120 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a121 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a122 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a123 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a124 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a125 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a126 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a127 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a128 to i8*) )		; <i32> [#uses=2]
	%tmp2692602 = icmp slt i32 %eh_select264, 0		; <i1> [#uses=1]
	br i1 %tmp2692602, label %filter270, label %cleanup279

filter270:		; preds = %unwind261
	invoke void @__cxa_call_unexpected( i8* %eh_ptr262 )
			to label %UnifiedUnreachableBlock1 unwind label %unwind272

unwind272:		; preds = %filter270
	%eh_ptr273 = tail call i8* @llvm.eh.exception( )		; <i8*> [#uses=3]
	%eh_select275 = tail call i32 (i8*, i8*, ...)* @llvm.eh.selector( i8* %eh_ptr273, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a000 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a001 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a002 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a003 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a004 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a005 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a006 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a007 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a008 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a009 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a010 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a011 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a012 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a013 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a014 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a015 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a016 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a017 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a018 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a019 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a020 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a021 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a022 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a023 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a024 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a025 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a026 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a027 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a028 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a029 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a030 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a031 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a032 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a033 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a034 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a035 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a036 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a037 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a038 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a039 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a040 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a041 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a042 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a043 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a044 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a045 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a046 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a047 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a048 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a049 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a050 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a051 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a052 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a053 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a054 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a055 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a056 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a057 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a058 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a059 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a060 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a061 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a062 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a063 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a064 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a065 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a066 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a067 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a068 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a069 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a070 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a071 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a072 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a073 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a074 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a075 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a076 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a077 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a078 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a079 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a080 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a081 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a082 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a083 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a084 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a085 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a086 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a087 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a088 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a089 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a090 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a091 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a092 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a093 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a094 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a095 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a096 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a097 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a098 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a099 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a100 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a101 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a102 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a103 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a104 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a105 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a106 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a107 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a108 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a109 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a110 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a111 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a112 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a113 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a114 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a115 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a116 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a117 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a118 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a119 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a120 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a121 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a122 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a123 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a124 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a125 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a126 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a127 to i8*), i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a128 to i8*) )		; <i32> [#uses=2]
	%eh_typeid2863 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a000 to i8*) )		; <i32> [#uses=1]
	%tmp2812865 = icmp eq i32 %eh_select275, %eh_typeid2863		; <i1> [#uses=1]
	br i1 %tmp2812865, label %eh_then, label %eh_else

cleanup279:		; preds = %unwind261, %unwind
	%eh_exception.1 = phi i8* [ %eh_ptr, %unwind ], [ %eh_ptr262, %unwind261 ]		; <i8*> [#uses=2]
	%eh_selector.1 = phi i32 [ %eh_select, %unwind ], [ %eh_select264, %unwind261 ]		; <i32> [#uses=2]
	%eh_typeid = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a000 to i8*) )		; <i32> [#uses=1]
	%tmp281 = icmp eq i32 %eh_selector.1, %eh_typeid		; <i1> [#uses=1]
	br i1 %tmp281, label %eh_then, label %eh_else

eh_then:		; preds = %cleanup279, %unwind272
	%eh_exception.12604.0 = phi i8* [ %eh_ptr273, %unwind272 ], [ %eh_exception.1, %cleanup279 ]		; <i8*> [#uses=1]
	%tmp284 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.0 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else:		; preds = %cleanup279, %unwind272
	%eh_exception.12604.1 = phi i8* [ %eh_ptr273, %unwind272 ], [ %eh_exception.1, %cleanup279 ]		; <i8*> [#uses=129]
	%eh_selector.12734.1 = phi i32 [ %eh_select275, %unwind272 ], [ %eh_selector.1, %cleanup279 ]		; <i32> [#uses=128]
	%eh_typeid295 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a001 to i8*) )		; <i32> [#uses=1]
	%tmp297 = icmp eq i32 %eh_selector.12734.1, %eh_typeid295		; <i1> [#uses=1]
	br i1 %tmp297, label %eh_then298, label %eh_else312

eh_then298:		; preds = %eh_else
	%tmp301 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else312:		; preds = %eh_else
	%eh_typeid313 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a002 to i8*) )		; <i32> [#uses=1]
	%tmp315 = icmp eq i32 %eh_selector.12734.1, %eh_typeid313		; <i1> [#uses=1]
	br i1 %tmp315, label %eh_then316, label %eh_else330

eh_then316:		; preds = %eh_else312
	%tmp319 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else330:		; preds = %eh_else312
	%eh_typeid331 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a003 to i8*) )		; <i32> [#uses=1]
	%tmp333 = icmp eq i32 %eh_selector.12734.1, %eh_typeid331		; <i1> [#uses=1]
	br i1 %tmp333, label %eh_then334, label %eh_else348

eh_then334:		; preds = %eh_else330
	%tmp337 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else348:		; preds = %eh_else330
	%eh_typeid349 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a004 to i8*) )		; <i32> [#uses=1]
	%tmp351 = icmp eq i32 %eh_selector.12734.1, %eh_typeid349		; <i1> [#uses=1]
	br i1 %tmp351, label %eh_then352, label %eh_else366

eh_then352:		; preds = %eh_else348
	%tmp355 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else366:		; preds = %eh_else348
	%eh_typeid367 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a005 to i8*) )		; <i32> [#uses=1]
	%tmp369 = icmp eq i32 %eh_selector.12734.1, %eh_typeid367		; <i1> [#uses=1]
	br i1 %tmp369, label %eh_then370, label %eh_else384

eh_then370:		; preds = %eh_else366
	%tmp373 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else384:		; preds = %eh_else366
	%eh_typeid385 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a006 to i8*) )		; <i32> [#uses=1]
	%tmp387 = icmp eq i32 %eh_selector.12734.1, %eh_typeid385		; <i1> [#uses=1]
	br i1 %tmp387, label %eh_then388, label %eh_else402

eh_then388:		; preds = %eh_else384
	%tmp391 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else402:		; preds = %eh_else384
	%eh_typeid403 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a007 to i8*) )		; <i32> [#uses=1]
	%tmp405 = icmp eq i32 %eh_selector.12734.1, %eh_typeid403		; <i1> [#uses=1]
	br i1 %tmp405, label %eh_then406, label %eh_else420

eh_then406:		; preds = %eh_else402
	%tmp409 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else420:		; preds = %eh_else402
	%eh_typeid421 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a008 to i8*) )		; <i32> [#uses=1]
	%tmp423 = icmp eq i32 %eh_selector.12734.1, %eh_typeid421		; <i1> [#uses=1]
	br i1 %tmp423, label %eh_then424, label %eh_else438

eh_then424:		; preds = %eh_else420
	%tmp427 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else438:		; preds = %eh_else420
	%eh_typeid439 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a009 to i8*) )		; <i32> [#uses=1]
	%tmp441 = icmp eq i32 %eh_selector.12734.1, %eh_typeid439		; <i1> [#uses=1]
	br i1 %tmp441, label %eh_then442, label %eh_else456

eh_then442:		; preds = %eh_else438
	%tmp445 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else456:		; preds = %eh_else438
	%eh_typeid457 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a010 to i8*) )		; <i32> [#uses=1]
	%tmp459 = icmp eq i32 %eh_selector.12734.1, %eh_typeid457		; <i1> [#uses=1]
	br i1 %tmp459, label %eh_then460, label %eh_else474

eh_then460:		; preds = %eh_else456
	%tmp463 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else474:		; preds = %eh_else456
	%eh_typeid475 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a011 to i8*) )		; <i32> [#uses=1]
	%tmp477 = icmp eq i32 %eh_selector.12734.1, %eh_typeid475		; <i1> [#uses=1]
	br i1 %tmp477, label %eh_then478, label %eh_else492

eh_then478:		; preds = %eh_else474
	%tmp481 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else492:		; preds = %eh_else474
	%eh_typeid493 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a012 to i8*) )		; <i32> [#uses=1]
	%tmp495 = icmp eq i32 %eh_selector.12734.1, %eh_typeid493		; <i1> [#uses=1]
	br i1 %tmp495, label %eh_then496, label %eh_else510

eh_then496:		; preds = %eh_else492
	%tmp499 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else510:		; preds = %eh_else492
	%eh_typeid511 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a013 to i8*) )		; <i32> [#uses=1]
	%tmp513 = icmp eq i32 %eh_selector.12734.1, %eh_typeid511		; <i1> [#uses=1]
	br i1 %tmp513, label %eh_then514, label %eh_else528

eh_then514:		; preds = %eh_else510
	%tmp517 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else528:		; preds = %eh_else510
	%eh_typeid529 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a014 to i8*) )		; <i32> [#uses=1]
	%tmp531 = icmp eq i32 %eh_selector.12734.1, %eh_typeid529		; <i1> [#uses=1]
	br i1 %tmp531, label %eh_then532, label %eh_else546

eh_then532:		; preds = %eh_else528
	%tmp535 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else546:		; preds = %eh_else528
	%eh_typeid547 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a015 to i8*) )		; <i32> [#uses=1]
	%tmp549 = icmp eq i32 %eh_selector.12734.1, %eh_typeid547		; <i1> [#uses=1]
	br i1 %tmp549, label %eh_then550, label %eh_else564

eh_then550:		; preds = %eh_else546
	%tmp553 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else564:		; preds = %eh_else546
	%eh_typeid565 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a016 to i8*) )		; <i32> [#uses=1]
	%tmp567 = icmp eq i32 %eh_selector.12734.1, %eh_typeid565		; <i1> [#uses=1]
	br i1 %tmp567, label %eh_then568, label %eh_else582

eh_then568:		; preds = %eh_else564
	%tmp571 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else582:		; preds = %eh_else564
	%eh_typeid583 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a017 to i8*) )		; <i32> [#uses=1]
	%tmp585 = icmp eq i32 %eh_selector.12734.1, %eh_typeid583		; <i1> [#uses=1]
	br i1 %tmp585, label %eh_then586, label %eh_else600

eh_then586:		; preds = %eh_else582
	%tmp589 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else600:		; preds = %eh_else582
	%eh_typeid601 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a018 to i8*) )		; <i32> [#uses=1]
	%tmp603 = icmp eq i32 %eh_selector.12734.1, %eh_typeid601		; <i1> [#uses=1]
	br i1 %tmp603, label %eh_then604, label %eh_else618

eh_then604:		; preds = %eh_else600
	%tmp607 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else618:		; preds = %eh_else600
	%eh_typeid619 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a019 to i8*) )		; <i32> [#uses=1]
	%tmp621 = icmp eq i32 %eh_selector.12734.1, %eh_typeid619		; <i1> [#uses=1]
	br i1 %tmp621, label %eh_then622, label %eh_else636

eh_then622:		; preds = %eh_else618
	%tmp625 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else636:		; preds = %eh_else618
	%eh_typeid637 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a020 to i8*) )		; <i32> [#uses=1]
	%tmp639 = icmp eq i32 %eh_selector.12734.1, %eh_typeid637		; <i1> [#uses=1]
	br i1 %tmp639, label %eh_then640, label %eh_else654

eh_then640:		; preds = %eh_else636
	%tmp643 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else654:		; preds = %eh_else636
	%eh_typeid655 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a021 to i8*) )		; <i32> [#uses=1]
	%tmp657 = icmp eq i32 %eh_selector.12734.1, %eh_typeid655		; <i1> [#uses=1]
	br i1 %tmp657, label %eh_then658, label %eh_else672

eh_then658:		; preds = %eh_else654
	%tmp661 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else672:		; preds = %eh_else654
	%eh_typeid673 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a022 to i8*) )		; <i32> [#uses=1]
	%tmp675 = icmp eq i32 %eh_selector.12734.1, %eh_typeid673		; <i1> [#uses=1]
	br i1 %tmp675, label %eh_then676, label %eh_else690

eh_then676:		; preds = %eh_else672
	%tmp679 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else690:		; preds = %eh_else672
	%eh_typeid691 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a023 to i8*) )		; <i32> [#uses=1]
	%tmp693 = icmp eq i32 %eh_selector.12734.1, %eh_typeid691		; <i1> [#uses=1]
	br i1 %tmp693, label %eh_then694, label %eh_else708

eh_then694:		; preds = %eh_else690
	%tmp697 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else708:		; preds = %eh_else690
	%eh_typeid709 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a024 to i8*) )		; <i32> [#uses=1]
	%tmp711 = icmp eq i32 %eh_selector.12734.1, %eh_typeid709		; <i1> [#uses=1]
	br i1 %tmp711, label %eh_then712, label %eh_else726

eh_then712:		; preds = %eh_else708
	%tmp715 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else726:		; preds = %eh_else708
	%eh_typeid727 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a025 to i8*) )		; <i32> [#uses=1]
	%tmp729 = icmp eq i32 %eh_selector.12734.1, %eh_typeid727		; <i1> [#uses=1]
	br i1 %tmp729, label %eh_then730, label %eh_else744

eh_then730:		; preds = %eh_else726
	%tmp733 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else744:		; preds = %eh_else726
	%eh_typeid745 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a026 to i8*) )		; <i32> [#uses=1]
	%tmp747 = icmp eq i32 %eh_selector.12734.1, %eh_typeid745		; <i1> [#uses=1]
	br i1 %tmp747, label %eh_then748, label %eh_else762

eh_then748:		; preds = %eh_else744
	%tmp751 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else762:		; preds = %eh_else744
	%eh_typeid763 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a027 to i8*) )		; <i32> [#uses=1]
	%tmp765 = icmp eq i32 %eh_selector.12734.1, %eh_typeid763		; <i1> [#uses=1]
	br i1 %tmp765, label %eh_then766, label %eh_else780

eh_then766:		; preds = %eh_else762
	%tmp769 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else780:		; preds = %eh_else762
	%eh_typeid781 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a028 to i8*) )		; <i32> [#uses=1]
	%tmp783 = icmp eq i32 %eh_selector.12734.1, %eh_typeid781		; <i1> [#uses=1]
	br i1 %tmp783, label %eh_then784, label %eh_else798

eh_then784:		; preds = %eh_else780
	%tmp787 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else798:		; preds = %eh_else780
	%eh_typeid799 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a029 to i8*) )		; <i32> [#uses=1]
	%tmp801 = icmp eq i32 %eh_selector.12734.1, %eh_typeid799		; <i1> [#uses=1]
	br i1 %tmp801, label %eh_then802, label %eh_else816

eh_then802:		; preds = %eh_else798
	%tmp805 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else816:		; preds = %eh_else798
	%eh_typeid817 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a030 to i8*) )		; <i32> [#uses=1]
	%tmp819 = icmp eq i32 %eh_selector.12734.1, %eh_typeid817		; <i1> [#uses=1]
	br i1 %tmp819, label %eh_then820, label %eh_else834

eh_then820:		; preds = %eh_else816
	%tmp823 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else834:		; preds = %eh_else816
	%eh_typeid835 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a031 to i8*) )		; <i32> [#uses=1]
	%tmp837 = icmp eq i32 %eh_selector.12734.1, %eh_typeid835		; <i1> [#uses=1]
	br i1 %tmp837, label %eh_then838, label %eh_else852

eh_then838:		; preds = %eh_else834
	%tmp841 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else852:		; preds = %eh_else834
	%eh_typeid853 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a032 to i8*) )		; <i32> [#uses=1]
	%tmp855 = icmp eq i32 %eh_selector.12734.1, %eh_typeid853		; <i1> [#uses=1]
	br i1 %tmp855, label %eh_then856, label %eh_else870

eh_then856:		; preds = %eh_else852
	%tmp859 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else870:		; preds = %eh_else852
	%eh_typeid871 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a033 to i8*) )		; <i32> [#uses=1]
	%tmp873 = icmp eq i32 %eh_selector.12734.1, %eh_typeid871		; <i1> [#uses=1]
	br i1 %tmp873, label %eh_then874, label %eh_else888

eh_then874:		; preds = %eh_else870
	%tmp877 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else888:		; preds = %eh_else870
	%eh_typeid889 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a034 to i8*) )		; <i32> [#uses=1]
	%tmp891 = icmp eq i32 %eh_selector.12734.1, %eh_typeid889		; <i1> [#uses=1]
	br i1 %tmp891, label %eh_then892, label %eh_else906

eh_then892:		; preds = %eh_else888
	%tmp895 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else906:		; preds = %eh_else888
	%eh_typeid907 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a035 to i8*) )		; <i32> [#uses=1]
	%tmp909 = icmp eq i32 %eh_selector.12734.1, %eh_typeid907		; <i1> [#uses=1]
	br i1 %tmp909, label %eh_then910, label %eh_else924

eh_then910:		; preds = %eh_else906
	%tmp913 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else924:		; preds = %eh_else906
	%eh_typeid925 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a036 to i8*) )		; <i32> [#uses=1]
	%tmp927 = icmp eq i32 %eh_selector.12734.1, %eh_typeid925		; <i1> [#uses=1]
	br i1 %tmp927, label %eh_then928, label %eh_else942

eh_then928:		; preds = %eh_else924
	%tmp931 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else942:		; preds = %eh_else924
	%eh_typeid943 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a037 to i8*) )		; <i32> [#uses=1]
	%tmp945 = icmp eq i32 %eh_selector.12734.1, %eh_typeid943		; <i1> [#uses=1]
	br i1 %tmp945, label %eh_then946, label %eh_else960

eh_then946:		; preds = %eh_else942
	%tmp949 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else960:		; preds = %eh_else942
	%eh_typeid961 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a038 to i8*) )		; <i32> [#uses=1]
	%tmp963 = icmp eq i32 %eh_selector.12734.1, %eh_typeid961		; <i1> [#uses=1]
	br i1 %tmp963, label %eh_then964, label %eh_else978

eh_then964:		; preds = %eh_else960
	%tmp967 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else978:		; preds = %eh_else960
	%eh_typeid979 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a039 to i8*) )		; <i32> [#uses=1]
	%tmp981 = icmp eq i32 %eh_selector.12734.1, %eh_typeid979		; <i1> [#uses=1]
	br i1 %tmp981, label %eh_then982, label %eh_else996

eh_then982:		; preds = %eh_else978
	%tmp985 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else996:		; preds = %eh_else978
	%eh_typeid997 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a040 to i8*) )		; <i32> [#uses=1]
	%tmp999 = icmp eq i32 %eh_selector.12734.1, %eh_typeid997		; <i1> [#uses=1]
	br i1 %tmp999, label %eh_then1000, label %eh_else1014

eh_then1000:		; preds = %eh_else996
	%tmp1003 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1014:		; preds = %eh_else996
	%eh_typeid1015 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a041 to i8*) )		; <i32> [#uses=1]
	%tmp1017 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1015		; <i1> [#uses=1]
	br i1 %tmp1017, label %eh_then1018, label %eh_else1032

eh_then1018:		; preds = %eh_else1014
	%tmp1021 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1032:		; preds = %eh_else1014
	%eh_typeid1033 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a042 to i8*) )		; <i32> [#uses=1]
	%tmp1035 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1033		; <i1> [#uses=1]
	br i1 %tmp1035, label %eh_then1036, label %eh_else1050

eh_then1036:		; preds = %eh_else1032
	%tmp1039 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1050:		; preds = %eh_else1032
	%eh_typeid1051 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a043 to i8*) )		; <i32> [#uses=1]
	%tmp1053 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1051		; <i1> [#uses=1]
	br i1 %tmp1053, label %eh_then1054, label %eh_else1068

eh_then1054:		; preds = %eh_else1050
	%tmp1057 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1068:		; preds = %eh_else1050
	%eh_typeid1069 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a044 to i8*) )		; <i32> [#uses=1]
	%tmp1071 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1069		; <i1> [#uses=1]
	br i1 %tmp1071, label %eh_then1072, label %eh_else1086

eh_then1072:		; preds = %eh_else1068
	%tmp1075 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1086:		; preds = %eh_else1068
	%eh_typeid1087 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a045 to i8*) )		; <i32> [#uses=1]
	%tmp1089 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1087		; <i1> [#uses=1]
	br i1 %tmp1089, label %eh_then1090, label %eh_else1104

eh_then1090:		; preds = %eh_else1086
	%tmp1093 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1104:		; preds = %eh_else1086
	%eh_typeid1105 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a046 to i8*) )		; <i32> [#uses=1]
	%tmp1107 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1105		; <i1> [#uses=1]
	br i1 %tmp1107, label %eh_then1108, label %eh_else1122

eh_then1108:		; preds = %eh_else1104
	%tmp1111 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1122:		; preds = %eh_else1104
	%eh_typeid1123 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a047 to i8*) )		; <i32> [#uses=1]
	%tmp1125 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1123		; <i1> [#uses=1]
	br i1 %tmp1125, label %eh_then1126, label %eh_else1140

eh_then1126:		; preds = %eh_else1122
	%tmp1129 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1140:		; preds = %eh_else1122
	%eh_typeid1141 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a048 to i8*) )		; <i32> [#uses=1]
	%tmp1143 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1141		; <i1> [#uses=1]
	br i1 %tmp1143, label %eh_then1144, label %eh_else1158

eh_then1144:		; preds = %eh_else1140
	%tmp1147 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1158:		; preds = %eh_else1140
	%eh_typeid1159 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a049 to i8*) )		; <i32> [#uses=1]
	%tmp1161 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1159		; <i1> [#uses=1]
	br i1 %tmp1161, label %eh_then1162, label %eh_else1176

eh_then1162:		; preds = %eh_else1158
	%tmp1165 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1176:		; preds = %eh_else1158
	%eh_typeid1177 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a050 to i8*) )		; <i32> [#uses=1]
	%tmp1179 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1177		; <i1> [#uses=1]
	br i1 %tmp1179, label %eh_then1180, label %eh_else1194

eh_then1180:		; preds = %eh_else1176
	%tmp1183 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1194:		; preds = %eh_else1176
	%eh_typeid1195 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a051 to i8*) )		; <i32> [#uses=1]
	%tmp1197 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1195		; <i1> [#uses=1]
	br i1 %tmp1197, label %eh_then1198, label %eh_else1212

eh_then1198:		; preds = %eh_else1194
	%tmp1201 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1212:		; preds = %eh_else1194
	%eh_typeid1213 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a052 to i8*) )		; <i32> [#uses=1]
	%tmp1215 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1213		; <i1> [#uses=1]
	br i1 %tmp1215, label %eh_then1216, label %eh_else1230

eh_then1216:		; preds = %eh_else1212
	%tmp1219 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1230:		; preds = %eh_else1212
	%eh_typeid1231 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a053 to i8*) )		; <i32> [#uses=1]
	%tmp1233 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1231		; <i1> [#uses=1]
	br i1 %tmp1233, label %eh_then1234, label %eh_else1248

eh_then1234:		; preds = %eh_else1230
	%tmp1237 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1248:		; preds = %eh_else1230
	%eh_typeid1249 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a054 to i8*) )		; <i32> [#uses=1]
	%tmp1251 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1249		; <i1> [#uses=1]
	br i1 %tmp1251, label %eh_then1252, label %eh_else1266

eh_then1252:		; preds = %eh_else1248
	%tmp1255 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1266:		; preds = %eh_else1248
	%eh_typeid1267 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a055 to i8*) )		; <i32> [#uses=1]
	%tmp1269 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1267		; <i1> [#uses=1]
	br i1 %tmp1269, label %eh_then1270, label %eh_else1284

eh_then1270:		; preds = %eh_else1266
	%tmp1273 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1284:		; preds = %eh_else1266
	%eh_typeid1285 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a056 to i8*) )		; <i32> [#uses=1]
	%tmp1287 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1285		; <i1> [#uses=1]
	br i1 %tmp1287, label %eh_then1288, label %eh_else1302

eh_then1288:		; preds = %eh_else1284
	%tmp1291 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1302:		; preds = %eh_else1284
	%eh_typeid1303 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a057 to i8*) )		; <i32> [#uses=1]
	%tmp1305 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1303		; <i1> [#uses=1]
	br i1 %tmp1305, label %eh_then1306, label %eh_else1320

eh_then1306:		; preds = %eh_else1302
	%tmp1309 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1320:		; preds = %eh_else1302
	%eh_typeid1321 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a058 to i8*) )		; <i32> [#uses=1]
	%tmp1323 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1321		; <i1> [#uses=1]
	br i1 %tmp1323, label %eh_then1324, label %eh_else1338

eh_then1324:		; preds = %eh_else1320
	%tmp1327 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1338:		; preds = %eh_else1320
	%eh_typeid1339 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a059 to i8*) )		; <i32> [#uses=1]
	%tmp1341 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1339		; <i1> [#uses=1]
	br i1 %tmp1341, label %eh_then1342, label %eh_else1356

eh_then1342:		; preds = %eh_else1338
	%tmp1345 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1356:		; preds = %eh_else1338
	%eh_typeid1357 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a060 to i8*) )		; <i32> [#uses=1]
	%tmp1359 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1357		; <i1> [#uses=1]
	br i1 %tmp1359, label %eh_then1360, label %eh_else1374

eh_then1360:		; preds = %eh_else1356
	%tmp1363 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1374:		; preds = %eh_else1356
	%eh_typeid1375 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a061 to i8*) )		; <i32> [#uses=1]
	%tmp1377 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1375		; <i1> [#uses=1]
	br i1 %tmp1377, label %eh_then1378, label %eh_else1392

eh_then1378:		; preds = %eh_else1374
	%tmp1381 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1392:		; preds = %eh_else1374
	%eh_typeid1393 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a062 to i8*) )		; <i32> [#uses=1]
	%tmp1395 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1393		; <i1> [#uses=1]
	br i1 %tmp1395, label %eh_then1396, label %eh_else1410

eh_then1396:		; preds = %eh_else1392
	%tmp1399 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1410:		; preds = %eh_else1392
	%eh_typeid1411 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a063 to i8*) )		; <i32> [#uses=1]
	%tmp1413 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1411		; <i1> [#uses=1]
	br i1 %tmp1413, label %eh_then1414, label %eh_else1428

eh_then1414:		; preds = %eh_else1410
	%tmp1417 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1428:		; preds = %eh_else1410
	%eh_typeid1429 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a064 to i8*) )		; <i32> [#uses=1]
	%tmp1431 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1429		; <i1> [#uses=1]
	br i1 %tmp1431, label %eh_then1432, label %eh_else1446

eh_then1432:		; preds = %eh_else1428
	%tmp1435 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1446:		; preds = %eh_else1428
	%eh_typeid1447 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a065 to i8*) )		; <i32> [#uses=1]
	%tmp1449 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1447		; <i1> [#uses=1]
	br i1 %tmp1449, label %eh_then1450, label %eh_else1464

eh_then1450:		; preds = %eh_else1446
	%tmp1453 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1464:		; preds = %eh_else1446
	%eh_typeid1465 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a066 to i8*) )		; <i32> [#uses=1]
	%tmp1467 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1465		; <i1> [#uses=1]
	br i1 %tmp1467, label %eh_then1468, label %eh_else1482

eh_then1468:		; preds = %eh_else1464
	%tmp1471 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1482:		; preds = %eh_else1464
	%eh_typeid1483 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a067 to i8*) )		; <i32> [#uses=1]
	%tmp1485 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1483		; <i1> [#uses=1]
	br i1 %tmp1485, label %eh_then1486, label %eh_else1500

eh_then1486:		; preds = %eh_else1482
	%tmp1489 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1500:		; preds = %eh_else1482
	%eh_typeid1501 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a068 to i8*) )		; <i32> [#uses=1]
	%tmp1503 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1501		; <i1> [#uses=1]
	br i1 %tmp1503, label %eh_then1504, label %eh_else1518

eh_then1504:		; preds = %eh_else1500
	%tmp1507 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1518:		; preds = %eh_else1500
	%eh_typeid1519 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a069 to i8*) )		; <i32> [#uses=1]
	%tmp1521 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1519		; <i1> [#uses=1]
	br i1 %tmp1521, label %eh_then1522, label %eh_else1536

eh_then1522:		; preds = %eh_else1518
	%tmp1525 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1536:		; preds = %eh_else1518
	%eh_typeid1537 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a070 to i8*) )		; <i32> [#uses=1]
	%tmp1539 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1537		; <i1> [#uses=1]
	br i1 %tmp1539, label %eh_then1540, label %eh_else1554

eh_then1540:		; preds = %eh_else1536
	%tmp1543 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1554:		; preds = %eh_else1536
	%eh_typeid1555 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a071 to i8*) )		; <i32> [#uses=1]
	%tmp1557 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1555		; <i1> [#uses=1]
	br i1 %tmp1557, label %eh_then1558, label %eh_else1572

eh_then1558:		; preds = %eh_else1554
	%tmp1561 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1572:		; preds = %eh_else1554
	%eh_typeid1573 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a072 to i8*) )		; <i32> [#uses=1]
	%tmp1575 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1573		; <i1> [#uses=1]
	br i1 %tmp1575, label %eh_then1576, label %eh_else1590

eh_then1576:		; preds = %eh_else1572
	%tmp1579 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1590:		; preds = %eh_else1572
	%eh_typeid1591 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a073 to i8*) )		; <i32> [#uses=1]
	%tmp1593 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1591		; <i1> [#uses=1]
	br i1 %tmp1593, label %eh_then1594, label %eh_else1608

eh_then1594:		; preds = %eh_else1590
	%tmp1597 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1608:		; preds = %eh_else1590
	%eh_typeid1609 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a074 to i8*) )		; <i32> [#uses=1]
	%tmp1611 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1609		; <i1> [#uses=1]
	br i1 %tmp1611, label %eh_then1612, label %eh_else1626

eh_then1612:		; preds = %eh_else1608
	%tmp1615 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1626:		; preds = %eh_else1608
	%eh_typeid1627 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a075 to i8*) )		; <i32> [#uses=1]
	%tmp1629 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1627		; <i1> [#uses=1]
	br i1 %tmp1629, label %eh_then1630, label %eh_else1644

eh_then1630:		; preds = %eh_else1626
	%tmp1633 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1644:		; preds = %eh_else1626
	%eh_typeid1645 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a076 to i8*) )		; <i32> [#uses=1]
	%tmp1647 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1645		; <i1> [#uses=1]
	br i1 %tmp1647, label %eh_then1648, label %eh_else1662

eh_then1648:		; preds = %eh_else1644
	%tmp1651 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1662:		; preds = %eh_else1644
	%eh_typeid1663 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a077 to i8*) )		; <i32> [#uses=1]
	%tmp1665 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1663		; <i1> [#uses=1]
	br i1 %tmp1665, label %eh_then1666, label %eh_else1680

eh_then1666:		; preds = %eh_else1662
	%tmp1669 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1680:		; preds = %eh_else1662
	%eh_typeid1681 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a078 to i8*) )		; <i32> [#uses=1]
	%tmp1683 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1681		; <i1> [#uses=1]
	br i1 %tmp1683, label %eh_then1684, label %eh_else1698

eh_then1684:		; preds = %eh_else1680
	%tmp1687 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1698:		; preds = %eh_else1680
	%eh_typeid1699 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a079 to i8*) )		; <i32> [#uses=1]
	%tmp1701 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1699		; <i1> [#uses=1]
	br i1 %tmp1701, label %eh_then1702, label %eh_else1716

eh_then1702:		; preds = %eh_else1698
	%tmp1705 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1716:		; preds = %eh_else1698
	%eh_typeid1717 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a080 to i8*) )		; <i32> [#uses=1]
	%tmp1719 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1717		; <i1> [#uses=1]
	br i1 %tmp1719, label %eh_then1720, label %eh_else1734

eh_then1720:		; preds = %eh_else1716
	%tmp1723 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1734:		; preds = %eh_else1716
	%eh_typeid1735 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a081 to i8*) )		; <i32> [#uses=1]
	%tmp1737 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1735		; <i1> [#uses=1]
	br i1 %tmp1737, label %eh_then1738, label %eh_else1752

eh_then1738:		; preds = %eh_else1734
	%tmp1741 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1752:		; preds = %eh_else1734
	%eh_typeid1753 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a082 to i8*) )		; <i32> [#uses=1]
	%tmp1755 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1753		; <i1> [#uses=1]
	br i1 %tmp1755, label %eh_then1756, label %eh_else1770

eh_then1756:		; preds = %eh_else1752
	%tmp1759 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1770:		; preds = %eh_else1752
	%eh_typeid1771 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a083 to i8*) )		; <i32> [#uses=1]
	%tmp1773 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1771		; <i1> [#uses=1]
	br i1 %tmp1773, label %eh_then1774, label %eh_else1788

eh_then1774:		; preds = %eh_else1770
	%tmp1777 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1788:		; preds = %eh_else1770
	%eh_typeid1789 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a084 to i8*) )		; <i32> [#uses=1]
	%tmp1791 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1789		; <i1> [#uses=1]
	br i1 %tmp1791, label %eh_then1792, label %eh_else1806

eh_then1792:		; preds = %eh_else1788
	%tmp1795 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1806:		; preds = %eh_else1788
	%eh_typeid1807 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a085 to i8*) )		; <i32> [#uses=1]
	%tmp1809 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1807		; <i1> [#uses=1]
	br i1 %tmp1809, label %eh_then1810, label %eh_else1824

eh_then1810:		; preds = %eh_else1806
	%tmp1813 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1824:		; preds = %eh_else1806
	%eh_typeid1825 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a086 to i8*) )		; <i32> [#uses=1]
	%tmp1827 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1825		; <i1> [#uses=1]
	br i1 %tmp1827, label %eh_then1828, label %eh_else1842

eh_then1828:		; preds = %eh_else1824
	%tmp1831 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1842:		; preds = %eh_else1824
	%eh_typeid1843 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a087 to i8*) )		; <i32> [#uses=1]
	%tmp1845 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1843		; <i1> [#uses=1]
	br i1 %tmp1845, label %eh_then1846, label %eh_else1860

eh_then1846:		; preds = %eh_else1842
	%tmp1849 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1860:		; preds = %eh_else1842
	%eh_typeid1861 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a088 to i8*) )		; <i32> [#uses=1]
	%tmp1863 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1861		; <i1> [#uses=1]
	br i1 %tmp1863, label %eh_then1864, label %eh_else1878

eh_then1864:		; preds = %eh_else1860
	%tmp1867 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1878:		; preds = %eh_else1860
	%eh_typeid1879 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a089 to i8*) )		; <i32> [#uses=1]
	%tmp1881 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1879		; <i1> [#uses=1]
	br i1 %tmp1881, label %eh_then1882, label %eh_else1896

eh_then1882:		; preds = %eh_else1878
	%tmp1885 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1896:		; preds = %eh_else1878
	%eh_typeid1897 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a090 to i8*) )		; <i32> [#uses=1]
	%tmp1899 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1897		; <i1> [#uses=1]
	br i1 %tmp1899, label %eh_then1900, label %eh_else1914

eh_then1900:		; preds = %eh_else1896
	%tmp1903 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1914:		; preds = %eh_else1896
	%eh_typeid1915 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a091 to i8*) )		; <i32> [#uses=1]
	%tmp1917 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1915		; <i1> [#uses=1]
	br i1 %tmp1917, label %eh_then1918, label %eh_else1932

eh_then1918:		; preds = %eh_else1914
	%tmp1921 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1932:		; preds = %eh_else1914
	%eh_typeid1933 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a092 to i8*) )		; <i32> [#uses=1]
	%tmp1935 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1933		; <i1> [#uses=1]
	br i1 %tmp1935, label %eh_then1936, label %eh_else1950

eh_then1936:		; preds = %eh_else1932
	%tmp1939 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1950:		; preds = %eh_else1932
	%eh_typeid1951 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a093 to i8*) )		; <i32> [#uses=1]
	%tmp1953 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1951		; <i1> [#uses=1]
	br i1 %tmp1953, label %eh_then1954, label %eh_else1968

eh_then1954:		; preds = %eh_else1950
	%tmp1957 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1968:		; preds = %eh_else1950
	%eh_typeid1969 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a094 to i8*) )		; <i32> [#uses=1]
	%tmp1971 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1969		; <i1> [#uses=1]
	br i1 %tmp1971, label %eh_then1972, label %eh_else1986

eh_then1972:		; preds = %eh_else1968
	%tmp1975 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else1986:		; preds = %eh_else1968
	%eh_typeid1987 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a095 to i8*) )		; <i32> [#uses=1]
	%tmp1989 = icmp eq i32 %eh_selector.12734.1, %eh_typeid1987		; <i1> [#uses=1]
	br i1 %tmp1989, label %eh_then1990, label %eh_else2004

eh_then1990:		; preds = %eh_else1986
	%tmp1993 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2004:		; preds = %eh_else1986
	%eh_typeid2005 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a096 to i8*) )		; <i32> [#uses=1]
	%tmp2007 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2005		; <i1> [#uses=1]
	br i1 %tmp2007, label %eh_then2008, label %eh_else2022

eh_then2008:		; preds = %eh_else2004
	%tmp2011 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2022:		; preds = %eh_else2004
	%eh_typeid2023 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a097 to i8*) )		; <i32> [#uses=1]
	%tmp2025 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2023		; <i1> [#uses=1]
	br i1 %tmp2025, label %eh_then2026, label %eh_else2040

eh_then2026:		; preds = %eh_else2022
	%tmp2029 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2040:		; preds = %eh_else2022
	%eh_typeid2041 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a098 to i8*) )		; <i32> [#uses=1]
	%tmp2043 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2041		; <i1> [#uses=1]
	br i1 %tmp2043, label %eh_then2044, label %eh_else2058

eh_then2044:		; preds = %eh_else2040
	%tmp2047 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2058:		; preds = %eh_else2040
	%eh_typeid2059 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a099 to i8*) )		; <i32> [#uses=1]
	%tmp2061 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2059		; <i1> [#uses=1]
	br i1 %tmp2061, label %eh_then2062, label %eh_else2076

eh_then2062:		; preds = %eh_else2058
	%tmp2065 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2076:		; preds = %eh_else2058
	%eh_typeid2077 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a100 to i8*) )		; <i32> [#uses=1]
	%tmp2079 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2077		; <i1> [#uses=1]
	br i1 %tmp2079, label %eh_then2080, label %eh_else2094

eh_then2080:		; preds = %eh_else2076
	%tmp2083 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2094:		; preds = %eh_else2076
	%eh_typeid2095 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a101 to i8*) )		; <i32> [#uses=1]
	%tmp2097 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2095		; <i1> [#uses=1]
	br i1 %tmp2097, label %eh_then2098, label %eh_else2112

eh_then2098:		; preds = %eh_else2094
	%tmp2101 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2112:		; preds = %eh_else2094
	%eh_typeid2113 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a102 to i8*) )		; <i32> [#uses=1]
	%tmp2115 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2113		; <i1> [#uses=1]
	br i1 %tmp2115, label %eh_then2116, label %eh_else2130

eh_then2116:		; preds = %eh_else2112
	%tmp2119 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2130:		; preds = %eh_else2112
	%eh_typeid2131 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a103 to i8*) )		; <i32> [#uses=1]
	%tmp2133 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2131		; <i1> [#uses=1]
	br i1 %tmp2133, label %eh_then2134, label %eh_else2148

eh_then2134:		; preds = %eh_else2130
	%tmp2137 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2148:		; preds = %eh_else2130
	%eh_typeid2149 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a104 to i8*) )		; <i32> [#uses=1]
	%tmp2151 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2149		; <i1> [#uses=1]
	br i1 %tmp2151, label %eh_then2152, label %eh_else2166

eh_then2152:		; preds = %eh_else2148
	%tmp2155 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2166:		; preds = %eh_else2148
	%eh_typeid2167 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a105 to i8*) )		; <i32> [#uses=1]
	%tmp2169 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2167		; <i1> [#uses=1]
	br i1 %tmp2169, label %eh_then2170, label %eh_else2184

eh_then2170:		; preds = %eh_else2166
	%tmp2173 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2184:		; preds = %eh_else2166
	%eh_typeid2185 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a106 to i8*) )		; <i32> [#uses=1]
	%tmp2187 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2185		; <i1> [#uses=1]
	br i1 %tmp2187, label %eh_then2188, label %eh_else2202

eh_then2188:		; preds = %eh_else2184
	%tmp2191 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2202:		; preds = %eh_else2184
	%eh_typeid2203 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a107 to i8*) )		; <i32> [#uses=1]
	%tmp2205 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2203		; <i1> [#uses=1]
	br i1 %tmp2205, label %eh_then2206, label %eh_else2220

eh_then2206:		; preds = %eh_else2202
	%tmp2209 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2220:		; preds = %eh_else2202
	%eh_typeid2221 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a108 to i8*) )		; <i32> [#uses=1]
	%tmp2223 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2221		; <i1> [#uses=1]
	br i1 %tmp2223, label %eh_then2224, label %eh_else2238

eh_then2224:		; preds = %eh_else2220
	%tmp2227 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2238:		; preds = %eh_else2220
	%eh_typeid2239 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a109 to i8*) )		; <i32> [#uses=1]
	%tmp2241 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2239		; <i1> [#uses=1]
	br i1 %tmp2241, label %eh_then2242, label %eh_else2256

eh_then2242:		; preds = %eh_else2238
	%tmp2245 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2256:		; preds = %eh_else2238
	%eh_typeid2257 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a110 to i8*) )		; <i32> [#uses=1]
	%tmp2259 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2257		; <i1> [#uses=1]
	br i1 %tmp2259, label %eh_then2260, label %eh_else2274

eh_then2260:		; preds = %eh_else2256
	%tmp2263 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2274:		; preds = %eh_else2256
	%eh_typeid2275 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a111 to i8*) )		; <i32> [#uses=1]
	%tmp2277 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2275		; <i1> [#uses=1]
	br i1 %tmp2277, label %eh_then2278, label %eh_else2292

eh_then2278:		; preds = %eh_else2274
	%tmp2281 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2292:		; preds = %eh_else2274
	%eh_typeid2293 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a112 to i8*) )		; <i32> [#uses=1]
	%tmp2295 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2293		; <i1> [#uses=1]
	br i1 %tmp2295, label %eh_then2296, label %eh_else2310

eh_then2296:		; preds = %eh_else2292
	%tmp2299 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2310:		; preds = %eh_else2292
	%eh_typeid2311 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a113 to i8*) )		; <i32> [#uses=1]
	%tmp2313 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2311		; <i1> [#uses=1]
	br i1 %tmp2313, label %eh_then2314, label %eh_else2328

eh_then2314:		; preds = %eh_else2310
	%tmp2317 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2328:		; preds = %eh_else2310
	%eh_typeid2329 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a114 to i8*) )		; <i32> [#uses=1]
	%tmp2331 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2329		; <i1> [#uses=1]
	br i1 %tmp2331, label %eh_then2332, label %eh_else2346

eh_then2332:		; preds = %eh_else2328
	%tmp2335 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2346:		; preds = %eh_else2328
	%eh_typeid2347 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a115 to i8*) )		; <i32> [#uses=1]
	%tmp2349 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2347		; <i1> [#uses=1]
	br i1 %tmp2349, label %eh_then2350, label %eh_else2364

eh_then2350:		; preds = %eh_else2346
	%tmp2353 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2364:		; preds = %eh_else2346
	%eh_typeid2365 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a116 to i8*) )		; <i32> [#uses=1]
	%tmp2367 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2365		; <i1> [#uses=1]
	br i1 %tmp2367, label %eh_then2368, label %eh_else2382

eh_then2368:		; preds = %eh_else2364
	%tmp2371 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2382:		; preds = %eh_else2364
	%eh_typeid2383 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a117 to i8*) )		; <i32> [#uses=1]
	%tmp2385 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2383		; <i1> [#uses=1]
	br i1 %tmp2385, label %eh_then2386, label %eh_else2400

eh_then2386:		; preds = %eh_else2382
	%tmp2389 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2400:		; preds = %eh_else2382
	%eh_typeid2401 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a118 to i8*) )		; <i32> [#uses=1]
	%tmp2403 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2401		; <i1> [#uses=1]
	br i1 %tmp2403, label %eh_then2404, label %eh_else2418

eh_then2404:		; preds = %eh_else2400
	%tmp2407 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2418:		; preds = %eh_else2400
	%eh_typeid2419 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a119 to i8*) )		; <i32> [#uses=1]
	%tmp2421 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2419		; <i1> [#uses=1]
	br i1 %tmp2421, label %eh_then2422, label %eh_else2436

eh_then2422:		; preds = %eh_else2418
	%tmp2425 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2436:		; preds = %eh_else2418
	%eh_typeid2437 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a120 to i8*) )		; <i32> [#uses=1]
	%tmp2439 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2437		; <i1> [#uses=1]
	br i1 %tmp2439, label %eh_then2440, label %eh_else2454

eh_then2440:		; preds = %eh_else2436
	%tmp2443 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2454:		; preds = %eh_else2436
	%eh_typeid2455 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a121 to i8*) )		; <i32> [#uses=1]
	%tmp2457 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2455		; <i1> [#uses=1]
	br i1 %tmp2457, label %eh_then2458, label %eh_else2472

eh_then2458:		; preds = %eh_else2454
	%tmp2461 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2472:		; preds = %eh_else2454
	%eh_typeid2473 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a122 to i8*) )		; <i32> [#uses=1]
	%tmp2475 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2473		; <i1> [#uses=1]
	br i1 %tmp2475, label %eh_then2476, label %eh_else2490

eh_then2476:		; preds = %eh_else2472
	%tmp2479 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2490:		; preds = %eh_else2472
	%eh_typeid2491 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a123 to i8*) )		; <i32> [#uses=1]
	%tmp2493 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2491		; <i1> [#uses=1]
	br i1 %tmp2493, label %eh_then2494, label %eh_else2508

eh_then2494:		; preds = %eh_else2490
	%tmp2497 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2508:		; preds = %eh_else2490
	%eh_typeid2509 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a124 to i8*) )		; <i32> [#uses=1]
	%tmp2511 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2509		; <i1> [#uses=1]
	br i1 %tmp2511, label %eh_then2512, label %eh_else2526

eh_then2512:		; preds = %eh_else2508
	%tmp2515 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2526:		; preds = %eh_else2508
	%eh_typeid2527 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a125 to i8*) )		; <i32> [#uses=1]
	%tmp2529 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2527		; <i1> [#uses=1]
	br i1 %tmp2529, label %eh_then2530, label %eh_else2544

eh_then2530:		; preds = %eh_else2526
	%tmp2533 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2544:		; preds = %eh_else2526
	%eh_typeid2545 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a126 to i8*) )		; <i32> [#uses=1]
	%tmp2547 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2545		; <i1> [#uses=1]
	br i1 %tmp2547, label %eh_then2548, label %eh_else2562

eh_then2548:		; preds = %eh_else2544
	%tmp2551 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2562:		; preds = %eh_else2544
	%eh_typeid2563 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a127 to i8*) )		; <i32> [#uses=1]
	%tmp2565 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2563		; <i1> [#uses=1]
	br i1 %tmp2565, label %eh_then2566, label %eh_else2580

eh_then2566:		; preds = %eh_else2562
	%tmp2569 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

eh_else2580:		; preds = %eh_else2562
	%eh_typeid2581 = tail call i32 @llvm.eh.typeid.for( i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI4a128 to i8*) )		; <i32> [#uses=1]
	%tmp2583 = icmp eq i32 %eh_selector.12734.1, %eh_typeid2581		; <i1> [#uses=1]
	br i1 %tmp2583, label %eh_then2584, label %Unwind

eh_then2584:		; preds = %eh_else2580
	%tmp2587 = tail call i8* @__cxa_begin_catch( i8* %eh_exception.12604.1 )		; <i8*> [#uses=0]
	tail call void @__cxa_end_catch( )
	ret void

Unwind:		; preds = %eh_else2580
	tail call i32 (...)* @_Unwind_Resume( i8* %eh_exception.12604.1 )		; <i32>:0 [#uses=0]
	unreachable

UnifiedUnreachableBlock1:		; preds = %filter270, %filter
	unreachable

UnifiedReturnBlock2:		; preds = %entry
	ret void
}

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()
