; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -relocation-model=pic -disable-fp-elim -color-ss-with-regs -stats -info-output-file - > %t
; RUN:   grep stackcoloring %t | grep "stack slot refs replaced with reg refs"  | grep 14

	type { [62 x %struct.Bitvec*] }		; type %0
	type { i8* }		; type %1
	type { double }		; type %2
	%struct..5sPragmaType = type { i8*, i32 }
	%struct.AggInfo = type { i8, i8, i32, %struct.ExprList*, i32, %struct.AggInfo_col*, i32, i32, i32, %struct.AggInfo_func*, i32, i32 }
	%struct.AggInfo_col = type { %struct.Table*, i32, i32, i32, i32, %struct.Expr* }
	%struct.AggInfo_func = type { %struct.Expr*, %struct.FuncDef*, i32, i32 }
	%struct.AuxData = type { i8*, void (i8*)* }
	%struct.Bitvec = type { i32, i32, i32, %0 }
	%struct.BtCursor = type { %struct.Btree*, %struct.BtShared*, %struct.BtCursor*, %struct.BtCursor*, i32 (i8*, i32, i8*, i32, i8*)*, i8*, i32, %struct.MemPage*, i32, %struct.CellInfo, i8, i8, i8*, i64, i32, i8, i32* }
	%struct.BtLock = type { %struct.Btree*, i32, i8, %struct.BtLock* }
	%struct.BtShared = type { %struct.Pager*, %struct.sqlite3*, %struct.BtCursor*, %struct.MemPage*, i8, i8, i8, i8, i8, i8, i8, i8, i32, i16, i16, i32, i32, i32, i32, i8, i32, i8*, void (i8*)*, %struct.sqlite3_mutex*, %struct.BusyHandler, i32, %struct.BtShared*, %struct.BtLock*, %struct.Btree* }
	%struct.Btree = type { %struct.sqlite3*, %struct.BtShared*, i8, i8, i8, i32, %struct.Btree*, %struct.Btree* }
	%struct.BtreeMutexArray = type { i32, [11 x %struct.Btree*] }
	%struct.BusyHandler = type { i32 (i8*, i32)*, i8*, i32 }
	%struct.CellInfo = type { i8*, i64, i32, i32, i16, i16, i16, i16 }
	%struct.CollSeq = type { i8*, i8, i8, i8*, i32 (i8*, i32, i8*, i32, i8*)*, void (i8*)* }
	%struct.Column = type { i8*, %struct.Expr*, i8*, i8*, i8, i8, i8, i8 }
	%struct.Context = type { i64, i32, %struct.Fifo }
	%struct.CountCtx = type { i64 }
	%struct.Cursor = type { %struct.BtCursor*, i32, i64, i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i64, %struct.Btree*, i32, i8*, i64, i8*, %struct.KeyInfo*, i32, i64, %struct.sqlite3_vtab_cursor*, %struct.sqlite3_module*, i32, i32, i32*, i32*, i8* }
	%struct.Db = type { i8*, %struct.Btree*, i8, i8, i8*, void (i8*)*, %struct.Schema* }
	%struct.DbPage = type { %struct.Pager*, i32, %struct.DbPage*, %struct.DbPage*, %struct.PagerLruLink, %struct.DbPage*, i8, i8, i8, i8, i8, i16, %struct.DbPage*, %struct.DbPage*, i8* }
	%struct.Expr = type { i8, i8, i16, %struct.CollSeq*, %struct.Expr*, %struct.Expr*, %struct.ExprList*, %struct..5sPragmaType, %struct..5sPragmaType, i32, i32, %struct.AggInfo*, i32, i32, %struct.Select*, %struct.Table*, i32 }
	%struct.ExprList = type { i32, i32, i32, %struct.ExprList_item* }
	%struct.ExprList_item = type { %struct.Expr*, i8*, i8, i8, i8 }
	%struct.FKey = type { %struct.Table*, %struct.FKey*, i8*, %struct.FKey*, i32, %struct.sColMap*, i8, i8, i8, i8 }
	%struct.Fifo = type { i32, %struct.FifoPage*, %struct.FifoPage* }
	%struct.FifoPage = type { i32, i32, i32, %struct.FifoPage*, [1 x i64] }
	%struct.FuncDef = type { i16, i8, i8, i8, i8*, %struct.FuncDef*, void (%struct.sqlite3_context*, i32, %struct.Mem**)*, void (%struct.sqlite3_context*, i32, %struct.Mem**)*, void (%struct.sqlite3_context*)*, [1 x i8] }
	%struct.Hash = type { i8, i8, i32, i32, %struct.HashElem*, %struct._ht* }
	%struct.HashElem = type { %struct.HashElem*, %struct.HashElem*, i8*, i8*, i32 }
	%struct.IdList = type { %struct..5sPragmaType*, i32, i32 }
	%struct.Index = type { i8*, i32, i32*, i32*, %struct.Table*, i32, i8, i8, i8*, %struct.Index*, %struct.Schema*, i8*, i8** }
	%struct.KeyInfo = type { %struct.sqlite3*, i8, i8, i8, i32, i8*, [1 x %struct.CollSeq*] }
	%struct.Mem = type { %struct.CountCtx, double, %struct.sqlite3*, i8*, i32, i16, i8, i8, void (i8*)* }
	%struct.MemPage = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i16, i16, i16, i16, i16, i16, [5 x %struct._OvflCell], %struct.BtShared*, i8*, %struct.DbPage*, i32, %struct.MemPage* }
	%struct.Module = type { %struct.sqlite3_module*, i8*, i8*, void (i8*)* }
	%struct.Op = type { i8, i8, i8, i8, i32, i32, i32, %1 }
	%struct.Pager = type { %struct.sqlite3_vfs*, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.Bitvec*, %struct.Bitvec*, i8*, i8*, i8*, i8*, %struct.sqlite3_file*, %struct.sqlite3_file*, %struct.sqlite3_file*, %struct.BusyHandler*, %struct.PagerLruList, %struct.DbPage*, %struct.DbPage*, %struct.DbPage*, i64, i64, i64, i64, i64, i32, void (%struct.DbPage*, i32)*, void (%struct.DbPage*, i32)*, i32, %struct.DbPage**, i8*, [16 x i8] }
	%struct.PagerLruLink = type { %struct.DbPage*, %struct.DbPage* }
	%struct.PagerLruList = type { %struct.DbPage*, %struct.DbPage*, %struct.DbPage* }
	%struct.Schema = type { i32, %struct.Hash, %struct.Hash, %struct.Hash, %struct.Hash, %struct.Table*, i8, i8, i16, i32, %struct.sqlite3* }
	%struct.Select = type { %struct.ExprList*, i8, i8, i8, i8, i8, i8, i8, %struct.SrcList*, %struct.Expr*, %struct.ExprList*, %struct.Expr*, %struct.ExprList*, %struct.Select*, %struct.Select*, %struct.Select*, %struct.Expr*, %struct.Expr*, i32, i32, [3 x i32] }
	%struct.SrcList = type { i16, i16, [1 x %struct.SrcList_item] }
	%struct.SrcList_item = type { i8*, i8*, i8*, %struct.Table*, %struct.Select*, i8, i8, i32, %struct.Expr*, %struct.IdList*, i64 }
	%struct.Table = type { i8*, i32, %struct.Column*, i32, %struct.Index*, i32, %struct.Select*, i32, %struct.Trigger*, %struct.FKey*, i8*, %struct.Expr*, i32, i8, i8, i8, i8, i8, i8, i8, %struct.Module*, %struct.sqlite3_vtab*, i32, i8**, %struct.Schema* }
	%struct.Trigger = type { i8*, i8*, i8, i8, %struct.Expr*, %struct.IdList*, %struct..5sPragmaType, %struct.Schema*, %struct.Schema*, %struct.TriggerStep*, %struct.Trigger* }
	%struct.TriggerStep = type { i32, i32, %struct.Trigger*, %struct.Select*, %struct..5sPragmaType, %struct.Expr*, %struct.ExprList*, %struct.IdList*, %struct.TriggerStep*, %struct.TriggerStep* }
	%struct.Vdbe = type { %struct.sqlite3*, %struct.Vdbe*, %struct.Vdbe*, i32, i32, %struct.Op*, i32, i32, i32*, %struct.Mem**, %struct.Mem*, i32, %struct.Cursor**, i32, %struct.Mem*, i8**, i32, i32, i32, %struct.Mem*, i32, i32, %struct.Fifo, i32, i32, %struct.Context*, i32, i32, i32, i32, i32, [25 x i32], i32, i32, i8**, i8*, %struct.Mem*, i8, i8, i8, i8, i8, i8, i32, i64, i32, %struct.BtreeMutexArray, i32, i8*, i32 }
	%struct.VdbeFunc = type { %struct.FuncDef*, i32, [1 x %struct.AuxData] }
	%struct._OvflCell = type { i8*, i16 }
	%struct._ht = type { i32, %struct.HashElem* }
	%struct.sColMap = type { i32, i8* }
	%struct.sqlite3 = type { %struct.sqlite3_vfs*, i32, %struct.Db*, i32, i32, i32, i32, i8, i8, i8, i8, i32, %struct.CollSeq*, i64, i64, i32, i32, i32, %struct.sqlite3_mutex*, %struct.sqlite3InitInfo, i32, i8**, %struct.Vdbe*, i32, void (i8*, i8*)*, i8*, void (i8*, i8*, i64)*, i8*, i8*, i32 (i8*)*, i8*, void (i8*)*, i8*, void (i8*, i32, i8*, i8*, i64)*, void (i8*, %struct.sqlite3*, i32, i8*)*, void (i8*, %struct.sqlite3*, i32, i8*)*, i8*, %struct.Mem*, i8*, i8*, %2, i32 (i8*, i32, i8*, i8*, i8*, i8*)*, i8*, i32 (i8*)*, i8*, i32, %struct.Hash, %struct.Table*, %struct.sqlite3_vtab**, i32, %struct.Hash, %struct.Hash, %struct.BusyHandler, i32, [2 x %struct.Db], i8 }
	%struct.sqlite3InitInfo = type { i32, i32, i8 }
	%struct.sqlite3_context = type { %struct.FuncDef*, %struct.VdbeFunc*, %struct.Mem, %struct.Mem*, i32, %struct.CollSeq* }
	%struct.sqlite3_file = type { %struct.sqlite3_io_methods* }
	%struct.sqlite3_index_constraint = type { i32, i8, i8, i32 }
	%struct.sqlite3_index_constraint_usage = type { i32, i8 }
	%struct.sqlite3_index_info = type { i32, %struct.sqlite3_index_constraint*, i32, %struct.sqlite3_index_constraint_usage*, %struct.sqlite3_index_constraint_usage*, i32, i8*, i32, i32, double }
	%struct.sqlite3_io_methods = type { i32, i32 (%struct.sqlite3_file*)*, i32 (%struct.sqlite3_file*, i8*, i32, i64)*, i32 (%struct.sqlite3_file*, i8*, i32, i64)*, i32 (%struct.sqlite3_file*, i64)*, i32 (%struct.sqlite3_file*, i32)*, i32 (%struct.sqlite3_file*, i64*)*, i32 (%struct.sqlite3_file*, i32)*, i32 (%struct.sqlite3_file*, i32)*, i32 (%struct.sqlite3_file*)*, i32 (%struct.sqlite3_file*, i32, i8*)*, i32 (%struct.sqlite3_file*)*, i32 (%struct.sqlite3_file*)* }
	%struct.sqlite3_module = type { i32, i32 (%struct.sqlite3*, i8*, i32, i8**, %struct.sqlite3_vtab**, i8**)*, i32 (%struct.sqlite3*, i8*, i32, i8**, %struct.sqlite3_vtab**, i8**)*, i32 (%struct.sqlite3_vtab*, %struct.sqlite3_index_info*)*, i32 (%struct.sqlite3_vtab*)*, i32 (%struct.sqlite3_vtab*)*, i32 (%struct.sqlite3_vtab*, %struct.sqlite3_vtab_cursor**)*, i32 (%struct.sqlite3_vtab_cursor*)*, i32 (%struct.sqlite3_vtab_cursor*, i32, i8*, i32, %struct.Mem**)*, i32 (%struct.sqlite3_vtab_cursor*)*, i32 (%struct.sqlite3_vtab_cursor*)*, i32 (%struct.sqlite3_vtab_cursor*, %struct.sqlite3_context*, i32)*, i32 (%struct.sqlite3_vtab_cursor*, i64*)*, i32 (%struct.sqlite3_vtab*, i32, %struct.Mem**, i64*)*, i32 (%struct.sqlite3_vtab*)*, i32 (%struct.sqlite3_vtab*)*, i32 (%struct.sqlite3_vtab*)*, i32 (%struct.sqlite3_vtab*)*, i32 (%struct.sqlite3_vtab*, i32, i8*, void (%struct.sqlite3_context*, i32, %struct.Mem**)**, i8**)*, i32 (%struct.sqlite3_vtab*, i8*)* }
	%struct.sqlite3_mutex = type opaque
	%struct.sqlite3_vfs = type { i32, i32, i32, %struct.sqlite3_vfs*, i8*, i8*, i32 (%struct.sqlite3_vfs*, i8*, %struct.sqlite3_file*, i32, i32*)*, i32 (%struct.sqlite3_vfs*, i8*, i32)*, i32 (%struct.sqlite3_vfs*, i8*, i32)*, i32 (%struct.sqlite3_vfs*, i32, i8*)*, i32 (%struct.sqlite3_vfs*, i8*, i32, i8*)*, i8* (%struct.sqlite3_vfs*, i8*)*, void (%struct.sqlite3_vfs*, i32, i8*)*, i8* (%struct.sqlite3_vfs*, i8*, i8*)*, void (%struct.sqlite3_vfs*, i8*)*, i32 (%struct.sqlite3_vfs*, i32, i8*)*, i32 (%struct.sqlite3_vfs*, i32)*, i32 (%struct.sqlite3_vfs*, double*)* }
	%struct.sqlite3_vtab = type { %struct.sqlite3_module*, i32, i8* }
	%struct.sqlite3_vtab_cursor = type { %struct.sqlite3_vtab* }
@llvm.used = appending global [1 x i8*] [i8* bitcast (void (%struct.MemPage*, i32, i32)* @dropCell to i8*)], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define fastcc void @dropCell(%struct.MemPage* nocapture %pPage, i32 %idx, i32 %sz) nounwind ssp {
entry:
	%0 = getelementptr %struct.MemPage* %pPage, i64 0, i32 18		; <i8**> [#uses=1]
	%1 = load i8** %0, align 8		; <i8*> [#uses=34]
	%2 = getelementptr %struct.MemPage* %pPage, i64 0, i32 12		; <i16*> [#uses=1]
	%3 = load i16* %2, align 2		; <i16> [#uses=1]
	%4 = zext i16 %3 to i32		; <i32> [#uses=2]
	%5 = shl i32 %idx, 1		; <i32> [#uses=2]
	%6 = add i32 %4, %5		; <i32> [#uses=1]
	%7 = sext i32 %6 to i64		; <i64> [#uses=2]
	%8 = getelementptr i8* %1, i64 %7		; <i8*> [#uses=1]
	%9 = load i8* %8, align 1		; <i8> [#uses=2]
	%10 = zext i8 %9 to i32		; <i32> [#uses=1]
	%11 = shl i32 %10, 8		; <i32> [#uses=1]
	%.sum3 = add i64 %7, 1		; <i64> [#uses=1]
	%12 = getelementptr i8* %1, i64 %.sum3		; <i8*> [#uses=1]
	%13 = load i8* %12, align 1		; <i8> [#uses=2]
	%14 = zext i8 %13 to i32		; <i32> [#uses=1]
	%15 = or i32 %11, %14		; <i32> [#uses=3]
	%16 = icmp slt i32 %sz, 4		; <i1> [#uses=1]
	%size_addr.0.i = select i1 %16, i32 4, i32 %sz		; <i32> [#uses=3]
	%17 = getelementptr %struct.MemPage* %pPage, i64 0, i32 8		; <i8*> [#uses=5]
	%18 = load i8* %17, align 8		; <i8> [#uses=1]
	%19 = zext i8 %18 to i32		; <i32> [#uses=4]
	%20 = add i32 %19, 1		; <i32> [#uses=2]
	br label %bb3.i

bb3.i:		; preds = %bb3.i, %entry
	%addr.0.i = phi i32 [ %20, %entry ], [ %29, %bb3.i ]		; <i32> [#uses=1]
	%21 = sext i32 %addr.0.i to i64		; <i64> [#uses=2]
	%22 = getelementptr i8* %1, i64 %21		; <i8*> [#uses=2]
	%23 = load i8* %22, align 1		; <i8> [#uses=2]
	%24 = zext i8 %23 to i32		; <i32> [#uses=1]
	%25 = shl i32 %24, 8		; <i32> [#uses=1]
	%.sum34.i = add i64 %21, 1		; <i64> [#uses=1]
	%26 = getelementptr i8* %1, i64 %.sum34.i		; <i8*> [#uses=2]
	%27 = load i8* %26, align 1		; <i8> [#uses=2]
	%28 = zext i8 %27 to i32		; <i32> [#uses=1]
	%29 = or i32 %25, %28		; <i32> [#uses=3]
	%.not.i = icmp uge i32 %29, %15		; <i1> [#uses=1]
	%30 = icmp eq i32 %29, 0		; <i1> [#uses=1]
	%or.cond.i = or i1 %30, %.not.i		; <i1> [#uses=1]
	br i1 %or.cond.i, label %bb5.i, label %bb3.i

bb5.i:		; preds = %bb3.i
	store i8 %9, i8* %22, align 1
	store i8 %13, i8* %26, align 1
	%31 = zext i32 %15 to i64		; <i64> [#uses=2]
	%32 = getelementptr i8* %1, i64 %31		; <i8*> [#uses=1]
	store i8 %23, i8* %32, align 1
	%.sum32.i = add i64 %31, 1		; <i64> [#uses=1]
	%33 = getelementptr i8* %1, i64 %.sum32.i		; <i8*> [#uses=1]
	store i8 %27, i8* %33, align 1
	%34 = add i32 %15, 2		; <i32> [#uses=1]
	%35 = zext i32 %34 to i64		; <i64> [#uses=2]
	%36 = getelementptr i8* %1, i64 %35		; <i8*> [#uses=1]
	%37 = lshr i32 %size_addr.0.i, 8		; <i32> [#uses=1]
	%38 = trunc i32 %37 to i8		; <i8> [#uses=1]
	store i8 %38, i8* %36, align 1
	%39 = trunc i32 %size_addr.0.i to i8		; <i8> [#uses=1]
	%.sum31.i = add i64 %35, 1		; <i64> [#uses=1]
	%40 = getelementptr i8* %1, i64 %.sum31.i		; <i8*> [#uses=1]
	store i8 %39, i8* %40, align 1
	%41 = getelementptr %struct.MemPage* %pPage, i64 0, i32 14		; <i16*> [#uses=4]
	%42 = load i16* %41, align 2		; <i16> [#uses=1]
	%43 = trunc i32 %size_addr.0.i to i16		; <i16> [#uses=1]
	%44 = add i16 %42, %43		; <i16> [#uses=1]
	store i16 %44, i16* %41, align 2
	%45 = load i8* %17, align 8		; <i8> [#uses=1]
	%46 = zext i8 %45 to i32		; <i32> [#uses=1]
	%47 = add i32 %46, 1		; <i32> [#uses=1]
	br label %bb11.outer.i

bb11.outer.i:		; preds = %bb6.i, %bb5.i
	%addr.1.ph.i = phi i32 [ %47, %bb5.i ], [ %111, %bb6.i ]		; <i32> [#uses=1]
	%48 = sext i32 %addr.1.ph.i to i64		; <i64> [#uses=2]
	%49 = getelementptr i8* %1, i64 %48		; <i8*> [#uses=1]
	%.sum30.i = add i64 %48, 1		; <i64> [#uses=1]
	%50 = getelementptr i8* %1, i64 %.sum30.i		; <i8*> [#uses=1]
	br label %bb11.i

bb6.i:		; preds = %bb11.i
	%51 = zext i32 %111 to i64		; <i64> [#uses=2]
	%52 = getelementptr i8* %1, i64 %51		; <i8*> [#uses=2]
	%53 = load i8* %52, align 1		; <i8> [#uses=1]
	%54 = zext i8 %53 to i32		; <i32> [#uses=1]
	%55 = shl i32 %54, 8		; <i32> [#uses=1]
	%.sum24.i = add i64 %51, 1		; <i64> [#uses=1]
	%56 = getelementptr i8* %1, i64 %.sum24.i		; <i8*> [#uses=2]
	%57 = load i8* %56, align 1		; <i8> [#uses=3]
	%58 = zext i8 %57 to i32		; <i32> [#uses=1]
	%59 = or i32 %55, %58		; <i32> [#uses=5]
	%60 = add i32 %111, 2		; <i32> [#uses=1]
	%61 = zext i32 %60 to i64		; <i64> [#uses=2]
	%62 = getelementptr i8* %1, i64 %61		; <i8*> [#uses=2]
	%63 = load i8* %62, align 1		; <i8> [#uses=1]
	%64 = zext i8 %63 to i32		; <i32> [#uses=1]
	%65 = shl i32 %64, 8		; <i32> [#uses=1]
	%.sum23.i = add i64 %61, 1		; <i64> [#uses=1]
	%66 = getelementptr i8* %1, i64 %.sum23.i		; <i8*> [#uses=2]
	%67 = load i8* %66, align 1		; <i8> [#uses=2]
	%68 = zext i8 %67 to i32		; <i32> [#uses=1]
	%69 = or i32 %65, %68		; <i32> [#uses=1]
	%70 = add i32 %111, 3		; <i32> [#uses=1]
	%71 = add i32 %70, %69		; <i32> [#uses=1]
	%72 = icmp sge i32 %71, %59		; <i1> [#uses=1]
	%73 = icmp ne i32 %59, 0		; <i1> [#uses=1]
	%74 = and i1 %72, %73		; <i1> [#uses=1]
	br i1 %74, label %bb9.i, label %bb11.outer.i

bb9.i:		; preds = %bb6.i
	%75 = load i8* %17, align 8		; <i8> [#uses=1]
	%76 = zext i8 %75 to i32		; <i32> [#uses=1]
	%77 = add i32 %76, 7		; <i32> [#uses=1]
	%78 = zext i32 %77 to i64		; <i64> [#uses=1]
	%79 = getelementptr i8* %1, i64 %78		; <i8*> [#uses=2]
	%80 = load i8* %79, align 1		; <i8> [#uses=1]
	%81 = sub i8 %109, %57		; <i8> [#uses=1]
	%82 = add i8 %81, %67		; <i8> [#uses=1]
	%83 = add i8 %82, %80		; <i8> [#uses=1]
	store i8 %83, i8* %79, align 1
	%84 = zext i32 %59 to i64		; <i64> [#uses=2]
	%85 = getelementptr i8* %1, i64 %84		; <i8*> [#uses=1]
	%86 = load i8* %85, align 1		; <i8> [#uses=1]
	store i8 %86, i8* %52, align 1
	%.sum22.i = add i64 %84, 1		; <i64> [#uses=1]
	%87 = getelementptr i8* %1, i64 %.sum22.i		; <i8*> [#uses=1]
	%88 = load i8* %87, align 1		; <i8> [#uses=1]
	store i8 %88, i8* %56, align 1
	%89 = add i32 %59, 2		; <i32> [#uses=1]
	%90 = zext i32 %89 to i64		; <i64> [#uses=2]
	%91 = getelementptr i8* %1, i64 %90		; <i8*> [#uses=1]
	%92 = load i8* %91, align 1		; <i8> [#uses=1]
	%93 = zext i8 %92 to i32		; <i32> [#uses=1]
	%94 = shl i32 %93, 8		; <i32> [#uses=1]
	%.sum20.i = add i64 %90, 1		; <i64> [#uses=1]
	%95 = getelementptr i8* %1, i64 %.sum20.i		; <i8*> [#uses=2]
	%96 = load i8* %95, align 1		; <i8> [#uses=1]
	%97 = zext i8 %96 to i32		; <i32> [#uses=1]
	%98 = or i32 %94, %97		; <i32> [#uses=1]
	%99 = sub i32 %59, %111		; <i32> [#uses=1]
	%100 = add i32 %99, %98		; <i32> [#uses=1]
	%101 = lshr i32 %100, 8		; <i32> [#uses=1]
	%102 = trunc i32 %101 to i8		; <i8> [#uses=1]
	store i8 %102, i8* %62, align 1
	%103 = load i8* %95, align 1		; <i8> [#uses=1]
	%104 = sub i8 %57, %109		; <i8> [#uses=1]
	%105 = add i8 %104, %103		; <i8> [#uses=1]
	store i8 %105, i8* %66, align 1
	br label %bb11.i

bb11.i:		; preds = %bb9.i, %bb11.outer.i
	%106 = load i8* %49, align 1		; <i8> [#uses=1]
	%107 = zext i8 %106 to i32		; <i32> [#uses=1]
	%108 = shl i32 %107, 8		; <i32> [#uses=1]
	%109 = load i8* %50, align 1		; <i8> [#uses=3]
	%110 = zext i8 %109 to i32		; <i32> [#uses=1]
	%111 = or i32 %108, %110		; <i32> [#uses=6]
	%112 = icmp eq i32 %111, 0		; <i1> [#uses=1]
	br i1 %112, label %bb12.i, label %bb6.i

bb12.i:		; preds = %bb11.i
	%113 = zext i32 %20 to i64		; <i64> [#uses=2]
	%114 = getelementptr i8* %1, i64 %113		; <i8*> [#uses=2]
	%115 = load i8* %114, align 1		; <i8> [#uses=2]
	%116 = add i32 %19, 5		; <i32> [#uses=1]
	%117 = zext i32 %116 to i64		; <i64> [#uses=2]
	%118 = getelementptr i8* %1, i64 %117		; <i8*> [#uses=3]
	%119 = load i8* %118, align 1		; <i8> [#uses=1]
	%120 = icmp eq i8 %115, %119		; <i1> [#uses=1]
	br i1 %120, label %bb13.i, label %bb1.preheader

bb13.i:		; preds = %bb12.i
	%121 = add i32 %19, 2		; <i32> [#uses=1]
	%122 = zext i32 %121 to i64		; <i64> [#uses=1]
	%123 = getelementptr i8* %1, i64 %122		; <i8*> [#uses=1]
	%124 = load i8* %123, align 1		; <i8> [#uses=1]
	%125 = add i32 %19, 6		; <i32> [#uses=1]
	%126 = zext i32 %125 to i64		; <i64> [#uses=1]
	%127 = getelementptr i8* %1, i64 %126		; <i8*> [#uses=1]
	%128 = load i8* %127, align 1		; <i8> [#uses=1]
	%129 = icmp eq i8 %124, %128		; <i1> [#uses=1]
	br i1 %129, label %bb14.i, label %bb1.preheader

bb14.i:		; preds = %bb13.i
	%130 = zext i8 %115 to i32		; <i32> [#uses=1]
	%131 = shl i32 %130, 8		; <i32> [#uses=1]
	%.sum29.i = add i64 %113, 1		; <i64> [#uses=1]
	%132 = getelementptr i8* %1, i64 %.sum29.i		; <i8*> [#uses=1]
	%133 = load i8* %132, align 1		; <i8> [#uses=1]
	%134 = zext i8 %133 to i32		; <i32> [#uses=1]
	%135 = or i32 %134, %131		; <i32> [#uses=2]
	%136 = zext i32 %135 to i64		; <i64> [#uses=1]
	%137 = getelementptr i8* %1, i64 %136		; <i8*> [#uses=1]
	%138 = bitcast i8* %137 to i16*		; <i16*> [#uses=1]
	%139 = bitcast i8* %114 to i16*		; <i16*> [#uses=1]
	%tmp.i = load i16* %138, align 1		; <i16> [#uses=1]
	store i16 %tmp.i, i16* %139, align 1
	%140 = load i8* %118, align 1		; <i8> [#uses=1]
	%141 = zext i8 %140 to i32		; <i32> [#uses=1]
	%142 = shl i32 %141, 8		; <i32> [#uses=1]
	%.sum28.i = add i64 %117, 1		; <i64> [#uses=1]
	%143 = getelementptr i8* %1, i64 %.sum28.i		; <i8*> [#uses=2]
	%144 = load i8* %143, align 1		; <i8> [#uses=2]
	%145 = zext i8 %144 to i32		; <i32> [#uses=1]
	%146 = or i32 %142, %145		; <i32> [#uses=1]
	%147 = add i32 %135, 2		; <i32> [#uses=1]
	%148 = zext i32 %147 to i64		; <i64> [#uses=2]
	%149 = getelementptr i8* %1, i64 %148		; <i8*> [#uses=1]
	%150 = load i8* %149, align 1		; <i8> [#uses=1]
	%151 = zext i8 %150 to i32		; <i32> [#uses=1]
	%152 = shl i32 %151, 8		; <i32> [#uses=1]
	%.sum27.i = add i64 %148, 1		; <i64> [#uses=1]
	%153 = getelementptr i8* %1, i64 %.sum27.i		; <i8*> [#uses=2]
	%154 = load i8* %153, align 1		; <i8> [#uses=1]
	%155 = zext i8 %154 to i32		; <i32> [#uses=1]
	%156 = or i32 %152, %155		; <i32> [#uses=1]
	%157 = add i32 %156, %146		; <i32> [#uses=1]
	%158 = lshr i32 %157, 8		; <i32> [#uses=1]
	%159 = trunc i32 %158 to i8		; <i8> [#uses=1]
	store i8 %159, i8* %118, align 1
	%160 = load i8* %153, align 1		; <i8> [#uses=1]
	%161 = add i8 %160, %144		; <i8> [#uses=1]
	store i8 %161, i8* %143, align 1
	br label %bb1.preheader

bb1.preheader:		; preds = %bb14.i, %bb13.i, %bb12.i
	%i.08 = add i32 %idx, 1		; <i32> [#uses=2]
	%162 = getelementptr %struct.MemPage* %pPage, i64 0, i32 15		; <i16*> [#uses=4]
	%163 = load i16* %162, align 4		; <i16> [#uses=2]
	%164 = zext i16 %163 to i32		; <i32> [#uses=1]
	%165 = icmp sgt i32 %164, %i.08		; <i1> [#uses=1]
	br i1 %165, label %bb, label %bb2

bb:		; preds = %bb, %bb1.preheader
	%indvar = phi i64 [ 0, %bb1.preheader ], [ %indvar.next, %bb ]		; <i64> [#uses=3]
	%tmp16 = add i32 %5, %4		; <i32> [#uses=1]
	%tmp.17 = sext i32 %tmp16 to i64		; <i64> [#uses=1]
	%tmp19 = shl i64 %indvar, 1		; <i64> [#uses=1]
	%ctg2.sum = add i64 %tmp.17, %tmp19		; <i64> [#uses=4]
	%ctg229 = getelementptr i8* %1, i64 %ctg2.sum		; <i8*> [#uses=1]
	%ctg229.sum31 = add i64 %ctg2.sum, 2		; <i64> [#uses=1]
	%166 = getelementptr i8* %1, i64 %ctg229.sum31		; <i8*> [#uses=1]
	%167 = load i8* %166, align 1		; <i8> [#uses=1]
	store i8 %167, i8* %ctg229
	%ctg229.sum30 = add i64 %ctg2.sum, 3		; <i64> [#uses=1]
	%168 = getelementptr i8* %1, i64 %ctg229.sum30		; <i8*> [#uses=1]
	%169 = load i8* %168, align 1		; <i8> [#uses=1]
	%ctg229.sum = add i64 %ctg2.sum, 1		; <i64> [#uses=1]
	%170 = getelementptr i8* %1, i64 %ctg229.sum		; <i8*> [#uses=1]
	store i8 %169, i8* %170, align 1
	%indvar15 = trunc i64 %indvar to i32		; <i32> [#uses=1]
	%i.09 = add i32 %indvar15, %i.08		; <i32> [#uses=1]
	%i.0 = add i32 %i.09, 1		; <i32> [#uses=1]
	%171 = load i16* %162, align 4		; <i16> [#uses=2]
	%172 = zext i16 %171 to i32		; <i32> [#uses=1]
	%173 = icmp sgt i32 %172, %i.0		; <i1> [#uses=1]
	%indvar.next = add i64 %indvar, 1		; <i64> [#uses=1]
	br i1 %173, label %bb, label %bb2

bb2:		; preds = %bb, %bb1.preheader
	%174 = phi i16 [ %163, %bb1.preheader ], [ %171, %bb ]		; <i16> [#uses=1]
	%175 = add i16 %174, -1		; <i16> [#uses=2]
	store i16 %175, i16* %162, align 4
	%176 = load i8* %17, align 8		; <i8> [#uses=1]
	%177 = zext i8 %176 to i32		; <i32> [#uses=1]
	%178 = add i32 %177, 3		; <i32> [#uses=1]
	%179 = zext i32 %178 to i64		; <i64> [#uses=1]
	%180 = getelementptr i8* %1, i64 %179		; <i8*> [#uses=1]
	%181 = lshr i16 %175, 8		; <i16> [#uses=1]
	%182 = trunc i16 %181 to i8		; <i8> [#uses=1]
	store i8 %182, i8* %180, align 1
	%183 = load i8* %17, align 8		; <i8> [#uses=1]
	%184 = zext i8 %183 to i32		; <i32> [#uses=1]
	%185 = add i32 %184, 3		; <i32> [#uses=1]
	%186 = zext i32 %185 to i64		; <i64> [#uses=1]
	%187 = load i16* %162, align 4		; <i16> [#uses=1]
	%188 = trunc i16 %187 to i8		; <i8> [#uses=1]
	%.sum = add i64 %186, 1		; <i64> [#uses=1]
	%189 = getelementptr i8* %1, i64 %.sum		; <i8*> [#uses=1]
	store i8 %188, i8* %189, align 1
	%190 = load i16* %41, align 2		; <i16> [#uses=1]
	%191 = add i16 %190, 2		; <i16> [#uses=1]
	store i16 %191, i16* %41, align 2
	%192 = getelementptr %struct.MemPage* %pPage, i64 0, i32 1		; <i8*> [#uses=1]
	store i8 1, i8* %192, align 1
	ret void
}
