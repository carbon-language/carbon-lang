; RUN: llc < %s -mtriple=i386-apple-darwin -relocation-model=pic -stats |& grep {machine-licm} | grep 1
; rdar://7274692

%0 = type { [125 x i32] }
%1 = type { i32 }
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
%struct.FILE = type { i8*, i32, i32, i16, i16, %struct..5sPragmaType, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct..5sPragmaType, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct..5sPragmaType, i32, i64 }
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
%struct._RuneCharClass = type { [14 x i8], i32 }
%struct._RuneEntry = type { i32, i32, i32, i32* }
%struct._RuneLocale = type { [8 x i8], [32 x i8], i32 (i8*, i32, i8**)*, i32 (i32, i8*, i32, i8**)*, i32, [256 x i32], [256 x i32], [256 x i32], %struct._RuneRange, %struct._RuneRange, %struct._RuneRange, i8*, i32, i32, %struct._RuneCharClass* }
%struct._RuneRange = type { i32, %struct._RuneEntry* }
%struct.__sFILEX = type opaque
%struct._ht = type { i32, %struct.HashElem* }
%struct.callback_data = type { %struct.sqlite3*, i32, i32, %struct.FILE*, i32, i32, i32, i8*, [20 x i8], [100 x i32], [100 x i32], [20 x i8], %struct.previous_mode_data, [1024 x i8], i8* }
%struct.previous_mode_data = type { i32, i32, i32, [100 x i32] }
%struct.sColMap = type { i32, i8* }
%struct.sqlite3 = type { %struct.sqlite3_vfs*, i32, %struct.Db*, i32, i32, i32, i32, i8, i8, i8, i8, i32, %struct.CollSeq*, i64, i64, i32, i32, i32, %struct.sqlite3_mutex*, %struct.sqlite3InitInfo, i32, i8**, %struct.Vdbe*, i32, void (i8*, i8*)*, i8*, void (i8*, i8*, i64)*, i8*, i8*, i32 (i8*)*, i8*, void (i8*)*, i8*, void (i8*, i32, i8*, i8*, i64)*, void (i8*, %struct.sqlite3*, i32, i8*)*, void (i8*, %struct.sqlite3*, i32, i8*)*, i8*, %struct.Mem*, i8*, i8*, %union.anon, i32 (i8*, i32, i8*, i8*, i8*, i8*)*, i8*, i32 (i8*)*, i8*, i32, %struct.Hash, %struct.Table*, %struct.sqlite3_vtab**, i32, %struct.Hash, %struct.Hash, %struct.BusyHandler, i32, [2 x %struct.Db], i8 }
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
%union.anon = type { double }

@_DefaultRuneLocale = external global %struct._RuneLocale ; <%struct._RuneLocale*> [#uses=2]
@__stderrp = external global %struct.FILE*        ; <%struct.FILE**> [#uses=1]
@.str10 = internal constant [16 x i8] c"Out of memory!\0A\00", align 1 ; <[16 x i8]*> [#uses=1]
@llvm.used = appending global [1 x i8*] [i8* bitcast (void (%struct.callback_data*, i8*)* @set_table_name to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define fastcc void @set_table_name(%struct.callback_data* nocapture %p, i8* %zName) nounwind ssp {
entry:
  %0 = getelementptr inbounds %struct.callback_data* %p, i32 0, i32 7 ; <i8**> [#uses=3]
  %1 = load i8** %0, align 4                      ; <i8*> [#uses=2]
  %2 = icmp eq i8* %1, null                       ; <i1> [#uses=1]
  br i1 %2, label %bb1, label %bb

bb:                                               ; preds = %entry
  free i8* %1
  store i8* null, i8** %0, align 4
  br label %bb1

bb1:                                              ; preds = %bb, %entry
  %3 = icmp eq i8* %zName, null                   ; <i1> [#uses=1]
  br i1 %3, label %return, label %bb2

bb2:                                              ; preds = %bb1
  %4 = load i8* %zName, align 1                   ; <i8> [#uses=2]
  %5 = zext i8 %4 to i32                          ; <i32> [#uses=2]
  %6 = icmp sgt i8 %4, -1                         ; <i1> [#uses=1]
  br i1 %6, label %bb.i.i, label %bb1.i.i

bb.i.i:                                           ; preds = %bb2
  %7 = getelementptr inbounds %struct._RuneLocale* @_DefaultRuneLocale, i32 0, i32 5, i32 %5 ; <i32*> [#uses=1]
  %8 = load i32* %7, align 4                      ; <i32> [#uses=1]
  %9 = and i32 %8, 256                            ; <i32> [#uses=1]
  br label %isalpha.exit

bb1.i.i:                                          ; preds = %bb2
  %10 = tail call i32 @__maskrune(i32 %5, i32 256) nounwind ; <i32> [#uses=1]
  br label %isalpha.exit

isalpha.exit:                                     ; preds = %bb1.i.i, %bb.i.i
  %storemerge.in.in.i.i = phi i32 [ %9, %bb.i.i ], [ %10, %bb1.i.i ] ; <i32> [#uses=1]
  %storemerge.in.i.i = icmp eq i32 %storemerge.in.in.i.i, 0 ; <i1> [#uses=1]
  br i1 %storemerge.in.i.i, label %bb3, label %bb5

bb3:                                              ; preds = %isalpha.exit
  %11 = load i8* %zName, align 1                  ; <i8> [#uses=2]
  %12 = icmp eq i8 %11, 95                        ; <i1> [#uses=1]
  br i1 %12, label %bb5, label %bb12.preheader

bb5:                                              ; preds = %bb3, %isalpha.exit
  %.pre = load i8* %zName, align 1                ; <i8> [#uses=1]
  br label %bb12.preheader

bb12.preheader:                                   ; preds = %bb5, %bb3
  %13 = phi i8 [ %.pre, %bb5 ], [ %11, %bb3 ]     ; <i8> [#uses=1]
  %needQuote.1.ph = phi i32 [ 0, %bb5 ], [ 1, %bb3 ] ; <i32> [#uses=2]
  %14 = icmp eq i8 %13, 0                         ; <i1> [#uses=1]
  br i1 %14, label %bb13, label %bb7

bb7:                                              ; preds = %bb11, %bb12.preheader
  %i.011 = phi i32 [ %tmp17, %bb11 ], [ 0, %bb12.preheader ] ; <i32> [#uses=2]
  %n.110 = phi i32 [ %26, %bb11 ], [ 0, %bb12.preheader ] ; <i32> [#uses=3]
  %needQuote.19 = phi i32 [ %needQuote.0, %bb11 ], [ %needQuote.1.ph, %bb12.preheader ] ; <i32> [#uses=2]
  %scevgep16 = getelementptr i8* %zName, i32 %i.011 ; <i8*> [#uses=2]
  %tmp17 = add i32 %i.011, 1                      ; <i32> [#uses=2]
  %scevgep18 = getelementptr i8* %zName, i32 %tmp17 ; <i8*> [#uses=1]
  %15 = load i8* %scevgep16, align 1              ; <i8> [#uses=2]
  %16 = zext i8 %15 to i32                        ; <i32> [#uses=2]
  %17 = icmp sgt i8 %15, -1                       ; <i1> [#uses=1]
  br i1 %17, label %bb.i.i2, label %bb1.i.i3

bb.i.i2:                                          ; preds = %bb7
  %18 = getelementptr inbounds %struct._RuneLocale* @_DefaultRuneLocale, i32 0, i32 5, i32 %16 ; <i32*> [#uses=1]
  %19 = load i32* %18, align 4                    ; <i32> [#uses=1]
  %20 = and i32 %19, 1280                         ; <i32> [#uses=1]
  br label %isalnum.exit

bb1.i.i3:                                         ; preds = %bb7
  %21 = tail call i32 @__maskrune(i32 %16, i32 1280) nounwind ; <i32> [#uses=1]
  br label %isalnum.exit

isalnum.exit:                                     ; preds = %bb1.i.i3, %bb.i.i2
  %storemerge.in.in.i.i4 = phi i32 [ %20, %bb.i.i2 ], [ %21, %bb1.i.i3 ] ; <i32> [#uses=1]
  %storemerge.in.i.i5 = icmp eq i32 %storemerge.in.in.i.i4, 0 ; <i1> [#uses=1]
  br i1 %storemerge.in.i.i5, label %bb8, label %bb11

bb8:                                              ; preds = %isalnum.exit
  %22 = load i8* %scevgep16, align 1              ; <i8> [#uses=2]
  %23 = icmp eq i8 %22, 95                        ; <i1> [#uses=1]
  br i1 %23, label %bb11, label %bb9

bb9:                                              ; preds = %bb8
  %24 = icmp eq i8 %22, 39                        ; <i1> [#uses=1]
  %25 = zext i1 %24 to i32                        ; <i32> [#uses=1]
  %.n.1 = add i32 %n.110, %25                     ; <i32> [#uses=1]
  br label %bb11

bb11:                                             ; preds = %bb9, %bb8, %isalnum.exit
  %needQuote.0 = phi i32 [ 1, %bb9 ], [ %needQuote.19, %isalnum.exit ], [ %needQuote.19, %bb8 ] ; <i32> [#uses=2]
  %n.0 = phi i32 [ %.n.1, %bb9 ], [ %n.110, %isalnum.exit ], [ %n.110, %bb8 ] ; <i32> [#uses=1]
  %26 = add nsw i32 %n.0, 1                       ; <i32> [#uses=2]
  %27 = load i8* %scevgep18, align 1              ; <i8> [#uses=1]
  %28 = icmp eq i8 %27, 0                         ; <i1> [#uses=1]
  br i1 %28, label %bb13, label %bb7

bb13:                                             ; preds = %bb11, %bb12.preheader
  %n.1.lcssa = phi i32 [ 0, %bb12.preheader ], [ %26, %bb11 ] ; <i32> [#uses=2]
  %needQuote.1.lcssa = phi i32 [ %needQuote.1.ph, %bb12.preheader ], [ %needQuote.0, %bb11 ] ; <i32> [#uses=1]
  %29 = add nsw i32 %n.1.lcssa, 2                 ; <i32> [#uses=1]
  %30 = icmp eq i32 %needQuote.1.lcssa, 0         ; <i1> [#uses=3]
  %n.1. = select i1 %30, i32 %n.1.lcssa, i32 %29  ; <i32> [#uses=1]
  %31 = add nsw i32 %n.1., 1                      ; <i32> [#uses=1]
  %32 = malloc i8, i32 %31                        ; <i8*> [#uses=7]
  store i8* %32, i8** %0, align 4
  %33 = icmp eq i8* %32, null                     ; <i1> [#uses=1]
  br i1 %33, label %bb16, label %bb17

bb16:                                             ; preds = %bb13
  %34 = load %struct.FILE** @__stderrp, align 4   ; <%struct.FILE*> [#uses=1]
  %35 = bitcast %struct.FILE* %34 to i8*          ; <i8*> [#uses=1]
  %36 = tail call i32 @"\01_fwrite$UNIX2003"(i8* getelementptr inbounds ([16 x i8]* @.str10, i32 0, i32 0), i32 1, i32 15, i8* %35) nounwind ; <i32> [#uses=0]
  tail call void @exit(i32 1) noreturn nounwind
  unreachable

bb17:                                             ; preds = %bb13
  br i1 %30, label %bb23.preheader, label %bb18

bb18:                                             ; preds = %bb17
  store i8 39, i8* %32, align 4
  br label %bb23.preheader

bb23.preheader:                                   ; preds = %bb18, %bb17
  %n.3.ph = phi i32 [ 1, %bb18 ], [ 0, %bb17 ]    ; <i32> [#uses=2]
  %37 = load i8* %zName, align 1                  ; <i8> [#uses=1]
  %38 = icmp eq i8 %37, 0                         ; <i1> [#uses=1]
  br i1 %38, label %bb24, label %bb20

bb20:                                             ; preds = %bb22, %bb23.preheader
  %storemerge18 = phi i32 [ %tmp, %bb22 ], [ 0, %bb23.preheader ] ; <i32> [#uses=2]
  %n.37 = phi i32 [ %n.4, %bb22 ], [ %n.3.ph, %bb23.preheader ] ; <i32> [#uses=3]
  %scevgep = getelementptr i8* %zName, i32 %storemerge18 ; <i8*> [#uses=1]
  %tmp = add i32 %storemerge18, 1                 ; <i32> [#uses=2]
  %scevgep15 = getelementptr i8* %zName, i32 %tmp ; <i8*> [#uses=1]
  %39 = load i8* %scevgep, align 1                ; <i8> [#uses=2]
  %40 = getelementptr inbounds i8* %32, i32 %n.37 ; <i8*> [#uses=1]
  store i8 %39, i8* %40, align 1
  %41 = add nsw i32 %n.37, 1                      ; <i32> [#uses=2]
  %42 = icmp eq i8 %39, 39                        ; <i1> [#uses=1]
  br i1 %42, label %bb21, label %bb22

bb21:                                             ; preds = %bb20
  %43 = getelementptr inbounds i8* %32, i32 %41   ; <i8*> [#uses=1]
  store i8 39, i8* %43, align 1
  %44 = add nsw i32 %n.37, 2                      ; <i32> [#uses=1]
  br label %bb22

bb22:                                             ; preds = %bb21, %bb20
  %n.4 = phi i32 [ %44, %bb21 ], [ %41, %bb20 ]   ; <i32> [#uses=2]
  %45 = load i8* %scevgep15, align 1              ; <i8> [#uses=1]
  %46 = icmp eq i8 %45, 0                         ; <i1> [#uses=1]
  br i1 %46, label %bb24, label %bb20

bb24:                                             ; preds = %bb22, %bb23.preheader
  %n.3.lcssa = phi i32 [ %n.3.ph, %bb23.preheader ], [ %n.4, %bb22 ] ; <i32> [#uses=3]
  br i1 %30, label %bb26, label %bb25

bb25:                                             ; preds = %bb24
  %47 = getelementptr inbounds i8* %32, i32 %n.3.lcssa ; <i8*> [#uses=1]
  store i8 39, i8* %47, align 1
  %48 = add nsw i32 %n.3.lcssa, 1                 ; <i32> [#uses=1]
  br label %bb26

bb26:                                             ; preds = %bb25, %bb24
  %n.5 = phi i32 [ %48, %bb25 ], [ %n.3.lcssa, %bb24 ] ; <i32> [#uses=1]
  %49 = getelementptr inbounds i8* %32, i32 %n.5  ; <i8*> [#uses=1]
  store i8 0, i8* %49, align 1
  ret void

return:                                           ; preds = %bb1
  ret void
}

declare i32 @"\01_fwrite$UNIX2003"(i8*, i32, i32, i8*)

declare void @exit(i32) noreturn nounwind

declare i32 @__maskrune(i32, i32)
