;; Register allocation is doing a very poor job on this routine from yyparse
;; in Burg:
;; -- at least two long-lived values are being allocated to %o? registers
;; -- even worse, those registers are being saved and restored repeatedly
;;    at function calls, even though there are no intervening uses.
;; -- outgoing args of some function calls have to be swapped, causing
;;    another write/read from stack to do the exchange (use -dregalloc=y).
;;
	%Arity = type %struct.arity*
	%Binding = type %struct.binding*
	%DeltaCost = type [4 x short]
	%Dimension = type %struct.dimension*
	%Index_Map = type { int, %Item_Set* }
	%IntList = type %struct.intlist*
	%Item = type { %DeltaCost, %Rule }
	%ItemArray = type %Item*
	%Item_Set = type %struct.item_set*
	%List = type %struct.list*
	%Mapping = type %struct.mapping*
	%NonTerminal = type %struct.nonterminal*
	%Operator = type %struct.operator*
	%Pattern = type %struct.pattern*
	%PatternAST = type %struct.patternAST*
	%Plank = type %struct.plank*
	%PlankMap = type %struct.plankMap*
	%ReadFn = type int ()*
	%Rule = type %struct.rule*
	%RuleAST = type %struct.ruleAST*
	%StateMap = type %struct.stateMap*
	%StrTableElement = type %struct.strTableElement*
	%Symbol = type %struct.symbol*
	%Table = type %struct.table*
	%YYSTYPE = type { %IntList }
	%struct.arity = type { int, %List }
	%struct.binding = type { sbyte*, int }
	%struct.dimension = type { short*, %Index_Map, %Mapping, int, %PlankMap }
	%struct.index_map = type { int, %Item_Set* }
	%struct.intlist = type { int, %IntList }
	%struct.item = type { %DeltaCost, %Rule }
	%struct.item_set = type { int, int, %Operator, [2 x %Item_Set], %Item_Set, short*, %ItemArray, %ItemArray }
	%struct.list = type { sbyte*, %List }
	%struct.mapping = type { %List*, int, int, int, %Item_Set* }
	%struct.nonterminal = type { sbyte*, int, int, int, %PlankMap, %Rule }
	%struct.operator = type { sbyte*, uint, int, int, int, int, %Table }
	%struct.pattern = type { %NonTerminal, %Operator, [2 x %NonTerminal] }
	%struct.patternAST = type { %Symbol, sbyte*, %List }
	%struct.plank = type { sbyte*, %List, int }
	%struct.plankMap = type { %List, int, %StateMap }
	%struct.rule = type { %DeltaCost, int, int, int, %NonTerminal, %Pattern, uint }
	%struct.ruleAST = type { sbyte*, %PatternAST, int, %IntList, %Rule, %StrTableElement, %StrTableElement }
	%struct.stateMap = type { sbyte*, %Plank, int, short* }
	%struct.strTableElement = type { sbyte*, %IntList, sbyte* }
	%struct.symbol = type { sbyte*, int, { %Operator } }
	%struct.table = type { %Operator, %List, short*, [2 x %Dimension], %Item_Set* }
%yylval = external global %YYSTYPE		; <%YYSTYPE*> [#uses=1]
%yylhs = external global [25 x short]		; <[25 x short]*> [#uses=1]
%yylen = external global [25 x short]		; <[25 x short]*> [#uses=1]
%yydefred = external global [43 x short]		; <[43 x short]*> [#uses=1]
%yydgoto = external global [12 x short]		; <[12 x short]*> [#uses=1]
%yysindex = external global [43 x short]		; <[43 x short]*> [#uses=2]
%yyrindex = external global [43 x short]		; <[43 x short]*> [#uses=1]
%yygindex = external global [12 x short]		; <[12 x short]*> [#uses=1]
%yytable = external global [263 x short]		; <[263 x short]*> [#uses=4]
%yycheck = external global [263 x short]		; <[263 x short]*> [#uses=4]
%yynerrs = external global int		; <int*> [#uses=3]
%yyerrflag = external global int		; <int*> [#uses=6]
%yychar = external global int		; <int*> [#uses=15]
%yyssp = external global short*		; <short**> [#uses=15]
%yyvsp = external global %YYSTYPE*		; <%YYSTYPE**> [#uses=30]
%yyval = external global %YYSTYPE		; <%YYSTYPE*> [#uses=1]
%yyss = external global short*		; <short**> [#uses=3]
%yysslim = external global short*		; <short**> [#uses=3]
%yyvs = external global %YYSTYPE*		; <%YYSTYPE**> [#uses=1]
%.LC01 = external global [13 x sbyte]		; <[13 x sbyte]*> [#uses=1]
%.LC1 = external global [20 x sbyte]		; <[20 x sbyte]*> [#uses=1]

implementation   ; Functions:

int %yyparse() {
bb0:		; No predecessors!
	store int 0, int* %yynerrs
	store int 0, int* %yyerrflag
	store int -1, int* %yychar
	%reg113 = load short** %yyss		; <short*> [#uses=1]
	%cond581 = setne short* %reg113, null		; <bool> [#uses=1]
	br bool %cond581, label %bb3, label %bb2

bb2:		; preds = %bb0
	%reg584 = call int %yygrowstack( )		; <int> [#uses=1]
	%cond584 = setne int %reg584, 0		; <bool> [#uses=1]
	br bool %cond584, label %bb113, label %bb3

bb3:		; preds = %bb2, %bb0
	%reg115 = load short** %yyss		; <short*> [#uses=1]
	store short* %reg115, short** %yyssp
	%reg116 = load %YYSTYPE** %yyvs		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %reg116, %YYSTYPE** %yyvsp
	%reg117 = load short** %yyssp		; <short*> [#uses=1]
	store short 0, short* %reg117
	br label %bb4

bb4:		; preds = %bb14, %bb15, %bb31, %bb35, %bb102, %bb112, %bb3
	%reg458 = phi uint [ %reg476, %bb112 ], [ 1, %bb102 ], [ %reg458, %bb35 ], [ %cast768, %bb31 ], [ %cast658, %bb15 ], [ %cast658, %bb14 ], [ 0, %bb3 ]		; <uint> [#uses=2]
	%reg458-idxcast = cast uint %reg458 to long		; <long> [#uses=3]
	%reg594 = getelementptr [43 x short]* %yydefred, long 0, long %reg458-idxcast		; <short*> [#uses=1]
	%reg125 = load short* %reg594		; <short> [#uses=1]
	%cast599 = cast short %reg125 to int		; <int> [#uses=2]
	%cond600 = setne int %cast599, 0		; <bool> [#uses=1]
	br bool %cond600, label %bb36, label %bb5

bb5:		; preds = %bb4
	%reg127 = load int* %yychar		; <int> [#uses=1]
	%cond603 = setge int %reg127, 0		; <bool> [#uses=1]
	br bool %cond603, label %bb8, label %bb6

bb6:		; preds = %bb5
	%reg607 = call int %yylex( )		; <int> [#uses=1]
	store int %reg607, int* %yychar
	%reg129 = load int* %yychar		; <int> [#uses=1]
	%cond609 = setge int %reg129, 0		; <bool> [#uses=1]
	br bool %cond609, label %bb8, label %bb7

bb7:		; preds = %bb6
	store int 0, int* %yychar
	br label %bb8

bb8:		; preds = %bb7, %bb6, %bb5
	%reg615 = getelementptr [43 x short]* %yysindex, long 0, long %reg458-idxcast		; <short*> [#uses=1]
	%reg137 = load short* %reg615		; <short> [#uses=1]
	%cast620 = cast short %reg137 to int		; <int> [#uses=2]
	%cond621 = seteq int %cast620, 0		; <bool> [#uses=1]
	br bool %cond621, label %bb16, label %bb9

bb9:		; preds = %bb8
	%reg139 = load int* %yychar		; <int> [#uses=2]
	%reg460 = add int %cast620, %reg139		; <int> [#uses=3]
	%cond624 = setlt int %reg460, 0		; <bool> [#uses=1]
	br bool %cond624, label %bb16, label %bb10

bb10:		; preds = %bb9
	%cond627 = setgt int %reg460, 262		; <bool> [#uses=1]
	br bool %cond627, label %bb16, label %bb11

bb11:		; preds = %bb10
	%reg460-idxcast = cast int %reg460 to long		; <long> [#uses=2]
	%reg632 = getelementptr [263 x short]* %yycheck, long 0, long %reg460-idxcast		; <short*> [#uses=1]
	%reg148 = load short* %reg632		; <short> [#uses=1]
	%cast637 = cast short %reg148 to int		; <int> [#uses=1]
	%cond639 = setne int %cast637, %reg139		; <bool> [#uses=1]
	br bool %cond639, label %bb16, label %bb12

bb12:		; preds = %bb11
	%reg150 = load short** %yyssp		; <short*> [#uses=1]
	%cast640 = cast short* %reg150 to sbyte*		; <sbyte*> [#uses=1]
	%reg151 = load short** %yysslim		; <short*> [#uses=1]
	%cast641 = cast short* %reg151 to sbyte*		; <sbyte*> [#uses=1]
	%cond642 = setlt sbyte* %cast640, %cast641		; <bool> [#uses=1]
	br bool %cond642, label %bb14, label %bb13

bb13:		; preds = %bb12
	%reg644 = call int %yygrowstack( )		; <int> [#uses=1]
	%cond644 = setne int %reg644, 0		; <bool> [#uses=1]
	br bool %cond644, label %bb113, label %bb14

bb14:		; preds = %bb13, %bb12
	%reg153 = load short** %yyssp		; <short*> [#uses=1]
	%reg647 = getelementptr short* %reg153, long 1		; <short*> [#uses=2]
	store short* %reg647, short** %yyssp
	%reg653 = getelementptr [263 x short]* %yytable, long 0, long %reg460-idxcast		; <short*> [#uses=1]
	%reg162 = load short* %reg653		; <short> [#uses=2]
	%cast658 = cast short %reg162 to uint		; <uint> [#uses=2]
	store short %reg162, short* %reg647
	%reg164 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg661 = getelementptr %YYSTYPE* %reg164, long 1		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %reg661, %YYSTYPE** %yyvsp
	%reg167 = load %IntList* getelementptr (%YYSTYPE* %yylval, long 0, ubyte 0)		; <%IntList> [#uses=1]
	%reg661.idx1 = getelementptr %YYSTYPE* %reg164, long 1, ubyte 0		; <%IntList*> [#uses=1]
	store %IntList %reg167, %IntList* %reg661.idx1
	store int -1, int* %yychar
	%reg169 = load int* %yyerrflag		; <int> [#uses=2]
	%cond669 = setle int %reg169, 0		; <bool> [#uses=1]
	br bool %cond669, label %bb4, label %bb15

bb15:		; preds = %bb14
	%reg171 = add int %reg169, -1		; <int> [#uses=1]
	store int %reg171, int* %yyerrflag
	br label %bb4

bb16:		; preds = %bb11, %bb10, %bb9, %bb8
	%reg677 = getelementptr [43 x short]* %yyrindex, long 0, long %reg458-idxcast		; <short*> [#uses=1]
	%reg178 = load short* %reg677		; <short> [#uses=1]
	%cast682 = cast short %reg178 to int		; <int> [#uses=2]
	%cond683 = seteq int %cast682, 0		; <bool> [#uses=1]
	br bool %cond683, label %bb21, label %bb17

bb17:		; preds = %bb16
	%reg180 = load int* %yychar		; <int> [#uses=2]
	%reg463 = add int %cast682, %reg180		; <int> [#uses=3]
	%cond686 = setlt int %reg463, 0		; <bool> [#uses=1]
	br bool %cond686, label %bb21, label %bb18

bb18:		; preds = %bb17
	%cond689 = setgt int %reg463, 262		; <bool> [#uses=1]
	br bool %cond689, label %bb21, label %bb19

bb19:		; preds = %bb18
	%reg463-idxcast = cast int %reg463 to long		; <long> [#uses=2]
	%reg694 = getelementptr [263 x short]* %yycheck, long 0, long %reg463-idxcast		; <short*> [#uses=1]
	%reg189 = load short* %reg694		; <short> [#uses=1]
	%cast699 = cast short %reg189 to int		; <int> [#uses=1]
	%cond701 = setne int %cast699, %reg180		; <bool> [#uses=1]
	br bool %cond701, label %bb21, label %bb20

bb20:		; preds = %bb19
	%reg704 = getelementptr [263 x short]* %yytable, long 0, long %reg463-idxcast		; <short*> [#uses=1]
	%reg197 = load short* %reg704		; <short> [#uses=1]
	%cast709 = cast short %reg197 to int		; <int> [#uses=1]
	br label %bb36

bb21:		; preds = %bb19, %bb18, %bb17, %bb16
	%reg198 = load int* %yyerrflag		; <int> [#uses=1]
	%cond711 = setne int %reg198, 0		; <bool> [#uses=1]
	br bool %cond711, label %bb23, label %bb22

bb22:		; preds = %bb21
	call void %yyerror( sbyte* getelementptr ([13 x sbyte]* %.LC01, long 0, long 0) )
	%reg200 = load int* %yynerrs		; <int> [#uses=1]
	%reg201 = add int %reg200, 1		; <int> [#uses=1]
	store int %reg201, int* %yynerrs
	br label %bb23

bb23:		; preds = %bb22, %bb21
	%reg202 = load int* %yyerrflag		; <int> [#uses=1]
	%cond719 = setgt int %reg202, 2		; <bool> [#uses=1]
	br bool %cond719, label %bb34, label %bb24

bb24:		; preds = %bb23
	store int 3, int* %yyerrflag
	%reg241 = load short** %yyss		; <short*> [#uses=1]
	%cast778 = cast short* %reg241 to sbyte*		; <sbyte*> [#uses=1]
	br label %bb25

bb25:		; preds = %bb33, %bb24
	%reg204 = load short** %yyssp		; <short*> [#uses=4]
	%reg206 = load short* %reg204		; <short> [#uses=1]
	%reg206-idxcast = cast short %reg206 to long		; <long> [#uses=1]
	%reg727 = getelementptr [43 x short]* %yysindex, long 0, long %reg206-idxcast		; <short*> [#uses=1]
	%reg212 = load short* %reg727		; <short> [#uses=2]
	%cast732 = cast short %reg212 to int		; <int> [#uses=2]
	%cond733 = seteq int %cast732, 0		; <bool> [#uses=1]
	br bool %cond733, label %bb32, label %bb26

bb26:		; preds = %bb25
	%reg466 = add int %cast732, 256		; <int> [#uses=2]
	%cond736 = setlt int %reg466, 0		; <bool> [#uses=1]
	br bool %cond736, label %bb32, label %bb27

bb27:		; preds = %bb26
	%cond739 = setgt int %reg466, 262		; <bool> [#uses=1]
	br bool %cond739, label %bb32, label %bb28

bb28:		; preds = %bb27
	%reg212-idxcast = cast short %reg212 to long		; <long> [#uses=1]
	%reg212-idxcast-offset = add long %reg212-idxcast, 256		; <long> [#uses=2]
	%reg744 = getelementptr [263 x short]* %yycheck, long 0, long %reg212-idxcast-offset		; <short*> [#uses=1]
	%reg221 = load short* %reg744		; <short> [#uses=1]
	%cond748 = setne short %reg221, 256		; <bool> [#uses=1]
	br bool %cond748, label %bb32, label %bb29

bb29:		; preds = %bb28
	%cast750 = cast short* %reg204 to sbyte*		; <sbyte*> [#uses=1]
	%reg223 = load short** %yysslim		; <short*> [#uses=1]
	%cast751 = cast short* %reg223 to sbyte*		; <sbyte*> [#uses=1]
	%cond752 = setlt sbyte* %cast750, %cast751		; <bool> [#uses=1]
	br bool %cond752, label %bb31, label %bb30

bb30:		; preds = %bb29
	%reg754 = call int %yygrowstack( )		; <int> [#uses=1]
	%cond754 = setne int %reg754, 0		; <bool> [#uses=1]
	br bool %cond754, label %bb113, label %bb31

bb31:		; preds = %bb30, %bb29
	%reg225 = load short** %yyssp		; <short*> [#uses=1]
	%reg757 = getelementptr short* %reg225, long 1		; <short*> [#uses=2]
	store short* %reg757, short** %yyssp
	%reg763 = getelementptr [263 x short]* %yytable, long 0, long %reg212-idxcast-offset		; <short*> [#uses=1]
	%reg234 = load short* %reg763		; <short> [#uses=2]
	%cast768 = cast short %reg234 to uint		; <uint> [#uses=1]
	store short %reg234, short* %reg757
	%reg236 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg771 = getelementptr %YYSTYPE* %reg236, long 1		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %reg771, %YYSTYPE** %yyvsp
	%reg239 = load %IntList* getelementptr (%YYSTYPE* %yylval, long 0, ubyte 0)		; <%IntList> [#uses=1]
	%reg771.idx1 = getelementptr %YYSTYPE* %reg236, long 1, ubyte 0		; <%IntList*> [#uses=1]
	store %IntList %reg239, %IntList* %reg771.idx1
	br label %bb4

bb32:		; preds = %bb28, %bb27, %bb26, %bb25
	%cast777 = cast short* %reg204 to sbyte*		; <sbyte*> [#uses=1]
	%cond779 = setle sbyte* %cast777, %cast778		; <bool> [#uses=1]
	br bool %cond779, label %UnifiedExitNode, label %bb33

bb33:		; preds = %bb32
	%reg781 = getelementptr short* %reg204, long -1		; <short*> [#uses=1]
	store short* %reg781, short** %yyssp
	%reg244 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=1]
	%reg786 = getelementptr %YYSTYPE* %reg244, long -1		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %reg786, %YYSTYPE** %yyvsp
	br label %bb25

bb34:		; preds = %bb23
	%reg246 = load int* %yychar		; <int> [#uses=1]
	%cond791 = seteq int %reg246, 0		; <bool> [#uses=1]
	br bool %cond791, label %UnifiedExitNode, label %bb35

bb35:		; preds = %bb34
	store int -1, int* %yychar
	br label %bb4

bb36:		; preds = %bb20, %bb4
	%reg468 = phi int [ %cast709, %bb20 ], [ %cast599, %bb4 ]		; <int> [#uses=31]
	%reg468-idxcast = cast int %reg468 to long		; <long> [#uses=2]
	%reg796 = getelementptr [25 x short]* %yylen, long 0, long %reg468-idxcast		; <short*> [#uses=1]
	%reg254 = load short* %reg796		; <short> [#uses=2]
	%reg259 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=1]
	%reg254-idxcast = cast short %reg254 to long		; <long> [#uses=1]
	%reg254-idxcast-scale = mul long %reg254-idxcast, -1		; <long> [#uses=1]
	%reg254-idxcast-scale-offset = add long %reg254-idxcast-scale, 1		; <long> [#uses=1]
	%reg261.idx1 = getelementptr %YYSTYPE* %reg259, long %reg254-idxcast-scale-offset, ubyte 0		; <%IntList*> [#uses=1]
	%reg261 = load %IntList* %reg261.idx1		; <%IntList> [#uses=1]
	store %IntList %reg261, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	%cond812 = seteq int %reg468, 13		; <bool> [#uses=1]
	br bool %cond812, label %bb85, label %bb37

bb37:		; preds = %bb36
	%cond814 = setgt int %reg468, 13		; <bool> [#uses=1]
	br bool %cond814, label %bb56, label %bb38

bb38:		; preds = %bb37
	%cond817 = seteq int %reg468, 7		; <bool> [#uses=1]
	br bool %cond817, label %bb79, label %bb39

bb39:		; preds = %bb38
	%cond819 = setgt int %reg468, 7		; <bool> [#uses=1]
	br bool %cond819, label %bb48, label %bb40

bb40:		; preds = %bb39
	%cond822 = seteq int %reg468, 4		; <bool> [#uses=1]
	br bool %cond822, label %bb76, label %bb41

bb41:		; preds = %bb40
	%cond824 = setgt int %reg468, 4		; <bool> [#uses=1]
	br bool %cond824, label %bb45, label %bb42

bb42:		; preds = %bb41
	%cond827 = seteq int %reg468, 2		; <bool> [#uses=1]
	br bool %cond827, label %bb74, label %bb43

bb43:		; preds = %bb42
	%cond829 = seteq int %reg468, 3		; <bool> [#uses=1]
	br bool %cond829, label %bb75, label %bb97

bb45:		; preds = %bb41
	%cond831 = seteq int %reg468, 5		; <bool> [#uses=1]
	br bool %cond831, label %bb77, label %bb46

bb46:		; preds = %bb45
	%cond833 = seteq int %reg468, 6		; <bool> [#uses=1]
	br bool %cond833, label %bb78, label %bb97

bb48:		; preds = %bb39
	%cond835 = seteq int %reg468, 10		; <bool> [#uses=1]
	br bool %cond835, label %bb82, label %bb49

bb49:		; preds = %bb48
	%cond837 = setgt int %reg468, 10		; <bool> [#uses=1]
	br bool %cond837, label %bb53, label %bb50

bb50:		; preds = %bb49
	%cond840 = seteq int %reg468, 8		; <bool> [#uses=1]
	br bool %cond840, label %bb80, label %bb51

bb51:		; preds = %bb50
	%cond842 = seteq int %reg468, 9		; <bool> [#uses=1]
	br bool %cond842, label %bb81, label %bb97

bb53:		; preds = %bb49
	%cond844 = seteq int %reg468, 11		; <bool> [#uses=1]
	br bool %cond844, label %bb83, label %bb54

bb54:		; preds = %bb53
	%cond846 = seteq int %reg468, 12		; <bool> [#uses=1]
	br bool %cond846, label %bb84, label %bb97

bb56:		; preds = %bb37
	%cond848 = seteq int %reg468, 19		; <bool> [#uses=1]
	br bool %cond848, label %bb91, label %bb57

bb57:		; preds = %bb56
	%cond850 = setgt int %reg468, 19		; <bool> [#uses=1]
	br bool %cond850, label %bb66, label %bb58

bb58:		; preds = %bb57
	%cond853 = seteq int %reg468, 16		; <bool> [#uses=1]
	br bool %cond853, label %bb88, label %bb59

bb59:		; preds = %bb58
	%cond855 = setgt int %reg468, 16		; <bool> [#uses=1]
	br bool %cond855, label %bb63, label %bb60

bb60:		; preds = %bb59
	%cond858 = seteq int %reg468, 14		; <bool> [#uses=1]
	br bool %cond858, label %bb86, label %bb61

bb61:		; preds = %bb60
	%cond860 = seteq int %reg468, 15		; <bool> [#uses=1]
	br bool %cond860, label %bb87, label %bb97

bb63:		; preds = %bb59
	%cond862 = seteq int %reg468, 17		; <bool> [#uses=1]
	br bool %cond862, label %bb89, label %bb64

bb64:		; preds = %bb63
	%cond864 = seteq int %reg468, 18		; <bool> [#uses=1]
	br bool %cond864, label %bb90, label %bb97

bb66:		; preds = %bb57
	%cond866 = seteq int %reg468, 22		; <bool> [#uses=1]
	br bool %cond866, label %bb94, label %bb67

bb67:		; preds = %bb66
	%cond868 = setgt int %reg468, 22		; <bool> [#uses=1]
	br bool %cond868, label %bb71, label %bb68

bb68:		; preds = %bb67
	%cond871 = seteq int %reg468, 20		; <bool> [#uses=1]
	br bool %cond871, label %bb92, label %bb69

bb69:		; preds = %bb68
	%cond873 = seteq int %reg468, 21		; <bool> [#uses=1]
	br bool %cond873, label %bb93, label %bb97

bb71:		; preds = %bb67
	%cond875 = seteq int %reg468, 23		; <bool> [#uses=1]
	br bool %cond875, label %bb95, label %bb72

bb72:		; preds = %bb71
	%cond877 = seteq int %reg468, 24		; <bool> [#uses=1]
	br bool %cond877, label %bb96, label %bb97

bb74:		; preds = %bb42
	call void %yyfinished( )
	br label %bb97

bb75:		; preds = %bb43
	%reg262 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg264.idx1 = getelementptr %YYSTYPE* %reg262, long -2, ubyte 0		; <%IntList*> [#uses=1]
	%reg264 = load %IntList* %reg264.idx1		; <%IntList> [#uses=1]
	%reg265.idx = getelementptr %YYSTYPE* %reg262, long 0, ubyte 0		; <%IntList*> [#uses=1]
	%reg265 = load %IntList* %reg265.idx		; <%IntList> [#uses=1]
	%cast889 = cast %IntList %reg265 to %List		; <%List> [#uses=1]
	%cast890 = cast %IntList %reg264 to %List		; <%List> [#uses=1]
	call void %doSpec( %List %cast890, %List %cast889 )
	br label %bb97

bb76:		; preds = %bb40
	store %IntList null, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb77:		; preds = %bb45
	%reg269 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%cast894 = getelementptr %YYSTYPE* %reg269, long 0, ubyte 0		; <%IntList*> [#uses=1]
	%reg271 = load %IntList* %cast894		; <%IntList> [#uses=1]
	%reg271 = cast %IntList %reg271 to sbyte*		; <sbyte*> [#uses=1]
	%reg272.idx1 = getelementptr %YYSTYPE* %reg269, long -1, ubyte 0		; <%IntList*> [#uses=1]
	%reg272 = load %IntList* %reg272.idx1		; <%IntList> [#uses=1]
	%cast901 = cast %IntList %reg272 to %List		; <%List> [#uses=1]
	%reg901 = call %List %newList( sbyte* %reg271, %List %cast901 )		; <%List> [#uses=1]
	cast %List %reg901 to %IntList		; <%IntList>:0 [#uses=1]
	store %IntList %0, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb78:		; preds = %bb46
	%reg275 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=1]
	%reg277.idx = getelementptr %YYSTYPE* %reg275, long 0, ubyte 0		; <%IntList*> [#uses=1]
	%reg277 = load %IntList* %reg277.idx		; <%IntList> [#uses=1]
	%cast907 = cast %IntList %reg277 to %List		; <%List> [#uses=1]
	%reg907 = call %Arity %newArity( int -1, %List %cast907 )		; <%Arity> [#uses=1]
	cast %Arity %reg907 to %IntList		; <%IntList>:1 [#uses=1]
	store %IntList %1, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb79:		; preds = %bb38
	store %IntList null, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	%reg281 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=1]
	%cast912 = getelementptr %YYSTYPE* %reg281, long 0, ubyte 0		; <%IntList*> [#uses=1]
	%reg282 = load %IntList* %cast912		; <%IntList> [#uses=1]
	%reg282 = cast %IntList %reg282 to %List		; <%List> [#uses=1]
	call void %doGram( %List %reg282 )
	br label %bb97

bb80:		; preds = %bb50
	store %IntList null, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	%reg285 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=1]
	%cast917 = getelementptr %YYSTYPE* %reg285, long 0, ubyte 0		; <%IntList*> [#uses=1]
	%reg286 = load %IntList* %cast917		; <%IntList> [#uses=1]
	%reg286 = cast %IntList %reg286 to sbyte*		; <sbyte*> [#uses=1]
	call void %doStart( sbyte* %reg286 )
	br label %bb97

bb81:		; preds = %bb51
	store %IntList null, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb82:		; preds = %bb48
	%reg290 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%cast923 = getelementptr %YYSTYPE* %reg290, long 0, ubyte 0		; <%IntList*> [#uses=1]
	%reg292 = load %IntList* %cast923		; <%IntList> [#uses=1]
	%reg292 = cast %IntList %reg292 to sbyte*		; <sbyte*> [#uses=1]
	%reg293.idx1 = getelementptr %YYSTYPE* %reg290, long -1, ubyte 0		; <%IntList*> [#uses=1]
	%reg293 = load %IntList* %reg293.idx1		; <%IntList> [#uses=1]
	%cast930 = cast %IntList %reg293 to %List		; <%List> [#uses=1]
	%reg930 = call %List %newList( sbyte* %reg292, %List %cast930 )		; <%List> [#uses=1]
	cast %List %reg930 to %IntList		; <%IntList>:2 [#uses=1]
	store %IntList %2, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb83:		; preds = %bb53
	store %IntList null, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb84:		; preds = %bb54
	%reg298 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%cast936 = getelementptr %YYSTYPE* %reg298, long 0, ubyte 0		; <%IntList*> [#uses=1]
	%reg300 = load %IntList* %cast936		; <%IntList> [#uses=1]
	%reg300 = cast %IntList %reg300 to sbyte*		; <sbyte*> [#uses=1]
	%reg301.idx1 = getelementptr %YYSTYPE* %reg298, long -1, ubyte 0		; <%IntList*> [#uses=1]
	%reg301 = load %IntList* %reg301.idx1		; <%IntList> [#uses=1]
	%cast943 = cast %IntList %reg301 to %List		; <%List> [#uses=1]
	%reg943 = call %List %newList( sbyte* %reg300, %List %cast943 )		; <%List> [#uses=1]
	cast %List %reg943 to %IntList		; <%IntList>:3 [#uses=1]
	store %IntList %3, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb85:		; preds = %bb36
	%reg304 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%cast9521 = getelementptr %YYSTYPE* %reg304, long -2, ubyte 0		; <%IntList*> [#uses=1]
	%reg306 = load %IntList* %cast9521		; <%IntList> [#uses=1]
	%reg306 = cast %IntList %reg306 to sbyte*		; <sbyte*> [#uses=1]
	%cast953 = cast %YYSTYPE* %reg304 to int*		; <int*> [#uses=1]
	%reg307 = load int* %cast953		; <int> [#uses=1]
	%reg955 = call %Binding %newBinding( sbyte* %reg306, int %reg307 )		; <%Binding> [#uses=1]
	cast %Binding %reg955 to %IntList		; <%IntList>:4 [#uses=1]
	store %IntList %4, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb86:		; preds = %bb60
	store %IntList null, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb87:		; preds = %bb61
	%reg312 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%cast961 = getelementptr %YYSTYPE* %reg312, long 0, ubyte 0		; <%IntList*> [#uses=1]
	%reg314 = load %IntList* %cast961		; <%IntList> [#uses=1]
	%reg314 = cast %IntList %reg314 to sbyte*		; <sbyte*> [#uses=1]
	%reg315.idx1 = getelementptr %YYSTYPE* %reg312, long -1, ubyte 0		; <%IntList*> [#uses=1]
	%reg315 = load %IntList* %reg315.idx1		; <%IntList> [#uses=1]
	%cast968 = cast %IntList %reg315 to %List		; <%List> [#uses=1]
	%reg968 = call %List %newList( sbyte* %reg314, %List %cast968 )		; <%List> [#uses=1]
	cast %List %reg968 to %IntList		; <%IntList>:5 [#uses=1]
	store %IntList %5, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb88:		; preds = %bb58
	%reg318 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=4]
	%cast9791 = getelementptr %YYSTYPE* %reg318, long -6, ubyte 0		; <%IntList*> [#uses=1]
	%reg322 = load %IntList* %cast9791		; <%IntList> [#uses=1]
	%reg322 = cast %IntList %reg322 to sbyte*		; <sbyte*> [#uses=1]
	%reg323.idx1 = getelementptr %YYSTYPE* %reg318, long -4, ubyte 0		; <%IntList*> [#uses=1]
	%reg323 = load %IntList* %reg323.idx1		; <%IntList> [#uses=1]
	%reg987 = getelementptr %YYSTYPE* %reg318, long -2		; <%YYSTYPE*> [#uses=1]
	%cast989 = cast %YYSTYPE* %reg987 to int*		; <int*> [#uses=1]
	%reg324 = load int* %cast989		; <int> [#uses=1]
	%reg325.idx1 = getelementptr %YYSTYPE* %reg318, long -1, ubyte 0		; <%IntList*> [#uses=1]
	%reg325 = load %IntList* %reg325.idx1		; <%IntList> [#uses=1]
	%cast998 = cast %IntList %reg323 to %PatternAST		; <%PatternAST> [#uses=1]
	%reg996 = call %RuleAST %newRuleAST( sbyte* %reg322, %PatternAST %cast998, int %reg324, %IntList %reg325 )		; <%RuleAST> [#uses=1]
	cast %RuleAST %reg996 to %IntList		; <%IntList>:6 [#uses=1]
	store %IntList %6, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb89:		; preds = %bb63
	%reg328 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=1]
	%cast1002 = getelementptr %YYSTYPE* %reg328, long 0, ubyte 0		; <%IntList*> [#uses=1]
	%reg329 = load %IntList* %cast1002		; <%IntList> [#uses=1]
	%reg329 = cast %IntList %reg329 to sbyte*		; <sbyte*> [#uses=1]
	%reg1004 = call %PatternAST %newPatternAST( sbyte* %reg329, %List null )		; <%PatternAST> [#uses=1]
	cast %PatternAST %reg1004 to %IntList		; <%IntList>:7 [#uses=1]
	store %IntList %7, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb90:		; preds = %bb64
	%reg333 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%cast10131 = getelementptr %YYSTYPE* %reg333, long -1, ubyte 0		; <%IntList*> [#uses=1]
	%reg335 = load %IntList* %cast10131		; <%IntList> [#uses=1]
	%reg335 = cast %IntList %reg335 to sbyte*		; <sbyte*> [#uses=1]
	%reg1015 = call %List %newList( sbyte* %reg335, %List null )		; <%List> [#uses=1]
	%cast10211 = getelementptr %YYSTYPE* %reg333, long -3, ubyte 0		; <%IntList*> [#uses=1]
	%reg338 = load %IntList* %cast10211		; <%IntList> [#uses=1]
	%reg338 = cast %IntList %reg338 to sbyte*		; <sbyte*> [#uses=1]
	%reg1023 = call %PatternAST %newPatternAST( sbyte* %reg338, %List %reg1015 )		; <%PatternAST> [#uses=1]
	cast %PatternAST %reg1023 to %IntList		; <%IntList>:8 [#uses=1]
	store %IntList %8, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb91:		; preds = %bb56
	%reg341 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=3]
	%cast10331 = getelementptr %YYSTYPE* %reg341, long -1, ubyte 0		; <%IntList*> [#uses=1]
	%reg344 = load %IntList* %cast10331		; <%IntList> [#uses=1]
	%reg344 = cast %IntList %reg344 to sbyte*		; <sbyte*> [#uses=1]
	%reg1035 = call %List %newList( sbyte* %reg344, %List null )		; <%List> [#uses=1]
	%cast10411 = getelementptr %YYSTYPE* %reg341, long -3, ubyte 0		; <%IntList*> [#uses=1]
	%reg347 = load %IntList* %cast10411		; <%IntList> [#uses=1]
	%reg347 = cast %IntList %reg347 to sbyte*		; <sbyte*> [#uses=1]
	%reg1043 = call %List %newList( sbyte* %reg347, %List %reg1035 )		; <%List> [#uses=1]
	%cast10491 = getelementptr %YYSTYPE* %reg341, long -5, ubyte 0		; <%IntList*> [#uses=1]
	%reg349 = load %IntList* %cast10491		; <%IntList> [#uses=1]
	%reg349 = cast %IntList %reg349 to sbyte*		; <sbyte*> [#uses=1]
	%reg1051 = call %PatternAST %newPatternAST( sbyte* %reg349, %List %reg1043 )		; <%PatternAST> [#uses=1]
	cast %PatternAST %reg1051 to %IntList		; <%IntList>:9 [#uses=1]
	store %IntList %9, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb92:		; preds = %bb68
	store %IntList null, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb93:		; preds = %bb69
	%reg354 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg1059 = getelementptr %YYSTYPE* %reg354, long -2		; <%YYSTYPE*> [#uses=1]
	%cast1061 = cast %YYSTYPE* %reg1059 to int*		; <int*> [#uses=1]
	%reg356 = load int* %cast1061		; <int> [#uses=1]
	%reg357.idx1 = getelementptr %YYSTYPE* %reg354, long -1, ubyte 0		; <%IntList*> [#uses=1]
	%reg357 = load %IntList* %reg357.idx1		; <%IntList> [#uses=1]
	%reg1068 = call %IntList %newIntList( int %reg356, %IntList %reg357 )		; <%IntList> [#uses=1]
	store %IntList %reg1068, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb94:		; preds = %bb66
	store %IntList null, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb95:		; preds = %bb71
	%reg362 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg1076 = getelementptr %YYSTYPE* %reg362, long -1		; <%YYSTYPE*> [#uses=1]
	%cast1078 = cast %YYSTYPE* %reg1076 to int*		; <int*> [#uses=1]
	%reg364 = load int* %cast1078		; <int> [#uses=1]
	%reg365.idx = getelementptr %YYSTYPE* %reg362, long 0, ubyte 0		; <%IntList*> [#uses=1]
	%reg365 = load %IntList* %reg365.idx		; <%IntList> [#uses=1]
	%reg1081 = call %IntList %newIntList( int %reg364, %IntList %reg365 )		; <%IntList> [#uses=1]
	store %IntList %reg1081, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb96:		; preds = %bb72
	%reg368 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg1088 = getelementptr %YYSTYPE* %reg368, long -1		; <%YYSTYPE*> [#uses=1]
	%cast1090 = cast %YYSTYPE* %reg1088 to int*		; <int*> [#uses=1]
	%reg370 = load int* %cast1090		; <int> [#uses=1]
	%reg371.idx = getelementptr %YYSTYPE* %reg368, long 0, ubyte 0		; <%IntList*> [#uses=1]
	%reg371 = load %IntList* %reg371.idx		; <%IntList> [#uses=1]
	%reg1093 = call %IntList %newIntList( int %reg370, %IntList %reg371 )		; <%IntList> [#uses=1]
	store %IntList %reg1093, %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)
	br label %bb97

bb97:		; preds = %bb96, %bb95, %bb94, %bb93, %bb92, %bb91, %bb90, %bb89, %bb88, %bb87, %bb86, %bb85, %bb84, %bb83, %bb82, %bb81, %bb80, %bb79, %bb78, %bb77, %bb76, %bb75, %bb74, %bb72, %bb69, %bb64, %bb61, %bb54, %bb51, %bb46, %bb43
	%cast1097 = cast short %reg254 to ulong		; <ulong> [#uses=3]
	%reg375 = add ulong %cast1097, %cast1097		; <ulong> [#uses=1]
	%reg377 = load short** %yyssp		; <short*> [#uses=1]
	%cast379 = cast short* %reg377 to ulong		; <ulong> [#uses=1]
	%reg381 = sub ulong %cast379, %reg375		; <ulong> [#uses=1]
	%cast1099 = cast ulong %reg381 to short*		; <short*> [#uses=1]
	store short* %cast1099, short** %yyssp
	%reg382 = load short** %yyssp		; <short*> [#uses=3]
	%reg383 = load short* %reg382		; <short> [#uses=1]
	%cast1103 = cast short %reg383 to int		; <int> [#uses=3]
	%reg385 = mul ulong %cast1097, 8		; <ulong> [#uses=1]
	%reg387 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=1]
	%cast389 = cast %YYSTYPE* %reg387 to ulong		; <ulong> [#uses=1]
	%reg391 = sub ulong %cast389, %reg385		; <ulong> [#uses=1]
	%cast1108 = cast ulong %reg391 to %YYSTYPE*		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %cast1108, %YYSTYPE** %yyvsp
	%reg1111 = getelementptr [25 x short]* %yylhs, long 0, long %reg468-idxcast		; <short*> [#uses=1]
	%reg398 = load short* %reg1111		; <short> [#uses=2]
	%cast1116 = cast short %reg398 to int		; <int> [#uses=1]
	%cond1117 = setne int %cast1103, 0		; <bool> [#uses=1]
	br bool %cond1117, label %bb104, label %bb98

bb98:		; preds = %bb97
	%cond1119 = setne int %cast1116, 0		; <bool> [#uses=1]
	br bool %cond1119, label %bb104, label %bb99

bb99:		; preds = %bb98
	%reg1122 = getelementptr short* %reg382, long 1		; <short*> [#uses=2]
	store short* %reg1122, short** %yyssp
	store short 1, short* %reg1122
	%reg403 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg1128 = getelementptr %YYSTYPE* %reg403, long 1		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %reg1128, %YYSTYPE** %yyvsp
	%reg406 = load %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)		; <%IntList> [#uses=1]
	%reg1128.idx1 = getelementptr %YYSTYPE* %reg403, long 1, ubyte 0		; <%IntList*> [#uses=1]
	store %IntList %reg406, %IntList* %reg1128.idx1
	%reg407 = load int* %yychar		; <int> [#uses=1]
	%cond1135 = setge int %reg407, 0		; <bool> [#uses=1]
	br bool %cond1135, label %bb102, label %bb100

bb100:		; preds = %bb99
	%reg1139 = call int %yylex( )		; <int> [#uses=1]
	store int %reg1139, int* %yychar
	%reg409 = load int* %yychar		; <int> [#uses=1]
	%cond1141 = setge int %reg409, 0		; <bool> [#uses=1]
	br bool %cond1141, label %bb102, label %bb101

bb101:		; preds = %bb100
	store int 0, int* %yychar
	br label %bb102

bb102:		; preds = %bb101, %bb100, %bb99
	%reg411 = load int* %yychar		; <int> [#uses=1]
	%cond1146 = setne int %reg411, 0		; <bool> [#uses=1]
	br bool %cond1146, label %bb4, label %UnifiedExitNode

bb104:		; preds = %bb98, %bb97
	%reg398-idxcast = cast short %reg398 to long		; <long> [#uses=2]
	%reg1150 = getelementptr [12 x short]* %yygindex, long 0, long %reg398-idxcast		; <short*> [#uses=1]
	%reg418 = load short* %reg1150		; <short> [#uses=1]
	%cast1155 = cast short %reg418 to int		; <int> [#uses=2]
	%cond1156 = seteq int %cast1155, 0		; <bool> [#uses=1]
	br bool %cond1156, label %bb109, label %bb105

bb105:		; preds = %bb104
	%reg473 = add int %cast1155, %cast1103		; <int> [#uses=3]
	%cond1158 = setlt int %reg473, 0		; <bool> [#uses=1]
	br bool %cond1158, label %bb109, label %bb106

bb106:		; preds = %bb105
	%cond1161 = setgt int %reg473, 262		; <bool> [#uses=1]
	br bool %cond1161, label %bb109, label %bb107

bb107:		; preds = %bb106
	%reg473-idxcast = cast int %reg473 to long		; <long> [#uses=2]
	%reg1166 = getelementptr [263 x short]* %yycheck, long 0, long %reg473-idxcast		; <short*> [#uses=1]
	%reg428 = load short* %reg1166		; <short> [#uses=1]
	%cast1171 = cast short %reg428 to int		; <int> [#uses=1]
	%cond1172 = setne int %cast1171, %cast1103		; <bool> [#uses=1]
	br bool %cond1172, label %bb109, label %bb108

bb108:		; preds = %bb107
	%reg1175 = getelementptr [263 x short]* %yytable, long 0, long %reg473-idxcast		; <short*> [#uses=1]
	%reg435 = load short* %reg1175		; <short> [#uses=1]
	%cast1180 = cast short %reg435 to uint		; <uint> [#uses=1]
	br label %bb110

bb109:		; preds = %bb107, %bb106, %bb105, %bb104
	%reg1183 = getelementptr [12 x short]* %yydgoto, long 0, long %reg398-idxcast		; <short*> [#uses=1]
	%reg442 = load short* %reg1183		; <short> [#uses=1]
	%cast1188 = cast short %reg442 to uint		; <uint> [#uses=1]
	br label %bb110

bb110:		; preds = %bb109, %bb108
	%reg476 = phi uint [ %cast1188, %bb109 ], [ %cast1180, %bb108 ]		; <uint> [#uses=2]
	%cast1189 = cast short* %reg382 to sbyte*		; <sbyte*> [#uses=1]
	%reg444 = load short** %yysslim		; <short*> [#uses=1]
	%cast1190 = cast short* %reg444 to sbyte*		; <sbyte*> [#uses=1]
	%cond1191 = setlt sbyte* %cast1189, %cast1190		; <bool> [#uses=1]
	br bool %cond1191, label %bb112, label %bb111

bb111:		; preds = %bb110
	%reg1193 = call int %yygrowstack( )		; <int> [#uses=1]
	%cond1193 = setne int %reg1193, 0		; <bool> [#uses=1]
	br bool %cond1193, label %bb113, label %bb112

bb112:		; preds = %bb111, %bb110
	%reg446 = load short** %yyssp		; <short*> [#uses=1]
	%reg1196 = getelementptr short* %reg446, long 1		; <short*> [#uses=2]
	store short* %reg1196, short** %yyssp
	%cast1357 = cast uint %reg476 to short		; <short> [#uses=1]
	store short %cast1357, short* %reg1196
	%reg449 = load %YYSTYPE** %yyvsp		; <%YYSTYPE*> [#uses=2]
	%reg1202 = getelementptr %YYSTYPE* %reg449, long 1		; <%YYSTYPE*> [#uses=1]
	store %YYSTYPE* %reg1202, %YYSTYPE** %yyvsp
	%reg452 = load %IntList* getelementptr (%YYSTYPE* %yyval, long 0, ubyte 0)		; <%IntList> [#uses=1]
	%reg1202.idx1 = getelementptr %YYSTYPE* %reg449, long 1, ubyte 0		; <%IntList*> [#uses=1]
	store %IntList %reg452, %IntList* %reg1202.idx1
	br label %bb4

bb113:		; preds = %bb111, %bb30, %bb13, %bb2
	call void %yyerror( sbyte* getelementptr ([20 x sbyte]* %.LC1, long 0, long 0) )
	br label %UnifiedExitNode

UnifiedExitNode:		; preds = %bb113, %bb102, %bb34, %bb32
	%UnifiedRetVal = phi int [ 1, %bb113 ], [ 1, %bb34 ], [ 1, %bb32 ], [ 0, %bb102 ]		; <int> [#uses=1]
	ret int %UnifiedRetVal
}

declare %List %newList(sbyte*, %List)

declare %IntList %newIntList(int, %IntList)

declare void %doStart(sbyte*)

declare void %yyerror(sbyte*)

declare void %doSpec(%List, %List)

declare %Arity %newArity(int, %List)

declare %Binding %newBinding(sbyte*, int)

declare %PatternAST %newPatternAST(sbyte*, %List)

declare %RuleAST %newRuleAST(sbyte*, %PatternAST, int, %IntList)

declare void %yyfinished()

declare int %yylex()

declare void %doGram(%List)

declare int %yygrowstack()
