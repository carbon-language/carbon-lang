; This file contains the output from the following compiled C code:
; typedef struct list {
;   struct list *Next;
;   int Data;
; } list;
;
; // Iterative insert fn
; void InsertIntoListTail(list **L, int Data) {
;   while (*L)
;     L = &(*L)->Next;
;   *L = (list*)malloc(sizeof(list));
;   (*L)->Data = Data;
;   (*L)->Next = 0;
; }
;
; // Recursive list search fn
; list *FindData(list *L, int Data) {
;   if (L == 0) return 0;
;   if (L->Data == Data) return L;
;   return FindData(L->Next, Data);
; }
;
; void DoListStuff() {
;   list *MyList = 0;
;   InsertIntoListTail(&MyList, 100);
;   InsertIntoListTail(&MyList, 12);
;   InsertIntoListTail(&MyList, 42);
;   InsertIntoListTail(&MyList, 1123);
;   InsertIntoListTail(&MyList, 1213);
;
;   if (FindData(MyList, 75)) foundIt();
;   if (FindData(MyList, 42)) foundIt();
;   if (FindData(MyList, 700)) foundIt();
; }

%list = type { %list*, int }

declare sbyte *"malloc"(uint)

;;**********************
implementation
;;**********************

void "InsertIntoListTail"(%list** %L, int %Data)
begin
bb1:
        %reg116 = load %list** %L                               ;;<%list*>
        %cast1004 = cast ulong 0 to %list*                      ;;<%list*>
        %cond1000 = seteq %list* %reg116, %cast1004             ;;<bool>
        br bool %cond1000, label %bb3, label %bb2

bb2:
        %reg117 = phi %list** [ %reg118, %bb2 ], [ %L, %bb1 ]   ;;<%list**>
        %cast1010 = cast %list** %reg117 to %list***            ;;<%list***>
        %reg118 = load %list*** %cast1010                       ;;<%list**>
        %reg109 = load %list** %reg118                          ;;<%list*>
        %cast1005 = cast ulong 0 to %list*                      ;;<%list*>
        %cond1001 = setne %list* %reg109, %cast1005             ;;<bool>
        br bool %cond1001, label %bb2, label %bb3

bb3:
        %reg119 = phi %list** [ %reg118, %bb2 ], [ %L, %bb1 ]   ;;<%list**>
        %cast1006 = cast %list** %reg119 to sbyte**             ;;<sbyte**>
        %reg111 = call sbyte* %malloc(uint 16)                  ;;<sbyte*>
        store sbyte* %reg111, sbyte** %cast1006                 ;;<void>
	%reg111 = cast sbyte* %reg111 to ulong
	%reg1002 = add ulong %reg111, 8
        %reg1002 = cast ulong %reg1002 to sbyte*             ;;<sbyte*>
        %cast1008 = cast sbyte* %reg1002 to int*                ;;<int*>
        store int %Data, int* %cast1008                         ;;<void>
        %cast1003 = cast ulong 0 to ulong*                      ;;<ulong*>
        %cast1009 = cast sbyte* %reg111 to ulong**              ;;<ulong**>
        store ulong* %cast1003, ulong** %cast1009               ;;<void>
        ret void
end

%list* "FindData"(%list* %L, int %Data)
begin
bb1:
        br label %bb2

bb2:
        %reg115 = phi %list* [ %reg116, %bb6 ], [ %L, %bb1 ]    ;;<%list*>
        %cast1014 = cast ulong 0 to %list*                      ;;<%list*>
        %cond1011 = setne %list* %reg115, %cast1014             ;;<bool>
        br bool %cond1011, label %bb4, label %bb3

bb3:
        ret %list* null

bb4:
	%idx = getelementptr %list* %reg115, long 0, ubyte 1                  ;;<int>
        %reg111 = load int* %idx
        %cond1013 = setne int %reg111, %Data                    ;;<bool>
        br bool %cond1013, label %bb6, label %bb5

bb5:
        ret %list* %reg115

bb6:
	%idx2 = getelementptr %list* %reg115, long 0, ubyte 0                  ;;<%list*>
        %reg116 = load %list** %idx2
        br label %bb2
end
