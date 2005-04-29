; Test that the PutsCatOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*fputs'
;
%struct._IO_FILE = type { int, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, %struct._IO_marker*, %struct._IO_FILE*, int, int, int, ushort, sbyte, [1 x sbyte], sbyte*, long, sbyte*, sbyte*, int, [52 x sbyte] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, int }
%stdout = external global %struct._IO_FILE*		; <%struct._IO_FILE**> [#uses=1]

declare int %fputs(sbyte*, %struct._IO_FILE*)

%empty = constant [1 x sbyte] c"\00"
%len1  = constant [2 x sbyte] c"A\00"
%long  = constant [7 x sbyte] c"hello\0A\00"

implementation   ; Functions:

int %main() {
entry:
  %out = load %struct._IO_FILE** %stdout
  %s1 = getelementptr [1 x sbyte]* %empty, int 0, int 0
  %s2 = getelementptr [2 x sbyte]* %len1, int 0, int 0
  %s3 = getelementptr [7 x sbyte]* %long, int 0, int 0
  %a = call int %fputs( sbyte* %s1, %struct._IO_FILE* %out )
  %b = call int %fputs( sbyte* %s2, %struct._IO_FILE* %out )
  %c = call int %fputs( sbyte* %s3, %struct._IO_FILE* %out )
  ret int 0
}
