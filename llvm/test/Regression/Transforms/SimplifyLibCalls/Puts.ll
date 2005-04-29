; Test that the StrCatOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*fputc'
%struct._IO_FILE = type { int, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, %struct._IO_marker*, %struct._IO_FILE*, int, int, int, ushort, sbyte, [1 x sbyte], sbyte*, long, sbyte*, sbyte*, int, [52 x sbyte] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, int }
%stdout = external global %struct._IO_FILE*		; <%struct._IO_FILE**> [#uses=1]

implementation   ; Functions:

declare int %fputc(int, %struct._IO_FILE*)

int %main() {
entry:
    %tmp.1 = load %struct._IO_FILE** %stdout		; <%struct._IO_FILE*> [#uses=1]
    %tmp.0 = call int %fputc( int 61, %struct._IO_FILE* %tmp.1 )		; <int> [#uses=0]
    ret int 0
}
