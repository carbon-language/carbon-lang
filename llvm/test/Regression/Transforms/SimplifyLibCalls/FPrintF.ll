; Test that the FPrintFOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*fprintf'
;

%struct._IO_FILE = type { int, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, %struct._IO_marker*, %struct._IO_FILE*, int, int, int, ushort, sbyte, [1 x sbyte], sbyte*, long, sbyte*, sbyte*, int, [52 x sbyte] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, int }

%str = constant [3 x sbyte] c"%s\00"		
%chr = constant [3 x sbyte] c"%c\00"		
%hello = constant [13 x sbyte] c"hello world\0A\00"
%stdout = external global %struct._IO_FILE*		

declare int %fprintf(%struct._IO_FILE*, sbyte*, ...)

implementation  

int %foo() 
{
entry:
	%tmp.1 = load %struct._IO_FILE** %stdout
	%tmp.0 = call int (%struct._IO_FILE*, sbyte*, ...)* %fprintf( %struct._IO_FILE* %tmp.1, sbyte* getelementptr ([13 x sbyte]* %hello, int 0, int 0) )
	%tmp.4 = load %struct._IO_FILE** %stdout
	%tmp.3 = call int (%struct._IO_FILE*, sbyte*, ...)* %fprintf( %struct._IO_FILE* %tmp.4, sbyte* getelementptr ([3 x sbyte]* %str, int 0, int 0), sbyte* getelementptr ([13 x sbyte]* %hello, int 0, int 0) )
	%tmp.8 = load %struct._IO_FILE** %stdout
	%tmp.7 = call int (%struct._IO_FILE*, sbyte*, ...)* %fprintf( %struct._IO_FILE* %tmp.8, sbyte* getelementptr ([3 x sbyte]* %chr, int 0, int 0), int 33 )
	ret int 0
}

