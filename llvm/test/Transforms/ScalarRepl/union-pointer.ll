; PR892
; RUN: llvm-upgrade < %s | llvm-as | opt -scalarrepl | llvm-dis | \
; RUN:   not grep alloca
; RUN: llvm-upgrade < %s | llvm-as | opt -scalarrepl | llvm-dis | grep {ret i8}

target endian = little
target pointersize = 32
target triple = "i686-apple-darwin8.7.2"
        
%struct.Val = type { int*, int }

implementation   ; Functions:

sbyte* %test(short* %X) {
        %X_addr = alloca short*
        store short* %X, short** %X_addr
        %X_addr = cast short** %X_addr to sbyte**
        %tmp = load sbyte** %X_addr
        ret sbyte* %tmp
}

void %test2(long %Op.0) {
        %tmp = alloca %struct.Val, align 8              
        %tmp1 = alloca %struct.Val, align 8             
        %tmp = call ulong %_Z3foov( )           
        %tmp1 = cast %struct.Val* %tmp1 to ulong*               
        store ulong %tmp, ulong* %tmp1
        %tmp = getelementptr %struct.Val* %tmp, int 0, uint 0           
        %tmp2 = getelementptr %struct.Val* %tmp1, int 0, uint 0         
        %tmp = load int** %tmp2         
        store int* %tmp, int** %tmp
        %tmp3 = getelementptr %struct.Val* %tmp, int 0, uint 1          
        %tmp4 = getelementptr %struct.Val* %tmp1, int 0, uint 1         
        %tmp = load int* %tmp4          
        store int %tmp, int* %tmp3
        %tmp7 = cast %struct.Val* %tmp to { long }*             
        %tmp8 = getelementptr { long }* %tmp7, int 0, uint 0            
        %tmp9 = load long* %tmp8                
        call void %_Z3bar3ValS_( long %Op.0, long %tmp9 )
        ret void
}

declare ulong %_Z3foov()

declare void %_Z3bar3ValS_(long, long)

