; The expr analysis routines were being too aggressive across cast instructions!

; RUN: llvm-as < %s | opt -raise | llvm-dis | not grep 4294967295

target endian = big
target pointersize = 64
	%struct..istack_struct = type { %struct..istack_struct*, %struct..istk_entry*, uint }
	%struct..istk_entry = type { double, int, int, double, double, sbyte* }

implementation   ; Functions:
bool %Intersection(%struct..istack_struct* %tmp.0, uint %tmp.12) {              ; No predecessors!
        %tmp.8 = getelementptr %struct..istack_struct* %tmp.0, long 0, ubyte 1          ; <%struct..istk_entry**> [#uses=1]
        %tmp.9 = load %struct..istk_entry** %tmp.8              ; <%struct..istk_entry*> [#uses=1]
        %dec = sub uint %tmp.12, 1              ; <uint> [#uses=1]
        %tmp.13 = cast uint %dec to ulong               ; <ulong> [#uses=1]
        %tmp.14 = mul ulong %tmp.13, 40         ; <ulong> [#uses=1]
        %tmp.16 = cast %struct..istk_entry* %tmp.9 to long              ; <long> [#uses=1]
        %tmp.17 = cast ulong %tmp.14 to long            ; <long> [#uses=1]
        %tmp.18 = add long %tmp.16, %tmp.17             ; <long> [#uses=1]
        %tmp.19 = cast long %tmp.18 to %struct..istk_entry*             ; <%struct..istk_entry*> [#uses=1]
        %tmp.21 = setne %struct..istk_entry* %tmp.19, null              ; <bool> [#uses=1]
        ret bool %tmp.21
}

