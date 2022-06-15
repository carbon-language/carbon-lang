@a = hidden global [6 x i8] c"hello\00", align 1
@b = hidden global [8 x i8] c"goodbye\00", align 1
@c = hidden global [9 x i8] c"whatever\00", align 1
@d = hidden global i32 42, align 4

@e = private constant [9 x i8] c"constant\00", align 1
@f = private constant i8 43, align 4

; RUN: llc --mtriple=wasm32-unknown-unknown -mattr=+bulk-memory,+atomics -filetype=obj %s -o %t.passive.o
; RUN: llc --mtriple=wasm64-unknown-unknown -mattr=+bulk-memory,+atomics -filetype=obj %s -o %t.passive64.o
; RUN: llc --mtriple=wasm32-unknown-unknown -filetype=obj %s -o %t.o

; RUN: wasm-ld -no-gc-sections --no-entry -o %t.merged.wasm %t.o
; RUN: obj2yaml %t.merged.wasm | FileCheck %s --check-prefix=MERGE

; MERGE-LABEL: - Type:            DATA
; MERGE-NEXT:    Segments:
; MERGE-NEXT:      - SectionOffset:   7
; MERGE-NEXT:        InitFlags:       0
; MERGE-NEXT:        Offset:
; MERGE:             Content:         636F6E7374616E74000000002B
; MERGE-NEXT:      - SectionOffset:   26
; MERGE-NEXT:        InitFlags:       0
; MERGE-NEXT:        Offset:
; MERGE:             Content:         68656C6C6F00676F6F6462796500776861746576657200002A000000
; MERGE-NEXT:  - Type:            CUSTOM
; MERGE-NEXT:    Name:            name
; MERGE-NEXT:    FunctionNames:
; MERGE-NEXT:      - Index:           0
; MERGE-NEXT:        Name:            __wasm_call_ctors
; MERGE-NEXT:    GlobalNames:
; MERGE-NEXT:      - Index:           0
; MERGE-NEXT:        Name:            __stack_pointer
; MERGE-NEXT:    DataSegmentNames:
; MERGE-NEXT:      - Index:           0
; MERGE-NEXT:        Name:            .rodata

; RUN: wasm-ld -no-gc-sections --no-entry --no-merge-data-segments -o %t.separate.wasm %t.o
; RUN: obj2yaml %t.separate.wasm | FileCheck %s --check-prefix=SEPARATE

; SEPARATE-NOT:                  DATACOUNT
; SEPARATE-LABEL: - Type:            DATA
; SEPARATE-NEXT:    Segments:
; SEPARATE-NEXT:      - SectionOffset:   7
; SEPARATE-NEXT:        InitFlags:       0
; SEPARATE-NEXT:        Offset:
; SEPARATE:             Content:         636F6E7374616E7400
; SEPARATE-NEXT:      - SectionOffset:   22
; SEPARATE-NEXT:        InitFlags:       0
; SEPARATE-NEXT:        Offset:
; SEPARATE:             Content:         2B
; SEPARATE-NEXT:      - SectionOffset:   29
; SEPARATE-NEXT:        InitFlags:       0
; SEPARATE-NEXT:        Offset:
; SEPARATE:             Content:         68656C6C6F00
; SEPARATE-NEXT:      - SectionOffset:   41
; SEPARATE-NEXT:        InitFlags:       0
; SEPARATE-NEXT:        Offset:
; SEPARATE:             Content:         676F6F6462796500
; SEPARATE-NEXT:      - SectionOffset:   55
; SEPARATE-NEXT:        InitFlags:       0
; SEPARATE-NEXT:        Offset:
; SEPARATE:             Content:         '776861746576657200'
; SEPARATE-NEXT:      - SectionOffset:   70
; SEPARATE-NEXT:        InitFlags:       0
; SEPARATE-NEXT:        Offset:
; SEPARATE:             Content:         2A000000
; SEPARATE-NEXT:  - Type:            CUSTOM
; SEPARATE-NEXT:    Name:            name
; SEPARATE-NEXT:    FunctionNames:
; SEPARATE-NEXT:      - Index:           0
; SEPARATE-NEXT:        Name:            __wasm_call_ctors
; SEPARATE-NEXT:    GlobalNames:
; SEPARATE-NEXT:      - Index:           0
; SEPARATE-NEXT:        Name:            __stack_pointer
; SEPARATE-NEXT:    DataSegmentNames:
; SEPARATE-NEXT:      - Index:           0
; SEPARATE-NEXT:        Name:            .rodata

; RUN: wasm-ld -no-gc-sections --no-entry --shared-memory --max-memory=131072 -o %t.merged.passive.wasm %t.passive.o
; RUN: obj2yaml %t.merged.passive.wasm | FileCheck %s --check-prefix=PASSIVE-MERGE
; RUN: wasm-ld -mwasm64 -no-gc-sections --no-entry --shared-memory --max-memory=131072 -o %t.merged.passive64.wasm %t.passive64.o
; RUN: obj2yaml %t.merged.passive64.wasm | FileCheck %s --check-prefix=PASSIVE-MERGE

; PASSIVE-MERGE-LABEL: - Type:            DATACOUNT
; PASSIVE-MERGE-NEXT:    Count:           2
; PASSIVE-MERGE-LABEL: - Type:            DATA
; PASSIVE-MERGE-NEXT:    Segments:
; PASSIVE-MERGE-NEXT:      - SectionOffset:   3
; PASSIVE-MERGE-NEXT:        InitFlags:       1
; PASSIVE-MERGE-NEXT:        Content:         636F6E7374616E74000000002B
; PASSIVE-MERGE-NEXT:      - SectionOffset:   18
; PASSIVE-MERGE-NEXT:        InitFlags:       1
; PASSIVE-MERGE-NEXT:        Content:         68656C6C6F00676F6F6462796500776861746576657200002A000000
; PASSIVE-MERGE-NEXT:  - Type:            CUSTOM
; PASSIVE-MERGE-NEXT:    Name:            name
; PASSIVE-MERGE-NEXT:    FunctionNames:
; PASSIVE-MERGE-NEXT:      - Index:           0
; PASSIVE-MERGE-NEXT:        Name:            __wasm_call_ctors
; PASSIVE-MERGE-NEXT:      - Index:           1
; PASSIVE-MERGE-NEXT:        Name:            __wasm_init_tls
; PASSIVE-MERGE-NEXT:      - Index:           2
; PASSIVE-MERGE-NEXT:        Name:            __wasm_init_memory

; RUN: wasm-ld -no-gc-sections --no-entry --shared-memory --max-memory=131072 --no-merge-data-segments -o %t.separate.passive.wasm %t.passive.o
; RUN: obj2yaml %t.separate.passive.wasm | FileCheck %s --check-prefix=PASSIVE-SEPARATE
; RUN: wasm-ld -mwasm64 -no-gc-sections --no-entry --shared-memory --max-memory=131072 --no-merge-data-segments -o %t.separate.passive64.wasm %t.passive64.o
; RUN: obj2yaml %t.separate.passive64.wasm | FileCheck %s --check-prefix=PASSIVE-SEPARATE

; PASSIVE-SEPARATE-LABEL: - Type:            DATACOUNT
; PASSIVE-SEPARATE-NEXT:    Count:           6
; PASSIVE-SEPARATE-LABEL: - Type:            DATA
; PASSIVE-SEPARATE-NEXT:    Segments:
; PASSIVE-SEPARATE-NEXT:      - SectionOffset:   3
; PASSIVE-SEPARATE-NEXT:        InitFlags:       1
; PASSIVE-SEPARATE-NEXT:        Content:         636F6E7374616E7400
; PASSIVE-SEPARATE-NEXT:      - SectionOffset:   14
; PASSIVE-SEPARATE-NEXT:        InitFlags:       1
; PASSIVE-SEPARATE-NEXT:        Content:         2B
; PASSIVE-SEPARATE-NEXT:      - SectionOffset:   17
; PASSIVE-SEPARATE-NEXT:        InitFlags:       1
; PASSIVE-SEPARATE-NEXT:        Content:         68656C6C6F00
; PASSIVE-SEPARATE-NEXT:      - SectionOffset:   25
; PASSIVE-SEPARATE-NEXT:        InitFlags:       1
; PASSIVE-SEPARATE-NEXT:        Content:         676F6F6462796500
; PASSIVE-SEPARATE-NEXT:      - SectionOffset:   35
; PASSIVE-SEPARATE-NEXT:        InitFlags:       1
; PASSIVE-SEPARATE-NEXT:        Content:         '776861746576657200'
; PASSIVE-SEPARATE-NEXT:      - SectionOffset:   46
; PASSIVE-SEPARATE-NEXT:        InitFlags:       1
; PASSIVE-SEPARATE-NEXT:        Content:         2A000000
; PASSIVE-SEPARATE-NEXT:    - Type:            CUSTOM
; PASSIVE-SEPARATE-NEXT:      Name:            name
; PASSIVE-SEPARATE-NEXT:      FunctionNames:
; PASSIVE-SEPARATE-NEXT:        - Index:           0
; PASSIVE-SEPARATE-NEXT:          Name:            __wasm_call_ctors
; PASSIVE-SEPARATE-NEXT:        - Index:           1
; PASSIVE-SEPARATE-NEXT:          Name:            __wasm_init_tls
; PASSIVE-SEPARATE-NEXT:        - Index:           2
; PASSIVE-SEPARATE-NEXT:          Name:            __wasm_init_memory
