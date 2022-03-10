; Test that internal symbols can still be GC'd when with --export-dynamic.
; RUN: llc -filetype=obj %s -o %t.o
; RUN: wasm-ld --export-dynamic -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s

target triple = "wasm32-unknown-unknown"

@foo = default global i32 1, align 4
@bar = internal default global i32 3, align 4

define internal i32 @baz() local_unnamed_addr {
entry:
  %0 = load i32, i32* @bar, align 4
  ret i32 %0
}

define void @_start() local_unnamed_addr {
entry:
  call i32 @baz()
  ret void
}

; CHECK:      --- !WASM
; CHECK-NEXT: FileHeader:
; CHECK-NEXT:   Version:         0x1
; CHECK-NEXT: Sections:
; CHECK-NEXT:   - Type:            TYPE
; CHECK-NEXT:     Signatures:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:         ReturnTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:         ReturnTypes:     []
; CHECK-NEXT:   - Type:            FUNCTION
; CHECK-NEXT:     FunctionTypes:   [ 0, 1 ]
; CHECK-NEXT:   - Type:            MEMORY
; CHECK-NEXT:     Memories:
; CHECK-NEXT:       - Minimum:         0x2
; CHECK-NEXT:   - Type:            GLOBAL
; CHECK-NEXT:     Globals:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         true
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           66576
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1024
; CHECK-NEXT:   - Type:            EXPORT
; CHECK-NEXT:     Exports:
; CHECK-NEXT:       - Name:            memory
; CHECK-NEXT:         Kind:            MEMORY
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:       - Name:            _start
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:       - Name:            foo
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:   - Type:            CODE
; CHECK-NEXT:     Functions:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Locals:
; CHECK-NEXT:         Body:            4100280284888080000B
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Locals:
; CHECK-NEXT:         Body:            1080808080001A0B
; CHECK-NEXT:   - Type:            DATA
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - SectionOffset:   7
; CHECK-NEXT:         InitFlags:       0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1024
; CHECK-NEXT:         Content:         '0100000003000000'
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            baz
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            _start
; CHECK-NEXT:     GlobalNames:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            __stack_pointer
; CHECK-NEXT:     DataSegmentNames:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            .data
; CHECK-NEXT: ...
