# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+simd128,+nontrapping-fptoint,+exception-handling < %s | FileCheck %s
# Check that it converts to .o without errors:
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -mattr=+simd128,+nontrapping-fptoint,+exception-handling < %s | obj2yaml | FileCheck -check-prefixes=BIN,BIN32 %s

# Same again for wasm64

# RUN: llvm-mc -triple=wasm64-unknown-unknown -mattr=+simd128,+nontrapping-fptoint,+exception-handling < %s | FileCheck %s
# Check that it converts to .o without errors:
# RUN: llvm-mc -triple=wasm64-unknown-unknown -filetype=obj -mattr=+simd128,+nontrapping-fptoint,+exception-handling < %s | obj2yaml | FileCheck -check-prefixes=BIN,BIN64 %s

# Minimal test for data sections.

test0:
    .functype   test0 () -> (i32)
    i32.const .L.str
.Ltestlabel:
    end_function

    .section	.rodata..L.str,"",@
.L.str:
    .int8	100
    .size	.L.str, 1


# CHECK:           .text
# CHECK-LABEL: test0:
# CHECK-NEXT:      .functype test0 () -> (i32)
# CHECK-NEXT:      i32.const .L.str
# CHECK-NEXT:  .Ltestlabel:
# CHECK-NEXT:      end_function

# CHECK:	    .section	.rodata..L.str,"",@
# CHECK-NEXT:.L.str:
# CHECK-NEXT:	.int8	100


# BIN:      --- !WASM
# BIN-NEXT: FileHeader:
# BIN-NEXT:   Version:         0x1
# BIN-NEXT: Sections:
# BIN-NEXT:   - Type:            TYPE
# BIN-NEXT:     Signatures:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         ParamTypes:      []
# BIN-NEXT:         ReturnTypes:
# BIN-NEXT:           - I32
# BIN-NEXT:   - Type:            IMPORT
# BIN-NEXT:     Imports:
# BIN-NEXT:       - Module:          env
# BIN-NEXT:         Field:           __linear_memory
# BIN-NEXT:         Kind:            MEMORY
# BIN-NEXT:         Memory:
# BIN64-NEXT:         Flags:           [ IS_64 ]
# BIN-NEXT:           Minimum:         0x1
# BIN-NEXT:   - Type:            FUNCTION
# BIN-NEXT:     FunctionTypes:   [ 0 ]
# BIN-NEXT:   - Type:            DATACOUNT
# BIN-NEXT:     Count:           1
# BIN-NEXT:   - Type:            CODE
# BIN-NEXT:     Relocations:
# BIN-NEXT:       - Type:            R_WASM_MEMORY_ADDR_SLEB
# BIN-NEXT:         Index:           1
# BIN-NEXT:         Offset:          0x4
# BIN-NEXT:     Functions:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         Locals:          []
# BIN-NEXT:         Body:            4180808080000B
# BIN-NEXT:   - Type:            DATA
# BIN-NEXT:     Segments:
# BIN-NEXT:       - SectionOffset:   6
# BIN-NEXT:         InitFlags:       0
# BIN-NEXT:         Offset:
# BIN32-NEXT:         Opcode:          I32_CONST
# BIN64-NEXT:         Opcode:          I64_CONST
# BIN-NEXT:           Value:           0
# BIN-NEXT:         Content:         '64'
# BIN-NEXT:   - Type:            CUSTOM
# BIN-NEXT:     Name:            linking
# BIN-NEXT:     Version:         2
# BIN-NEXT:     SymbolTable:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         Kind:            FUNCTION
# BIN-NEXT:         Name:            test0
# BIN-NEXT:         Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:         Function:        0
# BIN-NEXT:       - Index:           1
# BIN-NEXT:         Kind:            DATA
# BIN-NEXT:         Name:            .L.str
# BIN-NEXT:         Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:         Segment:         0
# BIN-NEXT:         Size:            1
# BIN-NEXT:     SegmentInfo:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         Name:            .rodata..L.str
# BIN-NEXT:         Alignment:       0
# BIN-NEXT:         Flags:           [  ]
# BIN-NEXT: ...
