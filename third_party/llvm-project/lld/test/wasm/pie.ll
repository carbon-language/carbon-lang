; RUN: llc -relocation-model=pic -mattr=+mutable-globals -filetype=obj %s -o %t.o
; RUN: wasm-ld --no-gc-sections --experimental-pic -pie -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s
; RUN: llvm-objdump --disassemble-symbols=__wasm_call_ctors --no-show-raw-insn --no-leading-addr %t.wasm | FileCheck %s --check-prefixes DISASSEM

target triple = "wasm32-unknown-emscripten"

@data = global i32 2, align 4
@data_external = external global i32
@indirect_func = local_unnamed_addr global i32 ()* @foo, align 4

@data_addr = local_unnamed_addr global i32* @data, align 4
@data_addr_external = local_unnamed_addr global i32* @data_external, align 4

define hidden i32 @foo() {
entry:
  ; To ensure we use __stack_pointer
  %ptr = alloca i32
  %0 = load i32, i32* @data, align 4
  %1 = load i32 ()*, i32 ()** @indirect_func, align 4
  call i32 %1()
  ret i32 %0
}

define default i32** @get_data_address() {
entry:
  ret i32** @data_addr_external
}

define void @_start() {
  call void @external_func()
  ret void
}

declare void @external_func()

; CHECK:      Sections:
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            dylink.0
; CHECK-NEXT:     MemorySize:      16
; CHECK-NEXT:     MemoryAlignment: 2
; CHECK-NEXT:     TableSize:       1
; CHECK-NEXT:     TableAlignment:  0
; CHECK-NEXT:     Needed:          []

; CHECK:        - Type:            IMPORT
; CHECK-NEXT:     Imports:
; CHECK-NEXT:      - Module:          env
; CHECK-NEXT:        Field:           __indirect_function_table
; CHECK-NEXT:        Kind:            TABLE
; CHECK-NEXT:        Table:
; CHECK-NEXT:          Index:           0
; CHECK-NEXT:          ElemType:        FUNCREF
; CHECK-NEXT:          Limits:
; CHECK-NEXT:            Minimum:         0x1
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __stack_pointer
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         GlobalType:      I32
; CHECK-NEXT:         GlobalMutable:   true
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __memory_base
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         GlobalType:      I32
; CHECK-NEXT:         GlobalMutable:   false
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __table_base
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         GlobalType:      I32
; CHECK-NEXT:         GlobalMutable:   false

; CHECK:        - Type:            START
; CHECK-NEXT:     StartFunction:   3

; CHECK:        - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            external_func
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            __wasm_call_ctors
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Name:            __wasm_apply_data_relocs
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Name:            __wasm_apply_global_relocs
; CHECK-NEXT:       - Index:           4
; CHECK-NEXT:         Name:            foo
; CHECK-NEXT:       - Index:           5
; CHECK-NEXT:         Name:            get_data_address
; CHECK-NEXT:       - Index:           6
; CHECK-NEXT:         Name:            _start
; CHECK-NEXT:     GlobalNames:

; DISASSEM:       <__wasm_call_ctors>:
; DISASSEM-EMPTY:
; DISASSEM-NEXT:   call 2
; DISASSEM-NEXT:   end

; Run the same test with extended-const support.  When this is available
; we don't need __wasm_apply_global_relocs and instead rely on the add
; instruction in the InitExpr.  We also, therefore, do not need these globals
; to be mutable.

; RUN: llc -relocation-model=pic -mattr=+extended-const,+mutable-globals,+atomics,+bulk-memory -filetype=obj %s -o %t.extended.o
; RUN: wasm-ld --no-gc-sections --allow-undefined --experimental-pic -pie -o %t.extended.wasm %t.extended.o
; RUN: obj2yaml %t.extended.wasm | FileCheck %s --check-prefix=EXTENDED-CONST

; EXTENDED-CONST-NOT: __wasm_apply_global_relocs

; EXTENDED-CONST:       - Type:            GLOBAL
; EXTENDED-CONST-NEXT:    Globals:
; EXTENDED-CONST-NEXT:      - Index:           4
; EXTENDED-CONST-NEXT:        Type:            I32
; EXTENDED-CONST-NEXT:        Mutable:         false
; EXTENDED-CONST-NEXT:        InitExpr:
; EXTENDED-CONST-NEXT:          Opcode:        GLOBAL_GET
; EXTENDED-CONST-NEXT:          Index:         1
; EXTENDED-CONST-NEXT:      - Index:           5
; EXTENDED-CONST-NEXT:        Type:            I32
; EXTENDED-CONST-NEXT:        Mutable:         false
; EXTENDED-CONST-NEXT:        InitExpr:
; EXTENDED-CONST-NEXT:          Extended:        true
; EXTENDED-CONST-NEXT:          Body:            230141046A0B
; EXTENDED-CONST-NEXT:      - Index:           6
; EXTENDED-CONST-NEXT:        Type:            I32
; EXTENDED-CONST-NEXT:        Mutable:         false
; EXTENDED-CONST-NEXT:        InitExpr:
; EXTENDED-CONST-NEXT:          Extended:        true
; This instruction sequence decodes to:
; (global.get[0x23] 0x1 i32.const[0x41] 0x0C i32.add[0x6A] end[0x0b])
; EXTENDED-CONST-NEXT:          Body:            2301410C6A0B

;  EXTENDED-CONST-NOT:  - Type:            START

;      EXTENDED-CONST:    FunctionNames:
; EXTENDED-CONST-NEXT:      - Index:           0
; EXTENDED-CONST-NEXT:        Name:            external_func
; EXTENDED-CONST-NEXT:      - Index:           1
; EXTENDED-CONST-NEXT:        Name:            __wasm_call_ctors
; EXTENDED-CONST-NEXT:      - Index:           2
; EXTENDED-CONST-NEXT:        Name:            __wasm_apply_data_relocs

; Run the same test with threading support.  In this mode
; we expect __wasm_init_memory and __wasm_apply_data_relocs
; to be generated along with __wasm_start as the start
; function.

; RUN: llc -relocation-model=pic -mattr=+mutable-globals,+atomics,+bulk-memory -filetype=obj %s -o %t.shmem.o
; RUN: wasm-ld --no-gc-sections --shared-memory --allow-undefined --experimental-pic -pie -o %t.shmem.wasm %t.shmem.o
; RUN: obj2yaml %t.shmem.wasm | FileCheck %s --check-prefix=SHMEM
; RUN: llvm-objdump --disassemble-symbols=__wasm_start --no-show-raw-insn --no-leading-addr %t.shmem.wasm | FileCheck %s --check-prefix DISASSEM-SHMEM

; SHMEM:         - Type:            START
; SHMEM-NEXT:      StartFunction:   6

; DISASSEM-SHMEM:       <__wasm_start>:
; DISASSEM-SHMEM-EMPTY:
; DISASSEM-SHMEM-NEXT:   call 5
; DISASSEM-SHMEM-NEXT:   call 3
; DISASSEM-SHMEM-NEXT:   end

; SHMEM:         FunctionNames:
; SHMEM-NEXT:      - Index:           0
; SHMEM-NEXT:        Name:            external_func
; SHMEM-NEXT:      - Index:           1
; SHMEM-NEXT:        Name:            __wasm_call_ctors
; SHMEM-NEXT:      - Index:           2
; SHMEM-NEXT:        Name:            __wasm_init_tls
; SHMEM-NEXT:      - Index:           3
; SHMEM-NEXT:        Name:            __wasm_init_memory
; SHMEM-NEXT:      - Index:           4
; SHMEM-NEXT:        Name:            __wasm_apply_data_relocs
; SHMEM-NEXT:      - Index:           5
; SHMEM-NEXT:        Name:            __wasm_apply_global_relocs
; SHMEM-NEXT:      - Index:           6
; SHMEM-NEXT:        Name:            __wasm_start
; SHMEM-NEXT:      - Index:           7
; SHMEM-NEXT:        Name:            foo
; SHMEM-NEXT:      - Index:           8
; SHMEM-NEXT:        Name:            get_data_address
; SHMEM-NEXT:      - Index:           9
; SHMEM-NEXT:        Name:            _start
