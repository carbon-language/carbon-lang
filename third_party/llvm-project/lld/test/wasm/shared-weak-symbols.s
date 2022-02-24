# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --experimental-pic -shared -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d %t.wasm | FileCheck %s -check-prefix=ASM

# Verify the weakly defined fuctions (weak_func) are both
# imported and exported, and that internal usage (direct call)
# always uses the imported version.


.globl weak_func
.weak weak_func
weak_func:
  .functype weak_func () -> (i32)
  i32.const 0
  end_function

.globl call_weak
call_weak:
# ASM: <call_weak>:
  .functype call_weak () -> (i32)
  call weak_func
# ASM:           10 80 80 80 80 00      call  0
  end_function
# ASM-NEXT:      0b                     end

# CHECK:       - Type:            IMPORT
# CHECK-NEXT:    Imports:
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           memory
# CHECK-NEXT:        Kind:            MEMORY
# CHECK-NEXT:        Memory:
# CHECK-NEXT:          Minimum:         0x0
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           __memory_base
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        GlobalType:      I32
# CHECK-NEXT:        GlobalMutable:   false
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           __table_base
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        GlobalType:      I32
# CHECK-NEXT:        GlobalMutable:   false
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           weak_func
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        SigIndex:        0

#      CHECK:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            name
# CHECK-NEXT:     FunctionNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            weak_func
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            __wasm_call_ctors
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Name:            __wasm_apply_data_relocs
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Name:            weak_func
