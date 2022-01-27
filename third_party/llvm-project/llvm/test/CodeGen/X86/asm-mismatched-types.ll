; RUN: llc -o - %s -no-integrated-as | FileCheck %s
target triple = "x86_64--"

; Allow to specify any of the 8/16/32/64 register names interchangeably in
; constraints

; Produced by C-programs like this:
; void foo(int p) { register int reg __asm__("r8") = p;
; __asm__ __volatile__("# REG: %0" : : "r" (reg)); }

; CHECK-LABEL: reg64_as_32:
; CHECK: # REG: %r8d
define void @reg64_as_32(i32 %p) {
  call void asm sideeffect "# REG: $0", "{r8}"(i32 %p)
  ret void
}

; CHECK-LABEL: reg64_as_32_float:
; CHECK: # REG: %r8d
define void @reg64_as_32_float(float %p) {
  call void asm sideeffect "# REG: $0", "{r8}"(float %p)
  ret void
}

; CHECK-LABEL: reg64_as_16:
; CHECK: # REG: %r9w
define void @reg64_as_16(i16 %p) {
  call void asm sideeffect "# REG: $0", "{r9}"(i16 %p)
  ret void
}

; CHECK-LABEL: reg64_as_8:
; CHECK: # REG: %bpl
define void @reg64_as_8(i8 %p) {
  call void asm sideeffect "# REG: $0", "{rbp}"(i8 %p)
  ret void
}

; CHECK-LABEL: reg32_as_16:
; CHECK: # REG: %r15w
define void @reg32_as_16(i16 %p) {
  call void asm sideeffect "# REG: $0", "{r15d}"(i16 %p)
  ret void
}

; CHECK-LABEL: reg32_as_8:
; CHECK: # REG: %r12b
define void @reg32_as_8(i8 %p) {
  call void asm sideeffect "# REG: $0", "{r12d}"(i8 %p)
  ret void
}

; CHECK-LABEL: reg16_as_8:
; CHECK: # REG: %cl
define void @reg16_as_8(i8 %p) {
  call void asm sideeffect "# REG: $0", "{cx}"(i8 %p)
  ret void
}

; CHECK-LABEL: reg32_as_64:
; CHECK: # REG: %rbp
define void @reg32_as_64(i64 %p) {
  call void asm sideeffect "# REG: $0", "{ebp}"(i64 %p)
  ret void
}

; CHECK-LABEL: reg32_as_64_float:
; CHECK: # REG: %rbp
define void @reg32_as_64_float(double %p) {
  call void asm sideeffect "# REG: $0", "{ebp}"(double %p)
  ret void
}

; CHECK-LABEL: reg16_as_64:
; CHECK: # REG: %r13
define void @reg16_as_64(i64 %p) {
  call void asm sideeffect "# REG: $0", "{r13w}"(i64 %p)
  ret void
}

; CHECK-LABEL: reg16_as_64_float:
; CHECK: # REG: %r13
define void @reg16_as_64_float(double %p) {
  call void asm sideeffect "# REG: $0", "{r13w}"(double %p)
  ret void
}

; CHECK-LABEL: reg8_as_64:
; CHECK: # REG: %rax
define void @reg8_as_64(i64 %p) {
  call void asm sideeffect "# REG: $0", "{al}"(i64 %p)
  ret void
}

; CHECK-LABEL: reg8_as_64_float:
; CHECK: # REG: %rax
define void @reg8_as_64_float(double %p) {
  call void asm sideeffect "# REG: $0", "{al}"(double %p)
  ret void
}

; CHECK-LABEL: reg16_as_32:
; CHECK: # REG: %r11d
define void @reg16_as_32(i32 %p) {
  call void asm sideeffect "# REG: $0", "{r11w}"(i32 %p)
  ret void
}

; CHECK-LABEL: reg16_as_32_float:
; CHECK: # REG: %r11d
define void @reg16_as_32_float(float %p) {
  call void asm sideeffect "# REG: $0", "{r11w}"(float %p)
  ret void
}

; CHECK-LABEL: reg8_as_32:
; CHECK: # REG: %r9d
define void @reg8_as_32(i32 %p) {
  call void asm sideeffect "# REG: $0", "{r9b}"(i32 %p)
  ret void
}

; CHECK-LABEL: reg8_as_32_float:
; CHECK: # REG: %r9d
define void @reg8_as_32_float(float %p) {
  call void asm sideeffect "# REG: $0", "{r9b}"(float %p)
  ret void
}

; CHECK-LABEL: reg8_as_16:
; CHECK: # REG: %di
define void @reg8_as_16(i16 %p) {
  call void asm sideeffect "# REG: $0", "{dil}"(i16 %p)
  ret void
}
