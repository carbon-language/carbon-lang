; RUN: llc -march=amdgcn -mcpu=gfx900 < %s | FileCheck --check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 < %s | FileCheck --check-prefixes=GCN,GFX10,GFX10W32,GFX1010,GFX1010W32 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize64 < %s | FileCheck --check-prefixes=GCN,GFX10,GFX10W64,GFX1010,GFX1010W64 %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 < %s | FileCheck --check-prefixes=GCN,GFX10,GFX10W32,GFX1030,GFX1030W32 %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 -mattr=+wavefrontsize64 < %s | FileCheck --check-prefixes=GCN,GFX10,GFX10W64,GFX1030,GFX1030W64 %s

; GCN-LABEL: {{^}}max_occupancy:
; GFX9:       ; Occupancy: 10
; GFX1010:    ; Occupancy: 20
; GFX1030:    ; Occupancy: 16
define amdgpu_kernel void @max_occupancy() {
  ret void
}

; GCN-LABEL: {{^}}limited_occupancy_3:
; GFX9:       ; Occupancy: 3
; GFX10W64:   ; Occupancy: 3
; GFX10W32:   ; Occupancy: 4
define amdgpu_kernel void @limited_occupancy_3() #0 {
  ret void
}

; GCN-LABEL: {{^}}limited_occupancy_18:
; GFX9:       ; Occupancy: 10
; GFX1010:    ; Occupancy: 18
; GFX1030:    ; Occupancy: 16
define amdgpu_kernel void @limited_occupancy_18() #1 {
  ret void
}

; GCN-LABEL: {{^}}limited_occupancy_19:
; GFX9:       ; Occupancy: 10
; GFX1010:    ; Occupancy: 18
; GFX1030:    ; Occupancy: 16
define amdgpu_kernel void @limited_occupancy_19() #2 {
  ret void
}

; GCN-LABEL: {{^}}used_24_vgprs:
; GFX9:       ; Occupancy: 10
; GFX1010:    ; Occupancy: 20
; GFX1030:    ; Occupancy: 16
define amdgpu_kernel void @used_24_vgprs() {
  call void asm sideeffect "", "~{v23}" ()
  ret void
}

; GCN-LABEL: {{^}}used_28_vgprs:
; GFX9:       ; Occupancy: 9
; GFX1010W64: ; Occupancy: 18
; GFX1010W32: ; Occupancy: 20
; GFX1030:    ; Occupancy: 16
define amdgpu_kernel void @used_28_vgprs() {
  call void asm sideeffect "", "~{v27}" ()
  ret void
}

; GCN-LABEL: {{^}}used_32_vgprs:
; GFX9:       ; Occupancy: 8
; GFX10W64:   ; Occupancy: 16
; GFX1010W32: ; Occupancy: 20
; GFX1030W32: ; Occupancy: 16
define amdgpu_kernel void @used_32_vgprs() {
  call void asm sideeffect "", "~{v31}" ()
  ret void
}

; GCN-LABEL: {{^}}used_36_vgprs:
; GFX9:       ; Occupancy: 7
; GFX1010W64: ; Occupancy: 14
; GFX1010W32: ; Occupancy: 20
; GFX1030W64: ; Occupancy: 12
; GFX1030W32: ; Occupancy: 16
define amdgpu_kernel void @used_36_vgprs() {
  call void asm sideeffect "", "~{v35}" ()
  ret void
}

; GCN-LABEL: {{^}}used_40_vgprs:
; GFX9:       ; Occupancy: 6
; GFX10W64:   ; Occupancy: 12
; GFX1010W32: ; Occupancy: 20
; GFX1030W32: ; Occupancy: 16
define amdgpu_kernel void @used_40_vgprs() {
  call void asm sideeffect "", "~{v39}" ()
  ret void
}

; GCN-LABEL: {{^}}used_44_vgprs:
; GFX9:       ; Occupancy: 5
; GFX1010W64: ; Occupancy: 11
; GFX1010W32: ; Occupancy: 20
; GFX1030W64: ; Occupancy: 10
; GFX1030W32: ; Occupancy: 16
define amdgpu_kernel void @used_44_vgprs() {
  call void asm sideeffect "", "~{v43}" ()
  ret void
}

; GCN-LABEL: {{^}}used_48_vgprs:
; GFX9:       ; Occupancy: 5
; GFX10W64:   ; Occupancy: 10
; GFX1010W32: ; Occupancy: 20
; GFX1030W32: ; Occupancy: 16
define amdgpu_kernel void @used_48_vgprs() {
  call void asm sideeffect "", "~{v47}" ()
  ret void
}

; GCN-LABEL: {{^}}used_56_vgprs:
; GFX9:       ; Occupancy: 4
; GFX10W64:   ; Occupancy: 9
; GFX1010W32: ; Occupancy: 18
; GFX1030W32: ; Occupancy: 16
define amdgpu_kernel void @used_56_vgprs() {
  call void asm sideeffect "", "~{v55}" ()
  ret void
}

; GCN-LABEL: {{^}}used_64_vgprs:
; GFX9:       ; Occupancy: 4
; GFX10W64:   ; Occupancy: 8
; GFX10W32:   ; Occupancy: 16
define amdgpu_kernel void @used_64_vgprs() {
  call void asm sideeffect "", "~{v63}" ()
  ret void
}

; GCN-LABEL: {{^}}used_72_vgprs:
; GFX9:       ; Occupancy: 3
; GFX10W64:   ; Occupancy: 7
; GFX1010W32: ; Occupancy: 14
; GFX1030W32: ; Occupancy: 12
define amdgpu_kernel void @used_72_vgprs() {
  call void asm sideeffect "", "~{v71}" ()
  ret void
}

; GCN-LABEL: {{^}}used_80_vgprs:
; GFX9:       ; Occupancy: 3
; GFX10W64:   ; Occupancy: 6
; GFX10W32:   ; Occupancy: 12
define amdgpu_kernel void @used_80_vgprs() {
  call void asm sideeffect "", "~{v79}" ()
  ret void
}

; GCN-LABEL: {{^}}used_84_vgprs:
; GFX9:       ; Occupancy: 3
; GFX1010W64: ; Occupancy: 6
; GFX1010W32: ; Occupancy: 11
; GFX1030W64: ; Occupancy: 5
; GFX1030W32: ; Occupancy: 10
define amdgpu_kernel void @used_84_vgprs() {
  call void asm sideeffect "", "~{v83}" ()
  ret void
}

; GCN-LABEL: {{^}}used_88_vgprs:
; GFX9:       ; Occupancy: 2
; GFX10W64:   ; Occupancy: 5
; GFX1010W32: ; Occupancy: 11
; GFX1030W32: ; Occupancy: 10
define amdgpu_kernel void @used_88_vgprs() {
  call void asm sideeffect "", "~{v87}" ()
  ret void
}

; GCN-LABEL: {{^}}used_96_vgprs:
; GFX9:       ; Occupancy: 2
; GFX10W64:   ; Occupancy: 5
; GFX10W32:   ; Occupancy: 10
define amdgpu_kernel void @used_96_vgprs() {
  call void asm sideeffect "", "~{v95}" ()
  ret void
}

; GCN-LABEL: {{^}}used_100_vgprs:
; GFX9:       ; Occupancy: 2
; GFX1010W64: ; Occupancy: 5
; GFX1030W64: ; Occupancy: 4
; GFX10W32:   ; Occupancy: 9
define amdgpu_kernel void @used_100_vgprs() {
  call void asm sideeffect "", "~{v99}" ()
  ret void
}

; GCN-LABEL: {{^}}used_112_vgprs:
; GFX9:       ; Occupancy: 2
; GFX10W64:   ; Occupancy: 4
; GFX10W32:   ; Occupancy: 9
define amdgpu_kernel void @used_112_vgprs() {
  call void asm sideeffect "", "~{v111}" ()
  ret void
}

; GCN-LABEL: {{^}}used_128_vgprs:
; GFX9:       ; Occupancy: 2
; GFX10W64:   ; Occupancy: 4
; GFX10W32:   ; Occupancy: 8
define amdgpu_kernel void @used_128_vgprs() {
  call void asm sideeffect "", "~{v127}" ()
  ret void
}

; GCN-LABEL: {{^}}used_144_vgprs:
; GFX9:       ; Occupancy: 1
; GFX10W64:   ; Occupancy: 3
; GFX10W32:   ; Occupancy: 7
define amdgpu_kernel void @used_144_vgprs() {
  call void asm sideeffect "", "~{v143}" ()
  ret void
}

; GCN-LABEL: {{^}}used_168_vgprs:
; GFX9:       ; Occupancy: 1
; GFX10W64:   ; Occupancy: 3
; GFX1010W32: ; Occupancy: 6
; GFX1030W32: ; Occupancy: 5
define amdgpu_kernel void @used_168_vgprs() {
  call void asm sideeffect "", "~{v167}" ()
  ret void
}

; GCN-LABEL: {{^}}used_200_vgprs:
; GFX9:       ; Occupancy: 1
; GFX10W64:   ; Occupancy: 2
; GFX1010W32: ; Occupancy: 5
; GFX1030W32: ; Occupancy: 4
define amdgpu_kernel void @used_200_vgprs() {
  call void asm sideeffect "", "~{v199}" ()
  ret void
}

; GCN-LABEL: {{^}}used_256_vgprs:
; GFX9:       ; Occupancy: 1
; GFX10W64:   ; Occupancy: 2
; GFX10W32:   ; Occupancy: 4
define amdgpu_kernel void @used_256_vgprs() {
  call void asm sideeffect "", "~{v255}" ()
  ret void
}

; GCN-LABEL: {{^}}used_80_sgprs:
; GFX9:       ; Occupancy: 10
; GFX1010:    ; Occupancy: 20
; GFX1030:    ; Occupancy: 16
define amdgpu_kernel void @used_80_sgprs() {
  call void asm sideeffect "", "~{s79}" ()
  ret void
}

; GCN-LABEL: {{^}}used_88_sgprs:
; GFX9:       ; Occupancy: 9
; GFX1010:    ; Occupancy: 20
; GFX1030:    ; Occupancy: 16
define amdgpu_kernel void @used_88_sgprs() {
  call void asm sideeffect "", "~{s87}" ()
  ret void
}

; GCN-LABEL: {{^}}used_100_sgprs:
; GFX9:       ; Occupancy: 8
; GFX1010:    ; Occupancy: 20
; GFX1030:    ; Occupancy: 16
define amdgpu_kernel void @used_100_sgprs() {
  call void asm sideeffect "", "~{s99}" ()
  ret void
}

; GCN-LABEL: {{^}}used_101_sgprs:
; GFX9:       ; Occupancy: 7
; GFX1010:    ; Occupancy: 20
; GFX1030:    ; Occupancy: 16
define amdgpu_kernel void @used_101_sgprs() {
  call void asm sideeffect "", "~{s100}" ()
  ret void
}

; GCN-LABEL: {{^}}used_lds_6552:
; GFX9:       ; Occupancy: 10
; GFX1010:    ; Occupancy: 20
; GFX1030:    ; Occupancy: 16
@lds6552 = internal addrspace(3) global [6552 x i8] undef, align 4
define amdgpu_kernel void @used_lds_6552() {
  %p = bitcast [6552 x i8] addrspace(3)* @lds6552 to i8 addrspace(3)*
  store volatile i8 1, i8 addrspace(3)* %p
  ret void
}

; GCN-LABEL: {{^}}used_lds_6556:
; GFX9:       ; Occupancy: 10
; GFX1010:    ; Occupancy: 20
; GFX1030:    ; Occupancy: 16
@lds6556 = internal addrspace(3) global [6556 x i8] undef, align 4
define amdgpu_kernel void @used_lds_6556() {
  %p = bitcast [6556 x i8] addrspace(3)* @lds6556 to i8 addrspace(3)*
  store volatile i8 1, i8 addrspace(3)* %p
  ret void
}

; GCN-LABEL: {{^}}used_lds_13112:
; GFX9:       ; Occupancy: 10
; GFX1010:    ; Occupancy: 20
; GFX1030:    ; Occupancy: 16
@lds13112 = internal addrspace(3) global [13112 x i8] undef, align 4
define amdgpu_kernel void @used_lds_13112() {
  %p = bitcast [13112 x i8] addrspace(3)* @lds13112 to i8 addrspace(3)*
  store volatile i8 1, i8 addrspace(3)* %p
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_64:
; GFX9:       ; Occupancy: 7{{$}}
; GFX10W64:   ; Occupancy: 7{{$}}
; GFX10W32:   ; Occupancy: 14{{$}}
@lds8252 = internal addrspace(3) global [8252 x i8] undef, align 4
define amdgpu_kernel void @used_lds_8252_max_group_size_64() #3 {
  %p = bitcast [8252 x i8] addrspace(3)* @lds8252 to i8 addrspace(3)*
  store volatile i8 1, i8 addrspace(3)* %p
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_96:
; GFX9:       ; Occupancy: 10{{$}}
; GFX10W64:   ; Occupancy: 14{{$}}
; GFX1010W32: ; Occupancy: 20{{$}}
; GFX1030W32: ; Occupancy: 16{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_96() #4 {
  %p = bitcast [8252 x i8] addrspace(3)* @lds8252 to i8 addrspace(3)*
  store volatile i8 1, i8 addrspace(3)* %p
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_128:
; GFX9:       ; Occupancy: 10{{$}}
; GFX10W64:   ; Occupancy: 14{{$}}
; GFX1010W32: ; Occupancy: 20{{$}}
; GFX1030W32: ; Occupancy: 16{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_128() #5 {
  %p = bitcast [8252 x i8] addrspace(3)* @lds8252 to i8 addrspace(3)*
  store volatile i8 1, i8 addrspace(3)* %p
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_192:
; GFX9:       ; Occupancy: 10{{$}}
; GFX1010:    ; Occupancy: 20{{$}}
; GFX1030:    ; Occupancy: 16{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_192() #6 {
  %p = bitcast [8252 x i8] addrspace(3)* @lds8252 to i8 addrspace(3)*
  store volatile i8 1, i8 addrspace(3)* %p
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_256:
; GFX9:       ; Occupancy: 10{{$}}
; GFX1010:    ; Occupancy: 20{{$}}
; GFX1030:    ; Occupancy: 16{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_256() #7 {
  %p = bitcast [8252 x i8] addrspace(3)* @lds8252 to i8 addrspace(3)*
  store volatile i8 1, i8 addrspace(3)* %p
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_512:
; GFX9:       ; Occupancy: 10{{$}}
; GFX1010:    ; Occupancy: 20{{$}}
; GFX1030:    ; Occupancy: 16{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_512() #8 {
  %p = bitcast [8252 x i8] addrspace(3)* @lds8252 to i8 addrspace(3)*
  store volatile i8 1, i8 addrspace(3)* %p
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_1024:
; GFX9:       ; Occupancy: 10{{$}}
; GFX1010:    ; Occupancy: 20{{$}}
; GFX1030:    ; Occupancy: 16{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_1024() #9 {
  %p = bitcast [8252 x i8] addrspace(3)* @lds8252 to i8 addrspace(3)*
  store volatile i8 1, i8 addrspace(3)* %p
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_32:
; GFX9:       ; Occupancy: 7{{$}}
; GFX10:      ; Occupancy: 7{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_32() #10 {
  %p = bitcast [8252 x i8] addrspace(3)* @lds8252 to i8 addrspace(3)*
  store volatile i8 1, i8 addrspace(3)* %p
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="2,3" "amdgpu-flat-work-group-size"="1,64" }
attributes #1 = { "amdgpu-waves-per-eu"="18,18" }
attributes #2 = { "amdgpu-waves-per-eu"="19,19" }
attributes #3 = { "amdgpu-flat-work-group-size"="1,64" }
attributes #4 = { "amdgpu-flat-work-group-size"="1,96" }
attributes #5 = { "amdgpu-flat-work-group-size"="1,128" }
attributes #6 = { "amdgpu-flat-work-group-size"="1,192" }
attributes #7 = { "amdgpu-flat-work-group-size"="1,256" }
attributes #8 = { "amdgpu-flat-work-group-size"="1,512" }
attributes #9 = { "amdgpu-flat-work-group-size"="1,1024" }
attributes #10 = { "amdgpu-flat-work-group-size"="1,32" }
