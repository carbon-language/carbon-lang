; RUN: llc <%s -march=sparc   -relocation-model=static | FileCheck %s --check-prefix=v8abs
; RUN: llc <%s -march=sparcv9 -relocation-model=static | FileCheck %s --check-prefix=v9abs
; RUN: llc <%s -march=sparc   -relocation-model=pic    | FileCheck %s --check-prefix=pic
; RUN: llc <%s -march=sparcv9 -relocation-model=pic    | FileCheck %s --check-prefix=pic


@local_symbol = internal thread_local global i32 0
@extern_symbol = external thread_local global i32

; v8abs-LABEL:  test_tls_local
; v8abs:        sethi  %tle_hix22(local_symbol), [[R0:%[goli][0-7]]]
; v8abs:        xor    [[R0]], %tle_lox10(local_symbol), [[R1:%[goli][0-7]]]
; v8abs:        ld     [%g7+[[R1]]]

; v9abs-LABEL:  test_tls_local
; v9abs:        sethi  %tle_hix22(local_symbol), [[R0:%[goli][0-7]]]
; v9abs:        xor    [[R0]], %tle_lox10(local_symbol), [[R1:%[goli][0-7]]]
; v9abs:        ld     [%g7+[[R1]]]

; pic-LABEL:  test_tls_local
; pic:        or     {{%[goli][0-7]}}, %lo(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[PC:%[goli][0-7]]]
; pic:        add    [[PC]], %o7, [[GOTBASE:%[goli][0-7]]]
; pic-DAG:    sethi  %tldm_hi22(local_symbol), [[R0:%[goli][0-7]]]
; pic-DAG:    add    [[R0]], %tldm_lo10(local_symbol), [[R1:%[goli][0-7]]]
; pic-DAG:    add    [[GOTBASE]], [[R1]], %o0, %tldm_add(local_symbol)
; pic-DAG:    call   __tls_get_addr, %tldm_call(local_symbol)
; pic-DAG:    sethi  %tldo_hix22(local_symbol), [[R2:%[goli][0-7]]]
; pic-DAG:    xor    [[R2]], %tldo_lox10(local_symbol), [[R3:%[goli][0-7]]]
; pic:        add    %o0, [[R3]], {{.+}}, %tldo_add(local_symbol)

define i32 @test_tls_local() {
entry:
  %0 = load i32* @local_symbol, align 4
  %1 = add i32 %0, 1
  store i32 %1, i32* @local_symbol, align 4
  ret i32 %1
}


; v8abs-LABEL:  test_tls_extern
; v8abs:        or     {{%[goli][0-7]}}, %lo(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[PC:%[goli][0-7]]]
; v8abs:        add    [[PC]], %o7, %[[GOTBASE:[goli][0-7]]]
; v8abs:        sethi  %tie_hi22(extern_symbol), [[R1:%[goli][0-7]]]
; v8abs:        add    [[R1]], %tie_lo10(extern_symbol), %[[R2:[goli][0-7]]]
; v8abs:        ld     [%[[GOTBASE]]+%[[R2]]], [[R3:%[goli][0-7]]], %tie_ld(extern_symbol)
; v8abs:        add    %g7, [[R3]], %[[R4:[goli][0-7]]], %tie_add(extern_symbol)
; v8abs:        ld     [%[[R4]]]

; v9abs-LABEL:  test_tls_extern
; v9abs:        or     {{%[goli][0-7]}}, %lo(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[PC:%[goli][0-7]]]
; v9abs:        add    [[PC]], %o7, %[[GOTBASE:[goli][0-7]]]
; v9abs:        sethi  %tie_hi22(extern_symbol), [[R1:%[goli][0-7]]]
; v9abs:        add    [[R1]], %tie_lo10(extern_symbol), %[[R2:[goli][0-7]]]
; v9abs:        ldx    [%[[GOTBASE]]+%[[R2]]], [[R3:%[goli][0-7]]], %tie_ldx(extern_symbol)
; v9abs:        add    %g7, [[R3]], %[[R4:[goli][0-7]]], %tie_add(extern_symbol)
; v9abs:        ld     [%[[R4]]]

; pic-LABEL:  test_tls_extern
; pic:        or     {{%[goli][0-7]}}, %lo(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[PC:%[goli][0-7]]]
; pic:        add    [[PC]], %o7, [[GOTBASE:%[goli][0-7]]]
; pic:        sethi  %tgd_hi22(extern_symbol), [[R0:%[goli][0-7]]]
; pic:        add    [[R0]], %tgd_lo10(extern_symbol), [[R1:%[goli][0-7]]]
; pic:        add    [[GOTBASE]], [[R1]], %o0, %tgd_add(extern_symbol)
; pic:        call   __tls_get_addr, %tgd_call(extern_symbol)
; pic-NEXT:   nop

define i32 @test_tls_extern() {
entry:
  %0 = load i32* @extern_symbol, align 4
  %1 = add i32 %0, 1
  store i32 %1, i32* @extern_symbol, align 4
  ret i32 %1
}
