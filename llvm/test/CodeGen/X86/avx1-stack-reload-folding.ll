; RUN: llc -O3 -disable-peephole -mcpu=corei7-avx -mattr=+avx < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Function Attrs: nounwind readonly uwtable
define <32 x double> @_Z14vstack_foldDv32_dS_(<32 x double> %a, <32 x double> %b) #0 {
  %1 = fadd <32 x double> %a, %b
  %2 = fsub <32 x double> %a, %b
  %3 = fmul <32 x double> %1, %2
  ret <32 x double> %3

  ;CHECK-NOT:  vmovapd {{.*#+}} 32-byte Reload
  ;CHECK:       vmulpd {{[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  ;CHECK-NOT:  vmovapd {{.*#+}} 32-byte Reload
}
