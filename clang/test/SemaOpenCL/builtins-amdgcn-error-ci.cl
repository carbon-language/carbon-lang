// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu tahiti -verify -S -o - %s

void test_ci_builtins()
{
  __builtin_amdgcn_s_dcache_inv_vol(); // expected-error {{'__builtin_amdgcn_s_dcache_inv_vol' needs target feature ci-insts}}
  __builtin_amdgcn_buffer_wbinvl1_vol(); // expected-error {{'__builtin_amdgcn_buffer_wbinvl1_vol' needs target feature ci-insts}}
  __builtin_amdgcn_ds_gws_sema_release_all(0); // expected-error {{'__builtin_amdgcn_ds_gws_sema_release_all' needs target feature ci-insts}}
}
