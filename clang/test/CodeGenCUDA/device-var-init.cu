// REQUIRES: nvptx-registered-target

// Make sure we don't allow dynamic initialization for device
// variables, but accept empty constructors allowed by CUDA.

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++11 \
// RUN:     -fno-threadsafe-statics -emit-llvm -o - %s | FileCheck %s

#ifdef __clang__
#include "Inputs/cuda.h"
#endif

// Base classes with different initializer variants.

// trivial constructor -- allowed
struct T {
  int t;
};

// empty constructor
struct EC {
  int ec;
  __device__ EC() {}     // -- allowed
  __device__ EC(int) {}  // -- not allowed
};

// empty templated constructor -- allowed with no arguments
struct ETC {
  template <typename... T> __device__ ETC(T...) {}
};

// undefined constructor -- not allowed
struct UC {
  int uc;
  __device__ UC();
};

// empty constructor w/ initializer list -- not allowed
struct ECI {
  int eci;
  __device__ ECI() : eci(1) {}
};

// non-empty constructor -- not allowed
struct NEC {
  int nec;
  __device__ NEC() { nec = 1; }
};

// no-constructor,  virtual method -- not allowed
struct NCV {
  int ncv;
  __device__ virtual void vm() {}
};

// dynamic in-class field initializer -- not allowed
__device__ int f();
struct NCF {
  int ncf = f();
};

// static in-class field initializer.  NVCC does not allow it, but
// clang generates static initializer for this, so we'll accept it.
// We still can't use it on __shared__ vars as they don't allow *any*
// initializers.
struct NCFS {
  int ncfs = 3;
};

// undefined templated constructor -- not allowed
struct UTC {
  template <typename... T> __device__ UTC(T...);
};

// non-empty templated constructor -- not allowed
struct NETC {
  int netc;
  template <typename... T> __device__ NETC(T...) { netc = 1; }
};

__device__ int d_v;
// CHECK: @d_v = addrspace(1) externally_initialized global i32 0,
__shared__ int s_v;
// CHECK: @s_v = addrspace(3) global i32 undef,
__constant__ int c_v;
// CHECK: addrspace(4) externally_initialized global i32 0,

__device__ int d_v_i = 1;
// CHECK: @d_v_i = addrspace(1) externally_initialized global i32 1,

__device__ T d_t;
// CHECK: @d_t = addrspace(1) externally_initialized global %struct.T zeroinitializer
__shared__ T s_t;
// CHECK: @s_t = addrspace(3) global %struct.T undef,
__constant__ T c_t;
// CHECK: @c_t = addrspace(4) externally_initialized global %struct.T zeroinitializer,

__device__ T d_t_i = {2};
// CHECK: @d_t_i = addrspace(1) externally_initialized global %struct.T { i32 2 },
__constant__ T c_t_i = {2};
// CHECK: @c_t_i = addrspace(4) externally_initialized global %struct.T { i32 2 },

__device__ EC d_ec;
// CHECK: @d_ec = addrspace(1) externally_initialized global %struct.EC zeroinitializer,
__shared__ EC s_ec;
// CHECK: @s_ec = addrspace(3) global %struct.EC undef,
__constant__ EC c_ec;
// CHECK: @c_ec = addrspace(4) externally_initialized global %struct.EC zeroinitializer,

__device__ ETC d_etc;
// CHECK: @d_etc = addrspace(1) externally_initialized global %struct.ETC zeroinitializer,
__shared__ ETC s_etc;
// CHECK: @s_etc = addrspace(3) global %struct.ETC undef,
__constant__ ETC c_etc;
// CHECK: @c_etc = addrspace(4) externally_initialized global %struct.ETC zeroinitializer,

__device__ NCFS d_ncfs;
// CHECK: @d_ncfs = addrspace(1) externally_initialized global %struct.NCFS { i32 3 }
__constant__ NCFS c_ncfs;
// CHECK: @c_ncfs = addrspace(4) externally_initialized global %struct.NCFS { i32 3 }

// Regular base class -- allowed
struct T_B_T : T {};
__device__ T_B_T d_t_b_t;
// CHECK: @d_t_b_t = addrspace(1) externally_initialized global %struct.T_B_T zeroinitializer,
__shared__ T_B_T s_t_b_t;
// CHECK: @s_t_b_t = addrspace(3) global %struct.T_B_T undef,
__constant__ T_B_T c_t_b_t;
// CHECK: @c_t_b_t = addrspace(4) externally_initialized global %struct.T_B_T zeroinitializer,

// Incapsulated object of allowed class -- allowed
struct T_F_T {
  T t;
};
__device__ T_F_T d_t_f_t;
// CHECK: @d_t_f_t = addrspace(1) externally_initialized global %struct.T_F_T zeroinitializer,
__shared__ T_F_T s_t_f_t;
// CHECK: @s_t_f_t = addrspace(3) global %struct.T_F_T undef,
__constant__ T_F_T c_t_f_t;
// CHECK: @c_t_f_t = addrspace(4) externally_initialized global %struct.T_F_T zeroinitializer,

// array of allowed objects -- allowed
struct T_FA_T {
  T t[2];
};
__device__ T_FA_T d_t_fa_t;
// CHECK: @d_t_fa_t = addrspace(1) externally_initialized global %struct.T_FA_T zeroinitializer,
__shared__ T_FA_T s_t_fa_t;
// CHECK: @s_t_fa_t = addrspace(3) global %struct.T_FA_T undef,
__constant__ T_FA_T c_t_fa_t;
// CHECK: @c_t_fa_t = addrspace(4) externally_initialized global %struct.T_FA_T zeroinitializer,


// Calling empty base class initializer is OK
struct EC_I_EC : EC {
  __device__ EC_I_EC() : EC() {}
};
__device__ EC_I_EC d_ec_i_ec;
// CHECK: @d_ec_i_ec = addrspace(1) externally_initialized global %struct.EC_I_EC zeroinitializer,
__shared__ EC_I_EC s_ec_i_ec;
// CHECK: @s_ec_i_ec = addrspace(3) global %struct.EC_I_EC undef,
__constant__ EC_I_EC c_ec_i_ec;
// CHECK: @c_ec_i_ec = addrspace(4) externally_initialized global %struct.EC_I_EC zeroinitializer,

// .. though passing arguments is not allowed.
struct EC_I_EC1 : EC {
  __device__ EC_I_EC1() : EC(1) {}
};

// Virtual base class -- not allowed
struct T_V_T : virtual T {};

// Inherited from or incapsulated class with non-empty constructor --
// not allowed
struct T_B_NEC : NEC {};
struct T_F_NEC {
  NEC nec;
};
struct T_FA_NEC {
  NEC nec[2];
};

// We should not emit global initializers for device-side variables.
// CHECK-NOT: @__cxx_global_var_init

// Make sure that initialization restrictions do not apply to local
// variables.
__device__ void df() {
  T t;
  // CHECK-NOT: call
  EC ec;
  // CHECK:   call void @_ZN2ECC1Ev(%struct.EC* %ec)
  ETC etc;
  // CHECK:   call void @_ZN3ETCC1IJEEEDpT_(%struct.ETC* %etc)
  UC uc;
  // CHECK:   call void @_ZN2UCC1Ev(%struct.UC* %uc)
  ECI eci;
  // CHECK:   call void @_ZN3ECIC1Ev(%struct.ECI* %eci)
  NEC nec;
  // CHECK:   call void @_ZN3NECC1Ev(%struct.NEC* %nec)
  NCV ncv;
  // CHECK-NOT: call
  NCF ncf;
  // CHECK:   call void @_ZN3NCFC1Ev(%struct.NCF* %ncf)
  NCFS ncfs;
  // CHECK:   call void @_ZN4NCFSC1Ev(%struct.NCFS* %ncfs)
  UTC utc;
  // CHECK:   call void @_ZN3UTCC1IJEEEDpT_(%struct.UTC* %utc)
  NETC netc;
  // CHECK:   call void @_ZN4NETCC1IJEEEDpT_(%struct.NETC* %netc)
  T_B_T t_b_t;
  // CHECK-NOT: call
  T_F_T t_f_t;
  // CHECK-NOT: call
  T_FA_T t_fa_t;
  // CHECK-NOT: call
  EC_I_EC ec_i_ec;
  // CHECK:   call void @_ZN7EC_I_ECC1Ev(%struct.EC_I_EC* %ec_i_ec)
  EC_I_EC1 ec_i_ec1;
  // CHECK:   call void @_ZN8EC_I_EC1C1Ev(%struct.EC_I_EC1* %ec_i_ec1)
  T_V_T t_v_t;
  // CHECK:   call void @_ZN5T_V_TC1Ev(%struct.T_V_T* %t_v_t)
  T_B_NEC t_b_nec;
  // CHECK:   call void @_ZN7T_B_NECC1Ev(%struct.T_B_NEC* %t_b_nec)
  T_F_NEC t_f_nec;
  // CHECK:   call void @_ZN7T_F_NECC1Ev(%struct.T_F_NEC* %t_f_nec)
  T_FA_NEC t_fa_nec;
  // CHECK:   call void @_ZN8T_FA_NECC1Ev(%struct.T_FA_NEC* %t_fa_nec)
  static __shared__ EC s_ec;
  // CHECK-NOT: call void @_ZN2ECC1Ev(%struct.EC* addrspacecast (%struct.EC addrspace(3)* @_ZZ2dfvE4s_ec to %struct.EC*))
  static __shared__ ETC s_etc;
  // CHECK-NOT: call void @_ZN3ETCC1IJEEEDpT_(%struct.ETC* addrspacecast (%struct.ETC addrspace(3)* @_ZZ2dfvE5s_etc to %struct.ETC*))

  // anchor point separating constructors and destructors
  df(); // CHECK: call void @_Z2dfv()

  // CHECK-NOT: call

  // CHECK: ret void
}

// We should not emit global init function.
// CHECK-NOT: @_GLOBAL__sub_I
