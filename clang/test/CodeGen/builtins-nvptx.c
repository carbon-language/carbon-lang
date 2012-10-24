// RUN: %clang_cc1 -triple nvptx-unknown-unknown -emit-llvm -o %t %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -emit-llvm -o %t %s

int read_tid() {

  int x = __builtin_ptx_read_tid_x();
  int y = __builtin_ptx_read_tid_y();
  int z = __builtin_ptx_read_tid_z();
  int w = __builtin_ptx_read_tid_w();

  return x + y + z + w;

}

int read_ntid() {

  int x = __builtin_ptx_read_ntid_x();
  int y = __builtin_ptx_read_ntid_y();
  int z = __builtin_ptx_read_ntid_z();
  int w = __builtin_ptx_read_ntid_w();

  return x + y + z + w;

}

int read_ctaid() {

  int x = __builtin_ptx_read_ctaid_x();
  int y = __builtin_ptx_read_ctaid_y();
  int z = __builtin_ptx_read_ctaid_z();
  int w = __builtin_ptx_read_ctaid_w();

  return x + y + z + w;

}

int read_nctaid() {

  int x = __builtin_ptx_read_nctaid_x();
  int y = __builtin_ptx_read_nctaid_y();
  int z = __builtin_ptx_read_nctaid_z();
  int w = __builtin_ptx_read_nctaid_w();

  return x + y + z + w;

}

int read_ids() {

  int a = __builtin_ptx_read_laneid();
  int b = __builtin_ptx_read_warpid();
  int c = __builtin_ptx_read_nwarpid();
  int d = __builtin_ptx_read_smid();
  int e = __builtin_ptx_read_nsmid();
  int f = __builtin_ptx_read_gridid();

  return a + b + c + d + e + f;

}

int read_lanemasks() {

  int a = __builtin_ptx_read_lanemask_eq();
  int b = __builtin_ptx_read_lanemask_le();
  int c = __builtin_ptx_read_lanemask_lt();
  int d = __builtin_ptx_read_lanemask_ge();
  int e = __builtin_ptx_read_lanemask_gt();

  return a + b + c + d + e;

}


long read_clocks() {

  int a = __builtin_ptx_read_clock();
  long b = __builtin_ptx_read_clock64();

  return (long)a + b;

}

int read_pms() {

  int a = __builtin_ptx_read_pm0();
  int b = __builtin_ptx_read_pm1();
  int c = __builtin_ptx_read_pm2();
  int d = __builtin_ptx_read_pm3();

  return a + b + c + d;

}

void sync() {

  __builtin_ptx_bar_sync(0);

}
