// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-apple-darwin9 | FileCheck %s

int atomic(void) {
  // non-sensical test for sync functions
  int old;
  int val = 1;
  char valc = 1;
  _Bool valb = 0;
  unsigned int uval = 1;
  int cmp = 0;

  old = __sync_fetch_and_add(&val, 1);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.load.add.i32.p0i32(i32* %val, i32 1)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  old = __sync_fetch_and_sub(&valc, 2);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i8 @llvm.atomic.load.sub.i8.p0i8(i8* %valc, i8 2)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  old = __sync_fetch_and_min(&val, 3);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.load.min.i32.p0i32(i32* %val, i32 3)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  old = __sync_fetch_and_max(&val, 4);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.load.max.i32.p0i32(i32* %val, i32 4)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  old = __sync_fetch_and_umin(&uval, 5u);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.load.umin.i32.p0i32(i32* %uval, i32 5)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  old = __sync_fetch_and_umax(&uval, 6u);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.load.umax.i32.p0i32(i32* %uval, i32 6)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  old = __sync_lock_test_and_set(&val, 7);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.swap.i32.p0i32(i32* %val, i32 7)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  old = __sync_swap(&val, 8);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.swap.i32.p0i32(i32* %val, i32 8)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  old = __sync_val_compare_and_swap(&val, 4, 1976);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* %val, i32 4, i32 1976)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  old = __sync_bool_compare_and_swap(&val, 4, 1976);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* %val, i32 4, i32 1976)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)

  old = __sync_fetch_and_and(&val, 0x9);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.load.and.i32.p0i32(i32* %val, i32 9)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)

  old = __sync_fetch_and_or(&val, 0xa);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.load.or.i32.p0i32(i32* %val, i32 10)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)

  old = __sync_fetch_and_xor(&val, 0xb);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.load.xor.i32.p0i32(i32* %val, i32 11)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  old = __sync_add_and_fetch(&val, 1);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.load.add.i32.p0i32(i32* %val, i32 1)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)

  old = __sync_sub_and_fetch(&val, 2);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %val, i32 2)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)

  old = __sync_and_and_fetch(&valc, 3);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i8 @llvm.atomic.load.and.i8.p0i8(i8* %valc, i8 3)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)

  old = __sync_or_and_fetch(&valc, 4);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i8 @llvm.atomic.load.or.i8.p0i8(i8* %valc, i8 4)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)

  old = __sync_xor_and_fetch(&valc, 5);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i8 @llvm.atomic.load.xor.i8.p0i8(i8* %valc, i8 5)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  
  __sync_val_compare_and_swap((void **)0, (void *)0, (void *)0);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* null, i32 0, i32 0)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)

  if ( __sync_val_compare_and_swap(&valb, 0, 1)) {
    // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
    // CHECK: call i8 @llvm.atomic.cmp.swap.i8.p0i8(i8* %valb, i8 0, i8 1)
    // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
    old = 42;
  }
  
  __sync_bool_compare_and_swap((void **)0, (void *)0, (void *)0);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* null, i32 0, i32 0)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  __sync_lock_release(&val);
  // CHECK: store volatile i32 0, i32* 
  
  __sync_synchronize ();
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 false)

  return old;
}

// CHECK: @release_return
void release_return(int *lock) {
  // Ensure this is actually returning void all the way through.
  return __sync_lock_release(lock);
  // CHECK: store volatile i32 0, i32* 
}


// rdar://8461279 - Atomics with address spaces.
// CHECK: @addrspace
void addrspace(int  __attribute__((address_space(256))) * P) {
  __sync_bool_compare_and_swap(P, 0, 1);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.cmp.swap.i32.p256i32(i32 addrspace(256)*{{.*}}, i32 0, i32 1)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  
  __sync_val_compare_and_swap(P, 0, 1);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.cmp.swap.i32.p256i32(i32 addrspace(256)*{{.*}}, i32 0, i32 1)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
  
  __sync_xor_and_fetch(P, 123);
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  // CHECK: call i32 @llvm.atomic.load.xor.i32.p256i32(i32 addrspace(256)* {{.*}}, i32 123)
  // CHECK: call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  
}

