// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -triple spir-unknown-unknown | FileCheck -check-prefixes=COMMON,SPIR %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -triple amdgcn-amd-amdhsa | FileCheck -check-prefixes=COMMON,AMDGCN %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -debug-info-kind=limited -triple spir-unknown-unknown | FileCheck -check-prefixes=CHECK-DEBUG %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -O0 -debug-info-kind=limited -triple amdgcn-amd-amdhsa | FileCheck -check-prefixes=CHECK-DEBUG %s

// SPIR: %struct.__opencl_block_literal_generic = type { i32, i32, i8 addrspace(4)* }
// AMDGCN: %struct.__opencl_block_literal_generic = type { i32, i32, i8* }
// SPIR: @__block_literal_global = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 12, i32 4, i8 addrspace(4)* addrspacecast (i8* bitcast (void (i8 addrspace(4)*, i8 addrspace(3)*)* @block_A_block_invoke to i8*) to i8 addrspace(4)*) }
// AMDGCN: @__block_literal_global = internal addrspace(1) constant { i32, i32, i8* } { i32 16, i32 8, i8* bitcast (void (i8*, i8 addrspace(3)*)* @block_A_block_invoke to i8*) }
// COMMON-NOT: .str

// SPIR-LABEL: define internal {{.*}}void @block_A_block_invoke(i8 addrspace(4)* noundef %.block_descriptor, i8 addrspace(3)* noundef %a)
// AMDGCN-LABEL: define internal {{.*}}void @block_A_block_invoke(i8* noundef %.block_descriptor, i8 addrspace(3)* noundef %a)
void (^block_A)(local void *) = ^(local void *a) {
  return;
};

// COMMON-LABEL: define {{.*}}void @foo()
void foo(){
  int i;
  // COMMON-NOT: %block.isa
  // COMMON-NOT: %block.flags
  // COMMON-NOT: %block.reserved
  // COMMON-NOT: %block.descriptor
  // SPIR: %[[block_size:.*]] = getelementptr inbounds <{ i32, i32, i8 addrspace(4)*, i32 }>, <{ i32, i32, i8 addrspace(4)*, i32 }>* %block, i32 0, i32 0
  // AMDGCN: %[[block_size:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }> addrspace(5)* %block, i32 0, i32 0
  // SPIR: store i32 16, i32* %[[block_size]]
  // AMDGCN: store i32 20, i32 addrspace(5)* %[[block_size]]
  // SPIR: %[[block_align:.*]] = getelementptr inbounds <{ i32, i32, i8 addrspace(4)*, i32 }>, <{ i32, i32, i8 addrspace(4)*, i32 }>* %block, i32 0, i32 1
  // AMDGCN: %[[block_align:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }> addrspace(5)* %block, i32 0, i32 1
  // SPIR: store i32 4, i32* %[[block_align]]
  // AMDGCN: store i32 8, i32 addrspace(5)* %[[block_align]]
  // SPIR: %[[block_invoke:.*]] = getelementptr inbounds <{ i32, i32, i8 addrspace(4)*, i32 }>, <{ i32, i32, i8 addrspace(4)*, i32 }>* %[[block:.*]], i32 0, i32 2
  // SPIR: store i8 addrspace(4)* addrspacecast (i8* bitcast (i32 (i8 addrspace(4)*)* @__foo_block_invoke to i8*) to i8 addrspace(4)*), i8 addrspace(4)** %[[block_invoke]]
  // SPIR: %[[block_captured:.*]] = getelementptr inbounds <{ i32, i32, i8 addrspace(4)*, i32 }>, <{ i32, i32, i8 addrspace(4)*, i32 }>* %[[block]], i32 0, i32 3
  // SPIR: %[[i_value:.*]] = load i32, i32* %i
  // SPIR: store i32 %[[i_value]], i32* %[[block_captured]],
  // SPIR: %[[blk_ptr:.*]] = bitcast <{ i32, i32, i8 addrspace(4)*, i32 }>* %[[block]] to %struct.__opencl_block_literal_generic*
  // SPIR: %[[blk_gen_ptr:.*]] = addrspacecast %struct.__opencl_block_literal_generic* %[[blk_ptr]] to %struct.__opencl_block_literal_generic addrspace(4)*
  // SPIR: store %struct.__opencl_block_literal_generic addrspace(4)* %[[blk_gen_ptr]], %struct.__opencl_block_literal_generic addrspace(4)** %[[block_B:.*]],
  // SPIR: %[[block_literal:.*]] = load %struct.__opencl_block_literal_generic addrspace(4)*, %struct.__opencl_block_literal_generic addrspace(4)** %[[block_B]]
  // SPIR: %[[blk_gen_ptr:.*]] = bitcast %struct.__opencl_block_literal_generic addrspace(4)* %[[block_literal]] to i8 addrspace(4)*
  // SPIR: call {{.*}}i32 @__foo_block_invoke(i8 addrspace(4)* noundef %[[blk_gen_ptr]])
  // AMDGCN: %[[block_invoke:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }> addrspace(5)* %[[block:.*]], i32 0, i32 2
  // AMDGCN: store i8* bitcast (i32 (i8*)* @__foo_block_invoke to i8*), i8* addrspace(5)* %[[block_invoke]]
  // AMDGCN: %[[block_captured:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }> addrspace(5)* %[[block]], i32 0, i32 3
  // AMDGCN: %[[i_value:.*]] = load i32, i32 addrspace(5)* %i
  // AMDGCN: store i32 %[[i_value]], i32 addrspace(5)* %[[block_captured]],
  // AMDGCN: %[[blk_ptr:.*]] = bitcast <{ i32, i32, i8*, i32 }> addrspace(5)* %[[block]] to %struct.__opencl_block_literal_generic addrspace(5)*
  // AMDGCN: %[[blk_gen_ptr:.*]] = addrspacecast %struct.__opencl_block_literal_generic addrspace(5)* %[[blk_ptr]] to %struct.__opencl_block_literal_generic*
  // AMDGCN: store %struct.__opencl_block_literal_generic* %[[blk_gen_ptr]], %struct.__opencl_block_literal_generic* addrspace(5)* %[[block_B:.*]],
  // AMDGCN: %[[block_literal:.*]] = load %struct.__opencl_block_literal_generic*, %struct.__opencl_block_literal_generic* addrspace(5)* %[[block_B]]
  // AMDGCN: %[[blk_gen_ptr:.*]] = bitcast %struct.__opencl_block_literal_generic* %[[block_literal]] to i8*
  // AMDGCN: call {{.*}}i32 @__foo_block_invoke(i8* noundef %[[blk_gen_ptr]])

  int (^ block_B)(void) = ^{
    return i;
  };
  block_B();
}

// SPIR-LABEL: define internal {{.*}}i32 @__foo_block_invoke(i8 addrspace(4)* noundef %.block_descriptor)
// SPIR:  %[[block:.*]] = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32, i8 addrspace(4)*, i32 }> addrspace(4)*
// SPIR:  %[[block_capture_addr:.*]] = getelementptr inbounds <{ i32, i32, i8 addrspace(4)*, i32 }>, <{ i32, i32, i8 addrspace(4)*, i32 }> addrspace(4)* %[[block]], i32 0, i32 3
// SPIR:  %[[block_capture:.*]] = load i32, i32 addrspace(4)* %[[block_capture_addr]]
// AMDGCN-LABEL: define internal {{.*}}i32 @__foo_block_invoke(i8* noundef %.block_descriptor)
// AMDGCN:  %[[block:.*]] = bitcast i8* %.block_descriptor to <{ i32, i32, i8*, i32 }>*
// AMDGCN:  %[[block_capture_addr:.*]] = getelementptr inbounds <{ i32, i32, i8*, i32 }>, <{ i32, i32, i8*, i32 }>* %[[block]], i32 0, i32 3
// AMDGCN:  %[[block_capture:.*]] = load i32, i32* %[[block_capture_addr]]

// COMMON-NOT: define{{.*}}@__foo_block_invoke_kernel

// COMMON-LABEL: define {{.*}}@call_block
// call {{.*}}@__call_block_block_invoke
int call_block() {
  return ^int(int num) { return num; } (11);
}

// CHECK-DEBUG: !DIDerivedType(tag: DW_TAG_member, name: "__size"
// CHECK-DEBUG: !DIDerivedType(tag: DW_TAG_member, name: "__align"

// CHECK-DEBUG-NOT: !DIDerivedType(tag: DW_TAG_member, name: "__isa"
// CHECK-DEBUG-NOT: !DIDerivedType(tag: DW_TAG_member, name: "__flags"
// CHECK-DEBUG-NOT: !DIDerivedType(tag: DW_TAG_member, name: "__reserved"
// CHECK-DEBUG-NOT: !DIDerivedType(tag: DW_TAG_member, name: "__FuncPtr"
