// RUN: %clang_cc1 %s -g -triple %itanium_abi_triple -fno-use-cxa-atexit -S -emit-llvm -o - \
// RUN:     | FileCheck %s --check-prefix=CHECK-NOKEXT
// RUN: %clang_cc1 %s -g -triple %itanium_abi_triple -fno-use-cxa-atexit -fapple-kext -S -emit-llvm -o - \
// RUN:     | FileCheck %s --check-prefix=CHECK-KEXT

class A {
 public:
  A() {}
  virtual ~A() {}
};

A glob;
A array[2];

void foo() {
  static A stat;
}

// CHECK-NOKEXT: !MDSubprogram(name: "__cxx_global_var_init",{{.*}} line: 12,{{.*}} isLocal: true, isDefinition: true
// CHECK-NOKEXT: !MDSubprogram(name: "__dtor_glob",{{.*}} line: 12,{{.*}} isLocal: true, isDefinition: true
// CHECK-NOKEXT: !MDSubprogram(name: "__cxx_global_var_init1",{{.*}} line: 13,{{.*}} isLocal: true, isDefinition: true
// CHECK-NOKEXT: !MDSubprogram(name: "__cxx_global_array_dtor",{{.*}} line: 13,{{.*}} isLocal: true, isDefinition: true
// CHECK-NOKEXT: !MDSubprogram(name: "__dtor_array",{{.*}} line: 13,{{.*}} isLocal: true, isDefinition: true
// CHECK-NOKEXT: !MDSubprogram(name: "__dtor__ZZ3foovE4stat",{{.*}} line: 16,{{.*}} isLocal: true, isDefinition: true
// CHECK-NOKEXT: !MDSubprogram({{.*}} isLocal: true, isDefinition: true

// CHECK-KEXT: !MDSubprogram({{.*}} isLocal: true, isDefinition: true
