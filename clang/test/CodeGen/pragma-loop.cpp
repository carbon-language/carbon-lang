// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

// Verify while loop is recognized after sequence of pragma clang loop directives.
void while_test(int *List, int Length) {
  // CHECK: define {{.*}} @_Z10while_test
  int i = 0;

#pragma clang loop vectorize(enable)
#pragma clang loop interleave_count(4)
#pragma clang loop vectorize_width(4)
#pragma clang loop unroll(enable)
  while (i < Length) {
    // CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_1:.*]]
    List[i] = i * 2;
    i++;
  }
}

// Verify do loop is recognized after multi-option pragma clang loop directive.
void do_test(int *List, int Length) {
  int i = 0;

#pragma clang loop vectorize_width(8) interleave_count(4) unroll(disable)
  do {
    // CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_2:.*]]
    List[i] = i * 2;
    i++;
  } while (i < Length);
}

// Verify for loop is recognized after sequence of pragma clang loop directives.
void for_test(int *List, int Length) {
#pragma clang loop interleave(enable)
#pragma clang loop interleave_count(4)
#pragma clang loop unroll_count(8)
  for (int i = 0; i < Length; i++) {
    // CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_3:.*]]
    List[i] = i * 2;
  }
}

// Verify c++11 for range loop is recognized after
// sequence of pragma clang loop directives.
void for_range_test() {
  double List[100];

#pragma clang loop vectorize_width(2) interleave_count(2)
  for (int i : List) {
    // CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_4:.*]]
    List[i] = i;
  }
}

// Verify disable pragma clang loop directive generates correct metadata
void disable_test(int *List, int Length) {
#pragma clang loop vectorize(disable) unroll(disable)
  for (int i = 0; i < Length; i++) {
    // CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_5:.*]]
    List[i] = i * 2;
  }
}

#define VECWIDTH 2
#define INTCOUNT 2
#define UNROLLCOUNT 8

// Verify defines are correctly resolved in pragma clang loop directive
void for_define_test(int *List, int Length, int Value) {
#pragma clang loop vectorize_width(VECWIDTH) interleave_count(INTCOUNT)
#pragma clang loop unroll_count(UNROLLCOUNT)
  for (int i = 0; i < Length; i++) {
    // CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_6:.*]]
    List[i] = i * Value;
  }
}

// Verify metadata is generated when template is used.
template <typename A>
void for_template_test(A *List, int Length, A Value) {

#pragma clang loop vectorize_width(8) interleave_count(8) unroll_count(8)
  for (int i = 0; i < Length; i++) {
    // CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_7:.*]]
    List[i] = i * Value;
  }
}

// Verify define is resolved correctly when template is used.
template <typename A>
void for_template_define_test(A *List, int Length, A Value) {
#pragma clang loop vectorize_width(VECWIDTH) interleave_count(INTCOUNT)
#pragma clang loop unroll_count(UNROLLCOUNT)
  for (int i = 0; i < Length; i++) {
    // CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_8:.*]]
    List[i] = i * Value;
  }
}

#undef VECWIDTH
#undef INTCOUNT
#undef UNROLLCOUNT

// Use templates defined above. Test verifies metadata is generated correctly.
void template_test(double *List, int Length) {
  double Value = 10;

  for_template_test<double>(List, Length, Value);
  for_template_define_test<double>(List, Length, Value);
}

// CHECK: ![[LOOP_1]] = metadata !{metadata ![[LOOP_1]], metadata ![[UNROLLENABLE_1:.*]], metadata ![[WIDTH_4:.*]], metadata ![[INTERLEAVE_4:.*]], metadata ![[INTENABLE_1:.*]]}
// CHECK: ![[UNROLLENABLE_1]] = metadata !{metadata !"llvm.loop.unroll.enable", i1 true}
// CHECK: ![[WIDTH_4]] = metadata !{metadata !"llvm.loop.vectorize.width", i32 4}
// CHECK: ![[INTERLEAVE_4]] = metadata !{metadata !"llvm.loop.vectorize.unroll", i32 4}
// CHECK: ![[INTENABLE_1]] = metadata !{metadata !"llvm.loop.vectorize.enable", i1 true}
// CHECK: ![[LOOP_2]] = metadata !{metadata ![[LOOP_2:.*]], metadata ![[UNROLLENABLE_0:.*]], metadata ![[INTERLEAVE_4:.*]], metadata ![[WIDTH_8:.*]]}
// CHECK: ![[UNROLLENABLE_0]] = metadata !{metadata !"llvm.loop.unroll.enable", i1 false}
// CHECK: ![[WIDTH_8]] = metadata !{metadata !"llvm.loop.vectorize.width", i32 8}
// CHECK: ![[LOOP_3]] = metadata !{metadata ![[LOOP_3]], metadata ![[UNROLL_8:.*]], metadata ![[INTERLEAVE_4:.*]], metadata ![[ENABLE_1:.*]]}
// CHECK: ![[UNROLL_8]] = metadata !{metadata !"llvm.loop.unroll.count", i32 8}
// CHECK: ![[LOOP_4]] = metadata !{metadata ![[LOOP_4]], metadata ![[INTERLEAVE_2:.*]], metadata ![[WIDTH_2:.*]]}
// CHECK: ![[INTERLEAVE_2]] = metadata !{metadata !"llvm.loop.vectorize.unroll", i32 2}
// CHECK: ![[WIDTH_2]] = metadata !{metadata !"llvm.loop.vectorize.width", i32 2}
// CHECK: ![[LOOP_5]] = metadata !{metadata ![[LOOP_5]], metadata ![[UNROLLENABLE_0:.*]], metadata ![[WIDTH_1:.*]]}
// CHECK: ![[WIDTH_1]] = metadata !{metadata !"llvm.loop.vectorize.width", i32 1}
// CHECK: ![[LOOP_6]] = metadata !{metadata ![[LOOP_6]], metadata ![[UNROLL_8:.*]], metadata ![[INTERLEAVE_2:.*]], metadata ![[WIDTH_2:.*]]}
// CHECK: ![[LOOP_7]] = metadata !{metadata ![[LOOP_7]], metadata ![[UNROLL_8:.*]], metadata ![[INTERLEAVE_8:.*]], metadata ![[WIDTH_8:.*]]}
// CHECK: ![[INTERLEAVE_8]] = metadata !{metadata !"llvm.loop.vectorize.unroll", i32 8}
// CHECK: ![[LOOP_8]] = metadata !{metadata ![[LOOP_8]], metadata ![[UNROLL_8:.*]], metadata ![[INTERLEAVE_2:.*]], metadata ![[WIDTH_2:.*]]}
