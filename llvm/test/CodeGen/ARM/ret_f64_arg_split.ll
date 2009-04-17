; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2

define double @test_double_arg_split(i64 %a1, i32 %a2, double %a3) {
        ret double %a3
}

