; RUN: llc < %s -march=nvptx -mcpu=sm_13

define ptx_device void @test_function({i8, i8}*) {
  ret void
}
