; RUN: llc < %s -march=nvptx -mcpu=sm_20

define ptx_device void @test_function({i8, i8}*) {
  ret void
}
