// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s
// PR10204
static void loopback_VertexAttribI4ubv() {}

void _mesa_loopback_init_api_table() {
  (void) loopback_VertexAttribI4ubv;
}
