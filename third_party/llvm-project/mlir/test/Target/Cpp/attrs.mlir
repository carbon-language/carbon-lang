// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

// CHECK-LABEL: void opaque_attrs() {
func @opaque_attrs() {
  // CHECK-NEXT: f(OPAQUE_ENUM_VALUE);
  emitc.call "f"() {args = [#emitc.opaque<"OPAQUE_ENUM_VALUE">]} : () -> ()
  // CHECK-NEXT: f("some string");
  emitc.call "f"() {args = [#emitc.opaque<"\"some string\"">]} : () -> ()
  return
}
