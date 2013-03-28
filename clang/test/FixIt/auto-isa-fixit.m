// RUN: cp %s %t
// RUN: %clang_cc1 -x objective-c -fixit %t
// RUN: %clang_cc1 -x objective-c -Werror %t
// rdar://13503456

void object_setClass(id, id);
Class object_getClass(id);

id rhs();

Class pr6302(id x123) {
  x123->isa  = 0;
  x123->isa = rhs();
  x123->isa = (id)(x123->isa);
  x123->isa = (id)x123->isa;
  x123->isa = (x123->isa);
  x123->isa = (id)(x123->isa);
  return x123->isa;
}
